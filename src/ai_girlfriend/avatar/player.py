from __future__ import annotations

import io
import logging
import queue
import tempfile
import threading
import time
import wave
import winsound
from pathlib import Path
from typing import Any

import cv2
import pygame
import pythoncom
import win32com.client

from ai_girlfriend.avatar.face import precompute_frames
from ai_girlfriend.avatar.lipsync import VisemeCue, run_rhubarb
from ai_girlfriend.avatar.visemes import IDLE_SHAPE

logger = logging.getLogger(__name__)

# SAPI5 enum values (SpeechLib), used the same literal-constant way speaker.py
# uses SVSF_* — avoids depending on win32com's gencache-generated constants.
_SAFT_22KHZ_16BIT_MONO = 22
_SSFM_CREATE_FOR_WRITE = 3

# How often the worker re-checks stop_event/interrupt_event/timeout, and
# pumps pygame's event queue, while a phrase is playing.
_POLL_SECONDS = 0.02

_SHUTDOWN = object()


class Avatar:
    """Speaks text out loud (via SAPI5) while animating an uploaded 2D image's mouth.

    Mirrors `ai_girlfriend.tts.speaker.Speaker`'s interface (`say`/`interrupt`/
    `stop`) and threading design exactly, so `main.py` can use either
    interchangeably — see docs/design/004-talking-head.md for why. Where
    `Speaker` speaks live via SAPI5's async `Speak()`, `Avatar` instead:
    1. renders the reply to an in-memory WAV via SAPI5 (`SpFileStream`),
    2. runs Rhubarb Lip Sync on that WAV to get viseme timing cues,
    3. plays the WAV (`winsound`, from memory) while swapping between 9
       precomputed warped-mouth frames of the uploaded image on a pygame
       window, timed against the same cues.

    Precomputing all 9 viseme frames once at construction (`face.py`) means
    playback is just picking which cached frame to blit — no per-frame
    warping cost, so this stays real-time on CPU regardless of image size.
    """

    def __init__(
        self,
        image_path: str,
        rhubarb_path: str = "rhubarb.exe",
        timeout: float = 15.0,
    ) -> None:
        self._rhubarb_path = rhubarb_path
        self._timeout = timeout
        # Done once, on the calling (startup) thread — precompute_frames is a
        # one-off cost, unlike everything below it which must be fast per-turn.
        self._raw_frames = precompute_frames(image_path)
        self._local = threading.local()
        self._queue: queue.Queue[Any] = queue.Queue()
        self._stop_event = threading.Event()
        self._interrupt_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def say(self, text: str) -> None:
        """Queue `text` to be spoken and lip-synced; returns immediately without blocking."""
        if not text:
            return
        self._queue.put((text, time.monotonic()))

    def interrupt(self) -> None:
        """Cut off current playback and drop anything queued; keep the worker alive."""
        self._interrupt_event.set()
        self._drain_queue()

    def stop(self) -> None:
        """Stop playback, drop anything still queued, and shut down the worker + window."""
        self._stop_event.set()
        self._drain_queue()
        self._queue.put(_SHUTDOWN)
        self._worker.join(timeout=10)
        if self._worker.is_alive():
            logger.error("Avatar worker thread did not stop within 10s")

    def _drain_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    # -- SAPI5 (worker thread only; COM objects are bound to their creating thread) --

    def _get_voice(self) -> Any:
        voice = getattr(self._local, "voice", None)
        if voice is None:
            pythoncom.CoInitialize()
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            self._local.voice = voice
        return voice

    def _synthesize(self, text: str, wav_path: str) -> None:
        """Render `text` to a WAV file at `wav_path` via SAPI5, without playing it live."""
        voice = self._get_voice()
        fmt = win32com.client.Dispatch("SAPI.SpAudioFormat")
        fmt.Type = _SAFT_22KHZ_16BIT_MONO
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        stream.Format = fmt
        stream.Open(wav_path, _SSFM_CREATE_FOR_WRITE)
        voice.AudioOutputStream = stream
        voice.Speak(text)  # synchronous: blocks until the file is fully written
        stream.Close()

    # -- pygame window (worker thread only) --

    def _init_window(self) -> None:
        pygame.init()
        first = next(iter(self._raw_frames.values()))
        h, w = first.shape[:2]
        self._window = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Avatar")
        self._surfaces = {
            shape: pygame.surfarray.make_surface(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
            )
            for shape, frame in self._raw_frames.items()
        }
        self._show_frame(IDLE_SHAPE)

    def _show_frame(self, shape: str) -> None:
        surface = self._surfaces.get(shape, self._surfaces[IDLE_SHAPE])
        self._window.blit(surface, (0, 0))
        pygame.display.flip()

    def _pump_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logger.info("Avatar window closed")
                self._stop_event.set()

    def _close_window(self) -> None:
        pygame.quit()

    # -- playback loop --

    def _worker_loop(self) -> None:
        self._init_window()
        while True:
            # Polled with a short timeout rather than a plain blocking get():
            # pygame needs its event queue pumped regularly or Windows marks
            # the window "Not Responding" almost immediately, even while
            # just idling between phrases waiting for the next say() call.
            try:
                item = self._queue.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                self._pump_events()
                if self._stop_event.is_set():
                    self._close_window()
                    return
                continue
            if item is _SHUTDOWN:
                self._close_window()
                return
            text, t_queued = item
            t_started = time.monotonic()
            logger.info(
                "Latency: Avatar queued -> started = %.0fms (queue_depth=%d)",
                (t_started - t_queued) * 1000,
                self._queue.qsize(),
            )
            try:
                self._play_now(text)
            except Exception:
                logger.exception("Avatar playback failed")
            finally:
                logger.info(
                    "Latency: Avatar started -> done = %.0fms (total = %.0fms)",
                    (time.monotonic() - t_started) * 1000,
                    (time.monotonic() - t_queued) * 1000,
                )

    def _play_now(self, text: str) -> None:
        # Cleared here, not by whoever calls interrupt() — same reasoning as
        # Speaker._speak_now: an interrupt for a prior phrase must not cancel
        # this one too.
        self._interrupt_event.clear()
        self._phrase_start = time.monotonic()

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = str(Path(tmp_dir) / "speech.wav")
            self._synthesize(text, wav_path)
            wav_bytes = Path(wav_path).read_bytes()
            try:
                cues: list[VisemeCue] = run_rhubarb(
                    self._rhubarb_path, wav_path, dialog_text=text, timeout=self._timeout
                )
            except Exception:
                logger.exception("Rhubarb lip-sync failed; playing audio with a static mouth")
                cues = []

        duration = _wav_duration(wav_bytes)
        start = time.monotonic()
        try:
            winsound.PlaySound(wav_bytes, winsound.SND_MEMORY | winsound.SND_ASYNC)
            for cue in cues:
                if self._wait_until(start + cue.start):
                    return
                self._show_frame(cue.shape)
            self._wait_until(start + duration)
        finally:
            winsound.PlaySound(None, winsound.SND_PURGE)
            self._show_frame(IDLE_SHAPE)

    def _wait_until(self, deadline: float) -> bool:
        """Block until `deadline`; return True if cut short by interrupt/stop/timeout."""
        while True:
            self._pump_events()
            if self._stop_event.is_set() or self._interrupt_event.is_set():
                return True
            if time.monotonic() - self._phrase_start > self._timeout:
                logger.warning("Avatar playback timed out after %.1fs", self._timeout)
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(remaining, _POLL_SECONDS))


def _wav_duration(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()
