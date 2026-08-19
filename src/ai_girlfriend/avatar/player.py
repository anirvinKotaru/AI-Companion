from __future__ import annotations

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

from ai_girlfriend.avatar.blink import BlinkScheduler
from ai_girlfriend.avatar.bob import BOB_AMPLITUDE_PX, IdleBob
from ai_girlfriend.avatar.face import precompute_frames
from ai_girlfriend.avatar.jitter import JITTER_AMPLITUDE_PX, Jitter
from ai_girlfriend.avatar.lipsync import VisemeCue, run_rhubarb
from ai_girlfriend.avatar.visemes import IDLE_SHAPE

_EYE_IDS = ("left", "right")

logger = logging.getLogger(__name__)

# SAPI5 enum values (SpeechLib), used the same literal-constant way speaker.py
# uses SVSF_* — avoids depending on win32com's gencache-generated constants.
_SAFT_22KHZ_16BIT_MONO = 22
_SSFM_CREATE_FOR_WRITE = 3
_SVSF_FLAGS_ASYNC = 1
_SVSF_PURGE_BEFORE_SPEAK = 2

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
    1. renders the reply to a WAV file via SAPI5 (`SpFileStream`),
    2. runs Rhubarb Lip Sync on that WAV to get viseme timing cues,
    3. plays the WAV (`winsound`) while swapping between 9 precomputed
       warped-mouth frames of the uploaded image on a pygame window, timed
       against the same cues, and independently blinks one eye and then the
       other on its own random schedule (`blink.py`), bobs the whole frame
       up and down on a continuous triangle-wave cycle (`bob.py`), and
       layers small abrupt random jitter on top of that (`jitter.py`) — all
       regardless of speech, so she never looks perfectly frozen.

    Precomputing all 9 viseme frames plus both closed-eye overlays once at
    construction (`face.py`) means playback is just picking which cached
    frame(s) to blit — no per-frame warping cost, so this stays real-time on
    CPU regardless of image size.
    """

    def __init__(
        self,
        image_path: str,
        rhubarb_path: str = "rhubarb.exe",
        voice: str = "",
        timeout: float = 15.0,
    ) -> None:
        self._rhubarb_path = rhubarb_path
        self._voice_name = voice
        self._timeout = timeout
        # Done once, on the calling (startup) thread — precompute_frames is a
        # one-off cost, unlike everything below it which must be fast per-turn.
        self._assets = precompute_frames(image_path)
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
            if self._voice_name:
                self._select_voice(voice, self._voice_name)
            self._local.voice = voice
        return voice

    def _select_voice(self, voice: Any, name: str) -> None:
        for candidate in voice.GetVoices():
            if name.lower() in candidate.GetDescription().lower():
                voice.Voice = candidate
                return
        logger.warning("TTS voice %r not found; using the default voice", name)

    def _synthesize(self, text: str, wav_path: str) -> None:
        """Render `text` to a WAV file at `wav_path` via SAPI5, without playing it live.

        Spoken asynchronously and polled with a timeout (mirroring
        Speaker._speak_now), rather than the simpler synchronous `Speak(text)`
        call: that blocks the worker thread — and stops pygame's event queue
        from being pumped — for however long SAPI5 takes, with no way to
        bound it. A hung SAPI5 call would wedge this thread permanently,
        `stop()` would time out waiting for it, and the window would sit
        frozen ("Not Responding") for the whole call, not just at idle.
        """
        voice = self._get_voice()
        fmt = win32com.client.Dispatch("SAPI.SpAudioFormat")
        fmt.Type = _SAFT_22KHZ_16BIT_MONO
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        stream.Format = fmt
        stream.Open(wav_path, _SSFM_CREATE_FOR_WRITE)
        voice.AudioOutputStream = stream
        try:
            voice.Speak(text, _SVSF_FLAGS_ASYNC)
            start = time.monotonic()
            while not voice.WaitUntilDone(int(_POLL_SECONDS * 1000)):
                self._pump_events()
                self._blink.tick()
                self._jitter.tick()
                self._render()
                if self._stop_event.is_set():
                    voice.Speak("", _SVSF_PURGE_BEFORE_SPEAK)
                    break
                if time.monotonic() - start > self._timeout:
                    logger.warning(
                        "SAPI5 synthesis timed out after %.1fs; WAV may be incomplete",
                        self._timeout,
                    )
                    voice.Speak("", _SVSF_PURGE_BEFORE_SPEAK)
                    break
        finally:
            stream.Close()

    # -- pygame window (worker thread only) --

    def _init_window(self) -> None:
        pygame.init()
        first = next(iter(self._assets.mouth_frames.values()))
        h, w = first.shape[:2]
        self._window = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Avatar")
        # Each frame gets replicated-edge padding — vertically for the bob
        # plus jitter, horizontally for jitter alone — bigger than the
        # window itself. Idle motion then blits it at an offset within that
        # padding rather than resizing or moving the window, so it never
        # exposes empty space at an edge.
        self._x_margin = round(JITTER_AMPLITUDE_PX)
        self._y_margin = round(BOB_AMPLITUDE_PX + JITTER_AMPLITUDE_PX)
        xm, ym = self._x_margin, self._y_margin
        self._surfaces = {
            shape: pygame.surfarray.make_surface(
                cv2.cvtColor(
                    cv2.copyMakeBorder(frame, ym, ym, xm, xm, cv2.BORDER_REPLICATE),
                    cv2.COLOR_BGR2RGB,
                ).swapaxes(0, 1)
            )
            for shape, frame in self._assets.mouth_frames.items()
        }
        self._eye_overlays = {
            eye_id: pygame.image.frombuffer(
                cv2.cvtColor(
                    cv2.copyMakeBorder(bgra, ym, ym, xm, xm, cv2.BORDER_REPLICATE),
                    cv2.COLOR_BGRA2RGBA,
                ).tobytes(),
                (w + 2 * xm, h + 2 * ym),
                "RGBA",
            ).convert_alpha()
            for eye_id, bgra in self._assets.eye_overlays.items()
        }
        self._bob = IdleBob()
        self._jitter = Jitter()
        self._blink = BlinkScheduler(_EYE_IDS)
        self._current_mouth_shape = IDLE_SHAPE
        self._render()

    def _show_frame(self, shape: str) -> None:
        self._current_mouth_shape = shape
        self._render()

    def _render(self) -> None:
        dx = round(self._jitter.dx) - self._x_margin
        dy = round(self._bob.offset_px() + self._jitter.dy) - self._y_margin
        surface = self._surfaces.get(self._current_mouth_shape, self._surfaces[IDLE_SHAPE])
        self._window.blit(surface, (dx, dy))
        for eye_id, closed in self._blink.closed.items():
            if closed:
                self._window.blit(self._eye_overlays[eye_id], (dx, dy))
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
                self._blink.tick()
                self._jitter.tick()
                self._render()
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
            try:
                cues: list[VisemeCue] = run_rhubarb(
                    self._rhubarb_path, wav_path, dialog_text=text, timeout=self._timeout
                )
            except Exception:
                logger.exception("Rhubarb lip-sync failed; playing audio with a static mouth")
                cues = []

            # Played by filename, not SND_MEMORY: winsound raises RuntimeError
            # if SND_ASYNC is combined with SND_MEMORY, since the OS can't
            # safely play async from a Python buffer that might get garbage
            # collected mid-playback. Playing from the temp file (kept alive
            # for the duration of this `with` block) avoids that restriction.
            duration = _wav_duration(wav_path)
            start = time.monotonic()
            try:
                winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
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
            self._blink.tick()
            self._jitter.tick()
            self._render()
            if self._stop_event.is_set() or self._interrupt_event.is_set():
                return True
            if time.monotonic() - self._phrase_start > self._timeout:
                logger.warning("Avatar playback timed out after %.1fs", self._timeout)
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(remaining, _POLL_SECONDS))


def _wav_duration(wav_path: str) -> float:
    with wave.open(wav_path, "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()
