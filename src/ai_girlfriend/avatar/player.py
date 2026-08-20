from __future__ import annotations

import logging
import queue
import tempfile
import threading
import time
import wave
import winsound
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import pygame
import pythoncom
import win32com.client

from ai_girlfriend.avatar.blink import BlinkScheduler
from ai_girlfriend.avatar.bob import BOB_AMPLITUDE_PX, IdleBob
from ai_girlfriend.avatar.face import precompute_frames
from ai_girlfriend.avatar.lipsync import VisemeCue, run_rhubarb
from ai_girlfriend.avatar.squash import squash_stretch_scale
from ai_girlfriend.avatar.twitch import HeadTwitch
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
       warped-mouth cutouts of her on a pygame window, timed against the
       same cues.

    `face.py` segments her from the background once at construction, so
    everything below animates *her* — not the whole photo: a static
    background plate is drawn first, then her cutout is composited on top,
    bobbing up and down on a continuous triangle-wave cycle with a
    squash-and-stretch tied to it (`bob.py`, `squash.py`), independently
    blinking one eye and then the other on its own random schedule
    (`blink.py`), and once in a long while snapping her head back and forth
    in a quick twitch (`twitch.py`) — all regardless of speech, so she never
    looks perfectly frozen, without the background moving with her.

    Precomputing all 9 viseme cutouts plus both closed-eye overlays once at
    construction (`face.py`) means playback is just picking which cached
    frame(s) to transform and blit — no per-frame warping cost, so this
    stays real-time on CPU regardless of image size.
    """

    def __init__(
        self,
        image_path: str,
        rhubarb_path: str = "rhubarb.exe",
        voice: str = "",
        timeout: float = 15.0,
        on_playback_start: Callable[[], None] | None = None,
        on_playback_end: Callable[[], None] | None = None,
    ) -> None:
        self._rhubarb_path = rhubarb_path
        self._voice_name = voice
        self._timeout = timeout
        # Bracket only the audible WAV playback (not WAV synthesis, which is
        # rendered silently to a file) so a caller can mute the microphone
        # for exactly as long as her voice is actually audible — see
        # Listener.mute()'s docstring for why.
        self._on_playback_start = on_playback_start
        self._on_playback_end = on_playback_end
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
                self._twitch.tick()
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
        bg = self._assets.background
        h, w = bg.shape[:2]
        self._window = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Avatar")
        # Static — drawn as-is every frame, never itself transformed. Her
        # cutout (below) is composited on top of it and animated instead, so
        # moving/rotating/scaling her doesn't drag the backdrop along too.
        # No replicated-edge padding needed here the way the animated layer
        # below used to require: wherever she moves away from, this is
        # already a plausible reconstruction of what's behind her.
        self._background_surface = pygame.surfarray.make_surface(
            cv2.cvtColor(bg, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        )
        self._surfaces = {
            shape: pygame.image.frombuffer(
                cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA).tobytes(), (w, h), "RGBA"
            ).convert_alpha()
            for shape, frame in self._assets.mouth_frames.items()
        }
        self._eye_overlays = {
            eye_id: pygame.image.frombuffer(
                cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA).tobytes(), (w, h), "RGBA"
            ).convert_alpha()
            for eye_id, bgra in self._assets.eye_overlays.items()
        }
        self._bob = IdleBob()
        self._blink = BlinkScheduler(_EYE_IDS)
        self._twitch = HeadTwitch()
        self._current_mouth_shape = IDLE_SHAPE
        self._render()

    def _show_frame(self, shape: str) -> None:
        self._current_mouth_shape = shape
        self._render()

    def _render(self) -> None:
        offset = self._bob.offset_px()
        dy = round(offset)
        angle = self._twitch.angle_deg
        normalized_offset = offset / BOB_AMPLITUDE_PX if BOB_AMPLITUDE_PX else 0.0
        scale_x, scale_y = squash_stretch_scale(normalized_offset)

        self._window.blit(self._background_surface, (0, 0))
        surface = self._surfaces.get(self._current_mouth_shape, self._surfaces[IDLE_SHAPE])
        self._blit_transformed(surface, 0, dy, angle, scale_x, scale_y)
        for eye_id, closed in self._blink.closed.items():
            if closed:
                self._blit_transformed(
                    self._eye_overlays[eye_id], 0, dy, angle, scale_x, scale_y
                )
        pygame.display.flip()

    def _blit_transformed(
        self,
        surface: pygame.Surface,
        dx: int,
        dy: int,
        angle: float,
        scale_x: float,
        scale_y: float,
    ) -> None:
        """Blit `surface` at (dx, dy), scaled and then rotated."""
        w0, h0 = surface.get_size()
        if scale_x != 1.0 or scale_y != 1.0:
            new_w, new_h = max(1, round(w0 * scale_x)), max(1, round(h0 * scale_y))
            surface = pygame.transform.smoothscale(surface, (new_w, new_h))
            # Anchored at the bottom, not the center: the squash/stretch
            # should read as her standing on a fixed floor and growing
            # taller/shorter from there, not swelling from her own middle
            # (which visibly lifted her feet off the ground on a stretch).
            dx += (w0 - new_w) // 2
            dy += h0 - new_h
        if angle:
            w_pre_rotate, h_pre_rotate = surface.get_size()
            surface = pygame.transform.rotate(surface, angle)
            w1, h1 = surface.get_size()
            dx += (w_pre_rotate - w1) // 2
            dy += (h_pre_rotate - h1) // 2
        self._window.blit(surface, (dx, dy))

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
                self._twitch.tick()
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
            if self._on_playback_start is not None:
                self._on_playback_start()
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
                if self._on_playback_end is not None:
                    self._on_playback_end()

    def _wait_until(self, deadline: float) -> bool:
        """Block until `deadline`; return True if cut short by interrupt/stop/timeout."""
        while True:
            self._pump_events()
            self._blink.tick()
            self._twitch.tick()
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
