from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import pythoncom
import win32com.client

logger = logging.getLogger(__name__)

SVSF_PURGE_BEFORE_SPEAK = 2
SVSF_FLAGS_ASYNC = 1

# How often the worker re-checks stop_event/timeout while a phrase is
# playing. SAPI5's WaitUntilDone only accepts a fixed timeout, not an
# interruptible event, so this is polled in small slices.
_WAIT_POLL_MS = 200

# Sentinel telling the worker thread to exit its loop.
_SHUTDOWN = object()


class Speaker:
    """Speaks text out loud through the Windows SAPI5 voice.

    The only file in the project that talks to SAPI5 — everything else just
    calls `say()`. Goes straight to SAPI5 via pywin32 rather than through
    pyttsx3/RealtimeTTS's SystemEngine: that layer reliably hangs or silently
    drops audio on the second call within a process, whereas a single
    SAPI.SpVoice COM object handling repeated Speak() calls is its intended,
    reliable usage pattern.

    `say()` is non-blocking: it queues `text` and returns immediately. A
    single dedicated worker thread owns the SAPI5 COM voice object and speaks
    queued phrases one at a time, in order. This keeps a slow or hung TTS
    call from ever blocking whatever thread calls `say()` (namely the STT
    callback thread). The COM object is created lazily on the worker thread
    (and COM initialized there) rather than in __init__, because COM objects
    are bound to the apartment/thread that created them.

    `interrupt()` cuts off whatever's currently playing and drops anything
    still queued, but leaves the worker thread running so future `say()`
    calls keep working — meant for barge-in (the user starts talking again
    while she's still replying). `stop()` does the same but also shuts the
    worker thread down for good, meant for process shutdown.
    """

    def __init__(
        self,
        voice: str = "",
        timeout: float = 15.0,
        on_playback_start: Callable[[], None] | None = None,
        on_playback_end: Callable[[], None] | None = None,
    ) -> None:
        self._voice_name = voice
        self._timeout = timeout
        # Bracket actual audio output (not queueing/synthesis) so a caller can
        # mute the microphone for exactly as long as her voice is audible —
        # see Listener.mute()'s docstring for why.
        self._on_playback_start = on_playback_start
        self._on_playback_end = on_playback_end
        self._local = threading.local()
        self._queue: queue.Queue[Any] = queue.Queue()
        self._stop_event = threading.Event()
        self._interrupt_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

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

    def say(self, text: str) -> None:
        """Queue `text` to be spoken; returns immediately without blocking."""
        if not text:
            return
        self._queue.put((text, time.monotonic()))

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SHUTDOWN:
                return
            text, t_queued = item
            t_started = time.monotonic()
            logger.info(
                "Latency: TTS queued -> started = %.0fms (queue_depth=%d)",
                (t_started - t_queued) * 1000,
                self._queue.qsize(),
            )
            try:
                self._speak_now(text)
            except Exception:
                logger.exception("TTS playback failed")
            finally:
                logger.info(
                    "Latency: TTS started -> done = %.0fms (total = %.0fms)",
                    (time.monotonic() - t_started) * 1000,
                    (time.monotonic() - t_queued) * 1000,
                )

    def _speak_now(self, text: str) -> None:
        # Cleared here, not by whoever calls interrupt(): an interrupt that
        # arrives before this phrase started (queue already drained) must
        # not cancel *this* phrase too, only whatever it replaced.
        self._interrupt_event.clear()
        voice = self._get_voice()
        if self._on_playback_start is not None:
            self._on_playback_start()
        try:
            voice.Speak(text, SVSF_FLAGS_ASYNC)
            start = time.monotonic()
            while True:
                if voice.WaitUntilDone(_WAIT_POLL_MS):
                    return
                if self._stop_event.is_set():
                    logger.debug("TTS interrupted by stop()")
                    voice.Speak("", SVSF_PURGE_BEFORE_SPEAK)
                    return
                if self._interrupt_event.is_set():
                    logger.debug("TTS interrupted by barge-in")
                    voice.Speak("", SVSF_PURGE_BEFORE_SPEAK)
                    return
                if time.monotonic() - start > self._timeout:
                    logger.warning("TTS timed out after %.1fs; skipping phrase", self._timeout)
                    voice.Speak("", SVSF_PURGE_BEFORE_SPEAK)
                    return
        finally:
            if self._on_playback_end is not None:
                self._on_playback_end()

    def _drain_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def interrupt(self) -> None:
        """Cut off current playback and drop anything queued; keep the worker alive."""
        self._interrupt_event.set()
        self._drain_queue()

    def stop(self) -> None:
        """Stop playback, drop anything still queued, and shut down the worker."""
        self._stop_event.set()
        self._drain_queue()
        self._queue.put(_SHUTDOWN)
        self._worker.join(timeout=5)
        if self._worker.is_alive():
            logger.error("TTS worker thread did not stop within 5s")
