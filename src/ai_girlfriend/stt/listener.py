from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import torch.hub
from RealtimeSTT import AudioToTextRecorder

logger = logging.getLogger(__name__)

OnText = Callable[[str], None]

# _run() retry/backoff tuning: bounds how hard a persistently broken
# microphone/model gets hammered instead of spinning and flooding the logs.
MAX_CONSECUTIVE_FAILURES = 5
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0

# AudioToTextRecorder construction retry tuning: its first run resolves the
# silero-vad model via torch.hub, which fetches https://github.com/... over
# the network with no retry of its own. A transient connection reset there
# (observed in practice) kills the whole app on startup otherwise.
INIT_MAX_ATTEMPTS = 3
INIT_RETRY_BACKOFF_SECONDS = 2.0

# How long stop() waits for the worker thread to exit before giving up on it.
SHUTDOWN_JOIN_TIMEOUT_SECONDS = 5.0


def _trust_silero_vad_repo() -> None:
    """Pre-trust the silero-vad torch.hub repo that RealtimeSTT loads internally.

    Without this, torch.hub prompts interactively on first use ("Do you trust
    this repository...?"), which hangs/fails when stdin isn't interactive.
    """
    trusted_list = Path(torch.hub.get_dir()) / "trusted_list"
    trusted_list.parent.mkdir(parents=True, exist_ok=True)
    entry = "snakers4_silero-vad"
    existing = trusted_list.read_text().splitlines() if trusted_list.exists() else []
    if entry not in existing:
        with trusted_list.open("a") as f:
            f.write(entry + "\n")


class Listener:
    """Continuously transcribes microphone audio to text.

    The only file in the project that imports RealtimeSTT — everything else
    talks to this plain callback interface instead.
    """

    def __init__(
        self,
        on_final_text: OnText,
        on_partial_text: OnText | None = None,
        on_fatal_error: Callable[[], None] | None = None,
        on_speech_start: Callable[[], None] | None = None,
        *,
        model: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        silero_sensitivity: float = 0.4,
    ) -> None:
        self._on_final_text = on_final_text
        self._on_fatal_error = on_fatal_error
        self._on_speech_start = on_speech_start
        _trust_silero_vad_repo()
        self._recorder = self._create_recorder(
            model=model,
            device=device,
            compute_type=compute_type,
            language=language,
            silero_sensitivity=silero_sensitivity,
            on_partial_text=on_partial_text,
        )
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._failed = False
        self._t_speech_start: float | None = None
        self._t_speech_end: float | None = None

    @property
    def failed(self) -> bool:
        """True if the listener gave up on its own after repeated STT failures."""
        return self._failed

    def mute(self) -> None:
        """Stop feeding microphone audio into the recognizer.

        Meant to be held for the duration of her own speech: without this,
        her voice leaking from the speakers into the microphone (there's no
        acoustic echo cancellation here) gets picked up by the VAD as the
        user talking, which both self-interrupts her mid-reply and
        occasionally gets transcribed and answered as if the user said it.
        """
        self._recorder.set_microphone(False)

    def unmute(self) -> None:
        """Resume feeding microphone audio into the recognizer after `mute()`."""
        self._recorder.set_microphone(True)

    def _create_recorder(
        self,
        *,
        model: str,
        device: str,
        compute_type: str,
        language: str,
        silero_sensitivity: float,
        on_partial_text: OnText | None,
    ) -> AudioToTextRecorder:
        last_exc: Exception | None = None
        for attempt in range(1, INIT_MAX_ATTEMPTS + 1):
            try:
                return AudioToTextRecorder(
                    model=model,
                    device=device,
                    compute_type=compute_type,
                    language=language,
                    silero_sensitivity=silero_sensitivity,
                    enable_realtime_transcription=on_partial_text is not None,
                    on_realtime_transcription_update=on_partial_text,
                    on_recording_start=self._on_recording_start,
                    on_recording_stop=self._on_recording_stop,
                    spinner=False,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Failed to initialize STT recorder (attempt %d/%d): %s",
                    attempt,
                    INIT_MAX_ATTEMPTS,
                    exc,
                )
                if attempt < INIT_MAX_ATTEMPTS:
                    time.sleep(INIT_RETRY_BACKOFF_SECONDS)
        logger.error("Giving up initializing STT recorder after %d attempts", INIT_MAX_ATTEMPTS)
        assert last_exc is not None
        raise last_exc

    def start(self) -> None:
        """Begin listening on a background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> bool:
        """Stop listening and release the microphone/model.

        Returns True if the worker thread terminated cleanly within the
        timeout, False if it's still alive afterwards. In the False case,
        `recorder.shutdown()` is skipped (the worker thread may still be
        using it) rather than pretending shutdown succeeded.
        """
        self._stop_event.set()
        self._recorder.abort()
        if self._thread is not None:
            self._thread.join(timeout=SHUTDOWN_JOIN_TIMEOUT_SECONDS)
            if self._thread.is_alive():
                logger.error(
                    "Listener worker thread did not stop within %.0fs; it may "
                    "still be running. Skipping recorder.shutdown() since the "
                    "recorder may still be in use on that thread.",
                    SHUTDOWN_JOIN_TIMEOUT_SECONDS,
                )
                return False
        self._recorder.shutdown()
        return True

    def _on_recording_start(self) -> None:
        self._t_speech_start = time.monotonic()
        logger.debug("Speech detected")
        if self._on_speech_start is not None:
            try:
                self._on_speech_start()
            except Exception:
                logger.exception("on_speech_start callback failed")

    def _on_recording_stop(self) -> None:
        self._t_speech_end = time.monotonic()
        logger.debug("Speech ended")

    def _run(self) -> None:
        consecutive_failures = 0
        while not self._stop_event.is_set():
            try:
                text = self._recorder.text()
            except Exception:
                consecutive_failures += 1
                logger.exception(
                    "STT transcription failed (%d/%d consecutive failures)",
                    consecutive_failures,
                    MAX_CONSECUTIVE_FAILURES,
                )
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        "STT failed %d times in a row; giving up.",
                        consecutive_failures,
                    )
                    self._failed = True
                    self._stop_event.set()
                    if self._on_fatal_error is not None:
                        try:
                            self._on_fatal_error()
                        except Exception:
                            logger.exception("on_fatal_error callback failed")
                    break
                backoff = min(
                    INITIAL_BACKOFF_SECONDS * 2 ** (consecutive_failures - 1),
                    MAX_BACKOFF_SECONDS,
                )
                logger.warning("Retrying in %.1fs", backoff)
                self._stop_event.wait(backoff)
                continue

            consecutive_failures = 0
            if not text:
                continue

            t_final = time.monotonic()
            if self._t_speech_end is not None:
                logger.info(
                    "Latency: speech_end -> final_transcript = %.0fms",
                    (t_final - self._t_speech_end) * 1000,
                )

            t_cb_start = time.monotonic()
            try:
                self._on_final_text(text)
            except Exception:
                logger.exception("on_final_text callback failed")
            finally:
                logger.info(
                    "Latency: on_final_text callback = %.0fms",
                    (time.monotonic() - t_cb_start) * 1000,
                )
