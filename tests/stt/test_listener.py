from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import ai_girlfriend.stt.listener as listener_module
from ai_girlfriend.stt.listener import Listener


class FakeRecorder:
    """Stand-in for RealtimeSTT's AudioToTextRecorder.

    `text()` mimics its real blocking behavior: it returns queued utterances
    one at a time, then blocks until `abort()` is called (simulating waiting
    on the microphone) instead of busy-looping.
    """

    def __init__(self, texts: list[str] | None = None) -> None:
        self._texts = list(texts or [])
        self._unblocked = threading.Event()
        self.abort_called = False
        self.shutdown_called = False

    def text(self) -> str:
        if self._texts:
            return self._texts.pop(0)
        self._unblocked.wait()
        return ""

    def abort(self) -> None:
        self.abort_called = True
        self._unblocked.set()

    def shutdown(self) -> None:
        self.shutdown_called = True


class StuckRecorder:
    """A recorder whose text() never returns, even after abort()."""

    def __init__(self) -> None:
        self.shutdown_called = False
        self._never = threading.Event()

    def text(self) -> str:
        self._never.wait()
        return ""

    def abort(self) -> None:
        pass  # does NOT unblock text() -- simulates a truly hung worker

    def shutdown(self) -> None:
        self.shutdown_called = True


class FailingRecorder:
    """A recorder whose text() always raises, simulating a broken mic/model."""

    def __init__(self) -> None:
        self.call_count = 0
        self.shutdown_called = False

    def text(self) -> str:
        self.call_count += 1
        raise RuntimeError("microphone unavailable")

    def abort(self) -> None:
        pass

    def shutdown(self) -> None:
        self.shutdown_called = True


def _make_listener(recorder: object, **kwargs: object) -> Listener:
    with (
        patch.object(listener_module, "AudioToTextRecorder", return_value=recorder),
        patch.object(listener_module, "_trust_silero_vad_repo"),
    ):
        return Listener(**kwargs)  # type: ignore[arg-type]


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_listener_starts_and_stops_cleanly() -> None:
    recorder = FakeRecorder()
    listener = _make_listener(recorder, on_final_text=lambda t: None)

    listener.start()
    time.sleep(0.05)
    assert listener.stop() is True
    assert recorder.abort_called
    assert recorder.shutdown_called


def test_final_transcript_callback_is_invoked() -> None:
    recorder = FakeRecorder(["hello world"])
    received: list[str] = []
    listener = _make_listener(recorder, on_final_text=received.append)

    listener.start()
    assert _wait_until(lambda: received == ["hello world"])
    listener.stop()


def test_partial_transcript_callback_is_wired_through() -> None:
    on_partial = MagicMock()
    with (
        patch.object(listener_module, "AudioToTextRecorder") as recorder_cls,
        patch.object(listener_module, "_trust_silero_vad_repo"),
    ):
        recorder_cls.return_value = FakeRecorder()
        Listener(on_final_text=lambda t: None, on_partial_text=on_partial)

    _, kwargs = recorder_cls.call_args
    assert kwargs["on_realtime_transcription_update"] is on_partial
    assert kwargs["enable_realtime_transcription"] is True


def test_recorder_construction_retries_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(listener_module, "INIT_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(listener_module, "INIT_RETRY_BACKOFF_SECONDS", 0.001)

    recorder = FakeRecorder()
    with (
        patch.object(listener_module, "AudioToTextRecorder") as recorder_cls,
        patch.object(listener_module, "_trust_silero_vad_repo"),
    ):
        recorder_cls.side_effect = [
            RuntimeError("network blip"),
            RuntimeError("network blip"),
            recorder,
        ]
        listener = Listener(on_final_text=lambda t: None)

    assert listener._recorder is recorder
    assert recorder_cls.call_count == 3


def test_recorder_construction_gives_up_after_max_attempts(monkeypatch) -> None:
    monkeypatch.setattr(listener_module, "INIT_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(listener_module, "INIT_RETRY_BACKOFF_SECONDS", 0.001)

    with (
        patch.object(listener_module, "AudioToTextRecorder") as recorder_cls,
        patch.object(listener_module, "_trust_silero_vad_repo"),
    ):
        recorder_cls.side_effect = RuntimeError("network blip")
        with pytest.raises(RuntimeError, match="network blip"):
            Listener(on_final_text=lambda t: None)

    assert recorder_cls.call_count == 2


def test_on_speech_start_fires_when_recording_starts() -> None:
    speech_start_calls: list[int] = []
    with (
        patch.object(listener_module, "AudioToTextRecorder") as recorder_cls,
        patch.object(listener_module, "_trust_silero_vad_repo"),
    ):
        recorder_cls.return_value = FakeRecorder()
        Listener(
            on_final_text=lambda t: None,
            on_speech_start=lambda: speech_start_calls.append(1),
        )

    _, kwargs = recorder_cls.call_args
    on_recording_start = kwargs["on_recording_start"]
    on_recording_start()  # simulate RealtimeSTT invoking it on VAD speech-start

    assert speech_start_calls == [1]


def test_stop_reports_failure_when_worker_thread_does_not_terminate(monkeypatch) -> None:
    monkeypatch.setattr(listener_module, "SHUTDOWN_JOIN_TIMEOUT_SECONDS", 0.05)
    recorder = StuckRecorder()
    listener = _make_listener(recorder, on_final_text=lambda t: None)

    listener.start()
    time.sleep(0.05)  # let the worker actually enter the blocking text() call

    assert listener.stop() is False
    # shutdown() must be skipped while the worker thread might still be using it.
    assert recorder.shutdown_called is False


def test_run_retries_with_backoff_then_gives_up(monkeypatch) -> None:
    monkeypatch.setattr(listener_module, "INITIAL_BACKOFF_SECONDS", 0.001)
    monkeypatch.setattr(listener_module, "MAX_BACKOFF_SECONDS", 0.005)
    monkeypatch.setattr(listener_module, "MAX_CONSECUTIVE_FAILURES", 3)

    recorder = FailingRecorder()
    fatal_calls: list[int] = []
    listener = _make_listener(
        recorder,
        on_final_text=lambda t: None,
        on_fatal_error=lambda: fatal_calls.append(1),
    )

    listener.start()
    assert _wait_until(lambda: not listener._thread.is_alive(), timeout=2.0)

    assert listener.failed is True
    assert recorder.call_count == 3  # bounded: stopped after MAX_CONSECUTIVE_FAILURES
    assert fatal_calls == [1]


def test_transient_failures_do_not_trip_the_fatal_threshold(monkeypatch) -> None:
    monkeypatch.setattr(listener_module, "INITIAL_BACKOFF_SECONDS", 0.001)
    monkeypatch.setattr(listener_module, "MAX_BACKOFF_SECONDS", 0.005)
    monkeypatch.setattr(listener_module, "MAX_CONSECUTIVE_FAILURES", 3)

    class FlakyThenFineRecorder:
        def __init__(self) -> None:
            self.call_count = 0
            self._unblocked = threading.Event()

        def text(self) -> str:
            self.call_count += 1
            if self.call_count <= 2:
                raise RuntimeError("transient glitch")
            self._unblocked.wait()
            return ""

        def abort(self) -> None:
            self._unblocked.set()

        def shutdown(self) -> None:
            pass

    recorder = FlakyThenFineRecorder()
    listener = _make_listener(recorder, on_final_text=lambda t: None)

    listener.start()
    assert _wait_until(lambda: recorder.call_count >= 3)
    time.sleep(0.05)
    assert listener.failed is False
    listener.stop()
