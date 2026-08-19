from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from ai_girlfriend.tts.speaker import SVSF_FLAGS_ASYNC, SVSF_PURGE_BEFORE_SPEAK, Speaker


@pytest.fixture
def sapi_voice():
    """Patch out real COM/SAPI5 so tests never actually speak out loud."""
    voice = MagicMock()
    voice.WaitUntilDone.return_value = True  # phrases "finish" instantly by default
    voice.GetVoices.return_value = []
    with (
        patch("ai_girlfriend.tts.speaker.pythoncom.CoInitialize"),
        patch("ai_girlfriend.tts.speaker.win32com.client.Dispatch", return_value=voice),
    ):
        yield voice


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_say_dispatches_text_to_sapi(sapi_voice) -> None:
    speaker = Speaker()
    speaker.say("hello world")
    assert _wait_until(lambda: sapi_voice.Speak.called)
    speaker.stop()
    sapi_voice.Speak.assert_any_call("hello world", SVSF_FLAGS_ASYNC)


def test_say_ignores_empty_text(sapi_voice) -> None:
    speaker = Speaker()
    speaker.say("")
    speaker.stop()
    sapi_voice.Speak.assert_not_called()


def test_say_does_not_block_caller_even_when_tts_is_slow(sapi_voice) -> None:
    release = threading.Event()
    sapi_voice.WaitUntilDone.side_effect = lambda ms: release.wait(timeout=ms / 1000) or False

    speaker = Speaker()
    start = time.monotonic()
    speaker.say("this would hang for a while")
    elapsed = time.monotonic() - start

    assert elapsed < 0.2  # say() must return ~immediately regardless of TTS speed
    release.set()
    speaker.stop()


def test_multiple_phrases_are_spoken_in_order(sapi_voice) -> None:
    speaker = Speaker()
    speaker.say("first")
    speaker.say("second")
    speaker.say("third")
    assert _wait_until(lambda: sapi_voice.Speak.call_count >= 3)
    speaker.stop()

    spoken = [call.args[0] for call in sapi_voice.Speak.call_args_list if call.args[0]]
    assert spoken == ["first", "second", "third"]


def test_hung_tts_call_times_out_and_purges(sapi_voice) -> None:
    sapi_voice.WaitUntilDone.return_value = False  # never finishes on its own

    speaker = Speaker(timeout=0.05)
    speaker.say("stuck forever")
    assert _wait_until(
        lambda: any(
            call.args == ("", SVSF_PURGE_BEFORE_SPEAK) for call in sapi_voice.Speak.call_args_list
        ),
        timeout=2.0,
    )
    speaker.stop()


def test_stop_drops_queued_phrases_and_purges_current_playback(sapi_voice) -> None:
    hold = threading.Event()
    sapi_voice.WaitUntilDone.side_effect = lambda ms: hold.wait(timeout=ms / 1000) and False

    speaker = Speaker()
    speaker.say("first")
    speaker.say("second")
    speaker.say("third")
    assert _wait_until(lambda: sapi_voice.Speak.called)  # "first" is now in flight

    speaker.stop()

    spoken = [call.args[0] for call in sapi_voice.Speak.call_args_list if call.args[0]]
    assert "second" not in spoken
    assert "third" not in spoken


def test_interrupt_purges_current_playback_and_drops_queue(sapi_voice) -> None:
    hold = threading.Event()
    sapi_voice.WaitUntilDone.side_effect = lambda ms: hold.wait(timeout=ms / 1000) and False

    speaker = Speaker()
    speaker.say("first")
    speaker.say("second")
    assert _wait_until(lambda: sapi_voice.Speak.called)  # "first" is now in flight

    speaker.interrupt()

    assert _wait_until(
        lambda: any(
            call.args == ("", SVSF_PURGE_BEFORE_SPEAK) for call in sapi_voice.Speak.call_args_list
        )
    )
    spoken = [call.args[0] for call in sapi_voice.Speak.call_args_list if call.args[0]]
    assert "second" not in spoken
    speaker.stop()


def test_interrupt_leaves_worker_alive_for_future_phrases(sapi_voice) -> None:
    speaker = Speaker()
    speaker.say("first")
    assert _wait_until(lambda: sapi_voice.Speak.called)

    speaker.interrupt()
    speaker.say("after interruption")

    assert _wait_until(
        lambda: any(
            call.args[0] == "after interruption" for call in sapi_voice.Speak.call_args_list
        )
    )
    speaker.stop()
