from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai_girlfriend.llm.brain import Brain


def _completion(content: str) -> SimpleNamespace:
    """Build a fake object shaped like a Groq ChatCompletion response."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def groq_client():
    """Patch out the real Groq client so tests never hit the network."""
    client = MagicMock()
    with patch("ai_girlfriend.llm.brain.Groq", return_value=client):
        yield client


def test_reply_returns_model_content(groq_client) -> None:
    groq_client.chat.completions.create.return_value = _completion("hey you")
    brain = Brain(api_key="test-key")

    assert brain.reply("hi") == "hey you"


def test_reply_sends_system_prompt_and_user_message(groq_client) -> None:
    groq_client.chat.completions.create.return_value = _completion("hey you")
    brain = Brain(api_key="test-key", system_prompt="be nice", model="some-model")

    brain.reply("hi there")

    _, kwargs = groq_client.chat.completions.create.call_args
    assert kwargs["model"] == "some-model"
    assert kwargs["messages"][0] == {"role": "system", "content": "be nice"}
    assert kwargs["messages"][-1] == {"role": "user", "content": "hi there"}


def test_reply_includes_prior_turns_in_next_call(groq_client) -> None:
    groq_client.chat.completions.create.side_effect = [
        _completion("first reply"),
        _completion("second reply"),
    ]
    brain = Brain(api_key="test-key")

    brain.reply("first message")
    brain.reply("second message")

    _, kwargs = groq_client.chat.completions.create.call_args
    messages = kwargs["messages"]
    assert {"role": "user", "content": "first message"} in messages
    assert {"role": "assistant", "content": "first reply"} in messages
    assert messages[-1] == {"role": "user", "content": "second message"}


def test_history_is_trimmed_to_configured_turns(groq_client) -> None:
    groq_client.chat.completions.create.return_value = _completion("ok")
    brain = Brain(api_key="test-key", history_turns=1)

    brain.reply("one")
    brain.reply("two")
    brain.reply("three")

    _, kwargs = groq_client.chat.completions.create.call_args
    messages = kwargs["messages"]
    # system prompt + 1 turn of history (user+assistant) + the new user message
    assert len(messages) == 4
    assert {"role": "user", "content": "one"} not in messages


def test_reply_propagates_api_errors(groq_client) -> None:
    groq_client.chat.completions.create.side_effect = RuntimeError("network exploded")
    brain = Brain(api_key="test-key")

    with pytest.raises(RuntimeError, match="network exploded"):
        brain.reply("hi")
