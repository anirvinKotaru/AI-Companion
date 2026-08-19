from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_girlfriend.config import Settings


def test_valid_stt_device_accepted() -> None:
    assert Settings(stt_device="cpu").stt_device == "cpu"
    assert Settings(stt_device="cuda").stt_device == "cuda"
    assert Settings(stt_device="auto").stt_device == "auto"


def test_invalid_stt_device_rejected_immediately() -> None:
    with pytest.raises(ValidationError, match="stt_device"):
        Settings(stt_device="quantum")


def test_valid_stt_compute_type_accepted() -> None:
    assert Settings(stt_compute_type="int8").stt_compute_type == "int8"
    assert Settings(stt_compute_type="float32").stt_compute_type == "float32"


def test_invalid_stt_compute_type_rejected_immediately() -> None:
    with pytest.raises(ValidationError, match="stt_compute_type"):
        Settings(stt_compute_type="ultra-precise")


def test_groq_api_key_defaults_empty() -> None:
    # _env_file=None isolates this from a real developer .env (which legitimately
    # has a real key in it) so this actually tests the field's class-level default.
    assert Settings(_env_file=None).groq_api_key.get_secret_value() == ""  # type: ignore[call-arg]


def test_groq_api_key_is_redacted_in_repr() -> None:
    settings = Settings(groq_api_key="super-secret-value")
    assert "super-secret-value" not in repr(settings)
    assert settings.groq_api_key.get_secret_value() == "super-secret-value"
