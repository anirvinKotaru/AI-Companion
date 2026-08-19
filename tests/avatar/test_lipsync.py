from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_girlfriend.avatar.lipsync import VisemeCue, run_rhubarb


def _fake_rhubarb(cues=(), capture: dict | None = None):
    """A subprocess.run stand-in that writes Rhubarb's expected JSON to `-o`'s path."""

    def _run(args, **kwargs):
        if capture is not None:
            capture["args"] = args
            if "-d" in args:
                capture["dialog_text"] = Path(args[args.index("-d") + 1]).read_text(
                    encoding="utf-8"
                )
        output_path = Path(args[args.index("-o") + 1])
        output_path.write_text(
            json.dumps({"metadata": {}, "mouthCues": list(cues)}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(args, 0)

    return _run


def test_parses_mouth_cues_into_viseme_cues() -> None:
    cues = [
        {"start": 0.0, "end": 0.05, "value": "X"},
        {"start": 0.05, "end": 0.27, "value": "D"},
    ]
    with patch("ai_girlfriend.avatar.lipsync.subprocess.run", side_effect=_fake_rhubarb(cues)):
        result = run_rhubarb("rhubarb.exe", "speech.wav")

    assert result == [
        VisemeCue(start=0.0, end=0.05, shape="X"),
        VisemeCue(start=0.05, end=0.27, shape="D"),
    ]


def test_dialog_text_is_passed_via_dash_d_file() -> None:
    capture: dict = {}
    with patch(
        "ai_girlfriend.avatar.lipsync.subprocess.run", side_effect=_fake_rhubarb((), capture)
    ):
        run_rhubarb("rhubarb.exe", "speech.wav", dialog_text="hello there")

    assert capture["dialog_text"] == "hello there"


def test_no_dash_d_flag_when_dialog_text_is_empty() -> None:
    capture: dict = {}
    with patch(
        "ai_girlfriend.avatar.lipsync.subprocess.run", side_effect=_fake_rhubarb((), capture)
    ):
        run_rhubarb("rhubarb.exe", "speech.wav", dialog_text="")

    assert "-d" not in capture["args"]


def test_wav_path_is_the_last_argument() -> None:
    capture: dict = {}
    with patch(
        "ai_girlfriend.avatar.lipsync.subprocess.run", side_effect=_fake_rhubarb((), capture)
    ):
        run_rhubarb("rhubarb.exe", "speech.wav")

    assert capture["args"][-1] == "speech.wav"


def test_nonzero_exit_raises() -> None:
    error = subprocess.CalledProcessError(1, ["rhubarb.exe"])
    with (
        patch("ai_girlfriend.avatar.lipsync.subprocess.run", side_effect=error),
        pytest.raises(subprocess.CalledProcessError),
    ):
        run_rhubarb("rhubarb.exe", "speech.wav")


def test_timeout_raises() -> None:
    error = subprocess.TimeoutExpired(["rhubarb.exe"], timeout=1.0)
    with (
        patch("ai_girlfriend.avatar.lipsync.subprocess.run", side_effect=error),
        pytest.raises(subprocess.TimeoutExpired),
    ):
        run_rhubarb("rhubarb.exe", "speech.wav", timeout=1.0)
