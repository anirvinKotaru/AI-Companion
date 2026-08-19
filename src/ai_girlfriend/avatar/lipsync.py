from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VisemeCue:
    """One entry from Rhubarb's mouthCues timeline: `shape` from `start` to `end` (seconds)."""

    start: float
    end: float
    shape: str


def run_rhubarb(
    rhubarb_path: str,
    wav_path: str,
    dialog_text: str = "",
    timeout: float = 15.0,
) -> list[VisemeCue]:
    """Run Rhubarb Lip Sync on `wav_path`, returning its timestamped mouth cues.

    `dialog_text` is passed via Rhubarb's `-d`/`--dialogFile` option — per its
    own README, this "usually leads to more reliable mouth animation, even if
    the text is not completely accurate."

    Raises FileNotFoundError if `rhubarb_path` isn't runnable, and
    subprocess.CalledProcessError/TimeoutExpired on failure — callers are
    expected to catch and log, the same convention Brain.reply() uses.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "cues.json"
        # str(Path(...)) normalizes slash direction: on Windows, CreateProcess
        # (via subprocess) fails to locate a relative executable path that
        # uses forward slashes, even though os.path.exists() accepts them.
        args = [str(Path(rhubarb_path)), "-f", "json", "-o", str(output_path)]
        if dialog_text:
            dialog_path = Path(tmp_dir) / "dialog.txt"
            dialog_path.write_text(dialog_text, encoding="utf-8")
            args += ["-d", str(dialog_path)]
        args.append(str(Path(wav_path)))

        subprocess.run(args, capture_output=True, timeout=timeout, check=True, text=True)
        data = json.loads(output_path.read_text(encoding="utf-8"))

    return [
        VisemeCue(start=cue["start"], end=cue["end"], shape=cue["value"])
        for cue in data["mouthCues"]
    ]
