# 001 — Real-time speech-to-text

## Problem
The companion app needs to hear the user speak and turn that into text a chatbot can respond to, with low enough latency to feel like a live conversation, running on this machine (no NVIDIA GPU — CPU only).

## Goals
- Continuously listen to the default microphone and produce finalized transcripts of each utterance.
- Keep the speech-to-text engine fully decoupled from whatever consumes the text (chatbot, logger, etc.) via a small callback interface.
- Make model/device/quantization configurable without code changes, so behavior can be tuned (e.g. if a GPU is added later) via `.env`.

## Non-goals
- Wake-word detection, multi-speaker diarization, or non-English support (can be revisited later; `stt_language` is already a config knob).
- Any chatbot or TTS logic — this module only produces text.

## Chosen approach
Build on [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT), which already implements microphone capture, voice-activity detection, and streaming transcription via `faster-whisper`. It's wrapped in `ai_girlfriend.stt.listener.Listener`, which is the only file that imports `RealtimeSTT` — every other module talks to `Listener`'s plain callback interface instead.

Configured for CPU: `faster-whisper` backend, `device="cpu"`, `compute_type="int8"`, `model="base.en"` by default (upgradable to `small.en` if accuracy isn't sufficient — this is a latency/accuracy tradeoff to validate by ear, not something to guess up front).

## Interface contract
```python
from ai_girlfriend.stt.listener import Listener

def on_final_text(text: str) -> None:
    ...  # e.g. send to chatbot

listener = Listener(on_final_text=on_final_text)
listener.start()
...
listener.stop()
```
`on_partial_text` is available for live/partial captions if a UI wants them later. This is the seam a future chatbot module plugs into — it never needs to know Whisper or RealtimeSTT exist.

## Alternatives considered
- **whisper_streaming**: a research-oriented streaming wrapper around `faster-whisper` with local-agreement buffering. More flexible but requires hand-rolling mic capture and VAD — more glue code for no benefit here.
- **OpenAI Whisper API**: simplest integration, but costs money per request, needs network, and adds latency for a feature meant to feel like live conversation. Local RealtimeSTT was chosen instead since this machine can run a small model on CPU.

## Future work
- Wire `on_final_text` into the chatbot module once it exists.
- Reassess model size / device once a GPU is available or if `base.en` proves inaccurate.
