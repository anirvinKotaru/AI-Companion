# 002 — Real-time text-to-speech

## Problem
The companion app needs a voice: given text (from the chatbot, or for now an echo of what the user said), speak it out loud with low enough latency to feel like a live conversation, running on this machine (no NVIDIA GPU, no API keys, no cost).

## Goals
- Speak arbitrary text out loud through the default audio output.
- Keep the TTS engine fully decoupled from whatever produces the text, via a small `say(text)` interface — mirrors how `Listener` decouples STT.
- Make the voice configurable without code changes (`.env`), so a different engine can be swapped in later without touching callers.

## Non-goals
- Streaming word-by-word playback while the chatbot is still generating (can be revisited later).
- Voice cloning or emotional/expressive TTS.
- Any chatbot logic — this module only speaks text it's given.

## Chosen approach
Talk to Windows SAPI5 directly via `win32com.client` (pywin32). It's wrapped in `ai_girlfriend.tts.speaker.Speaker`, which is the only file that imports `win32com` — every other module just calls `speaker.say(text)`.

This matches the constraint already set for STT: fully local, zero cost, no API key, no GPU. Quality is more robotic than a neural voice, but latency is minimal and there's nothing to download or configure. A single `SAPI.SpVoice` COM object is created once per `Speaker` and reused for every `say()` call — that's SAPI5's intended usage pattern (it's built to handle repeated `Speak()` calls, e.g. screen readers keep one instance alive for a whole session).

## Interface contract
```python
from ai_girlfriend.tts.speaker import Speaker

speaker = Speaker(voice="")  # voice name/substring from .env, "" = OS default
speaker.say("Hello there!")  # blocks until playback finishes
...
speaker.stop()                # purge the speech queue, stopping mid-utterance
```
This is the seam a future chatbot module plugs into — it never needs to know SAPI5 exists.

## Alternatives considered
- **RealtimeTTS's `SystemEngine` (pyttsx3-based)**: tried first, since `RealtimeTTS` is the sibling project to `RealtimeSTT` (same author) and shares its conventions. Rejected after live testing: it reliably works for exactly one `say()` call per process, then either silently drops audio or hangs the calling thread indefinitely on the next call — a known pyttsx3-on-Windows issue with reusing (or even freshly re-creating) its SAPI5 engine within one process. Talking to SAPI5 directly sidesteps pyttsx3 entirely and was confirmed reliable across many repeated calls in the same process.
- **Coqui (local neural TTS)**: much more natural voice, fully offline, but a large model download (~1GB+) and noticeably slower on CPU-only hardware — a latency/quality tradeoff not worth making until the system voice proves too robotic.
- **edge-tts**: free, good quality, no API key — but requires network access, breaking the "fully local" property the STT side already has.
- **OpenAI TTS**: highest quality, but costs money per request and needs an API key and network — ruled out to keep the whole pipeline free and offline for now.

## Future work
- Swap in `on_partial_text` from `Listener` for a full mic → STT → TTS echo demo once useful (`main.py` currently does this for validation).
- Reassess engine choice (e.g. Coqui) once voice quality is validated by ear against the robotic system voice.
- Wire the chatbot's output into `Speaker.say()` once the chatbot module exists, replacing the echo in `main.py`.
