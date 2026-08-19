# ai-girlfriend

A real-time voice companion app: continuous microphone-to-text transcription (the ear) feeds a Groq-hosted LLM (the brain), whose reply is spoken back out loud (the mouth).

## Status
- [x] Real-time speech-to-text (`src/ai_girlfriend/stt/`)
- [x] Chatbot (`src/ai_girlfriend/llm/`) — Groq API, free tier
- [x] Text-to-speech / voice output (`src/ai_girlfriend/tts/`)
- [x] Talking-head avatar, optional (`src/ai_girlfriend/avatar/`) — animates a 2D image you upload

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
copy .env.example .env
```

Runs on CPU by default (`STT_DEVICE=cpu` in `.env`) — no GPU required. If you have an NVIDIA GPU with CUDA installed, set `STT_DEVICE=cuda` for lower latency. TTS uses the OS's built-in voices (no GPU, no API key, no network needed either).

The chat layer needs a free Groq API key:
1. Sign up at [console.groq.com](https://console.groq.com) (no credit card required for the free tier).
2. Create a key at [console.groq.com/keys](https://console.groq.com/keys).
3. Paste it into `.env` as `GROQ_API_KEY=...`.

Without a key set, `python -m ai_girlfriend.main` exits immediately with a message telling you where to get one, instead of failing deep inside a network call.

### Optional: talking-head avatar
By default the app is audio-only. To show a window that animates a 2D image's mouth in sync with speech instead (still fully local, no GPU needed — see [docs/design/004-talking-head.md](docs/design/004-talking-head.md) for how):
1. Download `rhubarb.exe` for Windows from the [Rhubarb Lip Sync releases page](https://github.com/DanielSWolf/rhubarb-lip-sync/releases) and unzip it somewhere.
2. In `.env`, set `AVATAR_ENABLED=true`, `AVATAR_IMAGE_PATH=path\to\your\image.png` (a clear, front-facing face/mouth works best), and `RHUBARB_PATH=path\to\rhubarb.exe`.
3. The first run also downloads a small (~4MB) MediaPipe face-landmark model automatically.

This is a cheap geometric warp of your single image, not a rendered video — expect a stylized, visibly-synthetic "talking photo" look rather than photorealism.

## Run the voice demo
```
python -m ai_girlfriend.main
```
Speak into your microphone; the finalized transcript is sent to the LLM, and its reply is logged and spoken back out loud. Press Ctrl+C to stop.

## Development
```
pytest                        # unit tests
pre-commit run --all-files    # lint, format, type-check
```

See [docs/design/001-realtime-stt.md](docs/design/001-realtime-stt.md), [docs/design/002-realtime-tts.md](docs/design/002-realtime-tts.md), [docs/design/003-llm-brain.md](docs/design/003-llm-brain.md), and [docs/design/004-talking-head.md](docs/design/004-talking-head.md) for the design rationale behind the speech-to-text, text-to-speech, chat, and avatar modules.
