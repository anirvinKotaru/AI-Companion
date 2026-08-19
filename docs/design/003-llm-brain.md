# 003 — LLM brain

## Problem
The companion app can already hear (STT) and speak (TTS), but has no brain: `main.py` just echoes back whatever the user said. It needs something that turns a transcript into a reply, fast enough to keep the conversation feeling live, without a GPU or ongoing cost.

## Goals
- Turn a final transcript into a reply through a small `reply(text) -> str` interface, mirroring how `Listener` and `Speaker` decouple their engines from the rest of the app.
- Keep enough recent conversation history that replies feel continuous rather than isolated Q&A turns.
- Fail fast and clearly if the API key isn't configured, rather than failing deep inside a network call.
- Never block the STT callback thread on a network call — the same problem `Speaker.say()` had before it got a worker thread.

## Non-goals
- Streaming token-by-token replies (can be revisited once word-by-word TTS playback is worth building).
- Persistent memory across process restarts — history is in-memory only, lost on exit.
- Prompt-engineering the persona in depth; `LLM_SYSTEM_PROMPT` is a placeholder to be tuned by ear later.

## Chosen approach
Talk to the [Groq API](https://console.groq.com) via its official `groq` Python SDK. It's wrapped in `ai_girlfriend.llm.brain.Brain`, the only file that imports `groq` — everything else calls `reply()`.

Groq was chosen over the alternatives considered for 001/002's reasons — free, no GPU needed — plus:
- **Genuinely free tier**: rate-limited but no credit card required, unlike most API-based LLM providers.
- **Low latency**: Groq runs open models (Llama, etc.) on custom inference hardware built for speed, which matters for a "feels realtime" conversation loop, since a cloud LLM call is what a local, GPU-free machine can't do at comparable speed itself.
- Default model `llama-3.1-8b-instant` favors latency over the larger `llama-3.3-70b-versatile`, consistent with `base.en`'s tradeoff in the STT config — swappable via `.env` if replies feel shallow.

`Brain` keeps a `deque(maxlen=2 * history_turns)` of the most recent user/assistant message pairs and resends them as context on every call — the simplest form of conversation memory, no database needed yet.

`main.py` fails fast with `SystemExit` if `GROQ_API_KEY` is unset, before starting the listener, and dispatches transcripts to `Brain.reply()` on a dedicated worker thread (queue-fed, same pattern as `Speaker`'s worker) rather than calling it inline from the STT callback thread. A hung or slow Groq call would otherwise block `Listener` from picking up the next utterance.

### Barge-in
When the user starts talking again — whether to interrupt her mid-reply or just to say the next thing — `Listener`'s `on_speech_start` fires and `main.py` does two things: calls `Speaker.interrupt()` (cuts off whatever's currently playing, drops anything queued, but keeps the worker thread alive for future `say()` calls — distinct from `Speaker.stop()`, which shuts it down for good) and bumps a `turn_generation` counter. Each transcript is tagged with the generation active when it was heard; when `brain_worker` gets a reply back, it checks whether that generation is still current before speaking it. If the user has since started another utterance, the generation has moved on and the stale reply is logged and dropped instead of being spoken over whatever she should be responding to now.

## Interface contract
```python
from ai_girlfriend.llm.brain import Brain

brain = Brain(api_key="...", model="llama-3.1-8b-instant")
reply = brain.reply("hey, how's it going?")  # blocks on the network call
```
`reply()` raises on failure (timeout, API error) rather than swallowing it — callers are expected to catch and log, the same convention `Listener`/`Speaker` use for their own callbacks.

## Alternatives considered
- **character.ai**: no official public API. The only integration path is unofficial reverse-engineered clients that scrape the web client — against character.ai's Terms of Service, fragile (breaks whenever their frontend changes), and risks account bans. Ruled out.
- **Google Gemini (AI Studio free tier)**: also genuinely free and a reasonable choice; Groq's speed edge was the deciding factor for a live-conversation feel.
- **Local LLM via Ollama**: would match the fully-offline philosophy of `Listener`/`Speaker`, but this machine is CPU-only — a locally-run model small enough to be fast would likely trade too much quality, and larger models would be slow enough to hurt the realtime feel. Worth revisiting if a GPU becomes available.
- **OpenRouter free models**: viable fallback if Groq's rate limits prove too tight in practice.

## Future work
- Tune `LLM_SYSTEM_PROMPT` by ear once voice + reply quality can be evaluated together.
- Consider streaming replies sentence-by-sentence to `Speaker.say()` as they arrive, once TTS latency (not LLM latency) becomes the bottleneck.
- Reassess model choice/provider if free-tier rate limits are hit in normal use.
