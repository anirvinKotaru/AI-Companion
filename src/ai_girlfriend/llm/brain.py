from __future__ import annotations

import logging
from collections import deque

from groq import Groq

logger = logging.getLogger(__name__)

# Kept short since it's read aloud by TTS. Mirrors the default in config.py
# (kept independent, same as Listener/Speaker's own defaults) — update both
# if you change the persona.
DEFAULT_SYSTEM_PROMPT = (
    "You are a warm, playful, supportive girlfriend having a live spoken "
    "conversation. Keep replies short, 1 to 3 sentences, since a "
    "text-to-speech voice reads them aloud."
)


class Brain:
    """Turns a transcript into a reply using the Groq chat completions API.

    The only file in the project that imports `groq` — everything else calls
    `reply()`. Keeps a rolling window of the last `history_turns` user/
    assistant exchanges so replies read as a continuous conversation instead
    of isolated Q&A.

    `reply()` blocks on a network call and is not thread-safe to call
    concurrently (history mutation isn't locked) — callers should invoke it
    from a single dedicated thread, same as `Listener` owns its recorder.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "llama-3.1-8b-instant",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        history_turns: int = 6,
        timeout: float = 10.0,
    ) -> None:
        self._client = Groq(api_key=api_key, timeout=timeout)
        self._model = model
        self._system_prompt = system_prompt
        self._history: deque[dict[str, str]] = deque(maxlen=max(history_turns, 0) * 2)

    def reply(self, text: str) -> str:
        """Send `text` plus recent history to the model and return its reply.

        Raises whatever the Groq client raises (timeout, API error, etc.) on
        failure — callers are expected to catch and log, same as the
        STT/TTS callbacks do.
        """
        messages = [
            {"role": "system", "content": self._system_prompt},
            *self._history,
            {"role": "user", "content": text},
        ]
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
        )
        reply_text = completion.choices[0].message.content or ""
        self._history.append({"role": "user", "content": text})
        self._history.append({"role": "assistant", "content": reply_text})
        return reply_text
