from __future__ import annotations

import logging
import queue
import threading
import time
from typing import cast

from ai_girlfriend.config import Settings
from ai_girlfriend.llm.brain import Brain
from ai_girlfriend.logging_setup import configure_logging
from ai_girlfriend.stt.listener import Listener
from ai_girlfriend.tts.speaker import Speaker

logger = logging.getLogger(__name__)

# Sentinel telling the LLM worker thread to exit its loop.
_SHUTDOWN = object()


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)

    if not settings.groq_api_key.get_secret_value():
        raise SystemExit(
            "GROQ_API_KEY is not set. Get a free key from "
            "https://console.groq.com/keys and add it to your .env file."
        )

    # Avatar (talking-head window) and Speaker (audio-only) share the same
    # say()/interrupt()/stop() interface — see docs/design/004-talking-head.md
    # — so everything below just calls `speaker` without caring which one it is.
    if settings.avatar_enabled:
        if not settings.avatar_image_path:
            raise SystemExit(
                "AVATAR_ENABLED is true but AVATAR_IMAGE_PATH is not set. "
                "Point it at a 2D image with a visible face in your .env file."
            )
        from ai_girlfriend.avatar.player import Avatar

        speaker: Speaker | Avatar = Avatar(
            image_path=settings.avatar_image_path,
            rhubarb_path=settings.rhubarb_path,
            timeout=settings.avatar_timeout_seconds,
        )
    else:
        speaker = Speaker(voice=settings.tts_voice, timeout=settings.tts_timeout_seconds)
    brain = Brain(
        api_key=settings.groq_api_key.get_secret_value(),
        model=settings.llm_model,
        system_prompt=settings.llm_system_prompt,
        history_turns=settings.llm_history_turns,
        timeout=settings.llm_timeout_seconds,
    )

    # Final transcripts are handed off to a queue rather than answered inline
    # in on_final_text: a Groq call takes real network time, and running it
    # on the STT callback thread would block Listener from picking up the
    # next utterance until the reply came back — the same blocking problem
    # TTS had before Speaker got its own worker thread.
    transcripts: queue.Queue[object] = queue.Queue()

    # Bumped every time the user starts speaking, so an in-flight LLM reply
    # for a turn the user has since talked over can recognize itself as
    # stale (see brain_worker below) instead of getting spoken late over
    # whatever she should be responding to now. Plain int, but guarded by a
    # lock rather than relying on the GIL, since it's written from
    # Listener's audio thread and read from brain_worker.
    turn_generation = 0
    generation_lock = threading.Lock()

    def on_speech_start() -> None:
        nonlocal turn_generation
        with generation_lock:
            turn_generation += 1
        speaker.interrupt()

    def on_final_text(text: str) -> None:
        logger.info("Heard: %s", text)
        with generation_lock:
            gen = turn_generation
        transcripts.put((text, gen))

    def on_partial_text(text: str) -> None:
        logger.debug("Partial: %s", text)

    def brain_worker() -> None:
        while True:
            item = transcripts.get()
            if item is _SHUTDOWN:
                return
            text, gen = cast("tuple[str, int]", item)
            t_start = time.monotonic()
            try:
                reply = brain.reply(text)
            except Exception:
                logger.exception("LLM reply failed")
                continue
            logger.info(
                "Latency: LLM reply = %.0fms", (time.monotonic() - t_start) * 1000
            )
            with generation_lock:
                stale = gen != turn_generation
            if stale:
                logger.info("Dropping reply for interrupted turn: %r", reply)
                continue
            speaker.say(reply)

    brain_thread = threading.Thread(target=brain_worker, daemon=True)
    brain_thread.start()

    # Blocks until Ctrl+C; nothing else sets this event under normal operation.
    # All real work happens on Listener's/brain's background threads, so the
    # main thread just waits. If the listener gives up after repeated
    # failures, it sets this too, so main() doesn't hang forever with a dead
    # listener.
    shutdown_requested = threading.Event()

    def on_fatal_error() -> None:
        logger.error("Listener stopped after repeated STT failures; shutting down.")
        shutdown_requested.set()

    listener = Listener(
        on_final_text=on_final_text,
        on_partial_text=on_partial_text,
        on_fatal_error=on_fatal_error,
        on_speech_start=on_speech_start,
        model=settings.stt_model,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
        language=settings.stt_language,
    )

    logger.info("Listening... speak into your microphone (Ctrl+C to stop).")
    listener.start()
    try:
        shutdown_requested.wait()
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        if not listener.stop():
            logger.error("Listener did not shut down cleanly")
        transcripts.put(_SHUTDOWN)
        brain_thread.join(timeout=5)
        if brain_thread.is_alive():
            logger.error("LLM worker thread did not stop within 5s")
        speaker.stop()


if __name__ == "__main__":
    main()
