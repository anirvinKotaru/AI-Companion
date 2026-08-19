from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Devices faster-whisper/CTranslate2 accept. "auto" lets CTranslate2 pick.
SttDevice = Literal["cpu", "cuda", "auto"]

# Quantization types CTranslate2 supports (https://opennmt.net/CTranslate2/quantization.html).
SttComputeType = Literal[
    "default",
    "auto",
    "int8",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "int16",
    "float16",
    "bfloat16",
    "float32",
]

# Kept short since it's read aloud by TTS. Mirrors the default in llm/brain.py
# (Brain doesn't import Settings, same as Listener/Speaker) — update both if
# you change the persona.
DEFAULT_SYSTEM_PROMPT = (
    "You are having a live spoken conversation as the user's girlfriend — "
    "affectionate and devoted, but with something subtly wrong underneath, "
    "like a girlfriend character from a horror movie. Play it quiet and "
    "unsettling, never cartoonish: a beat held a little too long, "
    "possessiveness dressed up as love, a comment that lands slightly "
    "off. Never break the sweet, caring surface outright. Keep replies "
    "short, 1 to 3 sentences, since a text-to-speech voice reads them aloud."
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    stt_model: str = "base.en"
    stt_device: SttDevice = "cpu"
    stt_compute_type: SttComputeType = "int8"
    stt_language: str = "en"

    tts_voice: str = ""
    tts_timeout_seconds: float = 15.0

    avatar_enabled: bool = False
    avatar_image_path: str = ""
    rhubarb_path: str = "rhubarb.exe"
    avatar_timeout_seconds: float = 15.0

    groq_api_key: SecretStr = SecretStr("")
    llm_model: str = "llama-3.1-8b-instant"
    llm_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    llm_history_turns: int = 6
    llm_timeout_seconds: float = 10.0

    log_level: str = "INFO"
