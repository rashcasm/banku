"""Configuration. Everything tunable lives here and comes from the environment.

The model names are deliberately in config, not hardcoded, so the STT engine,
the LLM and the TTS voice can each be swapped without touching logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _path(name: str, default: Path) -> Path:
    return Path(os.getenv(name) or default).expanduser()


# Per-provider defaults, so switching is one env var. Both speak the OpenAI
# chat-completions shape; only auth and the base URL differ.
_LLM_DEFAULTS = {
    "sarvam": ("sarvam-105b", "https://api.sarvam.ai/v2/chat/completions"),
    "groq": ("openai/gpt-oss-20b", "https://api.groq.com/openai/v1/chat/completions"),
}


def _llm_provider() -> str:
    return os.getenv("BANKU_LLM_PROVIDER", "sarvam").strip().lower()


def _llm_default(index: int) -> str:
    model, endpoint = _LLM_DEFAULTS.get(_llm_provider(), _LLM_DEFAULTS["sarvam"])
    return (model, endpoint)[index]


def _llm_key() -> str:
    if _llm_provider() == "groq":
        return os.getenv("GROQ_API_KEY", "")
    return os.getenv("SARVAM_API_KEY", "")


@dataclass(frozen=True)
class LLMConfig:
    provider: str = field(default_factory=_llm_provider)
    model: str = field(
        default_factory=lambda: os.getenv("BANKU_LLM_MODEL") or _llm_default(0)
    )
    endpoint: str = field(
        default_factory=lambda: os.getenv("BANKU_LLM_ENDPOINT") or _llm_default(1)
    )
    api_key: str = field(default_factory=_llm_key)
    temperature: float = float(os.getenv("BANKU_LLM_TEMPERATURE", "0.2"))
    # This model spends most of its budget on internal reasoning before it says
    # anything. Measured: 400-430 reasoning tokens on an ordinary question, so a
    # 400-token cap produced finish_reason="length" with content=None — the
    # assistant thought, ran out, and stayed silent. 1600 leaves room for the
    # thought and the sentence.
    max_tokens: int = int(os.getenv("BANKU_LLM_MAX_TOKENS", "1600"))
    # Measured on the tool-calling path: "low" answers in 1.3-1.5 s versus 1.6-1.7
    # s by default, and with far less variance. Thinking longer buys nothing for
    # "where is the nail cutter".
    reasoning_effort: str = os.getenv("BANKU_LLM_REASONING", "low")
    timeout_s: float = float(os.getenv("BANKU_LLM_TIMEOUT", "40"))
    # Guard against a tool-calling loop burning money and time.
    max_tool_rounds: int = int(os.getenv("BANKU_LLM_MAX_TOOL_ROUNDS", "3"))

    def headers(self) -> dict[str, str]:
        """Auth differs per provider; the rest of the request is OpenAI-shaped."""
        if self.provider in ("groq", "openai", "cerebras"):
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        # Sarvam uses its own header name.
        return {"api-subscription-key": self.api_key, "Content-Type": "application/json"}


@dataclass(frozen=True)
class TTSConfig:
    """How the phone should speak. Returned to the phone on every reply so the
    language of the voice lives in config on this machine, not on the phone."""

    lang: str = os.getenv("BANKU_TTS_LANG", "en")
    region: str = os.getenv("BANKU_TTS_REGION", "IN")
    rate: float = float(os.getenv("BANKU_TTS_RATE", "1.0"))
    engine: str = os.getenv("BANKU_TTS_ENGINE", "com.google.android.tts")


@dataclass(frozen=True)
class ServerConfig:
    host: str = os.getenv("BANKU_HOST", "0.0.0.0")
    port: int = int(os.getenv("BANKU_PORT", "8765"))


@dataclass(frozen=True)
class STTConfig:
    # saaras:v4 is the current default and supports code-mixed output + keyterms.
    provider: str = os.getenv("BANKU_STT_PROVIDER", "sarvam")
    model: str = os.getenv("BANKU_STT_MODEL", "saaras:v4")
    language: str = os.getenv("BANKU_STT_LANGUAGE", "mr-IN")
    # "codemix" keeps Marathi in Devanagari and English words in Latin script.
    # Aai and Baba will say "TV unit", "nail cutter", so this matters.
    mode: str = os.getenv("BANKU_STT_MODE", "codemix")
    api_key: str = os.getenv("SARVAM_API_KEY", "")
    endpoint: str = os.getenv("BANKU_STT_ENDPOINT", "https://api.sarvam.ai/speech-to-text")
    timeout_s: float = float(os.getenv("BANKU_STT_TIMEOUT", "60"))
    # Lift quiet far-field speech to a usable level before sending.
    normalize: bool = os.getenv("BANKU_STT_NORMALIZE", "1") == "1"
    # Below this peak level we assume nobody is talking and skip the API call
    # entirely, so a silent room cannot come back as fluent invented Marathi.
    min_peak_db: float = float(os.getenv("BANKU_STT_MIN_PEAK_DB", "-38"))


@dataclass(frozen=True)
class Paths:
    db: Path = field(default_factory=lambda: _path("BANKU_DB", ROOT / "data" / "banku.db"))
    recordings: Path = field(
        default_factory=lambda: _path("BANKU_RECORDINGS", ROOT / "data" / "recordings")
    )


@dataclass(frozen=True)
class MatchConfig:
    """Fuzzy matching knobs for the item lookup.

    The dataset is a few hundred aliases at most, so we score in Python rather
    than reaching for FTS or embeddings. That stays accurate and debuggable.
    """

    threshold: int = int(os.getenv("BANKU_MATCH_THRESHOLD", "78"))
    ambiguous_gap: int = int(os.getenv("BANKU_MATCH_GAP", "6"))


@dataclass(frozen=True)
class LanguageConfig:
    """The MVP runs in English so mistakes are obvious to the person testing.

    Flipping back to Marathi is one variable: BANKU_LANG=mr plus the matching
    BANKU_STT_LANGUAGE / BANKU_TTS_LANG. Nothing else in the code changes.
    """

    lang: str = os.getenv("BANKU_LANG", "en")

    @property
    def is_english(self) -> bool:
        return self.lang.lower().startswith("en")


stt = STTConfig()
llm = LLMConfig()
tts = TTSConfig()
language = LanguageConfig()
server = ServerConfig()
paths = Paths()
match = MatchConfig()
