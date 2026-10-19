"""
config.py — centralised settings, loaded from environment variables or .env.

Nothing here is required at import time, so tests and `make ingest` work
without API keys.  Keys are checked only when something actually needs them:
ANTHROPIC_API_KEY for answers, OPENAI_API_KEY for the voice app's speech.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings."""

    # ── Anthropic (answer generation) ─────────────────────────────────────────
    anthropic_api_key: str = ""
    answer_model: str = "claude-haiku-5-5"
    max_answer_tokens: int = 1024

    # ── OpenAI (voice app only: Whisper speech-to-text + text-to-speech) ──────
    openai_api_key: str = ""
    transcribe_model: str = "whisper-1"
    tts_model: str = "tts-1"
    tts_voice: str = "nova"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    docs_dir: Path = Path("data/docs")
    index_dir: Path = Path(".index")
    chunk_size: int = 1000       # characters per chunk
    chunk_overlap: int = 150     # characters shared between neighbouring chunks
    top_k: int = 5               # passages sent to Claude per question

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
