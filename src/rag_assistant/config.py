"""
config.py — centralised settings, loaded from environment variables or .env.

Nothing here is required at import time, so tests and `make ingest` work
without an API key.  The key is checked only when `ask` actually calls Claude.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings."""

    # ── Anthropic (answer generation) ─────────────────────────────────────────
    anthropic_api_key: str = ""
    answer_model: str = "claude-haiku-5-5"
    max_answer_tokens: int = 1024

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
