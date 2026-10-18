"""Runtime configuration, driven by environment variables.

Defaults are chosen so the whole system runs offline with zero setup:
SQLite for storage and a deterministic offline LLM fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # SQLite by default; set DATABASE_URL to a PostgreSQL DSN
    # (e.g. postgresql+psycopg://user:pass@localhost:5432/merchant_intel)
    # to use the docker-compose database instead.
    database_url: str = os.environ.get(
        "DATABASE_URL", "sqlite:///./merchant_intel.db"
    )
    # "offline" (default, deterministic, no API key) or "openai".
    llm_provider: str = os.environ.get("MERCHANT_LLM_PROVIDER", "offline")
    openai_api_key: str | None = os.environ.get("OPENAI_API_KEY")
    openai_model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    # Seed for the synthetic dataset generator.
    data_seed: int = int(os.environ.get("MERCHANT_DATA_SEED", "42"))
    seed_merchant_count: int = int(os.environ.get("MERCHANT_SEED_COUNT", "24"))


settings = Settings()
