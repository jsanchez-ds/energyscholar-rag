"""Typed config loader — reads .env + configs/config.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    """Environment variables (secrets + per-environment tweaks)."""

    llm_provider: Literal["groq", "anthropic", "openai", "openrouter"] = "groq"
    llm_model_override: str = ""

    groq_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "energy_papers"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    papers_dir: str = "./data/papers"
    cache_dir: str = "./data/cache"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def active_api_key(self) -> str:
        """Return the API key matching `llm_provider`. Raises if missing."""
        key_map = {
            "groq": self.groq_api_key,
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "openrouter": self.openrouter_api_key,
        }
        key = key_map[self.llm_provider]
        if not key:
            raise RuntimeError(
                f"Provider '{self.llm_provider}' selected but no API key set. "
                f"Set {self.llm_provider.upper()}_API_KEY in .env."
            )
        return key


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_yaml_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Load the main YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_env() -> EnvSettings:
    return EnvSettings()  # type: ignore[call-arg]
