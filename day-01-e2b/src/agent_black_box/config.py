from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic.functional_validators import BeforeValidator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing import Annotated


def _parse_allowed_origins(value: object) -> list[str]:
    if value is None:
        return [
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        ]
    if isinstance(value, str):
        return [origin.strip() for origin in value.split(",") if origin.strip()]
    if isinstance(value, (list, tuple)):
        return [str(origin).strip() for origin in value if str(origin).strip()]
    raise TypeError(f"Unsupported frontend origins value: {value!r}")


def _parse_csv_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError(f"Unsupported comma-separated list value: {value!r}")


class Settings(BaseSettings):
    e2b_api_key: str = Field(alias="E2B_API_KEY")
    ollama_api_key: str = Field(alias="OLLAMA_API_KEY")
    ollama_base_url: str = Field(
        default="https://ollama.com/api",
        alias="OLLAMA_BASE_URL",
    )
    ollama_model: str = Field(default="minimax-m2.7:cloud", alias="OLLAMA_MODEL")
    ollama_fallback_model: str = Field(
        default="glm-5.1:cloud",
        alias="OLLAMA_FALLBACK_MODEL",
    )
    ollama_arena_models: Annotated[
        list[str],
        NoDecode,
        BeforeValidator(_parse_csv_list),
    ] = Field(
        default_factory=lambda: [
            "kimi-k2.6:cloud",
            "glm-5.1:cloud",
            "gemma4:31b",
            "qwen3.5:397b",
        ],
        alias="OLLAMA_ARENA_MODELS",
    )
    ollama_max_attempts_per_model: int = Field(
        default=2,
        alias="OLLAMA_MAX_ATTEMPTS_PER_MODEL",
    )
    ollama_retry_base_delay_seconds: float = Field(
        default=0.5,
        alias="OLLAMA_RETRY_BASE_DELAY_SECONDS",
    )
    ollama_tags_timeout_seconds: float = Field(
        default=5.0,
        alias="OLLAMA_TAGS_TIMEOUT_SECONDS",
    )
    ollama_chat_timeout_seconds: float = Field(
        default=300.0,
        alias="OLLAMA_CHAT_TIMEOUT_SECONDS",
    )
    ollama_tags_cache_seconds: float = Field(
        default=600.0,
        alias="OLLAMA_TAGS_CACHE_SECONDS",
    )
    ollama_keep_alive: str | None = Field(default="10m", alias="OLLAMA_KEEP_ALIVE")
    ollama_think: str | None = Field(default="false", alias="OLLAMA_THINK")
    ollama_num_predict: int | None = Field(
        default=None,
        alias="OLLAMA_NUM_PREDICT",
    )
    ollama_num_ctx: int | None = Field(default=None, alias="OLLAMA_NUM_CTX")
    ollama_temperature: float | None = Field(default=0.0, alias="OLLAMA_TEMPERATURE")
    model_protocol_repair_attempts: int = Field(
        default=2,
        alias="MODEL_PROTOCOL_REPAIR_ATTEMPTS",
    )
    ollama_timeout_demotion_threshold: int = Field(
        default=2,
        alias="OLLAMA_TIMEOUT_DEMOTION_THRESHOLD",
    )
    ollama_timeout_demotion_window_seconds: float = Field(
        default=600.0,
        alias="OLLAMA_TIMEOUT_DEMOTION_WINDOW_SECONDS",
    )
    ollama_timeout_demotion_seconds: float = Field(
        default=600.0,
        alias="OLLAMA_TIMEOUT_DEMOTION_SECONDS",
    )
    run_root: Path = Field(
        default=Path(
            "/Users/georgepickett/devrel-jobs/30-days-of-agentic-apis/day-01-e2b/runs"
        ),
        alias="RUN_ROOT",
    )
    arena_root: Path = Field(
        default=Path(
            "/Users/georgepickett/devrel-jobs/30-days-of-agentic-apis/day-01-e2b/arenas"
        ),
        alias="ARENA_ROOT",
    )
    e2b_template: str = Field(default="agent-black-box", alias="E2B_TEMPLATE")
    sandbox_timeout_seconds: int = Field(default=3600, alias="SANDBOX_TIMEOUT_SECONDS")
    max_turns: int = Field(default=20, alias="MAX_TURNS")
    preview_port: int = Field(default=4173, alias="PREVIEW_PORT")
    fixture_root: Path = Field(
        default=Path(
            "/Users/georgepickett/devrel-jobs/30-days-of-agentic-apis/day-01-e2b/fixtures"
        ),
        alias="FIXTURE_ROOT",
    )
    frontend_allowed_origins: Annotated[
        list[str],
        BeforeValidator(_parse_allowed_origins),
    ] = Field(
        default_factory=lambda: [
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        ],
        alias="FRONTEND_ALLOWED_ORIGINS",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
