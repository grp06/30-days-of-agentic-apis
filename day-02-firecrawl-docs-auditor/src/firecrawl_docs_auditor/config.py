from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field
from pydantic.functional_validators import BeforeValidator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_frontend_origins(value: object) -> list[str]:
    if value is None:
        return [
            "http://127.0.0.1:3122",
            "http://localhost:3122",
        ]
    if isinstance(value, str):
        return [origin.strip() for origin in value.split(",") if origin.strip()]
    if isinstance(value, (list, tuple)):
        return [str(origin).strip() for origin in value if str(origin).strip()]
    raise TypeError(f"Unsupported frontend origins value: {value!r}")


class Settings(BaseSettings):
    host: str = Field(default="127.0.0.1", alias="FIRECRAWL_DOCS_AUDITOR_HOST")
    port: int = Field(default=8122, alias="FIRECRAWL_DOCS_AUDITOR_PORT")
    frontend_origins: Annotated[
        list[str],
        BeforeValidator(_parse_frontend_origins),
    ] = Field(
        default_factory=lambda: [
            "http://127.0.0.1:3122",
            "http://localhost:3122",
        ],
        alias="FIRECRAWL_DOCS_AUDITOR_FRONTEND_ORIGINS",
    )
    codex_bin: str | None = Field(default=None, alias="CODEX_BIN")
    codex_app_server_timeout_seconds: float = Field(
        default=30.0,
        alias="CODEX_APP_SERVER_TIMEOUT_SECONDS",
    )
    firecrawl_api_key: str | None = Field(default=None, alias="FIRECRAWL_API_KEY")
    firecrawl_api_base_url: str = Field(
        default="https://api.firecrawl.dev/v2",
        alias="FIRECRAWL_API_BASE_URL",
    )
    firecrawl_timeout_seconds: float = Field(
        default=10.0,
        alias="FIRECRAWL_TIMEOUT_SECONDS",
    )
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2],
        alias="FIRECRAWL_DOCS_AUDITOR_PROJECT_ROOT",
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
