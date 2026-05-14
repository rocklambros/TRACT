from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from tract.config import PHASE1D_DEPLOYMENT_MODEL_DIR


class ApiSettings(BaseSettings):
    """
    Runtime config for the REST API
    All fields are overridable using env vars (prefixed - TRACT_API_)
    """
    model_config = SettingsConfigDict(
        env_prefix="TRACT_API_",
        env_file=None,
    )
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    model_dir: Path = PHASE1D_DEPLOYMENT_MODEL_DIR
    model_version: str = "tract-cre-assignment-v1.0"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_batch_size: int = 256
    max_text_length: int = 8192
    max_top_k: int = 50

    # avoiding privileged ports until given root access
    @field_validator("port")
    @classmethod
    def _validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"Invalid port: {v} (must be 1..65535)")
        if v < 1024:
            raise ValueError(f"Privileged port: {v} (below 1024 requires root)")
        return v

    # converting CORS string automatically
    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    """Cached settings factory. Returns the same instance for the process.
    Tests override by clearing the cache: get_settings.cache_clear().
    """
    return ApiSettings()
