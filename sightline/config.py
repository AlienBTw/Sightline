"""Environment and YAML config loading. Fail-loud on missing required settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from sightline.errors import SightlineError

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sightline_mode: Literal["live", "replay"] = Field(alias="SIGHTLINE_MODE")
    api_url: str = Field(default="http://127.0.0.1:8000", alias="API_URL")
    detector_backend: Literal["llmdet", "owlv2", "rtdetr", "dfine", "replay"] = Field(
        alias="DETECTOR_BACKEND"
    )
    report_backend: Literal["ollama", "meta_api"] = Field(alias="REPORT_BACKEND")
    # Sparse VLM evidence gate: any OpenAI-compatible endpoint (base URL includes /v1).
    # Local default: Ollama Muse Glimmer at http://localhost:11434/v1. Empty = gate off.
    vlm_base_url: str = Field(default="", alias="VLM_BASE_URL")
    vlm_model: str = Field(default="", alias="VLM_MODEL")
    vlm_api_key: str = Field(default="", alias="VLM_API_KEY")
    # Report backend "ollama": local Muse Glimmer through Ollama's OpenAI-compatible API.
    ollama_base_url: str = Field(default="http://localhost:11434/v1", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="muse-glimmer:30b", alias="OLLAMA_MODEL")
    # Report backend "meta_api": Meta Model API (Muse Spark). Set from the Meta Model API docs.
    meta_api_base_url: str = Field(default="", alias="META_API_BASE_URL")
    meta_api_key: str = Field(default="", alias="META_API_KEY")
    meta_api_model: str = Field(default="", alias="META_API_MODEL")
    torch_device: str = Field(default="cuda", alias="TORCH_DEVICE")
    sightline_db: str = Field(default="./data/sightline.db", alias="SIGHTLINE_DB")
    sample_fps: float = Field(default=2.0, alias="SAMPLE_FPS")

    @field_validator("sightline_mode", "detector_backend", "report_backend", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().lower()
        return v


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()  # type: ignore[call-arg]
    except Exception as e:
        raise SightlineError(
            error_code="config_invalid",
            component="config",
            message=f"Failed to load settings from environment: {e}",
            hint="Copy .env.example to .env and set SIGHTLINE_MODE, DETECTOR_BACKEND, REPORT_BACKEND",
        ) from e


def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise SightlineError(
            error_code="config_file_missing",
            component="config",
            message=f"Missing config file: {path}",
            hint=f"Create {name} under config/",
        )
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SightlineError(
            error_code="config_file_invalid",
            component="config",
            message=f"{name} must be a YAML mapping",
            hint="Check YAML syntax",
        )
    return data
