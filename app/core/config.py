"""
Application configuration from environment variables.
PHI-safe: no sensitive data in defaults or logs.
"""
import json
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Firebase configuration
    firebase_project_id: str = Field(
        ...,
        description="Firebase project ID for token verification"
    )
    firebase_credentials_json: Optional[str] = Field(
        default=None,
        description="Firebase service account JSON string (optional, uses ADC if not set)"
    )
    google_application_credentials: Optional[str] = Field(
        default=None,
        description="Path to service account JSON file (GOOGLE_APPLICATION_CREDENTIALS)"
    )

    # Service configuration
    service_env: Literal["dev", "staging", "prod"] = Field(
        default="dev",
        description="Service environment"
    )

    # Authentication mode
    auth_mode: Literal["firebase", "dev"] = Field(
        default="firebase",
        description="Auth mode: 'firebase' for production, 'dev' for local testing without Firebase"
    )
    dev_bearer_token: str = Field(
        default="dev-token",
        description="Bearer token accepted in dev auth mode (only used when AUTH_MODE=dev)"
    )

    # Extractor backend configuration
    extractor_backend: Literal["mock", "vllm", "openai_compat"] = Field(
        default="mock",
        description="Extraction backend: 'mock' for testing, 'vllm' or 'openai_compat' for inference"
    )

    # vLLM configuration
    vllm_base_url: str = Field(
        default="http://127.0.0.1:8000",
        description="Base URL for vLLM OpenAI-compatible API"
    )
    vllm_model: str = Field(
        default="",
        description="Model name/path for vLLM inference"
    )
    vllm_timeout_ms: int = Field(
        default=4500,
        ge=1000,
        le=30000,
        description="Timeout for vLLM requests in milliseconds"
    )

    # OpenAI-compatible backend configuration (LM Studio, Ollama, etc.)
    openai_compat_base_url: str = Field(
        default="http://127.0.0.1:1234/v1",
        description="Base URL for OpenAI-compatible API (e.g., LM Studio, Ollama)"
    )
    openai_compat_model: str = Field(
        default="",
        description="Model name for OpenAI-compatible inference"
    )
    openai_compat_timeout_ms: int = Field(
        default=30000,
        ge=1000,
        le=120000,
        description="Timeout for OpenAI-compatible requests in milliseconds"
    )

    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8080, description="Server port")

    # Drift Guard configuration
    drift_guard_mode: Literal["off", "warn", "safe"] = Field(
        default="warn",
        description=(
            "Drift guard mode: "
            "'off' = ignore contract drift, "
            "'warn' = emit telemetry event on drift, "
            "'safe' = emit event AND force fallback_baseline on drift"
        )
    )
    drift_guard_cooldown_s: int = Field(
        default=3600,
        ge=0,
        le=86400,
        description="Cooldown in seconds between drift telemetry events (0 = no cooldown)"
    )

    # Intelligent Chunking configuration
    chunking_enabled: bool = Field(
        default=False,
        description=(
            "Enable intelligent chunking with token limits. "
            "False = use basic duration-only chunking (legacy). "
            "True = use token + duration limits (recommended for long transcripts)."
        )
    )
    chunking_hard_token_limit: int = Field(
        default=2048,
        ge=256,
        le=32768,
        description="Maximum estimated tokens per chunk (hard limit)"
    )
    chunking_soft_duration_limit_ms: Optional[int] = Field(
        default=180000,
        ge=0,
        le=1800000,
        description="Soft duration target in ms (triggers split when exceeded). None to disable."
    )
    chunking_max_duration_ms: int = Field(
        default=300000,
        ge=60000,
        le=1800000,
        description="Maximum duration per chunk in ms (hard limit)"
    )
    chunking_min_segments_per_chunk: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Minimum segments per chunk before allowing split"
    )

    # Epic 15: Map Extractor Mode configuration
    map_extractor_mode: Literal["lite", "full"] = Field(
        default="lite",
        description=(
            "Extractor mode for MAP stage: "
            "'lite' = cheap baseline/short prompt (default), "
            "'full' = full MedGemma extraction per chunk"
        )
    )
    include_evidence_in_response: bool = Field(
        default=False,
        description=(
            "Include chunk evidence in API response metadata. "
            "False = backward compatible (no chunkEvidence field). "
            "True = include chunkEvidence array in metadata."
        )
    )
    lite_extractor_max_tokens: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Max tokens for lite extractor response"
    )
    evidence_max_snippets_per_chunk: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum evidence snippets to extract per chunk"
    )

    @field_validator("firebase_credentials_json", mode="before")
    @classmethod
    def validate_credentials_json(cls, v: Optional[str]) -> Optional[str]:
        """Validate that credentials JSON is valid if provided."""
        if v is None or v == "":
            return None
        try:
            json.loads(v)
            return v
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in FIREBASE_CREDENTIALS_JSON: {e}")

    @field_validator("auth_mode", mode="after")
    @classmethod
    def validate_auth_mode_not_dev_in_prod(cls, v: str, info) -> str:
        """Prevent dev auth mode in production environment."""
        # Access other fields via info.data
        service_env = info.data.get("service_env", "dev")
        if v == "dev" and service_env == "prod":
            raise ValueError(
                "SECURITY ERROR: AUTH_MODE=dev is forbidden when SERVICE_ENV=prod. "
                "This would bypass Firebase authentication in production."
            )
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
