"""
Centralized application configuration.

All runtime settings are loaded from environment variables 
"""
from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Core ---
    app_name: str = "AI Vision Inspection Engine"
    environment: str = Field(default="development")  # development | staging | production
    log_level: str = Field(default="INFO")

    # --- Gemini / Vision provider ---
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    candidate_models: List[str] = Field(
        default=[
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
        ]
    )
    max_retries_per_model: int = Field(default=3, ge=1, le=5)
    request_timeout_seconds: int = Field(default=30, ge=5, le=120)

    # --- Security ---
    api_key: str = Field(..., description="Shared secret clients must send in X-API-Key to call /inspect")
    allowed_origins: List[str] = Field(default=["http://localhost:3000"])

    # --- Rate limiting ---
    rate_limit_per_minute: int = Field(default=20, ge=1)

    # --- Upload constraints ---
    max_image_size_mb: float = Field(default=5.0, gt=0)
    allowed_image_types: List[str] = Field(default=["image/png", "image/jpeg", "image/webp"])

    # --- Local detector (fast first-pass, avoids a Gemini call on every frame) ---
    use_local_detector: bool = Field(default=True)
    local_model_path: str = Field(default="models/yolov8n.onnx")
    local_escalation_confidence: float = Field(
        default=0.50, ge=0.0, le=1.0,
        description="If the local detector's best detection is below this confidence "
        "(or it finds nothing), escalate the frame to Gemini instead of trusting it.",
    )

    @field_validator("candidate_models", "allowed_origins", "allowed_image_types", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache
def get_settings() -> "Settings":
    """Cached settings instance — env is read once per process."""
    return Settings()
