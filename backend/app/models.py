"""Request / response schemas shared across the API."""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ImagePayload(BaseModel):
    image_base64: str = Field(..., description="Data URL or raw base64-encoded image string")
    deep_analysis: bool = Field(
        default=False,
        description="Force escalation to the Gemini semantic-reasoning path even if the "
        "fast local detector is confident. Use for frames needing judgment calls "
        "beyond 'what objects are here' (e.g. subtle hazards, context-dependent risk).",
    )

    @field_validator("image_base64")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or len(v) < 20:
            raise ValueError("image_base64 is missing or too short to be a valid image")
        return v


class DetectionItem(BaseModel):
    label: str
    risk: str = Field(..., pattern="^(Low|Medium|High)$")
    box_2d: List[int] = Field(..., min_length=4, max_length=4)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class InspectionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    engine: str = Field(..., description="'local' (on-device YOLOv8n) or 'gemini' (API escalation)")
    model_used: str
    latency_ms: int
    detections: List[DetectionItem]


class HealthResponse(BaseModel):
    status: str
    environment: str
    engine: str


class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str
