"""AI Vision Inspection Engine

Accepts a base64-encoded image frame, runs it through a multi-model
Gemini vision pipeline with retry/failover, and returns normalized
bounding boxes with risk classifications.
"""
import logging
import time

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .config import get_settings
from .local_detector import bytes_to_bgr_image, get_local_detector
from .logging_config import configure_logging
from .models import ErrorResponse, HealthResponse, ImagePayload, InspectionResponse
from .security import require_api_key
from .vision_engine import ImageValidationError, decode_and_validate_image, run_inspection

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
gemini_client = genai.Client(api_key=settings.gemini_api_key)
local_detector = get_local_detector(settings.local_model_path) if settings.use_local_detector else None

app = FastAPI(
    title=settings.app_name,
    description="Multimodal safety hazard detection and spatial analysis powered by Gemini Flash",
    version="1.0.0",
    docs_url="/docs" if not settings.is_production else None,  # hide interactive docs in prod
    redoc_url=None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = int((time.monotonic() - start) * 1000)
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled_exception path=%s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(detail="Internal server error.").model_dump(),
    )


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_check():
    """Liveness/readiness probe — no auth required, used by the hosting platform."""
    return HealthResponse(status="online", environment=settings.environment, engine="Gemini Multimodal API")


@app.post(
    "/api/v1/inspect",
    response_model=InspectionResponse,
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    dependencies=[Depends(require_api_key)],
    tags=["inspection"],
)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
def inspect_frame(request: Request, payload: ImagePayload):
    try:
        image_bytes, mime_type = decode_and_validate_image(payload.image_base64, settings)
    except ImageValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    # Fast path: try the local YOLOv8n model first. It's free, runs in
    # ~100ms on CPU, and handles the common case (clear objects, decent
    # confidence) without ever calling Gemini. We only escalate when the
    # caller explicitly wants deeper semantic reasoning, or when the local
    # model isn't confident enough to trust on its own.
    if local_detector is not None and not payload.deep_analysis:
        cv_image = bytes_to_bgr_image(image_bytes)
        if cv_image is not None:
            local_detections, local_latency_ms = local_detector.detect(cv_image)
            best_confidence = max((d.confidence or 0.0) for d in local_detections) if local_detections else 0.0

            if best_confidence >= settings.local_escalation_confidence:
                return InspectionResponse(
                    status="success",
                    engine="local",
                    model_used="yolov8n (on-device, ONNX Runtime)",
                    latency_ms=local_latency_ms,
                    detections=local_detections,
                )
            logger.info(
                "escalating_to_gemini reason=low_local_confidence best_confidence=%.2f threshold=%.2f",
                best_confidence, settings.local_escalation_confidence,
            )

    # Escalation path: either the local model wasn't confident, deep_analysis
    # was requested, or the local detector isn't available in this deployment.
    model_used, detections, latency_ms = run_inspection(gemini_client, image_bytes, mime_type, settings)

    return InspectionResponse(
        status="success",
        engine="gemini",
        model_used=model_used,
        latency_ms=latency_ms,
        detections=detections,
    )
