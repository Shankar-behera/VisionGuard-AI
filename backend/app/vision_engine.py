"""Core inference logic: image validation + multi-model failover against
Google Gemini's vision API, with exponential backoff per model and
automatic promotion to the next model on persistent failure.
"""
import base64
import binascii
import json
import logging
import time
from typing import Any, Dict, List, Tuple

from fastapi import HTTPException, status
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from .config import Settings
from .models import DetectionItem

logger = logging.getLogger(__name__)

INSPECTION_PROMPT = (
    "Analyze this image carefully for a safety and quality inspection.\n"
    "Identify key objects, people, or hazards visible.\n"
    "Return a strictly valid JSON object with a single key 'detections', "
    "an array of objects each containing:\n"
    "- 'label': concise object or hazard name (string)\n"
    "- 'risk': one of 'Low', 'Medium', 'High'\n"
    "- 'box_2d': bounding box as [ymin, xmin, ymax, xmax], integers 0-1000\n"
    "- 'confidence': a float between 0 and 1\n"
    "If nothing notable is visible, return {\"detections\": []}. "
    "Return ONLY the JSON object, no markdown fences, no commentary."
)


class ImageValidationError(Exception):
    pass


def decode_and_validate_image(raw_b64: str, settings: Settings) -> Tuple[bytes, str]:
    """Strip data-URL prefix, base64-decode, and enforce size/type limits.

    Returns (raw_bytes, mime_type).
    """
    mime_type = "image/png"
    payload = raw_b64
    if payload.startswith("data:"):
        try:
            header, payload = payload.split(",", 1)
            mime_type = header.split(";")[0].replace("data:", "") or mime_type
        except ValueError:
            raise ImageValidationError("Malformed data URL.")

    if mime_type not in settings.allowed_image_types:
        raise ImageValidationError(
            f"Unsupported image type '{mime_type}'. Allowed: {settings.allowed_image_types}"
        )

    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise ImageValidationError("Invalid base64 payload — could not decode image.")

    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > settings.max_image_size_mb:
        raise ImageValidationError(
            f"Image too large ({size_mb:.2f} MB). Max allowed is {settings.max_image_size_mb} MB."
        )
    if size_mb == 0:
        raise ImageValidationError("Decoded image is empty.")

    return image_bytes, mime_type


def _parse_model_output(text: str) -> List[Dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned non-JSON output: {e}")

    detections = parsed.get("detections", [])
    if not isinstance(detections, list):
        raise ValueError("'detections' field was not a list.")
    return detections


def run_inspection(
    client: genai.Client,
    image_bytes: bytes,
    mime_type: str,
    settings: Settings,
) -> Tuple[str, List[DetectionItem], int]:
    """Try each candidate model in order, with exponential backoff on
    transient server errors, failing over to the next model on
    persistent failure. Returns (model_used, detections, latency_ms).
    """
    start = time.monotonic()
    last_error: str = "unknown error"

    for model_name in settings.candidate_models:
        for attempt in range(settings.max_retries_per_model):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        INSPECTION_PROMPT,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                raw_detections = _parse_model_output(response.text or "{}")
                detections = [DetectionItem(**d) for d in raw_detections]
                latency_ms = int((time.monotonic() - start) * 1000)
                logger.info(
                    "inspection_success model=%s attempt=%s latency_ms=%s detections=%s",
                    model_name, attempt + 1, latency_ms, len(detections),
                )
                return model_name, detections, latency_ms

            except ServerError as e:
                # Transient (5xx) — retry same model with exponential backoff
                last_error = str(e)
                wait = 2 ** attempt
                logger.warning(
                    "transient_error model=%s attempt=%s wait_s=%s error=%s",
                    model_name, attempt + 1, wait, last_error,
                )
                if attempt < settings.max_retries_per_model - 1:
                    time.sleep(wait)

            except ClientError as e:
                # 4xx (bad request, quota, auth) — no point retrying this model
                last_error = str(e)
                logger.error("client_error model=%s error=%s", model_name, last_error)
                break

            except (ValueError, TypeError) as e:
                # Model responded but payload didn't match our schema
                last_error = str(e)
                logger.error("schema_error model=%s error=%s", model_name, last_error)
                break

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"All vision models unavailable or returned invalid output. Last error: {last_error}",
    )
