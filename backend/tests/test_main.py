"""Basic API tests — mock the Gemini client so tests run without network
access or a real API key.
"""
import base64
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY") or "test-key"
os.environ["API_KEY"] = os.environ.get("API_KEY") or "test-shared-secret"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)
VALID_API_KEY = get_settings().api_key  # read the key actually loaded, whatever its source

# 1x1 transparent PNG, base64-encoded
TINY_PNG_B64 = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_health_check_no_auth_required():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "online"


def test_inspect_rejects_missing_api_key():
    resp = client.post("/api/v1/inspect", json={"image_base64": TINY_PNG_B64})
    assert resp.status_code == 401


def test_inspect_rejects_bad_api_key():
    resp = client.post(
        "/api/v1/inspect",
        json={"image_base64": TINY_PNG_B64},
        headers={"X-API-Key": "wrong"},
    )
    assert resp.status_code == 401


def test_inspect_rejects_malformed_base64():
    resp = client.post(
        "/api/v1/inspect",
        json={"image_base64": "data:image/png;base64,not-valid-base64!!!"},
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert resp.status_code == 422


@patch("app.main.gemini_client")
def test_inspect_success_returns_detections(mock_client):
    mock_response = MagicMock()
    mock_response.text = (
        '{"detections": [{"label": "forklift", "risk": "Medium", '
        '"box_2d": [100, 100, 400, 400], "confidence": 0.87}]}'
    )
    mock_client.models.generate_content.return_value = mock_response

    resp = client.post(
        "/api/v1/inspect",
        json={"image_base64": TINY_PNG_B64},
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["engine"] == "gemini"
    assert len(body["detections"]) == 1
    assert body["detections"][0]["label"] == "forklift"


def test_inspect_uses_local_detector_when_confident(monkeypatch):
    """When the local detector is confident, Gemini should never be called."""
    import app.main as main_module
    from app.models import DetectionItem

    fake_detections = [DetectionItem(label="person", risk="Medium", box_2d=[10, 10, 500, 500], confidence=0.91)]

    class FakeDetector:
        def detect(self, image, confidence_threshold=0.40):
            return fake_detections, 42

    monkeypatch.setattr(main_module, "local_detector", FakeDetector())
    with patch.object(main_module, "run_inspection") as mock_run_inspection:
        resp = client.post(
            "/api/v1/inspect",
            json={"image_base64": TINY_PNG_B64},
            headers={"X-API-Key": VALID_API_KEY},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["engine"] == "local"
        assert body["model_used"].startswith("yolov8n")
        mock_run_inspection.assert_not_called()  # Gemini must not be touched


def test_inspect_escalates_to_gemini_when_local_unconfident(monkeypatch):
    """Low-confidence (or empty) local results should fall through to Gemini."""
    import app.main as main_module

    class FakeDetector:
        def detect(self, image, confidence_threshold=0.40):
            return [], 30  # nothing found locally

    monkeypatch.setattr(main_module, "local_detector", FakeDetector())

    mock_response = MagicMock()
    mock_response.text = '{"detections": []}'
    with patch.object(main_module.gemini_client.models, "generate_content", return_value=mock_response):
        resp = client.post(
            "/api/v1/inspect",
            json={"image_base64": TINY_PNG_B64},
            headers={"X-API-Key": VALID_API_KEY},
        )
        assert resp.status_code == 200
        assert resp.json()["engine"] == "gemini"


def test_inspect_deep_analysis_skips_local_detector(monkeypatch):
    """deep_analysis=True should go straight to Gemini even if local would be confident."""
    import app.main as main_module
    from app.models import DetectionItem

    class FakeDetector:
        def detect(self, image, confidence_threshold=0.40):
            # Would normally short-circuit to 'local' — deep_analysis must override this.
            return [DetectionItem(label="person", risk="Low", box_2d=[0, 0, 100, 100], confidence=0.99)], 5

    monkeypatch.setattr(main_module, "local_detector", FakeDetector())

    mock_response = MagicMock()
    mock_response.text = '{"detections": []}'
    with patch.object(main_module.gemini_client.models, "generate_content", return_value=mock_response):
        resp = client.post(
            "/api/v1/inspect",
            json={"image_base64": TINY_PNG_B64, "deep_analysis": True},
            headers={"X-API-Key": VALID_API_KEY},
        )
        assert resp.status_code == 200
        assert resp.json()["engine"] == "gemini"
