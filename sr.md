

**Real-Time Hybrid AI Vision Inspection System**
*FastAPI · ONNX Runtime · Google Gemini · Docker · Vercel · Render*

- Designed and shipped a hybrid inference pipeline that routes frames between
  an on-device YOLOv8n model (ONNX Runtime, ~110ms p50 on CPU, measured via
  a custom benchmark script) and a Gemini vision API escalation path,
  cutting API calls and latency on the common case while preserving
  semantic reasoning for ambiguous frames.
- Built an evaluation harness with IoU-based precision/recall/F1 scoring
  against hand-labeled ground truth, wired as a CI regression gate
  (`--min-f1` threshold) rather than an unverified accuracy claim.
- Implemented the full YOLO inference path from raw bytes to results —
  letterbox preprocessing, tensor layout conversion, greedy NMS, and
  coordinate remapping back to source resolution — without relying on a
  higher-level detection framework.
- Hardened the service for deployment: environment-driven configuration,
  API-key auth, per-IP rate limiting, strict image validation ahead of
  every inference call, and a typed response contract validated with
  Pydantic so malformed output fails server-side, not in the client.
- Covered both routing paths and the auth/validation layer with a pytest
  suite (8 tests, Gemini client mocked) that runs without any API key,
  and containerized the backend with a non-root Docker image whose build
  step fetches model weights rather than vendoring them in git.

## Shorter single-line version 
Hybrid real-time vision inspection service (FastAPI + local YOLOv8n/ONNX +
Gemini escalation) with a measured eval harness, latency benchmarks, and
production hardening (auth, rate limiting, Docker, CI-gated tests).

## Numbers to have ready if asked in an interview
- Local detector: p50 111ms / p95 122ms per frame on CPU (measured, `scripts/benchmark.py`)
- Eval set: 2 images / 7 hand-labeled objects, precision 1.00 / recall 1.00 / F1 1.00 at IoU≥0.5
  — be upfront that this is a correctness smoke test, not a statistically powered
  accuracy claim, and say what a larger eval set would take (more labeled frames,
  ideally from the deployment's actual camera angle/lighting).
- Escalation threshold: local detections below 0.50 confidence (or none found) fall
  through to Gemini automatically; a `deep_analysis` flag forces escalation for
  frames needing contextual risk judgment a COCO-trained detector can't make.
