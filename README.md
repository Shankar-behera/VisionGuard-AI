# VisionGuard AI — Real-Time Edge & Cloud Vision Inspection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU%20Optimized-005CED.svg)](https://onnxruntime.ai/)
[![Gemini API](https://img.shields.io/badge/Google%20Gemini-Multimodal%20VLM-8E75B2.svg)](https://ai.google.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4%2B-38B2AC.svg)](https://tailwindcss.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

VisionGuard AI is a  hybrid-inference computer vision platform engineered for real-time safety and defect inspection. It pairs lightweight on-device detection (YOLOv8n via ONNX Runtime) with multimodal Vision-Language Model (VLM) escalation via Google Gemini.

By executing a low-latency local pass first, VisionGuard processes clean frames in ~105ms at zero cloud inference cost, reserving multimodal cloud calls strictly for ambiguous, low-confidence frames or deep contextual scene reasoning.

---

## Architecture: Hybrid Tiered Inference

Pure cloud VLM inspection introduces latency overhead, bandwidth consumption, and prohibitive token costs per frame. Pure edge object detection fails at semantic context (e.g., distinguishing between a ladder standing idle vs. a worker using a ladder unsafely). VisionGuard resolves this trade-off using a tiered routing architecture.

```mermaid
flowchart TD
    %% Define Styles
    classDef client fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc,rx:8px,ry:8px
    classDef gateway fill:#1e293b,stroke:#6366f1,stroke-width:2px,color:#f8fafc,rx:8px,ry:8px
    classDef logic fill:#334155,stroke:#94a3b8,stroke-width:1px,color:#f8fafc
    classDef edge fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#ecfdf5,rx:8px,ry:8px
    classDef cloud fill:#4c1d95,stroke:#8b5cf6,stroke-width:2px,color:#f5f3ff,rx:8px,ry:8px
    classDef output fill:#0f172a,stroke:#e2e8f0,stroke-width:2px,color:#f8fafc,rx:4px,ry:4px

    %% Nodes
    UI[Browser UI <br/> Live Frame Capture] ::: client
    API[FastAPI Gateway <br/> Rate Limits, Auth, Validation] ::: gateway
    DeepCheck{deep_analysis <br/> == true?} ::: logic
    Edge[Tier 1: Local Edge Pass <br/> YOLOv8n ONNX <br/> ~105ms CPU Latency] ::: edge
    ConfCheck{Confidence <br/> >= 0.50?} ::: logic
    Cloud[Tier 2: Cloud VLM <br/> Google Gemini <br/> Multimodal Escalation] ::: cloud
    ReturnEdge(Immediate Return <br/> engine='local') ::: edge
    JSON[/Standardized JSON Detections <br/> Bounding Boxes & Risk/] ::: output

    %% Connections
    UI --> API
    API --> DeepCheck
    DeepCheck -- "Yes" --> Cloud
    DeepCheck -- "No" --> Edge
    Edge --> ConfCheck
    ConfCheck -- "Yes" --> ReturnEdge
    ConfCheck -- "No" --> Cloud
    ReturnEdge --> JSON
    Cloud --> JSON
```

### Routing Logic

- **Tier 1 (Local Edge Pass):** Processes the frame locally using an ONNX-quantized YOLOv8n model on CPU. If target hazards are identified with a confidence threshold $\ge 0.50$, the response returns immediately (`engine="local"`).
- **Tier 2 (Cloud Escalation):** If confidence falls below the threshold or no target is resolved, the frame is escalated to Google Gemini (`engine="gemini"`) for multimodal inspection.
- **Contextual Override (`deep_analysis: true`):** Callers can bypass Tier 1 to demand semantic risk evaluation (e.g., ergonomics, safety compliance, spatial hazards).

---

## Key Features

- **Sub-110ms Edge Detection:** Native ONNX Runtime implementation featuring custom letterboxing, Non-Maximum Suppression (NMS), and normalized coordinate remapping.
- **Multimodal Cloud Failover:** Automated exponential backoff and model failover routines targeting transient 5xx faults and model deprecations.
- **Production-Hardened Security:**
  - Configured via strict Pydantic Settings (`pydantic-settings`).
  - Request gating via shared `X-API-Key` headers.
  - Per-IP rate limiting powered by `slowapi`.
  - Explicit CORS whitelisting (rejects wildcard origins).
  - Strict input validation against image payload size, aspect ratio, and MIME types.
- **Auditing & Telemetry:** Structured JSON logging for downstream ingestion (DataDog, Elastic, CloudWatch) without client-side stack trace leakage.
- **Evaluation & Benchmarking Harness:** Integrated intersection-over-union (IoU) evaluation engine and automated p50/p95/p99 latency measurement scripts.

---

## Performance & Verification

The local detector was benchmarked on commodity x86 hardware (single-threaded CPU ONNX Runtime):

### Latency Distribution (`scripts/benchmark.py`)

Target: `eval/images/bus.jpg` | Sample Size: 200 iterations (+10 warmup discarded)

| Metric           | Value              |
| ---------------- | ------------------ |
| Mean Latency     | 110.7 ms           |
| Median (p50)     | 104.8 ms           |
| 95th Percentile  | 143.9 ms           |
| 99th Percentile  | 217.1 ms           |
| Min / Max        | 91.5 ms / 355.5 ms |
| Throughput       | ~9.5 FPS (CPU bound) |

### Detection Accuracy (`eval/run_eval.py`)

Dataset: `eval/images/` (IoU Threshold: 0.50)

| Metric          | Value |
| --------------- | ----- |
| Precision       | 1.000 |
| Recall          | 1.000 |
| F1-Score        | 1.000 |
| True Positives  | 7     |
| False Positives | 0     |
| False Negatives | 0     |

> \* Verifies end-to-end correctness of preprocessing, NMS, and coordinate projection.

---

## Repository Structure

```text
ai-vision-inspection/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application, routing, and lifecycle hooks
│   │   ├── config.py            # Pydantic environment configuration
│   │   ├── models.py            # Request/Response validation schemas
│   │   ├── local_detector.py    # YOLOv8n ONNX runtime pipeline (NMS, letterboxing)
│   │   ├── vision_engine.py     # Gemini client orchestration, retries, failovers
│   │   ├── security.py          # API key auth & origin enforcement
│   │   └── logging_config.py    # Structured JSON logging formatter
│   ├── eval/
│   │   ├── run_eval.py          # IoU-based precision/recall/F1 evaluation harness
│   │   ├── ground_truth.json    # Labeled ground truth annotations
│   │   └── images/              # Benchmark image assets
│   ├── scripts/
│   │   ├── benchmark.py         # Latency profiling tool (p50, p95, p99)
│   │   └── download_model.sh    # Script to fetch latest YOLO weights
│   ├── models/
│   │   └── yolov8n.onnx         # ONNX weights file (runtime artifact)
│   ├── tests/
│   │   └── test_main.py         # Comprehensive unit and integration test suite
│   ├── Dockerfile               # Multi-stage non-root container definition
│   ├── requirements.txt         # Production runtime dependencies
│   └── requirements-dev.txt     # Linting, testing, and benchmark tools
├── frontend/
│   ├── index.html               # Enterprise inspection dashboard UI
│   ├── app.js                   # Stream processing, canvas rendering, telemetry
│   ├── config.js                # Frontend runtime parameters
│   ├── styles.css               # Clean operational layout styles
│   └── vercel.json              # Static deployment configuration
├── docker-compose.yml           # Local multi-service orchestrator
└── README.md
```

---

## Getting Started

### Prerequisites

- **Python:** 3.10 or higher
- **Node / Static Server:** Any static web server (e.g., Python `http.server`, Node, or VSCode Live Server)
- **Google AI Studio Key:** Required for Tier 2 escalation ([Get API Key](https://ai.google.dev/))

### 1. Backend Setup

Navigate to the `backend` directory and create a virtual environment:

```bash
cd backend
python -m venv .venv

# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Configure environment variables:

```bash
cp .env.example .env
```

Edit `.env` with your settings. Lists must be valid JSON arrays using double quotes:

```env
ENVIRONMENT=development
LOG_LEVEL=INFO

# Gemini Configuration
GEMINI_API_KEY="AIzaSy..."
CANDIDATE_MODELS=["models/gemini-3.6-flash"]
MAX_RETRIES_PER_MODEL=3
REQUEST_TIMEOUT_SECONDS=30

# Security & Ingress
API_KEY="your-secure-internal-key"
ALLOWED_ORIGINS=["http://127.0.0.1:5500", "http://localhost:3000"]

# Performance & Rate Limits
RATE_LIMIT_PER_MINUTE=60
MAX_IMAGE_SIZE_MB=5
ALLOWED_IMAGE_TYPES=["image/png", "image/jpeg", "image/webp"]
```

Launch the application:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Frontend Setup

Open `frontend/config.js` and verify endpoint and key parameters match your backend:

```javascript
window.APP_CONFIG = {
  BACKEND_URL: "http://127.0.0.1:8000",
  API_KEY: "your-secure-internal-key",
};
```

Serve the `frontend` directory using a local HTTP server:

```bash
cd frontend
python -m http.server 5500
```

Open [http://127.0.0.1:5500](http://127.0.0.1:5500) in a WebRTC-supported browser (Chrome, Edge, Firefox).

### 3. Running with Docker Compose

To deploy both services in isolated containers:

```bash
# Ensure backend/.env is populated first
docker compose up --build -d
```

---

## Automated Verification & Benchmarks

Run the test suite to validate authentication, schema boundaries, and mocked routing failover paths:

```bash
cd backend
pytest -v
```

Execute the evaluation regression gate:

```bash
# Fails with a non-zero exit code if F1 falls below threshold
python -m eval.run_eval --min-f1 0.85
```

Profile model execution latency on target deployment hardware:

```bash
python -m scripts.benchmark --image eval/images/bus.jpg --runs 200
```

---

## API Specification

### Health Check

`GET /health`

Response (200 OK):

```json
{
  "status": "healthy",
  "timestamp": 1726960000.123
}
```

### Frame Inspection

`POST /api/v1/inspect`

Headers:

```text
Content-Type: application/json
X-API-Key: <configured_api_key>
```

Payload:

```json
{
  "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "deep_analysis": false
}
```

Response (200 OK):

```json
{
  "status": "success",
  "engine": "local",
  "model_used": "yolov8n (on-device, ONNX Runtime)",
  "latency_ms": 104,
  "detections": [
    {
      "label": "person",
      "risk": "Medium",
      "box_2d": [352, 827, 814, 999],
      "confidence": 0.892
    }
  ]
}
```

### Field Schema

| Field                  | Type                          | Description                                                              |
| ---------------------- | ----------------------------- | ------------------------------------------------------------------------ |
| `status`               | string                        | Execution result (`success` or `failed`).                                |
| `engine`               | string                        | Tier engine responsible for resolution (`local` or `gemini`).            |
| `model_used`           | string                        | Specific runtime weight or cloud model signature.                        |
| `latency_ms`           | integer                       | Wall-clock execution time of inference pipeline in milliseconds.         |
| `detections`           | array                         | Array of identified objects and calculated risk evaluations.             |
| `detections[].box_2d`  | [ymin, xmin, ymax, xmax]      | Normalized bounding box coordinates on scale 0-1000.                     |