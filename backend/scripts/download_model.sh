#!/usr/bin/env bash
# Fetches the YOLOv8n ONNX weights used by the local detector.
# Model weights are never committed to git (see backend/.gitignore) —
# this script is run once at setup time or as a Docker build step.
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
MODEL_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx"
MODEL_PATH="${MODEL_DIR}/yolov8n.onnx"

mkdir -p "${MODEL_DIR}"

if [ -f "${MODEL_PATH}" ]; then
    echo "Model already present at ${MODEL_PATH}, skipping download."
    exit 0
fi

echo "Downloading YOLOv8n ONNX weights (~12.8 MB) to ${MODEL_PATH} ..."
curl -sL -o "${MODEL_PATH}" "${MODEL_URL}"
echo "Done."
