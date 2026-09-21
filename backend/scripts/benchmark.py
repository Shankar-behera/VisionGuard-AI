"""Latency benchmark for the local detector.

Measures wall-clock inference latency over N runs on a real image and
reports p50/p95/p99 — the numbers that actually matter for deciding
whether a frame needs to hit Gemini or can be served locally.

Usage:
    python -m scripts.benchmark --image eval/images/bus.jpg --runs 200
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.local_detector import LocalDetector  # noqa: E402

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolov8n.onnx"


def percentile(data, p):
    data = sorted(data)
    k = (len(data) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    if f == c:
        return data[f]
    return data[f] + (data[c] - data[f]) * (k - f)


def main():
    parser = argparse.ArgumentParser(description="Benchmark local detector inference latency.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--image", required=True)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    detector = LocalDetector(args.model)
    image = cv2.imread(args.image)
    if image is None:
        print(f"Could not read {args.image}", file=sys.stderr)
        sys.exit(1)

    for _ in range(args.warmup):
        detector.detect(image)

    latencies = []
    for _ in range(args.runs):
        start = time.perf_counter()
        detector.detect(image)
        latencies.append((time.perf_counter() - start) * 1000)

    print(f"Runs: {args.runs} (+{args.warmup} warmup, discarded) | Image: {args.image}")
    print(f"  mean: {statistics.mean(latencies):.1f} ms")
    print(f"  p50:  {percentile(latencies, 50):.1f} ms")
    print(f"  p95:  {percentile(latencies, 95):.1f} ms")
    print(f"  p99:  {percentile(latencies, 99):.1f} ms")
    print(f"  min:  {min(latencies):.1f} ms")
    print(f"  max:  {max(latencies):.1f} ms")


if __name__ == "__main__":
    main()
