"""Evaluation harness for the local detector.

Runs the ONNX detector against a small set of hand-labeled images and
reports precision / recall / F1 at IoU >= 0.5, using class-agnostic
matching (since the point here is measuring localization quality, not
COCO's 80-way classification accuracy — that's a solved problem
upstream in the pretrained weights).

Usage:
    python -m eval.run_eval [--confidence 0.40] [--iou 0.5]

Exit code is non-zero if F1 drops below --min-f1, so this is wired
into CI as a regression gate on eval/report.json.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.local_detector import LocalDetector  # noqa: E402
from app.models import DetectionItem  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = EVAL_DIR.parent / "models" / "yolov8n.onnx"


def box_iou(a: List[int], b: List[int]) -> float:
    """IoU between two [ymin, xmin, ymax, xmax] boxes."""
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    inter_y1, inter_x1 = max(ay1, by1), max(ax1, bx1)
    inter_y2, inter_x2 = min(ay2, by2), min(ax2, bx2)
    inter_h, inter_w = max(0, inter_y2 - inter_y1), max(0, inter_x2 - inter_x1)
    inter_area = inter_h * inter_w
    a_area = max(0, ay2 - ay1) * max(0, ax2 - ax1)
    b_area = max(0, by2 - by1) * max(0, bx2 - bx1)
    union = a_area + b_area - inter_area
    return inter_area / union if union > 0 else 0.0


def match_detections(
    predictions: List[DetectionItem], ground_truth: List[dict], iou_threshold: float
) -> Tuple[int, int, int]:
    """Greedy IoU matching. Returns (true_positives, false_positives, false_negatives)."""
    unmatched_gt = list(range(len(ground_truth)))
    tp = 0
    for pred in predictions:
        best_iou, best_idx = 0.0, -1
        for i in unmatched_gt:
            iou = box_iou(pred.box_2d, ground_truth[i]["box_2d"])
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_threshold:
            tp += 1
            unmatched_gt.remove(best_idx)

    fp = len(predictions) - tp
    fn = len(unmatched_gt)
    return tp, fp, fn


def run(model_path: str, confidence: float, iou_threshold: float) -> dict:
    detector = LocalDetector(model_path)
    ground_truth = json.loads((EVAL_DIR / "ground_truth.json").read_text())
    ground_truth.pop("_note", None)

    total_tp = total_fp = total_fn = 0
    per_image = []

    for image_name, gt_boxes in ground_truth.items():
        image_path = EVAL_DIR / "images" / image_name
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"WARNING: could not read {image_path}, skipping", file=sys.stderr)
            continue

        predictions, latency_ms = detector.detect(image, confidence_threshold=confidence)
        tp, fp, fn = match_detections(predictions, gt_boxes, iou_threshold)
        total_tp, total_fp, total_fn = total_tp + tp, total_fp + fp, total_fn + fn

        per_image.append({
            "image": image_name,
            "predictions": len(predictions),
            "ground_truth": len(gt_boxes),
            "tp": tp, "fp": fp, "fn": fn,
            "latency_ms": latency_ms,
        })

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "iou_threshold": iou_threshold,
        "confidence_threshold": confidence,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_image": per_image,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the local detector against labeled ground truth.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--min-f1", type=float, default=0.0, help="Exit non-zero if F1 falls below this.")
    parser.add_argument("--out", default=str(EVAL_DIR / "report.json"))
    args = parser.parse_args()

    report = run(args.model, args.confidence, args.iou)
    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"Precision: {report['precision']:.3f}  Recall: {report['recall']:.3f}  F1: {report['f1']:.3f}")
    print(f"TP={report['total_tp']} FP={report['total_fp']} FN={report['total_fn']}")
    for row in report["per_image"]:
        print(f"  {row['image']:15s} pred={row['predictions']} gt={row['ground_truth']} "
              f"tp={row['tp']} fp={row['fp']} fn={row['fn']} latency={row['latency_ms']}ms")

    if report["f1"] < args.min_f1:
        print(f"FAIL: F1 {report['f1']:.3f} below threshold {args.min_f1}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
