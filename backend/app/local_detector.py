"""Local object detection using a YOLOv8n ONNX model via ONNX Runtime.

This runs entirely on the backend host — no network call, no API cost,
sub-100ms on CPU for a single frame. It's used as a fast first pass:

  - Cheap frames (empty scene, nothing of interest) never touch Gemini.
  - Frames where the local model IS confident get an immediate response.
  - Frames where the local model is uncertain, or where scene-level
    reasoning about *risk* (not just "there is a forklift here") matters,
    escalate to Gemini for richer semantic judgment.

COCO's 80 classes aren't a safety-hazard taxonomy, so class -> risk
mapping below is a coarse heuristic (tunable via HAZARD_CLASS_RISK),
not a substitute for a model actually trained on hazard labels 
"""
import logging
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime as ort

from .models import DetectionItem

logger = logging.getLogger(__name__)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Coarse hazard-relevance heuristic: classes that commonly matter in a
# workplace/industrial safety context get elevated risk. Everything else
# defaults to "Low". This is intentionally simple — see eval/ for how it
# performs against labeled ground truth.
HAZARD_CLASS_RISK = {
    "person": "Medium",       # presence in a monitored zone is often the point
    "car": "Medium", "truck": "Medium", "bus": "Medium", "motorcycle": "Medium",
    "bicycle": "Low",
    "knife": "High", "scissors": "Medium",
    "fire hydrant": "Low",
    "oven": "Medium", "toaster": "Low", "microwave": "Low",
}
DEFAULT_RISK = "Low"

INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45


class LocalDetector:
    """Wraps an ONNX Runtime session for YOLOv8n inference."""

    def __init__(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Local detector model not found at {model_path}. "
                f"Run scripts/download_model.sh to fetch it."
            )
        # CPU is intentional: this is meant to run on a $7/mo web dyno,
        # not a GPU box. A CUDA provider can be added for self-hosted deploys.
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        logger.info("local_detector_loaded path=%s", model_path)

    @staticmethod
    def _letterbox(image: np.ndarray, size: int = INPUT_SIZE) -> Tuple[np.ndarray, float, int, int]:
        """Resize keeping aspect ratio, pad to a square. Returns
        (padded_image, scale, pad_x, pad_y) so boxes can be mapped back.
        """
        h, w = image.shape[:2]
        scale = min(size / h, size / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))

        import cv2  # local import keeps opencv optional at module-import time

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas, scale, pad_x, pad_y

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Standard greedy NMS, implemented directly (no cv2.dnn.NMSBoxes
        dependency) so behavior is explicit and unit-testable."""
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= iou_threshold]
        return keep

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
    ) -> Tuple[List[DetectionItem], int]:
        """Run detection on a raw BGR numpy image. Returns
        (detections, latency_ms). Box coordinates are normalized to the
        same [0, 1000] scale the Gemini path uses, so both paths are
        drop-in interchangeable for the frontend.
        """
        start = time.monotonic()
        orig_h, orig_w = image_bgr.shape[:2]

        padded, scale, pad_x, pad_y = self._letterbox(image_bgr, INPUT_SIZE)
        blob = padded[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, normalize
        blob = blob.transpose(2, 0, 1)[None, ...]  # HWC -> CHW -> NCHW

        raw_output = self.session.run(None, {self.input_name: blob})[0]  # [1, 84, 8400]
        predictions = raw_output[0].T  # -> [8400, 84]

        class_scores = predictions[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores.max(axis=1)

        mask = confidences >= confidence_threshold
        predictions, class_ids, confidences = predictions[mask], class_ids[mask], confidences[mask]

        if len(predictions) == 0:
            return [], int((time.monotonic() - start) * 1000)

        cx, cy, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        boxes_letterboxed = np.stack([x1, y1, x2, y2], axis=1)

        keep = self._nms(boxes_letterboxed, confidences, iou_threshold)

        detections: List[DetectionItem] = []
        for idx in keep:
            bx1, by1, bx2, by2 = boxes_letterboxed[idx]
            # Undo letterbox padding + scale to map back to the original image
            ox1 = max(0.0, (bx1 - pad_x) / scale)
            oy1 = max(0.0, (by1 - pad_y) / scale)
            ox2 = min(orig_w, (bx2 - pad_x) / scale)
            oy2 = min(orig_h, (by2 - pad_y) / scale)

            # Normalize to 0-1000 scale (matches the Gemini response contract)
            box_2d = [
                int(oy1 / orig_h * 1000),
                int(ox1 / orig_w * 1000),
                int(oy2 / orig_h * 1000),
                int(ox2 / orig_w * 1000),
            ]
            class_name = COCO_CLASSES[int(class_ids[idx])] if int(class_ids[idx]) < len(COCO_CLASSES) else "object"
            risk = HAZARD_CLASS_RISK.get(class_name, DEFAULT_RISK)

            detections.append(
                DetectionItem(
                    label=class_name,
                    risk=risk,
                    box_2d=box_2d,
                    confidence=round(float(confidences[idx]), 4),
                )
            )

        latency_ms = int((time.monotonic() - start) * 1000)
        logger.info("local_detect detections=%s latency_ms=%s", len(detections), latency_ms)
        return detections, latency_ms


def bytes_to_bgr_image(image_bytes: bytes) -> "np.ndarray | None":
    """Decode raw image bytes (PNG/JPEG/WebP) into an OpenCV BGR array."""
    import cv2

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


_detector_instance: "LocalDetector | None" = None


def get_local_detector(model_path: str) -> "LocalDetector | None":
    """Lazily load and cache the detector. Returns None (rather than
    raising) if the model file isn't present, so the service can still
    boot and fall back to Gemini-only mode — e.g. in an environment
    where the weights haven't been fetched yet.
    """
    global _detector_instance
    if _detector_instance is None:
        try:
            _detector_instance = LocalDetector(model_path)
        except FileNotFoundError as e:
            logger.warning("local_detector_unavailable reason=%s", e)
            return None
    return _detector_instance
