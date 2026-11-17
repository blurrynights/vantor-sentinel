# -*- coding: utf-8 -*-
"""
Object‑detection utilities for VANTOR Sentinel.

This implementation uses the Ultralytics YOLOv5 model (default: "yolov5s").
It provides a single public function:

    detect(frame: np.ndarray, conf_thresh: float = 0.3) -> List[Dict]

where each dict contains:
    - ``bbox``   : (x1, y1, x2, y2) in pixel coordinates
    - ``label``  : human‑readable class name
    - ``confidence`` : detection confidence (0‑1)

The function works with a BGR OpenCV frame (as returned by ``cv2.VideoCapture``).
"""

import cv2
import numpy as np
from typing import List, Dict

# ---------------------------------------------------------------
# Load the YOLOv5 model (weights are downloaded automatically on first use)
# ---------------------------------------------------------------
from ultralytics import YOLO

# You can change the model name via an environment variable or config later.
_MODEL_NAME = "yolov5s"  # lightweight model suitable for real‑time on macOS
_model = YOLO(f"{_MODEL_NAME}.pt")  # will download weights if missing


def _format_results(results, conf_thresh: float) -> List[Dict]:
    """Convert Ultralytics results to our plain‑dict format.

    Parameters
    ----------
    results : ultralytics.engine.results.Results
        The prediction object returned by ``model(frame)``.
    conf_thresh : float
        Minimum confidence to keep a detection.
    """
    detections: List[Dict] = []
    for r in results:
        # ``r.boxes`` holds tensor of shape (N, 6): [x1, y1, x2, y2, conf, cls]
        if r.boxes is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()  # (N, 4)
        scores = r.boxes.conf.cpu().numpy()  # (N,)
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)  # (N,)
        for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, cls_ids):
            if score < conf_thresh:
                continue
            label = r.names[cls_id] if hasattr(r, "names") else str(cls_id)
            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "label": label,
                "confidence": float(score),
            })
    return detections


def detect(frame: np.ndarray, conf_thresh: float = 0.3) -> List[Dict]:
    """Run YOLOv5 object detection on an OpenCV frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image from OpenCV.
    conf_thresh : float, optional
        Confidence threshold (default 0.3).
    """
    # Convert BGR -> RGB because YOLO expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Ultralytics model accepts a NumPy array directly
    results = _model(rgb, verbose=False)  # returns a list with one Results object
    return _format_results(results, conf_thresh)
