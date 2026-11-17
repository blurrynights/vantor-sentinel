# -*- coding: utf-8 -*-
"""Webcam capture demo with YOLOv5 detection **and** simple tracking.
"""

import cv2
import sys
import time

from app.perception.detect import detect
from app.perception.tracker import Tracker
from app.config import (
    DETECTION_CONFIDENCE,
    TRACK_MAX_AGE,
    TRACK_MIN_HITS,
    TRACK_IOU_THRESHOLD,
)


def main():
    cap = cv2.VideoCapture(0)  # default webcam
    if not cap.isOpened():
        sys.exit('❌ Cannot open webcam')
    print('✅ Webcam opened – press q to quit')

    tracker = Tracker(
        max_age=TRACK_MAX_AGE,
        min_hits=TRACK_MIN_HITS,
        iou_threshold=TRACK_IOU_THRESHOLD,
    )

    prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print('⚠️ Frame capture failed')
            break

        # -------- Detection ----------
        detections = detect(frame, conf_thresh=DETECTION_CONFIDENCE)

        # DEBUG: show raw detections (first few)
        if detections:
            print('[DEBUG] detections:', detections[:3])
        else:
            print('[DEBUG] No detections on this frame')

        # -------- Tracking ----------
        detections = tracker.update(detections)

        # Draw detections + track IDs (guard against missing keys)
        for det in detections:
            if not all(k in det for k in ("bbox", "label", "confidence", "object_id")):
                continue
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['label']} {det['confidence']:.2f} (ID:{det['object_id']})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Show FPS
        now = time.time()
        fps = 1.0 / (now - prev)
        prev = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        cv2.imshow('Sentinel – Detection + Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
