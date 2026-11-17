# -*- coding: utf-8 -*-
"""Webcam capture demo with YOLOv5 object detection.
Uses the detection module in ``app.perception.detect`` and configuration
from ``app.config``.
"""

import cv2
import sys
import time
from app.perception.detect import detect
from app.config import DETECTION_CONFIDENCE


def main():
    cap = cv2.VideoCapture(0)  # default webcam
    if not cap.isOpened():
        sys.exit('❌ Cannot open webcam')
    print('✅ Webcam opened – press q to quit')
    prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print('⚠️ Frame capture failed')
            break

        # Run object detection
        detections = detect(frame, conf_thresh=DETECTION_CONFIDENCE)
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['label']} {det['confidence']:.2f}"
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

        cv2.imshow('Sentinel – Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
