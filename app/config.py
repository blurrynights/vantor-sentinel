# -*- coding: utf-8 -*-
"""Configuration for VANTOR Sentinel.
You can adjust these values without touching code.
"""

# Detection model name (YOLOv5 variants: yolov5s, yolov5m, yolov5l, yolov5x)
DETECTION_MODEL = "yolov5s"

# Minimum confidence for a detection to be considered
DETECTION_CONFIDENCE = 0.3

# Whether to enable GPU acceleration (if a compatible GPU is available)
USE_GPU = False
