# -*- coding: utf-8 -*-
"""
Configuration for VANTOR Sentinel.
Adjust these values without touching code.
"""

# ------------------- Detection -------------------
# YOLOv5 model variant ("yolov5s", "yolov5m", "yolov5l", "yolov5x")
DETECTION_MODEL = "yolov5s"

# Minimum confidence for a detection to be kept
DETECTION_CONFIDENCE = 0.3

# GPU usage flag (set True if a compatible GPU is available)
USE_GPU = False

# ------------------- Tracking --------------------
# Maximum number of consecutive missed frames before a track is removed
TRACK_MAX_AGE = 30

# Minimum number of hits (matches) before a track is considered "confirmed"
TRACK_MIN_HITS = 3

# IoU threshold required to associate a detection with an existing track
TRACK_IOU_THRESHOLD = 0.3

# Debug logging controls
DEBUG_LOGGING = True  # Set to False to disable file logging
DEBUG_LOG_PATH = "logs/capture.log"  # Relative to project root
