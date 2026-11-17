# -*- coding: utf-8 -*-
"""
Public interface for the `app.perception` package.

Exports:
    - detect   : object‑detection function (YOLOv5)
    - Tracker  : simple SORT‑style multi‑object tracker
"""

# Re‑export the detection function
from .detect import detect  # noqa: F401

# Re‑export the Tracker class
from .tracker import Tracker  # noqa: F401

# Explicitly declare the public symbols
__all__ = ["detect", "Tracker"]
