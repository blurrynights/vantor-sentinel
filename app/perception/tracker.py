# -*- coding: utf-8 -*-
"""
Simple SORT‑style tracker for VANTOR Sentinel.

Exports the ``Tracker`` class that maintains object IDs across frames
using an IoU‑based association.
"""

import numpy as np
from typing import List, Tuple, Dict


def _iou(bb1: Tuple[int, int, int, int], bb2: Tuple[int, int, int, int]) -> float:
    """Intersection‑over‑Union of two bounding boxes.

    Parameters
    ----------
    bb1, bb2: (x1, y1, x2, y2) integer coordinates.
    """
    x1, y1, x2, y2 = bb1
    X1, Y1, X2, Y2 = bb2

    inter_x1 = max(x1, X1)
    inter_y1 = max(y1, Y1)
    inter_x2 = min(x2, X2)
    inter_y2 = min(y2, Y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (X2 - X1) * (Y2 - Y1)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


class Track:
    """Internal representation of a single tracked object."""

    def __init__(self, bbox: Tuple[int, int, int, int], track_id: int):
        self.bbox = bbox
        self.id = track_id
        self.age = 1
        self.hits = 1
        self.no_match_frames = 0

    def update(self, bbox: Tuple[int, int, int, int]):
        self.bbox = bbox
        self.age += 1
        self.hits += 1
        self.no_match_frames = 0

    def miss(self):
        self.no_match_frames += 1
        self.age += 1


class Tracker:
    """Very lightweight SORT‑style multi‑object tracker.

    Parameters
    ----------
    max_age : int
        Number of consecutive missed frames before a track is deleted.
    min_hits : int
        Minimum number of matches before a track is considered confirmed.
    iou_threshold : float
        Minimum IoU required to associate a detection with an existing track.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self._next_id = 1

    # -----------------------------------------------------------------
    def _associate(self, detections: List[Dict]) -> Tuple[Dict[int, int], List[int]]:
        """Greedy IoU association.

        Returns a mapping ``track_index`` → ``detection_index`` and a list of
        detection indices that could not be matched.
        """
        if not self.tracks or not detections:
            return {}, list(range(len(detections)))

        iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=float)
        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = _iou(track.bbox, det["bbox"])

        matches: Dict[int, int] = {}
        used_dets = set()
        while True:
            t, d = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[t, d]
            if max_iou < self.iou_threshold:
                break
            if t in matches or d in used_dets:
                iou_matrix[t, d] = -1
                continue
            matches[t] = d
            used_dets.add(d)
            iou_matrix[t, :] = -1
            iou_matrix[:, d] = -1
        unmatched = [i for i in range(len(detections)) if i not in used_dets]
        return matches, unmatched

    # -----------------------------------------------------------------
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update the tracker with new detections.

        The function adds an ``object_id`` key to each detection dict and
        returns the augmented list.
        """
        matches, unmatched_det_idxs = self._associate(detections)

        # Update matched tracks
        for trk_idx, det_idx in matches.items():
            track = self.tracks[trk_idx]
            det = detections[det_idx]
            track.update(det["bbox"])

        # Mark unmatched tracks as missed
        for i, track in enumerate(self.tracks):
            if i not in matches:
                track.miss()

        # Remove stale tracks
        self.tracks = [trk for trk in self.tracks if trk.no_match_frames <= self.max_age]

        # Initialise new tracks for unmatched detections
        for det_idx in unmatched_det_idxs:
            det = detections[det_idx]
            new_trk = Track(det["bbox"], self._next_id)
            self._next_id += 1
            self.tracks.append(new_trk)

        # Attach object IDs to detections
        output: List[Dict] = []
        for det in detections:
            matching = [trk for trk in self.tracks if trk.bbox == det["bbox"]]
            det["object_id"] = matching[0].id if matching else -1
            output.append(det)
        return output
