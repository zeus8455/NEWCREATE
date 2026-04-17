from __future__ import annotations

import math
from typing import Iterable, List

from .models import BBox, Detection


def iou(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: Detection, b: Detection) -> float:
    ax, ay = a.center
    bx, by = b.center
    return math.hypot(ax - bx, ay - by)


def dedupe_detections(
    detections: Iterable[Detection],
    iou_threshold: float,
    center_threshold: float,
) -> List[Detection]:
    kept: List[Detection] = []
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    for det in sorted_dets:
        duplicate = False
        for best in kept:
            if det.label == best.label and (
                iou(det.bbox, best.bbox) >= iou_threshold
                or center_distance(det, best) <= center_threshold
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def filter_by_label(detections: Iterable[Detection], label: str) -> List[Detection]:
    return [d for d in detections if d.label == label]
