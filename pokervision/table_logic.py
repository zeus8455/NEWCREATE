from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .models import BBox, Detection
from .postprocess import center_distance, iou


POSITION_ORDER = {
    2: ["BB"],
    3: ["SB", "BB"],
    4: ["SB", "BB", "CO"],
    5: ["SB", "BB", "UTG", "CO"],
    6: ["SB", "BB", "UTG", "MP", "CO"],
}


def compute_table_center(seats: List[Detection], btn: Detection) -> tuple[float, float]:
    points = [d.center for d in seats] + [btn.center]
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    return (x, y)


def _angle(center: tuple[float, float], point: tuple[float, float]) -> float:
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    return math.degrees(math.atan2(dy, dx))


def _clockwise_delta(start_angle: float, angle: float) -> float:
    return (angle - start_angle) % 360.0


def assign_positions(
    seats: List[Detection],
    btn: Detection,
    player_count: int,
) -> tuple[tuple[float, float], Dict[str, Dict[str, object]]]:
    table_center = compute_table_center(seats, btn)
    btn_angle = _angle(table_center, btn.center)
    sorted_seats = sorted(seats, key=lambda d: _clockwise_delta(btn_angle, _angle(table_center, d.center)))
    labels = POSITION_ORDER[player_count]
    positions: Dict[str, Dict[str, object]] = {
        "BTN": {
            "center": {"x": btn.center[0], "y": btn.center[1]},
            "bbox": btn.bbox.to_dict(),
            "angle_deg": btn_angle,
            "is_hero": False,
        }
    }
    for seat, name in zip(sorted_seats, labels):
        positions[name] = {
            "center": {"x": seat.center[0], "y": seat.center[1]},
            "bbox": seat.bbox.to_dict(),
            "angle_deg": _angle(table_center, seat.center),
            "is_hero": False,
        }
    return table_center, positions


def determine_hero_position(
    positions: Dict[str, Dict[str, object]],
    active_hero_detections: Optional[List[Detection]] = None,
    seat_match_max_distance_px: float = 85.0,
) -> str:
    if active_hero_detections:
        active = max(active_hero_detections, key=lambda d: d.confidence)
        best_name = None
        best_score = None
        for name, payload in positions.items():
            bbox = BBox(**payload["bbox"])
            pos_det = Detection(name, bbox, 1.0)
            overlap = iou(active.bbox, bbox)
            distance = center_distance(active, pos_det)
            score = (overlap > 0, overlap, -distance)
            if overlap > 0 or distance <= seat_match_max_distance_px:
                if best_score is None or score > best_score:
                    best_score = score
                    best_name = name
        if best_name is not None:
            return best_name
    return max(positions.items(), key=lambda item: item[1]["center"]["y"])[0]
