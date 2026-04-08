from __future__ import annotations

from typing import Dict, List, Tuple

from .models import Detection, TableAmountState, ValidationResult
from .validators import parse_numeric_tokens, validate_table_amount_regions


Point = tuple[float, float]


def _distance(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def _lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _projection_ratio(point: Point, start: Point, end: Point) -> float:
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq <= 1e-9:
        return 0.0
    px = point[0] - start[0]
    py = point[1] - start[1]
    return (px * vx + py * vy) / seg_len_sq


def _point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    t = _projection_ratio(point, start, end)
    t = max(0.0, min(1.0, t))
    closest = _lerp(start, end, t)
    return _distance(point, closest)


def _build_position_centers(positions: Dict[str, Dict[str, object]]) -> Dict[str, Point]:
    return {
        name: (
            float(payload.get("center", {}).get("x", 0.0)),
            float(payload.get("center", {}).get("y", 0.0)),
        )
        for name, payload in positions.items()
    }




def _region_key(region: Detection) -> str:
    bbox = region.bbox
    return (
        f"{region.label}@"
        f"{round(bbox.x1, 1)}:{round(bbox.y1, 1)}:{round(bbox.x2, 1)}:{round(bbox.y2, 1)}"
    )


def _resolve_region_digits(
    region: Detection,
    cleaned_index: int,
    digit_detection_map: Dict[str, List[Detection]],
) -> List[Detection]:
    preferred_keys = [
        _region_key(region),
        f"{region.label}_{cleaned_index}",
    ]
    for key in preferred_keys:
        digits = digit_detection_map.get(key)
        if digits is not None:
            return digits

    same_label_keys = [key for key in digit_detection_map.keys() if key.startswith(f"{region.label}_") or key.startswith(f"{region.label}@")]
    same_label_keys = list(dict.fromkeys(same_label_keys))
    if len(same_label_keys) == 1:
        return digit_detection_map.get(same_label_keys[0], [])
    return []

def _candidate_positions(
    positions: Dict[str, Dict[str, object]],
    player_states: Dict[str, Dict[str, object]] | None,
) -> List[str]:
    names = list(positions.keys())
    if not player_states:
        return names
    live = [name for name in names if not bool(player_states.get(name, {}).get("is_fold", False))]
    return live or names


def _score_chip_candidates(
    region_center: Point,
    candidate_names: List[str],
    position_centers: Dict[str, Point],
    table_center: Point | None,
    settings,
) -> List[Tuple[str, float, Dict[str, float]]]:
    scored: List[Tuple[str, float, Dict[str, float]]] = []
    for pos_name in candidate_names:
        pos_center = position_centers[pos_name]
        if table_center is None:
            seat_dist = _distance(region_center, pos_center)
            scored.append((pos_name, seat_dist, {
                "score": seat_dist,
                "target_distance_px": seat_dist,
                "seat_distance_px": seat_dist,
                "line_distance_px": seat_dist,
                "projection_ratio": 0.0,
                "outside_penalty_px": 0.0,
            }))
        else:
            score, meta = _chips_match_score(region_center, pos_center, table_center, settings)
            scored.append((pos_name, score, meta))
    scored.sort(key=lambda item: item[1])
    return scored


def _chips_match_score(chips_center: Point, position_center: Point, table_center: Point, settings) -> tuple[float, Dict[str, float]]:
    anchor_t = float(getattr(settings, "chips_target_towards_table_center", 0.58))
    segment_slack = float(getattr(settings, "chips_projection_outside_segment_slack", 0.22))
    outside_penalty_scale = float(getattr(settings, "chips_projection_penalty_scale_px", 220.0))
    line_weight = float(getattr(settings, "chips_line_distance_weight", 0.85))

    target_point = _lerp(position_center, table_center, anchor_t)
    target_dist = _distance(chips_center, target_point)
    seat_dist = _distance(chips_center, position_center)
    line_dist = _point_to_segment_distance(chips_center, position_center, table_center)
    projection = _projection_ratio(chips_center, position_center, table_center)

    outside_penalty = 0.0
    if projection < -segment_slack:
        outside_penalty = abs(projection + segment_slack) * outside_penalty_scale
    elif projection > 1.0 + segment_slack:
        outside_penalty = abs(projection - (1.0 + segment_slack)) * outside_penalty_scale

    score = min(target_dist, seat_dist) + line_weight * line_dist + outside_penalty
    return score, {
        "score": score,
        "target_distance_px": target_dist,
        "seat_distance_px": seat_dist,
        "line_distance_px": line_dist,
        "projection_ratio": projection,
        "outside_penalty_px": outside_penalty,
    }


def build_table_amount_state(
    region_detections: List[Detection],
    digit_detection_map: Dict[str, List[Detection]],
    positions: Dict[str, Dict[str, object]],
    player_states: Dict[str, Dict[str, object]] | None,
    table_center: tuple[float, float] | None,
    street: str,
    settings,
) -> ValidationResult:
    region_val = validate_table_amount_regions(
        region_detections,
        settings.table_amount_region_iou_threshold,
        settings.table_amount_region_center_threshold_px,
    )
    warnings = list(region_val.warnings)
    errors = list(region_val.errors)
    state = TableAmountState()

    if not region_val.ok:
        state.errors.extend(errors)
        return ValidationResult(ok=False, errors=errors, warnings=warnings, meta={"table_amount_state": state.to_dict()})

    position_centers = _build_position_centers(positions)
    live_positions = _candidate_positions(positions, player_states)

    for idx, region in enumerate(region_val.meta["cleaned"]):
        digits = _resolve_region_digits(region, idx, digit_detection_map)
        digits_val = parse_numeric_tokens(
            digits,
            settings.table_amount_digit_iou_threshold,
            settings.table_amount_digit_center_threshold_px,
        )
        entry = {
            "raw_text": digits_val.meta.get("raw_text", ""),
            "amount_bb": digits_val.meta.get("amount"),
            "bbox": region.bbox.to_dict(),
            "digit_tokens": list(digits_val.meta.get("digit_tokens", [])),
            "warnings": list(digits_val.warnings),
            "source": region.label,
        }
        if digits_val.errors:
            entry.setdefault("warnings", []).extend(digits_val.errors)
            warnings.extend([f"{region.label}: {msg}" for msg in digits_val.errors])

        if region.label == "TotalPot":
            state.total_pot = entry
            continue

        if region.label in {"SB", "BB"}:
            best_pos = None
            best_dist = None
            region_center = region.center
            for pos_name, pos_center in position_centers.items():
                dist = _distance(region_center, pos_center)
                if best_dist is None or dist < best_dist:
                    best_pos = pos_name
                    best_dist = dist
            entry["matched_position"] = best_pos
            if best_pos != region.label or (best_dist is not None and best_dist > settings.blind_marker_to_position_max_distance_px):
                entry.setdefault("warnings", []).append("Blind marker did not confidently match logical blind position")
                warnings.append(f"{region.label}: ambiguous blind marker match")
            state.posted_blinds[region.label] = entry
            continue

        if region.label == "Chips":
            region_center = region.center
            if table_center is not None and _distance(region_center, table_center) <= settings.table_pot_center_exclusion_radius_px:
                entry.setdefault("warnings", []).append("Chips region inside pot exclusion radius")
                state.unassigned_chips.append(entry)
                continue

            live_candidate_names = [name for name in live_positions if name in position_centers]
            all_candidate_names = [name for name in position_centers.keys()]
            if not live_candidate_names:
                live_candidate_names = list(all_candidate_names)
            if not all_candidate_names:
                entry.setdefault("warnings", []).append("No logical positions available for chips matching")
                state.unassigned_chips.append(entry)
                continue

            scored_live = _score_chip_candidates(region_center, live_candidate_names, position_centers, table_center, settings)
            scored_all = _score_chip_candidates(region_center, all_candidate_names, position_centers, table_center, settings)

            scored = scored_live
            match_scope = "live_only"
            best_pos, best_score, best_meta = scored_live[0]
            second_score = scored_live[1][1] if len(scored_live) > 1 else None
            ambiguous_live = second_score is not None and abs(second_score - best_score) <= settings.chips_ambiguity_margin_px
            max_distance = settings.chips_to_position_max_distance_px
            fallback_margin = max(6.0, float(getattr(settings, "chips_ambiguity_margin_px", 0.0)) / 2.0)

            if street == "preflop" and scored_all:
                best_all_pos, best_all_score, best_all_meta = scored_all[0]
                second_all_score = scored_all[1][1] if len(scored_all) > 1 else None
                ambiguous_all = second_all_score is not None and abs(second_all_score - best_all_score) <= settings.chips_ambiguity_margin_px
                best_all_is_folded = bool(player_states and player_states.get(best_all_pos, {}).get("is_fold", False))
                should_fallback = False
                if best_all_is_folded and best_all_score <= max_distance:
                    if best_score > max_distance or ambiguous_live:
                        should_fallback = True
                    elif best_all_pos != best_pos and best_all_score + fallback_margin < best_score:
                        should_fallback = True
                if should_fallback and not ambiguous_all:
                    scored = scored_all
                    match_scope = "all_positions_fallback"
                    best_pos, best_score, best_meta = best_all_pos, best_all_score, best_all_meta
                    second_score = second_all_score
                    entry.setdefault("warnings", []).append("Matched chips to folded position as residual preflop contribution")

            entry["matched_position"] = best_pos
            entry["match_score_px"] = round(best_score, 3)
            entry["match_details"] = best_meta
            entry["candidate_positions"] = [name for name, _, _ in scored]
            entry["street"] = street
            entry["matched_among_live_positions_only"] = match_scope == "live_only"
            entry["match_scope"] = match_scope
            entry["matched_to_folded_position"] = bool(player_states and player_states.get(best_pos, {}).get("is_fold", False))

            if best_score > max_distance:
                entry.setdefault("warnings", []).append("Chips too far from any logical betting lane")
                state.unassigned_chips.append(entry)
                continue
            if second_score is not None and abs(second_score - best_score) <= settings.chips_ambiguity_margin_px:
                entry.setdefault("warnings", []).append("Chips ambiguous between two positions")
                entry["second_best_match_score_px"] = round(second_score, 3)
                state.unassigned_chips.append(entry)
                continue

            state.bets_by_position[best_pos] = entry

    state.warnings.extend(warnings)
    state.errors.extend(errors)
    return ValidationResult(ok=not state.errors, errors=state.errors, warnings=state.warnings, meta={"table_amount_state": state.to_dict()})
