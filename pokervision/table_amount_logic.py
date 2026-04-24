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

    same_label_keys = [
        key
        for key in digit_detection_map.keys()
        if key.startswith(f"{region.label}_") or key.startswith(f"{region.label}@")
    ]
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


def _occupied_positions(positions: Dict[str, Dict[str, object]]) -> List[str]:
    return list(positions.keys())


def _bbox_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    x1 = max(float(a.get("x1", 0.0)), float(b.get("x1", 0.0)))
    y1 = max(float(a.get("y1", 0.0)), float(b.get("y1", 0.0)))
    x2 = min(float(a.get("x2", 0.0)), float(b.get("x2", 0.0)))
    y2 = min(float(a.get("y2", 0.0)), float(b.get("y2", 0.0)))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a.get("x2", 0.0)) - float(a.get("x1", 0.0))) * max(
        0.0, float(a.get("y2", 0.0)) - float(a.get("y1", 0.0))
    )
    area_b = max(0.0, float(b.get("x2", 0.0)) - float(b.get("x1", 0.0))) * max(
        0.0, float(b.get("y2", 0.0)) - float(b.get("y1", 0.0))
    )
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return inter / union


def _bbox_center(bbox: Dict[str, float]) -> Point:
    return (
        (float(bbox.get("x1", 0.0)) + float(bbox.get("x2", 0.0))) / 2.0,
        (float(bbox.get("y1", 0.0)) + float(bbox.get("y2", 0.0))) / 2.0,
    )


def _same_money_object(existing: Dict[str, object], new_entry: Dict[str, object], settings) -> bool:
    existing_bbox = existing.get("bbox") if isinstance(existing.get("bbox"), dict) else {}
    new_bbox = new_entry.get("bbox") if isinstance(new_entry.get("bbox"), dict) else {}
    if not existing_bbox or not new_bbox:
        return False
    if _bbox_iou(existing_bbox, new_bbox) >= float(getattr(settings, "chips_conflict_iou_threshold", 0.30)):
        return True
    center_distance = _distance(_bbox_center(existing_bbox), _bbox_center(new_bbox))
    return center_distance <= float(
        getattr(
            settings,
            "chips_conflict_center_threshold_px",
            max(float(getattr(settings, "table_amount_region_center_threshold_px", 40.0)), 55.0),
        )
    )


def _merge_warning_lists(*warning_lists: object) -> List[str]:
    merged: List[str] = []
    for warnings in warning_lists:
        if not isinstance(warnings, list):
            continue
        for warning in warnings:
            text = str(warning)
            if text not in merged:
                merged.append(text)
    return merged


def _prefer_entry(existing: Dict[str, object], new_entry: Dict[str, object]) -> Dict[str, object]:
    existing_is_true_chips = str(existing.get("source")) == "Chips"
    new_is_true_chips = str(new_entry.get("source")) == "Chips"
    if existing_is_true_chips != new_is_true_chips:
        return existing if existing_is_true_chips else new_entry

    existing_score = existing.get("match_score_px")
    new_score = new_entry.get("match_score_px")
    if isinstance(existing_score, (int, float)) and isinstance(new_score, (int, float)):
        if new_score < existing_score:
            return new_entry
        if existing_score < new_score:
            return existing

    existing_amount = existing.get("amount_bb")
    new_amount = new_entry.get("amount_bb")
    if isinstance(existing_amount, (int, float)) and isinstance(new_amount, (int, float)):
        if new_amount > existing_amount:
            return new_entry
        if existing_amount > new_amount:
            return existing

    return existing


def _store_position_bet(state: TableAmountState, position: str, entry: Dict[str, object], settings) -> None:
    existing = state.bets_by_position.get(position)
    if existing is None:
        state.bets_by_position[position] = entry
        return

    same_object = _same_money_object(existing, entry, settings)
    chosen = _prefer_entry(existing, entry)
    rejected = entry if chosen is existing else existing

    chosen = dict(chosen)
    chosen["warnings"] = _merge_warning_lists(
        chosen.get("warnings"),
        rejected.get("warnings"),
        [
            "postflop_blind_marker_conflict_resolved_to_chips"
            if same_object
            else "multiple_money_objects_matched_to_same_position"
        ],
    )
    state.bets_by_position[position] = chosen


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
            scored.append(
                (
                    pos_name,
                    seat_dist,
                    {
                        "score": seat_dist,
                        "target_distance_px": seat_dist,
                        "seat_distance_px": seat_dist,
                        "line_distance_px": seat_dist,
                        "projection_ratio": 0.0,
                        "outside_penalty_px": 0.0,
                    },
                )
            )
        else:
            score, meta = _chips_match_score(region_center, pos_center, table_center, settings)
            scored.append((pos_name, score, meta))
    scored.sort(key=lambda item: item[1])
    return scored


def _chip_match_status(
    scored: List[Tuple[str, float, Dict[str, float]]],
    settings,
) -> tuple[str | None, float | None, Dict[str, float] | None, float | None, str | None]:
    if not scored:
        return None, None, None, None, "no_candidates"
    best_pos, best_score, best_meta = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else None
    if best_score > settings.chips_to_position_max_distance_px:
        return best_pos, best_score, best_meta, second_score, "too_far"
    if second_score is not None and abs(second_score - best_score) <= settings.chips_ambiguity_margin_px:
        return best_pos, best_score, best_meta, second_score, "ambiguous"
    return best_pos, best_score, best_meta, second_score, None


def _is_folded_position(
    player_states: Dict[str, Dict[str, object]] | None,
    position: str,
) -> bool:
    return bool((player_states or {}).get(position, {}).get("is_fold", False))


def _folded_override_margin_px(settings) -> float:
    return float(
        getattr(
            settings,
            "chips_folded_override_margin_px",
            max(float(getattr(settings, "chips_ambiguity_margin_px", 40.0)) * 2.0, 35.0),
        )
    )


def _logical_blind_position(blind_label: str, positions: Dict[str, Dict[str, object]]) -> str | None:
    if blind_label == "BB":
        return "BB" if "BB" in positions else None
    if blind_label == "SB":
        if "SB" in positions:
            return "SB"
        if "BTN" in positions:
            return "BTN"
    return None


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


def _assign_chip_like_entry(
    *,
    state: TableAmountState,
    entry: Dict[str, object],
    region_center: Point,
    position_centers: Dict[str, Point],
    positions: Dict[str, Dict[str, object]],
    player_states: Dict[str, Dict[str, object]] | None,
    live_positions: List[str],
    table_center: Point | None,
    street: str,
    settings,
) -> None:
    inside_pot_exclusion = False
    if table_center is not None and _distance(region_center, table_center) <= settings.table_pot_center_exclusion_radius_px:
        # Do NOT immediately discard Chips near the pot/table center.
        #
        # On compact table layouts the upper player's betting lane can be very close
        # to TotalPot. A real bet/open from the top seat can therefore fall inside
        # table_pot_center_exclusion_radius_px and was previously pushed straight to
        # unassigned_chips before seat geometry had any chance to match it.
        #
        # Keep the warning, continue through normal betting-lane matching, and only
        # allow the assignment if the match is strong enough. Weak/ambiguous center
        # chips are still dropped below.
        inside_pot_exclusion = True
        entry.setdefault("warnings", []).append("Chips region inside pot exclusion radius")

    live_candidate_names = [name for name in live_positions if name in position_centers]
    occupied_candidate_names = [name for name in _occupied_positions(positions) if name in position_centers]
    if not live_candidate_names:
        live_candidate_names = list(position_centers.keys())
    if not occupied_candidate_names:
        occupied_candidate_names = list(position_centers.keys())
    if not occupied_candidate_names:
        entry.setdefault("warnings", []).append("No logical positions available for chips matching")
        state.unassigned_chips.append(entry)
        return

    live_scored = _score_chip_candidates(region_center, live_candidate_names, position_centers, table_center, settings)
    live_best_pos, live_best_score, live_best_meta, live_second_score, live_rejection = _chip_match_status(live_scored, settings)

    occupied_scored: List[Tuple[str, float, Dict[str, float]]] = []
    occupied_best_pos = None
    occupied_best_score = None
    occupied_best_meta = None
    occupied_second_score = None
    occupied_rejection = None
    if street == "preflop" and player_states:
        # CRITICAL INVARIANT:
        # On preflop, Chips matching must NOT be restricted to live / non-folded
        # players only. A player can already be folded in the current frame while
        # their historical preflop contribution is still visible on the table.
        # If live-only matching is restored here, spots like
        # open -> flat -> squeeze/3bet -> opener folds will lose prior
        # contributions and preflop reconstruction will become false again.
        occupied_scored = _score_chip_candidates(region_center, occupied_candidate_names, position_centers, table_center, settings)
        occupied_best_pos, occupied_best_score, occupied_best_meta, occupied_second_score, occupied_rejection = _chip_match_status(occupied_scored, settings)

    chosen_pos = live_best_pos
    chosen_score = live_best_score
    chosen_meta = live_best_meta
    chosen_second_score = live_second_score
    chosen_rejection = live_rejection
    chosen_phase = "live_positions"
    matched_live_only = bool(player_states)

    if occupied_scored and occupied_best_pos is not None and occupied_rejection is None:
        override_margin = _folded_override_margin_px(settings)
        occupied_is_folded = _is_folded_position(player_states, occupied_best_pos)
        live_missing_or_rejected = live_best_pos is None or live_rejection is not None
        folded_override = False
        if occupied_is_folded:
            if live_missing_or_rejected:
                folded_override = True
            elif live_best_pos != occupied_best_pos and live_best_score is not None and occupied_best_score is not None:
                folded_override = (occupied_best_score + override_margin) < live_best_score
        if folded_override:
            chosen_pos = occupied_best_pos
            chosen_score = occupied_best_score
            chosen_meta = occupied_best_meta
            chosen_second_score = occupied_second_score
            chosen_rejection = None
            chosen_phase = "occupied_positions_fallback"
            matched_live_only = False
            entry.setdefault("warnings", []).append("matched_to_folded_position_as_residual_preflop_contribution")
            if live_rejection is not None:
                entry.setdefault("warnings", []).append(f"Live-only chips matching fallback used: {live_rejection}")
            else:
                entry.setdefault("warnings", []).append("Folded-position residual override used on preflop")

    entry["matched_position"] = chosen_pos
    entry["street"] = street
    entry["matched_among_live_positions_only"] = matched_live_only
    entry["match_phase"] = chosen_phase
    entry["candidate_positions"] = [name for name, _, _ in (live_scored if matched_live_only else occupied_scored or live_scored)]
    if live_scored:
        entry["candidate_positions_live_only"] = [name for name, _, _ in live_scored]
    if occupied_scored:
        entry["candidate_positions_all_occupied"] = [name for name, _, _ in occupied_scored]

    if chosen_score is not None:
        entry["match_score_px"] = round(chosen_score, 3)
    if chosen_meta is not None:
        entry["match_details"] = chosen_meta

    if inside_pot_exclusion and chosen_rejection is None:
        recovery_max_score = float(
            getattr(
                settings,
                "chips_pot_exclusion_recovery_max_score_px",
                min(float(getattr(settings, "chips_to_position_max_distance_px", 120.0)), 60.0),
            )
        )
        recovery_margin = float(
            getattr(
                settings,
                "chips_pot_exclusion_recovery_margin_px",
                max(float(getattr(settings, "chips_ambiguity_margin_px", 40.0)) * 2.0, 70.0),
            )
        )
        confident_score = chosen_score is not None and float(chosen_score) <= recovery_max_score
        confident_margin = (
            chosen_second_score is None
            or chosen_score is None
            or (float(chosen_second_score) - float(chosen_score)) >= recovery_margin
        )
        if not (confident_score and confident_margin):
            entry.setdefault("warnings", []).append(
                "Chips inside pot exclusion radius without confident betting-lane recovery"
            )
            entry["pot_exclusion_recovery_rejected"] = True
            entry["pot_exclusion_recovery_max_score_px"] = round(recovery_max_score, 3)
            entry["pot_exclusion_recovery_margin_px"] = round(recovery_margin, 3)
            if chosen_second_score is not None:
                entry["second_best_match_score_px"] = round(chosen_second_score, 3)
            state.unassigned_chips.append(entry)
            return
        entry.setdefault("warnings", []).append(
            "Chips inside pot exclusion radius recovered by confident betting-lane match"
        )
        entry["pot_exclusion_recovery"] = True
        entry["pot_exclusion_recovery_max_score_px"] = round(recovery_max_score, 3)
        entry["pot_exclusion_recovery_margin_px"] = round(recovery_margin, 3)

    if chosen_rejection == "too_far":
        entry.setdefault("warnings", []).append("Chips too far from any logical betting lane")
        state.unassigned_chips.append(entry)
        return
    if chosen_rejection == "ambiguous":
        entry.setdefault("warnings", []).append("Chips ambiguous between two positions")
        if chosen_second_score is not None:
            entry["second_best_match_score_px"] = round(chosen_second_score, 3)
        state.unassigned_chips.append(entry)
        return

    if chosen_pos is None:
        entry.setdefault("warnings", []).append("No confident chips match found")
        state.unassigned_chips.append(entry)
        return

    if chosen_second_score is not None and chosen_phase != "occupied_positions_fallback":
        entry["second_best_match_score_px"] = round(chosen_second_score, 3)

    _store_position_bet(state, chosen_pos, entry, settings)


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

        if region.label in {"SB", "BB"} and street != "preflop":
            # CRITICAL INVARIANT:
            # Blind labels (SB/BB) are meaningful only on preflop as diagnostics for
            # forced blinds. On flop/turn/river they must not survive as posted blinds.
            # Any postflop SB/BB detection must be reinterpreted as Chips or dropped as
            # noise, otherwise money-state reconstruction becomes inconsistent and later
            # solver input breaks.
            entry.setdefault("warnings", []).append("postflop_blind_marker_reinterpreted_as_chips")
            entry["reinterpreted_from_blind_marker"] = region.label
            _assign_chip_like_entry(
                state=state,
                entry=entry,
                region_center=region.center,
                position_centers=position_centers,
                positions=positions,
                player_states=player_states,
                live_positions=live_positions,
                table_center=table_center,
                street=street,
                settings=settings,
            )
            continue

        if region.label in {"SB", "BB"}:
            region_center = region.center
            logical_pos = _logical_blind_position(region.label, positions)

            nearest_pos = None
            nearest_dist = None
            for pos_name, pos_center in position_centers.items():
                dist = _distance(region_center, pos_center)
                if nearest_dist is None or dist < nearest_dist:
                    nearest_pos = pos_name
                    nearest_dist = dist

            matched_pos = nearest_pos
            logical_dist = None
            if logical_pos is not None and logical_pos in position_centers:
                logical_dist = _distance(region_center, position_centers[logical_pos])

            if street == "preflop" and logical_pos is not None:
                matched_pos = logical_pos
                if nearest_pos != logical_pos:
                    entry.setdefault("warnings", []).append(
                        f"Blind marker re-anchored from nearest {nearest_pos} to logical {logical_pos}"
                    )
                if logical_dist is not None and logical_dist > settings.blind_marker_to_position_max_distance_px:
                    entry.setdefault("warnings", []).append("Blind marker did not confidently match logical blind position")
                    warnings.append(f"{region.label}: ambiguous blind marker match")
            else:
                if logical_pos is not None and (
                    nearest_pos != logical_pos
                    or (nearest_dist is not None and nearest_dist > settings.blind_marker_to_position_max_distance_px)
                ):
                    entry.setdefault("warnings", []).append("Blind marker did not confidently match logical blind position")
                    warnings.append(f"{region.label}: ambiguous blind marker match")

            entry["matched_position"] = matched_pos
            if nearest_dist is not None:
                entry["nearest_position"] = nearest_pos
                entry["nearest_distance_px"] = round(nearest_dist, 3)
            if logical_dist is not None:
                entry["logical_position"] = logical_pos
                entry["logical_distance_px"] = round(logical_dist, 3)
            state.posted_blinds[region.label] = entry
            continue

        if region.label == "Chips":
            _assign_chip_like_entry(
                state=state,
                entry=entry,
                region_center=region.center,
                position_centers=position_centers,
                positions=positions,
                player_states=player_states,
                live_positions=live_positions,
                table_center=table_center,
                street=street,
                settings=settings,
            )

    state.warnings.extend(warnings)
    state.errors.extend(errors)
    return ValidationResult(ok=not state.errors, errors=state.errors, warnings=state.warnings, meta={"table_amount_state": state.to_dict()})
