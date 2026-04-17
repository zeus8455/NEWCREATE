from __future__ import annotations

import re
from typing import List, Sequence

from .card_format import detector_label_to_internal
from .models import Detection, PlayerState, PlayerToken, ValidationResult
from .postprocess import center_distance, dedupe_detections, iou


VALID_STREETS = {"preflop", "flop", "turn", "river"}
PLAYER_STATE_LABELS = {"fold", ".", "all-in", *[str(i) for i in range(10)]}
NUMERIC_PLAYER_STATE_LABELS = {".", *[str(i) for i in range(10)]}
NON_NUMERIC_PLAYER_STATE_LABELS = {"fold", "all-in"}
TABLE_AMOUNT_REGION_LABELS = {"Chips", "SB", "BB", "TotalPot"}
TABLE_AMOUNT_DIGIT_LABELS = {".", *[str(i) for i in range(10)]}


def validate_structure(detections: List[Detection], iou_threshold: float, center_threshold: float) -> ValidationResult:
    cleaned = dedupe_detections(detections, iou_threshold, center_threshold)
    btn = [d for d in cleaned if d.label == "BTN"]
    seats = [d for d in cleaned if d.label == "player_seat"]

    errors: list[str] = []
    if len(btn) != 1:
        errors.append(f"BTN count must be exactly 1, got {len(btn)}")
    if not (1 <= len(seats) <= 5):
        errors.append(f"player_seat count must be 1..5, got {len(seats)}")

    player_count = len(seats) + len(btn)
    table_format = f"{player_count}max" if player_count in {2, 3, 4, 5, 6} else None
    if table_format is None:
        errors.append("Unable to determine table format")

    return ValidationResult(
        ok=not errors,
        errors=errors,
        meta={
            "cleaned": cleaned,
            "btn": btn,
            "seats": seats,
            "player_count": player_count,
            "table_format": table_format,
        },
    )


def determine_street(detections: List[Detection]) -> ValidationResult:
    labels = {d.label for d in detections if d.label in {"Flop", "Turn", "River"}}
    mapping = {
        frozenset(): "preflop",
        frozenset({"Flop"}): "flop",
        frozenset({"Turn"}): "turn",
        frozenset({"River"}): "river",
    }
    street = mapping.get(frozenset(labels))
    if street is None:
        return ValidationResult(ok=False, errors=[f"Invalid street marker combination: {sorted(labels)}"])
    return ValidationResult(ok=True, meta={"street": street})


def validate_unique_cards(card_detections: List[Detection], expected_count: int) -> ValidationResult:
    errors: list[str] = []
    internal_cards: list[str] = []
    if len(card_detections) != expected_count:
        errors.append(f"Expected {expected_count} cards, got {len(card_detections)}")

    try:
        internal_cards = [detector_label_to_internal(d.label) for d in card_detections]
    except ValueError as exc:
        errors.append(str(exc))

    if len(internal_cards) != len(set(internal_cards)):
        errors.append("Duplicate cards by class detected")

    centers = [tuple(map(int, d.center)) for d in card_detections]
    if len(centers) != len(set(centers)):
        errors.append("Duplicate card positions detected")

    return ValidationResult(ok=not errors, errors=errors, meta={"cards": internal_cards})


def validate_hero_cards(card_detections: List[Detection]) -> ValidationResult:
    result = validate_unique_cards(card_detections, 2)
    if result.ok:
        sorted_cards = sorted(zip(card_detections, result.meta["cards"]), key=lambda item: item[0].bbox.x1)
        result.meta["cards"] = [card for _, card in sorted_cards]
    return result


def validate_board_cards(card_detections: List[Detection], street: str) -> ValidationResult:
    expected = {"flop": 3, "turn": 4, "river": 5}.get(street)
    if expected is None:
        return ValidationResult(ok=True, meta={"cards": []})
    result = validate_unique_cards(card_detections, expected)
    if result.ok:
        sorted_cards = sorted(zip(card_detections, result.meta["cards"]), key=lambda item: item[0].bbox.x1)
        result.meta["cards"] = [card for _, card in sorted_cards]
    return result


def _same_token_slot(a: Detection, b: Detection, iou_threshold: float, center_threshold: float) -> bool:
    overlap = iou(a.bbox, b.bbox)
    distance = center_distance(a, b)
    tight_iou_threshold = max(0.82, iou_threshold + 0.22)
    tight_center_threshold = max(4.0, center_threshold * 0.35)
    return overlap >= tight_iou_threshold or distance <= tight_center_threshold


def _dedupe_player_state_detections(detections: Sequence[Detection], iou_threshold: float, center_threshold: float) -> List[Detection]:
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda item: item.confidence, reverse=True):
        label = det.label.lower()
        if label not in PLAYER_STATE_LABELS:
            continue
        duplicate = False
        for best in kept:
            best_label = best.label.lower()
            if not _same_token_slot(det, best, iou_threshold, center_threshold):
                continue
            if label in NUMERIC_PLAYER_STATE_LABELS and best_label in NUMERIC_PLAYER_STATE_LABELS:
                duplicate = True
                break
            if label == best_label and label in NON_NUMERIC_PLAYER_STATE_LABELS:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def _parse_validate_player_state_args(args: tuple, kwargs: dict) -> tuple[str, List[Detection], int, float, float, float]:
    if len(args) < 2:
        raise TypeError("validate_player_state() requires at least position and detections")
    position = args[0]
    detections = args[1]
    crop_height = kwargs.pop("crop_height", None)
    lower_band_ratio = kwargs.pop("lower_band_ratio", None)
    iou_threshold = kwargs.pop("iou_threshold", None)
    center_threshold = kwargs.pop("center_threshold", None)
    remaining = list(args[2:])
    if crop_height is None and remaining:
        crop_height = remaining.pop(0)
    if lower_band_ratio is None and remaining:
        lower_band_ratio = remaining.pop(0)
    if iou_threshold is None and remaining:
        iou_threshold = remaining.pop(0)
    if center_threshold is None and remaining:
        center_threshold = remaining.pop(0)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"validate_player_state() got unexpected keyword arguments: {unexpected}")
    missing = [
        name for name, value in (
            ("crop_height", crop_height),
            ("lower_band_ratio", lower_band_ratio),
            ("iou_threshold", iou_threshold),
            ("center_threshold", center_threshold),
        ) if value is None
    ]
    if missing:
        raise TypeError(f"validate_player_state() missing required arguments: {', '.join(missing)}")
    return str(position), list(detections), int(crop_height), float(lower_band_ratio), float(iou_threshold), float(center_threshold)


def validate_player_state(*args, **kwargs) -> ValidationResult:
    position, detections, crop_height, lower_band_ratio, iou_threshold, center_threshold = _parse_validate_player_state_args(args, kwargs)
    cleaned = _dedupe_player_state_detections(detections, iou_threshold, center_threshold)
    lower_band_y = crop_height * lower_band_ratio

    state_markers = [detection for detection in cleaned if detection.label.lower() in NON_NUMERIC_PLAYER_STATE_LABELS]
    numeric_tokens = [detection for detection in cleaned if detection.label in NUMERIC_PLAYER_STATE_LABELS and detection.center[1] >= lower_band_y]
    ordered = sorted([*state_markers, *numeric_tokens], key=lambda detection: (detection.bbox.x1, detection.bbox.y1))

    warnings: list[str] = []
    errors: list[str] = []
    tokens = [PlayerToken(label=det.label, confidence=det.confidence, bbox=det.bbox.to_dict(), x_sort_key=det.bbox.x1) for det in ordered]
    labels = [det.label.lower() for det in ordered]
    is_fold = "fold" in labels
    is_all_in = "all-in" in labels
    numeric_parts = [det.label for det in ordered if det.label in NUMERIC_PLAYER_STATE_LABELS]
    stack_text_raw = "".join(numeric_parts)
    stack_bb = None

    if not ordered:
        warnings.append("No player-state tokens found")
    if is_fold and is_all_in:
        warnings.append("Conflicting player-state tokens: fold and all-in")
    if stack_text_raw:
        if stack_text_raw.count(".") > 1:
            errors.append(f"Invalid numeric stack sequence: {stack_text_raw}")
        elif not re.fullmatch(r"\d+(\.\d+)?", stack_text_raw):
            errors.append(f"Unsupported numeric stack sequence: {stack_text_raw}")
        else:
            stack_bb = float(stack_text_raw)
    if is_fold and stack_text_raw:
        warnings.append("Fold token detected together with numeric stack")
    if is_all_in and stack_text_raw:
        warnings.append("all-in token detected together with numeric stack")

    player_state = PlayerState(
        position=position,
        is_fold=is_fold,
        is_all_in=is_all_in,
        is_active=not is_fold,
        stack_text_raw=stack_text_raw,
        stack_bb=stack_bb,
        tokens=tokens,
        warnings=warnings,
        errors=errors,
    )
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, meta={"player_state": player_state.to_dict(), "filtered_labels": [det.label for det in ordered]})


def validate_table_amount_regions(detections: List[Detection], iou_threshold: float, center_threshold: float) -> ValidationResult:
    region_dets = [det for det in detections if det.label in TABLE_AMOUNT_REGION_LABELS]
    chips = [det for det in region_dets if det.label == "Chips"]
    singles: list[Detection] = []
    warnings: list[str] = []
    errors: list[str] = []
    for label in ("SB", "BB", "TotalPot"):
        current = dedupe_detections([det for det in region_dets if det.label == label], iou_threshold, center_threshold)
        if len(current) > 1:
            warnings.append(f"Multiple {label} detections; using best one")
        if current:
            singles.append(current[0])
    chips = dedupe_detections(chips, max(0.85, iou_threshold + 0.2), max(10.0, center_threshold * 0.5))
    cleaned = singles + chips
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, meta={"cleaned": cleaned, "chips": chips})


def _dedupe_numeric_token_detections(detections: Sequence[Detection], iou_threshold: float, center_threshold: float) -> List[Detection]:
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if det.label not in TABLE_AMOUNT_DIGIT_LABELS:
            continue
        duplicate = False
        for best in kept:
            if not _same_token_slot(det, best, iou_threshold, center_threshold):
                continue
            if det.label != best.label:
                duplicate = True
                break
            if det.label == best.label and (iou(det.bbox, best.bbox) >= 0.92 or center_distance(det, best) <= 3.0):
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def parse_numeric_tokens(detections: Sequence[Detection], iou_threshold: float, center_threshold: float) -> ValidationResult:
    cleaned = _dedupe_numeric_token_detections(detections, iou_threshold, center_threshold)
    ordered = sorted(cleaned, key=lambda det: (det.bbox.x1, det.bbox.y1))
    warnings: list[str] = []
    errors: list[str] = []
    raw_text = "".join(det.label for det in ordered)
    amount = None
    if raw_text:
        if raw_text.count(".") > 1:
            errors.append(f"Invalid numeric sequence: {raw_text}")
        elif not re.fullmatch(r"\d+(\.\d+)?", raw_text):
            errors.append(f"Unsupported numeric sequence: {raw_text}")
        else:
            amount = float(raw_text)
    else:
        warnings.append("No numeric tokens found")
    return ValidationResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        meta={
            "raw_text": raw_text,
            "amount": amount,
            "digit_tokens": [
                {"label": det.label, "confidence": det.confidence, "bbox": det.bbox.to_dict()}
                for det in ordered
            ],
            "cleaned": ordered,
        },
    )
