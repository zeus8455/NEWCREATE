from __future__ import annotations

from typing import Any, Dict, Iterable, List


EPSILON = 1e-9


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unique_positions(values: Iterable[str] | None) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values or []:
        pos = str(value)
        if not pos or pos in seen:
            continue
        seen.add(pos)
        out.append(pos)
    return out


def _canonical_positions(
    player_count: int | None,
    occupied_positions: Iterable[str] | None,
    hero_position: str | None,
    player_states: Dict[str, Dict[str, Any]] | None,
) -> List[str]:
    ordered = _unique_positions(occupied_positions)
    if hero_position and hero_position not in ordered:
        ordered.append(str(hero_position))
    for position in (player_states or {}).keys():
        if position not in ordered:
            ordered.append(str(position))
    # keep only non-empty strings
    return [pos for pos in ordered if pos]


def _forced_preflop_blinds(player_count: int | None, positions: List[str]) -> Dict[str, float]:
    pos_set = set(positions)
    if int(player_count or 0) == 2 or ({"BTN", "BB"} <= pos_set and "SB" not in pos_set):
        forced: Dict[str, float] = {}
        if "BTN" in pos_set:
            forced["BTN"] = 0.5
        if "BB" in pos_set:
            forced["BB"] = 1.0
        return forced

    forced = {}
    if "SB" in pos_set:
        forced["SB"] = 0.5
    if "BB" in pos_set:
        forced["BB"] = 1.0
    return forced


def normalize_amount_contributions(
    table_amount_state: Dict[str, Any] | None,
    street: str,
    player_count: int | None,
    occupied_positions: Iterable[str] | None,
    hero_position: str | None,
    player_states: Dict[str, Dict[str, Any]] | None,
) -> Dict[str, Any]:
    """
    Canonical contribution normalizer used between table_amount_logic and action_inference.

    Output contract:
    - final_contribution_bb_by_pos
    - final_contribution_street_bb_by_pos

    For preflop:
    - starts from forced blind baselines (SB=0.5, BB=1.0)
    - for HU: BTN/SB=0.5, BB=1.0
    - then overlays visible Chips regions
    - posted_blinds are treated only as confirmation / diagnostics

    For postflop:
    - only visible Chips are treated as current-street contributions
    - posted_blinds are kept only in diagnostics
    """
    amount_state = table_amount_state if isinstance(table_amount_state, dict) else {}
    positions = _canonical_positions(player_count, occupied_positions, hero_position, player_states)

    final_total: Dict[str, float] = {pos: 0.0 for pos in positions}
    final_street: Dict[str, float] = {pos: 0.0 for pos in positions}
    visible_bets: Dict[str, float] = {}
    warnings: List[str] = []

    forced_blinds = _forced_preflop_blinds(player_count, positions) if street == "preflop" else {}
    if street == "preflop":
        for pos, amount in forced_blinds.items():
            final_total[pos] = amount
            final_street[pos] = amount

    posted_blinds = amount_state.get("posted_blinds", {}) if isinstance(amount_state, dict) else {}
    blind_diagnostics: Dict[str, Dict[str, Any]] = {}
    for blind_name, payload in (posted_blinds or {}).items():
        if not isinstance(payload, dict):
            continue
        detected_amount = _safe_float(payload.get("amount_bb"))
        matched_position = str(payload.get("matched_position") or blind_name)
        expected_amount = forced_blinds.get(matched_position) if street == "preflop" else None
        blind_diagnostics[str(blind_name)] = {
            "matched_position": matched_position,
            "detected_amount_bb": detected_amount,
            "expected_amount_bb": expected_amount,
            "used_as_source_of_truth": False,
            "status": "ignored_for_street" if street != "preflop" else "diagnostic_only",
            "warnings": list(payload.get("warnings", [])) if isinstance(payload.get("warnings", []), list) else [],
        }
        if street == "preflop" and detected_amount is not None and expected_amount is not None:
            if abs(detected_amount - expected_amount) > 0.26:
                warnings.append(
                    f"{blind_name}: detected blind {detected_amount:.3f}bb differs from forced baseline "
                    f"{expected_amount:.3f}bb"
                )

    bets_by_position = amount_state.get("bets_by_position", {}) if isinstance(amount_state, dict) else {}
    for position, payload in (bets_by_position or {}).items():
        if not isinstance(payload, dict):
            continue
        amount = _safe_float(payload.get("amount_bb"))
        if amount is None:
            continue
        pos = str(position)
        visible_bets[pos] = amount
        previous = final_street.get(pos, 0.0)
        if street == "preflop":
            merged = max(previous, amount)
            final_street[pos] = merged
            final_total[pos] = max(final_total.get(pos, 0.0), merged)
        else:
            final_street[pos] = amount
            final_total[pos] = amount

    source_mode = "forced_blinds_plus_visible_chips" if street == "preflop" else "visible_chips_only"
    unassigned = amount_state.get("unassigned_chips", []) if isinstance(amount_state, dict) else []

    return {
        "street": street,
        "player_count": int(player_count or 0),
        "occupied_positions": list(positions),
        "hero_position": hero_position,
        "source_mode": source_mode,
        "forced_blinds_by_position": dict(forced_blinds),
        "visible_bets_by_position": dict(visible_bets),
        "final_contribution_bb_by_pos": {pos: round(amount, 4) for pos, amount in final_total.items()},
        "final_contribution_street_bb_by_pos": {pos: round(amount, 4) for pos, amount in final_street.items()},
        "posted_blinds_snapshot": {
            str(name): dict(payload) for name, payload in (posted_blinds or {}).items() if isinstance(payload, dict)
        },
        "blind_diagnostics": blind_diagnostics,
        "unassigned_chips_count": len(unassigned) if isinstance(unassigned, list) else 0,
        "warnings": list(warnings),
    }
