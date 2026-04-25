from __future__ import annotations

from typing import Any, Dict, List, Optional

CANONICAL_RING = {
    2: ["BTN", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["BTN", "SB", "BB", "CO"],
    5: ["BTN", "SB", "BB", "UTG", "CO"],
    6: ["BTN", "SB", "BB", "UTG", "MP", "CO"],
}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_forced_blinds(
    street: str,
    player_count: int,
    occupied_positions: List[str],
) -> Dict[str, float]:
    if street != "preflop":
        return {}
    occupied = set(occupied_positions)
    forced: Dict[str, float] = {}
    if player_count == 2:
        if "BTN" in occupied:
            forced["BTN"] = 0.5
        if "BB" in occupied:
            forced["BB"] = 1.0
        return forced
    if "SB" in occupied:
        forced["SB"] = 0.5
    if "BB" in occupied:
        forced["BB"] = 1.0
    return forced


def _payload_warnings(payload: Any) -> List[str]:
    if not isinstance(payload, dict):
        return []
    warnings = payload.get("warnings", [])
    if not isinstance(warnings, list):
        return []
    return [str(item) for item in warnings]


def _is_live(player_states: Dict[str, Dict[str, Any]], pos: str) -> bool:
    return not bool((player_states.get(pos, {}) if isinstance(player_states, dict) else {}).get("is_fold", False))


def _infer_sb_complete_from_pot_or_ambiguous_bb_marker(
    *,
    street: str,
    ordered_positions: List[str],
    player_states: Dict[str, Dict[str, Any]],
    posted_blinds: Dict[str, Any],
    final_total: Dict[str, float],
    final_street: Dict[str, float],
    visible_bets: Dict[str, float],
    total_pot_amount: Optional[float],
    warnings: List[str],
) -> None:
    if street != "preflop":
        return
    if "SB" not in ordered_positions or "BB" not in ordered_positions:
        return
    if not (_is_live(player_states, "SB") and _is_live(player_states, "BB")):
        return

    sb_amount = float(final_total.get("SB", 0.0))
    bb_amount = float(final_total.get("BB", 0.0))
    if sb_amount >= 1.0 - 1e-6:
        return
    if bb_amount < 1.0 - 1e-6:
        return

    inferred = False
    reason = ""

    bb_marker = posted_blinds.get("BB") if isinstance(posted_blinds, dict) else None
    if isinstance(bb_marker, dict):
        marker_amount = _safe_float(bb_marker.get("amount_bb"))
        nearest_position = str(bb_marker.get("nearest_position") or "")
        matched_position = str(bb_marker.get("matched_position") or "")
        logical_position = str(bb_marker.get("logical_position") or "")
        marker_warnings = _payload_warnings(bb_marker)
        if (
            marker_amount is not None
            and abs(marker_amount - 1.0) <= 0.25
            and nearest_position == "SB"
            and (matched_position == "BB" or logical_position == "BB")
            and any("re-anchored from nearest SB" in item for item in marker_warnings)
        ):
            inferred = True
            reason = "ambiguous_bb_marker_nearest_sb"

    # Fallback for frames where the duplicate BB marker was already filtered out:
    # total pot 2bb + only SB/BB live + no visible open raise means SB completed to 1bb.
    if not inferred and total_pot_amount is not None:
        live_positions = [pos for pos in ordered_positions if _is_live(player_states, pos)]
        folded_non_blinds = [pos for pos in ordered_positions if pos not in {"SB", "BB"} and not _is_live(player_states, pos)]
        if (
            set(live_positions) == {"SB", "BB"}
            and folded_non_blinds
            and 1.75 <= float(total_pot_amount) <= 2.25
            and abs(bb_amount - 1.0) <= 0.25
        ):
            inferred = True
            reason = "total_pot_2bb_sb_vs_bb_complete"

    if not inferred:
        return

    final_total["SB"] = 1.0
    final_street["SB"] = 1.0
    visible_bets["SB"] = 1.0
    warnings.append(f"SB complete inferred for BB-vs-SB limp spot: {reason}")


def normalize_amount_contributions(
    table_amount_state: Dict[str, Any],
    street: str,
    player_count: int,
    occupied_positions: List[str],
    hero_position: Optional[str],
    player_states: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    state = table_amount_state if isinstance(table_amount_state, dict) else {}
    posted_blinds = state.get("posted_blinds", {}) if isinstance(state.get("posted_blinds"), dict) else {}
    bets_by_position = state.get("bets_by_position", {}) if isinstance(state.get("bets_by_position"), dict) else {}

    ordered_positions = [
        pos
        for pos in CANONICAL_RING.get(int(player_count or 0), [])
        if pos in list(occupied_positions or [])
    ]
    if not ordered_positions:
        ordered_positions = list(occupied_positions or [])

    final_total: Dict[str, float] = {pos: 0.0 for pos in ordered_positions}
    final_street: Dict[str, float] = {pos: 0.0 for pos in ordered_positions}
    forced_blinds = _build_forced_blinds(street, int(player_count or 0), list(occupied_positions or []))
    visible_bets: Dict[str, float] = {}
    blind_diagnostics: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []

    for pos, amount in forced_blinds.items():
        if pos in final_total:
            final_total[pos] = amount
            final_street[pos] = amount

    for blind_label, payload in posted_blinds.items():
        if not isinstance(payload, dict):
            continue
        matched_position = str(payload.get("matched_position") or "")
        amount_bb = _safe_float(payload.get("amount_bb"))
        blind_diagnostics[blind_label] = {
            "matched_position": matched_position,
            "amount_bb": amount_bb,
            "warnings": _payload_warnings(payload),
            "source": payload.get("source", blind_label),
            "nearest_position": payload.get("nearest_position"),
            "logical_position": payload.get("logical_position"),
            "confirmed": bool(
                matched_position
                and matched_position in forced_blinds
                and amount_bb is not None
                and abs(forced_blinds[matched_position] - amount_bb) <= 1e-6
            ),
        }
        if matched_position and matched_position in forced_blinds and amount_bb is not None:
            expected = forced_blinds[matched_position]
            if abs(expected - amount_bb) > 1e-6:
                warnings.append(
                    f"{blind_label}: detected blind {amount_bb:.3f}bb differs from forced baseline {expected:.3f}bb"
                )

    for pos, payload in bets_by_position.items():
        if not isinstance(payload, dict):
            continue
        amount_bb = _safe_float(payload.get("amount_bb"))
        if amount_bb is None:
            continue
        visible_bets[pos] = amount_bb
        final_street[pos] = amount_bb
        if street == "preflop":
            final_total[pos] = max(final_total.get(pos, 0.0), amount_bb)
        else:
            final_total[pos] = amount_bb

    total_pot_payload = state.get("total_pot", {}) if isinstance(state.get("total_pot"), dict) else {}
    total_pot_amount = _safe_float(total_pot_payload.get("amount_bb"))

    _infer_sb_complete_from_pot_or_ambiguous_bb_marker(
        street=street,
        ordered_positions=ordered_positions,
        player_states=player_states if isinstance(player_states, dict) else {},
        posted_blinds=posted_blinds,
        final_total=final_total,
        final_street=final_street,
        visible_bets=visible_bets,
        total_pot_amount=total_pot_amount,
        warnings=warnings,
    )

    active_positions = [
        pos
        for pos in ordered_positions
        if not bool((player_states.get(pos, {}) if isinstance(player_states, dict) else {}).get("is_fold", False))
    ]

    return {
        "street": street,
        "player_count": int(player_count or 0),
        "occupied_positions": list(ordered_positions),
        "active_positions": active_positions,
        "hero_position": hero_position,
        "forced_blinds_by_position": forced_blinds,
        "visible_bets_by_position": visible_bets,
        "final_contribution_bb_by_pos": final_total,
        "final_contribution_street_bb_by_pos": final_street,
        "blind_diagnostics": blind_diagnostics,
        "total_pot_bb": total_pot_amount,
        "warnings": warnings,
    }
