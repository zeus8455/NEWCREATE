from __future__ import annotations

from .action_inference import CANONICAL_RING
from .models import HandState, RenderState


def _build_display_seat_order(hand: HandState) -> list[str]:
    available = [pos for pos in CANONICAL_RING.get(hand.player_count, []) if pos in hand.occupied_positions]
    if not available:
        return list(hand.occupied_positions)
    hero_position = hand.hero_position
    if hero_position in available:
        idx = available.index(hero_position)
        return available[idx:] + available[:idx]
    return available


def _build_legacy_solver_annotation(hand: HandState) -> dict:
    """Backward-compatible solver annotation.

    CRITICAL INVARIANT:
    The official solver state now lives in top-level normalized fields on
    HandState/RenderState. This legacy payload is kept only so older UI/debug
    code does not break while the project transitions away from ad-hoc dicts.
    """
    result = {
        "status": hand.solver_status or "not_run",
        "result": {
            "type": str(hand.hero_decision_debug.get("type") or ""),
            "raw_repr": str(hand.hero_decision_debug.get("raw_repr") or ""),
        },
    }
    if hand.solver_warnings:
        result["warnings"] = list(hand.solver_warnings)
    if hand.solver_errors:
        result["errors"] = list(hand.solver_errors)
    return result


def _derive_solver_ui_fields(hand: HandState) -> dict:
    engine_result = hand.engine_result or {}
    advisor_input = hand.advisor_input or {}
    solver_input = hand.solver_input or {}
    solver_context = hand.solver_context or {}
    action_raw = engine_result.get("engine_action")
    recommended_action = str(action_raw).upper() if action_raw else None
    amount_to = engine_result.get("amount_to")
    size_pct = engine_result.get("size_pct")
    node_type = (
        str(advisor_input.get("node_type") or "").strip()
        or str(solver_input.get("node_type") or "").strip()
        or str(solver_context.get("node_type") or "").strip()
        or str((hand.action_state or {}).get("node_type_preview") or "").strip()
    )
    engine_status = str(engine_result.get("status") or hand.solver_status or "not_run")
    return {
        "recommended_action": recommended_action,
        "recommended_amount_to": amount_to,
        "recommended_size_pct": size_pct,
        "node_type": node_type,
        "engine_status": engine_status,
    }


def build_render_state(hand: HandState, source_frame_id: str, source_timestamp: str) -> RenderState:
    players = {}
    action_state = hand.action_state or {}
    last_actions = dict(action_state.get("last_actions_by_position", {}))
    bet_map = (hand.table_amount_state or {}).get("bets_by_position", {}) if isinstance(hand.table_amount_state, dict) else {}
    amount_norm = hand.amount_normalization or {}
    normalized_street = (
        amount_norm.get("final_contribution_street_bb_by_pos", {})
        if isinstance(amount_norm, dict)
        else {}
    )
    seat_order = _build_display_seat_order(hand)

    for position in hand.occupied_positions:
        player_state = hand.player_states.get(position, {})
        is_fold = player_state.get("is_fold", False)
        is_hero = position == hand.hero_position
        bet_payload = bet_map.get(position, {}) if isinstance(bet_map, dict) else {}
        normalized_bet = normalized_street.get(position) if isinstance(normalized_street, dict) else None
        players[position] = {
            "position": position,
            "occupied": True,
            "is_hero": is_hero,
            "is_button": position == "BTN",
            "is_fold": is_fold,
            "is_all_in": player_state.get("is_all_in", False),
            "is_active": player_state.get("is_active", not is_fold),
            "stack_bb": player_state.get("stack_bb"),
            "stack_text_raw": player_state.get("stack_text_raw", ""),
            "state_warnings": list(player_state.get("warnings", [])),
            "cards_visible": is_hero and not is_fold,
            "show_card_backs": (not is_hero) and (not is_fold),
            "current_bet_bb": bet_payload.get("amount_bb") if bet_payload.get("amount_bb") is not None else normalized_bet,
            "current_bet_raw": bet_payload.get("raw_text", ""),
            "last_action": last_actions.get(position),
        }

    street = hand.street_state.get("current_street", "preflop")
    freshness = "live"
    warnings = []
    if hand.status == "stale":
        freshness = "stale"
        warnings.append("State is stale")
    elif hand.status == "closed":
        freshness = "closed"
        warnings.append("State is closed")
    elif hand.status == "error":
        freshness = "error"
        warnings.append("State is error")
    if hand.conflict_state:
        warnings.append(hand.conflict_state)

    solver_summary = _build_legacy_solver_annotation(hand)
    solver_ui = _derive_solver_ui_fields(hand)

    return RenderState(
        hand_id=hand.hand_id,
        player_count=hand.player_count,
        table_format=hand.table_format,
        street=street,
        hero_position=hand.hero_position,
        hero_cards=list(hand.hero_cards),
        board_cards=list(hand.board_cards),
        players=players,
        status="ok" if hand.status in {"active", "stale"} else hand.status,
        warnings=warnings,
        freshness=freshness,
        source_frame_id=source_frame_id,
        source_timestamp=source_timestamp,
        updated_at=hand.updated_at,
        seat_order=seat_order,
        table_amount_state=dict(hand.table_amount_state),
        amount_normalization=dict(hand.amount_normalization),
        action_annotations={
            "actions_log": list(hand.actions_log[-12:]),
            "last_actions_by_position": last_actions,
            "solver_bridge": solver_summary,
        },
        advisor_input=dict(hand.advisor_input),
        solver_input=dict(hand.solver_input),
        solver_output=dict(hand.solver_output),
        engine_result=dict(hand.engine_result),
        solver_context=dict(hand.solver_context),
        solver_status=hand.solver_status,
        solver_warnings=list(hand.solver_warnings),
        solver_errors=list(hand.solver_errors),
        hero_decision_debug=dict(hand.hero_decision_debug),
        recommended_action=solver_ui["recommended_action"],
        recommended_amount_to=solver_ui["recommended_amount_to"],
        recommended_size_pct=solver_ui["recommended_size_pct"],
        node_type=solver_ui["node_type"],
        engine_status=solver_ui["engine_status"],
    )
