
from __future__ import annotations

from .action_inference import CANONICAL_RING
from .models import (
    HandState,
    RenderState,
    _compute_engine_status,
    _compute_node_type,
    _compute_recommended_action,
    _compute_recommended_amount_to,
    _compute_recommended_size_pct,
    derive_amount_state,
    derive_analysis_panel,
    derive_reconstructed_postflop,
    derive_reconstructed_preflop,
)


def _build_display_seat_order(hand: HandState) -> list[str]:
    available = [pos for pos in CANONICAL_RING.get(hand.player_count, []) if pos in hand.occupied_positions]
    if not available:
        return list(hand.occupied_positions)
    hero_position = hand.hero_position
    if hero_position in available:
        idx = available.index(hero_position)
        return available[idx:] + available[:idx]
    return available


def _clear_legacy_solver_annotation(action_annotations: dict) -> dict:
    """Remove deprecated duplicated solver payload from action annotations."""
    if not isinstance(action_annotations, dict):
        return {}
    cleaned = dict(action_annotations)
    cleaned.pop("solver_bridge", None)
    return cleaned


def build_render_state(hand: HandState, source_frame_id: str, source_timestamp: str) -> RenderState:
    players: dict[str, dict] = {}
    action_state = hand.action_state or {}
    last_actions = dict(action_state.get("last_actions_by_position", {}))
    bet_map = (hand.table_amount_state or {}).get("bets_by_position", {}) if isinstance(hand.table_amount_state, dict) else {}
    seat_order = _build_display_seat_order(hand)

    for position in hand.occupied_positions:
        player_state = hand.player_states.get(position, {})
        is_fold = bool(player_state.get("is_fold", False))
        is_hero = position == hand.hero_position
        bet_payload = bet_map.get(position, {}) if isinstance(bet_map, dict) else {}
        players[position] = {
            "position": position,
            "occupied": True,
            "is_hero": is_hero,
            "is_button": position == "BTN",
            "is_fold": is_fold,
            "is_all_in": bool(player_state.get("is_all_in", False)),
            "is_active": bool(player_state.get("is_active", not is_fold)),
            "stack_bb": player_state.get("stack_bb"),
            "stack_text_raw": player_state.get("stack_text_raw", ""),
            "state_warnings": list(player_state.get("warnings", [])),
            "cards_visible": is_hero and not is_fold,
            "show_card_backs": (not is_hero) and (not is_fold),
            "current_bet_bb": bet_payload.get("amount_bb", 0.0),
            "current_bet_raw": bet_payload.get("raw_text", ""),
            "last_action": last_actions.get(position),
        }

    street = str(hand.street_state.get("current_street", "preflop"))
    freshness = "live"
    warnings: list[str] = []

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
        warnings.append(str(hand.conflict_state))

    reconstructed_preflop = hand.reconstructed_preflop or derive_reconstructed_preflop(
        hand.action_state,
        hero_position=hand.hero_position,
    )
    reconstructed_postflop = hand.reconstructed_postflop or derive_reconstructed_postflop(
        street=street,
        action_state=hand.action_state,
        advisor_input=hand.advisor_input,
    )

    analysis_panel = derive_analysis_panel(
        street=street,
        hero_position=hand.hero_position,
        occupied_positions=hand.occupied_positions,
        action_state=hand.action_state,
        advisor_input=hand.advisor_input,
        solver_input=hand.solver_input,
        solver_output=hand.solver_output,
        engine_result=hand.engine_result,
        hero_decision_debug=hand.hero_decision_debug,
        solver_status=hand.solver_status,
        solver_warnings=hand.solver_warnings,
        solver_errors=hand.solver_errors,
        solver_result_reused=hand.solver_result_reused,
        solver_reuse_reason=hand.solver_reuse_reason,
        solver_fingerprint=hand.solver_fingerprint,
    )

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
        amount_state=derive_amount_state(hand.table_amount_state, hand.amount_normalization),
        action_annotations=_clear_legacy_solver_annotation({
            "actions_log": list(hand.actions_log[-12:]),
            "last_actions_by_position": last_actions,
        }),
        reconstructed_preflop=reconstructed_preflop,
        reconstructed_postflop=reconstructed_postflop,
        advisor_input=dict(hand.advisor_input),
        solver_input=dict(hand.solver_input),
        solver_output=dict(hand.solver_output),
        engine_result=dict(hand.engine_result),
        solver_context=dict(hand.solver_context),
        solver_status=hand.solver_status,
        solver_warnings=list(hand.solver_warnings),
        solver_errors=list(hand.solver_errors),
        hero_decision_debug=dict(hand.hero_decision_debug),
        solver_fingerprint=hand.solver_fingerprint,
        solver_result_reused=bool(hand.solver_result_reused),
        solver_reuse_reason=hand.solver_reuse_reason,
        solver_run_frame_id=hand.solver_run_frame_id,
        solver_run_timestamp=hand.solver_run_timestamp,
        recommended_action=_compute_recommended_action(hand.engine_result, hand.solver_output),
        recommended_amount_to=_compute_recommended_amount_to(hand.engine_result, hand.solver_output),
        recommended_size_pct=_compute_recommended_size_pct(hand.engine_result, hand.solver_output),
        node_type=_compute_node_type(hand.action_state, hand.advisor_input),
        engine_status=_compute_engine_status(hand.engine_result, hand.solver_status),
        analysis_panel=analysis_panel,
    )
