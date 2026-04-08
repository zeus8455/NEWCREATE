from __future__ import annotations

from typing import Dict, List

CANONICAL_RING = {
    2: ["BTN", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["BTN", "SB", "BB", "CO"],
    5: ["BTN", "SB", "BB", "UTG", "CO"],
    6: ["BTN", "SB", "BB", "UTG", "MP", "CO"],
}


def get_street_actor_order(player_count: int, street: str, occupied_positions: List[str], player_states: Dict[str, Dict[str, object]]) -> List[str]:
    ring = [pos for pos in CANONICAL_RING.get(player_count, []) if pos in occupied_positions]
    if not ring:
        return []
    if street == "preflop":
        if player_count == 2:
            start_pos = "BTN"
        else:
            start_pos = "UTG" if "UTG" in ring else ("CO" if "CO" in ring else ring[0])
    else:
        after_btn_preference = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
        start_pos = next((pos for pos in after_btn_preference if pos in ring), ring[0])
    start_idx = ring.index(start_pos)
    ordered = ring[start_idx:] + ring[:start_idx]
    return [pos for pos in ordered if not player_states.get(pos, {}).get("is_fold", False)]


def infer_actions(previous_hand, analysis, settings) -> dict:
    street = analysis.street
    previous_action_state = dict(previous_hand.action_state) if previous_hand else {}
    if previous_action_state.get("street") != street:
        street_commitments: Dict[str, float] = {}
        current_highest = 0.0
        acted_positions: List[str] = []
        last_aggressor = None
        last_actions_by_position: Dict[str, str] = {}
    else:
        street_commitments = {k: float(v) for k, v in previous_action_state.get("street_commitments", {}).items()}
        current_highest = float(previous_action_state.get("current_highest_commitment", 0.0))
        acted_positions = list(previous_action_state.get("acted_positions", []))
        last_aggressor = previous_action_state.get("last_aggressor_position")
        last_actions_by_position = dict(previous_action_state.get("last_actions_by_position", {}))

    actions: List[dict] = []
    actor_order = get_street_actor_order(
        int(analysis.player_count or 0),
        street,
        list(analysis.occupied_positions),
        analysis.player_states,
    )

    current_bets = {
        pos: float(payload.get("amount_bb"))
        for pos, payload in (analysis.table_amount_state or {}).get("bets_by_position", {}).items()
        if payload.get("amount_bb") is not None
    }
    previous_player_states = previous_hand.player_states if previous_hand else {}
    any_positive_bets = any(amount > 0 for amount in current_bets.values())
    allow_check_inference = bool(getattr(settings, "infer_checks_without_explicit_evidence", False))

    for position in actor_order:
        player_state = analysis.player_states.get(position, {})
        prev_player_state = previous_player_states.get(position, {})
        prev_fold = bool(prev_player_state.get("is_fold", False))
        is_fold = bool(player_state.get("is_fold", False))
        if is_fold and not prev_fold:
            action = {
                "position": position,
                "street": street,
                "action": "FOLD",
                "amount_bb": None,
                "frame_id": analysis.frame_id,
                "timestamp": analysis.timestamp,
            }
            actions.append(action)
            last_actions_by_position[position] = "FOLD"
            if position not in acted_positions:
                acted_positions.append(position)
            continue

        current_amount = current_bets.get(position, 0.0)
        previous_amount = street_commitments.get(position, 0.0)

        if current_amount > previous_amount + 1e-9:
            if street == "preflop":
                if current_highest <= 0.0:
                    action_name = "LIMP" if abs(current_amount - 1.0) <= 1e-9 else "OPEN"
                else:
                    action_name = "CALL" if abs(current_amount - current_highest) <= 1e-9 else "RAISE"
            else:
                if current_highest <= 0.0:
                    action_name = "BET"
                else:
                    action_name = "CALL" if abs(current_amount - current_highest) <= 1e-9 else "RAISE"
            action = {
                "position": position,
                "street": street,
                "action": action_name,
                "amount_bb": current_amount,
                "frame_id": analysis.frame_id,
                "timestamp": analysis.timestamp,
            }
            actions.append(action)
            last_actions_by_position[position] = f"{action_name} {current_amount:.1f}" if current_amount is not None else action_name
            street_commitments[position] = current_amount
            if action_name in {"OPEN", "BET", "RAISE", "LIMP"}:
                current_highest = max(current_highest, current_amount)
                if action_name != "LIMP":
                    last_aggressor = position
            if position not in acted_positions:
                acted_positions.append(position)
            continue

        # Conservative policy:
        # do not invent CHECK actions from a single silent frame by default,
        # because OCR may simply have missed a bet amount in that frame.
        # This avoids false sequences like CHECK/CHECK -> BET 2.5 on the same street.
        if (
            allow_check_inference
            and street != "preflop"
            and not any_positive_bets
            and current_highest <= 0.0
            and position not in acted_positions
            and not is_fold
        ):
            action = {
                "position": position,
                "street": street,
                "action": "CHECK",
                "amount_bb": 0.0,
                "frame_id": analysis.frame_id,
                "timestamp": analysis.timestamp,
            }
            actions.append(action)
            last_actions_by_position[position] = "CHECK"
            acted_positions.append(position)

    action_state = {
        "street": street,
        "actor_order": actor_order,
        "street_commitments": street_commitments,
        "current_highest_commitment": current_highest,
        "last_aggressor_position": last_aggressor,
        "acted_positions": acted_positions,
        "last_actions_by_position": last_actions_by_position,
        "actions_this_frame": actions,
    }
    return action_state
