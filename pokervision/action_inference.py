
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

CANONICAL_RING = {
    2: ["BTN", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["BTN", "SB", "BB", "CO"],
    5: ["BTN", "SB", "BB", "UTG", "CO"],
    6: ["BTN", "SB", "BB", "UTG", "MP", "CO"],
}

PREFLOP_ORDER = {
    2: ["BTN", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["CO", "BTN", "SB", "BB"],
    5: ["UTG", "CO", "BTN", "SB", "BB"],
    6: ["UTG", "MP", "CO", "BTN", "SB", "BB"],
}

EPS = 1e-9


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _approx_eq(a: float, b: float, eps: float = EPS) -> bool:
    return abs(a - b) <= eps


def _player_is_folded(player_states: Dict[str, Dict[str, object]], position: str) -> bool:
    return bool((player_states or {}).get(position, {}).get("is_fold", False))


def get_street_actor_order(
    player_count: int,
    street: str,
    occupied_positions: List[str],
    player_states: Dict[str, Dict[str, object]],
) -> List[str]:
    ring = [pos for pos in CANONICAL_RING.get(player_count, []) if pos in occupied_positions]
    if not ring:
        return []
    if street == "preflop":
        ordered = [pos for pos in PREFLOP_ORDER.get(player_count, ring) if pos in ring]
    else:
        after_btn_preference = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
        start_pos = next((pos for pos in after_btn_preference if pos in ring), ring[0])
        start_idx = ring.index(start_pos)
        ordered = ring[start_idx:] + ring[:start_idx]
    return [pos for pos in ordered if not _player_is_folded(player_states, pos)]


def _legacy_action_name_from_semantic(semantic_action: str) -> str:
    mapping = {
        "limp": "LIMP",
        "open_raise": "OPEN",
        "iso_raise": "RAISE",
        "call": "CALL",
        "check": "CHECK",
        "3bet": "RAISE",
        "4bet": "RAISE",
        "cold_4bet": "RAISE",
        "5bet_jam": "RAISE",
        "bet": "BET",
        "raise": "RAISE",
    }
    return mapping.get(semantic_action, semantic_action.upper())


def _engine_action_from_semantic(semantic_action: str) -> str:
    if semantic_action in {"limp", "call"}:
        return "call"
    if semantic_action == "check":
        return "check"
    return "raise"


def _display_action(semantic_action: str, amount: Optional[float]) -> str:
    label_map = {
        "limp": "LIMP",
        "open_raise": "OPEN",
        "iso_raise": "ISO",
        "call": "CALL",
        "check": "CHECK",
        "3bet": "3BET",
        "4bet": "4BET",
        "cold_4bet": "COLD 4BET",
        "5bet_jam": "5BET JAM",
        "bet": "BET",
        "raise": "RAISE",
    }
    label = label_map.get(semantic_action, semantic_action.upper())
    if amount is None or semantic_action == "check":
        return label
    return f"{label} {amount:.1f}"


def _current_preflop_contributions(analysis) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    amount_norm = dict(getattr(analysis, "amount_normalization", {}) or {})
    final_street = {
        str(pos): _safe_float(value)
        for pos, value in dict(amount_norm.get("final_contribution_street_bb_by_pos", {}) or {}).items()
    }
    forced = {
        str(pos): _safe_float(value)
        for pos, value in dict(amount_norm.get("forced_blinds_by_position", {}) or {}).items()
    }
    visible = {
        str(pos): _safe_float(value)
        for pos, value in dict(amount_norm.get("visible_bets_by_position", {}) or {}).items()
    }
    if not final_street:
        table_amount_state = dict(getattr(analysis, "table_amount_state", {}) or {})
        bets = dict(table_amount_state.get("bets_by_position", {}) or {})
        for pos in getattr(analysis, "occupied_positions", []) or []:
            if pos in bets and bets[pos].get("amount_bb") is not None:
                final_street[pos] = _safe_float(bets[pos].get("amount_bb"))
            else:
                final_street[pos] = 0.0
        if getattr(analysis, "street", "") == "preflop":
            blinds = dict(table_amount_state.get("posted_blinds", {}) or {})
            sb_payload = blinds.get("SB") or {}
            bb_payload = blinds.get("BB") or {}
            if int(getattr(analysis, "player_count", 0) or 0) == 2:
                if "BTN" in getattr(analysis, "occupied_positions", []):
                    forced["BTN"] = _safe_float(sb_payload.get("amount_bb"), 0.5) or 0.5
                if "BB" in getattr(analysis, "occupied_positions", []):
                    forced["BB"] = _safe_float(bb_payload.get("amount_bb"), 1.0) or 1.0
            else:
                if "SB" in getattr(analysis, "occupied_positions", []):
                    forced["SB"] = _safe_float(sb_payload.get("amount_bb"), 0.5) or 0.5
                if "BB" in getattr(analysis, "occupied_positions", []):
                    forced["BB"] = _safe_float(bb_payload.get("amount_bb"), 1.0) or 1.0
            for pos, amount in forced.items():
                final_street[pos] = max(final_street.get(pos, 0.0), amount)
            for pos in getattr(analysis, "occupied_positions", []) or []:
                visible[pos] = max(0.0, final_street.get(pos, 0.0) - forced.get(pos, 0.0))
    return final_street, forced, visible




def _normalized_hero_cards_from_hand(previous_hand) -> tuple[str, ...]:
    if not previous_hand:
        return tuple()
    return tuple(sorted(str(card) for card in getattr(previous_hand, "hero_cards", []) or []))


def _normalized_hero_cards_from_analysis(analysis) -> tuple[str, ...]:
    return tuple(sorted(str(card) for card in getattr(analysis, "hero_cards", []) or []))


def _can_reuse_previous_action_state(previous_hand, analysis, street: str) -> bool:
    if not previous_hand:
        return False
    previous_action_state = dict(getattr(previous_hand, "action_state", {}) or {})
    if not _can_reuse_previous_action_state(previous_hand, analysis, street):
        return False
    return _normalized_hero_cards_from_hand(previous_hand) == _normalized_hero_cards_from_analysis(analysis)

def _initial_preflop_state(previous_hand, analysis, actor_order: List[str]) -> dict:
    current_contribs, forced, visible = _current_preflop_contributions(analysis)
    occupied = list(getattr(analysis, "occupied_positions", []) or [])
    if _can_reuse_previous_action_state(previous_hand, analysis, "preflop"):
        previous_action_state = dict(previous_hand.action_state or {})
        state = {
            "street_commitments": {str(k): _safe_float(v) for k, v in dict(previous_action_state.get("street_commitments", {}) or {}).items()},
            "current_price_to_call": _safe_float(previous_action_state.get("current_price_to_call"), 1.0),
            "raise_level": int(previous_action_state.get("raise_level", 0) or 0),
            "limpers": list(previous_action_state.get("limpers", []) or []),
            "opener_pos": previous_action_state.get("opener_pos"),
            "three_bettor_pos": previous_action_state.get("three_bettor_pos"),
            "four_bettor_pos": previous_action_state.get("four_bettor_pos"),
            "callers_after_open": list(previous_action_state.get("callers_after_open", []) or []),
            "acted_positions": list(previous_action_state.get("acted_positions", []) or []),
            "last_actions_by_position": dict(previous_action_state.get("last_actions_by_position", {}) or {}),
            "last_aggressor_position": previous_action_state.get("last_aggressor_position"),
            "action_history": list(previous_action_state.get("action_history", []) or []),
        }
        for pos in occupied:
            state["street_commitments"].setdefault(pos, forced.get(pos, 0.0 if pos != "BB" else forced.get("BB", 0.0)))
        return state | {"current_contribs": current_contribs, "forced": forced, "visible": visible}
    base_commitments = {pos: _safe_float(forced.get(pos)) for pos in occupied}
    state = {
        "street_commitments": base_commitments,
        "current_price_to_call": 1.0 if int(getattr(analysis, "player_count", 0) or 0) >= 2 else 0.0,
        "raise_level": 0,
        "limpers": [],
        "opener_pos": None,
        "three_bettor_pos": None,
        "four_bettor_pos": None,
        "callers_after_open": [],
        "acted_positions": [],
        "last_actions_by_position": {},
        "last_aggressor_position": None,
        "action_history": [],
    }
    return state | {"current_contribs": current_contribs, "forced": forced, "visible": visible}


def _build_preflop_action_record(
    *,
    analysis,
    position: str,
    semantic_action: str,
    final_contribution_bb: float,
    current_price_to_call_before: float,
    current_price_to_call_after: float,
    raise_level_before: int,
    raise_level_after: int,
    opener_pos: Optional[str],
    three_bettor_pos: Optional[str],
    four_bettor_pos: Optional[str],
    limpers: List[str],
    callers_after_open: List[str],
    call_vs: Optional[str] = None,
    spot_family: Optional[str] = None,
    open_family: Optional[str] = None,
    limp_family: Optional[str] = None,
) -> dict:
    action_name = _legacy_action_name_from_semantic(semantic_action)
    payload = {
        "position": position,
        "street": "preflop",
        "action": action_name,
        "semantic_action": semantic_action,
        "engine_action": _engine_action_from_semantic(semantic_action),
        "amount_bb": final_contribution_bb,
        "final_contribution_bb": final_contribution_bb,
        "current_price_to_call_before": current_price_to_call_before,
        "current_price_to_call_after": current_price_to_call_after,
        "raise_level_before_action": raise_level_before,
        "raise_level_after_action": raise_level_after,
        "frame_id": analysis.frame_id,
        "timestamp": analysis.timestamp,
        "opener_pos": opener_pos,
        "three_bettor_pos": three_bettor_pos,
        "four_bettor_pos": four_bettor_pos,
        "limpers": len(limpers),
        "callers_after_open": len(callers_after_open),
    }
    if call_vs:
        payload["call_vs"] = call_vs
    if spot_family:
        payload["spot_family"] = spot_family
    if open_family:
        payload["open_family"] = open_family
    if limp_family:
        payload["limp_family"] = limp_family
    return payload


def _infer_preflop_actions(previous_hand, analysis, settings) -> dict:
    player_count = int(getattr(analysis, "player_count", 0) or 0)
    actor_order = get_street_actor_order(
        player_count,
        "preflop",
        list(getattr(analysis, "occupied_positions", []) or []),
        getattr(analysis, "player_states", {}) or {},
    )
    state = _initial_preflop_state(previous_hand, analysis, actor_order)
    street_commitments: Dict[str, float] = dict(state["street_commitments"])
    current_contribs: Dict[str, float] = dict(state["current_contribs"])
    forced: Dict[str, float] = dict(state["forced"])
    visible: Dict[str, float] = dict(state["visible"])
    current_price_to_call = _safe_float(state["current_price_to_call"], 1.0)
    raise_level = int(state["raise_level"])
    limpers: List[str] = list(state["limpers"])
    opener_pos = state["opener_pos"]
    three_bettor_pos = state["three_bettor_pos"]
    four_bettor_pos = state["four_bettor_pos"]
    callers_after_open: List[str] = list(state["callers_after_open"])
    acted_positions: List[str] = list(state["acted_positions"])
    last_actions_by_position: Dict[str, str] = dict(state["last_actions_by_position"])
    last_aggressor = state["last_aggressor_position"]
    action_history: List[dict] = list(state["action_history"])
    previous_player_states = getattr(previous_hand, "player_states", {}) if previous_hand else {}
    actions: List[dict] = []

    for position in actor_order:
        player_state = (getattr(analysis, "player_states", {}) or {}).get(position, {}) or {}
        prev_player_state = (previous_player_states or {}).get(position, {}) or {}
        prev_fold = bool(prev_player_state.get("is_fold", False))
        is_fold = bool(player_state.get("is_fold", False))
        if is_fold and not prev_fold:
            action = {
                "position": position,
                "street": "preflop",
                "action": "FOLD",
                "semantic_action": "fold",
                "engine_action": "fold",
                "amount_bb": None,
                "final_contribution_bb": street_commitments.get(position, 0.0),
                "frame_id": analysis.frame_id,
                "timestamp": analysis.timestamp,
            }
            actions.append(action)
            action_history.append(dict(action))
            last_actions_by_position[position] = "FOLD"
            if position not in acted_positions:
                acted_positions.append(position)
            continue

        current_amount = _safe_float(current_contribs.get(position))
        previous_amount = _safe_float(street_commitments.get(position))
        if current_amount <= previous_amount + EPS:
            continue

        before_price = current_price_to_call
        before_level = raise_level
        semantic_action = ""
        call_vs = None
        spot_family = None
        open_family = None
        limp_family = None

        if raise_level == 0:
            if _approx_eq(current_amount, 1.0):
                semantic_action = "limp"
                if position == "SB":
                    limp_family = "sb_complete"
                elif limpers:
                    limp_family = "over_limp"
                else:
                    limp_family = "open_limp"
                if position not in limpers:
                    limpers.append(position)
            elif current_amount > 1.0 + EPS:
                if limpers:
                    semantic_action = "iso_raise"
                else:
                    semantic_action = "open_raise"
                open_family = "open_raise"
                opener_pos = position
                raise_level = 1
                current_price_to_call = current_amount
                last_aggressor = position
        elif _approx_eq(current_amount, current_price_to_call):
            semantic_action = "call"
            if raise_level == 1:
                call_vs = "open_raise"
                if position != opener_pos and position not in callers_after_open:
                    callers_after_open.append(position)
            elif raise_level == 2:
                call_vs = "3bet"
            elif raise_level >= 3:
                call_vs = "4bet"
        elif current_amount > current_price_to_call + EPS:
            if raise_level == 1:
                semantic_action = "3bet"
                three_bettor_pos = position
                raise_level = 2
            elif raise_level == 2:
                semantic_action = "4bet"
                if position not in {opener_pos, three_bettor_pos}:
                    spot_family = "cold_4bet"
                four_bettor_pos = position
                raise_level = 3
            else:
                semantic_action = "5bet_jam"
                raise_level = max(raise_level + 1, 4)
            current_price_to_call = current_amount
            last_aggressor = position
        else:
            continue

        if not semantic_action:
            continue

        street_commitments[position] = current_amount
        if position not in acted_positions:
            acted_positions.append(position)
        action = _build_preflop_action_record(
            analysis=analysis,
            position=position,
            semantic_action=semantic_action if spot_family != "cold_4bet" else "4bet",
            final_contribution_bb=current_amount,
            current_price_to_call_before=before_price,
            current_price_to_call_after=current_price_to_call,
            raise_level_before=before_level,
            raise_level_after=raise_level,
            opener_pos=opener_pos,
            three_bettor_pos=three_bettor_pos,
            four_bettor_pos=four_bettor_pos,
            limpers=limpers,
            callers_after_open=callers_after_open,
            call_vs=call_vs,
            spot_family=spot_family,
            open_family=open_family,
            limp_family=limp_family,
        )
        # Preserve distinct spot family while keeping canonical semantic action.
        if semantic_action == "5bet_jam":
            action["is_all_in_like"] = True
        action["legacy_from_forced_blind"] = _approx_eq(current_amount, forced.get(position, -999.0)) and _approx_eq(current_amount, previous_amount)
        actions.append(action)
        action_history.append(dict(action))
        display_semantic = semantic_action if spot_family != "cold_4bet" else "cold_4bet"
        last_actions_by_position[position] = _display_action(display_semantic, current_amount)

    # Conservative optional check inference for completed blind-vs-limp spots only.
    allow_check = bool(getattr(settings, "infer_checks_without_explicit_evidence", False))
    if allow_check and raise_level == 0 and "BB" in actor_order and "BB" not in acted_positions and limpers:
        bb_current = _safe_float(current_contribs.get("BB"))
        bb_prev = _safe_float(street_commitments.get("BB"))
        if _approx_eq(bb_current, 1.0) and _approx_eq(bb_prev, 1.0):
            action = _build_preflop_action_record(
                analysis=analysis,
                position="BB",
                semantic_action="check",
                final_contribution_bb=bb_current,
                current_price_to_call_before=current_price_to_call,
                current_price_to_call_after=current_price_to_call,
                raise_level_before=raise_level,
                raise_level_after=raise_level,
                opener_pos=opener_pos,
                three_bettor_pos=three_bettor_pos,
                four_bettor_pos=four_bettor_pos,
                limpers=limpers,
                callers_after_open=callers_after_open,
            )
            actions.append(action)
            action_history.append(dict(action))
            last_actions_by_position["BB"] = "CHECK"
            acted_positions.append("BB")

    # Ensure all occupied positions exist in commitments
    for pos in getattr(analysis, "occupied_positions", []) or []:
        street_commitments[pos] = _safe_float(current_contribs.get(pos, street_commitments.get(pos, 0.0)))

    return {
        "street": "preflop",
        "actor_order": actor_order,
        "street_commitments": street_commitments,
        "current_highest_commitment": current_price_to_call,
        "current_price_to_call": current_price_to_call,
        "raise_level": raise_level,
        "limpers": limpers,
        "opener_pos": opener_pos,
        "three_bettor_pos": three_bettor_pos,
        "four_bettor_pos": four_bettor_pos,
        "callers_after_open": callers_after_open,
        "last_aggressor_position": last_aggressor,
        "acted_positions": acted_positions,
        "last_actions_by_position": last_actions_by_position,
        "actions_this_frame": actions,
        "action_history": action_history,
        "final_contribution_bb_by_pos": dict(current_contribs),
        "final_contribution_street_bb_by_pos": dict(current_contribs),
        "forced_blinds_by_position": dict(forced),
        "visible_bets_by_position": dict(visible),
        "amount_normalization": dict(getattr(analysis, "amount_normalization", {}) or {}),
    }


def _infer_non_preflop_actions(previous_hand, analysis, settings) -> dict:
    street = analysis.street
    previous_action_state = dict(previous_hand.action_state) if previous_hand else {}
    if not _can_reuse_previous_action_state(previous_hand, analysis, street):
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
                "semantic_action": "fold",
                "engine_action": "fold",
                "amount_bb": None,
                "final_contribution_bb": street_commitments.get(position, 0.0),
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

        if current_amount > previous_amount + EPS:
            if current_highest <= 0.0:
                semantic_action = "bet"
            else:
                semantic_action = "call" if _approx_eq(current_amount, current_highest) else "raise"
            action_name = _legacy_action_name_from_semantic(semantic_action)
            action = {
                "position": position,
                "street": street,
                "action": action_name,
                "semantic_action": semantic_action,
                "engine_action": _engine_action_from_semantic(semantic_action),
                "amount_bb": current_amount,
                "final_contribution_bb": current_amount,
                "frame_id": analysis.frame_id,
                "timestamp": analysis.timestamp,
            }
            actions.append(action)
            last_actions_by_position[position] = _display_action(semantic_action, current_amount)
            street_commitments[position] = current_amount
            if semantic_action in {"bet", "raise"}:
                current_highest = max(current_highest, current_amount)
                last_aggressor = position
            if position not in acted_positions:
                acted_positions.append(position)
            continue

        if (
            allow_check_inference
            and not any_positive_bets
            and current_highest <= 0.0
            and position not in acted_positions
            and not is_fold
        ):
            action = {
                "position": position,
                "street": street,
                "action": "CHECK",
                "semantic_action": "check",
                "engine_action": "check",
                "amount_bb": 0.0,
                "final_contribution_bb": 0.0,
                "frame_id": analysis.frame_id,
                "timestamp": analysis.timestamp,
            }
            actions.append(action)
            last_actions_by_position[position] = "CHECK"
            acted_positions.append(position)

    return {
        "street": street,
        "actor_order": actor_order,
        "street_commitments": street_commitments,
        "current_highest_commitment": current_highest,
        "last_aggressor_position": last_aggressor,
        "acted_positions": acted_positions,
        "last_actions_by_position": last_actions_by_position,
        "actions_this_frame": actions,
        "amount_normalization": dict(getattr(analysis, "amount_normalization", {}) or {}),
    }


def infer_actions(previous_hand, analysis, settings) -> dict:
    street = getattr(analysis, "street", "")
    if str(street).lower() == "preflop":
        return _infer_preflop_actions(previous_hand, analysis, settings)
    return _infer_non_preflop_actions(previous_hand, analysis, settings)
