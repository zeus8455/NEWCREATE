from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _approx_eq(a: float, b: float, eps: float = EPS) -> bool:
    return abs(a - b) <= eps


def _engine_action_for(semantic_action: str) -> str:
    if semantic_action in {"limp", "call"}:
        return "call"
    if semantic_action == "check":
        return "check"
    return "raise"

LEGACY_ACTION_BY_SEMANTIC = {
    "limp": "LIMP",
    "open_raise": "OPEN",
    "iso_raise": "OPEN",
    "call": "CALL",
    "3bet": "RAISE",
    "4bet": "RAISE",
    "cold_4bet": "RAISE",
    "5bet_jam": "RAISE",
    "check": "CHECK",
    "bet": "BET",
    "raise": "RAISE",
    "fold": "FOLD",
}


def _legacy_action_name(semantic_action: str, engine_action: Optional[str] = None) -> str:
    semantic = str(semantic_action or "").lower()
    if semantic in LEGACY_ACTION_BY_SEMANTIC:
        return LEGACY_ACTION_BY_SEMANTIC[semantic]
    engine = str(engine_action or "").upper()
    return engine or semantic.upper()


def _format_amount_for_display(amount: Any) -> str:
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return str(amount)
    if abs(value - round(value)) <= EPS:
        return f"{value:.1f}"
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def _decorate_legacy_action_fields(event: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(event)
    legacy_action = _legacy_action_name(str(out.get("semantic_action") or ""), str(out.get("engine_action") or ""))
    out.setdefault("action", legacy_action)
    amount = out.get("final_contribution_street_bb")
    if amount is None:
        amount = out.get("final_contribution_bb")
    if amount is None:
        amount = out.get("amount_bb")
    if amount is None or legacy_action in {"CHECK", "FOLD"}:
        out.setdefault("action_display", legacy_action)
    else:
        out.setdefault("action_display", f"{legacy_action} {_format_amount_for_display(amount)}")
    return out


def _legacy_last_action_display(event: Dict[str, Any]) -> str:
    decorated = _decorate_legacy_action_fields(event)
    return str(decorated.get("action_display") or decorated.get("action") or "")


def _final_aggression_label(raise_level: int) -> Optional[str]:
    if raise_level <= 0:
        return None
    if raise_level == 1:
        return "open_raise"
    if raise_level == 2:
        return "3bet"
    if raise_level == 3:
        return "4bet"
    return "5bet_or_more"


def _player_is_folded(player_states: Dict[str, Dict[str, object]], position: str) -> bool:
    return bool((player_states or {}).get(position, {}).get("is_fold", False))


def _forced_preflop_blinds(player_count: int, occupied_positions: Iterable[str]) -> Dict[str, float]:
    positions = set(occupied_positions)
    if player_count == 2 or ({"BTN", "BB"} <= positions and "SB" not in positions):
        forced: Dict[str, float] = {}
        if "BTN" in positions:
            forced["BTN"] = 0.5
        if "BB" in positions:
            forced["BB"] = 1.0
        return forced

    forced = {}
    if "SB" in positions:
        forced["SB"] = 0.5
    if "BB" in positions:
        forced["BB"] = 1.0
    return forced


def get_street_actor_order(
    player_count: int,
    street: str,
    occupied_positions: List[str],
    player_states: Dict[str, Dict[str, object]],
    contributions: Optional[Dict[str, float]] = None,
) -> List[str]:
    ring = [pos for pos in CANONICAL_RING.get(player_count, []) if pos in occupied_positions]
    if not ring:
        return []

    if street == "preflop":
        ordered = [pos for pos in PREFLOP_ORDER.get(player_count, ring) if pos in ring]
        # CRITICAL INVARIANT:
        # Preflop reconstruction is historical, not purely live-state based.
        # A position that is folded in the current frame but still has money
        # committed on the table must remain eligible for actor order and
        # action reconstruction. Otherwise lines like
        # open -> flat -> squeeze/3bet -> opener folds collapse into a false
        # shorter history because the earlier contributor disappears too early.
        return [
            pos
            for pos in ordered
            if (not _player_is_folded(player_states, pos)) or _safe_float((contributions or {}).get(pos), 0.0) > EPS
        ]

    after_btn_preference = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
    start_pos = next((pos for pos in after_btn_preference if pos in ring), ring[0])
    start_idx = ring.index(start_pos)
    ordered = ring[start_idx:] + ring[:start_idx]
    return [pos for pos in ordered if not _player_is_folded(player_states, pos)]


def _derive_node_type_preview(
    *,
    hero_position: Optional[str],
    player_count: int,
    limpers: list[str],
    opener_pos: Optional[str],
    three_bettor_pos: Optional[str],
    four_bettor_pos: Optional[str],
    callers_after_open: int,
    action_history: list[dict],
) -> Optional[str]:
    hero_pos = str(hero_position or "")
    if not hero_pos:
        return None

    if player_count == 2 and limpers and limpers[0] == "BTN" and hero_pos == "BB" and not opener_pos:
        return "bb_vs_sb_limp"

    if four_bettor_pos:
        if hero_pos == three_bettor_pos:
            return "threebettor_vs_4bet"
        if hero_pos == four_bettor_pos and hero_pos not in {opener_pos, three_bettor_pos}:
            return "cold_4bet"

    if three_bettor_pos:
        if hero_pos == opener_pos:
            return "opener_vs_3bet"
        if opener_pos and hero_pos not in {opener_pos, three_bettor_pos}:
            return "facing_3bet"

    if opener_pos:
        if limpers and hero_pos in limpers:
            return "limper_vs_iso"
        if callers_after_open > 0:
            return "facing_open_callers"
        if hero_pos != opener_pos:
            return "facing_open"
        return "open_raise"

    limp_actions = [step for step in action_history if step.get("semantic_action") == "limp"]
    if limp_actions:
        if len(limp_actions) == 1:
            return "open_limp_first_in"
        return "over_limp_after_1_limper" if len(limp_actions) == 2 else "over_limp_after_2plus_limpers"

    return "unopened"


def _build_action_step(
    *,
    order: int,
    position: str,
    street: str,
    final_contribution_bb: float,
    semantic_action: str,
    current_price_to_call: float,
    raise_level: int,
    frame_id: Optional[str],
    timestamp: Optional[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "order": order,
        "position": position,
        "pos": position,
        "street": street,
        "final_contribution_bb": round(final_contribution_bb, 4),
        "amount_bb": round(final_contribution_bb, 4),
        "semantic_action": semantic_action,
        "engine_action": _engine_action_for(semantic_action),
        "raise_level_after_action": raise_level,
        "current_price_to_call_after_action": round(current_price_to_call, 4),
        "frame_id": frame_id,
        "timestamp": timestamp,
    }
    if extra:
        payload.update(extra)
    return _decorate_legacy_action_fields(payload)


def _infer_preflop_actions(previous_hand: Any, analysis: Any, settings: Any) -> Dict[str, Any]:
    amount_state = getattr(analysis, "amount_state", None) or getattr(analysis, "amount_normalization", None) or {}
    contributions = {
        str(pos): _safe_float(amount, 0.0)
        for pos, amount in (amount_state.get("final_contribution_bb_by_pos") or {}).items()
    }
    street_contribs = {
        str(pos): _safe_float(amount, 0.0)
        for pos, amount in (amount_state.get("final_contribution_street_bb_by_pos") or {}).items()
    }
    player_count = int(getattr(analysis, "player_count", 0) or 0)
    occupied_positions = list(getattr(analysis, "occupied_positions", []) or [])
    hero_position = getattr(analysis, "hero_position", None)
    player_states = dict(getattr(analysis, "player_states", {}) or {})
    forced = _forced_preflop_blinds(player_count, occupied_positions)

    actor_order = get_street_actor_order(
        player_count,
        "preflop",
        occupied_positions,
        player_states,
        contributions=contributions,
    )

    current_price_to_call = 1.0 if "BB" in occupied_positions else 0.0
    raise_level = 0
    limpers: List[str] = []
    opener_pos: Optional[str] = None
    three_bettor_pos: Optional[str] = None
    four_bettor_pos: Optional[str] = None
    callers_after_open = 0
    action_history: List[Dict[str, Any]] = []
    last_actions_by_position: Dict[str, str] = {}
    acted_positions: List[str] = []
    skipped_positions: List[str] = []

    for position in actor_order:
        amount = contributions.get(position, 0.0)
        forced_amount = forced.get(position, 0.0)
        is_fold = _player_is_folded(player_states, position)

        if amount < forced_amount:
            amount = forced_amount

        if amount <= forced_amount + EPS:
            if position == "BB" and raise_level == 0 and limpers and _approx_eq(amount, current_price_to_call):
                step = _build_action_step(
                    order=len(action_history) + 1,
                    position=position,
                    street="preflop",
                    final_contribution_bb=amount,
                    semantic_action="check",
                    current_price_to_call=current_price_to_call,
                    raise_level=raise_level,
                    frame_id=getattr(analysis, "frame_id", None),
                    timestamp=getattr(analysis, "timestamp", None),
                    extra={"reason": "bb_closes_action_vs_limp"},
                )
                action_history.append(step)
                last_actions_by_position[position] = _legacy_last_action_display(step)
                acted_positions.append(position)
            else:
                skipped_positions.append(position)
            continue

        if raise_level == 0:
            if _approx_eq(amount, current_price_to_call):
                semantic_action = "limp"
                limpers.append(position)
                extra: Dict[str, Any] = {}
                if position == "SB":
                    extra["limp_subtype"] = "sb_complete"
                elif position == "BTN" and player_count == 2:
                    extra["limp_subtype"] = "sb_complete_first_in"
                elif len(limpers) == 1:
                    extra["limp_subtype"] = "open_limp"
                else:
                    extra["limp_subtype"] = "over_limp"
            elif amount > current_price_to_call + EPS:
                semantic_action = "open_raise" if not limpers else "iso_raise"
                opener_pos = position
                raise_level = 1
                current_price_to_call = amount
                callers_after_open = 0
                extra = {}
                if limpers:
                    extra["open_family"] = "open_raise"
                    extra["isolates_limpers"] = list(limpers)
            else:
                skipped_positions.append(position)
                continue
        elif raise_level == 1:
            if _approx_eq(amount, current_price_to_call):
                semantic_action = "call"
                callers_after_open += 1
                extra = {"call_vs": "open_raise"}
            elif amount > current_price_to_call + EPS:
                semantic_action = "3bet"
                three_bettor_pos = position
                raise_level = 2
                current_price_to_call = amount
                extra = {}
            else:
                skipped_positions.append(position)
                continue
        elif raise_level == 2:
            if _approx_eq(amount, current_price_to_call):
                semantic_action = "call"
                extra = {"call_vs": "3bet"}
            elif amount > current_price_to_call + EPS:
                semantic_action = "4bet"
                if position not in {opener_pos, three_bettor_pos}:
                    extra = {"spot_family": "cold_4bet"}
                else:
                    extra = {}
                four_bettor_pos = position
                raise_level = 3
                current_price_to_call = amount
            else:
                skipped_positions.append(position)
                continue
        else:
            if _approx_eq(amount, current_price_to_call):
                semantic_action = "call"
                extra = {"call_vs": "4bet"}
            elif amount > current_price_to_call + EPS:
                semantic_action = "5bet_jam"
                raise_level += 1
                current_price_to_call = amount
                extra = {}
            else:
                skipped_positions.append(position)
                continue

        if is_fold and amount <= forced_amount + EPS:
            skipped_positions.append(position)
            continue

        step = _build_action_step(
            order=len(action_history) + 1,
            position=position,
            street="preflop",
            final_contribution_bb=amount,
            semantic_action=semantic_action,
            current_price_to_call=current_price_to_call,
            raise_level=raise_level,
            frame_id=getattr(analysis, "frame_id", None),
            timestamp=getattr(analysis, "timestamp", None),
            extra=extra,
        )
        action_history.append(step)
        last_actions_by_position[position] = _legacy_last_action_display(step)
        acted_positions.append(position)

    terminal_actions: List[Dict[str, Any]] = []
    final_aggression = _final_aggression_label(raise_level)
    if final_aggression is not None and current_price_to_call > EPS:
        for position in actor_order:
            if not _player_is_folded(player_states, position):
                continue
            amount = contributions.get(position, 0.0)
            forced_amount = forced.get(position, 0.0)
            if amount <= forced_amount + EPS:
                continue
            if amount >= current_price_to_call - EPS:
                continue
            prior_action = next((step for step in reversed(action_history) if step.get("position") == position), None)
            fold_event = _decorate_legacy_action_fields({
                "order": len(action_history) + len(terminal_actions) + 1,
                "position": position,
                "pos": position,
                "street": "preflop",
                "semantic_action": "fold",
                "engine_action": "fold",
                "amount_bb": None,
                "final_contribution_bb": round(amount, 4),
                "facing_price_to_call_bb": round(current_price_to_call, 4),
                "facing_aggression": final_aggression,
                "inferred_terminal_action": True,
                "reason": "folded_with_residual_preflop_contribution_below_final_price",
                "prior_semantic_action": prior_action.get("semantic_action") if prior_action else None,
                "frame_id": getattr(analysis, "frame_id", None),
                "timestamp": getattr(analysis, "timestamp", None),
            })
            terminal_actions.append(fold_event)
            last_actions_by_position[position] = _legacy_last_action_display(fold_event)

    node_type_preview = _derive_node_type_preview(
        hero_position=hero_position,
        player_count=player_count,
        limpers=limpers,
        opener_pos=opener_pos,
        three_bettor_pos=three_bettor_pos,
        four_bettor_pos=four_bettor_pos,
        callers_after_open=callers_after_open,
        action_history=action_history,
    )

    return {
        "street": "preflop",
        "source_mode": amount_state.get("source_mode", "forced_blinds_plus_visible_chips"),
        "actor_order": actor_order,
        "street_commitments": {pos: round(contributions.get(pos, 0.0), 4) for pos in actor_order},
        "current_highest_commitment": round(current_price_to_call, 4),
        "last_aggressor_position": four_bettor_pos or three_bettor_pos or opener_pos,
        "acted_positions": list(acted_positions),
        "last_actions_by_position": dict(last_actions_by_position),
        "actions_this_frame": [_decorate_legacy_action_fields(item) for item in [*action_history, *terminal_actions]],
        "action_history": [_decorate_legacy_action_fields(item) for item in [*action_history, *terminal_actions]],
        "historical_terminal_actions": [_decorate_legacy_action_fields(item) for item in terminal_actions],
        "final_contribution_bb_by_pos": {
            pos: round(value, 4) for pos, value in contributions.items()
        },
        "final_contribution_street_bb_by_pos": {
            pos: round(value, 4) for pos, value in street_contribs.items()
        },
        "semantic_action": ([*action_history, *terminal_actions][-1]["semantic_action"] if [*action_history, *terminal_actions] else None),
        "engine_action": ([*action_history, *terminal_actions][-1].get("engine_action") if [*action_history, *terminal_actions] else None),
        "raise_level_after_action": (action_history[-1]["raise_level_after_action"] if action_history else 0),
        "current_price_to_call_after_action": (action_history[-1]["current_price_to_call_after_action"] if action_history else round(current_price_to_call, 4)),
        "limpers": list(limpers),
        "limpers_count": len(limpers),
        "opener_pos": opener_pos,
        "three_bettor_pos": three_bettor_pos,
        "four_bettor_pos": four_bettor_pos,
        "callers_after_open": callers_after_open,
        "callers": callers_after_open,
        "node_type_preview": node_type_preview,
        "hero_position": hero_position,
        "hero_context_preview": {
            "hero_pos": hero_position,
            "node_type": node_type_preview,
            "opener_pos": opener_pos,
            "three_bettor_pos": three_bettor_pos,
            "four_bettor_pos": four_bettor_pos,
            "limpers": len(limpers),
            "callers": callers_after_open,
        },
        "skipped_positions": skipped_positions,
    }


def _infer_postflop_actions(previous_hand: Any, analysis: Any, settings: Any) -> Dict[str, Any]:
    street = getattr(analysis, "street", "preflop")
    previous_action_state = dict(getattr(previous_hand, "action_state", {}) or {}) if previous_hand else {}
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

    amount_state = getattr(analysis, "amount_state", None) or getattr(analysis, "amount_normalization", None) or {}
    current_bets = {
        str(pos): _safe_float(amount, 0.0)
        for pos, amount in (amount_state.get("final_contribution_street_bb_by_pos") or {}).items()
    }
    if not current_bets:
        current_bets = {
            pos: _safe_float(payload.get("amount_bb"), 0.0)
            for pos, payload in (getattr(analysis, "table_amount_state", {}) or {}).get("bets_by_position", {}).items()
            if isinstance(payload, dict) and payload.get("amount_bb") is not None
        }

    actions: List[dict] = []
    actor_order = get_street_actor_order(
        int(getattr(analysis, "player_count", 0) or 0),
        street,
        list(getattr(analysis, "occupied_positions", []) or []),
        dict(getattr(analysis, "player_states", {}) or {}),
        contributions=current_bets,
    )
    previous_player_states = getattr(previous_hand, "player_states", {}) if previous_hand else {}
    any_positive_bets = any(amount > 0 for amount in current_bets.values())
    allow_check_inference = bool(getattr(settings, "infer_checks_without_explicit_evidence", False))

    for position in actor_order:
        player_state = dict((getattr(analysis, "player_states", {}) or {}).get(position, {}))
        prev_player_state = dict((previous_player_states or {}).get(position, {}))
        prev_fold = bool(prev_player_state.get("is_fold", False))
        is_fold = bool(player_state.get("is_fold", False))
        if is_fold and not prev_fold:
            action = {
                "position": position,
                "pos": position,
                "street": street,
                "semantic_action": "fold",
                "engine_action": "fold",
                "amount_bb": None,
                "frame_id": getattr(analysis, "frame_id", None),
                "timestamp": getattr(analysis, "timestamp", None),
            }
            action = _decorate_legacy_action_fields(action)
            actions.append(action)
            last_actions_by_position[position] = _legacy_last_action_display(action)
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
            action = {
                "position": position,
                "pos": position,
                "street": street,
                "semantic_action": semantic_action,
                "engine_action": "call" if semantic_action == "call" else "raise",
                "amount_bb": round(current_amount, 4),
                "final_contribution_street_bb": round(current_amount, 4),
                "frame_id": getattr(analysis, "frame_id", None),
                "timestamp": getattr(analysis, "timestamp", None),
            }
            action = _decorate_legacy_action_fields(action)
            actions.append(action)
            last_actions_by_position[position] = _legacy_last_action_display(action)
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
                "pos": position,
                "street": street,
                "semantic_action": "check",
                "engine_action": "check",
                "amount_bb": 0.0,
                "frame_id": getattr(analysis, "frame_id", None),
                "timestamp": getattr(analysis, "timestamp", None),
            }
            action = _decorate_legacy_action_fields(action)
            actions.append(action)
            last_actions_by_position[position] = _legacy_last_action_display(action)
            acted_positions.append(position)

    last_action = actions[-1] if actions else {}
    return {
        "street": street,
        "actor_order": actor_order,
        "street_commitments": {k: round(v, 4) for k, v in street_commitments.items()},
        "current_highest_commitment": round(current_highest, 4),
        "last_aggressor_position": last_aggressor,
        "acted_positions": list(acted_positions),
        "last_actions_by_position": dict(last_actions_by_position),
        "actions_this_frame": [_decorate_legacy_action_fields(item) for item in actions],
        "action_history": [_decorate_legacy_action_fields(item) for item in actions],
        "final_contribution_bb_by_pos": {
            str(pos): round(_safe_float(amount, 0.0), 4)
            for pos, amount in (amount_state.get("final_contribution_bb_by_pos") or {}).items()
        },
        "final_contribution_street_bb_by_pos": {pos: round(v, 4) for pos, v in current_bets.items()},
        "semantic_action": last_action.get("semantic_action"),
        "engine_action": last_action.get("engine_action"),
        "raise_level_after_action": None,
        "current_price_to_call_after_action": round(current_highest, 4),
        "opener_pos": None,
        "three_bettor_pos": None,
        "four_bettor_pos": None,
        "limpers": [],
        "callers_after_open": 0,
        "node_type_preview": None,
    }


def infer_actions(previous_hand: Any, analysis: Any, settings: Any) -> dict:
    street = str(getattr(analysis, "street", "preflop") or "preflop")
    if street == "preflop":
        return _infer_preflop_actions(previous_hand, analysis, settings)
    return _infer_postflop_actions(previous_hand, analysis, settings)
