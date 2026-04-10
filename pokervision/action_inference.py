from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .preflop_reconstruction import (
    build_preflop_frame_observation,
    reconstruct_preflop_from_frame,
    reconcile_preflop_with_hand,
)

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

    aggressive_actions = {"open_raise", "iso_raise", "3bet", "4bet", "5bet_jam", "raise"}
    last_aggressive = None
    max_raise_level = 0
    for step in action_history:
        semantic = str(step.get("semantic_action") or "")
        if semantic in aggressive_actions:
            last_aggressive = str(step.get("position") or step.get("pos") or "") or last_aggressive
        max_raise_level = max(max_raise_level, int(step.get("raise_level_after_action") or 0))

    if player_count == 2 and limpers and limpers[0] == "BTN" and hero_pos == "BB" and not opener_pos:
        return "bb_vs_sb_limp"

    if max_raise_level >= 4 and four_bettor_pos:
        if hero_pos == four_bettor_pos and last_aggressive and last_aggressive != hero_pos:
            return "fourbettor_vs_5bet"
        if hero_pos == three_bettor_pos and last_aggressive and last_aggressive != hero_pos:
            return "threebettor_vs_4bet"

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




def _normalize_count_or_len(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _build_preflop_resolved_ledger(
    *,
    hero_position: Optional[str],
    actor_order: List[str],
    action_history: List[Dict[str, Any]],
    final_contribution_bb_by_pos: Dict[str, float],
    final_contribution_street_bb_by_pos: Dict[str, float],
    current_price_to_call: float,
    opener_pos: Optional[str],
    three_bettor_pos: Optional[str],
    four_bettor_pos: Optional[str],
    limpers: List[str],
    callers_after_open: int,
    node_type_preview: Optional[str],
    source_mode: str,
    skipped_positions: List[str],
    same_hand_identity: bool,
) -> Dict[str, Any]:
    resolved_history = [_decorate_legacy_action_fields(dict(item)) for item in list(action_history or [])]
    limper_positions = [str(pos) for pos in list(limpers or [])]
    callers_count = _normalize_count_or_len(callers_after_open)
    hero_context_preview = {
        "hero_pos": hero_position,
        "node_type": node_type_preview,
        "opener_pos": opener_pos,
        "three_bettor_pos": three_bettor_pos,
        "four_bettor_pos": four_bettor_pos,
        "limpers": len(limper_positions),
        "callers": callers_count,
        "resolved": True,
        "projection_source": "reconstructed_preflop",
    }
    return {
        "street": "preflop",
        "source_mode": source_mode,
        "hero_position": hero_position,
        "node_type": node_type_preview,
        "opener_pos": opener_pos,
        "three_bettor_pos": three_bettor_pos,
        "four_bettor_pos": four_bettor_pos,
        "limpers": limper_positions,
        "limpers_count": len(limper_positions),
        "callers": callers_count,
        "callers_after_open": callers_count,
        "action_history": list(resolved_history),
        "action_history_resolved": list(resolved_history),
        "actor_order": [str(pos) for pos in list(actor_order or [])],
        "current_price_to_call": round(_safe_float(current_price_to_call, 0.0), 4),
        "last_aggressor_position": four_bettor_pos or three_bettor_pos or opener_pos,
        "final_contribution_bb_by_pos": {
            str(pos): round(_safe_float(value, 0.0), 4)
            for pos, value in (final_contribution_bb_by_pos or {}).items()
        },
        "final_contribution_street_bb_by_pos": {
            str(pos): round(_safe_float(value, 0.0), 4)
            for pos, value in (final_contribution_street_bb_by_pos or {}).items()
        },
        "hero_context_preview": hero_context_preview,
        "skipped_positions": [str(pos) for pos in list(skipped_positions or [])],
        "same_hand_identity": bool(same_hand_identity),
        "contract_version": "preflop_resolved_v1",
    }

def _infer_preflop_actions(previous_hand: Any, analysis: Any, settings: Any) -> Dict[str, Any]:
    observation = build_preflop_frame_observation(
        analysis,
        settings=settings,
        safe_float=_safe_float,
        forced_preflop_blinds=_forced_preflop_blinds,
        get_street_actor_order=get_street_actor_order,
    )
    frame_result = reconstruct_preflop_from_frame(
        observation,
        previous_hand=previous_hand,
        analysis=analysis,
        settings=settings,
        approx_eq=_approx_eq,
        build_action_step=_build_action_step,
        decorate_legacy_action_fields=_decorate_legacy_action_fields,
        legacy_last_action_display=_legacy_last_action_display,
        final_aggression_label=_final_aggression_label,
        same_hand_identity=_same_hand_identity,
        player_is_folded=_player_is_folded,
        eps=EPS,
    )
    return reconcile_preflop_with_hand(
        previous_hand,
        frame_result,
        analysis=analysis,
        settings=settings,
        safe_float=_safe_float,
        derive_node_type_preview=_derive_node_type_preview,
        build_preflop_resolved_ledger=_build_preflop_resolved_ledger,
        build_action_step=_build_action_step,
    )

def _normalized_hero_cards_for_identity(cards: Any) -> tuple[str, ...]:
    if not cards:
        return tuple()
    return tuple(sorted(str(card) for card in cards if card))


def _same_hand_identity(previous_hand: Any, analysis: Any) -> bool:
    """Return True only when previous_hand belongs to the same logical hand.

    CRITICAL INVARIANT:
    Postflop action-state must never leak across different hand_id instances.
    infer_actions() runs before HandStateManager decides whether the new frame
    updates the current hand or opens a new one, so the only safe identity key
    available here is HERO cards. If we reuse previous postflop commitments when
    HERO cards changed, a new hand can inherit stale bets/highest commitment and
    misclassify first-bet spots as raises.
    """
    if previous_hand is None:
        return False
    prev_cards = _normalized_hero_cards_for_identity(getattr(previous_hand, "hero_cards", []) or [])
    curr_cards = _normalized_hero_cards_for_identity(getattr(analysis, "hero_cards", []) or [])
    return bool(prev_cards) and prev_cards == curr_cards


def _infer_postflop_actions(previous_hand: Any, analysis: Any, settings: Any) -> Dict[str, Any]:
    street = getattr(analysis, "street", "preflop")
    same_hand = _same_hand_identity(previous_hand, analysis)
    previous_action_state = dict(getattr(previous_hand, "action_state", {}) or {}) if same_hand else {}
    can_carry_street_state = same_hand and previous_action_state.get("street") == street
    if not can_carry_street_state:
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
    previous_player_states = getattr(previous_hand, "player_states", {}) if same_hand and previous_hand else {}
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
        "same_hand_identity": same_hand,
        "carried_previous_street_state": can_carry_street_state,
    }


def infer_actions(previous_hand: Any, analysis: Any, settings: Any) -> dict:
    street = str(getattr(analysis, "street", "preflop") or "preflop")
    if street == "preflop":
        return _infer_preflop_actions(previous_hand, analysis, settings)
    return _infer_postflop_actions(previous_hand, analysis, settings)
