from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

SafeFloat = Callable[[Any, float], float]
ApproxEq = Callable[[float, float, float], bool]
BuildActionStep = Callable[..., Dict[str, Any]]
DecorateLegacyActionFields = Callable[[Dict[str, Any]], Dict[str, Any]]
LegacyLastActionDisplay = Callable[[Dict[str, Any]], str]
FinalAggressionLabel = Callable[[int], Optional[str]]
DeriveNodeTypePreview = Callable[..., Optional[str]]
BuildResolvedLedger = Callable[..., Dict[str, Any]]
SameHandIdentity = Callable[[Any, Any], bool]
ForcedPreflopBlinds = Callable[[int, List[str]], Dict[str, float]]
GetStreetActorOrder = Callable[..., List[str]]
PlayerIsFolded = Callable[[Dict[str, Dict[str, object]], str], bool]

FRAME_OBSERVATION_CONTRACT_VERSION = "preflop_frame_observation_v2"
HAND_RECONCILIATION_CONTRACT_VERSION = "preflop_hand_reconciliation_v2"
PREFLOP_PROJECTION_CONTRACT_VERSION = "preflop_projection_v1"
FRAME_ONLY_RECONCILIATION_MODE = "frame_local_only"
COMMITMENT_GROWTH_RECONCILIATION_MODE = "commitment_growth_reconciled"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clone_action_list(actions: Any) -> List[Dict[str, Any]]:
    return [dict(item) for item in list(actions or []) if isinstance(item, dict)]


def _round_map(values: Dict[str, float]) -> Dict[str, float]:
    return {str(pos): round(float(amount), 4) for pos, amount in dict(values or {}).items()}


def _clone_string_list(values: Iterable[Any]) -> List[str]:
    return [str(item) for item in list(values or []) if str(item)]


def _dedupe_preserve_order(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in values or []:
        value = str(item)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _normalize_count(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _action_position(action: Dict[str, Any]) -> str:
    return str(action.get("position") or action.get("pos") or "")


def _action_semantic(action: Dict[str, Any]) -> str:
    return str(action.get("semantic_action") or "")


def _legacy_action_identity(action: Dict[str, Any]) -> Tuple[str, str, float, str, str]:
    return (
        _action_position(action),
        _action_semantic(action),
        round(float(action.get("final_contribution_bb") or action.get("amount_bb") or 0.0), 4),
        str(action.get("frame_id") or ""),
        str(action.get("timestamp") or ""),
    )


def _normalize_action_orders(actions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for index, action in enumerate(list(actions or []), start=1):
        payload = dict(action)
        payload["order"] = index
        normalized.append(payload)
    return normalized


def _extract_previous_action_state(previous_hand: Any) -> Dict[str, Any]:
    if previous_hand is None:
        return {}
    try:
        return dict(getattr(previous_hand, "action_state", {}) or {})
    except Exception:
        return {}


def _extract_previous_resolved_preflop(previous_hand: Any) -> Dict[str, Any]:
    if previous_hand is None:
        return {}
    candidates = []
    reconstructed = getattr(previous_hand, "reconstructed_preflop", None)
    if reconstructed:
        candidates.append(dict(reconstructed))
    action_state = _extract_previous_action_state(previous_hand)
    if action_state.get("reconstructed_preflop"):
        candidates.append(dict(action_state.get("reconstructed_preflop") or {}))
    if action_state:
        candidates.append(action_state)
    for candidate in candidates:
        if candidate.get("action_history_resolved") or candidate.get("action_history"):
            return candidate
    return candidates[0] if candidates else {}


def _extract_previous_commitments(previous_resolved: Dict[str, Any]) -> Dict[str, float]:
    if not previous_resolved:
        return {}
    values = (
        previous_resolved.get("resolved_commitments_by_pos")
        or previous_resolved.get("final_contribution_bb_by_pos")
        or {}
    )
    out: Dict[str, float] = {}
    for pos, amount in dict(values).items():
        try:
            out[str(pos)] = float(amount)
        except (TypeError, ValueError):
            continue
    return out


def _derive_raise_level_from_actions(actions: Sequence[Dict[str, Any]]) -> int:
    max_raise_level = 0
    for action in actions or []:
        try:
            max_raise_level = max(max_raise_level, int(action.get("raise_level_after_action") or 0))
        except (TypeError, ValueError):
            continue
    return max_raise_level


def _derive_state_from_actions(actions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    limpers: List[str] = []
    opener_pos: Optional[str] = None
    three_bettor_pos: Optional[str] = None
    four_bettor_pos: Optional[str] = None
    callers_after_open = 0
    current_price_to_call = 0.0
    raise_level = 0
    last_aggressor_position: Optional[str] = None

    for action in actions or []:
        semantic = _action_semantic(action)
        position = _action_position(action)
        amount = float(action.get("final_contribution_bb") or action.get("amount_bb") or 0.0)
        if semantic == "limp":
            if position and position not in limpers:
                limpers.append(position)
            current_price_to_call = max(current_price_to_call, amount)
        elif semantic in {"open_raise", "iso_raise"}:
            opener_pos = position or opener_pos
            raise_level = max(raise_level, 1)
            current_price_to_call = max(current_price_to_call, amount)
            last_aggressor_position = position or last_aggressor_position
        elif semantic == "3bet":
            three_bettor_pos = position or three_bettor_pos
            raise_level = max(raise_level, 2)
            current_price_to_call = max(current_price_to_call, amount)
            last_aggressor_position = position or last_aggressor_position
        elif semantic == "4bet":
            four_bettor_pos = position or four_bettor_pos
            raise_level = max(raise_level, 3)
            current_price_to_call = max(current_price_to_call, amount)
            last_aggressor_position = position or last_aggressor_position
        elif semantic == "5bet_jam":
            raise_level = max(raise_level, 4)
            current_price_to_call = max(current_price_to_call, amount)
            last_aggressor_position = position or last_aggressor_position
        elif semantic == "call":
            call_vs = str(action.get("call_vs") or "")
            if call_vs == "open_raise":
                callers_after_open += 1
            current_price_to_call = max(current_price_to_call, amount)
        elif semantic in {"check", "fold"}:
            pass

    if current_price_to_call <= 0.0:
        current_price_to_call = float(actions[-1].get("current_price_to_call_after_action") or 0.0) if actions else 0.0

    return {
        "limpers": list(limpers),
        "opener_pos": opener_pos,
        "three_bettor_pos": three_bettor_pos,
        "four_bettor_pos": four_bettor_pos,
        "callers_after_open": callers_after_open,
        "current_price_to_call": current_price_to_call,
        "raise_level": max(raise_level, _derive_raise_level_from_actions(actions)),
        "last_aggressor_position": last_aggressor_position,
    }


def _compute_positions_closed_to_action(
    actor_order: Sequence[str],
    *,
    current_price_to_call: float,
    commitments: Dict[str, float],
    player_states: Dict[str, Dict[str, object]],
    last_aggressor_position: Optional[str],
    forced_blinds: Optional[Dict[str, float]] = None,
    eps: float = 1e-9,
) -> List[str]:
    forced = dict(forced_blinds or {})
    if not actor_order or not last_aggressor_position or last_aggressor_position not in actor_order:
        return []

    start_index = actor_order.index(last_aggressor_position)
    closed: List[str] = []
    for position in actor_order[start_index + 1 :]:
        committed = float(commitments.get(position, 0.0))
        forced_amount = float(forced.get(position, 0.0))
        is_fold = bool((player_states or {}).get(position, {}).get("is_fold", False))
        if is_fold:
            closed.append(position)
            continue
        if committed >= current_price_to_call - eps:
            closed.append(position)
            continue
        if committed <= forced_amount + eps:
            closed.append(position)
    return _dedupe_preserve_order(closed)


def _append_terminal_folds(
    *,
    resolved_actions: List[Dict[str, Any]],
    actor_order: Sequence[str],
    commitments: Dict[str, float],
    forced_blinds: Dict[str, float],
    player_states: Dict[str, Dict[str, object]],
    current_price_to_call: float,
    final_aggression_label: Optional[str],
    frame_id: Optional[str],
    timestamp: Optional[str],
    decorate_legacy_action_fields: Optional[DecorateLegacyActionFields],
) -> List[Dict[str, Any]]:
    terminal_actions: List[Dict[str, Any]] = []
    if final_aggression_label is None or current_price_to_call <= 0.0:
        return terminal_actions

    existing_fold_keys = {
        (
            _action_position(action),
            round(float(action.get("final_contribution_bb") or action.get("amount_bb") or 0.0), 4),
            str(action.get("semantic_action") or ""),
        )
        for action in resolved_actions
        if _action_semantic(action) == "fold"
    }

    for position in actor_order:
        is_fold = bool((player_states or {}).get(position, {}).get("is_fold", False))
        if not is_fold:
            continue
        committed = float(commitments.get(position, 0.0))
        forced_amount = float(forced_blinds.get(position, 0.0))
        if committed <= forced_amount + 1e-9:
            continue
        if committed >= current_price_to_call - 1e-9:
            continue
        key = (position, round(committed, 4), "fold")
        if key in existing_fold_keys:
            continue
        event: Dict[str, Any] = {
            "order": len(resolved_actions) + len(terminal_actions) + 1,
            "position": position,
            "pos": position,
            "street": "preflop",
            "semantic_action": "fold",
            "engine_action": "fold",
            "amount_bb": None,
            "final_contribution_bb": round(committed, 4),
            "facing_price_to_call_bb": round(current_price_to_call, 4),
            "facing_aggression": final_aggression_label,
            "inferred_terminal_action": True,
            "reason": "folded_with_residual_preflop_contribution_below_final_price",
            "frame_id": frame_id,
            "timestamp": timestamp,
        }
        if decorate_legacy_action_fields is not None:
            event = decorate_legacy_action_fields(event)
        terminal_actions.append(event)
        existing_fold_keys.add(key)

    return terminal_actions


def _build_projection_mapping(
    *,
    node_type_preview: Optional[str],
    opener_pos: Optional[str],
    three_bettor_pos: Optional[str],
    four_bettor_pos: Optional[str],
    last_aggressor_position: Optional[str],
) -> Dict[str, Optional[str]]:
    projection_node_type = node_type_preview
    advisor_node_type = node_type_preview
    advisor_four_bettor_pos = four_bettor_pos
    advisor_mapping_reason: Optional[str] = None

    if node_type_preview == "fourbettor_vs_5bet":
        advisor_node_type = "threebettor_vs_4bet"
        advisor_four_bettor_pos = last_aggressor_position or three_bettor_pos
        advisor_mapping_reason = "closest_supported_rereraise_defense_node"
    elif node_type_preview == "cold_4bet":
        advisor_node_type = "cold_4bet"
        advisor_four_bettor_pos = four_bettor_pos
    elif node_type_preview == "opener_vs_3bet":
        advisor_node_type = "opener_vs_3bet"
        advisor_four_bettor_pos = four_bettor_pos
    else:
        advisor_four_bettor_pos = four_bettor_pos

    return {
        "projection_node_type": projection_node_type,
        "advisor_node_type": advisor_node_type,
        "advisor_four_bettor_pos": advisor_four_bettor_pos,
        "advisor_mapping_reason": advisor_mapping_reason,
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def build_preflop_frame_observation(
    analysis: Any,
    *,
    settings: Any,
    safe_float: SafeFloat,
    forced_preflop_blinds: ForcedPreflopBlinds,
    get_street_actor_order: GetStreetActorOrder,
) -> Dict[str, Any]:
    """Build the canonical frame-local preflop observation.

    This layer must answer only one question:
    what is minimally and safely visible on the current final snapshot?
    """
    amount_state = getattr(analysis, "amount_state", None) or getattr(analysis, "amount_normalization", None) or {}
    contributions = {
        str(pos): safe_float(amount, 0.0)
        for pos, amount in dict(amount_state.get("final_contribution_bb_by_pos") or {}).items()
    }
    street_contribs = {
        str(pos): safe_float(amount, 0.0)
        for pos, amount in dict(amount_state.get("final_contribution_street_bb_by_pos") or {}).items()
    }
    player_count = int(getattr(analysis, "player_count", 0) or 0)
    occupied_positions = [str(pos) for pos in list(getattr(analysis, "occupied_positions", []) or [])]
    hero_position = getattr(analysis, "hero_position", None)
    player_states = dict(getattr(analysis, "player_states", {}) or {})
    forced = forced_preflop_blinds(player_count, occupied_positions)
    actor_order = get_street_actor_order(
        player_count,
        "preflop",
        occupied_positions,
        player_states,
        contributions=contributions,
    )
    street_commitments_current_frame = {
        str(pos): safe_float(street_contribs.get(pos, contributions.get(pos, 0.0)), 0.0)
        for pos in occupied_positions
    }
    return {
        "street": "preflop",
        "observation_contract_version": FRAME_OBSERVATION_CONTRACT_VERSION,
        "source_mode": amount_state.get("source_mode", "forced_blinds_plus_visible_chips"),
        "amount_state": amount_state,
        "contributions": contributions,
        "street_contribs": street_contribs,
        "street_commitments_current_frame": street_commitments_current_frame,
        "player_count": player_count,
        "occupied_positions": occupied_positions,
        "hero_position": hero_position,
        "player_states": player_states,
        "forced_blinds": forced,
        "actor_order": actor_order,
        "frame_id": getattr(analysis, "frame_id", None),
        "timestamp": getattr(analysis, "timestamp", None),
        "current_price_to_call_start": 1.0 if "BB" in occupied_positions else 0.0,
    }


def reconstruct_preflop_from_frame(
    observation: Dict[str, Any],
    *,
    previous_hand: Any,
    analysis: Any,
    settings: Any,
    approx_eq: ApproxEq,
    build_action_step: BuildActionStep,
    decorate_legacy_action_fields: DecorateLegacyActionFields,
    legacy_last_action_display: LegacyLastActionDisplay,
    final_aggression_label: FinalAggressionLabel,
    same_hand_identity: SameHandIdentity,
    player_is_folded: PlayerIsFolded,
    eps: float,
) -> Dict[str, Any]:
    """Run the single-pass frame-local reconstruction.

    This is the step-3.3 layer only. It may be incomplete and it must not claim
    to be the temporal truth of the whole hand.
    """
    contributions = dict(observation.get("contributions") or {})
    street_contribs = dict(observation.get("street_contribs") or {})
    street_commitments_current_frame = dict(observation.get("street_commitments_current_frame") or {})
    player_count = int(observation.get("player_count") or 0)
    hero_position = observation.get("hero_position")
    player_states = dict(observation.get("player_states") or {})
    forced = dict(observation.get("forced_blinds") or {})
    actor_order = list(observation.get("actor_order") or [])
    frame_id = observation.get("frame_id")
    timestamp = observation.get("timestamp")
    source_mode = str(observation.get("source_mode") or "forced_blinds_plus_visible_chips")
    current_price_to_call = float(observation.get("current_price_to_call_start") or 0.0)

    raise_level = 0
    limpers: List[str] = []
    opener_pos: Optional[str] = None
    three_bettor_pos: Optional[str] = None
    four_bettor_pos: Optional[str] = None
    callers_after_open = 0

    frame_local_actions: List[Dict[str, Any]] = []
    acted_positions: List[str] = []
    skipped_positions: List[str] = []
    unresolved_positions: List[str] = []
    last_actions_by_position: Dict[str, str] = {}

    for position in actor_order:
        amount = float(contributions.get(position, 0.0))
        forced_amount = float(forced.get(position, 0.0))
        is_fold = player_is_folded(player_states, position)

        if amount < forced_amount:
            amount = forced_amount

        if is_fold and amount <= forced_amount + eps:
            skipped_positions.append(position)
            continue

        if amount <= forced_amount + eps:
            if position == "BB" and raise_level == 0 and limpers and approx_eq(amount, current_price_to_call, eps):
                step = build_action_step(
                    order=len(frame_local_actions) + 1,
                    position=position,
                    street="preflop",
                    final_contribution_bb=amount,
                    semantic_action="check",
                    current_price_to_call=current_price_to_call,
                    raise_level=raise_level,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    extra={"reason": "bb_closes_action_vs_limp"},
                )
                frame_local_actions.append(step)
                acted_positions.append(position)
                last_actions_by_position[position] = legacy_last_action_display(step)
            else:
                skipped_positions.append(position)
            continue

        if raise_level == 0:
            if approx_eq(amount, current_price_to_call, eps):
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
            elif amount > current_price_to_call + eps:
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
                unresolved_positions.append(position)
                continue
        elif raise_level == 1:
            if approx_eq(amount, current_price_to_call, eps):
                semantic_action = "call"
                callers_after_open += 1
                extra = {"call_vs": "open_raise"}
            elif amount > current_price_to_call + eps:
                semantic_action = "3bet"
                three_bettor_pos = position
                raise_level = 2
                current_price_to_call = amount
                extra = {}
            else:
                unresolved_positions.append(position)
                continue
        elif raise_level == 2:
            if approx_eq(amount, current_price_to_call, eps):
                semantic_action = "call"
                extra = {"call_vs": "3bet"}
            elif amount > current_price_to_call + eps:
                semantic_action = "4bet"
                extra = {}
                if position not in {opener_pos, three_bettor_pos}:
                    extra["spot_family"] = "cold_4bet"
                four_bettor_pos = position
                raise_level = 3
                current_price_to_call = amount
            else:
                unresolved_positions.append(position)
                continue
        else:
            if approx_eq(amount, current_price_to_call, eps):
                semantic_action = "call"
                extra = {"call_vs": "4bet"}
            elif amount > current_price_to_call + eps:
                semantic_action = "5bet_jam"
                raise_level += 1
                current_price_to_call = amount
                extra = {}
            else:
                unresolved_positions.append(position)
                continue

        step = build_action_step(
            order=len(frame_local_actions) + 1,
            position=position,
            street="preflop",
            final_contribution_bb=amount,
            semantic_action=semantic_action,
            current_price_to_call=current_price_to_call,
            raise_level=raise_level,
            frame_id=frame_id,
            timestamp=timestamp,
            extra=extra,
        )
        frame_local_actions.append(step)
        acted_positions.append(position)
        last_actions_by_position[position] = legacy_last_action_display(step)

    frame_local_terminal_actions = _append_terminal_folds(
        resolved_actions=frame_local_actions,
        actor_order=actor_order,
        commitments=contributions,
        forced_blinds=forced,
        player_states=player_states,
        current_price_to_call=current_price_to_call,
        final_aggression_label=final_aggression_label(raise_level),
        frame_id=frame_id,
        timestamp=timestamp,
        decorate_legacy_action_fields=decorate_legacy_action_fields,
    )

    for action in frame_local_terminal_actions:
        frame_local_actions.append(action)
        position = _action_position(action)
        if position:
            last_actions_by_position[position] = legacy_last_action_display(action)

    frame_local_max_aggression = final_aggression_label(raise_level)

    return {
        "street": "preflop",
        "source_mode": source_mode,
        "player_count": player_count,
        "actor_order": list(actor_order),
        "street_commitments": {
            pos: round(float(street_commitments_current_frame.get(pos, contributions.get(pos, 0.0))), 4)
            for pos in actor_order
        },
        "current_highest_commitment": round(current_price_to_call, 4),
        "last_aggressor_position": four_bettor_pos or three_bettor_pos or opener_pos,
        "acted_positions": list(acted_positions),
        "last_actions_by_position": dict(last_actions_by_position),
        "actions_this_frame": _clone_action_list(frame_local_actions),
        "action_history": _clone_action_list(frame_local_actions),
        "action_history_resolved": _clone_action_list(frame_local_actions),
        "historical_terminal_actions": _clone_action_list(frame_local_terminal_actions),
        "final_contribution_bb_by_pos": _round_map(contributions),
        "final_contribution_street_bb_by_pos": _round_map(street_contribs),
        "semantic_action": (_action_semantic(frame_local_actions[-1]) if frame_local_actions else None),
        "engine_action": (frame_local_actions[-1].get("engine_action") if frame_local_actions else None),
        "raise_level_after_action": (int(frame_local_actions[-1].get("raise_level_after_action") or 0) if frame_local_actions else 0),
        "current_price_to_call_after_action": round(current_price_to_call, 4),
        "limpers": list(limpers),
        "limpers_count": len(limpers),
        "opener_pos": opener_pos,
        "three_bettor_pos": three_bettor_pos,
        "four_bettor_pos": four_bettor_pos,
        "callers_after_open": callers_after_open,
        "callers": callers_after_open,
        "node_type_preview": None,
        "hero_position": hero_position,
        "hero_context_preview": {},
        "reconstructed_preflop": {},
        "skipped_positions": _dedupe_preserve_order(skipped_positions),
        "unresolved_positions": _dedupe_preserve_order(unresolved_positions),
        "same_hand_identity": bool(same_hand_identity(previous_hand, analysis)),
        "frame_local_actions": _clone_action_list(frame_local_actions),
        "frame_local_terminal_actions": _clone_action_list(frame_local_terminal_actions),
        "frame_local_max_aggression": frame_local_max_aggression,
        "frame_observation": {
            "actor_order": list(actor_order),
            "final_contribution_bb_by_pos": _round_map(contributions),
            "final_contribution_street_bb_by_pos": _round_map(street_contribs),
            "forced_blinds": _round_map(forced),
            "street_commitments_current_frame": _round_map(street_commitments_current_frame),
            "source_mode": source_mode,
            "observation_contract_version": FRAME_OBSERVATION_CONTRACT_VERSION,
        },
    }


def build_preflop_projection(resolved_state: Dict[str, Any]) -> Dict[str, Any]:
    """Build the canonical preflop projection payload from resolved state.

    This layer is the official step-3.6 contract. Downstream consumers should
    read this object instead of rebuilding preview fields from frame-local or
    ad-hoc state.
    """
    resolved = dict(resolved_state or {})
    resolved_actions = _normalize_action_orders(
        _clone_action_list(resolved.get("action_history_resolved") or resolved.get("action_history") or [])
    )
    projection = _build_projection_mapping(
        node_type_preview=resolved.get("projection_node_type") or resolved.get("node_type") or resolved.get("node_type_preview"),
        opener_pos=resolved.get("opener_pos"),
        three_bettor_pos=resolved.get("three_bettor_pos"),
        four_bettor_pos=resolved.get("four_bettor_pos"),
        last_aggressor_position=resolved.get("last_aggressor_position"),
    )
    limpers_count = _normalize_count(
        resolved.get("limpers_count")
        if resolved.get("limpers_count") is not None
        else resolved.get("limpers")
    )
    callers_count = _normalize_count(
        resolved.get("callers")
        if resolved.get("callers") is not None
        else resolved.get("callers_after_open")
    )
    hero_context_preview = dict(resolved.get("hero_context_preview") or {})
    hero_context_preview.update(
        {
            "hero_pos": resolved.get("hero_position"),
            "node_type": projection.get("projection_node_type"),
            "projection_node_type": projection.get("projection_node_type"),
            "advisor_node_type": projection.get("advisor_node_type"),
            "advisor_four_bettor_pos": projection.get("advisor_four_bettor_pos"),
            "advisor_mapping_reason": projection.get("advisor_mapping_reason"),
            "opener_pos": resolved.get("opener_pos"),
            "three_bettor_pos": resolved.get("three_bettor_pos"),
            "four_bettor_pos": resolved.get("four_bettor_pos"),
            "limpers": limpers_count,
            "callers": callers_count,
            "resolved": True,
            "projection_source": "preflop_projection",
            "projection_contract_version": PREFLOP_PROJECTION_CONTRACT_VERSION,
            "reconciliation_mode": resolved.get("reconciliation_mode"),
            "reconciliation_applied": bool(resolved.get("reconciliation_applied", False)),
            "reconstruction_confidence": resolved.get("reconstruction_confidence"),
            "frame_local_only": str(resolved.get("reconciliation_mode") or "") == FRAME_ONLY_RECONCILIATION_MODE,
        }
    )
    return {
        "contract_version": PREFLOP_PROJECTION_CONTRACT_VERSION,
        "projection_source": "resolved_preflop_ledger",
        "projection_node_type": projection.get("projection_node_type"),
        "advisor_node_type": projection.get("advisor_node_type"),
        "advisor_four_bettor_pos": projection.get("advisor_four_bettor_pos"),
        "advisor_mapping_reason": projection.get("advisor_mapping_reason"),
        "node_type": projection.get("projection_node_type"),
        "hero_position": resolved.get("hero_position"),
        "opener_pos": resolved.get("opener_pos"),
        "three_bettor_pos": resolved.get("three_bettor_pos"),
        "four_bettor_pos": resolved.get("four_bettor_pos"),
        "limpers": limpers_count,
        "callers": callers_count,
        "action_history": list(resolved_actions),
        "action_history_resolved": list(resolved_actions),
        "final_contribution_bb_by_pos": _round_map(
            resolved.get("resolved_commitments_by_pos") or resolved.get("final_contribution_bb_by_pos") or {}
        ),
        "final_contribution_street_bb_by_pos": _round_map(
            resolved.get("final_contribution_street_bb_by_pos") or {}
        ),
        "positions_closed_to_action": _clone_string_list(resolved.get("positions_closed_to_action") or []),
        "reconciliation_mode": resolved.get("reconciliation_mode"),
        "reconciliation_applied": bool(resolved.get("reconciliation_applied", False)),
        "reconciliation_notes": _clone_string_list(resolved.get("reconciliation_notes") or []),
        "reconstruction_confidence": resolved.get("reconstruction_confidence"),
        "hero_context_preview": hero_context_preview,
    }


def reconcile_preflop_with_hand(
    previous_hand: Any,
    current_frame_result: Dict[str, Any],
    *,
    analysis: Any,
    settings: Any,
    safe_float: Optional[SafeFloat] = None,
    derive_node_type_preview: Optional[DeriveNodeTypePreview] = None,
    build_preflop_resolved_ledger: Optional[BuildResolvedLedger] = None,
    build_action_step: Optional[BuildActionStep] = None,
) -> Dict[str, Any]:
    """Reconcile frame-local preflop output with the same-hand resolved ledger.

    This is the real step-3.4 layer. It compares the current final commitments
    against the previous resolved ledger and appends only *escalation* actions
    that happened inside the same HERO-cards hand.
    """
    result = dict(current_frame_result or {})
    safe_float_fn = safe_float or (lambda value, default=0.0: default if value is None else float(value))

    actor_order = _clone_string_list(result.get("actor_order") or [])
    hero_position = result.get("hero_position") or getattr(analysis, "hero_position", None)
    player_count = int(result.get("player_count") or getattr(analysis, "player_count", 0) or 0)
    source_mode = str(result.get("source_mode") or "forced_blinds_plus_visible_chips")

    frame_local_actions = _clone_action_list(result.get("frame_local_actions") or result.get("action_history") or [])
    frame_local_terminal_actions = _clone_action_list(result.get("frame_local_terminal_actions") or result.get("historical_terminal_actions") or [])
    frame_local_max_aggression = result.get("frame_local_max_aggression")
    skipped_positions = _clone_string_list(result.get("skipped_positions") or [])
    unresolved_positions = _clone_string_list(result.get("unresolved_positions") or [])

    current_commitments = {
        str(pos): safe_float_fn(amount, 0.0)
        for pos, amount in dict(result.get("final_contribution_bb_by_pos") or {}).items()
    }
    current_street_commitments = {
        str(pos): safe_float_fn(amount, 0.0)
        for pos, amount in dict(result.get("final_contribution_street_bb_by_pos") or {}).items()
    }
    frame_observation = dict(result.get("frame_observation") or {})
    forced_blinds = {
        str(pos): safe_float_fn(amount, 0.0)
        for pos, amount in dict(frame_observation.get("forced_blinds") or {}).items()
    }
    player_states = dict(getattr(analysis, "player_states", {}) or {})
    frame_id = result.get("frame_id") or getattr(analysis, "frame_id", None)
    timestamp = result.get("timestamp") or getattr(analysis, "timestamp", None)
    same_hand = bool(result.get("same_hand_identity", False))

    previous_resolved = _extract_previous_resolved_preflop(previous_hand) if same_hand else {}
    previous_actions = _clone_action_list(
        previous_resolved.get("action_history_resolved")
        or previous_resolved.get("action_history")
        or []
    )
    previous_commitments = _extract_previous_commitments(previous_resolved)

    resolved_actions: List[Dict[str, Any]]
    actions_this_frame: List[Dict[str, Any]]
    terminal_actions: List[Dict[str, Any]]
    reconciliation_notes: List[str] = []
    reconciliation_applied = False

    if same_hand and previous_actions:
        resolved_actions = _clone_action_list(previous_actions)
        previous_state = _derive_state_from_actions(previous_actions)
        current_price_to_call = float(
            previous_state.get("current_price_to_call")
            or result.get("current_price_to_call_after_action")
            or result.get("current_highest_commitment")
            or 0.0
        )
        raise_level = int(previous_state.get("raise_level") or 0)
        limpers = _clone_string_list(previous_state.get("limpers") or [])
        opener_pos = previous_state.get("opener_pos")
        three_bettor_pos = previous_state.get("three_bettor_pos")
        four_bettor_pos = previous_state.get("four_bettor_pos")
        callers_after_open = int(previous_state.get("callers_after_open") or 0)
        last_aggressor_position = previous_state.get("last_aggressor_position")
        actions_this_frame = []

        for position in actor_order:
            current_amount = float(current_commitments.get(position, 0.0))
            forced_amount = float(forced_blinds.get(position, 0.0))
            previous_amount = float(previous_commitments.get(position, forced_amount))
            if current_amount < forced_amount:
                current_amount = forced_amount
            if previous_amount < forced_amount:
                previous_amount = forced_amount
            if current_amount <= previous_amount + 1e-9:
                continue
            if build_action_step is None:
                continue

            if raise_level == 0:
                if abs(current_amount - current_price_to_call) <= 1e-9:
                    semantic_action = "limp"
                    if position not in limpers:
                        limpers.append(position)
                    extra: Dict[str, Any] = {}
                else:
                    semantic_action = "open_raise" if not limpers else "iso_raise"
                    opener_pos = position
                    raise_level = 1
                    current_price_to_call = current_amount
                    callers_after_open = 0
                    extra = {}
                    if limpers:
                        extra["open_family"] = "open_raise"
                        extra["isolates_limpers"] = list(limpers)
                    last_aggressor_position = position
            elif raise_level == 1:
                if abs(current_amount - current_price_to_call) <= 1e-9:
                    semantic_action = "call"
                    callers_after_open += 1
                    extra = {"call_vs": "open_raise"}
                else:
                    semantic_action = "3bet"
                    three_bettor_pos = position
                    raise_level = 2
                    current_price_to_call = current_amount
                    extra = {}
                    last_aggressor_position = position
            elif raise_level == 2:
                if abs(current_amount - current_price_to_call) <= 1e-9:
                    semantic_action = "call"
                    extra = {"call_vs": "3bet"}
                else:
                    semantic_action = "4bet"
                    extra = {}
                    if position not in {opener_pos, three_bettor_pos}:
                        extra["spot_family"] = "cold_4bet"
                    four_bettor_pos = position
                    raise_level = 3
                    current_price_to_call = current_amount
                    last_aggressor_position = position
            else:
                if abs(current_amount - current_price_to_call) <= 1e-9:
                    semantic_action = "call"
                    extra = {"call_vs": "4bet"}
                else:
                    semantic_action = "5bet_jam"
                    raise_level += 1
                    current_price_to_call = current_amount
                    extra = {}
                    last_aggressor_position = position

            step = build_action_step(
                order=len(resolved_actions) + 1,
                position=position,
                street="preflop",
                final_contribution_bb=current_amount,
                semantic_action=semantic_action,
                current_price_to_call=current_price_to_call,
                raise_level=raise_level,
                frame_id=frame_id,
                timestamp=timestamp,
                extra=extra,
            )
            resolved_actions.append(step)
            actions_this_frame.append(step)
            previous_commitments[position] = current_amount

        terminal_actions = _append_terminal_folds(
            resolved_actions=resolved_actions,
            actor_order=actor_order,
            commitments=current_commitments,
            forced_blinds=forced_blinds,
            player_states=player_states,
            current_price_to_call=current_price_to_call,
            final_aggression_label=frame_local_max_aggression or ("5bet_or_more" if raise_level >= 4 else None),
            frame_id=frame_id,
            timestamp=timestamp,
            decorate_legacy_action_fields=None,
        )
        for action in terminal_actions:
            action["order"] = len(resolved_actions) + 1
            resolved_actions.append(action)
            actions_this_frame.append(action)

        reconciliation_applied = bool(actions_this_frame)
        if reconciliation_applied:
            reconciliation_notes.append("history_rebuilt_from_previous_resolved_ledger")
        else:
            reconciliation_notes.append("same_hand_no_commitment_growth_detected")
    else:
        resolved_actions = _clone_action_list(frame_local_actions)
        actions_this_frame = _clone_action_list(frame_local_actions)
        terminal_actions = _clone_action_list(frame_local_terminal_actions)
        current_state = _derive_state_from_actions(resolved_actions)
        current_price_to_call = float(
            current_state.get("current_price_to_call")
            or result.get("current_price_to_call_after_action")
            or result.get("current_highest_commitment")
            or 0.0
        )
        raise_level = int(current_state.get("raise_level") or 0)
        limpers = _clone_string_list(current_state.get("limpers") or [])
        opener_pos = current_state.get("opener_pos")
        three_bettor_pos = current_state.get("three_bettor_pos")
        four_bettor_pos = current_state.get("four_bettor_pos")
        callers_after_open = int(current_state.get("callers_after_open") or 0)
        last_aggressor_position = current_state.get("last_aggressor_position")
        if same_hand and not previous_actions:
            reconciliation_notes.append("same_hand_without_previous_resolved_history_used_frame_local_result")
        else:
            reconciliation_notes.append("frame_local_result_used")

    resolved_actions = _normalize_action_orders(resolved_actions)
    actions_this_frame = _normalize_action_orders(actions_this_frame)
    terminal_actions = _normalize_action_orders(terminal_actions)

    state_from_actions = _derive_state_from_actions(resolved_actions)
    limpers = _clone_string_list(state_from_actions.get("limpers") or limpers)
    opener_pos = state_from_actions.get("opener_pos") or opener_pos
    three_bettor_pos = state_from_actions.get("three_bettor_pos") or three_bettor_pos
    four_bettor_pos = state_from_actions.get("four_bettor_pos") or four_bettor_pos
    callers_after_open = int(state_from_actions.get("callers_after_open") or callers_after_open)
    current_price_to_call = float(state_from_actions.get("current_price_to_call") or current_price_to_call)
    raise_level = int(state_from_actions.get("raise_level") or raise_level)
    last_aggressor_position = state_from_actions.get("last_aggressor_position") or last_aggressor_position

    positions_closed_to_action = _compute_positions_closed_to_action(
        actor_order,
        current_price_to_call=current_price_to_call,
        commitments=current_commitments,
        player_states=player_states,
        last_aggressor_position=last_aggressor_position,
        forced_blinds=forced_blinds,
    )

    if derive_node_type_preview is not None:
        node_type_preview = derive_node_type_preview(
            hero_position=hero_position,
            player_count=player_count,
            limpers=limpers,
            opener_pos=opener_pos,
            three_bettor_pos=three_bettor_pos,
            four_bettor_pos=four_bettor_pos,
            callers_after_open=callers_after_open,
            action_history=resolved_actions,
        )
    else:
        node_type_preview = result.get("node_type_preview")

    projection_meta = _build_projection_mapping(
        node_type_preview=node_type_preview,
        opener_pos=opener_pos,
        three_bettor_pos=three_bettor_pos,
        four_bettor_pos=four_bettor_pos,
        last_aggressor_position=last_aggressor_position,
    )

    reconstruction_confidence = 1.0 if not unresolved_positions else 0.8

    projection_payload: Dict[str, Any]

    if build_preflop_resolved_ledger is not None:
        resolved_ledger = build_preflop_resolved_ledger(
            hero_position=hero_position,
            actor_order=actor_order,
            action_history=resolved_actions,
            final_contribution_bb_by_pos=current_commitments,
            final_contribution_street_bb_by_pos=current_street_commitments,
            current_price_to_call=current_price_to_call,
            opener_pos=opener_pos,
            three_bettor_pos=three_bettor_pos,
            four_bettor_pos=four_bettor_pos,
            limpers=limpers,
            callers_after_open=callers_after_open,
            node_type_preview=projection_meta["projection_node_type"],
            source_mode=source_mode,
            skipped_positions=skipped_positions,
            same_hand_identity=same_hand,
        )
    else:
        resolved_ledger = {
            "street": "preflop",
            "source_mode": source_mode,
            "hero_position": hero_position,
            "node_type": projection_meta["projection_node_type"],
            "opener_pos": opener_pos,
            "three_bettor_pos": three_bettor_pos,
            "four_bettor_pos": four_bettor_pos,
            "limpers": list(limpers),
            "limpers_count": len(limpers),
            "callers": callers_after_open,
            "callers_after_open": callers_after_open,
            "action_history": list(resolved_actions),
            "action_history_resolved": list(resolved_actions),
            "actor_order": list(actor_order),
            "current_price_to_call": round(current_price_to_call, 4),
            "last_aggressor_position": last_aggressor_position,
            "final_contribution_bb_by_pos": _round_map(current_commitments),
            "final_contribution_street_bb_by_pos": _round_map(current_street_commitments),
            "skipped_positions": list(skipped_positions),
            "same_hand_identity": same_hand,
        }

    resolved_ledger.update(
        {
            "action_history": list(resolved_actions),
            "action_history_resolved": list(resolved_actions),
            "node_type": projection_meta["projection_node_type"],
            "resolved_commitments_by_pos": _round_map(current_commitments),
            "last_frame_id": frame_id,
            "reconstruction_confidence": reconstruction_confidence,
            "reconciliation_mode": (
                COMMITMENT_GROWTH_RECONCILIATION_MODE if (same_hand and previous_actions) else FRAME_ONLY_RECONCILIATION_MODE
            ),
            "reconciliation_applied": reconciliation_applied,
            "reconciliation_notes": list(reconciliation_notes),
            "projection_node_type": projection_meta["projection_node_type"],
            "advisor_node_type": projection_meta["advisor_node_type"],
            "advisor_four_bettor_pos": projection_meta["advisor_four_bettor_pos"],
            "advisor_mapping_reason": projection_meta["advisor_mapping_reason"],
        }
    )

    projection_payload = build_preflop_projection(
        {
            **resolved_ledger,
            "hero_position": hero_position,
            "positions_closed_to_action": list(positions_closed_to_action),
        }
    )
    hero_context_preview = dict(projection_payload.get("hero_context_preview") or {})
    resolved_ledger["hero_context_preview"] = hero_context_preview
    resolved_ledger["preflop_projection"] = dict(projection_payload)

    last_actions_by_position: Dict[str, str] = {}
    for action in resolved_actions:
        position = _action_position(action)
        if not position:
            continue
        display = str(action.get("action_display") or action.get("action") or _action_semantic(action) or "")
        last_actions_by_position[position] = display

    return {
        "street": "preflop",
        "source_mode": source_mode,
        "player_count": player_count,
        "actor_order": list(actor_order),
        "street_commitments": {
            pos: round(float(current_street_commitments.get(pos, current_commitments.get(pos, 0.0))), 4)
            for pos in actor_order
        },
        "current_highest_commitment": round(current_price_to_call, 4),
        "last_aggressor_position": last_aggressor_position,
        "acted_positions": _dedupe_preserve_order(_action_position(action) for action in resolved_actions if _action_semantic(action) != "fold"),
        "last_actions_by_position": dict(last_actions_by_position),
        "actions_this_frame": list(actions_this_frame),
        "action_history": list(resolved_actions),
        "action_history_resolved": list(resolved_actions),
        "historical_terminal_actions": list(terminal_actions),
        "final_contribution_bb_by_pos": _round_map(current_commitments),
        "final_contribution_street_bb_by_pos": _round_map(current_street_commitments),
        "resolved_commitments_by_pos": _round_map(current_commitments),
        "semantic_action": (_action_semantic(resolved_actions[-1]) if resolved_actions else None),
        "engine_action": (resolved_actions[-1].get("engine_action") if resolved_actions else None),
        "raise_level_after_action": raise_level,
        "current_price_to_call_after_action": round(current_price_to_call, 4),
        "limpers": list(limpers),
        "limpers_count": len(limpers),
        "opener_pos": opener_pos,
        "three_bettor_pos": three_bettor_pos,
        "four_bettor_pos": four_bettor_pos,
        "callers_after_open": callers_after_open,
        "callers": callers_after_open,
        "node_type_preview": projection_payload["projection_node_type"],
        "projection_node_type": projection_payload["projection_node_type"],
        "advisor_node_type": projection_payload["advisor_node_type"],
        "advisor_four_bettor_pos": projection_payload["advisor_four_bettor_pos"],
        "advisor_mapping_reason": projection_payload["advisor_mapping_reason"],
        "hero_position": hero_position,
        "hero_context_preview": hero_context_preview,
        "preflop_projection": dict(projection_payload),
        "reconstructed_preflop": resolved_ledger,
        "skipped_positions": _dedupe_preserve_order(skipped_positions),
        "unresolved_positions": _dedupe_preserve_order(unresolved_positions),
        "positions_closed_to_action": list(positions_closed_to_action),
        "same_hand_identity": same_hand,
        "frame_local_actions": list(frame_local_actions),
        "frame_local_terminal_actions": list(frame_local_terminal_actions),
        "frame_local_max_aggression": frame_local_max_aggression,
        "frame_observation": dict(frame_observation),
        "reconstruction_confidence": reconstruction_confidence,
        "reconciliation": {
            "mode": COMMITMENT_GROWTH_RECONCILIATION_MODE if (same_hand and previous_actions) else FRAME_ONLY_RECONCILIATION_MODE,
            "applied": reconciliation_applied,
            "same_hand_identity": same_hand,
            "previous_hand_available": previous_hand is not None,
            "reconstruction_confidence": reconstruction_confidence,
            "positions_closed_to_action": list(positions_closed_to_action),
            "notes": list(reconciliation_notes),
        },
        "contract_version": HAND_RECONCILIATION_CONTRACT_VERSION,
    }
