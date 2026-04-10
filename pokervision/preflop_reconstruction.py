from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


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


FRAME_OBSERVATION_CONTRACT_VERSION = "preflop_frame_observation_v1"
FRAME_RECONCILIATION_MODE = "frame_only_passthrough"


def build_preflop_frame_observation(
    analysis: Any,
    *,
    settings: Any,
    safe_float: SafeFloat,
    forced_preflop_blinds: ForcedPreflopBlinds,
    get_street_actor_order: GetStreetActorOrder,
) -> Dict[str, Any]:
    """Build a canonical raw observation for a single preflop frame.

    This layer extracts only current-frame inputs and does not try to claim a
    full hand history. It is the answer to the question:
    "What is visible on the final snapshot right now?"
    """

    amount_state = getattr(analysis, "amount_state", None) or getattr(analysis, "amount_normalization", None) or {}
    contributions = {
        str(pos): safe_float(amount, 0.0)
        for pos, amount in (amount_state.get("final_contribution_bb_by_pos") or {}).items()
    }
    street_contribs = {
        str(pos): safe_float(amount, 0.0)
        for pos, amount in (amount_state.get("final_contribution_street_bb_by_pos") or {}).items()
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
    """Run a single-pass frame-local preflop reconstruction.

    CRITICAL INVARIANT FOR STEP 3.3:
    This pass may emit only a minimal plausible line for *the current final
    snapshot*. It may be incomplete. It must not pretend to be the full
    historical truth of the hand.
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
    frame_local_terminal_actions: List[Dict[str, Any]] = []
    last_actions_by_position: Dict[str, str] = {}
    acted_positions: List[str] = []
    skipped_positions: List[str] = []
    unresolved_positions: List[str] = []

    for position in actor_order:
        amount = float(contributions.get(position, 0.0))
        forced_amount = float(forced.get(position, 0.0))
        is_fold = player_is_folded(player_states, position)

        if amount < forced_amount:
            amount = forced_amount

        # Blind-only / no-voluntary-money spots are not unresolved.
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
                last_actions_by_position[position] = legacy_last_action_display(step)
                acted_positions.append(position)
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
                skipped_positions.append(position)
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
                skipped_positions.append(position)
                unresolved_positions.append(position)
                continue
        elif raise_level == 2:
            if approx_eq(amount, current_price_to_call, eps):
                semantic_action = "call"
                extra = {"call_vs": "3bet"}
            elif amount > current_price_to_call + eps:
                semantic_action = "4bet"
                extra = {"spot_family": "cold_4bet"} if position not in {opener_pos, three_bettor_pos} else {}
                four_bettor_pos = position
                raise_level = 3
                current_price_to_call = amount
            else:
                skipped_positions.append(position)
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
                skipped_positions.append(position)
                unresolved_positions.append(position)
                continue

        if is_fold and amount <= forced_amount + eps:
            skipped_positions.append(position)
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
        last_actions_by_position[position] = legacy_last_action_display(step)
        acted_positions.append(position)

    frame_local_max_aggression = final_aggression_label(raise_level)
    if frame_local_max_aggression is not None and current_price_to_call > eps:
        for position in actor_order:
            if not player_is_folded(player_states, position):
                continue
            amount = float(contributions.get(position, 0.0))
            forced_amount = float(forced.get(position, 0.0))
            if amount <= forced_amount + eps:
                continue
            if amount >= current_price_to_call - eps:
                continue
            prior_action = next((step for step in reversed(frame_local_actions) if step.get("position") == position), None)
            fold_event = decorate_legacy_action_fields(
                {
                    "order": len(frame_local_actions) + len(frame_local_terminal_actions) + 1,
                    "position": position,
                    "pos": position,
                    "street": "preflop",
                    "semantic_action": "fold",
                    "engine_action": "fold",
                    "amount_bb": None,
                    "final_contribution_bb": round(amount, 4),
                    "facing_price_to_call_bb": round(current_price_to_call, 4),
                    "facing_aggression": frame_local_max_aggression,
                    "inferred_terminal_action": True,
                    "reason": "folded_with_residual_preflop_contribution_below_final_price",
                    "prior_semantic_action": prior_action.get("semantic_action") if prior_action else None,
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                }
            )
            frame_local_terminal_actions.append(fold_event)
            last_actions_by_position[position] = legacy_last_action_display(fold_event)

    resolved_frame_actions = [
        decorate_legacy_action_fields(item)
        for item in [*frame_local_actions, *frame_local_terminal_actions]
    ]
    unresolved_positions = list(dict.fromkeys(str(pos) for pos in unresolved_positions))
    skipped_positions = list(dict.fromkeys(str(pos) for pos in skipped_positions))

    return {
        "street": "preflop",
        "source_mode": source_mode,
        "observation_contract_version": observation.get(
            "observation_contract_version",
            FRAME_OBSERVATION_CONTRACT_VERSION,
        ),
        "frame_local_only": True,
        "frame_local_scope": "current_final_snapshot_only",
        "actor_order": list(actor_order),
        "player_count": player_count,
        "hero_position": hero_position,
        "forced_blinds": {pos: round(value, 4) for pos, value in forced.items()},
        "final_contribution_bb_by_pos": {pos: round(value, 4) for pos, value in contributions.items()},
        "final_contribution_street_bb_by_pos": {pos: round(value, 4) for pos, value in street_contribs.items()},
        "street_commitments_current_frame": {
            pos: round(float(street_commitments_current_frame.get(pos, contributions.get(pos, 0.0))), 4)
            for pos in actor_order
        },
        "frame_local_actions": list(resolved_frame_actions),
        "frame_local_terminal_actions": [decorate_legacy_action_fields(item) for item in frame_local_terminal_actions],
        "skipped_positions": list(skipped_positions),
        "unresolved_positions": list(unresolved_positions),
        "frame_local_max_aggression": frame_local_max_aggression,
        "frame_local_raise_level": raise_level,
        "frame_local_current_price_to_call": round(current_price_to_call, 4),
        "frame_local_last_aggressor_position": four_bettor_pos or three_bettor_pos or opener_pos,
        "frame_local_opener_pos": opener_pos,
        "frame_local_three_bettor_pos": three_bettor_pos,
        "frame_local_four_bettor_pos": four_bettor_pos,
        "frame_local_limpers": list(limpers),
        "frame_local_callers_after_open": callers_after_open,
        "acted_positions": list(acted_positions),
        "last_actions_by_position": dict(last_actions_by_position),
        "actions_this_frame": list(resolved_frame_actions),
        "action_history": list(resolved_frame_actions),
        "action_history_resolved": list(resolved_frame_actions),
        "semantic_action": (resolved_frame_actions[-1]["semantic_action"] if resolved_frame_actions else None),
        "engine_action": (resolved_frame_actions[-1].get("engine_action") if resolved_frame_actions else None),
        "raise_level_after_action": raise_level,
        "current_price_to_call_after_action": round(current_price_to_call, 4),
        "same_hand_identity": bool(same_hand_identity(previous_hand, analysis)),
        "frame_observation": {
            "contract_version": observation.get("observation_contract_version", FRAME_OBSERVATION_CONTRACT_VERSION),
            "actor_order": list(actor_order),
            "final_contribution_bb_by_pos": {pos: round(value, 4) for pos, value in contributions.items()},
            "final_contribution_street_bb_by_pos": {pos: round(value, 4) for pos, value in street_contribs.items()},
            "forced_blinds": {pos: round(value, 4) for pos, value in forced.items()},
            "street_commitments_current_frame": {
                pos: round(float(street_commitments_current_frame.get(pos, contributions.get(pos, 0.0))), 4)
                for pos in actor_order
            },
            "frame_local_actions": list(resolved_frame_actions),
            "skipped_positions": list(skipped_positions),
            "unresolved_positions": list(unresolved_positions),
            "frame_local_max_aggression": frame_local_max_aggression,
            "source_mode": source_mode,
        },
    }


def _legacy_action_identity(action: Dict[str, Any]) -> tuple:
    return (
        str(action.get("street") or "preflop"),
        str(action.get("position") or action.get("pos") or ""),
        str(action.get("semantic_action") or ""),
        round(float(action.get("final_contribution_bb") or action.get("amount_bb") or 0.0), 4),
        round(float(action.get("current_price_to_call_after_action") or 0.0), 4),
    )


def _clone_action_list(actions: Any) -> List[Dict[str, Any]]:
    return [dict(item) for item in list(actions or []) if isinstance(item, dict)]


def _normalized_amount_map(values: Any, *, safe_float: Optional[SafeFloat] = None) -> Dict[str, float]:
    if safe_float is None:
        safe_float = lambda value, default=0.0: default if value is None else float(value)
    return {
        str(pos): round(float(safe_float(amount, 0.0)), 4)
        for pos, amount in dict(values or {}).items()
    }


def _derive_raise_level_from_actions(actions: List[Dict[str, Any]]) -> int:
    level = 0
    for action in actions:
        semantic = str(action.get("semantic_action") or "")
        if semantic in {"open_raise", "iso_raise"}:
            level = max(level, 1)
        elif semantic == "3bet":
            level = max(level, 2)
        elif semantic == "4bet":
            level = max(level, 3)
        elif semantic == "5bet_jam":
            level = max(level, 4)
    return level


def _derive_preflop_state_from_actions(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    limpers: List[str] = []
    opener_pos: Optional[str] = None
    three_bettor_pos: Optional[str] = None
    four_bettor_pos: Optional[str] = None
    callers_after_open = 0
    current_price_to_call = 1.0
    last_aggressor: Optional[str] = None
    raise_level = 0

    for action in actions:
        position = str(action.get("position") or action.get("pos") or "")
        semantic = str(action.get("semantic_action") or "")
        amount = float(action.get("final_contribution_bb") or action.get("amount_bb") or 0.0)
        if semantic == "limp":
            if position and position not in limpers:
                limpers.append(position)
            continue
        if semantic in {"open_raise", "iso_raise"}:
            opener_pos = opener_pos or position
            raise_level = max(raise_level, 1)
            current_price_to_call = amount
            last_aggressor = position
            continue
        if semantic == "3bet":
            three_bettor_pos = three_bettor_pos or position
            raise_level = max(raise_level, 2)
            current_price_to_call = amount
            last_aggressor = position
            continue
        if semantic == "4bet":
            four_bettor_pos = four_bettor_pos or position
            raise_level = max(raise_level, 3)
            current_price_to_call = amount
            last_aggressor = position
            continue
        if semantic == "5bet_jam":
            raise_level = max(raise_level, 4)
            current_price_to_call = amount
            last_aggressor = position
            continue
        if semantic == "call" and str(action.get("call_vs") or "") == "open_raise":
            callers_after_open += 1

    return {
        "limpers": list(limpers),
        "opener_pos": opener_pos,
        "three_bettor_pos": three_bettor_pos,
        "four_bettor_pos": four_bettor_pos,
        "callers_after_open": callers_after_open,
        "current_price_to_call": round(current_price_to_call, 4),
        "last_aggressor_position": last_aggressor,
        "raise_level": raise_level,
    }


def _previous_reconciler_state(previous_resolved: Dict[str, Any]) -> Dict[str, Any]:
    return dict(previous_resolved.get("reconciliation_state") or {})


def _extract_previous_preflop_resolved(previous_hand: Any) -> Dict[str, Any]:
    if previous_hand is None:
        return {}
    action_state = dict(getattr(previous_hand, "action_state", {}) or {})
    reconstructed = dict(action_state.get("reconstructed_preflop") or {})
    if reconstructed:
        return reconstructed
    if str(action_state.get("street") or "").lower() == "preflop":
        return dict(action_state)
    return {}


def _circular_actor_order(actor_order: List[str], start_after: Optional[str]) -> List[str]:
    if not actor_order:
        return []
    if start_after not in actor_order:
        return list(actor_order)
    index = actor_order.index(start_after) + 1
    return list(actor_order[index:] + actor_order[:index])


def _compute_positions_closed_to_action(
    actor_order: List[str],
    *,
    current_price_to_call: float,
    commitments: Dict[str, float],
    player_states: Dict[str, Dict[str, object]],
    last_aggressor_position: Optional[str],
    eps: float = 1e-9,
) -> List[str]:
    if not actor_order or current_price_to_call <= eps:
        return []
    closed: List[str] = []
    for position in _circular_actor_order(actor_order, last_aggressor_position):
        player_state = dict(player_states.get(position, {}) or {})
        if bool(player_state.get("is_fold", False)):
            closed.append(position)
            continue
        amount = float(commitments.get(position, 0.0))
        if amount + eps < current_price_to_call:
            break
        closed.append(position)
    return closed


def _reconciliation_confidence(*, applied: bool, unresolved_positions: List[str], notes: List[str]) -> float:
    confidence = 1.0 if applied else 0.9
    if unresolved_positions:
        confidence -= min(0.35, 0.1 * len(unresolved_positions))
    if any("regression" in note for note in notes):
        confidence = min(confidence, 0.35)
    if any("guard_break" in note for note in notes):
        confidence = min(confidence, 0.45)
    return round(max(0.05, min(confidence, 1.0)), 4)


def _append_unique_action(target: List[Dict[str, Any]], action: Dict[str, Any]) -> bool:
    signature = _legacy_action_identity(action)
    if any(_legacy_action_identity(item) == signature for item in target):
        return False
    target.append(dict(action))
    return True


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
    """Reconcile current preflop frame-local observation with the same hand.

    Step 3.4 goals:
    - compare the previous resolved preflop ledger with the current final commitments
    - treat the single-pass result only as a frame-local observation
    - build an updated aggression ladder for the same HERO cards instead of
      reinterpreting every new final snapshot from scratch.
    """

    result = dict(current_frame_result or {})
    safe_float_fn = safe_float or (lambda value, default=0.0: default if value is None else float(value))
    same_hand = bool(result.get("same_hand_identity", False))
    actor_order = [str(pos) for pos in list(result.get("actor_order") or [])]
    hero_position = result.get("hero_position") or getattr(analysis, "hero_position", None)
    player_count = int(result.get("player_count") or getattr(analysis, "player_count", 0) or 0)
    source_mode = str(result.get("source_mode") or "forced_blinds_plus_visible_chips")
    frame_local_actions = _clone_action_list(result.get("frame_local_actions") or [])
    frame_local_terminal_actions = _clone_action_list(result.get("frame_local_terminal_actions") or [])
    frame_local_max_aggression = result.get("frame_local_max_aggression")
    skipped_positions = [str(pos) for pos in list(result.get("skipped_positions") or [])]
    unresolved_positions = [str(pos) for pos in list(result.get("unresolved_positions") or [])]
    forced_blinds = _normalized_amount_map(result.get("forced_blinds") or {}, safe_float=safe_float_fn)
    current_commitments = _normalized_amount_map(result.get("final_contribution_bb_by_pos") or {}, safe_float=safe_float_fn)
    current_street_commitments = _normalized_amount_map(result.get("final_contribution_street_bb_by_pos") or {}, safe_float=safe_float_fn)
    player_states = dict(getattr(analysis, "player_states", {}) or result.get("player_states") or {})

    previous_resolved = _extract_previous_preflop_resolved(previous_hand) if same_hand else {}
    previous_actions = _clone_action_list(previous_resolved.get("action_history_resolved") or previous_resolved.get("action_history") or [])
    previous_commitments = _normalized_amount_map(previous_resolved.get("final_contribution_bb_by_pos") or {}, safe_float=safe_float_fn)
    previous_state = _previous_reconciler_state(previous_resolved)

    reconciliation_notes: List[str] = []
    reconciliation_applied = False
    fallback_to_frame_local = False

    if not same_hand or not previous_resolved or not previous_actions:
        resolved_actions = list(frame_local_actions)
        reconciliation_notes.append("no_previous_resolved_ledger_available")
    else:
        regression_positions = []
        for position in set(current_commitments) | set(previous_commitments):
            current_amount = float(current_commitments.get(position, forced_blinds.get(position, 0.0)))
            previous_amount = float(previous_commitments.get(position, forced_blinds.get(position, 0.0)))
            if current_amount + 1e-9 < previous_amount:
                regression_positions.append(str(position))
        if regression_positions:
            resolved_actions = list(frame_local_actions)
            reconciliation_notes.append(
                "commitment_regression_detected:" + ",".join(sorted(regression_positions))
            )
            fallback_to_frame_local = True
        else:
            resolved_actions = list(previous_actions)
            resolved_commitments = {
                str(pos): round(float(previous_commitments.get(pos, forced_blinds.get(pos, 0.0))), 4)
                for pos in set(actor_order) | set(current_commitments) | set(previous_commitments) | set(forced_blinds)
            }
            previous_state_from_actions = _derive_preflop_state_from_actions(resolved_actions)
            raise_level = int(previous_state.get("raise_level") or previous_state_from_actions.get("raise_level") or 0)
            current_price_to_call = float(
                previous_resolved.get("current_price_to_call")
                or previous_state.get("current_price_to_call")
                or previous_state_from_actions.get("current_price_to_call")
                or 0.0
            )
            last_action_position = previous_actions[-1].get("position") if previous_actions else None
            guard = 0
            appended_actions: List[Dict[str, Any]] = []
            unresolved_growth_positions: List[str] = []

            while True:
                pending_positions = [
                    pos
                    for pos in actor_order
                    if float(current_commitments.get(pos, forced_blinds.get(pos, 0.0)))
                    > float(resolved_commitments.get(pos, forced_blinds.get(pos, 0.0))) + 1e-9
                ]
                if not pending_positions:
                    break
                guard += 1
                if guard > max(8, len(actor_order) * 4):
                    reconciliation_notes.append("reconciliation_guard_break")
                    unresolved_growth_positions.extend(pending_positions)
                    break

                cycle = _circular_actor_order(actor_order, last_action_position)
                if not cycle:
                    cycle = list(actor_order)

                progressed_this_cycle = False
                raised_this_cycle = False
                for position in cycle:
                    current_amount = float(current_commitments.get(position, forced_blinds.get(position, 0.0)))
                    known_amount = float(resolved_commitments.get(position, forced_blinds.get(position, 0.0)))
                    if current_amount <= known_amount + 1e-9:
                        continue

                    semantic_action: Optional[str] = None
                    extra: Dict[str, Any] = {}

                    if raise_level <= 0:
                        if current_amount <= current_price_to_call + 1e-9:
                            semantic_action = "limp"
                            extra["limp_subtype"] = "over_limp" if resolved_actions else "open_limp"
                        else:
                            semantic_action = "open_raise"
                            if previous_state_from_actions.get("limpers"):
                                semantic_action = "iso_raise"
                            raise_level = 1
                            current_price_to_call = current_amount
                    elif current_amount <= current_price_to_call + 1e-9:
                        semantic_action = "call"
                        if raise_level == 1:
                            extra["call_vs"] = "open_raise"
                        elif raise_level == 2:
                            extra["call_vs"] = "3bet"
                        elif raise_level >= 3:
                            extra["call_vs"] = "4bet"
                    else:
                        if raise_level == 1:
                            semantic_action = "3bet"
                            raise_level = 2
                        elif raise_level == 2:
                            semantic_action = "4bet"
                            if position not in {
                                previous_state_from_actions.get("opener_pos"),
                                previous_state_from_actions.get("three_bettor_pos"),
                            }:
                                extra["spot_family"] = "cold_4bet"
                            raise_level = 3
                        else:
                            semantic_action = "5bet_jam"
                            raise_level = max(4, raise_level + 1)
                        current_price_to_call = current_amount

                    if semantic_action is None or build_action_step is None:
                        unresolved_growth_positions.append(position)
                        continue

                    step = build_action_step(
                        order=len(resolved_actions) + len(appended_actions) + 1,
                        position=position,
                        street="preflop",
                        final_contribution_bb=current_amount,
                        semantic_action=semantic_action,
                        current_price_to_call=current_price_to_call,
                        raise_level=raise_level,
                        frame_id=getattr(analysis, "frame_id", None),
                        timestamp=getattr(analysis, "timestamp", None),
                        extra=extra,
                    )
                    appended = _append_unique_action(resolved_actions, step)
                    resolved_commitments[position] = round(current_amount, 4)
                    if appended:
                        appended_actions.append(step)
                        progressed_this_cycle = True
                    last_action_position = position
                    if semantic_action in {"open_raise", "iso_raise", "3bet", "4bet", "5bet_jam"}:
                        raised_this_cycle = True
                        break

                if unresolved_growth_positions:
                    unresolved_positions.extend(unresolved_growth_positions)
                if raised_this_cycle:
                    continue
                if not progressed_this_cycle:
                    break

            for fold_event in frame_local_terminal_actions:
                if str(fold_event.get("semantic_action") or "") != "fold":
                    continue
                _append_unique_action(resolved_actions, fold_event)

            if appended_actions:
                reconciliation_notes.append(
                    "appended_reconciled_actions:" + ",".join(
                        f"{action.get('position')}={action.get('semantic_action')}:{action.get('final_contribution_bb')}"
                        for action in appended_actions
                    )
                )
            else:
                reconciliation_notes.append("no_new_commitment_growth_detected")

            reconciliation_applied = bool(appended_actions)

    state_from_actions = _derive_preflop_state_from_actions(resolved_actions)
    limpers = list(state_from_actions.get("limpers") or [])
    opener_pos = state_from_actions.get("opener_pos")
    three_bettor_pos = state_from_actions.get("three_bettor_pos")
    four_bettor_pos = state_from_actions.get("four_bettor_pos")
    callers_after_open = int(state_from_actions.get("callers_after_open") or 0)
    current_price_to_call = float(state_from_actions.get("current_price_to_call") or result.get("frame_local_current_price_to_call") or 0.0)
    last_aggressor_position = state_from_actions.get("last_aggressor_position")
    raise_level = int(state_from_actions.get("raise_level") or _derive_raise_level_from_actions(resolved_actions) or 0)

    unresolved_positions = list(dict.fromkeys(str(pos) for pos in unresolved_positions))
    skipped_positions = list(dict.fromkeys(str(pos) for pos in skipped_positions))

    positions_closed_to_action = _compute_positions_closed_to_action(
        actor_order,
        current_price_to_call=current_price_to_call,
        commitments=current_commitments,
        player_states=player_states,
        last_aggressor_position=last_aggressor_position,
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

    resolved_ledger: Dict[str, Any] = {}
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
            node_type_preview=node_type_preview,
            source_mode=source_mode,
            skipped_positions=skipped_positions,
            same_hand_identity=same_hand,
        )

    if not resolved_ledger:
        resolved_ledger = {
            "street": "preflop",
            "source_mode": source_mode,
            "hero_position": hero_position,
            "node_type": node_type_preview,
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
            "final_contribution_bb_by_pos": dict(current_commitments),
            "final_contribution_street_bb_by_pos": dict(current_street_commitments),
            "hero_context_preview": {
                "hero_pos": hero_position,
                "node_type": node_type_preview,
                "opener_pos": opener_pos,
                "three_bettor_pos": three_bettor_pos,
                "four_bettor_pos": four_bettor_pos,
                "limpers": len(limpers),
                "callers": callers_after_open,
                "resolved": True,
                "projection_source": "reconstructed_preflop",
            },
            "skipped_positions": list(skipped_positions),
            "same_hand_identity": same_hand,
            "contract_version": "preflop_resolved_v1",
        }

    confidence = _reconciliation_confidence(
        applied=reconciliation_applied and not fallback_to_frame_local,
        unresolved_positions=unresolved_positions,
        notes=reconciliation_notes,
    )
    resolution_scope = "hand_reconciled" if same_hand and not fallback_to_frame_local else "frame_local_only"
    frame_local_only = resolution_scope != "hand_reconciled"
    reconciliation_mode = "commitment_growth_reconciled" if resolution_scope == "hand_reconciled" else FRAME_RECONCILIATION_MODE

    reconciliation_state = {
        "resolved_actions": _clone_action_list(resolved_actions),
        "resolved_commitments_by_pos": dict(current_commitments),
        "last_aggressor": last_aggressor_position,
        "raise_level": raise_level,
        "opener_pos": opener_pos,
        "three_bettor_pos": three_bettor_pos,
        "four_bettor_pos": four_bettor_pos,
        "positions_closed_to_action": list(positions_closed_to_action),
        "last_frame_id": getattr(analysis, "frame_id", None),
        "reconstruction_confidence": confidence,
        "reconciliation_notes": list(reconciliation_notes),
    }

    resolved_ledger.update(
        {
            "resolution_scope": resolution_scope,
            "frame_local_only": frame_local_only,
            "frame_local_max_aggression": frame_local_max_aggression,
            "unresolved_positions": list(unresolved_positions),
            "frame_observation_contract_version": result.get(
                "observation_contract_version",
                FRAME_OBSERVATION_CONTRACT_VERSION,
            ),
            "reconciliation_mode": reconciliation_mode,
            "reconciliation_applied": bool(reconciliation_applied and not fallback_to_frame_local),
            "reconciliation_state": dict(reconciliation_state),
            "resolved_actions": _clone_action_list(resolved_actions),
            "resolved_commitments_by_pos": dict(current_commitments),
            "last_frame_id": getattr(analysis, "frame_id", None),
            "reconstruction_confidence": confidence,
            "reconciliation_notes": list(reconciliation_notes),
            "last_aggressor_position": last_aggressor_position,
            "current_price_to_call": round(current_price_to_call, 4),
            "action_history": _clone_action_list(resolved_actions),
            "action_history_resolved": _clone_action_list(resolved_actions),
            "final_contribution_bb_by_pos": dict(current_commitments),
            "final_contribution_street_bb_by_pos": dict(current_street_commitments),
        }
    )

    hero_context_preview = dict(resolved_ledger.get("hero_context_preview") or {})
    hero_context_preview.setdefault("hero_pos", hero_position)
    hero_context_preview["node_type"] = node_type_preview
    hero_context_preview["opener_pos"] = opener_pos
    hero_context_preview["three_bettor_pos"] = three_bettor_pos
    hero_context_preview["four_bettor_pos"] = four_bettor_pos
    hero_context_preview["limpers"] = len(limpers)
    hero_context_preview["callers"] = callers_after_open
    hero_context_preview["resolved"] = True
    hero_context_preview["projection_source"] = "reconstructed_preflop"
    hero_context_preview["resolution_scope"] = resolution_scope
    hero_context_preview["frame_local_only"] = frame_local_only
    hero_context_preview["reconciliation_confidence"] = confidence
    resolved_ledger["hero_context_preview"] = hero_context_preview

    actions_this_frame = []
    existing_signatures = {_legacy_action_identity(item) for item in previous_actions}
    for action in resolved_actions:
        if _legacy_action_identity(action) not in existing_signatures:
            actions_this_frame.append(dict(action))

    result.update(
        {
            "street": "preflop",
            "node_type_preview": node_type_preview,
            "hero_position": hero_position,
            "limpers": list(limpers),
            "limpers_count": len(limpers),
            "opener_pos": opener_pos,
            "three_bettor_pos": three_bettor_pos,
            "four_bettor_pos": four_bettor_pos,
            "callers_after_open": callers_after_open,
            "callers": callers_after_open,
            "hero_context_preview": dict(hero_context_preview),
            "reconstructed_preflop": dict(resolved_ledger),
            "actions_this_frame": list(actions_this_frame),
            "action_history": _clone_action_list(resolved_actions),
            "action_history_resolved": _clone_action_list(resolved_actions),
            "same_hand_identity": same_hand,
            "semantic_action": (resolved_actions[-1]["semantic_action"] if resolved_actions else None),
            "engine_action": (resolved_actions[-1].get("engine_action") if resolved_actions else None),
            "raise_level_after_action": raise_level,
            "current_price_to_call_after_action": round(current_price_to_call, 4),
            "reconciliation": {
                "mode": reconciliation_mode,
                "applied": bool(reconciliation_applied and not fallback_to_frame_local),
                "same_hand_identity": same_hand,
                "previous_hand_available": previous_hand is not None,
                "reconstruction_confidence": confidence,
                "positions_closed_to_action": list(positions_closed_to_action),
                "notes": list(reconciliation_notes),
            },
        }
    )
    return result
