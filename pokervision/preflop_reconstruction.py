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


def reconcile_preflop_with_hand(
    previous_hand: Any,
    current_frame_result: Dict[str, Any],
    *,
    analysis: Any,
    settings: Any,
    safe_float: Optional[SafeFloat] = None,
    derive_node_type_preview: Optional[DeriveNodeTypePreview] = None,
    build_preflop_resolved_ledger: Optional[BuildResolvedLedger] = None,
) -> Dict[str, Any]:
    """Hand-level reconciliation entry point.

    For step 3.3 we still keep a conservative passthrough mode so the rest of
    the project continues to work. The important change is that the current
    single-pass is now explicitly treated as frame-local observation, and the
    downstream "resolved" payload is marked as such.
    """

    result = dict(current_frame_result or {})
    same_hand = bool(result.get("same_hand_identity", False))
    actor_order = list(result.get("actor_order") or [])
    hero_position = result.get("hero_position") or getattr(analysis, "hero_position", None)
    player_count = int(result.get("player_count") or getattr(analysis, "player_count", 0) or 0)
    limpers = list(result.get("frame_local_limpers") or [])
    opener_pos = result.get("frame_local_opener_pos")
    three_bettor_pos = result.get("frame_local_three_bettor_pos")
    four_bettor_pos = result.get("frame_local_four_bettor_pos")
    callers_after_open = int(result.get("frame_local_callers_after_open") or 0)
    frame_local_actions = [dict(item) for item in list(result.get("frame_local_actions") or [])]
    current_price_to_call = float(result.get("frame_local_current_price_to_call") or 0.0)
    source_mode = str(result.get("source_mode") or "forced_blinds_plus_visible_chips")
    skipped_positions = list(result.get("skipped_positions") or [])
    unresolved_positions = list(result.get("unresolved_positions") or [])

    if derive_node_type_preview is not None:
        node_type_preview = derive_node_type_preview(
            hero_position=hero_position,
            player_count=player_count,
            limpers=limpers,
            opener_pos=opener_pos,
            three_bettor_pos=three_bettor_pos,
            four_bettor_pos=four_bettor_pos,
            callers_after_open=callers_after_open,
            action_history=frame_local_actions,
        )
    else:
        node_type_preview = result.get("node_type_preview")

    resolved_ledger: Dict[str, Any] = {}
    if build_preflop_resolved_ledger is not None:
        resolved_ledger = build_preflop_resolved_ledger(
            hero_position=hero_position,
            actor_order=actor_order,
            action_history=frame_local_actions,
            final_contribution_bb_by_pos=dict(result.get("final_contribution_bb_by_pos") or {}),
            final_contribution_street_bb_by_pos=dict(result.get("final_contribution_street_bb_by_pos") or {}),
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
            "action_history": list(frame_local_actions),
            "action_history_resolved": list(frame_local_actions),
            "actor_order": list(actor_order),
            "current_price_to_call": round(current_price_to_call, 4),
            "last_aggressor_position": four_bettor_pos or three_bettor_pos or opener_pos,
            "final_contribution_bb_by_pos": dict(result.get("final_contribution_bb_by_pos") or {}),
            "final_contribution_street_bb_by_pos": dict(result.get("final_contribution_street_bb_by_pos") or {}),
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

    resolved_ledger["resolution_scope"] = "frame_local_only"
    resolved_ledger["frame_local_only"] = True
    resolved_ledger["frame_local_max_aggression"] = result.get("frame_local_max_aggression")
    resolved_ledger["unresolved_positions"] = list(unresolved_positions)
    resolved_ledger["frame_observation_contract_version"] = result.get(
        "observation_contract_version",
        FRAME_OBSERVATION_CONTRACT_VERSION,
    )
    resolved_ledger["reconciliation_mode"] = FRAME_RECONCILIATION_MODE
    resolved_ledger["reconciliation_applied"] = False

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
    hero_context_preview["resolution_scope"] = "frame_local_only"
    hero_context_preview["frame_local_only"] = True
    resolved_ledger["hero_context_preview"] = hero_context_preview

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
            "actions_this_frame": list(frame_local_actions),
            "action_history": list(frame_local_actions),
            "action_history_resolved": list(frame_local_actions),
            "same_hand_identity": same_hand,
            "reconciliation": {
                "mode": FRAME_RECONCILIATION_MODE,
                "applied": False,
                "same_hand_identity": same_hand,
                "previous_hand_available": previous_hand is not None,
                "notes": [
                    "single-pass preflop result is now scoped as frame-local observation",
                    "hand-level temporal reconciliation is deferred to the next substep",
                ],
            },
        }
    )
    return result
