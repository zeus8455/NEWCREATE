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


def build_preflop_frame_observation(
    analysis: Any,
    *,
    settings: Any,
    safe_float: SafeFloat,
    forced_preflop_blinds: ForcedPreflopBlinds,
    get_street_actor_order: GetStreetActorOrder,
) -> Dict[str, Any]:
    """Build a canonical raw observation for a single preflop frame.

    This layer should stay focused on extracting and normalizing the raw inputs
    needed by preflop reconstruction. It intentionally does *not* derive the
    final semantic line yet.
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

    return {
        "street": "preflop",
        "source_mode": amount_state.get("source_mode", "forced_blinds_plus_visible_chips"),
        "amount_state": amount_state,
        "contributions": contributions,
        "street_contribs": street_contribs,
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
    derive_node_type_preview: DeriveNodeTypePreview,
    build_preflop_resolved_ledger: BuildResolvedLedger,
    same_hand_identity: SameHandIdentity,
    player_is_folded: PlayerIsFolded,
    eps: float,
) -> Dict[str, Any]:
    """Reconstruct the semantic preflop line from one frame observation.

    This layer is intentionally frame-local. It must not own persistence or
    downstream projection concerns.
    """

    contributions = dict(observation.get("contributions") or {})
    street_contribs = dict(observation.get("street_contribs") or {})
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
    action_history: List[Dict[str, Any]] = []
    last_actions_by_position: Dict[str, str] = {}
    acted_positions: List[str] = []
    skipped_positions: List[str] = []

    for position in actor_order:
        amount = float(contributions.get(position, 0.0))
        forced_amount = float(forced.get(position, 0.0))
        is_fold = player_is_folded(player_states, position)

        if amount < forced_amount:
            amount = forced_amount

        if amount <= forced_amount + eps:
            if position == "BB" and raise_level == 0 and limpers and approx_eq(amount, current_price_to_call, eps):
                step = build_action_step(
                    order=len(action_history) + 1,
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
                action_history.append(step)
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
                continue
        elif raise_level == 2:
            if approx_eq(amount, current_price_to_call, eps):
                semantic_action = "call"
                extra = {"call_vs": "3bet"}
            elif amount > current_price_to_call + eps:
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
                continue

        if is_fold and amount <= forced_amount + eps:
            skipped_positions.append(position)
            continue

        step = build_action_step(
            order=len(action_history) + 1,
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
        action_history.append(step)
        last_actions_by_position[position] = legacy_last_action_display(step)
        acted_positions.append(position)

    terminal_actions: List[Dict[str, Any]] = []
    final_aggression = final_aggression_label(raise_level)
    if final_aggression is not None and current_price_to_call > eps:
        for position in actor_order:
            if not player_is_folded(player_states, position):
                continue
            amount = float(contributions.get(position, 0.0))
            forced_amount = float(forced.get(position, 0.0))
            if amount <= forced_amount + eps:
                continue
            if amount >= current_price_to_call - eps:
                continue
            prior_action = next((step for step in reversed(action_history) if step.get("position") == position), None)
            fold_event = decorate_legacy_action_fields(
                {
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
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                }
            )
            terminal_actions.append(fold_event)
            last_actions_by_position[position] = legacy_last_action_display(fold_event)

    node_type_preview = derive_node_type_preview(
        hero_position=hero_position,
        player_count=player_count,
        limpers=limpers,
        opener_pos=opener_pos,
        three_bettor_pos=three_bettor_pos,
        four_bettor_pos=four_bettor_pos,
        callers_after_open=callers_after_open,
        action_history=action_history,
    )

    resolved_history = [decorate_legacy_action_fields(item) for item in [*action_history, *terminal_actions]]
    resolved_ledger = build_preflop_resolved_ledger(
        hero_position=hero_position,
        actor_order=actor_order,
        action_history=resolved_history,
        final_contribution_bb_by_pos=contributions,
        final_contribution_street_bb_by_pos=street_contribs,
        current_price_to_call=current_price_to_call,
        opener_pos=opener_pos,
        three_bettor_pos=three_bettor_pos,
        four_bettor_pos=four_bettor_pos,
        limpers=limpers,
        callers_after_open=callers_after_open,
        node_type_preview=node_type_preview,
        source_mode=source_mode,
        skipped_positions=skipped_positions,
        same_hand_identity=same_hand_identity(previous_hand, analysis),
    )

    return {
        "street": "preflop",
        "source_mode": source_mode,
        "actor_order": actor_order,
        "street_commitments": {pos: round(contributions.get(pos, 0.0), 4) for pos in actor_order},
        "current_highest_commitment": round(current_price_to_call, 4),
        "last_aggressor_position": four_bettor_pos or three_bettor_pos or opener_pos,
        "acted_positions": list(acted_positions),
        "last_actions_by_position": dict(last_actions_by_position),
        "actions_this_frame": list(resolved_history),
        "action_history": list(resolved_history),
        "action_history_resolved": list(resolved_history),
        "historical_terminal_actions": [decorate_legacy_action_fields(item) for item in terminal_actions],
        "final_contribution_bb_by_pos": {pos: round(value, 4) for pos, value in contributions.items()},
        "final_contribution_street_bb_by_pos": {pos: round(value, 4) for pos, value in street_contribs.items()},
        "semantic_action": (resolved_history[-1]["semantic_action"] if resolved_history else None),
        "engine_action": (resolved_history[-1].get("engine_action") if resolved_history else None),
        "raise_level_after_action": (
            action_history[-1]["raise_level_after_action"] if action_history else 0
        ),
        "current_price_to_call_after_action": (
            action_history[-1]["current_price_to_call_after_action"] if action_history else round(current_price_to_call, 4)
        ),
        "limpers": list(limpers),
        "limpers_count": len(limpers),
        "opener_pos": opener_pos,
        "three_bettor_pos": three_bettor_pos,
        "four_bettor_pos": four_bettor_pos,
        "callers_after_open": callers_after_open,
        "callers": callers_after_open,
        "node_type_preview": node_type_preview,
        "hero_position": hero_position,
        "hero_context_preview": dict(resolved_ledger.get("hero_context_preview") or {}),
        "reconstructed_preflop": dict(resolved_ledger),
        "skipped_positions": skipped_positions,
        "same_hand_identity": bool(resolved_ledger.get("same_hand_identity", False)),
        "frame_observation": {
            "actor_order": list(actor_order),
            "final_contribution_bb_by_pos": {pos: round(value, 4) for pos, value in contributions.items()},
            "final_contribution_street_bb_by_pos": {pos: round(value, 4) for pos, value in street_contribs.items()},
            "forced_blinds": {pos: round(value, 4) for pos, value in forced.items()},
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
) -> Dict[str, Any]:
    """Hand-level reconciliation entry point.

    In step 3.2 this remains intentionally conservative: it preserves the
    frame-local result while establishing a dedicated place for future temporal
    reconciliation logic.
    """

    result = dict(current_frame_result or {})
    same_hand = bool(result.get("same_hand_identity", False))
    result["reconciliation"] = {
        "mode": "frame_only_passthrough",
        "applied": False,
        "same_hand_identity": same_hand,
        "previous_hand_available": previous_hand is not None,
        "notes": [
            "preflop_reconstruction module owns the hand-level reconciliation entry point",
            "temporal reconciliation is deferred to the next substep",
        ],
    }

    reconstructed = dict(result.get("reconstructed_preflop") or {})
    reconstructed.setdefault("reconciliation_mode", "frame_only_passthrough")
    reconstructed.setdefault("reconciliation_applied", False)
    result["reconstructed_preflop"] = reconstructed
    return result
