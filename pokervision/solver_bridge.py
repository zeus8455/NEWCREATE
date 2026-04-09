
from __future__ import annotations

"""Solver bridge extracted from the external launcher.

This module intentionally contains only the decision/adapter layer that converts
PokerVision `hand` / `action_state` into the decision-layer contracts used by
the external solver files. UI, Qt rendering, CLI and debug-print code are kept
out on purpose.

CRITICAL INVARIANT:
This file must remain a thin adapter between PokerVision state and the
decision-layer contracts. Do not reintroduce UI/Qt/launcher responsibilities
here, otherwise bridge semantics will drift again between the main pipeline and
the external launcher.
"""

from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
import importlib
from types import ModuleType
from typing import Any, Dict, List, Optional

CANONICAL_RING = {
    2: ["BTN", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["BTN", "SB", "BB", "CO"],
    5: ["BTN", "SB", "BB", "UTG", "CO"],
    6: ["BTN", "SB", "BB", "UTG", "MP", "CO"],
}

GENERIC_WIDE_RANGE = (
    "22+ A2s+ K2s+ Q2s+ J2s+ T2s+ 92s+ 82s+ 72s+ 62s+ 52s+ 42s+ 32s "
    "A2o+ K2o+ Q2o+ J2o+ T2o+ 92o+ 82o+ 72o+ 62o+ 52o+"
)

STREET_ORDER = ["preflop", "flop", "turn", "river"]
POSTFLOP_STREETS = ["flop", "turn", "river"]


def _import_bridge_module(module_name: str) -> ModuleType:
    """Import decision-layer modules with package-safe fallback.

    Preferred order:
    1. sibling module inside the current package, e.g. ``pokervision.decision_types``
    2. top-level external module, e.g. ``decision_types``

    This keeps the bridge compatible both with vendored-in solver files and with
    the current external-file layout used by the project.
    """
    candidates: list[str] = []
    if __package__:
        candidates.append(f"{__package__}.{module_name}")
    candidates.append(module_name)

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except Exception as exc:  # pragma: no cover - exercised by fallback tests
            last_error = exc
    if last_error is None:
        raise ImportError(f"Could not import module: {module_name}")
    raise last_error


def _import_decision_types():
    module = _import_bridge_module("decision_types")
    return module.HeroDecision, module.PostflopContext, module.PreflopContext


def _solve_hero_preflop(context):
    module = _import_bridge_module("hero_decision")
    return module.solve_hero_preflop(context)


def _solve_hero_postflop(context, **kwargs):
    module = _import_bridge_module("hero_decision")
    return module.solve_hero_postflop(context, **kwargs)


def _serialize_for_payload(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return {
            field.name: _serialize_for_payload(getattr(value, field.name), depth=depth + 1)
            for field in dataclass_fields(value)
        }
    if isinstance(value, dict):
        return {str(k): _serialize_for_payload(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_for_payload(v, depth=depth + 1) for v in value]
    if hasattr(value, "__dict__"):
        return {
            str(k): _serialize_for_payload(v, depth=depth + 1)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return repr(value)


@dataclass(slots=True)
class ActorFlags:
    limped: bool = False
    opened: bool = False
    threebet: bool = False
    last_voluntary: Optional[str] = None


@dataclass(slots=True)
class ReplayState:
    limpers: List[str] = field(default_factory=list)
    opener: Optional[str] = None
    callers_after_open: List[str] = field(default_factory=list)
    three_bettor: Optional[str] = None
    four_bettor: Optional[str] = None
    flags_by_pos: Dict[str, ActorFlags] = field(default_factory=dict)

    def flags(self, pos: str) -> ActorFlags:
        if pos not in self.flags_by_pos:
            self.flags_by_pos[pos] = ActorFlags()
        return self.flags_by_pos[pos]


@dataclass(slots=True)
class SpotDescription:
    node_type: str
    hero_pos: str
    opener_pos: Optional[str] = None
    three_bettor_pos: Optional[str] = None
    four_bettor_pos: Optional[str] = None
    limpers: int = 0
    callers: int = 0

    def to_preflop_context(self, hero_hand: List[str], *, meta: Optional[Dict[str, object]] = None):
        _, _, PreflopContext = _import_decision_types()
        return PreflopContext(
            hero_hand=list(hero_hand),
            hero_pos=self.hero_pos,
            node_type=self.node_type,
            opener_pos=self.opener_pos,
            three_bettor_pos=self.three_bettor_pos,
            four_bettor_pos=self.four_bettor_pos,
            limpers=self.limpers,
            callers=self.callers,
            range_owner="hero",
            meta={} if meta is None else dict(meta),
        )


class EngineBridge:
    """Thin adapter from PokerVision hand/action state to decision-layer calls."""

    def __init__(self, settings=None) -> None:
        if settings is None:
            try:
                from .config import get_default_settings
                settings = get_default_settings()
            except Exception:
                settings = None
        self.settings = settings

    # ------------------------------
    # basic helpers
    # ------------------------------
    def _setting(self, name: str, default: Any) -> Any:
        if self.settings is None:
            return default
        return getattr(self.settings, name, default)

    def _safe_float(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _preflop_pos(self, pos: str, player_count: int) -> str:
        pos = str(pos or "").upper()
        if player_count == 2 and pos == "BTN":
            return "SB"
        return pos

    def _normalize_effective_stack(self, stack_bb: Optional[float]) -> Optional[float]:
        if stack_bb is None:
            return None
        value = float(stack_bb)
        if not self._setting("normalize_short_stack_to_40bb", False):
            return value
        min_bb = float(self._setting("short_stack_min_inclusive_bb", 0.0))
        max_bb = float(self._setting("short_stack_max_exclusive_bb", 0.0))
        forced = float(self._setting("short_stack_forced_value_bb", value))
        if min_bb <= value < max_bb:
            return forced
        return value

    def _current_bets(self, hand) -> Dict[str, float]:
        table_amount_state = hand.table_amount_state if isinstance(hand.table_amount_state, dict) else {}
        bets = table_amount_state.get("bets_by_position", {}) if isinstance(table_amount_state, dict) else {}
        out: Dict[str, float] = {}
        if isinstance(bets, dict):
            for pos, payload in bets.items():
                if not isinstance(payload, dict):
                    continue
                amount = self._safe_float(payload.get("amount_bb"))
                if amount is not None:
                    out[str(pos)] = amount
        return out

    def _pot_before_hero(self, hand) -> float:
        table_amount_state = hand.table_amount_state if isinstance(hand.table_amount_state, dict) else {}
        total_pot = table_amount_state.get("total_pot", {}) if isinstance(table_amount_state, dict) else {}
        if isinstance(total_pot, dict):
            amount = self._safe_float(total_pot.get("amount_bb"))
            if amount is not None:
                return max(0.0, amount)
        bets = self._current_bets(hand)
        return max(0.0, sum(bets.values()))

    def _to_call(self, hand) -> float:
        bets = self._current_bets(hand)
        hero_commit = bets.get(hand.hero_position, 0.0)
        highest = max(bets.values(), default=0.0)
        return max(0.0, highest - hero_commit)

    def _hero_in_position_postflop(self, hand) -> bool:
        player_count = int(hand.player_count)
        if player_count == 2:
            return hand.hero_position == "BTN"
        order = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
        hero_idx = order.index(hand.hero_position) if hand.hero_position in order else -1
        villain_indices = [
            order.index(pos)
            for pos in hand.occupied_positions
            if pos != hand.hero_position and pos in order and not hand.player_states.get(pos, {}).get("is_fold", False)
        ]
        if not villain_indices or hero_idx < 0:
            return True
        return hero_idx > max(villain_indices)

    def _effective_stack(self, hand) -> Optional[float]:
        hero_state = hand.player_states.get(hand.hero_position, {}) if isinstance(hand.player_states, dict) else {}
        hero_stack = self._normalize_effective_stack(self._safe_float(hero_state.get("stack_bb")))
        villain_stacks: List[float] = []
        if isinstance(hand.player_states, dict):
            for pos, payload in hand.player_states.items():
                if pos == hand.hero_position:
                    continue
                if payload.get("is_fold", False):
                    continue
                stack = self._normalize_effective_stack(self._safe_float(payload.get("stack_bb")))
                if stack is not None:
                    villain_stacks.append(stack)
        if hero_stack is None and not villain_stacks:
            return None
        if hero_stack is None:
            return min(villain_stacks)
        if not villain_stacks:
            return hero_stack
        return min([hero_stack, *villain_stacks])

    def _legacy_action_to_semantic(self, action_name: str) -> str:
        action = str(action_name or "").upper()
        if action == "LIMP":
            return "limp"
        if action == "OPEN":
            return "open_raise"
        if action == "CALL":
            return "call"
        if action == "CHECK":
            return "check"
        if action == "RAISE":
            return "raise"
        if action == "BET":
            return "bet"
        return action.lower()

    def _normalized_preflop_action(self, item: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(item)
        semantic = str(payload.get("semantic_action") or "").strip().lower()
        if not semantic:
            semantic = self._legacy_action_to_semantic(str(payload.get("action", "")))
            if semantic:
                payload["semantic_action"] = semantic
        if not payload.get("engine_action") and semantic:
            if semantic in {"limp", "call"}:
                payload["engine_action"] = "call"
            elif semantic == "check":
                payload["engine_action"] = "check"
            elif semantic == "fold":
                payload["engine_action"] = "fold"
            else:
                payload["engine_action"] = "raise"
        if payload.get("final_contribution_bb") is None and payload.get("amount_bb") is not None:
            payload["final_contribution_bb"] = payload.get("amount_bb")
        if payload.get("action") in {None, ""} and semantic:
            payload["action"] = semantic.upper()
        return payload

    def _street_actions(self, hand, street: str) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        for item in list(getattr(hand, "actions_log", []) or []):
            if str(item.get("street", "")).lower() != street.lower():
                continue
            payload = dict(item)
            if street.lower() == "preflop":
                payload = self._normalized_preflop_action(payload)
            actions.append(payload)
        return actions

    def _ordered_positions(self, hand) -> List[str]:
        ring = CANONICAL_RING.get(int(hand.player_count), [])
        return [pos for pos in ring if pos in hand.occupied_positions]

    def _street_rank(self, street: str) -> int:
        try:
            return STREET_ORDER.index(str(street).lower())
        except ValueError:
            return -1

    def _streets_up_to(self, street: str) -> List[str]:
        rank = self._street_rank(street)
        if rank < 0:
            return []
        return [s for s in STREET_ORDER[: rank + 1] if s in POSTFLOP_STREETS]

    def _sum_visible_total_pot(self, hand) -> float:
        return self._pot_before_hero(hand)

    # ------------------------------
    # preflop replay / spot inference
    # ------------------------------
    def _infer_actor_spot_before_action(self, state: ReplayState, actor_pos: str) -> Optional[SpotDescription]:
        flags = state.flags(actor_pos)

        if flags.last_voluntary is None:
            if state.opener is None:
                if state.limpers:
                    if actor_pos == "BB" and len(state.limpers) == 1 and state.limpers[0] == "SB":
                        return SpotDescription(node_type="bb_vs_sb_limp", hero_pos=actor_pos, opener_pos="SB", limpers=1)
                    return SpotDescription(
                        node_type="facing_limp",
                        hero_pos=actor_pos,
                        opener_pos=state.limpers[0] if state.limpers else None,
                        limpers=len(state.limpers),
                    )
                return SpotDescription(node_type="unopened", hero_pos=actor_pos)

            if state.three_bettor is None:
                return SpotDescription(
                    node_type="facing_open_callers" if state.callers_after_open else "facing_open",
                    hero_pos=actor_pos,
                    opener_pos=state.opener,
                    callers=len(state.callers_after_open),
                )

            if state.four_bettor is None:
                return SpotDescription(
                    node_type="cold_4bet",
                    hero_pos=actor_pos,
                    opener_pos=state.opener,
                    three_bettor_pos=state.three_bettor,
                )
            return None

        if flags.limped and state.opener and state.opener != actor_pos and state.three_bettor is None:
            return SpotDescription(node_type="limper_vs_iso", hero_pos=actor_pos, opener_pos=state.opener, limpers=1)

        if flags.opened and state.three_bettor and state.three_bettor != actor_pos and state.four_bettor is None:
            return SpotDescription(node_type="opener_vs_3bet", hero_pos=actor_pos, three_bettor_pos=state.three_bettor)

        if flags.threebet and state.four_bettor and state.four_bettor != actor_pos:
            return SpotDescription(node_type="threebettor_vs_4bet", hero_pos=actor_pos, four_bettor_pos=state.four_bettor)

        return None

    def _apply_preflop_action(self, state: ReplayState, actor_pos: str, action_payload: Dict[str, Any]) -> None:
        payload = self._normalized_preflop_action(action_payload)
        semantic = str(payload.get("semantic_action") or "").strip().lower()
        legacy = str(payload.get("action") or "").upper()
        action = semantic or self._legacy_action_to_semantic(legacy)
        flags = state.flags(actor_pos)

        if action == "check":
            flags.last_voluntary = "CHECK"
            return

        if action == "limp":
            if actor_pos not in state.limpers and state.opener is None:
                state.limpers.append(actor_pos)
            flags.limped = True
            flags.last_voluntary = "LIMP"
            return

        if action in {"open_raise", "iso_raise"}:
            if state.opener is None:
                state.opener = actor_pos
            flags.opened = True
            flags.last_voluntary = action.upper()
            return

        if action == "call":
            if state.opener is not None and state.three_bettor is None and actor_pos not in state.callers_after_open:
                state.callers_after_open.append(actor_pos)
            flags.last_voluntary = "CALL"
            return

        if action == "3bet":
            if state.opener is None:
                state.opener = actor_pos
                flags.opened = True
            elif state.three_bettor is None:
                state.three_bettor = actor_pos
                flags.threebet = True
            flags.last_voluntary = "3BET"
            return

        if action in {"4bet", "cold_4bet"}:
            if state.opener is None:
                state.opener = actor_pos
                flags.opened = True
            elif state.three_bettor is None:
                state.three_bettor = actor_pos
                flags.threebet = True
            state.four_bettor = actor_pos
            flags.last_voluntary = action.upper()
            return

        if action == "5bet_jam":
            if state.opener is None:
                state.opener = actor_pos
                flags.opened = True
            elif state.three_bettor is None:
                state.three_bettor = actor_pos
                flags.threebet = True
            elif state.four_bettor is None:
                state.four_bettor = actor_pos
            flags.last_voluntary = "5BET_JAM"
            return

        if legacy == "LIMP":
            if actor_pos not in state.limpers and state.opener is None:
                state.limpers.append(actor_pos)
            flags.limped = True
            flags.last_voluntary = legacy
            return

        if legacy == "OPEN":
            if state.opener is None:
                state.opener = actor_pos
            flags.opened = True
            flags.last_voluntary = legacy
            return

        if legacy == "CALL":
            if state.opener is not None and state.three_bettor is None and actor_pos not in state.callers_after_open:
                state.callers_after_open.append(actor_pos)
            flags.last_voluntary = legacy
            return

        if legacy == "RAISE":
            if state.opener is None:
                state.opener = actor_pos
                flags.opened = True
            elif state.three_bettor is None:
                state.three_bettor = actor_pos
                flags.threebet = True
            elif state.four_bettor is None:
                state.four_bettor = actor_pos
            flags.last_voluntary = legacy
            return

    def _map_action_for_spot(self, spot: Optional[SpotDescription], action_payload: Dict[str, Any]) -> Optional[str]:
        if spot is None:
            return None
        payload = self._normalized_preflop_action(action_payload)
        semantic = str(payload.get("semantic_action") or "").strip().lower()
        legacy = str(payload.get("action") or "").upper()
        if semantic in {"", "raise"}:
            semantic = self._legacy_action_to_semantic(legacy)
        if semantic == "fold" or legacy == "FOLD":
            return None

        if spot.node_type == "unopened":
            if semantic == "limp":
                return "limp"
            if semantic in {"open_raise", "iso_raise", "raise"}:
                return "raise"
            return None

        if spot.node_type in {"facing_limp", "bb_vs_sb_limp"}:
            if semantic in {"open_raise", "iso_raise", "raise"}:
                return "iso_raise" if spot.node_type == "facing_limp" else "raise"
            if semantic == "call":
                return "call"
            return None

        if spot.node_type == "limper_vs_iso":
            if semantic in {"3bet", "4bet", "cold_4bet", "5bet_jam", "raise"}:
                return "3bet"
            if semantic == "call":
                return "call"
            return None

        if spot.node_type in {"facing_open", "facing_open_callers"}:
            if semantic in {"3bet", "4bet", "cold_4bet", "5bet_jam", "raise"}:
                return "3bet"
            if semantic == "call":
                return "call"
            return None

        if spot.node_type == "opener_vs_3bet":
            if semantic in {"4bet", "cold_4bet", "5bet_jam", "raise"}:
                return "4bet"
            if semantic == "call":
                return "call"
            return None

        if spot.node_type == "threebettor_vs_4bet":
            if semantic in {"5bet_jam", "raise"}:
                return "5bet_jam"
            if semantic == "call":
                return "call"
            return None

        if spot.node_type == "cold_4bet":
            if semantic in {"4bet", "cold_4bet", "raise"}:
                return "4bet"
            if semantic == "call":
                return "call"
            return None

        return None

    def _build_hero_preflop_spot(self, hand) -> SpotDescription:
        state = ReplayState()
        hero_pos = self._preflop_pos(hand.hero_position, int(hand.player_count))
        hero_spot: Optional[SpotDescription] = None

        for action in self._street_actions(hand, "preflop"):
            actor_pos = self._preflop_pos(str(action.get("position", "")), int(hand.player_count))
            if actor_pos == hero_pos:
                hero_spot = self._infer_actor_spot_before_action(state, actor_pos)
            self._apply_preflop_action(state, actor_pos, action)

        flags = state.flags(hero_pos)
        if flags.last_voluntary is None:
            fresh_spot = self._infer_actor_spot_before_action(state, hero_pos)
            if fresh_spot is not None:
                hero_spot = fresh_spot

        if hero_spot is not None:
            return hero_spot
        return SpotDescription(node_type="unopened", hero_pos=hero_pos)

    def _build_villain_preflop_spots(self, hand) -> List[Dict[str, object]]:
        state = ReplayState()
        hero_pos = self._preflop_pos(hand.hero_position, int(hand.player_count))
        villain_spots: Dict[str, Dict[str, object]] = {}

        for action in self._street_actions(hand, "preflop"):
            actor_raw_pos = str(action.get("position", ""))
            actor_pos = self._preflop_pos(actor_raw_pos, int(hand.player_count))
            spot = self._infer_actor_spot_before_action(state, actor_pos)
            mapped_action = self._map_action_for_spot(spot, action) if spot is not None else None

            if actor_pos != hero_pos and spot is not None and mapped_action is not None:
                villain_spots[actor_pos] = {
                    "name": actor_raw_pos,
                    "node_type": spot.node_type,
                    "villain_pos": actor_pos,
                    "villain_action": mapped_action,
                    "opener_pos": spot.opener_pos,
                    "three_bettor_pos": spot.three_bettor_pos,
                    "four_bettor_pos": spot.four_bettor_pos,
                    "limpers": spot.limpers,
                    "callers": spot.callers,
                    "range_owner": "opponent",
                }

            self._apply_preflop_action(state, actor_pos, action)

        active_non_fold = [
            pos for pos in self._ordered_positions(hand)
            if pos != hand.hero_position and not hand.player_states.get(pos, {}).get("is_fold", False)
        ]
        result: List[Dict[str, object]] = []
        for pos in active_non_fold:
            mapped = self._preflop_pos(pos, int(hand.player_count))
            payload = villain_spots.get(mapped)
            if payload is None:
                result.append({
                    "name": pos,
                    "node_type": "facing_open",
                    "villain_pos": mapped,
                    "villain_action": "call",
                    "opener_pos": self._preflop_pos(hand.hero_position, int(hand.player_count)),
                    "callers": 0,
                    "limpers": 0,
                    "range_owner": "opponent",
                })
            else:
                result.append(payload)
        return result

    # ------------------------------
    # postflop line reconstruction
    # ------------------------------
    def _visible_total_pot_or_sum(self, hand) -> float:
        return self._sum_visible_total_pot(hand)

    def _street_final_commitments(self, hand, street: str) -> Dict[str, float]:
        commits: Dict[str, float] = {}
        for action in self._street_actions(hand, street):
            pos = str(action.get("position", ""))
            amount = self._safe_float(action.get("amount_bb"))
            if pos and amount is not None:
                commits[pos] = max(commits.get(pos, 0.0), amount)
        return commits

    def _street_contributions(self, hand) -> Dict[str, float]:
        out: Dict[str, float] = {street: 0.0 for street in STREET_ORDER}
        for street in STREET_ORDER:
            out[street] = sum(self._street_final_commitments(hand, street).values())
        return out

    def _street_start_pot(self, hand, street: str) -> float:
        contributions = self._street_contributions(hand)
        total = 0.0
        for s in STREET_ORDER:
            if s == street:
                break
            total += contributions.get(s, 0.0)
        visible_total = self._visible_total_pot_or_sum(hand)
        if street == str(hand.street_state.get("current_street", "")).lower() and visible_total > 0:
            current_contrib = contributions.get(street, 0.0)
            estimated = max(0.0, visible_total - current_contrib)
            return max(total, estimated)
        return total

    def _map_postflop_action(self, street_actions: List[Dict[str, Any]], idx: int) -> Optional[str]:
        action_name = str(street_actions[idx].get("action", "")).upper()
        actor = str(street_actions[idx].get("position", ""))
        if action_name in {"OPEN", "BET"}:
            return "bet"
        if action_name == "CALL":
            return "call"
        if action_name == "CHECK":
            return "check_back"
        if action_name != "RAISE":
            return None

        earlier = street_actions[:idx]
        earlier_aggressive = [a for a in earlier if str(a.get("action", "")).upper() in {"BET", "RAISE", "OPEN"}]
        actor_prior_aggressive = [a for a in earlier_aggressive if str(a.get("position", "")) == actor]
        if actor_prior_aggressive:
            return "reraise"
        if earlier_aggressive:
            return "check_raise"
        return "bet"

    def _build_postflop_events_for_position(self, hand, villain_pos: str, current_street: str) -> List[Dict[str, object]]:
        events: List[Dict[str, object]] = []
        streets = self._streets_up_to(current_street)
        for street in streets:
            street_actions = self._street_actions(hand, street)
            if not street_actions:
                continue

            start_pot = max(0.01, self._street_start_pot(hand, street))
            commitments: Dict[str, float] = {}
            pot_progress = start_pot

            for idx, action in enumerate(street_actions):
                actor = str(action.get("position", ""))
                amount = self._safe_float(action.get("amount_bb")) or 0.0
                prev_commit = commitments.get(actor, 0.0)
                increment = max(0.0, amount - prev_commit)
                if amount > prev_commit:
                    commitments[actor] = amount

                mapped = self._map_postflop_action(street_actions, idx)
                if actor == villain_pos and mapped is not None:
                    pct = 0.0
                    if mapped in {"bet", "call", "check_raise", "reraise"}:
                        pct = max(0.0, (increment / max(0.01, pot_progress)) * 100.0)
                    is_all_in = bool(hand.player_states.get(villain_pos, {}).get("is_all_in", False)) and street == current_street
                    events.append(
                        {
                            "street": street,
                            "action": mapped,
                            "bet_pct_pot": pct,
                            "is_all_in": is_all_in,
                        }
                    )

                pot_progress += increment
        return events

    def _build_villain_postflop_players(self, hand, current_street: str) -> List[Dict[str, object]]:
        villain_spots = self._build_villain_preflop_spots(hand)
        by_pos = {str(item.get("name") or item.get("villain_pos")): item for item in villain_spots}
        by_mapped_pos = {str(item.get("villain_pos")): item for item in villain_spots}

        players: List[Dict[str, object]] = []
        for pos in self._ordered_positions(hand):
            if pos == hand.hero_position:
                continue
            state = hand.player_states.get(pos, {})
            if state.get("is_fold", False):
                continue

            pre = by_pos.get(pos) or by_mapped_pos.get(pos)
            if pre is None:
                pre = {
                    "name": pos,
                    "node_type": "facing_open",
                    "villain_pos": pos,
                    "villain_action": "call",
                    "opener_pos": self._preflop_pos(hand.hero_position, int(hand.player_count)),
                    "callers": 0,
                    "limpers": 0,
                    "range_owner": "opponent",
                }

            events = self._build_postflop_events_for_position(hand, pos, current_street)
            players.append(
                {
                    "name": pos,
                    "node_type": pre.get("node_type"),
                    "villain_pos": pre.get("villain_pos", pos),
                    "villain_action": pre.get("villain_action", "call"),
                    "opener_pos": pre.get("opener_pos"),
                    "three_bettor_pos": pre.get("three_bettor_pos"),
                    "four_bettor_pos": pre.get("four_bettor_pos"),
                    "limpers": int(pre.get("limpers", 0) or 0),
                    "callers": int(pre.get("callers", 0) or 0),
                    "range_owner": pre.get("range_owner", "opponent"),
                    "events": events,
                }
            )
        return players

    # ------------------------------
    # recommendation construction
    # ------------------------------
    def build_recommendation(self, analysis, hand):
        if hand is None:
            return {"status": "not_run", "result": None, "reason": "no_hand"}
        hero_cards = list(getattr(hand, "hero_cards", None) or getattr(analysis, "hero_cards", None) or [])
        if len(hero_cards) != 2:
            return {"status": "not_run", "result": None, "reason": "hero_cards_not_exactly_two"}

        street_state = getattr(hand, "street_state", {}) if hand is not None else {}
        street = str(street_state.get("current_street") or getattr(analysis, "street", None) or "preflop").lower()
        if street == "preflop":
            return self._build_preflop_recommendation(analysis, hand, hero_cards)
        return self._build_postflop_recommendation(analysis, hand, hero_cards, street)

    def build_preflop_context(self, analysis, hand):
        _, _, PreflopContext = _import_decision_types()
        hero_cards = list(getattr(hand, "hero_cards", None) or getattr(analysis, "hero_cards", None) or [])
        if hand is None or len(hero_cards) != 2:
            return None

        action_state = getattr(hand, "action_state", {}) or {}
        hero_preview = dict(action_state.get("hero_context_preview") or {})
        fallback_spot: Optional[SpotDescription] = None

        node_type = str(hero_preview.get("node_type") or action_state.get("node_type_preview") or "").strip()
        if not node_type:
            fallback_spot = self._build_hero_preflop_spot(hand)
            node_type = fallback_spot.node_type

        opener_pos = hero_preview.get("opener_pos")
        three_bettor_pos = hero_preview.get("three_bettor_pos")
        four_bettor_pos = hero_preview.get("four_bettor_pos")
        limpers = hero_preview.get("limpers")
        callers = hero_preview.get("callers")

        if fallback_spot is None and (
            opener_pos is None and three_bettor_pos is None and four_bettor_pos is None and limpers is None and callers is None
        ):
            fallback_spot = self._build_hero_preflop_spot(hand)

        if fallback_spot is not None:
            opener_pos = fallback_spot.opener_pos if opener_pos is None else opener_pos
            three_bettor_pos = fallback_spot.three_bettor_pos if three_bettor_pos is None else three_bettor_pos
            four_bettor_pos = fallback_spot.four_bettor_pos if four_bettor_pos is None else four_bettor_pos
            limpers = fallback_spot.limpers if limpers is None else limpers
            callers = fallback_spot.callers if callers is None else callers

        action_history = list(action_state.get("action_history") or self._street_actions(hand, "preflop"))
        hero_pos = self._preflop_pos(hand.hero_position, int(hand.player_count))
        return PreflopContext(
            hero_hand=list(hero_cards),
            hero_pos=hero_pos,
            node_type=node_type or "unopened",
            opener_pos=opener_pos,
            three_bettor_pos=three_bettor_pos,
            four_bettor_pos=four_bettor_pos,
            limpers=int(limpers or 0),
            callers=int(callers or 0),
            range_owner="hero",
            action_history=action_history,
            meta={
                "source": "pokervision",
                "hand_id": hand.hand_id,
                "hero_original_position": hand.hero_position,
                "projection_source": "action_state_preview" if hero_preview else "replayed_actions_fallback",
                "actions_seen": self._street_actions(hand, "preflop"),
            },
        )

    def build_postflop_context(self, analysis, hand, street: Optional[str] = None):
        _, PostflopContext, _ = _import_decision_types()
        hero_cards = list(getattr(hand, "hero_cards", None) or getattr(analysis, "hero_cards", None) or [])
        if hand is None or len(hero_cards) != 2:
            return None
        resolved_street = str(street or getattr(hand, "street_state", {}).get("current_street") or getattr(analysis, "street", "") or "").lower()
        board = list(getattr(hand, "board_cards", None) or getattr(analysis, "board_cards", None) or [])
        if resolved_street not in POSTFLOP_STREETS or len(board) not in {3, 4, 5}:
            return None

        villain_positions = [
            pos for pos in self._ordered_positions(hand)
            if pos != hand.hero_position and not hand.player_states.get(pos, {}).get("is_fold", False)
        ]
        if not villain_positions:
            return None

        line_context = self._build_postflop_line_context(hand, resolved_street)
        return PostflopContext(
            hero_hand=list(hero_cards),
            board=list(board),
            pot_before_hero=self._pot_before_hero(hand),
            to_call=self._to_call(hand),
            effective_stack=self._effective_stack(hand),
            hero_position=hand.hero_position,
            villain_positions=list(villain_positions),
            line_context=line_context,
            dead_cards=[],
            street=resolved_street,
            player_count=int(hand.player_count),
            meta={
                "source": "pokervision",
                "hand_id": hand.hand_id,
                "hero_original_position": hand.hero_position,
                "projection_source": "postflop_runtime_projection",
                "hero_in_position": self._hero_in_position_postflop(hand),
            },
        )

    def _build_contract_payload(self, context: Any) -> Dict[str, Any]:
        contract_name = type(context).__name__
        payload = {"context_type": contract_name}
        payload.update(_serialize_for_payload(context))
        return payload

    def _build_preflop_recommendation(self, analysis, hand, hero_cards: List[str]):
        context = self.build_preflop_context(analysis, hand)
        if context is None:
            return {"status": "not_run", "result": None, "reason": "preflop_context_unavailable"}
        contract_payload = self._build_contract_payload(context)
        result = _solve_hero_preflop(context)
        return {
            "status": "ok",
            "context_type": "PreflopContext",
            "solver_context": dict(contract_payload),
            "advisor_input": dict(contract_payload),
            "projection_meta": _serialize_for_payload(getattr(context, "meta", {})),
            "result": result,
        }

    def _build_postflop_line_context(self, hand, street: str) -> Dict[str, object]:
        current_actions = [a for a in self._street_actions(hand, street)]
        preflop_actions = [a for a in self._street_actions(hand, "preflop")]

        hero_last_aggressor_preflop = False
        for item in reversed(preflop_actions):
            if str(item.get("action", "")).upper() in {"OPEN", "RAISE"}:
                hero_last_aggressor_preflop = str(item.get("position", "")) == hand.hero_position
                break

        current_aggressive = [
            a for a in current_actions
            if str(a.get("action", "")).upper() in {"BET", "RAISE", "OPEN"}
        ]
        to_call = self._to_call(hand)
        villain_positions = [
            pos for pos in self._ordered_positions(hand)
            if pos != hand.hero_position and not hand.player_states.get(pos, {}).get("is_fold", False)
        ]
        return {
            "hero_has_initiative": hero_last_aggressor_preflop,
            "hero_last_aggressor": hero_last_aggressor_preflop,
            "prior_aggression": bool(current_aggressive),
            "facing_raise": to_call > 0.0 and bool(current_aggressive),
            "checked_to_hero": to_call <= 0.0 and not bool(current_aggressive),
            "delayed_spot": False,
            "villain_in_position": not self._hero_in_position_postflop(hand),
            "villain_positions": villain_positions,
        }

    def _build_postflop_recommendation(self, analysis, hand, hero_cards: List[str], street: str):
        context = self.build_postflop_context(analysis, hand, street)
        if context is None:
            return {"status": "not_run", "result": None, "reason": "postflop_context_unavailable"}

        villain_positions = list(getattr(context, "villain_positions", []) or [])
        hero_in_position = self._hero_in_position_postflop(hand)
        board_now = list(getattr(context, "board", []) or [])
        full_runout_available = len(board_now) == 5

        # CRITICAL INVARIANT:
        # The external postflop range-builder currently requires a full 5-card
        # board_runout when villain_postflop_players are used. On flop/turn we
        # only know the current board (3 or 4 cards), so sending
        # villain_postflop_players first would create a false solver error even
        # though the hero decision can still be computed from preflop spots.
        #
        # Therefore:
        # - river (5 cards): prefer villain_postflop_players with full runout
        # - flop/turn (3/4 cards): skip that path and use villain_preflop_spots
        #   directly, avoiding noisy "board_runout должен содержать ровно 5 карт"
        #   errors for otherwise valid runtime spots.
        result = None
        warning_messages: List[str] = []

        if full_runout_available:
            try:
                villain_postflop_players = self._build_villain_postflop_players(hand, street)
                result = _solve_hero_postflop(
                    context,
                    villain_postflop_players=villain_postflop_players,
                    hero_in_position=hero_in_position,
                    trials=6000,
                    seed=42,
                )
            except Exception as exc:
                warning_messages.append(str(exc))

        if result is None:
            try:
                villain_preflop_spots = self._build_villain_preflop_spots(hand)
                result = _solve_hero_postflop(
                    context,
                    villain_preflop_spots=villain_preflop_spots,
                    hero_in_position=hero_in_position,
                    trials=6000,
                    seed=42,
                )
            except Exception as exc2:
                warning_messages.append(str(exc2))
                fallback_ranges = [GENERIC_WIDE_RANGE for _ in villain_positions]
                result = _solve_hero_postflop(
                    context,
                    villain_ranges=fallback_ranges,
                    hero_in_position=hero_in_position,
                    trials=5000,
                    seed=42,
                )

        contract_payload = self._build_contract_payload(context)
        payload = {
            "status": "ok",
            "context_type": "PostflopContext",
            "solver_context": dict(contract_payload),
            "advisor_input": dict(contract_payload),
            "projection_meta": _serialize_for_payload(getattr(context, "meta", {})),
            "result": result,
        }
        if warning_messages:
            payload["warnings"] = warning_messages
        return payload

def build_recommendation(analysis, hand, settings=None):
    """Module-level convenience wrapper used by future pipeline integration."""
    return EngineBridge(settings=settings).build_recommendation(analysis, hand)


__all__ = [
    "ActorFlags",
    "ReplayState",
    "SpotDescription",
    "EngineBridge",
    "build_recommendation",
]
