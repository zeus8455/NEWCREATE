from __future__ import annotations

"""Solver bridge extracted from the external launcher.

This module intentionally contains only the decision/adapter layer that converts
PokerVision `hand` / `action_state` into the decision-layer contracts used by
the external solver files.

UI, Qt rendering, CLI and debug-print code are kept out on purpose.

CRITICAL INVARIANT:
This file must remain a thin adapter between PokerVision state and the
decision-layer contracts. Do not reintroduce UI/Qt/launcher responsibilities
here, otherwise bridge semantics will drift again between the main pipeline and
the external launcher.
"""

from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
import hashlib
import importlib
import json
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence
from time import perf_counter

from .context_projection import ContextProjector

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

PROFILE_STAGE_KEYS = (
    "build_solver_contract_preview",
    "build_villain_postflop_range_sources",
    "solve_hero_postflop",
    "runtime_range_state_serialization",
    "fingerprint_build",
)


def _empty_solver_bridge_timings() -> Dict[str, float]:
    return {key: 0.0 for key in PROFILE_STAGE_KEYS}


def _record_solver_bridge_timing(timings: Optional[Dict[str, float]], key: str, started_at: float) -> None:
    if timings is None:
        return
    elapsed_ms = round((perf_counter() - started_at) * 1000.0, 3)
    timings[key] = round(float(timings.get(key, 0.0)) + elapsed_ms, 3)


def _build_solver_bridge_summary(
    phase: str,
    *,
    timings: Optional[Dict[str, float]] = None,
    status: Optional[str] = None,
    street: Optional[str] = None,
    context_type: Optional[str] = None,
    route_mode: Optional[str] = None,
    runtime_used: Optional[bool] = None,
    reused: Optional[bool] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_timings = _empty_solver_bridge_timings()
    if isinstance(timings, dict):
        for key in PROFILE_STAGE_KEYS:
            try:
                resolved_timings[key] = round(float(timings.get(key, 0.0) or 0.0), 3)
            except Exception:
                resolved_timings[key] = 0.0
    total_ms = round(sum(resolved_timings.values()), 3)
    summary: Dict[str, Any] = {
        "phase": phase,
        "status": status or "unknown",
        "street": street,
        "context_type": context_type,
        "timings_ms": resolved_timings,
        "total_profiled_ms": total_ms,
    }
    if route_mode not in (None, ""):
        summary["route_mode"] = route_mode
    if runtime_used is not None:
        summary["runtime_used"] = bool(runtime_used)
    if reused is not None:
        summary["reused"] = bool(reused)
    if note not in (None, ""):
        summary["note"] = str(note)
    return summary


def _attach_solver_bridge_summary(payload: Any, summary: Dict[str, Any]) -> Any:
    if not isinstance(payload, dict):
        return payload
    updated = dict(payload)
    updated["processing_summary"] = {"solver_bridge": dict(summary)}
    return updated


def _with_solver_bridge_summary(payload: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    payload["processing_summary"] = {"solver_bridge": dict(summary)}
    return payload



def _import_bridge_module(module_name: str) -> ModuleType:
    """Import decision-layer modules with package-safe fallback."""

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

    if last_error is None:  # pragma: no cover
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


def _summarize_range_source_payload(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {"name": None, "source_type": None, "combo_count": 0, "total_weight": 0.0}

    weighted = item.get("weighted_combos") or []
    total_weight = 0.0
    combo_count = 0
    for entry in weighted:
        combo_count += 1
        weight = None
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            try:
                weight = float(entry[1])
            except (TypeError, ValueError):
                weight = None
        elif isinstance(entry, dict):
            try:
                weight = float(entry.get("weight"))
            except (TypeError, ValueError):
                weight = None
        total_weight += float(weight or 0.0)

    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    return {
        "name": item.get("name"),
        "source_type": item.get("source_type"),
        "combo_count": combo_count,
        "total_weight": total_weight,
        "resolved_street": meta.get("resolved_street") or meta.get("street"),
        "villain_pos": meta.get("villain_pos"),
        "villain_action": meta.get("villain_action"),
        "range_build_path": meta.get("range_build_path"),
        "range_contract": meta.get("range_contract"),
    }


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
class FallbackPreflopContext:
    hero_hand: list[str]
    hero_pos: str
    node_type: str
    range_owner: str = "hero"
    opener_pos: Optional[str] = None
    three_bettor_pos: Optional[str] = None
    four_bettor_pos: Optional[str] = None
    limpers: int = 0
    callers: int = 0
    dead_cards: list[str] = field(default_factory=list)
    action_history: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class FallbackPostflopContext:
    hero_hand: list[str]
    board: list[str]
    pot_before_hero: float
    to_call: float = 0.0
    effective_stack: Optional[float] = None
    hero_position: Optional[str] = None
    villain_positions: list[str] = field(default_factory=list)
    line_context: dict[str, object] = field(default_factory=dict)
    dead_cards: list[str] = field(default_factory=list)
    street: Optional[str] = None
    player_count: Optional[int] = None
    meta: dict[str, object] = field(default_factory=dict)


def _resolve_decision_types():
    """Resolve external decision-layer dataclasses with a safe local fallback."""

    try:
        return _import_decision_types()
    except Exception:
        return None, FallbackPostflopContext, FallbackPreflopContext


def _canonical_context_type(context: Any) -> str:
    if context is None:
        return ""
    explicit = getattr(context, "context_type", None)
    if explicit:
        return str(explicit)
    if hasattr(context, "hero_pos") and hasattr(context, "node_type"):
        return "PreflopContext"
    if hasattr(context, "pot_before_hero") and hasattr(context, "board"):
        return "PostflopContext"
    return type(context).__name__


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
        _, _, PreflopContext = _resolve_decision_types()
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

    def __init__(self, settings=None, *, enable_runtime: bool = True, runtime: Any = None) -> None:
        if settings is None:
            try:
                from .config import get_default_settings

                settings = get_default_settings()
            except Exception:
                settings = None
        self.settings = settings
        self.runtime = runtime
        if enable_runtime and self.runtime is None:
            try:
                from .solver_runtime import get_solver_runtime

                self.runtime = get_solver_runtime(settings)
            except Exception:
                self.runtime = None
        self.projector = ContextProjector(
            bridge=self,
            resolve_decision_types=_resolve_decision_types,
            canonical_context_type=_canonical_context_type,
            serializer=_serialize_for_payload,
        )

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
            pos
            for pos in self._ordered_positions(hand)
            if pos != hand.hero_position and not hand.player_states.get(pos, {}).get("is_fold", False)
        ]
        result: List[Dict[str, object]] = []
        for pos in active_non_fold:
            mapped = self._preflop_pos(pos, int(hand.player_count))
            payload = villain_spots.get(mapped)
            if payload is None:
                result.append(
                    {
                        "name": pos,
                        "node_type": "facing_open",
                        "villain_pos": mapped,
                        "villain_action": "call",
                        "opener_pos": self._preflop_pos(hand.hero_position, int(hand.player_count)),
                        "callers": 0,
                        "limpers": 0,
                        "range_owner": "opponent",
                    }
                )
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

    def _build_villain_preflop_range_sources(self, villain_preflop_spots: Sequence[Dict[str, object]]) -> tuple[list[Any], Optional[str]]:
        try:
            module = _import_bridge_module("hero_decision")
            builder = getattr(module, "build_villain_ranges_from_preflop_spots", None)
            if builder is None:
                return [], "build_villain_ranges_from_preflop_spots unavailable"
            return list(builder(villain_preflop_spots)), None
        except Exception as exc:
            return [], str(exc)

    def _build_villain_postflop_range_sources(
        self,
        context: Any,
        villain_postflop_players: Sequence[Dict[str, object]],
        *,
        timings: Optional[Dict[str, float]] = None,
    ) -> tuple[list[Any], Optional[Dict[str, Any]], Optional[str]]:
        started_at = perf_counter()
        try:
            module = _import_bridge_module("hero_decision")
            builder = getattr(module, "build_villain_ranges_from_postflop_players", None)
            if builder is None:
                return [], None, "build_villain_ranges_from_postflop_players unavailable"
            sources, report = builder(
                hero_hand=list(getattr(context, "hero_hand", []) or []),
                board_runout=list(getattr(context, "board", []) or []),
                players=list(villain_postflop_players),
                dead_cards=list(getattr(context, "dead_cards", []) or []),
            )
            return list(sources), dict(report or {}), None
        except Exception as exc:
            return [], None, str(exc)
        finally:
            _record_solver_bridge_timing(timings, "build_villain_postflop_range_sources", started_at)

    def _build_postflop_runtime_routes(
        self,
        hand,
        context: Any,
        street: str,
        *,
        timings: Optional[Dict[str, float]] = None,
    ) -> tuple[list[Dict[str, Any]], list[str]]:
        hero_in_position = self._hero_in_position_postflop(hand)
        routes: list[Dict[str, Any]] = []
        build_warnings: list[str] = []
        board_now = list(getattr(context, "board", []) or [])
        board_count = len(board_now)

        villain_postflop_players = self._build_villain_postflop_players(hand, street)
        if villain_postflop_players:
            villain_sources, report, build_error = self._build_villain_postflop_range_sources(
                context,
                villain_postflop_players,
                timings=timings,
            )
            if build_error is None:
                mode = "postflop_players_full_runout" if board_count == 5 else "postflop_players_partial_board"
                routes.append(
                    {
                        "mode": mode,
                        "payload_kind": "villain_postflop_players",
                        "payload": villain_postflop_players,
                        "solver_kwargs": {"villain_postflop_players": villain_postflop_players},
                        "hero_in_position": hero_in_position,
                        "trials": 6000,
                        "seed": 42,
                        "villain_sources": villain_sources,
                        "range_report": report,
                    }
                )
            else:
                build_warnings.append(f"postflop_players_build_failed: {build_error}")

        villain_preflop_spots = self._build_villain_preflop_spots(hand)
        if villain_preflop_spots:
            villain_sources, build_error = self._build_villain_preflop_range_sources(villain_preflop_spots)
            if build_error is not None:
                build_warnings.append(f"villain_preflop_sources_build_failed: {build_error}")
            routes.append(
                {
                    "mode": "villain_preflop_spots",
                    "payload_kind": "villain_preflop_spots",
                    "payload": villain_preflop_spots,
                    "solver_kwargs": {"villain_preflop_spots": villain_preflop_spots},
                    "hero_in_position": hero_in_position,
                    "trials": 6000,
                    "seed": 42,
                    "villain_sources": villain_sources,
                    "range_report": None,
                }
            )

        villain_positions = list(getattr(context, "villain_positions", []) or [])
        fallback_ranges = [GENERIC_WIDE_RANGE for _ in villain_positions]
        routes.append(
            {
                "mode": "generic_wide_range_fallback",
                "payload_kind": "villain_ranges",
                "payload": fallback_ranges,
                "solver_kwargs": {"villain_ranges": fallback_ranges},
                "hero_in_position": hero_in_position,
                "trials": 5000,
                "seed": 42,
                "villain_sources": [],
                "range_report": None,
            }
        )
        return routes, build_warnings

    def _build_postflop_runtime_range_state(
        self,
        route: Dict[str, Any],
        warnings: Optional[Sequence[str]] = None,
        *,
        timings: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        started_at = perf_counter()
        state: Dict[str, Any] = {
            "range_build_mode": str(route.get("mode") or ""),
            "payload_kind": str(route.get("payload_kind") or ""),
            "hero_in_position": bool(route.get("hero_in_position", True)),
            "villain_sources": _serialize_for_payload(route.get("villain_sources") or []),
        }
        report = route.get("range_report")
        if isinstance(report, dict) and report:
            state["range_contract"] = report.get("range_contract")
            if report.get("resolved_street") is not None:
                state["resolved_street"] = report.get("resolved_street")
            if report.get("board") is not None:
                state["board"] = _serialize_for_payload(report.get("board"))
            if report.get("board_runout") is not None:
                state["board_runout"] = _serialize_for_payload(report.get("board_runout"))
            serialized_reports = _serialize_for_payload(report.get("villain_reports") or [])
            state["villain_reports"] = serialized_reports
            state["villain_range_reports"] = serialized_reports
            state["villain_sources_summary"] = [
                _summarize_range_source_payload(item)
                for item in state.get("villain_sources", [])
                if isinstance(item, dict)
            ]
        payload_kind = str(route.get("payload_kind") or "")
        payload = route.get("payload")
        if payload_kind and payload is not None:
            state[payload_kind] = _serialize_for_payload(payload)
        if warnings:
            state["warnings"] = [str(item) for item in warnings if str(item)]
        _record_solver_bridge_timing(timings, "runtime_range_state_serialization", started_at)
        return state

    def _build_enriched_postflop_solver_input(
        self,
        contract_payload: Dict[str, Any],
        route: Optional[Dict[str, Any]],
        warnings: Optional[Sequence[str]] = None,
        *,
        timings: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        solver_input = dict(contract_payload)
        if route is not None:
            solver_input["runtime_range_state"] = self._build_postflop_runtime_range_state(
                route,
                warnings,
                timings=timings,
            )
        elif warnings:
            solver_input["runtime_range_state"] = {"warnings": [str(item) for item in warnings if str(item)]}
        return solver_input

    def _build_fingerprint_from_solver_input(
        self,
        solver_input: Dict[str, Any],
        *,
        context_type: Optional[str] = None,
        timings: Optional[Dict[str, float]] = None,
    ) -> str:
        started_at = perf_counter()
        canonical_payload = {
            "context_type": context_type or solver_input.get("context_type"),
            "solver_input": solver_input,
        }
        raw = json.dumps(
            _serialize_for_payload(canonical_payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        fingerprint = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        _record_solver_bridge_timing(timings, "fingerprint_build", started_at)
        return fingerprint

    # ------------------------------
    # preview construction
    # ------------------------------
    def build_recommendation_preview(self, analysis, hand):
        """Build canonical solver inputs without running the solver.

        CRITICAL INVARIANT:
        Reuse/dedup decisions must be based on the exact same normalized contract
        that would be sent into the solver layer. Do not invent a separate
        pipeline-only heuristic signature here, otherwise repeated frames can look
        "equal" to the cache while producing a different advisor/solver input.
        """

        timings = _empty_solver_bridge_timings()
        started_total = perf_counter()
        street: Optional[str] = None
        context_type: Optional[str] = None
        route_mode: Optional[str] = None

        if hand is None:
            payload = {"status": "not_run", "reason": "no_hand", "fingerprint": ""}
            timings["build_solver_contract_preview"] = round((perf_counter() - started_total) * 1000.0, 3)
            return _attach_solver_bridge_summary(
                payload,
                _build_solver_bridge_summary(
                    "preview",
                    timings=timings,
                    status="not_run",
                    street=street,
                    context_type=context_type,
                    route_mode=route_mode,
                    runtime_used=False,
                    note="no_hand",
                ),
            )

        hero_cards = list(getattr(hand, "hero_cards", None) or getattr(analysis, "hero_cards", None) or [])
        if len(hero_cards) != 2:
            payload = {"status": "not_run", "reason": "hero_cards_not_exactly_two", "fingerprint": ""}
            timings["build_solver_contract_preview"] = round((perf_counter() - started_total) * 1000.0, 3)
            return _attach_solver_bridge_summary(
                payload,
                _build_solver_bridge_summary(
                    "preview",
                    timings=timings,
                    status="not_run",
                    street=street,
                    context_type=context_type,
                    route_mode=route_mode,
                    runtime_used=False,
                    note="hero_cards_not_exactly_two",
                ),
            )

        street_state = getattr(hand, "street_state", {}) if hand is not None else {}
        street = str(street_state.get("current_street") or getattr(analysis, "street", None) or "preflop").lower()
        if street == "preflop":
            context = self.build_preflop_context(analysis, hand)
        else:
            context = self.build_postflop_context(analysis, hand, street)
        if context is None:
            payload = {"status": "not_run", "reason": f"{street}_context_unavailable", "fingerprint": ""}
            timings["build_solver_contract_preview"] = round((perf_counter() - started_total) * 1000.0, 3)
            return _attach_solver_bridge_summary(
                payload,
                _build_solver_bridge_summary(
                    "preview",
                    timings=timings,
                    status="not_run",
                    street=street,
                    context_type=context_type,
                    route_mode=route_mode,
                    runtime_used=False,
                    note=f"{street}_context_unavailable",
                ),
            )

        contract_payload = self._build_contract_payload(context)
        context_type = _canonical_context_type(context)
        advisor_input = dict(contract_payload)
        solver_input = dict(contract_payload)
        preview_warnings: list[str] = []
        if street in POSTFLOP_STREETS and context_type == "PostflopContext":
            routes, build_warnings = self._build_postflop_runtime_routes(hand, context, street, timings=timings)
            preview_warnings.extend(build_warnings)
            selected_route = routes[0] if routes else None
            route_mode = str((selected_route or {}).get("mode") or "") or None
            solver_input = self._build_enriched_postflop_solver_input(
                contract_payload,
                selected_route,
                preview_warnings,
                timings=timings,
            )
            advisor_input = dict(solver_input)
        fingerprint = self._build_fingerprint_from_solver_input(
            solver_input,
            context_type=context_type,
            timings=timings,
        )
        payload = {
            "status": "ready",
            "context_type": context_type,
            "solver_context": dict(contract_payload),
            "advisor_input": dict(advisor_input),
            "solver_input": dict(solver_input),
            "projection_meta": _serialize_for_payload(getattr(context, "meta", {})),
            "fingerprint": fingerprint,
        }
        if preview_warnings:
            payload["warnings"] = [str(item) for item in preview_warnings if str(item)]
        timings["build_solver_contract_preview"] = round((perf_counter() - started_total) * 1000.0, 3)
        return _attach_solver_bridge_summary(
            payload,
            _build_solver_bridge_summary(
                "preview",
                timings=timings,
                status="ready",
                street=street,
                context_type=context_type,
                route_mode=route_mode,
                runtime_used=False,
            ),
        )

    # ------------------------------
    # recommendation construction
    # ------------------------------
    def _append_runtime_warning(self, payload: Any, message: str) -> Any:
        if not isinstance(payload, dict):
            return payload
        updated = dict(payload)
        warnings = list(updated.get("warnings") or [])
        warnings.append(str(message))
        updated["warnings"] = warnings
        return updated

    def _should_use_runtime(self, street: str) -> bool:
        if self.runtime is None:
            return False
        if street in POSTFLOP_STREETS:
            return True
        if street == "preflop":
            return bool(self._setting("solver_runtime_use_for_preflop", False))
        return False

    def _build_recommendation_inline(self, analysis, hand):
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

    def build_recommendation(self, analysis, hand):
        if hand is None:
            payload = {"status": "not_run", "result": None, "reason": "no_hand"}
            return _attach_solver_bridge_summary(
                payload,
                _build_solver_bridge_summary("recommendation", status="not_run", runtime_used=False, note="no_hand"),
            )
        street_state = getattr(hand, "street_state", {}) if hand is not None else {}
        street = str(street_state.get("current_street") or getattr(analysis, "street", None) or "preflop").lower()
        if self._should_use_runtime(street):
            try:
                return self.runtime.compute_postflop_recommendation(analysis, hand)
            except Exception as exc:
                payload = self._build_recommendation_inline(analysis, hand)
                payload = self._append_runtime_warning(payload, f"solver_runtime_fallback: {exc}")
                summary = (((payload.get("processing_summary") or {}).get("solver_bridge") or {}) if isinstance(payload, dict) else {})
                if isinstance(summary, dict):
                    summary = dict(summary)
                    summary["runtime_used"] = False
                    summary["note"] = f"solver_runtime_fallback: {exc}"
                    payload["processing_summary"] = {"solver_bridge": summary}
                return payload
        return self._build_recommendation_inline(analysis, hand)

    def build_preflop_context(self, analysis, hand):
        return self.projector.build_preflop_context(analysis, hand)

    def build_postflop_context(self, analysis, hand, street: Optional[str] = None):
        return self.projector.build_postflop_context(analysis, hand, street)

    def _build_contract_payload(self, context: Any) -> Dict[str, Any]:
        return self.projector.build_contract_payload(context)

    def _build_preflop_recommendation(self, analysis, hand, hero_cards: List[str]):
        context = self.build_preflop_context(analysis, hand)
        if context is None:
            payload = {"status": "not_run", "result": None, "reason": "preflop_context_unavailable"}
            return _attach_solver_bridge_summary(
                payload,
                _build_solver_bridge_summary(
                    "recommendation",
                    status="not_run",
                    street="preflop",
                    context_type="PreflopContext",
                    runtime_used=False,
                    note="preflop_context_unavailable",
                ),
            )
        contract_payload = self._build_contract_payload(context)
        try:
            result = _solve_hero_preflop(context)
        except Exception as exc:
            payload = {
                "status": "solver_unavailable",
                "context_type": "PreflopContext",
                "solver_context": dict(contract_payload),
                "advisor_input": dict(contract_payload),
                "solver_input": dict(contract_payload),
                "solver_output": {"result": None, "context_type": "PreflopContext"},
                "projection_meta": _serialize_for_payload(getattr(context, "meta", {})),
                "warnings": [str(exc)],
                "result": None,
            }
            return _attach_solver_bridge_summary(
                payload,
                _build_solver_bridge_summary(
                    "recommendation",
                    status="solver_unavailable",
                    street="preflop",
                    context_type="PreflopContext",
                    runtime_used=False,
                    note=str(exc),
                ),
            )
        payload = {
            "status": "ok",
            "context_type": "PreflopContext",
            "solver_context": dict(contract_payload),
            "advisor_input": dict(contract_payload),
            "solver_input": dict(contract_payload),
            "solver_output": {"result": _serialize_for_payload(result), "context_type": "PreflopContext"},
            "projection_meta": _serialize_for_payload(getattr(context, "meta", {})),
            "result": result,
        }
        return _attach_solver_bridge_summary(
            payload,
            _build_solver_bridge_summary(
                "recommendation",
                status="ok",
                street="preflop",
                context_type="PreflopContext",
                runtime_used=False,
            ),
        )

    def _can_use_postflop_line_builder(self, board_cards: Sequence[str], street: str) -> bool:
        return self.projector.can_use_postflop_line_builder(board_cards, street)

    def _build_postflop_line_context(self, hand, street: str) -> Dict[str, object]:
        current_actions = [a for a in self._street_actions(hand, street)]
        preflop_actions = [a for a in self._street_actions(hand, "preflop")]
        hero_last_aggressor_preflop = False
        for item in reversed(preflop_actions):
            if str(item.get("action", "")).upper() in {"OPEN", "RAISE"}:
                hero_last_aggressor_preflop = str(item.get("position", "")) == hand.hero_position
                break
        current_aggressive = [a for a in current_actions if str(a.get("action", "")).upper() in {"BET", "RAISE", "OPEN"}]
        to_call = self._to_call(hand)
        villain_positions = [
            pos
            for pos in self._ordered_positions(hand)
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
        timings = _empty_solver_bridge_timings()
        context = self.build_postflop_context(analysis, hand, street)
        if context is None:
            payload = {"status": "not_run", "result": None, "reason": "postflop_context_unavailable"}
            return _attach_solver_bridge_summary(
                payload,
                _build_solver_bridge_summary(
                    "recommendation",
                    timings=timings,
                    status="not_run",
                    street=street,
                    context_type="PostflopContext",
                    runtime_used=self._should_use_runtime(street),
                    note="postflop_context_unavailable",
                ),
            )
        route_warnings: List[str] = []
        routes, build_warnings = self._build_postflop_runtime_routes(hand, context, street, timings=timings)
        route_warnings.extend(build_warnings)
        result = None
        warning_messages: List[str] = []
        selected_route: Optional[Dict[str, Any]] = routes[0] if routes else None
        for route in routes:
            solve_started_at = perf_counter()
            try:
                result = _solve_hero_postflop(
                    context,
                    hero_in_position=bool(route.get("hero_in_position", True)),
                    trials=int(route.get("trials", 6000) or 6000),
                    seed=int(route.get("seed", 42) or 42),
                    **dict(route.get("solver_kwargs") or {}),
                )
                _record_solver_bridge_timing(timings, "solve_hero_postflop", solve_started_at)
                selected_route = route
                break
            except Exception as exc:
                _record_solver_bridge_timing(timings, "solve_hero_postflop", solve_started_at)
                warning_messages.append(str(exc))
                selected_route = route
        contract_payload = self._build_contract_payload(context)
        combined_warnings = [str(item) for item in [*route_warnings, *warning_messages] if str(item)]
        advisor_input = self._build_enriched_postflop_solver_input(
            contract_payload,
            selected_route,
            route_warnings,
            timings=timings,
        )
        solver_input = dict(advisor_input)
        solver_output = {
            "result": _serialize_for_payload(result),
            "context_type": "PostflopContext",
        }
        if selected_route is not None:
            solver_output["runtime_range_state"] = self._build_postflop_runtime_range_state(
                selected_route,
                combined_warnings,
                timings=timings,
            )
        payload = {
            "status": "ok" if result is not None else "solver_unavailable",
            "context_type": "PostflopContext",
            "solver_context": dict(contract_payload),
            "advisor_input": dict(advisor_input),
            "solver_input": dict(solver_input),
            "solver_output": solver_output,
            "projection_meta": _serialize_for_payload(getattr(context, "meta", {})),
            "result": result,
        }
        if combined_warnings:
            payload["warnings"] = combined_warnings
        return _attach_solver_bridge_summary(
            payload,
            _build_solver_bridge_summary(
                "recommendation",
                timings=timings,
                status=payload.get("status"),
                street=street,
                context_type="PostflopContext",
                route_mode=str((selected_route or {}).get("mode") or "") or None,
                runtime_used=self._should_use_runtime(street),
            ),
        )


def build_recommendation(analysis, hand, settings=None):
    """Module-level convenience wrapper used by pipeline integration."""

    return EngineBridge(settings=settings).build_recommendation(analysis, hand)


def build_solver_contract_preview(analysis, hand, settings=None):
    """Build normalized solver/advisor inputs without executing the solver."""

    return EngineBridge(settings=settings).build_recommendation_preview(analysis, hand)


__all__ = [
    "ActorFlags",
    "ReplayState",
    "SpotDescription",
    "EngineBridge",
    "build_recommendation",
    "build_solver_contract_preview",
]
