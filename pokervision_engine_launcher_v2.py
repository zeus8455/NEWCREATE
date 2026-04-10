from __future__ import annotations

"""
Единый запуск PokerVision + Engine_equity_range_postflop.

Положить файл сюда:
    C:/PokerAI/PokerVision/Python_PY/pokervision_engine_launcher_v2.py

Запуск:
    cd C:/PokerAI/PokerVision/Python_PY
    C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe pokervision_engine_launcher_v2.py --real

Что делает файл:
- запускает PokerVision как главный цикл приложения;
- по триггеру ActiveHero получает текущее состояние руки из PokerVision;
- на префлопе всегда сначала строит preflop spot и сверяет руку Hero с chart/range первого проекта;
- на постфлопе строит PostflopContext, восстанавливает диапазоны оппонентов,
  пытается сузить их по линии действий и считает EV-ветки;
- показывает отдельное окно со столом и отдельное окно с подробным engine-analysis.
"""

import argparse
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CURRENT_DIR = Path(__file__).resolve().parent
POKERVISION_ROOT = CURRENT_DIR
ENGINE_ROOT = Path(r"C:\PokerAI\Engine_equity_range_postflop")

for path in (POKERVISION_ROOT, ENGINE_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


from pokervision.capture import MockFrameSource, ScreenFrameSource
from pokervision.config import get_default_settings
from pokervision.detectors import MockDetectorBackend, YoloDetectorBackend
from pokervision.hand_state import HandStateManager
from pokervision.pipeline import PokerVisionPipeline
from pokervision.storage import StorageManager
from pokervision.ui_bridge import SharedState
from pokervision.visualizer import DebugMonitorWindow
from pokervision.card_renderer import render_card, render_card_back

from hero_decision import (
    format_hero_decision_report,
    solve_hero_postflop,
    solve_hero_preflop,
)
from decision_types import HeroDecision, PostflopContext, PreflopContext

try:
    from postflop_advisor import format_multiway_postflop_report
except Exception:
    format_multiway_postflop_report = None

try:
    from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    QtCore = QtGui = QtWidgets = None
    cv2 = None
    np = None


CANONICAL_RING = {
    2: ["BTN", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["BTN", "SB", "BB", "CO"],
    5: ["BTN", "SB", "BB", "UTG", "CO"],
    6: ["BTN", "SB", "BB", "UTG", "MP", "CO"],
}

DISPLAY_SLOTS = {
    2: [(0.50, 0.84), (0.50, 0.18)],
    3: [(0.50, 0.84), (0.18, 0.46), (0.82, 0.22)],
    4: [(0.50, 0.84), (0.16, 0.64), (0.18, 0.24), (0.84, 0.28)],
    5: [(0.50, 0.84), (0.16, 0.68), (0.18, 0.28), (0.50, 0.14), (0.84, 0.30)],
    6: [(0.50, 0.84), (0.16, 0.70), (0.18, 0.30), (0.50, 0.13), (0.82, 0.30), (0.84, 0.70)],
}

GENERIC_WIDE_RANGE = (
    "22+ A2s+ K2s+ Q2s+ J2s+ T2s+ 92s+ 82s+ 72s+ 62s+ 52s+ 42s+ 32s "
    "A2o+ K2o+ Q2o+ J2o+ T2o+ 92o+ 82o+ 72o+ 62o+ 52o+"
)

STREET_ORDER = ["preflop", "flop", "turn", "river"]
POSTFLOP_STREETS = ["flop", "turn", "river"]


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

    def to_preflop_context(self, hero_hand: List[str], *, meta: Optional[Dict[str, object]] = None) -> PreflopContext:
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
    def __init__(self, settings) -> None:
        self.settings = settings

    # ------------------------------
    # базовые хелперы
    # ------------------------------
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

    def _display_action(self, decision: HeroDecision) -> str:
        action = str(decision.engine_action).upper()
        if decision.size_pct is not None:
            return f"{action} {float(decision.size_pct):.0f}%"
        if decision.amount_to is not None:
            return f"{action} to {float(decision.amount_to):.2f}"
        return action

    def _normalize_effective_stack(self, stack_bb: Optional[float]) -> Optional[float]:
        if stack_bb is None:
            return None
        value = float(stack_bb)
        if not self.settings.normalize_short_stack_to_40bb:
            return value
        if self.settings.short_stack_min_inclusive_bb <= value < self.settings.short_stack_max_exclusive_bb:
            return float(self.settings.short_stack_forced_value_bb)
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
            else:
                payload["engine_action"] = "raise"
        if payload.get("final_contribution_bb") is None and payload.get("amount_bb") is not None:
            payload["final_contribution_bb"] = payload.get("amount_bb")
        if payload.get("action") in {None, ""} and semantic:
            payload["action"] = semantic.upper()
        return payload

    def _street_actions(self, hand, street: str) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        for item in list(hand.actions_log or []):
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

    def _map_action_for_spot(self, spot: SpotDescription, action_payload: Dict[str, Any]) -> Optional[str]:
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
                action_name = str(action.get("action", "")).upper()
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
    # формирование решения
    # ------------------------------
    def build_recommendation(self, analysis, hand) -> Optional[HeroDecision]:
        if hand is None:
            return None
        hero_cards = list(hand.hero_cards or analysis.hero_cards or [])
        if len(hero_cards) != 2:
            return None

        street = str(hand.street_state.get("current_street") or analysis.street or "preflop").lower()
        if street == "preflop":
            return self._build_preflop_recommendation(hand, hero_cards)
        return self._build_postflop_recommendation(hand, hero_cards, street)

    def _build_preflop_recommendation(self, hand, hero_cards: List[str]) -> Optional[HeroDecision]:
        spot = self._build_hero_preflop_spot(hand)
        context = spot.to_preflop_context(
            hero_cards,
            meta={
                "source": "pokervision",
                "hand_id": hand.hand_id,
                "hero_original_position": hand.hero_position,
                "actions_seen": self._street_actions(hand, "preflop"),
            },
        )
        return solve_hero_preflop(context)

    def _build_postflop_recommendation(self, hand, hero_cards: List[str], street: str) -> Optional[HeroDecision]:
        board = list(hand.board_cards or [])
        if len(board) not in {3, 4, 5}:
            return None

        villain_positions = [
            pos for pos in self._ordered_positions(hand)
            if pos != hand.hero_position and not hand.player_states.get(pos, {}).get("is_fold", False)
        ]
        if not villain_positions:
            return None

        line_context = self._build_postflop_line_context(hand, street)
        context = PostflopContext(
            hero_hand=list(hero_cards),
            board=list(board),
            pot_before_hero=self._pot_before_hero(hand),
            to_call=self._to_call(hand),
            effective_stack=self._effective_stack(hand),
            hero_position=hand.hero_position,
            villain_positions=list(villain_positions),
            line_context=line_context,
            dead_cards=[],
            street=street,
            player_count=int(hand.player_count),
            meta={
                "source": "pokervision",
                "hand_id": hand.hand_id,
            },
        )

        villain_postflop_players = self._build_villain_postflop_players(hand, street)
        try:
            return solve_hero_postflop(
                context,
                villain_postflop_players=villain_postflop_players,
                hero_in_position=self._hero_in_position_postflop(hand),
                trials=6000,
                seed=42,
            )
        except Exception:
            try:
                villain_preflop_spots = self._build_villain_preflop_spots(hand)
                return solve_hero_postflop(
                    context,
                    villain_preflop_spots=villain_preflop_spots,
                    hero_in_position=self._hero_in_position_postflop(hand),
                    trials=6000,
                    seed=42,
                )
            except Exception:
                fallback_ranges = [GENERIC_WIDE_RANGE for _ in villain_positions]
                return solve_hero_postflop(
                    context,
                    villain_ranges=fallback_ranges,
                    hero_in_position=self._hero_in_position_postflop(hand),
                    trials=5000,
                    seed=42,
                )

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

    # ------------------------------
    # данные для UI / text panel
    # ------------------------------
    def _format_range_source(self, source) -> str:
        if source is None:
            return "<none>"
        raw = getattr(source, "normalized_expr", None) or getattr(source, "raw_expr", None) or "<weighted/direct>"
        combo_count = getattr(source, "combo_count", None)
        if combo_count is None and hasattr(source, "weighted_combos"):
            combo_count = len(getattr(source, "weighted_combos") or [])
        return f"{raw} | combos={combo_count if combo_count is not None else '?'}"

    def _resolve_preflop_display_context(self, decision: HeroDecision, hand=None) -> Dict[str, str]:
        debug = dict(decision.debug or {})

        projection_node_type = None
        advisor_node_type = debug.get("node_type")
        advisor_mapping_reason = None
        projection_description = None
        advisor_description = None

        pre = getattr(decision, "preflop", None)
        if pre is not None and isinstance(getattr(pre, "meta", None), dict):
            advisor_description = pre.meta.get("description")

        if hand is not None:
            action_state = getattr(hand, "action_state", None) or {}
            if isinstance(action_state, dict):
                projection_node_type = (
                    action_state.get("projection_node_type")
                    or action_state.get("node_type_preview")
                    or action_state.get("node_type")
                    or projection_node_type
                )
                advisor_node_type = action_state.get("advisor_node_type") or advisor_node_type
                advisor_mapping_reason = action_state.get("advisor_mapping_reason") or advisor_mapping_reason
                projection_description = action_state.get("projection_description") or projection_description

            advisor_input = getattr(hand, "advisor_input", None) or {}
            if isinstance(advisor_input, dict):
                advisor_node_type = advisor_input.get("node_type") or advisor_node_type
                meta = advisor_input.get("meta") or {}
                if isinstance(meta, dict):
                    projection_node_type = meta.get("projection_node_type") or projection_node_type
                    advisor_mapping_reason = (
                        meta.get("advisor_mapping_reason")
                        or meta.get("mapping_reason")
                        or advisor_mapping_reason
                    )
                    projection_description = meta.get("projection_description") or projection_description

            reconstructed_preflop = getattr(hand, "reconstructed_preflop", None) or {}
            if isinstance(reconstructed_preflop, dict):
                projection_node_type = reconstructed_preflop.get("node_type") or projection_node_type
                projection_description = reconstructed_preflop.get("description") or projection_description

        display_node_type = projection_node_type or advisor_node_type or "-"
        return {
            "display_node_type": str(display_node_type),
            "projection_node_type": str(projection_node_type) if projection_node_type else "",
            "advisor_node_type": str(advisor_node_type) if advisor_node_type else "",
            "advisor_mapping_reason": str(advisor_mapping_reason) if advisor_mapping_reason else "",
            "projection_description": str(projection_description) if projection_description else "",
            "advisor_description": str(advisor_description) if advisor_description else "",
        }

    def _build_preflop_analysis_text(self, decision: HeroDecision, hand=None) -> str:
        lines: List[str] = []
        pre = decision.preflop

        lines.append("=== PREFLOP ANALYSIS ===")
        lines.append(f"Recommendation: {self._display_action(decision)}")
        lines.append(f"Reason: {decision.reason}")
        if decision.confidence is not None:
            lines.append(f"Confidence: {float(decision.confidence):.2f}")
        lines.append("")

        if hand is not None:
            lines.append(f"Hand ID: {hand.hand_id}")
            lines.append(f"Hero position (vision): {hand.hero_position}")
            lines.append(f"Hero cards: {' '.join(hand.hero_cards)}")
            lines.append("")

        if pre is None:
            lines.append("Preflop decision payload is empty.")
            return "\n".join(lines)

        debug = dict(decision.debug or {})
        display_ctx = self._resolve_preflop_display_context(decision, hand=hand)

        lines.append(f"Tree / node_type: {display_ctx['display_node_type']}")
        lines.append(f"Range owner: {debug.get('range_owner', '-')}")
        if (
            display_ctx["projection_node_type"]
            and display_ctx["advisor_node_type"]
            and display_ctx["projection_node_type"] != display_ctx["advisor_node_type"]
        ):
            lines.append(f"Advisor mapped node: {display_ctx['advisor_node_type']}")
            if display_ctx["advisor_mapping_reason"]:
                lines.append(f"Advisor mapping reason: {display_ctx['advisor_mapping_reason']}")

        lines.append(f"Hero hand class: {pre.hand_class}")

        description = display_ctx["projection_description"] or display_ctx["advisor_description"]
        if description:
            lines.append(f"Description: {description}")
        if (
            display_ctx["projection_description"]
            and display_ctx["advisor_description"]
            and display_ctx["projection_description"] != display_ctx["advisor_description"]
        ):
            lines.append(f"Advisor chart description: {display_ctx['advisor_description']}")

        lines.append(f"Matching actions: {', '.join(pre.matching_actions) if pre.matching_actions else '-'}")
        lines.append(f"Chosen action: {pre.action}")
        lines.append(f"Selected range expr: {pre.selected_range_expr or '-'}")
        if pre.fallback_reason:
            lines.append(f"Fallback reason: {pre.fallback_reason}")

        lines.append("")
        lines.append("Action map / chart branches:")
        if pre.action_map:
            for action_name, expr in pre.action_map.items():
                lines.append(f" {action_name:<10} -> {expr}")
        else:
            lines.append(" <empty>")

        lines.append("")
        lines.append(f"Chosen branch range source: {self._format_range_source(pre.range_source)}")
        lines.append("")
        lines.append(
            "Note: preflop layer is chart/tree based. EV branches are not calculated here; "
            "the file shows the exact node and chart branch used."
        )
        return "\n".join(lines)

    def _append_postflop_range_narrowing(self, lines: List[str], decision: HeroDecision) -> None:
        meta = {}
        if decision.postflop and isinstance(decision.postflop.meta, dict):
            meta = dict(decision.postflop.meta.get("postflop_report_meta") or {})
        if not meta:
            meta = dict(decision.debug.get("report_meta") or {})
        villain_reports = meta.get("villain_reports") if isinstance(meta, dict) else None
        if not villain_reports:
            lines.append("=== RANGE NARROWING ===")
            lines.append("No explicit narrowing report available.")
            return

        lines.append("=== RANGE NARROWING ===")
        if format_multiway_postflop_report is not None:
            try:
                lines.append(format_multiway_postflop_report(meta))
                return
            except Exception:
                pass

        for item in villain_reports:
            report = item.get("report", {}) if isinstance(item, dict) else {}
            lines.append(f"[{item.get('name', 'Villain')}] pos={item.get('villain_pos')} preflop={item.get('villain_action')}")
            lines.append(f"  start expr: {report.get('starting_range_expr', '-')}")
            start = report.get("starting_range", {}) if isinstance(report, dict) else {}
            if isinstance(start, dict):
                lines.append(
                    f"  start combos={start.get('combo_count', '?')} classes={start.get('class_count', '?')}"
                )
            for step in report.get("steps", []) if isinstance(report, dict) else []:
                after = step.get("range_after", {}) if isinstance(step, dict) else {}
                lines.append(
                    f"  {str(step.get('street', '')).upper()} {step.get('action', '')} {float(step.get('bet_pct_pot', 0.0)):.1f}% "
                    f"-> after {after.get('combo_count', '?')} combos"
                )
                reason = step.get("reason")
                if reason:
                    lines.append(f"    rule: {reason}")

    def _build_postflop_analysis_text(self, decision: HeroDecision, hand=None) -> str:
        lines: List[str] = []
        lines.append("=== POSTFLOP ANALYSIS ===")
        lines.append(f"Recommendation: {self._display_action(decision)}")
        lines.append(f"Reason: {decision.reason}")
        if decision.confidence is not None:
            lines.append(f"Confidence: {float(decision.confidence):.2f}")
        lines.append("")
        if hand is not None:
            lines.append(f"Hand ID: {hand.hand_id}")
            lines.append(f"Hero cards: {' '.join(hand.hero_cards)}")
            lines.append(f"Board: {' '.join(hand.board_cards) if hand.board_cards else '<empty>'}")
            lines.append(f"Vision current pot: {self._pot_before_hero(hand):.2f}")
            lines.append(f"Vision to call: {self._to_call(hand):.2f}")
            eff = self._effective_stack(hand)
            lines.append(f"Effective stack: {eff:.2f}" if eff is not None else "Effective stack: -")
            lines.append("")

        report = {}
        if decision.postflop and isinstance(decision.postflop.report, dict):
            report = dict(decision.postflop.report)

        if report:
            try:
                lines.append(format_hero_decision_report(report))
            except Exception:
                lines.append("Unable to format hero decision report; raw report follows.")
                lines.append(str(report))
        else:
            lines.append("Postflop report is empty.")

        lines.append("")
        self._append_postflop_range_narrowing(lines, decision)
        return "\n".join(lines)

    def build_analysis_text(self, decision: Optional[HeroDecision], hand=None) -> str:
        if decision is None:
            return "No decision available yet. Waiting for stable hand state / hero cards / board."
        if str(decision.street).lower() == "preflop":
            return self._build_preflop_analysis_text(decision, hand=hand)
        return self._build_postflop_analysis_text(decision, hand=hand)

    def _resolve_state_analysis_context(self, render_state: dict, hand=None) -> Dict[str, Any]:
        panel = render_state.get("analysis_panel") or {}
        range_debug = render_state.get("range_debug") or {}
        debug_payload = render_state.get("hero_decision_debug") or {}
        debug = debug_payload.get("debug") if isinstance(debug_payload, dict) else {}
        if not isinstance(debug, dict):
            debug = {}
        meta = range_debug.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        return {
            "street": str(render_state.get("street") or (getattr(hand, "street_state", {}) or {}).get("current_street") or "-"),
            "recommended_action": str(render_state.get("recommended_action") or "NO DECISION"),
            "reason": str(render_state.get("decision_reason") or ""),
            "confidence": render_state.get("decision_confidence"),
            "node_type": str(panel.get("node_type") or render_state.get("node_type") or meta.get("projection_node_type") or "-"),
            "projection_node_type": str(panel.get("projection_node_type") or render_state.get("node_type") or meta.get("projection_node_type") or ""),
            "advisor_node_type": str(panel.get("advisor_node_type") or meta.get("advisor_node_type") or debug.get("node_type") or ""),
            "advisor_mapping_reason": str(panel.get("advisor_mapping_reason") or meta.get("advisor_mapping_reason") or ""),
            "hero_position": str(render_state.get("hero_position") or getattr(hand, "hero_position", "-")),
            "hero_cards": list(render_state.get("hero_cards") or getattr(hand, "hero_cards", []) or []),
            "range_owner": str(debug.get("range_owner") or "hero"),
            "hand_class": str(range_debug.get("hand_class") or "-"),
            "description": str(meta.get("description") or ""),
            "matching_actions": list(range_debug.get("matching_actions") or []),
            "chosen_action": str(range_debug.get("action") or "-"),
            "selected_range_expr": str(range_debug.get("selected_range_expr") or "-"),
            "action_map": dict(range_debug.get("action_map") or {}),
            "fallback_reason": range_debug.get("fallback_reason"),
        }

    def build_analysis_text_from_render_state(self, render_state: dict, hand=None) -> str:
        ctx = self._resolve_state_analysis_context(render_state, hand=hand)
        lines: List[str] = []
        street = str(ctx["street"]).lower()
        lines.append("=== PREFLOP ANALYSIS ===" if street == "preflop" else "=== POSTFLOP ANALYSIS ===")
        lines.append(f"Recommendation: {ctx['recommended_action']}")
        lines.append(f"Reason: {ctx['reason']}")
        if ctx["confidence"] is not None:
            lines.append(f"Confidence: {float(ctx['confidence']):.2f}")
        lines.append("")
        if hand is not None:
            lines.append(f"Hand ID: {hand.hand_id}")
        lines.append(f"Hero position (vision): {ctx['hero_position']}")
        if ctx["hero_cards"]:
            lines.append(f"Hero cards: {' '.join(ctx['hero_cards'])}")
        lines.append("")
        lines.append(f"Tree / node_type: {ctx['node_type']}")
        if ctx["projection_node_type"] and ctx["advisor_node_type"] and ctx["projection_node_type"] != ctx["advisor_node_type"]:
            lines.append(f"Advisor mapped node: {ctx['advisor_node_type']}")
            if ctx["advisor_mapping_reason"]:
                lines.append(f"Advisor mapping reason: {ctx['advisor_mapping_reason']}")
        lines.append(f"Range owner: {ctx['range_owner']}")
        lines.append(f"Hero hand class: {ctx['hand_class']}")
        if ctx["description"]:
            lines.append(f"Description: {ctx['description']}")
        ma = ctx["matching_actions"]
        lines.append(f"Matching actions: {', '.join(ma) if ma else '-'}")
        lines.append(f"Chosen action: {ctx['chosen_action']}")
        lines.append(f"Selected range expr: {ctx['selected_range_expr']}")
        if ctx["fallback_reason"]:
            lines.append(f"Fallback reason: {ctx['fallback_reason']}")
        if ctx["action_map"]:
            lines.append("")
            lines.append("Action map / chart branches:")
            for action_name, expr in ctx["action_map"].items():
                lines.append(f" {action_name:<10} -> {expr}")
        return "\n".join(lines)

    def format_existing_render_state_for_ui(self, render_state: Optional[dict], hand=None) -> Optional[Dict[str, Any]]:
        if not isinstance(render_state, dict):
            return None
        recommended_action = render_state.get("recommended_action")
        decision_reason = render_state.get("decision_reason")
        if not recommended_action and not decision_reason:
            return None
        title = str(recommended_action or "NO DECISION")
        reason = str(decision_reason or "")
        confidence = render_state.get("decision_confidence")
        street = str(render_state.get("street") or (getattr(hand, "street_state", {}) or {}).get("current_street") or "")
        return {
            "title": title,
            "action": title.lower(),
            "reason": reason,
            "street": street,
            "confidence": confidence,
            "size_pct": render_state.get("recommended_size_pct"),
            "amount_to": render_state.get("recommended_amount_to"),
            "debug": dict((((render_state.get("hero_decision_debug") or {}) if isinstance(render_state.get("hero_decision_debug"), dict) else {}).get("debug") or {})),
            "analysis_text": self.build_analysis_text_from_render_state(render_state, hand=hand),
        }

    def format_for_ui(self, decision: Optional[HeroDecision], hand=None) -> Dict[str, Any]:
        if decision is None:
            return {
                "title": "NO DECISION",
                "action": "—",
                "reason": "Недостаточно данных для решения",
                "street": None,
                "confidence": None,
                "debug": {},
                "analysis_text": self.build_analysis_text(None, hand=hand),
            }

        debug = dict(decision.debug or {})
        title = self._display_action(decision)
        if hand is not None and str(decision.street).lower() == "preflop" and str(decision.engine_action).lower() == "fold":
            title = "FOLD"
        return {
            "title": title,
            "action": str(decision.engine_action),
            "reason": str(decision.reason or ""),
            "street": str(decision.street),
            "confidence": decision.confidence,
            "size_pct": decision.size_pct,
            "amount_to": decision.amount_to,
            "debug": debug,
            "analysis_text": self.build_analysis_text(decision, hand=hand),
        }


class IntegratedTableWindow:  # pragma: no cover
    def __init__(self, shared_state, settings):
        if QtWidgets is None:
            raise RuntimeError("PySide6 is not installed; UI window unavailable")
        self.shared_state = shared_state
        self.settings = settings
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("PokerVision + Engine Recommendation")
        self.window.resize(1280, 900)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.window.update)
        self.timer.start(settings.ui_refresh_ms)
        self.window.paintEvent = self.paintEvent  # type: ignore[assignment]

    def show(self):
        self.window.show()

    def _ordered_positions(self, render_state: dict) -> List[str]:
        explicit = render_state.get("seat_order") or []
        if explicit:
            return [pos for pos in explicit if pos in render_state.get("players", {})]
        player_count = int(render_state.get("player_count", 0) or 0)
        available = [pos for pos in CANONICAL_RING.get(player_count, []) if pos in render_state.get("players", {})]
        hero_position = render_state.get("hero_position")
        if hero_position in available:
            idx = available.index(hero_position)
            return available[idx:] + available[:idx]
        return available

    def _player_brush(self, player_payload, is_hero):
        if player_payload.get("is_fold"):
            return QtGui.QBrush(QtGui.QColor(62, 62, 62))
        if player_payload.get("is_all_in"):
            return QtGui.QBrush(QtGui.QColor(118, 48, 48))
        if is_hero:
            return QtGui.QBrush(QtGui.QColor(176, 148, 58))
        return QtGui.QBrush(QtGui.QColor(52, 52, 52))

    def _draw_badge(self, painter, x: int, y: int, text: str, bg: QtGui.QColor, width: int = 36):
        painter.setPen(QtGui.QPen(QtGui.QColor(250, 250, 250), 1))
        painter.setBrush(QtGui.QBrush(bg))
        painter.drawRoundedRect(x, y, width, 18, 8, 8)
        painter.drawText(QtCore.QRect(x, y, width, 18), QtCore.Qt.AlignmentFlag.AlignCenter, text)

    def _pixmap_from_bgr(self, image: "np.ndarray"):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        return QtGui.QPixmap.fromImage(qimg.copy())

    def _draw_card_pixmap(self, painter, image: "np.ndarray", x: int, y: int):
        painter.drawPixmap(x, y, self._pixmap_from_bgr(image))

    def _player_card_y(self, rect, py: int, box_y: int, box_h: int, card_h: int) -> int:
        if py >= int(rect.height() * 0.58):
            return box_y - card_h - 16
        return box_y + box_h + 12

    def _draw_player(self, painter, rect, pos_name: str, payload: dict, center_xy: Tuple[int, int], blind_text: str | None = None):
        px, py = center_xy
        box_w, box_h = 178, 106
        box_x = px - box_w // 2
        box_y = py - box_h // 2
        is_hero = payload.get("is_hero", False)

        painter.setBrush(self._player_brush(payload, is_hero))
        painter.setPen(QtGui.QPen(QtGui.QColor(245, 245, 245), 2))
        painter.drawRoundedRect(box_x, box_y, box_w, box_h, 14, 14)

        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.drawText(box_x + 10, box_y + 20, pos_name)
        if is_hero:
            self._draw_badge(painter, box_x + box_w - 86, box_y + 8, "HERO", QtGui.QColor(120, 88, 10), width=46)
        if payload.get("is_button"):
            self._draw_badge(painter, box_x + box_w - 36, box_y + 8, "D", QtGui.QColor(70, 110, 170), width=24)
        if blind_text:
            self._draw_badge(painter, box_x + 10, box_y - 20, blind_text, QtGui.QColor(95, 115, 55), width=max(34, 10 + len(blind_text) * 6))

        stack_text = "—"
        if payload.get("stack_bb") is not None:
            stack_text = f"{payload['stack_bb']:.1f} BB"
        elif payload.get("stack_text_raw"):
            stack_text = f"{payload['stack_text_raw']} BB"
        painter.drawText(box_x + 10, box_y + 42, f"Stack: {stack_text}")

        bet_text = "—"
        if payload.get("current_bet_bb") is not None:
            bet_text = f"{payload['current_bet_bb']:.1f} BB"
        elif payload.get("current_bet_raw"):
            bet_text = f"{payload['current_bet_raw']} BB"
        painter.drawText(box_x + 10, box_y + 62, f"Bet: {bet_text}")

        status_parts: List[str] = []
        if payload.get("is_fold"):
            status_parts.append("FOLD")
        if payload.get("is_all_in"):
            status_parts.append("ALL-IN")
        if payload.get("last_action"):
            status_parts.append(str(payload["last_action"]))
        warnings = payload.get("state_warnings", [])
        if warnings:
            status_parts.append(str(warnings[0]))
        status_text = " | ".join(status_parts)[:42]
        if status_text:
            painter.drawText(box_x + 10, box_y + 84, status_text)

        if payload.get("show_card_backs") and not payload.get("is_fold"):
            back = render_card_back()
            card_y = self._player_card_y(rect, py, box_y, box_h, back.shape[0])
            self._draw_card_pixmap(painter, back, box_x + 58, card_y)
            self._draw_card_pixmap(painter, back, box_x + 98, card_y)
        return {"box_x": box_x, "box_y": box_y, "box_w": box_w, "box_h": box_h, "center_x": px, "center_y": py}

    def _draw_recommendation(self, painter, rect, render_state: dict, status: dict):
        rec = status.get("recommendation") or render_state.get("hero_recommendation") or {}
        x = rect.width() - 390
        y = 52
        w = 350
        h = 120

        painter.setPen(QtGui.QPen(QtGui.QColor(235, 235, 235), 2))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(24, 24, 32, 220)))
        painter.drawRoundedRect(x, y, w, h, 14, 14)

        title = str(rec.get("title") or "NO DECISION")
        reason = str(rec.get("reason") or "")
        confidence = rec.get("confidence")

        painter.setPen(QtGui.QColor(255, 222, 120))
        font = painter.font()
        font.setPointSize(14)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(x + 14, y + 28, f"RECOMMEND: {title}")

        font.setPointSize(9)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(225, 225, 225))
        if confidence is not None:
            painter.drawText(x + 14, y + 50, f"Confidence: {float(confidence):.2f}")
        painter.drawText(QtCore.QRect(x + 14, y + 58, w - 28, h - 24), QtCore.Qt.TextWordWrap, reason[:220])

    def paintEvent(self, event):
        painter = QtGui.QPainter(self.window)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.window.rect()
        painter.fillRect(rect, QtGui.QColor(24, 44, 24))

        table_rect = QtCore.QRect(120, 110, rect.width() - 240, rect.height() - 240)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(30, 92, 36)))
        painter.setPen(QtGui.QPen(QtGui.QColor(220, 220, 220), 3))
        painter.drawEllipse(table_rect)

        _frame, render_state, status = self.shared_state.snapshot()
        if not render_state:
            painter.setPen(QtGui.QColor(240, 240, 240))
            painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, "Waiting for render_state")
            return

        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.drawText(20, 28, f"Hand: {render_state.get('hand_id', '-')}")
        painter.drawText(20, 50, f"Street: {render_state.get('street', '-')}")
        painter.drawText(20, 72, f"Status: {render_state.get('status', '-')} / {render_state.get('freshness', '-')}")
        painter.drawText(20, 94, f"Players: {render_state.get('player_count', '-')} ({render_state.get('table_format', '-')})")
        if render_state.get("warnings"):
            painter.drawText(20, 116, f"Warning: {str(render_state['warnings'][0])}")

        table_amount_state = render_state.get("table_amount_state", {}) if isinstance(render_state.get("table_amount_state", {}), dict) else {}
        total_pot = table_amount_state.get("total_pot", {}) if isinstance(table_amount_state, dict) else {}
        if isinstance(total_pot, dict) and total_pot.get("amount_bb") is not None:
            painter.drawText(rect.width() - 210, 28, f"Pot: {float(total_pot['amount_bb']):.1f} BB")

        self._draw_recommendation(painter, rect, render_state, status)

        blind_labels: Dict[str, str] = {}
        posted_blinds = table_amount_state.get("posted_blinds", {}) if isinstance(table_amount_state, dict) else {}
        if isinstance(posted_blinds, dict):
            for blind_name, payload in posted_blinds.items():
                if not isinstance(payload, dict):
                    continue
                pos = payload.get("matched_position") or blind_name
                amount = payload.get("amount_bb")
                if amount is not None:
                    blind_labels[str(pos)] = f"{blind_name} {float(amount):g}"

        player_count = int(render_state.get("player_count", 0) or 0)
        ordered_positions = self._ordered_positions(render_state)
        slots = DISPLAY_SLOTS.get(player_count, [])
        player_boxes = {}
        for pos_name, slot in zip(ordered_positions, slots):
            payload = render_state.get("players", {}).get(pos_name, {})
            center_xy = (int(slot[0] * rect.width()), int(slot[1] * rect.height()))
            player_boxes[pos_name] = self._draw_player(painter, rect, pos_name, payload, center_xy, blind_labels.get(pos_name))

        board_cards = render_state.get("board_cards", []) or []
        board_w = len(board_cards) * 82 - (8 if board_cards else 0)
        bx = rect.width() // 2 - board_w // 2
        by = rect.height() // 2 - 54
        for idx, card in enumerate(board_cards):
            self._draw_card_pixmap(painter, render_card(card), bx + idx * 82, by)

        hero_position = render_state.get("hero_position")
        hero_cards = render_state.get("hero_cards", []) or []
        hero_payload = render_state.get("players", {}).get(hero_position, {})
        hero_box = player_boxes.get(hero_position)
        if hero_cards and hero_box and not hero_payload.get("is_fold", False):
            card_h = 104
            total_w = len(hero_cards) * 82 - 8
            hx = hero_box["center_x"] - total_w // 2
            hy = hero_box["box_y"] - card_h - 18
            hy = max(175, hy)
            for idx, card in enumerate(hero_cards):
                self._draw_card_pixmap(painter, render_card(card), hx + idx * 82, hy)

        actions = render_state.get("action_annotations", {}).get("actions_log", [])[-6:]
        y = rect.height() - 108
        painter.setPen(QtGui.QColor(240, 240, 240))
        for action in actions:
            action_text = f"{str(action.get('street', '')).upper()} {action.get('position', '')}: {action.get('action', '')}"
            if action.get("amount_bb") not in (None, 0, 0.0):
                action_text += f" {float(action['amount_bb']):.1f}"
            painter.drawText(20, y, action_text[:100])
            y += 18


class DecisionDetailsWindow:  # pragma: no cover
    def __init__(self, shared_state, refresh_ms: int = 100):
        if QtWidgets is None:
            raise RuntimeError("PySide6 is not installed; DecisionDetailsWindow unavailable")
        self.shared_state = shared_state
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("PokerVision Engine Analysis")
        self.window.resize(860, 900)
        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        self.header = QtWidgets.QLabel("Waiting for engine analysis")
        self.header.setMinimumHeight(54)
        self.header.setWordWrap(True)
        self.header.setStyleSheet("font-size: 14px; font-weight: 600; padding: 8px;")
        self.info = QtWidgets.QPlainTextEdit()
        self.info.setReadOnly(True)
        layout.addWidget(self.header)
        layout.addWidget(self.info, stretch=1)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(refresh_ms)

    def show(self):
        self.window.show()

    def refresh(self):
        _frame, render_state, status = self.shared_state.snapshot()
        rec = status.get("recommendation") or {}
        title = rec.get("title") or "NO DECISION"
        street = rec.get("street") or (render_state or {}).get("street") or "-"
        reason = rec.get("reason") or ""
        self.header.setText(f"Street: {street} | Recommendation: {title} | {reason}")
        text = status.get("analysis_text") or "Waiting for analysis text"
        if status.get("exception"):
            text = f"{text}\n\n=== ENGINE EXCEPTION ===\n{status['exception']}"
        self.info.setPlainText(str(text))


class IntegratedRunner:
    def __init__(self, args):
        self.args = args
        self.settings = get_default_settings()
        self.settings.debug_mode = True
        self.bridge = EngineBridge(self.settings)
        self.source = MockFrameSource(*self.settings.mock_table_size) if args.mock else ScreenFrameSource(self.settings.monitor_index)
        self.detector = MockDetectorBackend(self.settings) if args.mock else YoloDetectorBackend(self.settings)
        self.storage = StorageManager(self.settings)
        self.hand_manager = HandStateManager(
            self.settings.schema_version,
            self.settings.hand_stale_timeout_sec,
            self.settings.hand_close_timeout_sec,
        )
        self.pipeline = PokerVisionPipeline(self.settings, self.detector, self.storage, self.hand_manager)

    def _inject_recommendation(self, result, recommendation_payload: Dict[str, Any]) -> Optional[dict]:
        render_state = None
        if result.render_state:
            render_state = dict(result.render_state)
        elif result.hand is not None and getattr(result.hand, "render_state_snapshot", None):
            render_state = dict(result.hand.render_state_snapshot)
        if render_state is None:
            return None
        render_state["hero_recommendation"] = dict(recommendation_payload)
        return render_state

    def _build_status(self, result, recommendation_payload: Dict[str, Any], analysis_text: str, exception_text: str = "") -> Dict[str, Any]:
        return {
            "frame_id": result.analysis.frame_id,
            "street": result.analysis.street,
            "errors": list(result.analysis.errors),
            "recommendation": dict(recommendation_payload),
            "analysis_text": analysis_text,
            "exception": exception_text,
        }

    def process_once(self):
        frame = self.source.next_frame()
        result = self.pipeline.process_frame(frame)
        decision = None
        recommendation_payload: Dict[str, Any]
        analysis_text = ""
        exception_text = ""
        existing_render_state = None
        if result.render_state:
            existing_render_state = dict(result.render_state)
        elif result.hand is not None and getattr(result.hand, "render_state_snapshot", None):
            existing_render_state = dict(result.hand.render_state_snapshot)
        try:
            recommendation_payload = self.bridge.format_existing_render_state_for_ui(existing_render_state, hand=result.hand) or {}
            if not recommendation_payload:
                decision = self.bridge.build_recommendation(result.analysis, result.hand)
                recommendation_payload = self.bridge.format_for_ui(decision, result.hand)
            analysis_text = str(recommendation_payload.get("analysis_text") or "")
        except Exception:
            exception_text = traceback.format_exc(limit=8)
            analysis_text = exception_text
            recommendation_payload = {
                "title": "ENGINE ERROR",
                "action": "—",
                "reason": exception_text.splitlines()[-1] if exception_text else "engine exception",
                "street": result.analysis.street,
                "confidence": None,
                "debug": {},
                "analysis_text": analysis_text,
            }

        render_state = self._inject_recommendation(result, recommendation_payload)
        status = self._build_status(result, recommendation_payload, analysis_text=analysis_text, exception_text=exception_text)
        return frame, result, decision, render_state, status


def run_headless(args) -> int:
    runner = IntegratedRunner(args)
    iterations = args.iterations or 12
    for _ in range(iterations):
        _frame, result, _decision, _render_state, status = runner.process_once()
        print({
            "frame_id": result.analysis.frame_id,
            "street": result.analysis.street,
            "hero_cards": result.analysis.hero_cards,
            "board": result.analysis.board_cards,
            "hand_id": result.hand.hand_id if result.hand else None,
            "decision": status.get("recommendation", {}),
            "analysis_text": status.get("analysis_text", "")[:800],
            "errors": result.analysis.errors,
        })
    return 0


def run_with_ui(args) -> int:  # pragma: no cover
    if QtWidgets is None:
        print("PySide6 is not installed; UI mode is unavailable")
        return 1

    runner = IntegratedRunner(args)
    shared = SharedState()
    stop_event = threading.Event()

    def worker():
        while not stop_event.is_set():
            frame, _result, _decision, render_state, status = runner.process_once()
            shared.update_frame(frame.image)
            if render_state is not None:
                shared.update_render_state(render_state)
            shared.update_status(status)
            time.sleep(runner.settings.frame_debounce_ms / 1000.0)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    debug_window = DebugMonitorWindow(shared, runner.settings.ui_refresh_ms)
    table_window = IntegratedTableWindow(shared, runner.settings)
    details_window = DecisionDetailsWindow(shared, runner.settings.ui_refresh_ms)
    debug_window.show()
    table_window.show()
    details_window.show()
    code = app.exec()
    stop_event.set()
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PokerVision + Engine integrated runner")
    parser.add_argument("--mock", action="store_true", help="Use mock frame source and mock detectors")
    parser.add_argument("--real", action="store_true", help="Use real screen capture and YOLO detectors")
    parser.add_argument("--headless", action="store_true", help="Run without UI")
    parser.add_argument("--iterations", type=int, default=0, help="Headless iteration count")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.real:
        args.mock = False
    elif not args.mock:
        args.mock = True

    if args.headless:
        return run_headless(args)
    return run_with_ui(args)


if __name__ == "__main__":
    raise SystemExit(main())
