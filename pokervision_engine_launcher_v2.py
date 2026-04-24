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
import copy
import ast
import json
import re
import sys
import threading
import time
import traceback
import re
from types import SimpleNamespace
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
    build_villain_ranges_from_postflop_players,
    build_villain_ranges_from_preflop_spots,
    format_hero_decision_report,
    solve_hero_postflop,
    solve_hero_preflop,
)
from decision_types import HeroDecision, PostflopContext, PreflopContext

try:
    from pokervision_auto_click_runtime import AutoClickConfig, AutoClickRuntime
except Exception:
    AutoClickConfig = None  # type: ignore[assignment]
    AutoClickRuntime = None  # type: ignore[assignment]

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

REGIONS = [
    (63, 93, 875, 681),      # table_01
    (875, 93, 1686, 681),    # table_02
    (1686, 93, 2498, 681),   # table_03
    (63, 681, 875, 1269),    # table_04
    (875, 681, 1686, 1269),  # table_05
    (1686, 681, 2498, 1269), # table_06
]
SLOT_IDS = tuple(f"table_{index:02d}" for index in range(1, len(REGIONS) + 1))
SLOT_BBOX_BY_ID = {slot_id: REGIONS[index] for index, slot_id in enumerate(SLOT_IDS)}
DEFAULT_SLOT_ID = SLOT_IDS[0]


def get_slot_bbox(slot_id: str) -> Tuple[int, int, int, int]:
    normalized = str(slot_id or DEFAULT_SLOT_ID).strip() or DEFAULT_SLOT_ID
    if normalized not in SLOT_BBOX_BY_ID:
        raise KeyError(f"Unknown slot_id: {normalized}")
    return tuple(int(value) for value in SLOT_BBOX_BY_ID[normalized])


def iter_slots_round_robin() -> Iterable[str]:
    for slot_id in SLOT_IDS:
        yield slot_id


def _slot_id_from_view(slot_view: Any) -> str:
    try:
        index = int(slot_view)
    except Exception:
        index = 1
    if index < 1:
        index = 1
    if index > len(SLOT_IDS):
        index = len(SLOT_IDS)
    return SLOT_IDS[index - 1]


def resolve_slot_paths(root_dir: Path, slot_id: str) -> Dict[str, Path]:
    slot_root = Path(root_dir) / "tables" / str(slot_id)
    return {
        "slot_root": slot_root,
        "hands": slot_root / "hands",
        "temp": slot_root / "temp",
        "render": slot_root / "render",
        "logs": slot_root / "logs",
    }


@dataclass(slots=True)
class SlotContext:
    slot_id: str
    bbox: Tuple[int, int, int, int]
    paths: Dict[str, Path]


@dataclass(slots=True)
class SlotRuntimeState:
    last_frame_ts: Optional[str] = None
    last_active_hero_seen_at: Optional[float] = None
    current_hand_id: Optional[str] = None
    is_active: bool = False
    last_render_state_path: Optional[str] = None
    last_decision_summary: Optional[Dict[str, Any]] = None



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
    def _decode_ui_scalar(self, value: Any) -> Any:
        current = value
        for _ in range(4):
            if not isinstance(current, str):
                break
            text = current.strip()
            if not text:
                return ""
            parsed = None
            for parser in (json.loads, ast.literal_eval):
                try:
                    candidate = parser(text)
                except Exception:
                    continue
                if candidate != current:
                    parsed = candidate
                    break
            if parsed is not None:
                current = parsed
                continue
            cleaned = text.replace('\\"', '"').replace("\\'", "'").strip()
            cleaned = cleaned.strip('"').strip("'")
            if cleaned != text:
                current = cleaned
                continue
            break
        return current

    def _normalize_ui_payload(self, value: Any) -> Any:
        value = self._decode_ui_scalar(value)
        if isinstance(value, list):
            return [self._normalize_ui_payload(item) for item in value]
        if isinstance(value, dict):
            return {str(self._decode_ui_scalar(key)): self._normalize_ui_payload(item) for key, item in value.items()}
        return value

    def _safe_float(self, value: Any) -> Optional[float]:
        try:
            value = self._decode_ui_scalar(value)
            if value is None:
                return None
            return float(value)
        except Exception:
            if isinstance(value, str):
                cleaned = re.sub(r'[^0-9eE+\-\.]', '', value)
                if cleaned and cleaned not in {'-', '+', '.', 'e', 'E'}:
                    try:
                        return float(cleaned)
                    except Exception:
                        return None
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

        hero_in_position = self._hero_in_position_postflop(hand)
        villain_postflop_players = self._build_villain_postflop_players(hand, street)
        if villain_postflop_players:
            try:
                build_villain_ranges_from_postflop_players(
                    hero_hand=context.hero_hand,
                    board_runout=context.board,
                    players=villain_postflop_players,
                    dead_cards=context.dead_cards,
                )
                return solve_hero_postflop(
                    context,
                    villain_postflop_players=villain_postflop_players,
                    hero_in_position=hero_in_position,
                    trials=6000,
                    seed=42,
                )
            except Exception:
                pass

        villain_preflop_spots = self._build_villain_preflop_spots(hand)
        if villain_preflop_spots:
            try:
                build_villain_ranges_from_preflop_spots(villain_preflop_spots)
                return solve_hero_postflop(
                    context,
                    villain_preflop_spots=villain_preflop_spots,
                    hero_in_position=hero_in_position,
                    trials=6000,
                    seed=42,
                )
            except Exception:
                pass

        fallback_ranges = [GENERIC_WIDE_RANGE for _ in villain_positions]
        return solve_hero_postflop(
            context,
            villain_ranges=fallback_ranges,
            hero_in_position=hero_in_position,
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
        if not isinstance(panel, dict):
            panel = {}
        solver_output = render_state.get("solver_output") if isinstance(render_state.get("solver_output"), dict) else {}
        solver_input = render_state.get("solver_input") if isinstance(render_state.get("solver_input"), dict) else {}
        advisor_input = render_state.get("advisor_input") if isinstance(render_state.get("advisor_input"), dict) else {}
        result_payload = solver_output.get("result") if isinstance(solver_output, dict) else {}
        if not isinstance(result_payload, dict):
            result_payload = {}
        postflop_result = result_payload.get("postflop") if isinstance(result_payload.get("postflop"), dict) else {}
        debug_payload = panel.get("hero_decision_debug") or render_state.get("hero_decision_debug") or {}
        if not isinstance(debug_payload, dict):
            debug_payload = {}
        debug = debug_payload.get("debug") if isinstance(debug_payload, dict) else {}
        if not isinstance(debug, dict):
            debug = {}
        range_debug = panel.get("range_debug") or render_state.get("range_debug") or []
        if isinstance(range_debug, dict):
            range_debug_items = [range_debug]
        elif isinstance(range_debug, list):
            range_debug_items = [item for item in range_debug if isinstance(item, dict)]
        else:
            range_debug_items = []
        first_range_debug = range_debug_items[0] if range_debug_items else {}
        meta = first_range_debug.get("meta") if isinstance(first_range_debug.get("meta"), dict) else {}
        postflop_payload = debug_payload.get("postflop") if isinstance(debug_payload.get("postflop"), dict) else {}
        if not isinstance(postflop_payload, dict):
            postflop_payload = {}
        report = {}
        for candidate in (
            postflop_result.get("report"),
            postflop_payload.get("report"),
            first_range_debug.get("report"),
        ):
            candidate = self._normalize_ui_payload(candidate)
            if isinstance(candidate, dict) and candidate:
                report = dict(candidate)
                break
        trace = {}
        for candidate in (
            panel.get("postflop_range_trace"),
            solver_output.get("postflop_range_trace"),
            solver_input.get("postflop_range_trace"),
            advisor_input.get("postflop_range_trace"),
            solver_output.get("runtime_range_state"),
            solver_input.get("runtime_range_state"),
            advisor_input.get("runtime_range_state"),
            debug_payload.get("postflop_range_trace"),
        ):
            candidate = self._normalize_ui_payload(candidate)
            if isinstance(candidate, dict) and candidate:
                trace = dict(candidate)
                break
        villain_sources = []
        for candidate in (
            postflop_result.get("villain_sources"),
            result_payload.get("villain_sources"),
            postflop_payload.get("villain_sources"),
        ):
            candidate = self._normalize_ui_payload(candidate)
            if isinstance(candidate, list) and candidate:
                villain_sources = list(candidate)
                break
        villain_summary_raw = self._normalize_ui_payload(trace.get("villain_sources_summary"))
        villain_reports_raw = self._normalize_ui_payload(trace.get("villain_range_reports"))
        villain_summary = villain_summary_raw if isinstance(villain_summary_raw, list) else []
        villain_reports = villain_reports_raw if isinstance(villain_reports_raw, list) else []
        hero_cards = list(render_state.get("hero_cards") or getattr(hand, "hero_cards", []) or [])
        board_cards = list(render_state.get("board_cards") or getattr(hand, "board_cards", []) or [])
        return {
            "street": str(render_state.get("street") or (getattr(hand, "street_state", {}) or {}).get("current_street") or "-"),
            "recommended_action": str(render_state.get("recommended_action") or panel.get("recommended_action") or result_payload.get("engine_action") or "NO DECISION"),
            "reason": str(render_state.get("decision_reason") or panel.get("decision_reason") or result_payload.get("reason") or ""),
            "confidence": render_state.get("decision_confidence") if render_state.get("decision_confidence") is not None else (panel.get("decision_confidence") if panel.get("decision_confidence") is not None else result_payload.get("confidence")),
            "node_type": str(panel.get("node_type") or render_state.get("node_type") or meta.get("projection_node_type") or "-"),
            "projection_node_type": str(panel.get("projection_node_type") or render_state.get("node_type") or meta.get("projection_node_type") or ""),
            "advisor_node_type": str(panel.get("advisor_node_type") or meta.get("advisor_node_type") or debug.get("node_type") or ""),
            "advisor_mapping_reason": str(panel.get("advisor_mapping_reason") or meta.get("advisor_mapping_reason") or ""),
            "hero_position": str(render_state.get("hero_position") or panel.get("hero_position") or getattr(hand, "hero_position", "-")),
            "hero_cards": hero_cards,
            "board_cards": board_cards,
            "range_owner": str(debug.get("range_owner") or "hero"),
            "hand_class": str(first_range_debug.get("hand_class") or "-"),
            "description": str(meta.get("description") or ""),
            "matching_actions": list(first_range_debug.get("matching_actions") or []),
            "chosen_action": str(first_range_debug.get("action") or "-"),
            "selected_range_expr": str(first_range_debug.get("selected_range_expr") or "-"),
            "action_map": dict(first_range_debug.get("action_map") or {}),
            "fallback_reason": first_range_debug.get("fallback_reason"),
            "recommended_amount_to": render_state.get("recommended_amount_to") if render_state.get("recommended_amount_to") is not None else (panel.get("recommended_amount_to") if panel.get("recommended_amount_to") is not None else result_payload.get("amount_to")),
            "recommended_size_pct": render_state.get("recommended_size_pct") if render_state.get("recommended_size_pct") is not None else (panel.get("recommended_size_pct") if panel.get("recommended_size_pct") is not None else result_payload.get("size_pct")),
            "engine_status": str(render_state.get("engine_status") or panel.get("engine_status") or ""),
            "solver_status": str(panel.get("solver_status") or render_state.get("solver_status") or ""),
            "solver_reused": bool(panel.get("solver_reused") or render_state.get("solver_result_reused") or False),
            "solver_reuse_reason": str(panel.get("solver_reuse_reason") or render_state.get("solver_reuse_reason") or ""),
            "pot_before_hero": report.get("pot_before_hero", self._pot_before_hero(hand) if hand is not None else None),
            "to_call": report.get("to_call", self._to_call(hand) if hand is not None else None),
            "effective_stack": report.get("effective_stack", self._effective_stack(hand) if hand is not None else None),
            "hero_equity": postflop_result.get("hero_equity") if postflop_result.get("hero_equity") is not None else postflop_payload.get("hero_equity"),
            "realized_equity": postflop_result.get("realized_equity") if postflop_result.get("realized_equity") is not None else postflop_payload.get("realized_equity"),
            "villain_sources": villain_sources,
            "villain_sources_summary": villain_summary,
            "villain_range_reports": villain_reports,
            "postflop_range_trace": trace,
            "report": report,
            "line_context": report.get("line_context") if isinstance(report, dict) else {},
            "hero_tags": list(report.get("hero_tags") or []),
            "range_debug": range_debug_items,
            "analysis_sections": list(panel.get("sections") or []),
        }

    def build_analysis_text_from_render_state(self, render_state: dict, hand=None) -> str:
        ctx = self._resolve_state_analysis_context(render_state, hand=hand)
        lines: List[str] = []
        street = str(ctx["street"]).lower()
        lines.append("=== PREFLOP ANALYSIS ===" if street == "preflop" else "=== POSTFLOP ANALYSIS ===")
        lines.append(f"Recommendation: {ctx['recommended_action']}")
        lines.append(f"Reason: {ctx['reason']}")
        if ctx["confidence"] is not None:
            conf = self._safe_float(ctx['confidence'])
            lines.append(f"Confidence: {(conf if conf is not None else 0.0):.2f}")
        lines.append("")
        if hand is not None:
            lines.append(f"Hand ID: {hand.hand_id}")
        lines.append(f"Hero position (vision): {ctx['hero_position']}")
        if ctx["hero_cards"]:
            lines.append(f"Hero cards: {' '.join(ctx['hero_cards'])}")
        if street != "preflop" and ctx.get("board_cards"):
            lines.append(f"Board: {' '.join(ctx['board_cards'])}")
        lines.append("")

        if street == "preflop":
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

        pot_before = ctx.get("pot_before_hero")
        to_call = ctx.get("to_call")
        effective_stack = ctx.get("effective_stack")
        if pot_before is not None:
            pot_value = self._safe_float(pot_before)
            if pot_value is not None:
                lines.append(f"Pot before HERO: {pot_value:.2f}")
        if to_call is not None:
            to_call_value = self._safe_float(to_call)
            if to_call_value is not None:
                lines.append(f"To call: {to_call_value:.2f}")
        if effective_stack is not None:
            effective_stack_value = self._safe_float(effective_stack)
            if effective_stack_value is not None:
                lines.append(f"Effective stack: {effective_stack_value:.2f}")
        if ctx.get("recommended_size_pct") is not None:
            size_value = self._safe_float(ctx['recommended_size_pct'])
            if size_value is not None:
                lines.append(f"Recommended size: {size_value:.1f}% pot")
        if ctx.get("recommended_amount_to") is not None:
            amount_value = self._safe_float(ctx['recommended_amount_to'])
            if amount_value is not None:
                lines.append(f"Recommended amount to: {amount_value:.2f}")
        if ctx.get("hero_equity") is not None:
            hero_eq = self._safe_float(ctx['hero_equity'])
            if hero_eq is not None:
                lines.append(f"Hero equity: {hero_eq:.4f}")
        if ctx.get("realized_equity") is not None:
            realized_eq = self._safe_float(ctx['realized_equity'])
            if realized_eq is not None:
                lines.append(f"Realized equity: {realized_eq:.4f}")
        if ctx.get("engine_status"):
            lines.append(f"Engine status: {ctx['engine_status']}")
        if ctx.get("solver_reused"):
            reuse_reason = ctx.get("solver_reuse_reason") or "same_fingerprint"
            lines.append(f"Solver reused: yes ({reuse_reason})")
        lines.append("")

        report = self._normalize_ui_payload(ctx.get("report") or {})
        if isinstance(report, dict) and report:
            try:
                lines.append(format_hero_decision_report(report))
            except Exception:
                recommended_option = report.get("recommended_option") if isinstance(report.get("recommended_option"), dict) else {}
                if recommended_option:
                    lines.append("=== POSTFLOP EV SUMMARY ===")
                    lines.append(f"Recommended option: {recommended_option.get('action', '-')}")
                    if recommended_option.get("ev") is not None:
                        ev_value = self._safe_float(recommended_option['ev'])
                        if ev_value is not None:
                            lines.append(f"Recommended EV: {ev_value:.6f}")
                    if recommended_option.get("size_pct") is not None:
                        size_pct_value = self._safe_float(recommended_option['size_pct'])
                        if size_pct_value is not None:
                            lines.append(f"Recommended size pct: {size_pct_value:.1f}")
                    if recommended_option.get("amount_to") is not None:
                        amount_to_value = self._safe_float(recommended_option['amount_to'])
                        if amount_to_value is not None:
                            lines.append(f"Recommended amount to: {amount_to_value:.2f}")
                size_reports = report.get("size_reports") if isinstance(report.get("size_reports"), list) else []
                if size_reports:
                    lines.append("")
                    lines.append("=== SIZE REPORTS ===")
                    for item in size_reports[:6]:
                        if not isinstance(item, dict):
                            continue
                        action = str(item.get("action") or "-")
                        ev_value = item.get("ev")
                        amount_to = item.get("amount_to")
                        size_pct = item.get("size_pct")
                        gate_status = str(item.get("gate_status") or "-")
                        fragments = [f"{action}"]
                        if size_pct is not None:
                            size_pct_value = self._safe_float(size_pct)
                            if size_pct_value is not None:
                                fragments.append(f"{size_pct_value:.1f}%")
                        if amount_to is not None:
                            amount_to_value = self._safe_float(amount_to)
                            if amount_to_value is not None:
                                fragments.append(f"to {amount_to_value:.2f}")
                        if ev_value is not None:
                            ev_value_num = self._safe_float(ev_value)
                            if ev_value_num is not None:
                                fragments.append(f"EV {ev_value_num:.6f}")
                        fragments.append(f"gate={gate_status}")
                        lines.append(" | ".join(fragments))
        else:
            lines.append("Postflop report is empty.")

        trace = ctx.get("postflop_range_trace") or {}
        if isinstance(trace, dict) and trace:
            lines.append("")
            lines.append("=== RANGE TRACE ===")
            requested = trace.get("requested_range_build_mode")
            if requested:
                lines.append(f"Requested mode: {requested}")
            route = trace.get("range_build_mode")
            if route:
                lines.append(f"Used route: {route}")
            payload_kind = trace.get("payload_kind")
            if payload_kind:
                lines.append(f"Payload: {payload_kind}")
            contract = trace.get("range_contract")
            if contract:
                lines.append(f"Contract: {contract}")

        villain_sources = ctx.get("villain_sources") or []
        villain_summary = ctx.get("villain_sources_summary") or []
        villain_reports = ctx.get("villain_range_reports") or []
        if villain_sources or villain_summary or villain_reports:
            lines.append("")
            lines.append("=== VILLAIN SOURCES ===")
            summary_by_name = {}
            for item in villain_summary:
                if isinstance(item, dict):
                    summary_by_name[str(item.get("name") or item.get("villain_pos") or "Villain")] = item
            report_by_name = {}
            for item in villain_reports:
                if isinstance(item, dict):
                    report_by_name[str(item.get("name") or item.get("villain_pos") or "Villain")] = item
            shown = set()
            iterable = villain_sources if villain_sources else [item for item in villain_summary if isinstance(item, dict)]
            for source in iterable:
                if not isinstance(source, dict):
                    continue
                name = str(source.get("name") or source.get("villain_pos") or "Villain")
                shown.add(name)
                source_type = str(source.get("source_type") or summary_by_name.get(name, {}).get("source_type") or "-")
                combo_count = source.get("combo_count")
                if combo_count is None:
                    weighted = source.get("weighted_combos")
                    if isinstance(weighted, list):
                        combo_count = len(weighted)
                if combo_count is None:
                    combo_count = summary_by_name.get(name, {}).get("combo_count")
                total_weight = summary_by_name.get(name, {}).get("total_weight")
                suffix = f" | combos={combo_count}" if combo_count is not None else ""
                if total_weight not in (None, ""):
                    suffix += f" | weight={total_weight}"
                lines.append(f"{name}: {source_type}{suffix}")
                report_item = report_by_name.get(name, {})
                start_source = report_item.get("range_source") if isinstance(report_item.get("range_source"), dict) else {}
                final_source = report_item.get("final_range_source") if isinstance(report_item.get("final_range_source"), dict) else source
                start_expr = start_source.get("normalized_expr") or start_source.get("raw_expr")
                if start_expr:
                    lines.append(f"  start: {start_expr}")
                final_expr = final_source.get("normalized_expr") or final_source.get("raw_expr")
                if final_expr:
                    lines.append(f"  final: {final_expr}")
                payload = report_item.get("report") if isinstance(report_item.get("report"), dict) else {}
                steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
                for step in steps[:4]:
                    if not isinstance(step, dict):
                        continue
                    step_street = str(step.get("street") or ctx.get("street") or "").upper()
                    action = str(step.get("action") or step.get("semantic_action") or "action").upper()
                    pct = step.get("bet_pct_pot")
                    step_line = f"  {step_street} {action}"
                    if pct not in (None, ""):
                        try:
                            step_line += f" {float(pct):.1f}%"
                        except Exception:
                            pass
                    before_source = step.get("range_before_source") if isinstance(step.get("range_before_source"), dict) else {}
                    after_source = step.get("range_after_source") if isinstance(step.get("range_after_source"), dict) else {}
                    before_count = len(before_source.get("weighted_combos", [])) if isinstance(before_source.get("weighted_combos"), list) else None
                    after_count = len(after_source.get("weighted_combos", [])) if isinstance(after_source.get("weighted_combos"), list) else None
                    if before_count is not None or after_count is not None:
                        step_line += f" | {before_count if before_count is not None else '?'}c→{after_count if after_count is not None else '?'}c"
                    lines.append(step_line)
            for name, item in summary_by_name.items():
                if name in shown:
                    continue
                source_type = str(item.get("source_type") or "-")
                combo_count = item.get("combo_count")
                total_weight = item.get("total_weight")
                suffix = f" | combos={combo_count}" if combo_count is not None else ""
                if total_weight not in (None, ""):
                    suffix += f" | weight={total_weight}"
                lines.append(f"{name}: {source_type}{suffix}")

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
        self.selected_slot_id = _slot_id_from_view(getattr(args, "slot_view", 1))
        self.slot_contexts: Dict[str, SlotContext] = {}
        self.slot_states: Dict[str, SlotRuntimeState] = {}
        self._round_robin_cursor = SLOT_IDS.index(self.selected_slot_id) if self.selected_slot_id in SLOT_IDS else 0
        self._last_processed_slot_id: Optional[str] = None
        self._bootstrap_slot_runtime()
        self.hand_manager = HandStateManager(
            self.settings.schema_version,
            self.settings.hand_stale_timeout_sec,
            self.settings.hand_close_timeout_sec,
        )
        self.pipeline = PokerVisionPipeline(self.settings, self.detector, self.storage, self.hand_manager)
        self.auto_click_runtime = self._build_auto_click_runtime(args)
        self._auto_click_cycle_started_at: Optional[float] = None
        self._auto_click_cycle_key: Optional[str] = None

    def _bootstrap_slot_runtime(self) -> None:
        for slot_id in iter_slots_round_robin():
            paths = resolve_slot_paths(self.settings.root_dir, slot_id)
            context = SlotContext(slot_id=slot_id, bbox=get_slot_bbox(slot_id), paths=paths)
            self.slot_contexts[slot_id] = context
            self.slot_states[slot_id] = SlotRuntimeState()
            print(f"[MultiTable] bootstrap slot {slot_id} bbox={context.bbox}")

    def _build_slot_frame(self, frame, slot_id: str):
        image = getattr(frame, "image", None)
        if image is None:
            return frame
        x1, y1, x2, y2 = get_slot_bbox(slot_id)
        frame_height = int(image.shape[0]) if hasattr(image, "shape") and len(image.shape) >= 2 else 0
        frame_width = int(image.shape[1]) if hasattr(image, "shape") and len(image.shape) >= 2 else 0
        x1 = max(0, min(frame_width, int(x1)))
        y1 = max(0, min(frame_height, int(y1)))
        x2 = max(0, min(frame_width, int(x2)))
        y2 = max(0, min(frame_height, int(y2)))
        if x2 <= x1 or y2 <= y1:
            cropped = image.copy() if hasattr(image, "copy") else image
        else:
            cropped = image[y1:y2, x1:x2].copy()
        frame_id = str(getattr(frame, "frame_id", "frame_unknown") or "frame_unknown")
        timestamp = getattr(frame, "timestamp", None)
        slot_frame = SimpleNamespace(
            frame_id=f"{frame_id}__{slot_id}",
            timestamp=timestamp,
            image=cropped,
            slot_id=slot_id,
            slot_bbox=(x1, y1, x2, y2),
            source_frame_id=frame_id,
            source_timestamp=timestamp,
            source_image=image,
        )
        return slot_frame

    def _build_slot_frames(self, frame) -> Dict[str, Any]:
        slot_frames: Dict[str, Any] = {}
        for slot_id in iter_slots_round_robin():
            slot_frames[slot_id] = self._build_slot_frame(frame, slot_id)
        return slot_frames

    def _iter_slot_ids_from_cursor(self) -> Iterable[str]:
        total = len(SLOT_IDS)
        if total <= 0:
            return []
        start = int(self._round_robin_cursor) % total
        return [SLOT_IDS[(start + offset) % total] for offset in range(total)]

    def _advance_round_robin_cursor(self, processed_slot_id: Optional[str] = None) -> None:
        total = len(SLOT_IDS)
        if total <= 0:
            self._round_robin_cursor = 0
            return
        if processed_slot_id in SLOT_IDS:
            self._round_robin_cursor = (SLOT_IDS.index(str(processed_slot_id)) + 1) % total
            return
        self._round_robin_cursor = (int(self._round_robin_cursor) + 1) % total

    def _detect_active_hero_fast(self, slot_frame) -> List[Any]:
        try:
            detections = list(self.detector.detect_active_hero(slot_frame) or [])
        except Exception:
            return []
        out: List[Any] = []
        for detection in detections:
            label = str(getattr(detection, "label", "") or "").strip().lower().replace("_", "")
            if label in {"activehero", "heroactive"}:
                out.append(detection)
        return out

    def _select_processing_slot(self, slot_frames: Dict[str, Any]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        scan_summary: List[Dict[str, Any]] = []
        chosen_slot_id: Optional[str] = None
        now = time.monotonic()
        for slot_id in self._iter_slot_ids_from_cursor():
            slot_frame = slot_frames.get(slot_id)
            detections = self._detect_active_hero_fast(slot_frame)
            active = bool(detections)
            scan_summary.append({
                "slot_id": slot_id,
                "active_hero_found": active,
                "active_hero_count": len(detections),
            })
            slot_state = self.slot_states.get(slot_id)
            if slot_state is not None:
                slot_state.last_frame_ts = getattr(slot_frame, "timestamp", None) if slot_frame is not None else None
                slot_state.is_active = active
                if active:
                    slot_state.last_active_hero_seen_at = now
            if active and chosen_slot_id is None:
                chosen_slot_id = slot_id
                break
        self._advance_round_robin_cursor(chosen_slot_id)
        if chosen_slot_id is not None:
            self._last_processed_slot_id = chosen_slot_id
        return chosen_slot_id, scan_summary

    def _build_idle_result(self, frame, scan_summary: List[Dict[str, Any]]):
        display_slot_id = self.selected_slot_id if self.selected_slot_id in SLOT_IDS else DEFAULT_SLOT_ID
        display_frame = frame
        analysis = SimpleNamespace(
            frame_id=str(getattr(display_frame, "frame_id", "frame_unknown") or "frame_unknown"),
            street="preflop",
            hero_cards=[],
            board_cards=[],
            errors=[],
            warnings=[],
            active_hero_found=False,
            slot_scan_summary=list(scan_summary),
        )
        result = SimpleNamespace(
            analysis=analysis,
            hand=None,
            render_state=None,
        )
        recommendation_payload = {
            "title": "WAITING ACTIVEHERO",
            "action": "—",
            "reason": "No ActiveHero detected in any slot on this pass",
            "street": "preflop",
            "confidence": None,
            "debug": {"slot_scan_summary": list(scan_summary)},
            "analysis_text": "",
        }
        analysis_lines = [
            "No ActiveHero detected in any slot on this pass.",
            "Round-robin quick gate:",
        ]
        analysis_lines.extend(
            f"- {item['slot_id']}: active_hero_found={item['active_hero_found']} count={item['active_hero_count']}"
            for item in scan_summary
        )
        analysis_text = "\n".join(analysis_lines)
        return display_frame, result, recommendation_payload, analysis_text

    def _build_auto_click_runtime(self, args):
        if not getattr(args, "autoclick", False):
            return None
        if AutoClickRuntime is None or AutoClickConfig is None:
            raise RuntimeError("pokervision_auto_click_runtime.py is unavailable, but --autoclick was requested")
        config = AutoClickConfig(
            enabled=True,
            enable_idle_movement=not bool(getattr(args, "autoclick_disable_idle", False)),
            button_model_path=str(getattr(args, "autoclick_model_path", AutoClickConfig.button_model_path)),
            force_primary_monitor_capture=True,
            scroll_enabled_probability=0.0,
        )
        return AutoClickRuntime(config=config)

    def _build_auto_click_cycle_key(self, result) -> Optional[str]:
        if not bool(getattr(result.analysis, "active_hero_found", False)):
            return None
        hand = getattr(result, "hand", None)
        hand_id = getattr(hand, "hand_id", None) if hand is not None else None
        if hand_id:
            return f"hand:{hand_id}"
        street = str(getattr(result.analysis, "street", "preflop") or "preflop")
        hero_cards = tuple(getattr(result.analysis, "hero_cards", []) or [])
        board_cards = tuple(getattr(result.analysis, "board_cards", []) or [])
        return f"active:{street}:{hero_cards}:{board_cards}"

    def _resolve_auto_click_started_at(self, result) -> float:
        now = time.monotonic()
        cycle_key = self._build_auto_click_cycle_key(result)
        if cycle_key is None:
            self._auto_click_cycle_started_at = None
            self._auto_click_cycle_key = None
            return now
        if self._auto_click_cycle_started_at is None or cycle_key != self._auto_click_cycle_key:
            self._auto_click_cycle_started_at = now
            self._auto_click_cycle_key = cycle_key
        return float(self._auto_click_cycle_started_at)

    def _normalize_frame_for_autoclick(self, image):
        if image is None:
            return None
        if np is not None and isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 4:
                return image[:, :, :3]
            return image
        return image

    def _safe_int(self, value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(round(float(value)))
        except Exception:
            return None

    def _clamp_action_panel_bbox(self, bbox: Tuple[int, int, int, int], frame_width: int, frame_height: int) -> Optional[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(frame_width, int(x1)))
        y1 = max(0, min(frame_height, int(y1)))
        x2 = max(0, min(frame_width, int(x2)))
        y2 = max(0, min(frame_height, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _infer_action_panel_bbox(self, frame_bgr, result) -> Optional[Tuple[int, int, int, int]]:
        if frame_bgr is None or not hasattr(frame_bgr, "shape") or len(frame_bgr.shape) < 2:
            return None
        frame_height = int(frame_bgr.shape[0])
        frame_width = int(frame_bgr.shape[1])
        default_bbox = self._clamp_action_panel_bbox(
            (
                int(frame_width * 0.14),
                int(frame_height * 0.54),
                int(frame_width * 0.86),
                int(frame_height * 0.995),
            ),
            frame_width,
            frame_height,
        )
        hand = getattr(result, "hand", None)
        if hand is None:
            return default_bbox
        hero_pos = getattr(hand, "hero_position", None)
        positions = getattr(hand, "positions", None)
        if not hero_pos or not isinstance(positions, dict):
            return default_bbox
        hero_payload = positions.get(hero_pos)
        if not isinstance(hero_payload, dict):
            return default_bbox
        bbox_payload = hero_payload.get("bbox")
        if not isinstance(bbox_payload, dict):
            return default_bbox

        hx1 = self._safe_int(bbox_payload.get("x1"))
        hy1 = self._safe_int(bbox_payload.get("y1"))
        hx2 = self._safe_int(bbox_payload.get("x2"))
        hy2 = self._safe_int(bbox_payload.get("y2"))
        if None in {hx1, hy1, hx2, hy2}:
            return default_bbox

        hero_center_x = int(round((int(hx1) + int(hx2)) / 2.0))
        hero_top = min(int(hy1), int(hy2))
        hero_bottom = max(int(hy1), int(hy2))
        hero_width = max(1, abs(int(hx2) - int(hx1)))

        panel_half_width = max(int(frame_width * 0.24), int(hero_width * 1.9))
        x1 = hero_center_x - panel_half_width
        x2 = hero_center_x + panel_half_width
        y1 = min(hero_top - int(frame_height * 0.12), int(frame_height * 0.56))
        y2 = max(hero_bottom + int(frame_height * 0.28), int(frame_height * 0.985))
        return self._clamp_action_panel_bbox((x1, y1, x2, y2), frame_width, frame_height) or default_bbox

    def _auto_click_result_to_dict(self, auto_click_result) -> Dict[str, Any]:
        if auto_click_result is None:
            return {
                "enabled": bool(self.auto_click_runtime is not None),
                "state": "DISABLED" if self.auto_click_runtime is None else "IDLE",
                "executed": False,
                "locked": False,
                "plan_name": None,
                "normalized_action": None,
                "raw_action": None,
                "events": [],
            }
        return {
            "enabled": bool(self.auto_click_runtime is not None),
            "state": str(getattr(auto_click_result, "state", "")),
            "executed": bool(getattr(auto_click_result, "executed", False)),
            "locked": bool(getattr(auto_click_result, "locked", False)),
            "plan_name": getattr(auto_click_result, "plan_name", None),
            "normalized_action": getattr(auto_click_result, "normalized_action", None),
            "raw_action": getattr(auto_click_result, "raw_action", None),
            "events": [
                {
                    "name": str(getattr(event, "name", "")),
                    "ts": float(getattr(event, "ts", 0.0) or 0.0),
                    "payload": dict(getattr(event, "payload", {}) or {}),
                }
                for event in list(getattr(auto_click_result, "events", []) or [])
            ],
        }

    def _inject_recommendation(self, result, recommendation_payload: Dict[str, Any], auto_click_payload: Optional[Dict[str, Any]] = None) -> Optional[dict]:
        render_state = None
        if result.render_state:
            render_state = dict(result.render_state)
        elif result.hand is not None and getattr(result.hand, "render_state_snapshot", None):
            render_state = dict(result.hand.render_state_snapshot)
        if render_state is None:
            return None
        render_state["hero_recommendation"] = dict(recommendation_payload)
        if auto_click_payload is not None:
            render_state["auto_click"] = dict(auto_click_payload)
        return render_state

    def _build_status(self, result, recommendation_payload: Dict[str, Any], analysis_text: str, exception_text: str = "", auto_click_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "slot_id": getattr(self, "selected_slot_id", DEFAULT_SLOT_ID),
            "frame_id": result.analysis.frame_id,
            "street": result.analysis.street,
            "errors": list(result.analysis.errors),
            "recommendation": dict(recommendation_payload),
            "analysis_text": analysis_text,
            "exception": exception_text,
            "auto_click": dict(auto_click_payload or self._auto_click_result_to_dict(None)),
        }

    def _extract_identity_from_render_state(self, render_state: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
        if not isinstance(render_state, dict):
            return {"decision_id": None, "solver_fingerprint": None, "source_frame_id": None, "engine_action": None}
        debug_payload = render_state.get("hero_decision_debug") if isinstance(render_state.get("hero_decision_debug"), dict) else {}
        raw_repr = str(debug_payload.get("raw_repr") or "")

        def _rx(name: str) -> Optional[str]:
            match = re.search(rf"{name}='([^']+)'", raw_repr)
            return match.group(1) if match else None

        engine_result = render_state.get("engine_result") if isinstance(render_state.get("engine_result"), dict) else {}
        return {
            "decision_id": _rx("decision_id"),
            "solver_fingerprint": str(render_state.get("solver_fingerprint") or _rx("solver_fingerprint") or "") or None,
            "source_frame_id": str(render_state.get("source_frame_id") or _rx("source_frame_id") or "") or None,
            "engine_action": str(engine_result.get("engine_action") or "") or None,
        }

    def _decision_identity_tuple(self, decision: Optional[HeroDecision]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        if decision is None:
            return (None, None, None, None)
        debug = getattr(decision, "debug", {}) or {}
        decision_id = getattr(decision, "decision_id", None)
        solver_fingerprint = getattr(decision, "solver_fingerprint", None)
        source_frame_id = getattr(decision, "source_frame_id", None)
        engine_action = getattr(decision, "engine_action", None)
        if isinstance(debug, dict):
            decision_id = decision_id or debug.get("decision_id")
            solver_fingerprint = solver_fingerprint or debug.get("solver_fingerprint")
            source_frame_id = source_frame_id or debug.get("source_frame_id")
            engine_action = engine_action or debug.get("engine_action") or debug.get("recommended_action")
        return (
            str(decision_id) if decision_id not in (None, "") else None,
            str(solver_fingerprint) if solver_fingerprint not in (None, "") else None,
            str(source_frame_id) if source_frame_id not in (None, "") else None,
            str(engine_action).lower() if engine_action not in (None, "") else None,
        )

    def _build_autoclick_decision_from_render_state(self, render_state: Optional[Dict[str, Any]], *, expected_frame_id: Optional[str] = None) -> Optional[HeroDecision]:
        if not isinstance(render_state, dict):
            return None
        identity = self._extract_identity_from_render_state(render_state)
        frame_id = identity.get("source_frame_id")
        if expected_frame_id and frame_id and str(frame_id) != str(expected_frame_id):
            return None
        hero_debug = render_state.get("hero_decision_debug") if isinstance(render_state.get("hero_decision_debug"), dict) else {}
        engine_result = render_state.get("engine_result") if isinstance(render_state.get("engine_result"), dict) else {}
        solver_context = render_state.get("solver_context") if isinstance(render_state.get("solver_context"), dict) else {}
        preflop = hero_debug.get("preflop") if isinstance(hero_debug.get("preflop"), dict) else None
        postflop = hero_debug.get("postflop") if isinstance(hero_debug.get("postflop"), dict) else None
        debug_payload: Dict[str, object] = {}
        for key in ("node_type", "opener_pos", "three_bettor_pos", "four_bettor_pos", "limpers", "callers"):
            value = solver_context.get(key)
            if value not in (None, ""):
                debug_payload[key] = value
        if identity.get("solver_fingerprint"):
            debug_payload["solver_fingerprint"] = identity["solver_fingerprint"]
        if identity.get("decision_id"):
            debug_payload["decision_id"] = identity["decision_id"]
        if identity.get("source_frame_id"):
            debug_payload["source_frame_id"] = identity["source_frame_id"]
        if identity.get("engine_action"):
            debug_payload["engine_action"] = identity["engine_action"]
            debug_payload["recommended_action"] = identity["engine_action"]
        street = str(engine_result.get("street") or render_state.get("street") or "preflop")
        debug_payload["street"] = street
        debug_payload["meta"] = {
            "source": "render_state_authoritative",
            "hand_id": render_state.get("hand_id"),
            "hero_original_position": render_state.get("hero_position"),
            "projection_node_type": solver_context.get("node_type"),
            "advisor_node_type": solver_context.get("node_type"),
            "actions_seen": list(solver_context.get("action_history") or []),
        }
        return SimpleNamespace(
            street=street,
            engine_action=str(engine_result.get("engine_action") or render_state.get("recommended_action") or "").lower(),
            amount_to=engine_result.get("amount_to"),
            size_pct=engine_result.get("size_pct") if engine_result.get("size_pct") is not None else render_state.get("recommended_size_pct"),
            actor_name=engine_result.get("actor_name") or "Hero",
            actor_pos=engine_result.get("actor_pos") or render_state.get("hero_position"),
            reason=str(hero_debug.get("reason") or engine_result.get("reason") or render_state.get("decision_reason") or ""),
            confidence=hero_debug.get("confidence") if hero_debug.get("confidence") is not None else engine_result.get("confidence"),
            source=hero_debug.get("source") or engine_result.get("source") or "render_state",
            solver_fingerprint=identity.get("solver_fingerprint"),
            decision_id=identity.get("decision_id"),
            source_frame_id=identity.get("source_frame_id"),
            preflop=SimpleNamespace(**preflop) if isinstance(preflop, dict) else None,
            postflop=SimpleNamespace(**postflop) if isinstance(postflop, dict) else None,
            debug=debug_payload,
        )

    def _select_autoclick_decision(
        self,
        *,
        live_decision: Optional[HeroDecision],
        render_state: Optional[Dict[str, Any]],
        analysis_frame_id: Optional[str],
        active_hero_present: bool,
    ) -> Tuple[Optional[HeroDecision], Optional[str]]:
        if not active_hero_present:
            return live_decision, None
        authoritative = self._build_autoclick_decision_from_render_state(render_state, expected_frame_id=analysis_frame_id)
        if authoritative is None:
            return live_decision, None
        live_identity = self._decision_identity_tuple(live_decision)
        authoritative_identity = self._decision_identity_tuple(authoritative)
        if live_decision is None:
            return authoritative, "render_state_authoritative_no_live_decision"
        if live_identity != authoritative_identity:
            return authoritative, "render_state_authoritative_mismatch"
        return live_decision, None

    def process_once(self):
        full_frame = self.source.next_frame()
        slot_frames = self._build_slot_frames(full_frame)
        processing_slot_id, scan_summary = self._select_processing_slot(slot_frames)

        if processing_slot_id is None:
            display_frame = slot_frames.get(self.selected_slot_id) or self._build_slot_frame(full_frame, self.selected_slot_id)
            frame, result, recommendation_payload, analysis_text = self._build_idle_result(display_frame, scan_summary)
            decision = None
            exception_text = ""
            auto_click_result = None
            existing_render_state = None
            if self.auto_click_runtime is not None:
                try:
                    frame_for_autoclick = self._normalize_frame_for_autoclick(getattr(full_frame, "image", None))
                    frame_width = 0
                    frame_height = 0
                    if frame_for_autoclick is not None and hasattr(frame_for_autoclick, "shape") and len(frame_for_autoclick.shape) >= 2:
                        frame_height = int(frame_for_autoclick.shape[0])
                        frame_width = int(frame_for_autoclick.shape[1])
                    snapshot = self.auto_click_runtime.build_snapshot_from_launcher(
                        active_hero_present=False,
                        hero_decision=None,
                        decision_ready=False,
                        decision_started_at=time.monotonic(),
                        hand=None,
                        critical_error_flag=False,
                        critical_error_text=None,
                        action_panel_bbox=None,
                        monitor_width=frame_width,
                        monitor_height=frame_height,
                    )
                    auto_click_result = self.auto_click_runtime.step(snapshot, frame_bgr=frame_for_autoclick)
                except Exception:
                    auto_click_trace = traceback.format_exc(limit=8)
                    exception_text = auto_click_trace
                    analysis_text = f"{analysis_text}\n\n=== AUTOCLICK EXCEPTION ===\n{auto_click_trace}" if analysis_text else auto_click_trace
                    auto_click_result = self._auto_click_result_to_dict(None)
                    if isinstance(auto_click_result, dict):
                        auto_click_result["state"] = "ERROR"
                        auto_click_result["events"] = [
                            {
                                "name": "autoclick_exception",
                                "ts": time.monotonic(),
                                "payload": {"error": auto_click_trace.splitlines()[-1] if auto_click_trace else "autoclick exception"},
                            }
                        ]
            auto_click_payload = self._auto_click_result_to_dict(auto_click_result) if not isinstance(auto_click_result, dict) else auto_click_result
            if auto_click_payload.get("enabled"):
                analysis_text = (
                    f"{analysis_text}\n\n=== AUTOCLICK ===\n"
                    f"State: {auto_click_payload.get('state')}\n"
                    f"Plan: {auto_click_payload.get('plan_name') or '-'}\n"
                    f"Normalized: {auto_click_payload.get('normalized_action') or '-'}\n"
                    f"Raw: {auto_click_payload.get('raw_action') or '-'}\n"
                    f"Executed: {auto_click_payload.get('executed')} | Locked: {auto_click_payload.get('locked')}"
                ).strip()
                recommendation_payload["analysis_text"] = analysis_text
            render_state = None
            status = {
                "frame_id": result.analysis.frame_id,
                "street": result.analysis.street,
                "errors": [],
                "recommendation": dict(recommendation_payload),
                "analysis_text": analysis_text,
                "exception": exception_text,
                "auto_click": dict(auto_click_payload),
                "slot_scan_summary": list(scan_summary),
            }
            return frame, result, decision, render_state, status

        frame = slot_frames.get(processing_slot_id) or self._build_slot_frame(full_frame, processing_slot_id)
        if hasattr(self.storage, "set_active_slot"):
            self.storage.set_active_slot(processing_slot_id)
        slot_state = self.slot_states.get(processing_slot_id)
        if slot_state is not None:
            slot_state.last_frame_ts = getattr(frame, "timestamp", None)
        result = self.pipeline.process_frame(frame)
        if slot_state is not None:
            slot_state.is_active = bool(getattr(result.analysis, "active_hero_found", False))
            if slot_state.is_active:
                slot_state.last_active_hero_seen_at = time.monotonic()
            slot_state.current_hand_id = getattr(result.hand, "hand_id", None) if getattr(result, "hand", None) is not None else None
        decision = None
        recommendation_payload: Dict[str, Any]
        analysis_text = ""
        exception_text = ""
        auto_click_result = None
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
            elif self.auto_click_runtime is not None and bool(getattr(result.analysis, "active_hero_found", False)):
                decision = self.bridge.build_recommendation(result.analysis, result.hand)
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

        quick_gate_note = "Round-robin quick gate:\n" + "\n".join(
            f"- {item['slot_id']}: active_hero_found={item['active_hero_found']} count={item['active_hero_count']}"
            for item in scan_summary
        )
        analysis_text = f"{analysis_text}\n\n=== SLOT GATE ===\nProcessed slot: {processing_slot_id}\n{quick_gate_note}".strip()
        recommendation_payload["analysis_text"] = analysis_text

        if self.auto_click_runtime is not None:
            try:
                frame_for_autoclick = self._normalize_frame_for_autoclick(getattr(full_frame, "image", None))
                frame_width = 0
                frame_height = 0
                if frame_for_autoclick is not None and hasattr(frame_for_autoclick, "shape") and len(frame_for_autoclick.shape) >= 2:
                    frame_height = int(frame_for_autoclick.shape[0])
                    frame_width = int(frame_for_autoclick.shape[1])
                active_hero_present = bool(getattr(result.analysis, "active_hero_found", False))
                autoclick_decision, autoclick_reason = self._select_autoclick_decision(
                    live_decision=decision,
                    render_state=existing_render_state,
                    analysis_frame_id=getattr(result.analysis, "frame_id", None),
                    active_hero_present=active_hero_present,
                )
                if autoclick_reason:
                    note = f"AUTOCLICK_DECISION_OVERRIDE: {autoclick_reason}"
                    analysis_text = f"{analysis_text}\n\n{note}" if analysis_text else note
                action_panel_bbox = None
                snapshot = self.auto_click_runtime.build_snapshot_from_launcher(
                    active_hero_present=active_hero_present,
                    hero_decision=autoclick_decision,
                    decision_ready=autoclick_decision is not None and not bool(exception_text),
                    decision_started_at=self._resolve_auto_click_started_at(result),
                    hand=result.hand,
                    critical_error_flag=bool(exception_text),
                    critical_error_text=(exception_text.splitlines()[-1] if exception_text else None),
                    action_panel_bbox=action_panel_bbox,
                    monitor_width=frame_width,
                    monitor_height=frame_height,
                )
                auto_click_result = self.auto_click_runtime.step(
                    snapshot,
                    frame_bgr=frame_for_autoclick,
                )
            except Exception:
                auto_click_trace = traceback.format_exc(limit=8)
                exception_text = f"{exception_text}\n\n=== AUTOCLICK EXCEPTION ===\n{auto_click_trace}" if exception_text else auto_click_trace
                analysis_text = f"{analysis_text}\n\n=== AUTOCLICK EXCEPTION ===\n{auto_click_trace}" if analysis_text else auto_click_trace
                auto_click_result = self._auto_click_result_to_dict(None)
                if isinstance(auto_click_result, dict):
                    auto_click_result["state"] = "ERROR"
                    auto_click_result["events"] = [
                        {
                            "name": "autoclick_exception",
                            "ts": time.monotonic(),
                            "payload": {"error": auto_click_trace.splitlines()[-1] if auto_click_trace else "autoclick exception"},
                        }
                    ]

        auto_click_payload = self._auto_click_result_to_dict(auto_click_result) if not isinstance(auto_click_result, dict) else auto_click_result
        if auto_click_payload.get("enabled"):
            analysis_text = (
                f"{analysis_text}\n\n=== AUTOCLICK ===\n"
                f"State: {auto_click_payload.get('state')}\n"
                f"Plan: {auto_click_payload.get('plan_name') or '-'}\n"
                f"Normalized: {auto_click_payload.get('normalized_action') or '-'}\n"
                f"Raw: {auto_click_payload.get('raw_action') or '-'}\n"
                f"Executed: {auto_click_payload.get('executed')} | Locked: {auto_click_payload.get('locked')}"
            ).strip()
            recommendation_payload["analysis_text"] = analysis_text

        render_state = self._inject_recommendation(result, recommendation_payload, auto_click_payload=auto_click_payload)
        status = self._build_status(result, recommendation_payload, analysis_text=analysis_text, exception_text=exception_text, auto_click_payload=auto_click_payload)
        status["slot_scan_summary"] = list(scan_summary)
        status["processed_slot_id"] = processing_slot_id
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
    parser.add_argument("--autoclick", action="store_true", help="Enable auto click runtime after final HeroDecision")
    parser.add_argument(
        "--autoclick-model-path",
        type=str,
        default=r"C:\PokerAI\AI_detect\AutoClick\weights",
        help="Path to auto click YOLO model file or weights directory",
    )
    parser.add_argument("--autoclick-disable-idle", action="store_true", help="Disable idle mouse movement while auto click runtime is enabled")
    parser.add_argument("--slot-view", type=int, default=1, help="UI/processing slot index for crop mode (1-6)")
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
