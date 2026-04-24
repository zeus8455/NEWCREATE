from __future__ import annotations

import json
import logging
import math
import os
import random
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

try:
    import ctypes
except Exception:  # pragma: no cover
    ctypes = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import mss  # type: ignore
except Exception:  # pragma: no cover
    mss = None

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None

try:
    from decision_types import HeroDecision
except Exception:  # pragma: no cover
    HeroDecision = Any  # type: ignore[misc,assignment]

LOGGER = logging.getLogger(__name__)

BUTTON_FOLD = "FOLD"
BUTTON_33 = "33%"
BUTTON_50 = "50%"
BUTTON_70 = "70%"
BUTTON_98 = "98%"
BUTTON_CALL = "CALL"
BUTTON_RAISE = "Raise"
BUTTON_CHECK_FOLD = "Check/fold"
BUTTON_CHECK = "check"
BUTTON_CLASS_NAMES = (
    BUTTON_FOLD,
    BUTTON_33,
    BUTTON_50,
    BUTTON_70,
    BUTTON_98,
    BUTTON_CALL,
    BUTTON_RAISE,
    BUTTON_CHECK_FOLD,
    BUTTON_CHECK,
)

ACTION_CLICK_FOLD = "CLICK_FOLD"
ACTION_CLICK_CHECK = "CLICK_CHECK"
ACTION_CLICK_CALL = "CLICK_CALL"
ACTION_CLICK_RAISE_ONLY = "CLICK_RAISE_ONLY"
ACTION_CLICK_SIZE_THEN_RAISE_33 = "CLICK_SIZE_THEN_RAISE_33"
ACTION_CLICK_SIZE_THEN_RAISE_50 = "CLICK_SIZE_THEN_RAISE_50"
ACTION_CLICK_SIZE_THEN_RAISE_70 = "CLICK_SIZE_THEN_RAISE_70"
ACTION_CLICK_98_THEN_RAISE = "CLICK_98_THEN_RAISE"
ACTION_CLICK_CHECK_FOLD = "CLICK_CHECK_FOLD"
ACTION_NO_AUTOCLICK = "NO_AUTOCLICK"

STATE_IDLE = "IDLE"
STATE_ACTIVE_HERO_WAITING_DECISION = "ACTIVE_HERO_WAITING_DECISION"
STATE_DECISION_READY = "DECISION_READY"
STATE_EXECUTING_PLAN = "EXECUTING_PLAN"
STATE_POST_CLICK_COOLDOWN = "POST_CLICK_COOLDOWN"
STATE_FAILSAFE_EXECUTION = "FAILSAFE_EXECUTION"
STATE_LOCKED_UNTIL_RESET = "LOCKED_UNTIL_RESET"

RAW_THREEBET_ACTIONS = {"3bet", "threebet", "three_bet"}
RAW_FOURBET_ACTIONS = {"4bet", "fourbet", "four_bet", "cold_4bet", "cold4bet"}
RAW_FIVEBET_ACTIONS = {"5bet", "5bet_jam", "fivebet", "five_bet", "jam", "all_in", "all-in"}
RAW_ISO_ACTIONS = {"iso_raise", "iso", "iso-raise"}
SIZE_TO_BUTTON = {33: BUTTON_33, 50: BUTTON_50, 70: BUTTON_70, 98: BUTTON_98}
AGGRESSIVE_NORMALIZED_ACTIONS = {
    "raise_only",
    "raise_33",
    "raise_50",
    "raise_70",
    "raise_98",
    "iso_raise_98",
    "threebet_98",
    "fourbet_98",
    "fivebet_98",
}
RERAISE_SPOT_ALLOWED_ACTIONS = {
    "opener_vs_3bet": {"fourbet_98"},
    "threebettor_vs_4bet": {"fivebet_98"},
    "fourbettor_vs_5bet": {"fivebet_98"},
}
IDENTITY_KEYS = (
    "solver_fingerprint",
    "decision_id",
    "source_frame_id",
    "node_type",
    "opener_pos",
    "three_bettor_pos",
    "four_bettor_pos",
    "limpers",
    "callers",
)


def canonicalize_button_class_name(name: object) -> Optional[str]:
    raw = str(name or "").strip()
    if not raw:
        return None
    lowered = raw.lower().replace("_", "").replace(" ", "").replace("-", "")
    lowered = lowered.replace("％", "%")
    aliases = {
        "fold": BUTTON_FOLD,
        "f": BUTTON_FOLD,
        "33": BUTTON_33,
        "33%": BUTTON_33,
        "50": BUTTON_50,
        "50%": BUTTON_50,
        "70": BUTTON_70,
        "70%": BUTTON_70,
        "98": BUTTON_98,
        "98%": BUTTON_98,
        "call": BUTTON_CALL,
        "c": BUTTON_CALL,
        "raise": BUTTON_RAISE,
        "r": BUTTON_RAISE,
        "check/fold": BUTTON_CHECK_FOLD,
        "checkfold": BUTTON_CHECK_FOLD,
        "check": BUTTON_CHECK,
    }
    if lowered in aliases:
        return aliases[lowered]
    if raw in BUTTON_CLASS_NAMES:
        return raw
    return None


def resolve_model_path(path_like: str | os.PathLike[str]) -> Path:
    path = Path(path_like)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Button model path does not exist: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Button model path is not a file or directory: {path}")
    for candidate in (path / "best.pt", path / "last.pt"):
        if candidate.exists() and candidate.is_file():
            return candidate
    pt_files = sorted(path.glob("*.pt"))
    if pt_files:
        return pt_files[0]
    raise FileNotFoundError(f"No .pt files found in model directory: {path}")


def _json_safe(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _stable_json_bytes(value: object) -> bytes:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _stable_hash(prefix: str, payload: object) -> str:
    digest = hashlib.sha1()
    digest.update(prefix.encode("utf-8"))
    digest.update(b"|")
    digest.update(_stable_json_bytes(payload))
    return digest.hexdigest()


def _first_non_empty(*values: object) -> Optional[str]:
    for value in values:
        if value in (None, ""):
            continue
        return str(value)
    return None


@dataclass(slots=True)
class ButtonDetection:
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    center_x: float
    center_y: float
    frame_ts: float
    source_class_name: Optional[str] = None
    local_bbox: Optional[Tuple[float, float, float, float]] = None
    global_bbox: Optional[Tuple[float, float, float, float]] = None
    coordinate_space: str = "global"
    slot_id: Optional[str] = None

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])


@dataclass(slots=True)
class AutoClickSnapshot:
    active_hero_present: bool
    hero_decision: Optional[HeroDecision]
    decision_ready: bool
    decision_started_at: float
    critical_error_flag: bool = False
    critical_error_text: Optional[str] = None
    total_pot_bb: Optional[float] = None
    hand_id: Optional[str] = None
    street: str = "preflop"
    solver_context_meta: Dict[str, object] = field(default_factory=dict)
    action_panel_bbox: Optional[Tuple[float, float, float, float]] = None
    monitor_name: str = "primary"
    monitor_left: int = 0
    monitor_top: int = 0
    monitor_width: int = 0
    monitor_height: int = 0
    frame_source: str = "launcher_frame"
    slot_id: Optional[str] = None
    slot_bbox: Optional[Tuple[float, float, float, float]] = None
    slot_global_offset: Tuple[float, float] = (0.0, 0.0)


@dataclass(slots=True)
class AutoClickEvent:
    name: str
    ts: float
    payload: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class AutoClickResult:
    state: str
    executed: bool = False
    locked: bool = False
    plan_name: Optional[str] = None
    normalized_action: Optional[str] = None
    raw_action: Optional[str] = None
    events: List[AutoClickEvent] = field(default_factory=list)


@dataclass(slots=True)
class AutoClickPlan:
    plan_name: str
    normalized_action: str
    raw_action: str
    primary_button: Optional[str]
    secondary_button: Optional[str] = None
    allow_scroll_between: bool = False
    is_failsafe: bool = False


@dataclass(slots=True)
class AutoClickSlotState:
    state: str = STATE_IDLE
    wait_started_at: Optional[float] = None
    error_started_at: Optional[float] = None
    post_click_until: float = 0.0
    last_cycle_key: Optional[str] = None
    last_hand_id: Optional[str] = None
    last_executed_decision_id: Optional[str] = None
    last_executed_solver_fingerprint: Optional[str] = None
    last_executed_source_frame_id: Optional[str] = None
    last_executed_hand_id: Optional[str] = None
    last_executed_street: Optional[str] = None
    last_executed_plan_name: Optional[str] = None
    last_executed_at: Optional[float] = None
    last_executed_decision_guard: Optional[str] = None
    locked_decision_id: Optional[str] = None
    locked_decision_guard: Optional[str] = None
    last_plan_name: Optional[str] = None
    last_normalized_action: Optional[str] = None
    last_raw_action: Optional[str] = None
    last_click_at: Optional[float] = None


@dataclass(slots=True)
class AutoClickConfig:
    enabled: bool = True
    enable_idle_movement: bool = True
    button_model_path: str = r"C:\PokerAI\AI_detect\AutoClick\weights"
    auto_build_button_detector: bool = True
    force_primary_monitor_capture: bool = True
    debug_root_dir: str = str(Path.cwd() / "autoclick_debug")
    detector_conf_default: float = 0.40
    detector_conf_raise: float = 0.35
    detector_conf_check: float = 0.35
    button_detection_ttl_ms: int = 900
    nms_iou_threshold: float = 0.45
    decision_settle_delay_sec: float = 0.0
    timeout_default_sec: float = 9.0
    execution_timeout_no_click_sec: float = 9.0
    timeout_big_pot_extra_sec: float = 9.0
    timeout_big_pot_threshold_bb: float = 20.0
    error_grace_sec: float = 4.0
    retry_count_max: int = 3
    retry_total_budget_ms: int = 900
    retry_sleep_ms_min: int = 40
    retry_sleep_ms_max: int = 120
    scroll_enabled_probability: float = 0.0
    scroll_direction_up_probability: float = 0.5
    scroll_steps_min: int = 1
    scroll_steps_max: int = 2
    scroll_pause_ms_min: int = 25
    scroll_pause_ms_max: int = 60
    move_duration_ms_min: int = 190
    move_duration_ms_max: int = 620
    target_jitter_px_min: int = 2
    target_jitter_px_max: int = 7
    click_target_inner_padding_px_min: int = 3
    click_target_inner_padding_px_max: int = 10
    click_target_inner_padding_ratio_min: float = 0.02
    click_target_inner_padding_ratio_max: float = 0.05
    mouse_curve_offset_px_min: int = 18
    mouse_curve_offset_px_max: int = 125
    mouse_overshoot_probability: float = 0.42
    mouse_overshoot_px_min: int = 6
    mouse_overshoot_px_max: int = 34
    click_down_up_delay_ms_min: int = 28
    click_down_up_delay_ms_max: int = 65
    post_click_cooldown_sec_min: float = 1.0
    post_click_cooldown_sec_max: float = 1.8
    idle_pause_sec_min: float = 11.0
    idle_pause_sec_max: float = 30.0
    idle_move_distance_px_min: int = 60
    idle_move_distance_px_max: int = 280
    idle_move_duration_ms_min: int = 380
    idle_move_duration_ms_max: int = 1200
    secondary_click_settle_ms_min: int = 15
    secondary_click_settle_ms_max: int = 45
    secondary_retry_recapture: bool = True
    disable_aggressive_auto_actions: bool = False
    require_identity_match: bool = True
    log_prefix: str = "[AutoClick]"


class ButtonDetectorProtocol(Protocol):
    def detect_buttons(self, frame_bgr: Any) -> List[ButtonDetection]: ...


class MouseBackendProtocol(Protocol):
    def get_position(self) -> Tuple[int, int]: ...
    def move_to(self, x: int, y: int) -> None: ...
    def left_down(self) -> None: ...
    def left_up(self) -> None: ...
    def wheel(self, amount: int) -> None: ...


class NoopMouseBackend:
    def get_position(self) -> Tuple[int, int]:
        return (0, 0)

    def move_to(self, x: int, y: int) -> None:
        return None

    def left_down(self) -> None:
        return None

    def left_up(self) -> None:
        return None

    def wheel(self, amount: int) -> None:
        return None


class WindowsMouseBackend:
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_WHEEL = 0x0800

    def __init__(self) -> None:
        if ctypes is None:
            raise RuntimeError("ctypes unavailable")
        self.user32 = ctypes.windll.user32  # type: ignore[attr-defined]

    def get_position(self) -> Tuple[int, int]:
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        point = POINT()
        self.user32.GetCursorPos(ctypes.byref(point))
        return int(point.x), int(point.y)

    def move_to(self, x: int, y: int) -> None:
        self.user32.SetCursorPos(int(x), int(y))

    def left_down(self) -> None:
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

    def left_up(self) -> None:
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def wheel(self, amount: int) -> None:
        self.user32.mouse_event(self.MOUSEEVENTF_WHEEL, 0, 0, int(amount), 0)


class YoloButtonDetector:
    def __init__(self, model_path: str | os.PathLike[str], conf: float = 0.40) -> None:
        if YOLO is None:
            raise RuntimeError("ultralytics is unavailable")
        self.model_path = resolve_model_path(model_path)
        self.conf = float(conf)
        self.model = YOLO(str(self.model_path))

    def detect_buttons(self, frame_bgr: Any) -> List[ButtonDetection]:
        if frame_bgr is None or cv2 is None or np is None:
            return []
        ts = time.time()
        results = self.model.predict(frame_bgr, conf=self.conf, verbose=False)
        detections: List[ButtonDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            names = getattr(result, "names", {}) or {}
            if boxes is None:
                continue
            xyxy = getattr(boxes, "xyxy", None)
            cls = getattr(boxes, "cls", None)
            conf = getattr(boxes, "conf", None)
            if xyxy is None or cls is None or conf is None:
                continue
            for bbox, cls_idx, score in zip(xyxy.tolist(), cls.tolist(), conf.tolist()):
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [float(v) for v in bbox]
                raw_name = str(names.get(int(cls_idx), int(cls_idx)))
                canonical = canonicalize_button_class_name(raw_name)
                if canonical is None:
                    continue
                detections.append(
                    ButtonDetection(
                        class_name=canonical,
                        confidence=float(score),
                        bbox=(x1, y1, x2, y2),
                        center_x=(x1 + x2) / 2.0,
                        center_y=(y1 + y2) / 2.0,
                        frame_ts=ts,
                        source_class_name=raw_name,
                    )
                )
        return detections


def build_default_button_detector(config: AutoClickConfig) -> ButtonDetectorProtocol:
    return YoloButtonDetector(config.button_model_path, conf=config.detector_conf_default)


class AutoClickRuntime:
    def __init__(
        self,
        config: Optional[AutoClickConfig] = None,
        *,
        mouse_backend: Optional[MouseBackendProtocol] = None,
        button_detector: Optional[ButtonDetectorProtocol] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.config = config or AutoClickConfig()
        self.rng = rng or random.Random()
        self.mouse = mouse_backend or self._build_default_mouse_backend()
        self.button_detector = button_detector
        if self.button_detector is None and self.config.auto_build_button_detector:
            try:
                self.button_detector = build_default_button_detector(self.config)
            except Exception:
                LOGGER.exception("%s failed to initialize button detector", self.config.log_prefix)
                self.button_detector = None

        self.slot_states: Dict[str, AutoClickSlotState] = {}
        self._active_slot_key: str = "single"

        self.state = STATE_IDLE
        self.wait_started_at: Optional[float] = None
        self.error_started_at: Optional[float] = None
        self.post_click_until = 0.0
        self.last_cycle_key: Optional[str] = None
        self.last_hand_id: Optional[str] = None
        self.last_executed_decision_id: Optional[str] = None
        self.last_executed_solver_fingerprint: Optional[str] = None
        self.last_executed_source_frame_id: Optional[str] = None
        self.last_executed_hand_id: Optional[str] = None
        self.last_executed_street: Optional[str] = None
        self.last_executed_plan_name: Optional[str] = None
        self.last_executed_at: Optional[float] = None
        self.last_executed_decision_guard: Optional[str] = None
        self.locked_decision_id: Optional[str] = None
        self.locked_decision_guard: Optional[str] = None
        self.last_plan_name: Optional[str] = None
        self.last_normalized_action: Optional[str] = None
        self.last_raw_action: Optional[str] = None
        self.last_click_at: Optional[float] = None
        self.next_idle_at = self._schedule_next_idle(time.monotonic())
        self.recent_events: List[AutoClickEvent] = []
        self.debug_root = Path(self.config.debug_root_dir).expanduser().resolve()
        self.debug_root.mkdir(parents=True, exist_ok=True)

    def _build_default_mouse_backend(self) -> MouseBackendProtocol:
        if os.name == "nt" and ctypes is not None:
            try:
                return WindowsMouseBackend()
            except Exception:
                LOGGER.exception("%s failed to initialize Windows mouse backend", self.config.log_prefix)
        return NoopMouseBackend()

    def _slot_key_from_snapshot(self, snapshot: Optional[AutoClickSnapshot]) -> str:
        if snapshot is None:
            return "single"
        raw_slot = getattr(snapshot, "slot_id", None)
        if raw_slot not in (None, ""):
            return str(raw_slot).strip() or "single"
        return "single"

    def _copy_current_to_slot_state(self, target: AutoClickSlotState) -> None:
        target.state = self.state
        target.wait_started_at = self.wait_started_at
        target.error_started_at = self.error_started_at
        target.post_click_until = self.post_click_until
        target.last_cycle_key = self.last_cycle_key
        target.last_hand_id = self.last_hand_id
        target.last_executed_decision_id = self.last_executed_decision_id
        target.last_executed_solver_fingerprint = self.last_executed_solver_fingerprint
        target.last_executed_source_frame_id = self.last_executed_source_frame_id
        target.last_executed_hand_id = self.last_executed_hand_id
        target.last_executed_street = self.last_executed_street
        target.last_executed_plan_name = self.last_executed_plan_name
        target.last_executed_at = self.last_executed_at
        target.last_executed_decision_guard = self.last_executed_decision_guard
        target.locked_decision_id = self.locked_decision_id
        target.locked_decision_guard = self.locked_decision_guard
        target.last_plan_name = self.last_plan_name
        target.last_normalized_action = self.last_normalized_action
        target.last_raw_action = self.last_raw_action
        target.last_click_at = self.last_click_at

    def _load_slot_state_to_current(self, source: AutoClickSlotState) -> None:
        self.state = source.state
        self.wait_started_at = source.wait_started_at
        self.error_started_at = source.error_started_at
        self.post_click_until = source.post_click_until
        self.last_cycle_key = source.last_cycle_key
        self.last_hand_id = source.last_hand_id
        self.last_executed_decision_id = source.last_executed_decision_id
        self.last_executed_solver_fingerprint = source.last_executed_solver_fingerprint
        self.last_executed_source_frame_id = source.last_executed_source_frame_id
        self.last_executed_hand_id = source.last_executed_hand_id
        self.last_executed_street = source.last_executed_street
        self.last_executed_plan_name = source.last_executed_plan_name
        self.last_executed_at = source.last_executed_at
        self.last_executed_decision_guard = source.last_executed_decision_guard
        self.locked_decision_id = source.locked_decision_id
        self.locked_decision_guard = source.locked_decision_guard
        self.last_plan_name = source.last_plan_name
        self.last_normalized_action = source.last_normalized_action
        self.last_raw_action = source.last_raw_action
        self.last_click_at = source.last_click_at

    def _sync_active_slot_state(self) -> None:
        target = self.slot_states.setdefault(self._active_slot_key, AutoClickSlotState())
        self._copy_current_to_slot_state(target)

    def _activate_slot_state(self, slot_key: str) -> None:
        normalized = str(slot_key or "single").strip() or "single"
        if self._active_slot_key != normalized:
            self._sync_active_slot_state()
            self._active_slot_key = normalized
        source = self.slot_states.setdefault(self._active_slot_key, AutoClickSlotState())
        self._load_slot_state_to_current(source)


    def _extract_launcher_decision_action(self, decision: Optional[HeroDecision]) -> str:
        if decision is None:
            return ""
        for source in (
            getattr(decision, "preflop", None),
            getattr(decision, "postflop", None),
        ):
            if source is None:
                continue
            action = getattr(source, "action", None)
            if action not in (None, ""):
                return str(action).strip().lower()
        for attr_name in ("engine_action", "reason"):
            value = getattr(decision, attr_name, None)
            if value not in (None, ""):
                return str(value).strip().lower()
        debug = getattr(decision, "debug", {}) or {}
        if isinstance(debug, dict):
            for key in ("engine_action", "recommended_action", "action", "raw_action"):
                value = debug.get(key)
                if value not in (None, ""):
                    return str(value).strip().lower()
        return ""

    def _sanitize_launcher_decision(
        self,
        hero_decision: Optional[HeroDecision],
        decision_ready: bool,
    ) -> Tuple[Optional[HeroDecision], bool, Optional[str]]:
        if hero_decision is None:
            return None, False, "no_hero_decision"

        engine_action = str(getattr(hero_decision, "engine_action", "") or "").strip().lower()
        extracted_action = self._extract_launcher_decision_action(hero_decision)
        debug = getattr(hero_decision, "debug", {}) or {}
        debug_meta = debug.get("meta") if isinstance(debug, dict) else {}
        if not isinstance(debug_meta, dict):
            debug_meta = {}
        solver_status = ""
        for candidate in (
            debug.get("solver_status") if isinstance(debug, dict) else None,
            debug_meta.get("solver_status"),
            debug.get("status") if isinstance(debug, dict) else None,
            debug_meta.get("status"),
        ):
            if candidate not in (None, ""):
                solver_status = str(candidate).strip().lower()
                break
        fallback_reason = str(getattr(hero_decision, "fallback_reason", "") or "").strip().lower()
        reason_text = str(getattr(hero_decision, "reason", "") or "").strip().lower()

        unusable_markers = {
            "solver_unavailable",
            "result=null",
            "result_null",
            "null_result",
            "no_result",
            "empty_result",
        }
        joined_reason = " | ".join(part for part in (solver_status, fallback_reason, reason_text) if part)
        has_unusable_marker = any(marker in joined_reason for marker in unusable_markers)
        has_clickable_action = bool(engine_action or extracted_action)
        if not has_clickable_action:
            return None, False, "missing_clickable_action"
        if has_unusable_marker and engine_action in ("", "none"):
            return None, False, "solver_unavailable"
        if not decision_ready:
            return hero_decision, False, None
        return hero_decision, True, None

    def build_snapshot_from_launcher(
        self,
        *,
        active_hero_present: bool,
        hero_decision: Optional[HeroDecision],
        decision_ready: bool,
        decision_started_at: float,
        hand: Any = None,
        critical_error_flag: bool = False,
        critical_error_text: Optional[str] = None,
        action_panel_bbox: Optional[Tuple[float, float, float, float]] = None,
        monitor_left: int = 0,
        monitor_top: int = 0,
        monitor_width: int = 0,
        monitor_height: int = 0,
        monitor_name: str = "primary",
        slot_id: Optional[str] = None,
        slot_bbox: Optional[Tuple[float, float, float, float]] = None,
        frame_source: str = "launcher_frame",
    ) -> AutoClickSnapshot:
        hero_decision, decision_ready, stripped_reason = self._sanitize_launcher_decision(hero_decision, decision_ready)
        street = "preflop"
        hand_id = None
        total_pot_bb = None
        meta: Dict[str, object] = {}
        normalized_slot_bbox = self._normalize_slot_bbox(slot_bbox)
        normalized_slot_id = str(slot_id).strip() if slot_id not in (None, "") else None
        slot_offset = (0.0, 0.0)
        if normalized_slot_bbox is not None:
            slot_offset = (float(normalized_slot_bbox[0]), float(normalized_slot_bbox[1]))
            if normalized_slot_id is None:
                normalized_slot_id = "table_01"
            meta["slot_bbox"] = [float(v) for v in normalized_slot_bbox]
            meta["slot_global_offset"] = [float(slot_offset[0]), float(slot_offset[1])]
        if normalized_slot_id:
            meta["slot_id"] = normalized_slot_id
        if stripped_reason:
            meta["autoclick_decision_absent_reason"] = stripped_reason
        if hero_decision is not None:
            street = str(getattr(hero_decision, "street", street) or street).lower()
            debug = getattr(hero_decision, "debug", {}) or {}
            if isinstance(debug, dict):
                for key in IDENTITY_KEYS:
                    if key in debug and debug[key] not in (None, ""):
                        meta[str(key)] = debug[key]
                for key in ("engine_action", "recommended_action", "hero_pos", "street"):
                    if key in debug and debug[key] not in (None, ""):
                        meta[str(key)] = debug[key]
                nested_meta = debug.get("meta") or {}
                if isinstance(nested_meta, dict):
                    for key in IDENTITY_KEYS:
                        if key in nested_meta and key not in meta and nested_meta[key] not in (None, ""):
                            meta[str(key)] = nested_meta[key]
            for key in ("solver_fingerprint", "decision_id", "source_frame_id", "engine_action"):
                value = getattr(hero_decision, key, None)
                if value is not None and key not in meta:
                    meta[key] = value
        if hand is not None:
            hand_id = getattr(hand, "hand_id", None)
            street_state = getattr(hand, "street_state", None)
            if isinstance(street_state, dict):
                street = str(street_state.get("current_street") or street).lower()
            action_state = getattr(hand, "action_state", None) or {}
            if isinstance(action_state, dict):
                for key in IDENTITY_KEYS:
                    value = action_state.get(key)
                    if value not in (None, "") and key not in meta:
                        meta[key] = value
                if "solver_fingerprint" in action_state:
                    meta.setdefault("solver_fingerprint", action_state.get("solver_fingerprint"))
                if "source_frame_id" in action_state:
                    meta.setdefault("source_frame_id", action_state.get("source_frame_id"))
            amount_state = getattr(hand, "amount_state", None) or getattr(hand, "table_amount_state", None) or {}
            if isinstance(amount_state, dict):
                total_pot = amount_state.get("total_pot") or {}
                if isinstance(total_pot, dict) and total_pot.get("amount_bb") is not None:
                    try:
                        total_pot_bb = float(total_pot.get("amount_bb"))
                    except Exception:
                        total_pot_bb = None
                if total_pot_bb is None and amount_state.get("total_pot_bb") is not None:
                    try:
                        total_pot_bb = float(amount_state.get("total_pot_bb"))
                    except Exception:
                        total_pot_bb = None
            for attr_name in ("source_frame_id", "solver_fingerprint", "decision_id"):
                value = getattr(hand, attr_name, None)
                if value not in (None, "") and attr_name not in meta:
                    meta[attr_name] = value
        return AutoClickSnapshot(
            active_hero_present=bool(active_hero_present),
            hero_decision=hero_decision,
            decision_ready=bool(decision_ready),
            decision_started_at=float(decision_started_at),
            critical_error_flag=bool(critical_error_flag),
            critical_error_text=critical_error_text,
            total_pot_bb=total_pot_bb,
            hand_id=None if hand_id is None else str(hand_id),
            street=street,
            solver_context_meta=meta,
            action_panel_bbox=action_panel_bbox,
            monitor_left=int(monitor_left),
            monitor_top=int(monitor_top),
            monitor_width=int(monitor_width),
            monitor_height=int(monitor_height),
            monitor_name=str(monitor_name or "primary"),
            frame_source=str(frame_source or "launcher_frame"),
            slot_id=normalized_slot_id,
            slot_bbox=normalized_slot_bbox,
            slot_global_offset=slot_offset,
        )

    def step(
        self,
        snapshot: AutoClickSnapshot,
        *,
        frame_bgr: Any = None,
        detections: Optional[Sequence[ButtonDetection]] = None,
    ) -> AutoClickResult:
        now = time.monotonic()
        events: List[AutoClickEvent] = []
        self._activate_slot_state(self._slot_key_from_snapshot(snapshot))
        if not self.config.enabled:
            return self._finalize(AutoClickResult(state=self.state, events=events))

        cycle_key = self._cycle_key(snapshot)
        if self._should_reset_cycle(snapshot, cycle_key):
            events.extend(self._reset_cycle(snapshot, cycle_key, now))

        if not snapshot.active_hero_present:
            self.state = STATE_IDLE
            self.locked_decision_id = None
            self.locked_decision_guard = None
            if self.config.enable_idle_movement:
                self._maybe_idle_move(now, events)
            return self._finalize(AutoClickResult(state=self.state, events=events))

        if self.wait_started_at is None:
            self.wait_started_at = float(snapshot.decision_started_at or now)
            events.append(self._event("active_hero_detected", hand_id=snapshot.hand_id or ""))
            events.append(self._event("decision_wait_started", started_at=float(self.wait_started_at)))

        frame_used, capture_meta = self._get_detection_frame(snapshot, frame_bgr)
        if detections is None:
            raw_detections: Sequence[ButtonDetection] = []
            if self.button_detector is not None and frame_used is not None:
                try:
                    raw_detections = self.button_detector.detect_buttons(frame_used)
                except Exception as exc:
                    LOGGER.exception("%s button detector failed", self.config.log_prefix)
                    events.append(self._event("button_detector_error", error=str(exc)))
                    raw_detections = []
        else:
            raw_detections = list(detections)
        filtered = self._prepare_detections(raw_detections, snapshot)
        self._write_debug_observation(snapshot, frame_used, raw_detections, filtered, capture_meta, events=events)

        if now < self.post_click_until:
            self.state = STATE_POST_CLICK_COOLDOWN
            return self._finalize(
                AutoClickResult(
                    state=self.state,
                    executed=False,
                    locked=True,
                    plan_name=self.last_plan_name,
                    normalized_action=self.last_normalized_action,
                    raw_action=self.last_raw_action,
                    events=events,
                )
            )

        if snapshot.critical_error_flag:
            if self.error_started_at is None:
                self.error_started_at = now
                events.append(self._event("critical_error_detected", error=snapshot.critical_error_text or "critical_error"))
            if (now - self.error_started_at) < float(self.config.error_grace_sec):
                self.state = STATE_ACTIVE_HERO_WAITING_DECISION
                events.append(self._event("critical_error_grace_wait", elapsed_sec=round(now - self.error_started_at, 3)))
                return self._finalize(AutoClickResult(state=self.state, events=events))
            self.state = STATE_FAILSAFE_EXECUTION
            plan = self._build_failsafe_plan(filtered, reason=snapshot.critical_error_text or "critical_error")
            executed = self._execute_plan(plan, filtered, snapshot, events)
            self._write_debug_plan(snapshot, plan, filtered, executed, events=events)
            return self._finalize(self._result_from_state(plan, executed, events, locked=executed))

        timeout_sec = float(self.config.timeout_default_sec)
        try:
            if snapshot.total_pot_bb is not None and float(snapshot.total_pot_bb) > float(self.config.timeout_big_pot_threshold_bb):
                timeout_sec += float(self.config.timeout_big_pot_extra_sec)
        except Exception:
            pass

        elapsed_since_wait = now - float(self.wait_started_at or now)
        if self.last_click_at is None and elapsed_since_wait >= float(self.config.execution_timeout_no_click_sec):
            self.state = STATE_FAILSAFE_EXECUTION
            events.append(
                self._event(
                    "execution_timeout_failsafe_triggered",
                    elapsed_sec=round(elapsed_since_wait, 3),
                    timeout_sec=float(self.config.execution_timeout_no_click_sec),
                    hand_id=snapshot.hand_id or "",
                    street=snapshot.street,
                    decision_ready=bool(snapshot.decision_ready),
                )
            )
            plan = self._build_failsafe_plan(filtered, reason="execution_timeout_no_click")
            executed = self._execute_plan(plan, filtered, snapshot, events)
            self._write_debug_plan(snapshot, plan, filtered, executed, events=events)
            return self._finalize(self._result_from_state(plan, executed, events, locked=executed))

        if not snapshot.decision_ready or snapshot.hero_decision is None:
            if elapsed_since_wait >= timeout_sec:
                self.state = STATE_FAILSAFE_EXECUTION
                plan = self._build_failsafe_plan(filtered, reason="decision_timeout")
                executed = self._execute_plan(plan, filtered, snapshot, events)
                self._write_debug_plan(snapshot, plan, filtered, executed, events=events)
                return self._finalize(self._result_from_state(plan, executed, events, locked=executed))
            self.state = STATE_ACTIVE_HERO_WAITING_DECISION
            return self._finalize(AutoClickResult(state=self.state, events=events))

        self.state = STATE_DECISION_READY
        raw_action = self._resolve_raw_action(snapshot, snapshot.hero_decision)
        normalized = self._normalize_action(snapshot)
        plan = self._build_click_plan(normalized, raw_action)
        decision_memory = self._build_decision_memory(snapshot, plan)
        execution_token = str(decision_memory.get("execution_token") or "")
        decision_id = str(decision_memory.get("decision_id") or "")
        decision_guard = str(decision_memory.get("guard_key") or "")
        self.last_plan_name = plan.plan_name
        self.last_normalized_action = plan.normalized_action
        self.last_raw_action = plan.raw_action
        events.append(self._event("decision_received", raw_action=raw_action, normalized_action=normalized, street=snapshot.street, decision_id=decision_id))
        events.append(
            self._event(
                "click_plan_built",
                plan_name=plan.plan_name,
                primary_button=plan.primary_button or "",
                secondary_button=plan.secondary_button or "",
                decision_id=decision_id,
                execution_token=execution_token,
            )
        )

        if plan.plan_name == ACTION_NO_AUTOCLICK:
            events.append(self._event("unsupported_auto_action", raw_action=raw_action, normalized_action=normalized))
            self._write_debug_plan(snapshot, plan, filtered, False, events=events)
            return self._finalize(
                AutoClickResult(
                    state=self.state,
                    executed=False,
                    locked=False,
                    plan_name=ACTION_NO_AUTOCLICK,
                    normalized_action=normalized,
                    raw_action=raw_action,
                    events=events,
                )
            )

        identity_ok, identity_info = self._validate_snapshot_identity(snapshot)
        if self.config.require_identity_match and not identity_ok:
            events.append(self._event("identity_mismatch_blocked", **identity_info))
            self._write_debug_plan(snapshot, plan, filtered, False, events=events)
            return self._finalize(
                AutoClickResult(
                    state=self.state,
                    executed=False,
                    locked=False,
                    plan_name=plan.plan_name,
                    normalized_action=normalized,
                    raw_action=raw_action,
                    events=events,
                )
            )

        reraise_ok, reraise_info = self._validate_reraise_spot_action(snapshot, normalized, raw_action)
        if reraise_info.get("guard_applied"):
            guard_event_name = "reraise_spot_guard_allowed" if reraise_ok else "reraise_spot_guard_blocked"
            guard_event_payload = dict(reraise_info)
            guard_event_payload["decision_id"] = decision_id
            guard_event_payload["plan_name"] = plan.plan_name
            events.append(self._event(guard_event_name, **guard_event_payload))
        if not reraise_ok:
            self._write_debug_plan(snapshot, plan, filtered, False, events=events)
            return self._finalize(
                AutoClickResult(
                    state=self.state,
                    executed=False,
                    locked=False,
                    plan_name=ACTION_NO_AUTOCLICK,
                    normalized_action=normalized,
                    raw_action=raw_action,
                    events=events,
                )
            )

        if decision_guard and self.locked_decision_guard == decision_guard:
            self.state = STATE_LOCKED_UNTIL_RESET
            events.append(
                self._event(
                    "decision_id_reused_click_blocked",
                    decision_id=decision_id,
                    last_executed_decision_id=self.last_executed_decision_id or "",
                    guard_source=decision_memory.get("guard_source") or "",
                    execution_token=execution_token,
                )
            )
            return self._finalize(
                AutoClickResult(
                    state=self.state,
                    executed=False,
                    locked=True,
                    plan_name=self.last_plan_name,
                    normalized_action=self.last_normalized_action,
                    raw_action=self.last_raw_action,
                    events=events,
                )
            )

        if decision_guard and self.last_executed_decision_guard == decision_guard:
            events.append(
                self._event(
                    "decision_id_reused_click_blocked",
                    decision_id=decision_id,
                    last_executed_decision_id=self.last_executed_decision_id or "",
                    guard_source=decision_memory.get("guard_source") or "",
                    execution_token=execution_token,
                )
            )
            return self._finalize(
                AutoClickResult(
                    state=self.state,
                    executed=False,
                    locked=False,
                    plan_name=plan.plan_name,
                    normalized_action=plan.normalized_action,
                    raw_action=plan.raw_action,
                    events=events,
                )
            )

        if decision_id and self.last_executed_decision_id and self.last_executed_decision_id != decision_id:
            events.append(
                self._event(
                    "decision_id_changed_new_click_allowed",
                    decision_id=decision_id,
                    last_executed_decision_id=self.last_executed_decision_id,
                )
            )

        settle_deadline = float(self.wait_started_at or now) + float(self.config.decision_settle_delay_sec)
        if now < settle_deadline:
            events.append(self._event("decision_settle_wait", remaining_sec=round(settle_deadline - now, 3)))
            return self._finalize(
                AutoClickResult(
                    state=self.state,
                    executed=False,
                    locked=False,
                    plan_name=plan.plan_name,
                    normalized_action=plan.normalized_action,
                    raw_action=plan.raw_action,
                    events=events,
                )
            )

        executed = self._execute_plan(plan, filtered, snapshot, events)
        self._write_debug_plan(snapshot, plan, filtered, executed, events=events)
        if executed:
            self._record_successful_execution(decision_memory, events)
        return self._finalize(self._result_from_state(plan, executed, events, locked=executed))

    def reset(self) -> None:
        self.state = STATE_IDLE
        self.wait_started_at = None
        self.error_started_at = None
        self.post_click_until = 0.0
        self.last_cycle_key = None
        self.last_hand_id = None
        self.last_executed_decision_id = None
        self.last_executed_solver_fingerprint = None
        self.last_executed_source_frame_id = None
        self.last_executed_hand_id = None
        self.last_executed_street = None
        self.last_executed_plan_name = None
        self.last_executed_at = None
        self.last_executed_decision_guard = None
        self.locked_decision_id = None
        self.locked_decision_guard = None
        self.last_plan_name = None
        self.last_normalized_action = None
        self.last_raw_action = None
        self.last_click_at = None
        self.slot_states.clear()
        self._active_slot_key = "single"
        self._sync_active_slot_state()

    def get_recent_events(self) -> List[AutoClickEvent]:
        return list(self.recent_events)

    def _cycle_key(self, snapshot: AutoClickSnapshot) -> str:
        slot_prefix = f"{snapshot.slot_id or 'single'}:"
        if snapshot.hand_id:
            return f"{slot_prefix}hand:{snapshot.hand_id}"
        if snapshot.active_hero_present and self.last_hand_id:
            return f"{slot_prefix}hand:{self.last_hand_id}"
        if snapshot.active_hero_present:
            return f"{slot_prefix}active_hero_session"
        return f"{slot_prefix}started:{float(snapshot.decision_started_at):.6f}"

    def _should_reset_cycle(self, snapshot: AutoClickSnapshot, cycle_key: str) -> bool:
        if self.last_cycle_key is None:
            self.last_cycle_key = cycle_key
            self.last_hand_id = snapshot.hand_id
            return False
        if not snapshot.active_hero_present:
            return True
        if cycle_key != self.last_cycle_key:
            return True
        if snapshot.hand_id and self.last_hand_id and snapshot.hand_id != self.last_hand_id:
            return True
        return False

    def _reset_cycle(self, snapshot: AutoClickSnapshot, cycle_key: str, now: float) -> List[AutoClickEvent]:
        stable_hand_id = snapshot.hand_id or self.last_hand_id
        self.wait_started_at = float(snapshot.decision_started_at or now)
        self.error_started_at = None
        self.post_click_until = 0.0
        self.last_cycle_key = cycle_key
        self.last_hand_id = stable_hand_id
        self.last_plan_name = None
        self.last_raw_action = None
        self.last_normalized_action = None
        self.last_executed_decision_id = None
        self.last_executed_solver_fingerprint = None
        self.last_executed_source_frame_id = None
        self.last_executed_hand_id = None
        self.last_executed_street = None
        self.last_executed_plan_name = None
        self.last_executed_at = None
        self.last_executed_decision_guard = None
        self.locked_decision_id = None
        self.locked_decision_guard = None
        self.state = STATE_ACTIVE_HERO_WAITING_DECISION if snapshot.active_hero_present else STATE_IDLE
        return [self._event("cycle_reset", hand_id=stable_hand_id or "", state=self.state, cycle_key=cycle_key)]

    def _schedule_next_idle(self, now: float) -> float:
        return now + self.rng.uniform(float(self.config.idle_pause_sec_min), float(self.config.idle_pause_sec_max))

    def _maybe_idle_move(self, now: float, events: List[AutoClickEvent]) -> None:
        if now < self.next_idle_at:
            return
        self.next_idle_at = self._schedule_next_idle(now)
        start_x, start_y = self.mouse.get_position()
        bounds = self._get_virtual_screen_bounds()
        distance = self.rng.randint(int(self.config.idle_move_distance_px_min), int(self.config.idle_move_distance_px_max))
        angle = self.rng.uniform(0.0, math.tau)
        dx = math.cos(angle) * distance
        dy = math.sin(angle) * distance * self.rng.uniform(0.72, 1.22)
        target_x = self._clamp_to_bounds(int(round(start_x + dx)), bounds[0], bounds[2])
        target_y = self._clamp_to_bounds(int(round(start_y + dy)), bounds[1], bounds[3])
        if target_x == start_x and target_y == start_y:
            events.append(self._event("idle_move_skipped_bounds", start_x=start_x, start_y=start_y))
            return
        duration_ms = self.rng.randint(int(self.config.idle_move_duration_ms_min), int(self.config.idle_move_duration_ms_max))
        self._move_mouse_human(target_x, target_y, duration_ms)
        events.append(
            self._event(
                "idle_move_performed",
                start_x=start_x,
                start_y=start_y,
                target_x=target_x,
                target_y=target_y,
                distance_px=int(round(math.hypot(target_x - start_x, target_y - start_y))),
                duration_ms=duration_ms,
            )
        )

    def _normalize_slot_bbox(self, slot_bbox: Optional[Sequence[float]]) -> Optional[Tuple[float, float, float, float]]:
        if slot_bbox is None:
            return None
        try:
            values = tuple(float(v) for v in slot_bbox)
        except Exception:
            return None
        if len(values) != 4:
            return None
        x1, y1, x2, y2 = values
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        if right <= left or bottom <= top:
            return None
        return (left, top, right, bottom)

    def _snapshot_uses_slot_local_coordinates(self, snapshot: AutoClickSnapshot) -> bool:
        if snapshot.slot_bbox is None:
            return False
        source = str(snapshot.frame_source or "").strip().lower()
        return source in {"slot_crop", "slot_frame", "launcher_slot_crop", "mss_slot_crop"}

    def _translate_local_bbox_to_global(
        self,
        bbox: Tuple[float, float, float, float],
        snapshot: AutoClickSnapshot,
    ) -> Tuple[float, float, float, float]:
        if not self._snapshot_uses_slot_local_coordinates(snapshot):
            return bbox
        offset_x, offset_y = snapshot.slot_global_offset
        return (
            float(bbox[0]) + float(offset_x),
            float(bbox[1]) + float(offset_y),
            float(bbox[2]) + float(offset_x),
            float(bbox[3]) + float(offset_y),
        )

    def _point_inside_slot(self, x: int, y: int, snapshot: AutoClickSnapshot) -> bool:
        slot_bbox = self._normalize_slot_bbox(snapshot.slot_bbox)
        if slot_bbox is None:
            return True
        x1, y1, x2, y2 = slot_bbox
        return float(x1) <= float(x) <= float(x2) and float(y1) <= float(y) <= float(y2)

    def _bbox_inside_slot(self, bbox: Tuple[float, float, float, float], snapshot: AutoClickSnapshot) -> bool:
        slot_bbox = self._normalize_slot_bbox(snapshot.slot_bbox)
        if slot_bbox is None:
            return True
        x1, y1, x2, y2 = bbox
        sx1, sy1, sx2, sy2 = slot_bbox
        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0
        return float(sx1) <= cx <= float(sx2) and float(sy1) <= cy <= float(sy2)

    def _get_detection_frame(self, snapshot: AutoClickSnapshot, frame_bgr: Any) -> Tuple[Any, Dict[str, object]]:
        if frame_bgr is not None:
            return frame_bgr, {
                "source": str(snapshot.frame_source or "launcher_frame"),
                "slot_id": snapshot.slot_id,
                "slot_bbox": list(snapshot.slot_bbox) if snapshot.slot_bbox is not None else None,
                "slot_global_offset": list(snapshot.slot_global_offset),
                "coordinates": "slot_local" if self._snapshot_uses_slot_local_coordinates(snapshot) else "global",
            }
        if mss is None or np is None:
            return None, {"source": "none"}
        slot_bbox = self._normalize_slot_bbox(snapshot.slot_bbox)
        if slot_bbox is not None:
            left, top, right, bottom = slot_bbox
            monitor = {
                "left": int(round(left)),
                "top": int(round(top)),
                "width": max(1, int(round(right - left))),
                "height": max(1, int(round(bottom - top))),
            }
            source_name = "mss_slot_crop"
        else:
            width = int(snapshot.monitor_width or 0)
            height = int(snapshot.monitor_height or 0)
            if width <= 0 or height <= 0:
                return None, {"source": "none"}
            monitor = {
                "left": int(snapshot.monitor_left),
                "top": int(snapshot.monitor_top),
                "width": width,
                "height": height,
            }
            source_name = "mss"
        try:
            with mss.mss() as sct:
                shot = sct.grab(monitor)
            frame = np.array(shot)
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            return frame, {
                "source": source_name,
                **monitor,
                "slot_id": snapshot.slot_id,
                "slot_bbox": list(slot_bbox) if slot_bbox is not None else None,
                "coordinates": "slot_local" if slot_bbox is not None else "global",
            }
        except Exception as exc:
            return None, {"source": "capture_error", "error": str(exc)}

    def _event(self, name: str, **payload: object) -> AutoClickEvent:
        return AutoClickEvent(name=name, ts=time.time(), payload={str(k): _json_safe(v) for k, v in payload.items()})

    def _append_events(self, events: Sequence[AutoClickEvent]) -> None:
        if not events:
            return
        self.recent_events.extend(events)
        if len(self.recent_events) > 200:
            self.recent_events = self.recent_events[-200:]

    def _finalize(self, result: AutoClickResult) -> AutoClickResult:
        self._append_events(result.events)
        self._sync_active_slot_state()
        return result

    def _result_from_state(self, plan: AutoClickPlan, executed: bool, events: List[AutoClickEvent], *, locked: bool) -> AutoClickResult:
        if executed:
            self.state = STATE_LOCKED_UNTIL_RESET
        return AutoClickResult(
            state=self.state,
            executed=executed,
            locked=locked,
            plan_name=plan.plan_name,
            normalized_action=plan.normalized_action,
            raw_action=plan.raw_action,
            events=events,
        )

    def _prepare_detections(self, items: Sequence[Any], snapshot: AutoClickSnapshot) -> List[ButtonDetection]:
        now_ts = time.time()
        detections: List[ButtonDetection] = []
        uses_slot_local = self._snapshot_uses_slot_local_coordinates(snapshot)
        for item in items:
            if isinstance(item, ButtonDetection):
                raw_bbox = tuple(float(v) for v in item.bbox)
                if len(raw_bbox) != 4:
                    continue
                source_class_name = item.source_class_name
                confidence = float(item.confidence)
                frame_ts = float(item.frame_ts)
                class_name = item.class_name
            else:
                raw_name = getattr(item, "class_name", None) or getattr(item, "name", None) or getattr(item, "label", None)
                canonical = canonicalize_button_class_name(raw_name)
                if canonical is None:
                    continue
                raw_bbox = tuple(float(v) for v in getattr(item, "bbox", (0.0, 0.0, 0.0, 0.0)))
                if len(raw_bbox) != 4:
                    continue
                source_class_name = str(raw_name) if raw_name is not None else None
                confidence = float(getattr(item, "confidence", 0.0))
                frame_ts = float(getattr(item, "frame_ts", now_ts))
                class_name = canonical

            # Button detections produced on a slot crop are local to that crop.
            # Convert them once here, before any mouse target is generated.
            local_bbox = raw_bbox if uses_slot_local else None
            global_bbox = self._translate_local_bbox_to_global(raw_bbox, snapshot)
            bbox = global_bbox if uses_slot_local else raw_bbox
            x1, y1, x2, y2 = bbox

            # action_panel_bbox belongs to the same frame space as the input detector.
            # In slot-crop mode it is local; in full-frame mode it is global.
            if snapshot.action_panel_bbox is not None:
                panel_bbox = tuple(float(v) for v in snapshot.action_panel_bbox)
                panel_check_bbox = raw_bbox if uses_slot_local else bbox
                if not self._bbox_inside_panel(panel_check_bbox, panel_bbox):
                    continue

            det = ButtonDetection(
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
                center_x=(x1 + x2) / 2.0,
                center_y=(y1 + y2) / 2.0,
                frame_ts=frame_ts,
                source_class_name=source_class_name,
                local_bbox=local_bbox,
                global_bbox=bbox,
                coordinate_space="global_from_slot_local" if uses_slot_local else "global",
                slot_id=snapshot.slot_id,
            )
            if (now_ts - float(det.frame_ts)) * 1000.0 > float(self.config.button_detection_ttl_ms):
                continue
            if snapshot.slot_bbox is not None and not self._bbox_inside_slot(det.bbox, snapshot):
                continue
            conf_threshold = self._threshold_for_button(det.class_name)
            if det.confidence < conf_threshold:
                continue
            detections.append(det)
        deduped = self._dedupe_by_class(detections)
        return sorted(deduped, key=lambda d: (d.center_y, d.center_x))

    def _threshold_for_button(self, class_name: str) -> float:
        if class_name == BUTTON_RAISE:
            return float(self.config.detector_conf_raise)
        if class_name in {BUTTON_CHECK, BUTTON_CHECK_FOLD}:
            return float(self.config.detector_conf_check)
        return float(self.config.detector_conf_default)

    def _bbox_inside_panel(self, bbox: Tuple[float, float, float, float], panel: Tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = bbox
        px1, py1, px2, py2 = panel
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return px1 <= cx <= px2 and py1 <= cy <= py2

    def _dedupe_by_class(self, detections: Sequence[ButtonDetection]) -> List[ButtonDetection]:
        best: Dict[str, ButtonDetection] = {}
        for det in detections:
            existing = best.get(det.class_name)
            if existing is None or det.confidence > existing.confidence:
                best[det.class_name] = det
        return list(best.values())

    def _resolve_raw_action(self, snapshot: AutoClickSnapshot, decision: HeroDecision) -> str:
        street = str(snapshot.street or getattr(decision, "street", "preflop") or "preflop").strip().lower()
        primary_source = getattr(decision, "preflop", None) if street == "preflop" else getattr(decision, "postflop", None)
        fallback_source = getattr(decision, "postflop", None) if street == "preflop" else getattr(decision, "preflop", None)

        if primary_source is not None:
            primary_action = getattr(primary_source, "action", None)
            if primary_action not in (None, ""):
                return str(primary_action).strip().lower()

        top_level_reason = getattr(decision, "reason", None)
        if top_level_reason not in (None, ""):
            return str(top_level_reason).strip().lower()

        engine_action = getattr(decision, "engine_action", None)
        if engine_action not in (None, ""):
            return str(engine_action).strip().lower()

        if fallback_source is not None:
            fallback_action = getattr(fallback_source, "action", None)
            if fallback_action not in (None, ""):
                return str(fallback_action).strip().lower()

        return ""

    def _extract_size_pct(self, decision: HeroDecision) -> Optional[float]:
        for source in (decision, getattr(decision, "preflop", None), getattr(decision, "postflop", None)):
            if source is None:
                continue
            value = getattr(source, "size_pct", None)
            if value is None:
                continue
            try:
                return float(value)
            except Exception:
                continue
        debug = getattr(decision, "debug", {}) or {}
        if isinstance(debug, dict):
            recommended = debug.get("recommended_option")
            if isinstance(recommended, dict):
                try:
                    value = recommended.get("size_pct")
                    if value is not None:
                        return float(value)
                except Exception:
                    pass
        return None

    def _nearest_supported_size(self, size_pct: Optional[float]) -> Optional[int]:
        if size_pct is None:
            return None
        try:
            value = float(size_pct)
        except Exception:
            return None
        supported = [33, 50, 70, 98]
        nearest = min(supported, key=lambda item: abs(item - value))
        return nearest if abs(nearest - value) <= 4.0 else None

    def _normalize_action(self, snapshot: AutoClickSnapshot) -> str:
        decision = snapshot.hero_decision
        if decision is None:
            return ""
        raw_action = self._resolve_raw_action(snapshot, decision)
        engine_action = str(getattr(decision, "engine_action", "") or "").strip().lower()
        street = str(snapshot.street or getattr(decision, "street", "preflop") or "preflop").lower()
        rounded_size = self._nearest_supported_size(self._extract_size_pct(decision))

        if raw_action in {"fold", "check", "call", "check_fold", "check/fold"}:
            return "check_fold" if raw_action in {"check_fold", "check/fold"} else raw_action
        if raw_action in RAW_ISO_ACTIONS:
            return "iso_raise_98"
        if raw_action in RAW_THREEBET_ACTIONS:
            return "threebet_98"
        if raw_action in RAW_FOURBET_ACTIONS:
            return "fourbet_98"
        if raw_action in RAW_FIVEBET_ACTIONS:
            return "fivebet_98"
        if rounded_size in SIZE_TO_BUTTON and (engine_action in {"raise", "bet", "all_in"} or raw_action.startswith("bet_") or raw_action.startswith("raise_")):
            return {33: "raise_33", 50: "raise_50", 70: "raise_70", 98: "raise_98"}[rounded_size]
        if street == "preflop" and engine_action == "raise" and rounded_size is None:
            return "raise_only"
        if engine_action in {"raise", "bet", "all_in"} or raw_action in {"raise", "bet", "jam", "all_in", "all-in"}:
            return "raise_only"
        return engine_action or raw_action

    def _build_click_plan(self, normalized_action: str, raw_action: str) -> AutoClickPlan:
        if normalized_action == "fold":
            return AutoClickPlan(ACTION_CLICK_FOLD, normalized_action, raw_action, BUTTON_FOLD)
        if normalized_action == "check":
            return AutoClickPlan(ACTION_CLICK_CHECK, normalized_action, raw_action, BUTTON_CHECK)
        if normalized_action == "call":
            return AutoClickPlan(ACTION_CLICK_CALL, normalized_action, raw_action, BUTTON_CALL)
        if normalized_action == "check_fold":
            return AutoClickPlan(ACTION_CLICK_CHECK_FOLD, normalized_action, raw_action, BUTTON_CHECK_FOLD)
        if normalized_action == "raise_only":
            return AutoClickPlan(ACTION_CLICK_RAISE_ONLY, normalized_action, raw_action, BUTTON_RAISE)
        if normalized_action == "raise_33":
            return AutoClickPlan(ACTION_CLICK_SIZE_THEN_RAISE_33, normalized_action, raw_action, BUTTON_33, BUTTON_RAISE, allow_scroll_between=True)
        if normalized_action == "raise_50":
            return AutoClickPlan(ACTION_CLICK_SIZE_THEN_RAISE_50, normalized_action, raw_action, BUTTON_50, BUTTON_RAISE, allow_scroll_between=True)
        if normalized_action == "raise_70":
            return AutoClickPlan(ACTION_CLICK_SIZE_THEN_RAISE_70, normalized_action, raw_action, BUTTON_70, BUTTON_RAISE, allow_scroll_between=True)
        if normalized_action in {"raise_98", "iso_raise_98", "threebet_98", "fourbet_98", "fivebet_98"}:
            return AutoClickPlan(ACTION_CLICK_98_THEN_RAISE, normalized_action, raw_action, BUTTON_98, BUTTON_RAISE, allow_scroll_between=True)
        return AutoClickPlan(ACTION_NO_AUTOCLICK, normalized_action, raw_action, None)

    def _extract_runtime_meta(self, snapshot: AutoClickSnapshot) -> Dict[str, object]:
        meta = dict(snapshot.solver_context_meta or {})
        if snapshot.slot_id:
            meta.setdefault("slot_id", snapshot.slot_id)
        if snapshot.slot_bbox is not None:
            meta.setdefault("slot_bbox", [float(v) for v in snapshot.slot_bbox])
            meta.setdefault("slot_global_offset", [float(snapshot.slot_global_offset[0]), float(snapshot.slot_global_offset[1])])
        decision = snapshot.hero_decision
        debug = getattr(decision, "debug", {}) or {}
        if isinstance(debug, dict):
            nested_meta = debug.get("meta") or {}
            if isinstance(nested_meta, dict):
                for key, value in nested_meta.items():
                    meta.setdefault(str(key), value)
            for key in IDENTITY_KEYS:
                value = debug.get(key)
                if value not in (None, "") and key not in meta:
                    meta[key] = value
        return meta

    def _validate_reraise_spot_action(self, snapshot: AutoClickSnapshot, normalized_action: str, raw_action: str) -> Tuple[bool, Dict[str, object]]:
        meta = self._extract_runtime_meta(snapshot)
        node_type = str(meta.get("node_type") or "").strip().lower()
        allowed_actions = RERAISE_SPOT_ALLOWED_ACTIONS.get(node_type)
        expected_action = sorted(allowed_actions)[0] if allowed_actions and len(allowed_actions) == 1 else ""
        info: Dict[str, object] = {
            "node_type": node_type,
            "normalized_action": normalized_action,
            "raw_action": raw_action,
            "allowed_actions": sorted(allowed_actions) if allowed_actions else [],
            "expected_action": expected_action,
            "opener_pos": meta.get("opener_pos"),
            "three_bettor_pos": meta.get("three_bettor_pos"),
            "four_bettor_pos": meta.get("four_bettor_pos"),
            "limpers": meta.get("limpers"),
            "callers": meta.get("callers"),
        }
        if not allowed_actions:
            info["guard_applied"] = False
            return (True, info)
        info["guard_applied"] = True
        if normalized_action not in AGGRESSIVE_NORMALIZED_ACTIONS:
            info["blocked_reason"] = "non_aggressive_action_allowed"
            return (True, info)
        if normalized_action in allowed_actions:
            info["blocked_reason"] = ""
            return (True, info)
        info["blocked_reason"] = "aggressive_action_not_allowed_for_reraise_spot"
        return (False, info)

    def _build_failsafe_plan(self, detections: Sequence[ButtonDetection], reason: str) -> AutoClickPlan:
        button_names = {d.class_name for d in detections}
        if BUTTON_CHECK in button_names:
            return AutoClickPlan(ACTION_CLICK_CHECK, "check", f"failsafe:{reason}", BUTTON_CHECK, is_failsafe=True)
        if BUTTON_CHECK_FOLD in button_names:
            return AutoClickPlan(ACTION_CLICK_CHECK_FOLD, "check_fold", f"failsafe:{reason}", BUTTON_CHECK_FOLD, is_failsafe=True)
        if BUTTON_FOLD in button_names:
            return AutoClickPlan(ACTION_CLICK_FOLD, "fold", f"failsafe:{reason}", BUTTON_FOLD, is_failsafe=True)
        return AutoClickPlan(ACTION_NO_AUTOCLICK, "", f"failsafe:{reason}", None, is_failsafe=True)

    def _identity_payload_from_decision(self, decision: Optional[HeroDecision]) -> Dict[str, object]:
        payload: Dict[str, object] = {}
        if decision is None:
            return payload
        debug = getattr(decision, "debug", {}) or {}
        if isinstance(debug, dict):
            for key in IDENTITY_KEYS:
                if key in debug:
                    payload[key] = debug[key]
            for key in ("engine_action", "recommended_action", "street"):
                if key in debug:
                    payload[key] = debug[key]
            nested_meta = debug.get("meta") or {}
            if isinstance(nested_meta, dict):
                for key in IDENTITY_KEYS:
                    if key in nested_meta and key not in payload:
                        payload[key] = nested_meta[key]
        for key in ("solver_fingerprint", "decision_id", "source_frame_id", "engine_action", "street"):
            value = getattr(decision, key, None)
            if value is not None:
                payload[key] = value
        return payload

    def _validate_snapshot_identity(self, snapshot: AutoClickSnapshot) -> Tuple[bool, Dict[str, object]]:
        decision_payload = self._identity_payload_from_decision(snapshot.hero_decision)
        meta = dict(snapshot.solver_context_meta or {})
        mismatches: List[str] = []
        compared: List[str] = []
        skipped_missing_on_meta: List[str] = []
        skipped_missing_on_decision: List[str] = []
        for key in IDENTITY_KEYS:
            meta_has = key in meta and meta.get(key) not in (None, "")
            decision_has = key in decision_payload and decision_payload.get(key) not in (None, "")
            if meta_has and decision_has:
                compared.append(key)
                if str(meta.get(key)) != str(decision_payload.get(key)):
                    mismatches.append(key)
                continue
            if meta_has and not decision_has:
                skipped_missing_on_decision.append(key)
            elif decision_has and not meta_has:
                skipped_missing_on_meta.append(key)
        info = {
            "compared": compared,
            "mismatches": mismatches,
            "meta": meta,
            "decision": decision_payload,
            "skipped_missing_on_meta": skipped_missing_on_meta,
            "skipped_missing_on_decision": skipped_missing_on_decision,
            "identity_mode": "soft_overlap_only",
        }
        if mismatches:
            return (False, info)
        return (True, info)

    def _build_decision_memory(self, snapshot: AutoClickSnapshot, plan: AutoClickPlan) -> Dict[str, object]:
        decision = snapshot.hero_decision
        debug = getattr(decision, "debug", {}) or {}
        meta = dict(snapshot.solver_context_meta or {})
        execution_token = self._execution_token(snapshot, plan)
        decision_id = _first_non_empty(
            getattr(decision, "decision_id", None),
            debug.get("decision_id") if isinstance(debug, dict) else None,
            meta.get("decision_id"),
        )
        solver_fingerprint = _first_non_empty(
            getattr(decision, "solver_fingerprint", None),
            debug.get("solver_fingerprint") if isinstance(debug, dict) else None,
            meta.get("solver_fingerprint"),
        )
        source_frame_id = _first_non_empty(
            getattr(decision, "source_frame_id", None),
            debug.get("source_frame_id") if isinstance(debug, dict) else None,
            meta.get("source_frame_id"),
        )
        hand_id = _first_non_empty(snapshot.hand_id, meta.get("hand_id"))
        street = _first_non_empty(snapshot.street, getattr(decision, "street", None), meta.get("street"))
        guard_key = decision_id if decision_id else f"execution_token:{execution_token}"
        guard_source = "decision_id" if decision_id else "execution_token_fallback"
        return {
            "decision_id": decision_id,
            "solver_fingerprint": solver_fingerprint,
            "source_frame_id": source_frame_id,
            "hand_id": hand_id,
            "street": street,
            "plan_name": plan.plan_name,
            "normalized_action": plan.normalized_action,
            "raw_action": plan.raw_action,
            "execution_token": execution_token,
            "guard_key": guard_key,
            "guard_source": guard_source,
        }

    def _record_successful_execution(self, decision_memory: Dict[str, object], events: List[AutoClickEvent]) -> None:
        self.locked_decision_id = _first_non_empty(decision_memory.get("decision_id"))
        self.locked_decision_guard = _first_non_empty(decision_memory.get("guard_key"))
        self.last_executed_decision_id = _first_non_empty(decision_memory.get("decision_id"))
        self.last_executed_solver_fingerprint = _first_non_empty(decision_memory.get("solver_fingerprint"))
        self.last_executed_source_frame_id = _first_non_empty(decision_memory.get("source_frame_id"))
        self.last_executed_hand_id = _first_non_empty(decision_memory.get("hand_id"))
        self.last_executed_street = _first_non_empty(decision_memory.get("street"))
        self.last_executed_plan_name = _first_non_empty(decision_memory.get("plan_name"))
        self.last_executed_at = time.time()
        self.last_executed_decision_guard = _first_non_empty(decision_memory.get("guard_key"))
        events.append(
            self._event(
                "decision_id_recorded_after_success",
                decision_id=self.last_executed_decision_id or "",
                solver_fingerprint=self.last_executed_solver_fingerprint or "",
                source_frame_id=self.last_executed_source_frame_id or "",
                hand_id=self.last_executed_hand_id or "",
                street=self.last_executed_street or "",
                plan_name=self.last_executed_plan_name or "",
                guard_source=decision_memory.get("guard_source") or "",
                execution_token=decision_memory.get("execution_token") or "",
                recorded_at=self.last_executed_at,
            )
        )

    def _execution_token(self, snapshot: AutoClickSnapshot, plan: AutoClickPlan) -> str:
        decision = snapshot.hero_decision
        debug = getattr(decision, "debug", {}) or {}
        meta = dict(snapshot.solver_context_meta or {})
        token_payload = {
            "hand_id": snapshot.hand_id,
            "street": snapshot.street,
            "solver_fingerprint": _first_non_empty(
                getattr(decision, "solver_fingerprint", None),
                debug.get("solver_fingerprint") if isinstance(debug, dict) else None,
                meta.get("solver_fingerprint"),
            ),
            "decision_id": _first_non_empty(
                getattr(decision, "decision_id", None),
                debug.get("decision_id") if isinstance(debug, dict) else None,
                meta.get("decision_id"),
            ),
            "source_frame_id": _first_non_empty(
                getattr(decision, "source_frame_id", None),
                debug.get("source_frame_id") if isinstance(debug, dict) else None,
                meta.get("source_frame_id"),
            ),
            "node_type": meta.get("node_type") or (debug.get("node_type") if isinstance(debug, dict) else None),
            "opener_pos": meta.get("opener_pos"),
            "three_bettor_pos": meta.get("three_bettor_pos"),
            "four_bettor_pos": meta.get("four_bettor_pos"),
            "limpers": meta.get("limpers"),
            "callers": meta.get("callers"),
            "engine_action": getattr(decision, "engine_action", None),
            "reason": getattr(decision, "reason", None),
            "plan_name": plan.plan_name,
            "normalized_action": plan.normalized_action,
        }
        return _stable_hash("execution_token_v2", token_payload)

    def _find_button(self, detections: Sequence[ButtonDetection], class_name: Optional[str]) -> Optional[ButtonDetection]:
        if not class_name:
            return None
        for det in detections:
            if det.class_name == class_name:
                return det
        return None

    def _execute_plan(
        self,
        plan: AutoClickPlan,
        detections: Sequence[ButtonDetection],
        snapshot: AutoClickSnapshot,
        events: List[AutoClickEvent],
    ) -> bool:
        if plan.primary_button is None:
            events.append(self._event("critical_click_failure", reason="missing_primary_button_in_plan"))
            return False
        primary = self._find_button(detections, plan.primary_button)
        if primary is None:
            events.append(self._event("primary_button_missing", button=plan.primary_button))
            primary = self._retry_find_button(plan.primary_button, snapshot, events)
            if primary is None:
                return False
        self.state = STATE_EXECUTING_PLAN
        if not self._click_detection(primary, events, role="primary", snapshot=snapshot):
            return False
        if plan.allow_scroll_between:
            self._maybe_scroll(events)
        if plan.secondary_button:
            settle_ms = self.rng.randint(int(self.config.secondary_click_settle_ms_min), int(self.config.secondary_click_settle_ms_max))
            if settle_ms > 0:
                time.sleep(settle_ms / 1000.0)
            secondary = self._find_button(detections, plan.secondary_button)
            if secondary is None and self.config.secondary_retry_recapture:
                secondary = self._retry_find_button(plan.secondary_button, snapshot, events)
            if secondary is None:
                events.append(self._event("critical_click_failure", reason="secondary_button_missing", button=plan.secondary_button))
                return False
            if not self._click_detection(secondary, events, role="secondary", snapshot=snapshot):
                return False
        self.post_click_until = time.monotonic() + self.rng.uniform(
            float(self.config.post_click_cooldown_sec_min),
            float(self.config.post_click_cooldown_sec_max),
        )
        return True

    def _retry_find_button(self, class_name: str, snapshot: AutoClickSnapshot, events: List[AutoClickEvent]) -> Optional[ButtonDetection]:
        if self.button_detector is None:
            events.append(self._event("critical_click_failure", reason="button_detector_unavailable", button=class_name))
            return None
        deadline = time.monotonic() + (float(self.config.retry_total_budget_ms) / 1000.0)
        for attempt in range(1, int(self.config.retry_count_max) + 1):
            if time.monotonic() >= deadline:
                break
            sleep_ms = self.rng.randint(int(self.config.retry_sleep_ms_min), int(self.config.retry_sleep_ms_max))
            time.sleep(sleep_ms / 1000.0)
            frame, capture_meta = self._get_detection_frame(snapshot, None)
            raw = self.button_detector.detect_buttons(frame) if frame is not None else []
            prepared = self._prepare_detections(raw, snapshot)
            self._write_debug_observation(snapshot, frame, raw, prepared, capture_meta, events=events)
            found = self._find_button(prepared, class_name)
            events.append(self._event("button_retry", button=class_name, attempt=attempt, found=bool(found)))
            if found is not None:
                return found
        events.append(self._event("critical_click_failure", reason="button_not_found_after_retry", button=class_name))
        return None

    def _maybe_scroll(self, events: List[AutoClickEvent]) -> None:
        if self.rng.random() > float(self.config.scroll_enabled_probability):
            return
        steps = self.rng.randint(int(self.config.scroll_steps_min), int(self.config.scroll_steps_max))
        amount = 120 if self.rng.random() < float(self.config.scroll_direction_up_probability) else -120
        for _ in range(steps):
            self.mouse.wheel(amount)
            pause_ms = self.rng.randint(int(self.config.scroll_pause_ms_min), int(self.config.scroll_pause_ms_max))
            time.sleep(pause_ms / 1000.0)
        events.append(self._event("scroll_applied", steps=steps, amount=amount))

    def _click_detection(
        self,
        detection: ButtonDetection,
        events: List[AutoClickEvent],
        *,
        role: str,
        snapshot: Optional[AutoClickSnapshot] = None,
    ) -> bool:
        target_x, target_y = self._pick_click_target(detection)
        if snapshot is not None and not self._point_inside_slot(target_x, target_y, snapshot):
            events.append(
                self._event(
                    "click_blocked_outside_slot_region",
                    role=role,
                    button=detection.class_name,
                    click_x=target_x,
                    click_y=target_y,
                    bbox=[round(v, 2) for v in detection.bbox],
                    local_bbox=[round(v, 2) for v in detection.local_bbox] if detection.local_bbox is not None else None,
                    global_bbox=[round(v, 2) for v in (detection.global_bbox or detection.bbox)],
                    slot_id=snapshot.slot_id,
                    slot_bbox=list(snapshot.slot_bbox) if snapshot.slot_bbox is not None else None,
                )
            )
            return False

        duration_ms = self.rng.randint(int(self.config.move_duration_ms_min), int(self.config.move_duration_ms_max))
        events.append(
            self._event(
                "mouse_move_started",
                role=role,
                target_x=target_x,
                target_y=target_y,
                duration_ms=duration_ms,
                bbox=[round(v, 2) for v in detection.bbox],
                local_bbox=[round(v, 2) for v in detection.local_bbox] if detection.local_bbox is not None else None,
                global_bbox=[round(v, 2) for v in (detection.global_bbox or detection.bbox)],
                coordinate_space=detection.coordinate_space,
                slot_id=detection.slot_id,
            )
        )
        self._move_mouse_human(target_x, target_y, duration_ms)
        down_up_ms = self.rng.randint(int(self.config.click_down_up_delay_ms_min), int(self.config.click_down_up_delay_ms_max))
        self.mouse.left_down()
        time.sleep(down_up_ms / 1000.0)
        self.mouse.left_up()
        self.last_click_at = time.monotonic()
        events.append(
            self._event(
                "button_clicked",
                role=role,
                button=detection.class_name,
                confidence=round(float(detection.confidence), 4),
                click_x=target_x,
                click_y=target_y,
                bbox=[round(v, 2) for v in detection.bbox],
                local_bbox=[round(v, 2) for v in detection.local_bbox] if detection.local_bbox is not None else None,
                global_bbox=[round(v, 2) for v in (detection.global_bbox or detection.bbox)],
                coordinate_space=detection.coordinate_space,
                slot_id=detection.slot_id,
            )
        )
        return True

    def _pick_click_target(self, detection: ButtonDetection) -> Tuple[int, int]:
        x1, y1, x2, y2 = detection.bbox
        left = int(math.floor(min(x1, x2)))
        top = int(math.floor(min(y1, y2)))
        right = int(math.ceil(max(x1, x2)))
        bottom = int(math.ceil(max(y1, y2)))
        width = max(1, right - left)
        height = max(1, bottom - top)

        ratio_min = max(0.0, float(self.config.click_target_inner_padding_ratio_min))
        ratio_max = max(ratio_min, float(self.config.click_target_inner_padding_ratio_max))
        padding_ratio_x = self.rng.uniform(ratio_min, ratio_max)
        padding_ratio_y = self.rng.uniform(ratio_min, ratio_max)
        padding_x = int(round(width * padding_ratio_x))
        padding_y = int(round(height * padding_ratio_y))

        padding_x = max(int(self.config.click_target_inner_padding_px_min), padding_x)
        padding_y = max(int(self.config.click_target_inner_padding_px_min), padding_y)
        padding_x = min(int(self.config.click_target_inner_padding_px_max), padding_x, max(1, width // 4))
        padding_y = min(int(self.config.click_target_inner_padding_px_max), padding_y, max(1, height // 4))

        safe_left = left + padding_x
        safe_top = top + padding_y
        safe_right = right - padding_x
        safe_bottom = bottom - padding_y
        if safe_left >= safe_right:
            safe_left, safe_right = left, right
        if safe_top >= safe_bottom:
            safe_top, safe_bottom = top, bottom

        target_x = self.rng.randint(int(safe_left), int(max(safe_left, safe_right)))
        target_y = self.rng.randint(int(safe_top), int(max(safe_top, safe_bottom)))
        jitter_x = self.rng.randint(int(self.config.target_jitter_px_min), int(self.config.target_jitter_px_max))
        jitter_y = self.rng.randint(int(self.config.target_jitter_px_min), int(self.config.target_jitter_px_max))
        target_x = self._clamp_to_bounds(target_x + self.rng.randint(-jitter_x, jitter_x), int(safe_left), int(max(safe_left, safe_right)))
        target_y = self._clamp_to_bounds(target_y + self.rng.randint(-jitter_y, jitter_y), int(safe_top), int(max(safe_top, safe_bottom)))
        return int(target_x), int(target_y)

    def _get_virtual_screen_bounds(self) -> Tuple[int, int, int, int]:
        if ctypes is not None:
            try:
                user32 = ctypes.windll.user32  # type: ignore[attr-defined]
                left = int(user32.GetSystemMetrics(76))
                top = int(user32.GetSystemMetrics(77))
                width = int(user32.GetSystemMetrics(78))
                height = int(user32.GetSystemMetrics(79))
                if width > 0 and height > 0:
                    return left, top, left + width - 1, top + height - 1
            except Exception:
                pass
        return 0, 0, 3839, 2159

    def _clamp_to_bounds(self, value: int, min_value: int, max_value: int) -> int:
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        return max(min_value, min(max_value, value))

    def _bezier_point(
        self,
        t: float,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
    ) -> Tuple[float, float]:
        inv = 1.0 - t
        x = (inv ** 3) * p0[0] + 3 * (inv ** 2) * t * p1[0] + 3 * inv * (t ** 2) * p2[0] + (t ** 3) * p3[0]
        y = (inv ** 3) * p0[1] + 3 * (inv ** 2) * t * p1[1] + 3 * inv * (t ** 2) * p2[1] + (t ** 3) * p3[1]
        return x, y

    def _move_mouse_curve(self, start_x: int, start_y: int, target_x: int, target_y: int, duration_ms: int) -> None:
        distance = math.hypot(target_x - start_x, target_y - start_y)
        if distance < 1.0:
            self.mouse.move_to(int(target_x), int(target_y))
            return
        bounds = self._get_virtual_screen_bounds()
        angle = math.atan2(target_y - start_y, target_x - start_x)
        side = self.rng.choice([-1.0, 1.0])
        curve_mag = self.rng.uniform(float(self.config.mouse_curve_offset_px_min), float(self.config.mouse_curve_offset_px_max))
        curve_mag = min(curve_mag, max(16.0, distance * 0.40))
        ctrl1_dist = self.rng.uniform(0.18, 0.34) * distance
        ctrl2_dist = self.rng.uniform(0.62, 0.84) * distance
        perp_angle = angle + side * (math.pi / 2.0)
        p0 = (float(start_x), float(start_y))
        p3 = (float(target_x), float(target_y))
        p1 = (
            start_x + math.cos(angle) * ctrl1_dist + math.cos(perp_angle) * curve_mag * self.rng.uniform(0.55, 1.10),
            start_y + math.sin(angle) * ctrl1_dist + math.sin(perp_angle) * curve_mag * self.rng.uniform(0.55, 1.10),
        )
        p2 = (
            start_x + math.cos(angle) * ctrl2_dist - math.cos(perp_angle) * curve_mag * self.rng.uniform(0.35, 0.95),
            start_y + math.sin(angle) * ctrl2_dist - math.sin(perp_angle) * curve_mag * self.rng.uniform(0.35, 0.95),
        )
        steps = max(12, min(64, int(duration_ms / 11) + int(distance / 34)))
        base_sleep = max(0.0012, duration_ms / 1000.0 / steps)
        for step in range(1, steps + 1):
            t = step / steps
            eased = t * t * (3.0 - 2.0 * t)
            x, y = self._bezier_point(eased, p0, p1, p2, p3)
            jitter_scale = max(0.18, 1.0 - eased)
            x += self.rng.uniform(-1.8, 1.8) * jitter_scale
            y += self.rng.uniform(-1.8, 1.8) * jitter_scale
            px = self._clamp_to_bounds(int(round(x)), bounds[0], bounds[2])
            py = self._clamp_to_bounds(int(round(y)), bounds[1], bounds[3])
            self.mouse.move_to(px, py)
            time.sleep(base_sleep * self.rng.uniform(0.82, 1.22))
        self.mouse.move_to(int(target_x), int(target_y))

    def _move_mouse_human(self, target_x: int, target_y: int, duration_ms: int) -> None:
        start_x, start_y = self.mouse.get_position()
        distance = math.hypot(target_x - start_x, target_y - start_y)
        if distance < 1.0:
            self.mouse.move_to(int(target_x), int(target_y))
            return
        if distance > 70.0 and self.rng.random() < float(self.config.mouse_overshoot_probability):
            angle = math.atan2(target_y - start_y, target_x - start_x)
            overshoot_px = self.rng.randint(int(self.config.mouse_overshoot_px_min), int(self.config.mouse_overshoot_px_max))
            overshoot_x = int(round(target_x + math.cos(angle) * overshoot_px))
            overshoot_y = int(round(target_y + math.sin(angle) * overshoot_px))
            bounds = self._get_virtual_screen_bounds()
            overshoot_x = self._clamp_to_bounds(overshoot_x, bounds[0], bounds[2])
            overshoot_y = self._clamp_to_bounds(overshoot_y, bounds[1], bounds[3])
            first_leg = max(80, int(duration_ms * self.rng.uniform(0.58, 0.76)))
            second_leg = max(45, duration_ms - first_leg)
            self._move_mouse_curve(start_x, start_y, overshoot_x, overshoot_y, first_leg)
            self._move_mouse_curve(overshoot_x, overshoot_y, int(target_x), int(target_y), second_leg)
            return
        self._move_mouse_curve(start_x, start_y, int(target_x), int(target_y), duration_ms)

    def _events_to_dict(self, events: Sequence[AutoClickEvent] | None) -> List[Dict[str, object]]:
        serialized: List[Dict[str, object]] = []
        for event in events or ():
            serialized.append(
                {
                    "ts": float(event.ts),
                    "name": event.name,
                    "payload": {str(k): _json_safe(v) for k, v in event.payload.items()},
                }
            )
        return serialized

    def _extract_guard_events(self, events: Sequence[AutoClickEvent] | None) -> List[Dict[str, object]]:
        relevant = []
        for event in events or ():
            if event.name in {"reraise_spot_guard_allowed", "reraise_spot_guard_blocked"}:
                relevant.append(
                    {
                        "ts": float(event.ts),
                        "name": event.name,
                        "payload": {str(k): _json_safe(v) for k, v in event.payload.items()},
                    }
                )
        return relevant

    def _write_debug_runtime_events(
        self,
        snapshot: AutoClickSnapshot,
        events: Sequence[AutoClickEvent] | None,
        *,
        channel: str,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        serialized_events = self._events_to_dict(events)
        if not serialized_events:
            return
        base_extra = {str(k): _json_safe(v) for k, v in (extra or {}).items()}
        for item in serialized_events:
            payload = {
                "ts": item["ts"],
                "event_name": item["name"],
                "event_payload": item["payload"],
                "slot_id": snapshot.slot_id,
                "slot_bbox": list(snapshot.slot_bbox) if snapshot.slot_bbox is not None else None,
                "hand_id": snapshot.hand_id,
                "street": snapshot.street,
                "channel": channel,
                "solver_context_meta": snapshot.solver_context_meta,
                **base_extra,
            }
            self._append_jsonl(self.debug_root / "runtime_events.jsonl", payload)

    def _write_debug_observation(
        self,
        snapshot: AutoClickSnapshot,
        frame_bgr: Any,
        raw_detections: Sequence[ButtonDetection],
        filtered_detections: Sequence[ButtonDetection],
        capture_meta: Dict[str, object],
        *,
        events: Sequence[AutoClickEvent] | None = None,
    ) -> None:
        serialized_events = self._events_to_dict(events)
        payload = {
            "ts": time.time(),
            "slot_id": snapshot.slot_id,
            "slot_bbox": list(snapshot.slot_bbox) if snapshot.slot_bbox is not None else None,
            "hand_id": snapshot.hand_id,
            "street": snapshot.street,
            "capture_meta": capture_meta,
            "solver_context_meta": snapshot.solver_context_meta,
            "raw_count": len(raw_detections),
            "filtered_count": len(filtered_detections),
            "raw_detections": [self._detection_to_dict(item) for item in raw_detections],
            "filtered_detections": [self._detection_to_dict(item) for item in filtered_detections],
            "event_names": [item["name"] for item in serialized_events],
            "events": serialized_events,
            "guard_events": self._extract_guard_events(events),
        }
        self._append_jsonl(self.debug_root / "observations.jsonl", payload)
        self._write_debug_runtime_events(snapshot, events, channel="observation")

    def _write_debug_plan(
        self,
        snapshot: AutoClickSnapshot,
        plan: AutoClickPlan,
        detections: Sequence[ButtonDetection],
        executed: bool,
        *,
        events: Sequence[AutoClickEvent] | None = None,
    ) -> None:
        serialized_events = self._events_to_dict(events)
        payload = {
            "ts": time.time(),
            "slot_id": snapshot.slot_id,
            "slot_bbox": list(snapshot.slot_bbox) if snapshot.slot_bbox is not None else None,
            "hand_id": snapshot.hand_id,
            "street": snapshot.street,
            "plan_name": plan.plan_name,
            "normalized_action": plan.normalized_action,
            "raw_action": plan.raw_action,
            "primary_button": plan.primary_button,
            "secondary_button": plan.secondary_button,
            "executed": bool(executed),
            "solver_context_meta": snapshot.solver_context_meta,
            "detections": [self._detection_to_dict(item) for item in detections],
            "event_names": [item["name"] for item in serialized_events],
            "events": serialized_events,
            "guard_events": self._extract_guard_events(events),
        }
        self._append_jsonl(self.debug_root / "plans.jsonl", payload)
        self._write_debug_runtime_events(
            snapshot,
            events,
            channel="plan",
            extra={
                "plan_name": plan.plan_name,
                "normalized_action": plan.normalized_action,
                "executed": bool(executed),
            },
        )

    def _append_jsonl(self, path: Path, payload: Dict[str, object]) -> None:
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(_json_safe(payload), ensure_ascii=False) + "\n")
        except Exception:
            LOGGER.exception("%s failed to write debug jsonl %s", self.config.log_prefix, path)

    def _detection_to_dict(self, detection: ButtonDetection) -> Dict[str, object]:
        return {
            "class_name": detection.class_name,
            "confidence": float(detection.confidence),
            "bbox": list(detection.bbox),
            "center_x": float(detection.center_x),
            "center_y": float(detection.center_y),
            "frame_ts": float(detection.frame_ts),
            "source_class_name": detection.source_class_name,
            "local_bbox": list(detection.local_bbox) if detection.local_bbox is not None else None,
            "global_bbox": list(detection.global_bbox) if detection.global_bbox is not None else list(detection.bbox),
            "coordinate_space": detection.coordinate_space,
            "slot_id": detection.slot_id,
        }


__all__ = [
    "BUTTON_FOLD",
    "BUTTON_33",
    "BUTTON_50",
    "BUTTON_70",
    "BUTTON_98",
    "BUTTON_CALL",
    "BUTTON_RAISE",
    "BUTTON_CHECK_FOLD",
    "BUTTON_CHECK",
    "ButtonDetection",
    "AutoClickSnapshot",
    "AutoClickEvent",
    "AutoClickResult",
    "AutoClickPlan",
    "AutoClickConfig",
    "ButtonDetectorProtocol",
    "MouseBackendProtocol",
    "NoopMouseBackend",
    "WindowsMouseBackend",
    "YoloButtonDetector",
    "build_default_button_detector",
    "AutoClickRuntime",
    "canonicalize_button_class_name",
    "resolve_model_path",
]
