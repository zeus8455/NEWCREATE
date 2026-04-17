from __future__ import annotations

import json
import logging
import math
import os
import random
import time
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
    import mss  # type: ignore
except Exception:  # pragma: no cover
    mss = None

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

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


@dataclass(slots=True)
class ButtonDetection:
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    center_x: float
    center_y: float
    frame_ts: float
    source_class_name: Optional[str] = None

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
    primary_button: str
    secondary_button: Optional[str] = None
    allow_scroll_between: bool = False
    is_failsafe: bool = False


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
    timeout_default_sec: float = 10.0
    timeout_big_pot_extra_sec: float = 9.0
    timeout_big_pot_threshold_bb: float = 20.0
    retry_count_max: int = 3
    retry_total_budget_ms: int = 900
    retry_sleep_ms_min: int = 40
    retry_sleep_ms_max: int = 120
    scroll_enabled_probability: float = 0.0
    scroll_direction_up_probability: float = 0.50
    scroll_steps_min: int = 1
    scroll_steps_max: int = 2
    scroll_pause_ms_min: int = 25
    scroll_pause_ms_max: int = 60
    move_duration_ms_min: int = 140
    move_duration_ms_max: int = 360
    target_jitter_px_min: int = 3
    target_jitter_px_max: int = 10
    click_down_up_delay_ms_min: int = 28
    click_down_up_delay_ms_max: int = 65
    post_click_cooldown_sec_min: float = 1.0
    post_click_cooldown_sec_max: float = 1.8
    idle_pause_sec_min: float = 4.0
    idle_pause_sec_max: float = 14.0
    idle_move_distance_px_min: int = 20
    idle_move_distance_px_max: int = 140
    secondary_click_settle_ms_min: int = 15
    secondary_click_settle_ms_max: int = 45
    secondary_retry_recapture: bool = True
    log_prefix: str = "[AutoClick]"


class ButtonDetectorProtocol(Protocol):
    def detect_buttons(self, frame_bgr: Any) -> List[ButtonDetection]:
        ...


class MouseBackendProtocol(Protocol):
    def get_position(self) -> Tuple[int, int]:
        ...

    def move_to(self, x: int, y: int) -> None:
        ...

    def left_down(self) -> None:
        ...

    def left_up(self) -> None:
        ...

    def wheel(self, amount: int) -> None:
        ...


class WindowsMouseBackend:
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_WHEEL = 0x0800

    def __init__(self) -> None:
        if ctypes is None:
            raise RuntimeError("ctypes unavailable")
        self._user32 = ctypes.windll.user32

    def get_position(self) -> Tuple[int, int]:
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
        point = POINT()
        self._user32.GetCursorPos(ctypes.byref(point))
        return int(point.x), int(point.y)

    def move_to(self, x: int, y: int) -> None:
        self._user32.SetCursorPos(int(x), int(y))

    def left_down(self) -> None:
        self._user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

    def left_up(self) -> None:
        self._user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def wheel(self, amount: int) -> None:
        self._user32.mouse_event(self.MOUSEEVENTF_WHEEL, 0, 0, int(amount), 0)


class NoopMouseBackend:
    def __init__(self) -> None:
        self._pos = (0, 0)

    def get_position(self) -> Tuple[int, int]:
        return self._pos

    def move_to(self, x: int, y: int) -> None:
        self._pos = (int(x), int(y))

    def left_down(self) -> None:
        return None

    def left_up(self) -> None:
        return None

    def wheel(self, amount: int) -> None:
        return None


class YoloButtonDetector:
    def __init__(self, model_path: str, *, conf: float = 0.25) -> None:
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed")
        self.model_path = resolve_model_path(model_path)
        self.model = YOLO(str(self.model_path))
        self.conf = float(conf)
        names = getattr(self.model, "names", {}) or {}
        self.model_names = {str(k): str(v) for k, v in dict(names).items()} if isinstance(names, dict) else {}

    def detect_buttons(self, frame_bgr: Any) -> List[ButtonDetection]:
        if frame_bgr is None:
            return []
        results = self.model.predict(frame_bgr, verbose=False, conf=self.conf)
        detections: List[ButtonDetection] = []
        ts = time.monotonic()
        for result in results or []:
            names = getattr(result, "names", {}) or self.model_names
            boxes = getattr(result, "boxes", None)
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
            self.button_detector = build_default_button_detector(self.config)
        self.state = STATE_IDLE
        self.execution_lock = False
        self.locked_until_reset = False
        self.last_cycle_key: Optional[str] = None
        self.last_hand_id: Optional[str] = None
        self.wait_started_at: Optional[float] = None
        self.post_click_until = 0.0
        self.last_idle_move_at = 0.0
        self.next_idle_at = self._schedule_next_idle(time.monotonic())
        self.last_plan_name: Optional[str] = None
        self.last_normalized_action: Optional[str] = None
        self.last_raw_action: Optional[str] = None
        self.debug_root = Path(self.config.debug_root_dir).expanduser().resolve()
        self.debug_root.mkdir(parents=True, exist_ok=True)

    def _build_default_mouse_backend(self) -> MouseBackendProtocol:
        if os.name == "nt" and ctypes is not None:
            try:
                return WindowsMouseBackend()
            except Exception:
                LOGGER.exception("%s failed to initialize Windows mouse backend", self.config.log_prefix)
        return NoopMouseBackend()

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
    ) -> AutoClickSnapshot:
        street = "preflop"
        hand_id = None
        total_pot_bb = None
        meta: Dict[str, object] = {}
        if hero_decision is not None:
            street = str(getattr(hero_decision, "street", street) or street).lower()
            debug = getattr(hero_decision, "debug", {}) or {}
            if isinstance(debug, dict):
                meta.update({str(k): v for k, v in debug.items() if k in {"node_type", "opener_pos", "three_bettor_pos", "four_bettor_pos", "limpers", "callers"}})
        if hand is not None:
            hand_id = getattr(hand, "hand_id", None)
            street_state = getattr(hand, "street_state", None)
            if isinstance(street_state, dict):
                street = str(street_state.get("current_street") or street).lower()
            table_amount_state = getattr(hand, "table_amount_state", None)
            if isinstance(table_amount_state, dict):
                total_pot = table_amount_state.get("total_pot")
                if isinstance(total_pot, dict):
                    amount = total_pot.get("amount_bb")
                    try:
                        total_pot_bb = None if amount is None else float(amount)
                    except Exception:
                        total_pot_bb = None
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
            frame_source="launcher_frame",
        )

    def step(self, snapshot: AutoClickSnapshot, *, frame_bgr: Any = None, detections: Optional[Sequence[ButtonDetection]] = None) -> AutoClickResult:
        now = time.monotonic()
        events: List[AutoClickEvent] = []
        if not self.config.enabled:
            return AutoClickResult(state=self.state, events=events)

        cycle_key = self._cycle_key(snapshot)
        if self._should_reset_cycle(snapshot, cycle_key):
            events.extend(self._reset_cycle(snapshot, cycle_key, now))

        if not snapshot.active_hero_present:
            self.state = STATE_IDLE
            self.locked_until_reset = False
            self.execution_lock = False
            if self.config.enable_idle_movement:
                self._maybe_idle_move(now, events)
            return AutoClickResult(state=self.state, events=events)

        if self.wait_started_at is None:
            self.wait_started_at = float(snapshot.decision_started_at or now)
            self.state = STATE_ACTIVE_HERO_WAITING_DECISION
            events.append(self._event("active_hero_detected", hand_id=snapshot.hand_id or "", street=snapshot.street))
            events.append(self._event("decision_wait_started", started_at=self.wait_started_at))

        frame_used, capture_meta = self._get_detection_frame(snapshot, frame_bgr)
        raw_detections = list(detections or [])
        if not raw_detections and frame_used is not None and self.button_detector is not None:
            try:
                raw_detections = list(self.button_detector.detect_buttons(frame_used))
            except Exception as exc:
                LOGGER.exception("%s button detector failed", self.config.log_prefix)
                events.append(self._event("button_detector_error", error=str(exc)))
        filtered = self._prepare_detections(raw_detections, snapshot)
        self._write_debug_observation(snapshot, frame_used, raw_detections, filtered, capture_meta)

        if now < self.post_click_until:
            self.state = STATE_POST_CLICK_COOLDOWN
            return AutoClickResult(state=self.state, locked=True, events=events)

        if snapshot.critical_error_flag:
            self.state = STATE_FAILSAFE_EXECUTION
            plan = self._build_failsafe_plan(filtered, reason=snapshot.critical_error_text or "critical_error")
            executed = self._execute_plan(plan, filtered, snapshot, events)
            self._write_debug_plan(snapshot, plan, filtered, executed)
            return self._result_from_state(plan, executed, events, locked=executed)

        if self.locked_until_reset:
            self.state = STATE_LOCKED_UNTIL_RESET
            events.append(self._event("execution_locked", hand_id=snapshot.hand_id or "", plan_name=self.last_plan_name or ""))
            return AutoClickResult(
                state=self.state,
                executed=False,
                locked=True,
                plan_name=self.last_plan_name,
                normalized_action=self.last_normalized_action,
                raw_action=self.last_raw_action,
                events=events,
            )

        timeout_seconds = self._resolve_timeout(snapshot)
        if now - self.wait_started_at >= timeout_seconds and not snapshot.decision_ready:
            self.state = STATE_FAILSAFE_EXECUTION
            plan = self._build_failsafe_plan(filtered, reason="decision_timeout")
            events.append(self._event("decision_timeout", timeout_seconds=timeout_seconds))
            executed = self._execute_plan(plan, filtered, snapshot, events)
            self._write_debug_plan(snapshot, plan, filtered, executed)
            return self._result_from_state(plan, executed, events, locked=executed)

        if not snapshot.decision_ready or snapshot.hero_decision is None:
            self.state = STATE_ACTIVE_HERO_WAITING_DECISION
            return AutoClickResult(state=self.state, events=events)

        self.state = STATE_DECISION_READY
        raw_action = self._resolve_raw_action(snapshot.hero_decision)
        normalized = self._normalize_action(snapshot)
        plan = self._build_click_plan(normalized, raw_action)
        self.last_plan_name = plan.plan_name
        self.last_normalized_action = plan.normalized_action
        self.last_raw_action = plan.raw_action
        events.append(self._event("decision_received", raw_action=raw_action, normalized_action=normalized, street=snapshot.street))
        events.append(self._event("click_plan_built", plan_name=plan.plan_name, primary_button=plan.primary_button, secondary_button=plan.secondary_button or ""))
        executed = self._execute_plan(plan, filtered, snapshot, events)
        self._write_debug_plan(snapshot, plan, filtered, executed)
        return self._result_from_state(plan, executed, events, locked=executed)

    def _cycle_key(self, snapshot: AutoClickSnapshot) -> str:
        if snapshot.hand_id:
            return f"hand:{snapshot.hand_id}"
        return f"started:{snapshot.decision_started_at:.6f}"

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
        self.execution_lock = False
        self.locked_until_reset = False
        self.wait_started_at = float(snapshot.decision_started_at or now)
        self.post_click_until = 0.0
        self.last_cycle_key = cycle_key
        self.last_hand_id = snapshot.hand_id
        self.last_plan_name = None
        self.last_raw_action = None
        self.last_normalized_action = None
        self.state = STATE_ACTIVE_HERO_WAITING_DECISION if snapshot.active_hero_present else STATE_IDLE
        return [self._event("cycle_reset", hand_id=snapshot.hand_id or "", state=self.state)]

    def _schedule_next_idle(self, now: float) -> float:
        return now + self.rng.uniform(self.config.idle_pause_sec_min, self.config.idle_pause_sec_max)

    def _maybe_idle_move(self, now: float, events: List[AutoClickEvent]) -> None:
        if now < self.next_idle_at:
            return
        distance = self.rng.randint(self.config.idle_move_distance_px_min, self.config.idle_move_distance_px_max)
        angle = self.rng.random() * math.tau
        start_x, start_y = self.mouse.get_position()
        target_x = int(start_x + math.cos(angle) * distance)
        target_y = int(start_y + math.sin(angle) * distance)
        self._human_move_to(target_x, target_y, events, idle=True)
        self.last_idle_move_at = now
        self.next_idle_at = self._schedule_next_idle(now)

    def _get_detection_frame(self, snapshot: AutoClickSnapshot, frame_bgr: Any) -> Tuple[Any, Dict[str, object]]:
        if self.config.force_primary_monitor_capture:
            primary = self._capture_primary_monitor()
            if primary is not None:
                frame, meta = primary
                snapshot.monitor_left = int(meta.get("left", 0))
                snapshot.monitor_top = int(meta.get("top", 0))
                snapshot.monitor_width = int(meta.get("width", 0))
                snapshot.monitor_height = int(meta.get("height", 0))
                snapshot.monitor_name = "primary"
                snapshot.frame_source = "mss_primary_monitor"
                return frame, meta
        snapshot.frame_source = "launcher_frame"
        return frame_bgr, {
            "name": snapshot.monitor_name,
            "left": snapshot.monitor_left,
            "top": snapshot.monitor_top,
            "width": snapshot.monitor_width,
            "height": snapshot.monitor_height,
        }

    def _capture_primary_monitor(self) -> Optional[Tuple[Any, Dict[str, object]]]:
        if mss is None or np is None:
            return None
        with mss.mss() as sct:
            if len(sct.monitors) < 2:
                return None
            monitor = dict(sct.monitors[1])
            shot = sct.grab(monitor)
            frame = np.array(shot)
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            return frame, {
                "name": "primary",
                "left": int(monitor.get("left", 0)),
                "top": int(monitor.get("top", 0)),
                "width": int(monitor.get("width", 0)),
                "height": int(monitor.get("height", 0)),
            }

    def _resolve_timeout(self, snapshot: AutoClickSnapshot) -> float:
        timeout = float(self.config.timeout_default_sec)
        try:
            pot = None if snapshot.total_pot_bb is None else float(snapshot.total_pot_bb)
        except Exception:
            pot = None
        if pot is not None and pot > float(self.config.timeout_big_pot_threshold_bb):
            timeout += float(self.config.timeout_big_pot_extra_sec)
        return timeout

    def _resolve_raw_action(self, decision: HeroDecision) -> str:
        pre = getattr(decision, "preflop", None)
        if pre is not None:
            value = getattr(pre, "action", None)
            if value:
                return str(value).strip().lower()
        post = getattr(decision, "postflop", None)
        if post is not None:
            value = getattr(post, "action", None)
            if value:
                return str(value).strip().lower()
        reason = getattr(decision, "reason", None)
        if reason:
            txt = str(reason).strip().lower()
            if ":" in txt:
                txt = txt.split(":")[-1].strip()
            return txt
        return str(getattr(decision, "engine_action", "") or "").strip().lower()

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
                pass
        return None

    def _nearest_supported_size(self, value: Optional[float]) -> Optional[int]:
        if value is None:
            return None
        candidates = [33, 50, 70, 98]
        numeric = float(value)
        nearest = min(candidates, key=lambda x: abs(x - numeric))
        return nearest if abs(nearest - numeric) <= 6.0 else None

    def _parse_sizing_suffix(self, raw_action: str) -> Optional[int]:
        for size in (33, 50, 70, 98):
            if str(size) in raw_action:
                return size
        return None

    def _normalize_action(self, snapshot: AutoClickSnapshot) -> str:
        decision = snapshot.hero_decision
        if decision is None:
            return ""
        raw_action = self._resolve_raw_action(decision)
        engine_action = str(getattr(decision, "engine_action", "") or "").strip().lower()
        street = str(snapshot.street or getattr(decision, "street", "preflop") or "preflop").lower()
        size_pct = self._extract_size_pct(decision)
        rounded_size = self._nearest_supported_size(size_pct)
        node_type = str(snapshot.solver_context_meta.get("node_type") or "").lower()

        if raw_action == "fold" or engine_action == "fold":
            return "fold"
        if raw_action == "check" or engine_action == "check":
            return "check"
        if raw_action in {"call", "limp"} or engine_action == "call":
            return "call"
        if raw_action in RAW_ISO_ACTIONS or (street == "preflop" and node_type == "facing_limp" and engine_action == "raise"):
            return "iso_raise_98"
        if raw_action in RAW_THREEBET_ACTIONS:
            return "threebet_98"
        if raw_action in RAW_FOURBET_ACTIONS:
            return "fourbet_98"
        if raw_action in RAW_FIVEBET_ACTIONS:
            return "fivebet_98"
        if raw_action.startswith("bet_") or raw_action.startswith("raise_"):
            parsed = self._parse_sizing_suffix(raw_action)
            if parsed in SIZE_TO_BUTTON:
                return f"raise_{parsed}"
        if rounded_size in {33, 50, 70}:
            return f"raise_{rounded_size}"
        if rounded_size == 98:
            return "raise_98"
        if street == "preflop" and engine_action == "raise" and rounded_size is None:
            return "raise_only"
        if engine_action in {"raise", "bet", "all_in"} or raw_action in {"raise", "bet", "jam", "all_in", "all-in"}:
            return "raise_only"
        return engine_action or raw_action

    def _build_click_plan(self, normalized_action: str, raw_action: str) -> AutoClickPlan:
        if normalized_action == "fold":
            return AutoClickPlan(ACTION_CLICK_FOLD, normalized_action, raw_action, BUTTON_CHECK)
        if normalized_action == "check":
            return AutoClickPlan(ACTION_CLICK_CHECK, normalized_action, raw_action, BUTTON_CHECK)
        if normalized_action == "call":
            return AutoClickPlan(ACTION_CLICK_CALL, normalized_action, raw_action, BUTTON_CALL)
        if normalized_action == "raise_only":
            return AutoClickPlan(ACTION_CLICK_RAISE_ONLY, normalized_action, raw_action, BUTTON_RAISE)
        if normalized_action == "raise_33":
            return AutoClickPlan(ACTION_CLICK_SIZE_THEN_RAISE_33, normalized_action, raw_action, BUTTON_33, BUTTON_RAISE, False)
        if normalized_action == "raise_50":
            return AutoClickPlan(ACTION_CLICK_SIZE_THEN_RAISE_50, normalized_action, raw_action, BUTTON_50, BUTTON_RAISE, False)
        if normalized_action == "raise_70":
            return AutoClickPlan(ACTION_CLICK_SIZE_THEN_RAISE_70, normalized_action, raw_action, BUTTON_70, BUTTON_RAISE, False)
        if normalized_action in {"raise_98", "iso_raise_98", "threebet_98", "fourbet_98", "fivebet_98"}:
            return AutoClickPlan(ACTION_CLICK_98_THEN_RAISE, normalized_action, raw_action, BUTTON_98, BUTTON_RAISE, False)
        if normalized_action == "check_fold":
            return AutoClickPlan(ACTION_CLICK_CHECK_FOLD, normalized_action, raw_action, BUTTON_CHECK_FOLD)
        return AutoClickPlan(ACTION_CLICK_RAISE_ONLY, normalized_action, raw_action, BUTTON_RAISE)

    def _build_failsafe_plan(self, detections_by_class: Dict[str, List[ButtonDetection]], *, reason: str) -> AutoClickPlan:
        if detections_by_class.get(BUTTON_CHECK):
            return AutoClickPlan(ACTION_CLICK_CHECK, "check", reason, BUTTON_CHECK, is_failsafe=True)
        if detections_by_class.get(BUTTON_FOLD):
            return AutoClickPlan(ACTION_CLICK_FOLD, "fold", reason, BUTTON_FOLD, is_failsafe=True)
        if detections_by_class.get(BUTTON_CHECK_FOLD):
            return AutoClickPlan(ACTION_CLICK_CHECK_FOLD, "check_fold", reason, BUTTON_CHECK_FOLD, is_failsafe=True)
        return AutoClickPlan(ACTION_CLICK_FOLD, "fold", reason, BUTTON_FOLD, is_failsafe=True)

    def _prepare_detections(self, detections: Sequence[ButtonDetection], snapshot: AutoClickSnapshot) -> Dict[str, List[ButtonDetection]]:
        now = time.monotonic()
        ttl_seconds = self.config.button_detection_ttl_ms / 1000.0
        thresholds = {BUTTON_RAISE: self.config.detector_conf_raise, BUTTON_CHECK: self.config.detector_conf_check}
        kept: List[ButtonDetection] = []
        for det in detections:
            canonical = canonicalize_button_class_name(det.class_name)
            if canonical is None:
                continue
            det.class_name = canonical
            if float(det.confidence) < thresholds.get(canonical, self.config.detector_conf_default):
                continue
            if now - float(det.frame_ts) > ttl_seconds:
                continue
            kept.append(det)
        deduped = self._nms_by_class(kept)
        grouped: Dict[str, List[ButtonDetection]] = {name: [] for name in BUTTON_CLASS_NAMES}
        for det in deduped:
            grouped[det.class_name].append(det)
        panel_center = self._panel_center(snapshot.action_panel_bbox)
        for name, items in grouped.items():
            if len(items) <= 1:
                continue
            if panel_center is None:
                items.sort(key=lambda item: item.confidence, reverse=True)
            else:
                items.sort(key=lambda item: (self._distance((item.center_x, item.center_y), panel_center), -item.confidence))
        return grouped

    def _nms_by_class(self, detections: Sequence[ButtonDetection]) -> List[ButtonDetection]:
        grouped: Dict[str, List[ButtonDetection]] = {}
        for det in detections:
            grouped.setdefault(det.class_name, []).append(det)
        final: List[ButtonDetection] = []
        for items in grouped.values():
            items_sorted = sorted(items, key=lambda item: item.confidence, reverse=True)
            kept: List[ButtonDetection] = []
            for candidate in items_sorted:
                if all(self._iou(candidate.bbox, existing.bbox) < self.config.nms_iou_threshold for existing in kept):
                    kept.append(candidate)
            final.extend(kept)
        return final

    def _execute_plan(self, plan: AutoClickPlan, detections_by_class: Dict[str, List[ButtonDetection]], snapshot: AutoClickSnapshot, events: List[AutoClickEvent]) -> bool:
        self.execution_lock = True
        self.state = STATE_EXECUTING_PLAN

        primary = self._resolve_primary_detection(plan, detections_by_class)
        if primary is None:
            primary = self._retry_find_button(plan.primary_button, snapshot, events)
        if primary is None:
            events.append(self._event("critical_click_failure", stage="primary_button_missing", button=plan.primary_button))
            self.execution_lock = False
            self._write_events(events)
            return False
        if not self._click_detection(primary, snapshot, events):
            events.append(self._event("critical_click_failure", stage="primary_click_failed", button=primary.class_name))
            self.execution_lock = False
            self._write_events(events)
            return False

        if plan.secondary_button:
            time.sleep(self.rng.randint(self.config.secondary_click_settle_ms_min, self.config.secondary_click_settle_ms_max) / 1000.0)
            secondary = self._select_button(detections_by_class, plan.secondary_button)
            if secondary is None or not self._detection_is_fresh(secondary):
                secondary = self._retry_find_button(plan.secondary_button, snapshot, events)
            if secondary is None:
                events.append(self._event("critical_click_failure", stage="secondary_button_missing", button=plan.secondary_button))
                self.execution_lock = False
                self._write_events(events)
                return False
            if not self._click_detection(secondary, snapshot, events):
                events.append(self._event("critical_click_failure", stage="secondary_click_failed", button=secondary.class_name))
                self.execution_lock = False
                self._write_events(events)
                return False

        cooldown = self.rng.uniform(self.config.post_click_cooldown_sec_min, self.config.post_click_cooldown_sec_max)
        self.post_click_until = time.monotonic() + cooldown
        self.locked_until_reset = True
        self.execution_lock = False
        self.state = STATE_LOCKED_UNTIL_RESET
        self._write_events(events)
        return True

    def _resolve_primary_detection(self, plan: AutoClickPlan, detections_by_class: Dict[str, List[ButtonDetection]]) -> Optional[ButtonDetection]:
        if plan.plan_name == ACTION_CLICK_FOLD:
            for button in (BUTTON_CHECK, BUTTON_FOLD, BUTTON_CHECK_FOLD):
                det = self._select_button(detections_by_class, button)
                if det is not None:
                    return det
            return None
        if plan.plan_name == ACTION_CLICK_CHECK:
            for button in (BUTTON_CHECK, BUTTON_CHECK_FOLD):
                det = self._select_button(detections_by_class, button)
                if det is not None:
                    return det
            return None
        if plan.plan_name == ACTION_CLICK_CALL:
            return self._select_button(detections_by_class, BUTTON_CALL)
        if plan.plan_name == ACTION_CLICK_CHECK_FOLD:
            return self._select_button(detections_by_class, BUTTON_CHECK_FOLD)
        return self._select_button(detections_by_class, plan.primary_button)

    def _select_button(self, detections_by_class: Dict[str, List[ButtonDetection]], button_name: str) -> Optional[ButtonDetection]:
        items = detections_by_class.get(button_name) or []
        return items[0] if items else None

    def _retry_find_button(self, button_name: str, snapshot: AutoClickSnapshot, events: List[AutoClickEvent]) -> Optional[ButtonDetection]:
        start = time.monotonic()
        budget = self.config.retry_total_budget_ms / 1000.0
        for attempt in range(1, self.config.retry_count_max + 1):
            if time.monotonic() - start > budget:
                break
            time.sleep(self.rng.randint(self.config.retry_sleep_ms_min, self.config.retry_sleep_ms_max) / 1000.0)
            frame, capture_meta = self._get_detection_frame(snapshot, None)
            if frame is None or self.button_detector is None:
                continue
            try:
                raw = list(self.button_detector.detect_buttons(frame))
            except Exception as exc:
                events.append(self._event("button_detector_error", error=str(exc), attempt=attempt))
                continue
            filtered = self._prepare_detections(raw, snapshot)
            self._write_debug_observation(snapshot, frame, raw, filtered, capture_meta)
            det = self._select_button(filtered, button_name)
            events.append(self._event("retry_attempt", button=button_name, attempt=attempt, found=bool(det)))
            if det is not None:
                return det
        return None

    def _click_detection(self, detection: ButtonDetection, snapshot: AutoClickSnapshot, events: List[AutoClickEvent]) -> bool:
        if not self._detection_is_fresh(detection):
            return False
        x, y = self._jittered_point(detection, snapshot)
        events.append(self._event("mouse_move_started", target_x=x, target_y=y, button=detection.class_name))
        self._human_move_to(x, y, events)
        time.sleep(self.rng.randint(self.config.click_down_up_delay_ms_min, self.config.click_down_up_delay_ms_max) / 1000.0)
        self.mouse.left_down()
        time.sleep(self.rng.randint(self.config.click_down_up_delay_ms_min, self.config.click_down_up_delay_ms_max) / 1000.0)
        self.mouse.left_up()
        click_payload = {
            "ts": time.monotonic(),
            "button": detection.class_name,
            "x": x,
            "y": y,
            "confidence": float(detection.confidence),
            "bbox": [float(v) for v in detection.bbox],
            "monitor_left": snapshot.monitor_left,
            "monitor_top": snapshot.monitor_top,
        }
        self._append_jsonl(self.debug_root / "clicks.jsonl", click_payload)
        events.append(self._event("button_clicked", **click_payload))
        return True

    def _maybe_apply_scroll(self, events: List[AutoClickEvent]) -> None:
        if self.rng.random() > self.config.scroll_enabled_probability:
            return
        direction = 1 if self.rng.random() < self.config.scroll_direction_up_probability else -1
        steps = self.rng.randint(self.config.scroll_steps_min, self.config.scroll_steps_max)
        for _ in range(steps):
            self.mouse.wheel(120 * direction)
            time.sleep(self.rng.randint(self.config.scroll_pause_ms_min, self.config.scroll_pause_ms_max) / 1000.0)
        events.append(self._event("scroll_applied", direction="up" if direction > 0 else "down", steps=steps))

    def _human_move_to(self, target_x: int, target_y: int, events: List[AutoClickEvent], *, idle: bool = False) -> None:
        start_x, start_y = self.mouse.get_position()
        duration_ms = self.rng.randint(self.config.move_duration_ms_min, self.config.move_duration_ms_max)
        duration_sec = duration_ms / 1000.0
        steps = max(8, min(40, int(duration_ms / 16)))
        cp1 = self._control_point(start_x, start_y, target_x, target_y, factor=self.rng.uniform(0.22, 0.45))
        cp2 = self._control_point(start_x, start_y, target_x, target_y, factor=self.rng.uniform(0.55, 0.78))
        start_ts = time.monotonic()
        for step in range(1, steps + 1):
            t = step / steps
            eased = self._ease_in_out(t)
            x = self._cubic_bezier(start_x, cp1[0], cp2[0], target_x, eased)
            y = self._cubic_bezier(start_y, cp1[1], cp2[1], target_y, eased)
            self.mouse.move_to(int(round(x)), int(round(y)))
            target_time = start_ts + duration_sec * t
            remaining = target_time - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        if idle:
            events.append(self._event("idle_move_applied", target_x=target_x, target_y=target_y))

    def _jittered_point(self, detection: ButtonDetection, snapshot: AutoClickSnapshot) -> Tuple[int, int]:
        max_jx = min(self.config.target_jitter_px_max, max(self.config.target_jitter_px_min, int(max(4.0, detection.width / 4.0))))
        max_jy = min(self.config.target_jitter_px_max, max(self.config.target_jitter_px_min, int(max(4.0, detection.height / 4.0))))
        dx = self.rng.randint(-max_jx, max_jx)
        dy = self.rng.randint(-max_jy, max_jy)
        abs_x = int(round(snapshot.monitor_left + detection.center_x + dx))
        abs_y = int(round(snapshot.monitor_top + detection.center_y + dy))
        return abs_x, abs_y

    def _panel_center(self, bbox: Optional[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float]]:
        if bbox is None:
            return None
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _iou(self, box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return 0.0 if denom <= 0 else inter / denom

    def _detection_is_fresh(self, detection: ButtonDetection) -> bool:
        return (time.monotonic() - float(detection.frame_ts)) <= (self.config.button_detection_ttl_ms / 1000.0)

    def _control_point(self, x1: int, y1: int, x2: int, y2: int, *, factor: float) -> Tuple[float, float]:
        mid_x = x1 + (x2 - x1) * factor
        mid_y = y1 + (y2 - y1) * factor
        offset_x = (y2 - y1) * self.rng.uniform(-0.12, 0.12)
        offset_y = (x1 - x2) * self.rng.uniform(-0.12, 0.12)
        return mid_x + offset_x, mid_y + offset_y

    def _ease_in_out(self, value: float) -> float:
        return 3 * value * value - 2 * value * value * value

    def _cubic_bezier(self, p0: float, p1: float, p2: float, p3: float, t: float) -> float:
        inv = 1.0 - t
        return inv * inv * inv * p0 + 3 * inv * inv * t * p1 + 3 * inv * t * t * p2 + t * t * t * p3

    def _write_debug_observation(self, snapshot: AutoClickSnapshot, frame_bgr: Any, raw: Sequence[ButtonDetection], filtered: Dict[str, List[ButtonDetection]], capture_meta: Dict[str, object]) -> None:
        detector_info = {
            "button_detector_enabled": bool(self.button_detector is not None),
            "button_model_path": str(getattr(self.button_detector, "model_path", "")) if self.button_detector is not None else None,
            "button_model_names": dict(getattr(self.button_detector, "model_names", {}) or {}),
            "raw_detection_count": len(raw),
            "filtered_detection_count": sum(len(v) for v in filtered.values()),
            "frame_source": snapshot.frame_source,
        }
        observation = {
            "ts": time.monotonic(),
            "state": self.state,
            "hand_id": snapshot.hand_id,
            "street": snapshot.street,
            "active_hero_present": snapshot.active_hero_present,
            "decision_ready": snapshot.decision_ready,
            "frame_source": snapshot.frame_source,
            "monitor": {
                "name": snapshot.monitor_name,
                "left": snapshot.monitor_left,
                "top": snapshot.monitor_top,
                "width": snapshot.monitor_width,
                "height": snapshot.monitor_height,
            },
            "capture_meta": capture_meta,
            "raw_detections": [
                {
                    "class_name": d.class_name,
                    "source_class_name": d.source_class_name,
                    "confidence": float(d.confidence),
                    "bbox": [float(v) for v in d.bbox],
                    "center_x": float(d.center_x),
                    "center_y": float(d.center_y),
                }
                for d in raw
            ],
            "filtered_detections": {
                key: [
                    {
                        "confidence": float(d.confidence),
                        "bbox": [float(v) for v in d.bbox],
                        "center_x": float(d.center_x),
                        "center_y": float(d.center_y),
                    }
                    for d in value
                ]
                for key, value in filtered.items() if value
            },
            "hero_decision": self._serialize_decision(snapshot.hero_decision),
        }
        self._write_json(self.debug_root / "last_observation.json", observation)
        self._write_json(self.debug_root / "detector_health.json", {"ts": time.monotonic(), **detector_info})
        if frame_bgr is not None and cv2 is not None:
            try:
                cv2.imwrite(str(self.debug_root / "frame_latest.png"), frame_bgr)
            except Exception:
                pass

    def _write_debug_plan(self, snapshot: AutoClickSnapshot, plan: AutoClickPlan, filtered: Dict[str, List[ButtonDetection]], executed: bool) -> None:
        payload = {
            "ts": time.monotonic(),
            "note": "decision_ready",
            "state": self.state,
            "hand_id": snapshot.hand_id,
            "street": snapshot.street,
            "plan_name": plan.plan_name,
            "normalized_action": plan.normalized_action,
            "raw_action": plan.raw_action,
            "primary_button": plan.primary_button,
            "secondary_button": plan.secondary_button,
            "allow_scroll_between": plan.allow_scroll_between,
            "is_failsafe": plan.is_failsafe,
            "visible_buttons": [key for key, value in filtered.items() if value],
            "frame_source": snapshot.frame_source,
            "monitor": {
                "name": snapshot.monitor_name,
                "left": snapshot.monitor_left,
                "top": snapshot.monitor_top,
                "width": snapshot.monitor_width,
                "height": snapshot.monitor_height,
            },
            "result": {"state": self.state, "executed": executed, "locked": self.locked_until_reset},
        }
        self._write_json(self.debug_root / "last_plan.json", payload)

    def _serialize_decision(self, decision: Optional[HeroDecision]) -> Dict[str, object]:
        if decision is None:
            return {}
        return {
            "engine_action": getattr(decision, "engine_action", None),
            "reason": getattr(decision, "reason", None),
            "size_pct": getattr(decision, "size_pct", None),
            "amount_to": getattr(decision, "amount_to", None),
        }

    def _write_events(self, events: Sequence[AutoClickEvent]) -> None:
        for event in events:
            self._append_jsonl(self.debug_root / "events.jsonl", {"name": event.name, "ts": event.ts, "payload": _json_safe(event.payload)})

    def _write_json(self, path: Path, payload: Dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_jsonl(self, path: Path, payload: Dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(payload), ensure_ascii=False) + "\n")

    def _result_from_state(self, plan: AutoClickPlan, executed: bool, events: List[AutoClickEvent], *, locked: bool) -> AutoClickResult:
        return AutoClickResult(
            state=self.state,
            executed=executed,
            locked=locked,
            plan_name=plan.plan_name,
            normalized_action=plan.normalized_action,
            raw_action=plan.raw_action,
            events=events,
        )

    def _event(self, name: str, **payload: object) -> AutoClickEvent:
        return AutoClickEvent(name=name, ts=time.monotonic(), payload={str(k): v for k, v in payload.items()})


def build_default_button_detector(config: Optional[AutoClickConfig] = None) -> Optional[YoloButtonDetector]:
    cfg = config or AutoClickConfig()
    try:
        return YoloButtonDetector(cfg.button_model_path)
    except Exception:
        LOGGER.exception("Failed to build YOLO button detector")
        return None


def create_default_auto_click_runtime(*, with_button_detector: bool = True) -> AutoClickRuntime:
    config = AutoClickConfig()
    detector = build_default_button_detector(config) if with_button_detector else None
    return AutoClickRuntime(config=config, button_detector=detector)


__all__ = [
    "AutoClickConfig",
    "AutoClickRuntime",
    "AutoClickSnapshot",
    "AutoClickResult",
    "AutoClickPlan",
    "AutoClickEvent",
    "ButtonDetection",
    "build_default_button_detector",
    "create_default_auto_click_runtime",
]
