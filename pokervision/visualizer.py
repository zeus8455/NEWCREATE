from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

try:
    from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
except Exception:  # pragma: no cover
    QtCore = QtGui = QtWidgets = None


def _to_json_safe(value: Any, _seen: set[int] | None = None) -> Any:
    """Convert arbitrary runtime/debug payloads into JSON-safe structures.

    This is intentionally defensive because solver/debug payloads may contain
    dataclass instances like RangeSource, tuples, sets, Paths and other helper
    objects that the standard json encoder cannot serialize directly.
    """
    if _seen is None:
        _seen = set()

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    obj_id = id(value)
    if obj_id in _seen:
        return "<recursive-ref>"

    if is_dataclass(value):
        _seen.add(obj_id)
        try:
            return _to_json_safe(asdict(value), _seen)
        finally:
            _seen.discard(obj_id)

    if isinstance(value, dict):
        _seen.add(obj_id)
        try:
            return {str(k): _to_json_safe(v, _seen) for k, v in value.items()}
        finally:
            _seen.discard(obj_id)

    if isinstance(value, (list, tuple, set, frozenset)):
        _seen.add(obj_id)
        try:
            return [_to_json_safe(v, _seen) for v in value]
        finally:
            _seen.discard(obj_id)

    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        _seen.add(obj_id)
        try:
            return _to_json_safe(value.to_dict(), _seen)
        except Exception:
            pass
        finally:
            _seen.discard(obj_id)

    if hasattr(value, "__dict__"):
        _seen.add(obj_id)
        try:
            return {
                str(k): _to_json_safe(v, _seen)
                for k, v in vars(value).items()
                if not str(k).startswith("_")
            }
        finally:
            _seen.discard(obj_id)

    return str(value)


def _extract_postflop_range_trace(render_state: Any) -> dict[str, Any]:
    if not isinstance(render_state, dict):
        return {}
    analysis_panel = render_state.get("analysis_panel")
    if isinstance(analysis_panel, dict):
        trace = analysis_panel.get("postflop_range_trace")
        if isinstance(trace, dict) and trace:
            return trace
    for block_name in ("solver_output", "solver_input", "advisor_input"):
        block = render_state.get(block_name)
        if not isinstance(block, dict):
            continue
        for key in ("postflop_range_trace", "runtime_range_state"):
            trace = block.get(key)
            if isinstance(trace, dict) and trace:
                return trace
    return {}


def _format_range_trace(trace: dict[str, Any]) -> list[str]:
    if not trace:
        return []
    lines: list[str] = []
    requested = trace.get("requested_range_build_mode")
    route = trace.get("range_build_mode")
    payload_kind = trace.get("payload_kind")
    resolved_street = trace.get("resolved_street")
    if requested:
        lines.append(f"Requested mode: {requested}")
    if route:
        lines.append(f"Used route: {route}")
    if payload_kind:
        lines.append(f"Payload: {payload_kind}")
    if resolved_street:
        lines.append(f"Street: {str(resolved_street).upper()}")
    for item in trace.get("villain_sources_summary", []) or []:
        if not isinstance(item, dict):
            continue
        actor = item.get("name") or item.get("villain_pos") or "Villain"
        source_type = item.get("source_type") or "range"
        combo_count = item.get("combo_count")
        total_weight = item.get("total_weight")
        line = f"{actor}: {source_type}"
        if combo_count not in (None, ""):
            line += f" | {combo_count}c"
        if total_weight not in (None, ""):
            line += f" | w={total_weight}"
        lines.append(line)
    reports = trace.get("villain_range_reports") or trace.get("villain_reports") or []
    for report in reports:
        if not isinstance(report, dict):
            continue
        actor = str(report.get("name") or report.get("villain_pos") or "Villain")
        payload = report.get("report") if isinstance(report.get("report"), dict) else {}
        steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
        for step in steps[:3]:
            if not isinstance(step, dict):
                continue
            street = str(step.get("street") or trace.get("resolved_street") or "").upper()
            action = str(step.get("action") or step.get("semantic_action") or "action").upper()
            before = step.get("range_before_source") if isinstance(step.get("range_before_source"), dict) else {}
            after = step.get("range_after_source") if isinstance(step.get("range_after_source"), dict) else {}
            before_count = len(before.get("weighted_combos", [])) if isinstance(before.get("weighted_combos"), list) else None
            after_count = len(after.get("weighted_combos", [])) if isinstance(after.get("weighted_combos"), list) else None
            detail = f"{actor} {street} {action}"
            if before_count is not None or after_count is not None:
                detail += f" | {before_count if before_count is not None else '?'}c→{after_count if after_count is not None else '?'}c"
            lines.append(detail)
    return lines


def _build_monitor_text(render_state: Any, status: Any, safe_payload: dict[str, Any]) -> str:
    lines: list[str] = []
    if isinstance(render_state, dict):
        lines.append(f"Hand: {render_state.get('hand_id') or '—'}")
        lines.append(f"Street: {str(render_state.get('street') or '—').upper()}")
        lines.append(f"Hero: {render_state.get('hero_position') or '—'}")
        if render_state.get('recommended_action'):
            lines.append(f"Action: {render_state.get('recommended_action')}")
        if render_state.get('solver_status'):
            lines.append(f"Solver: {render_state.get('solver_status')}")
    trace = _extract_postflop_range_trace(render_state)
    trace_lines = _format_range_trace(trace)
    if trace_lines:
        if lines:
            lines.append("")
        lines.append("Range trace")
        lines.append("-----------")
        lines.extend(trace_lines[:14])
    if lines:
        lines.append("")
        lines.append("Raw payload")
        lines.append("-----------")
    lines.append(json.dumps(safe_payload, ensure_ascii=False, indent=2, sort_keys=False))
    return "\n".join(lines)


class DebugMonitorWindow:  # pragma: no cover - UI is optional in tests
    def __init__(self, shared_state, refresh_ms: int = 100):
        if QtWidgets is None:
            raise RuntimeError("PySide6 is not installed; DebugMonitorWindow unavailable")

        self.shared_state = shared_state

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("PokerVision Debug Monitor")
        self.window.resize(1100, 760)

        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)

        self.image_label = QtWidgets.QLabel("Waiting for frames")
        self.image_label.setMinimumSize(900, 520)
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.info = QtWidgets.QPlainTextEdit()
        self.info.setReadOnly(True)

        layout.addWidget(self.image_label, stretch=7)
        layout.addWidget(self.info, stretch=3)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(refresh_ms)

    def show(self):
        self.window.show()

    def refresh(self):
        import cv2

        frame, render_state, status = self.shared_state.snapshot()

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            image = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(image).scaled(
                self.image_label.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(pix)

        payload = {
            "render_state": render_state,
            "status": status,
        }
        safe_payload = _to_json_safe(payload)
        self.info.setPlainText(_build_monitor_text(render_state, status, safe_payload))
