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
        self.info.setPlainText(json.dumps(safe_payload, ensure_ascii=False, indent=2, sort_keys=False))
