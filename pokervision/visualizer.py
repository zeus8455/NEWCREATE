from __future__ import annotations

import json
from typing import Optional

try:
    from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
except Exception:  # pragma: no cover
    QtCore = QtGui = QtWidgets = None


class DebugMonitorWindow:  # pragma: no cover - UI is optional in tests
    def __init__(self, shared_state, refresh_ms: int = 100):
        if QtWidgets is None:
            raise RuntimeError("PySide6 is not installed; DebugMonitorWindow unavailable")
        self.shared_state = shared_state
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("PokerVision Debug Monitor")
        self.window.resize(1180, 800)

        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.image_label = QtWidgets.QLabel("Waiting for frames")
        self.image_label.setMinimumSize(940, 520)
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

    def _build_info_payload(self, render_state: Optional[dict], status: dict) -> dict:
        if not render_state:
            return {"status": status}

        analysis_panel = render_state.get("analysis_panel", {}) if isinstance(render_state, dict) else {}
        payload = {
            "frame_status": status,
            "hand_id": render_state.get("hand_id"),
            "street": render_state.get("street"),
            "recommended_action": render_state.get("recommended_action"),
            "recommended_amount_to": render_state.get("recommended_amount_to"),
            "recommended_size_pct": render_state.get("recommended_size_pct"),
            "node_type": render_state.get("node_type"),
            "engine_status": render_state.get("engine_status"),
            "solver_status": render_state.get("solver_status"),
            "warnings": render_state.get("warnings"),
            "solver_warnings": render_state.get("solver_warnings"),
            "solver_errors": render_state.get("solver_errors"),
            "analysis_panel": analysis_panel,
        }
        return payload

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

        payload = self._build_info_payload(render_state, status)
        self.info.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False))
