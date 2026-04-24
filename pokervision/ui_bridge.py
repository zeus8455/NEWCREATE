from __future__ import annotations

import threading
from typing import Any, Optional


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[Any] = None
        self._render_state: Optional[dict] = None
        self._status: dict[str, Any] = {}

    def update_frame(self, frame: Any) -> None:
        with self._lock:
            self._frame = frame

    def update_render_state(self, render_state: dict) -> None:
        with self._lock:
            self._render_state = render_state

    def update_status(self, status: dict[str, Any]) -> None:
        with self._lock:
            self._status = dict(status)

    def snapshot(self) -> tuple[Optional[Any], Optional[dict], dict]:
        with self._lock:
            return self._frame, self._render_state, dict(self._status)
