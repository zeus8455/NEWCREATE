from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from .models import CapturedFrame

try:
    import mss  # type: ignore
except Exception:  # pragma: no cover
    mss = None


class FrameSource:
    def next_frame(self) -> CapturedFrame:
        raise NotImplementedError


@dataclass(slots=True)
class MockFrameSource(FrameSource):
    width: int = 1280
    height: int = 720
    counter: int = 0

    def next_frame(self) -> CapturedFrame:
        self.counter += 1
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(image, f"PokerVision Mock Frame {self.counter}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
        cv2.ellipse(image, (self.width // 2, self.height // 2), (350, 220), 0, 0, 360, (30, 90, 30), 4)
        return CapturedFrame(
            frame_id=f"frame_{self.counter:04d}",
            timestamp=datetime.utcnow().isoformat(timespec="milliseconds"),
            image=image,
        )


class ScreenFrameSource(FrameSource):
    def __init__(self, monitor_index: int = 1) -> None:
        if mss is None:
            raise RuntimeError("mss is not installed; real screen capture is unavailable")
        self.counter = 0
        self.monitor_index = max(1, int(monitor_index))

    def next_frame(self) -> CapturedFrame:
        self.counter += 1
        with mss.mss() as sct:  # pragma: no cover
            monitors = sct.monitors
            idx = self.monitor_index if self.monitor_index < len(monitors) else 1
            shot = sct.grab(monitors[idx])
            image = np.array(shot)[:, :, :3]
        return CapturedFrame(
            frame_id=f"frame_{self.counter:04d}",
            timestamp=datetime.utcnow().isoformat(timespec="milliseconds"),
            image=image,
        )
