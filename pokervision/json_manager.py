from __future__ import annotations

import json
from pathlib import Path

from .models import HandState


def save_hand_json(path: Path, hand: HandState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(hand.to_dict(), f, ensure_ascii=False, indent=2)
