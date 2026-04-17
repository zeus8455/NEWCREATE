from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_hero_cards(cards: list[str] | tuple[str, ...] | None) -> list[str]:
    if not cards:
        return []
    normalized = [str(card).strip() for card in cards if str(card).strip()]
    if len(normalized) != 2:
        return normalized
    return sorted(normalized)


@dataclass(slots=True)
class HandMatch:
    hand_id: str
    hand_path: Path
    matched_existing: bool
    hand_payload: dict[str, Any]


class JsonStore:
    def __init__(self, *, base_dir: Path, state_json_path: Path, last_frame_json_path: Path) -> None:
        self.base_dir = Path(base_dir)
        self.state_json_path = Path(state_json_path)
        self.last_frame_json_path = Path(last_frame_json_path)
        self.hands_dir = self.base_dir / "hands"
        self.index_path = self.base_dir / "hands_index.json"

    def ensure_dirs(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.hands_dir.mkdir(parents=True, exist_ok=True)
        self.state_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_frame_json_path.parent.mkdir(parents=True, exist_ok=True)

    def write_idle_state(self, *, frame_id: str, active_hero_found: bool, notes: list[str] | None = None) -> None:
        self.ensure_dirs()
        index = self._load_index()
        payload: dict[str, Any] = {
            "timestamp_utc": utc_now_iso(),
            "frame_id": frame_id,
            "status": "idle",
            "active_hero_found": active_hero_found,
            "notes": list(notes or []),
            "players": [],
            "hero_cards": [],
            "board_cards": [],
            "street": None,
            "current_hand_id": index.get("current_hand_id"),
            "current_hand_status": index.get("current_hand_status"),
            "output_base_dir": str(self.base_dir),
        }
        self._write_json(self.state_json_path, payload)
        self._write_json(self.last_frame_json_path, payload)

    def write_current_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ensure_dirs()
        payload = dict(payload)
        payload.setdefault("timestamp_utc", utc_now_iso())
        payload.setdefault("output_base_dir", str(self.base_dir))

        hero_cards = _normalize_hero_cards(payload.get("hero_cards") or [])
        if payload.get("active_hero_found") and len(hero_cards) == 2:
            match = self._resolve_hand(hero_cards)
            hand_payload = self._merge_into_hand(match, payload, hero_cards)
            self._write_json(match.hand_path, hand_payload)
            self._save_index_for_hand(hand_payload, match.hand_path)

            payload["hand_id"] = hand_payload["hand_id"]
            payload["matched_existing_hand"] = match.matched_existing
            payload["hand_status"] = hand_payload["status"]
            payload["hand_json_path"] = str(match.hand_path)
        else:
            index = self._load_index()
            payload.setdefault("current_hand_id", index.get("current_hand_id"))
            payload.setdefault("current_hand_status", index.get("current_hand_status"))

        self._write_json(self.state_json_path, payload)
        self._write_json(self.last_frame_json_path, payload)
        return payload

    def _resolve_hand(self, hero_cards: list[str]) -> HandMatch:
        index = self._load_index()
        known_hands = index.get("hands", [])

        matching_meta: dict[str, Any] | None = None
        for meta in sorted(known_hands, key=lambda item: str(item.get("last_seen_at") or ""), reverse=True):
            if _normalize_hero_cards(meta.get("hero_cards") or []) == hero_cards:
                matching_meta = meta
                break

        if matching_meta is not None:
            hand_id = str(matching_meta["hand_id"])
            hand_path = Path(str(matching_meta["hand_json_path"]))
            hand_payload = self._read_json(hand_path, default={})
            if not hand_payload:
                hand_payload = self._new_hand_payload(hand_id=hand_id, hero_cards=hero_cards)
            return HandMatch(
                hand_id=hand_id,
                hand_path=hand_path,
                matched_existing=True,
                hand_payload=hand_payload,
            )

        next_seq = int(index.get("next_hand_seq", 1))
        hand_id = f"hand_{next_seq:06d}"
        hand_dir = self.hands_dir / hand_id
        hand_dir.mkdir(parents=True, exist_ok=True)
        hand_path = hand_dir / "hand.json"
        hand_payload = self._new_hand_payload(hand_id=hand_id, hero_cards=hero_cards)
        return HandMatch(
            hand_id=hand_id,
            hand_path=hand_path,
            matched_existing=False,
            hand_payload=hand_payload,
        )

    def _new_hand_payload(self, *, hand_id: str, hero_cards: list[str]) -> dict[str, Any]:
        now = utc_now_iso()
        return {
            "schema_version": "1.0",
            "hand_id": hand_id,
            "status": "active",
            "created_at": now,
            "updated_at": now,
            "last_seen_at": now,
            "hero_cards": list(hero_cards),
            "street_history": [],
            "frames_log": [],
            "last_payload": {},
        }

    def _merge_into_hand(self, match: HandMatch, payload: dict[str, Any], hero_cards: list[str]) -> dict[str, Any]:
        hand = dict(match.hand_payload)
        timestamp = str(payload.get("timestamp_utc") or utc_now_iso())
        street = payload.get("street")
        frame_id = payload.get("frame_id")

        street_history = list(hand.get("street_history") or [])
        if street and street not in street_history:
            street_history.append(str(street))

        frames_log = list(hand.get("frames_log") or [])
        frames_log.append(
            {
                "timestamp_utc": timestamp,
                "frame_id": frame_id,
                "street": street,
                "matched_existing_hand": match.matched_existing,
                "cycle_id": payload.get("cycle_id"),
                "warnings": list(payload.get("warnings") or []),
                "errors": list(payload.get("errors") or []),
                "hero_cards": list(hero_cards),
            }
        )

        players = list(payload.get("players") or [])
        occupied_positions = [
            str(player.get("logical_pos"))
            for player in players
            if isinstance(player, dict) and player.get("logical_pos")
        ]

        hand.update(
            {
                "status": "active",
                "updated_at": timestamp,
                "last_seen_at": timestamp,
                "hero_cards": list(hero_cards),
                "street": street,
                "street_history": street_history,
                "player_count": len(players),
                "occupied_positions": occupied_positions,
                "players": players,
                "board_cards": list(payload.get("board_cards") or []),
                "warnings": list(payload.get("warnings") or []),
                "errors": list(payload.get("errors") or []),
                "last_payload": dict(payload),
                "frames_log": frames_log,
            }
        )
        return hand

    def _save_index_for_hand(self, hand_payload: dict[str, Any], hand_path: Path) -> None:
        index = self._load_index()
        hand_id = str(hand_payload["hand_id"])
        hand_meta = {
            "hand_id": hand_id,
            "hero_cards": list(hand_payload.get("hero_cards") or []),
            "status": str(hand_payload.get("status") or "active"),
            "last_seen_at": str(hand_payload.get("last_seen_at") or utc_now_iso()),
            "hand_json_path": str(hand_path),
        }

        hands = [meta for meta in index.get("hands", []) if str(meta.get("hand_id")) != hand_id]
        hands.append(hand_meta)
        hands.sort(key=lambda item: str(item.get("last_seen_at") or ""))

        current_seq = self._parse_hand_seq(hand_id)
        next_seq = max(int(index.get("next_hand_seq", 1)), current_seq + 1)

        new_index = {
            "current_hand_id": hand_id,
            "current_hand_status": hand_meta["status"],
            "next_hand_seq": next_seq,
            "hands": hands,
            "updated_at": utc_now_iso(),
        }
        self._write_json(self.index_path, new_index)

    def _load_index(self) -> dict[str, Any]:
        index = self._read_json(self.index_path, default=None)
        if isinstance(index, dict):
            index.setdefault("current_hand_id", None)
            index.setdefault("current_hand_status", None)
            index.setdefault("next_hand_seq", 1)
            index.setdefault("hands", [])
            return index

        hands: list[dict[str, Any]] = []
        next_seq = 1
        if self.hands_dir.exists():
            for hand_json_path in sorted(self.hands_dir.glob("hand_*/hand.json")):
                payload = self._read_json(hand_json_path, default={})
                if not isinstance(payload, dict):
                    continue
                hand_id = str(payload.get("hand_id") or hand_json_path.parent.name)
                hands.append(
                    {
                        "hand_id": hand_id,
                        "hero_cards": list(payload.get("hero_cards") or []),
                        "status": str(payload.get("status") or "active"),
                        "last_seen_at": str(payload.get("last_seen_at") or payload.get("updated_at") or ""),
                        "hand_json_path": str(hand_json_path),
                    }
                )
                next_seq = max(next_seq, self._parse_hand_seq(hand_id) + 1)

        current_hand_id = hands[-1]["hand_id"] if hands else None
        current_hand_status = hands[-1]["status"] if hands else None
        return {
            "current_hand_id": current_hand_id,
            "current_hand_status": current_hand_status,
            "next_hand_seq": next_seq,
            "hands": hands,
            "updated_at": utc_now_iso(),
        }

    @staticmethod
    def _parse_hand_seq(hand_id: str) -> int:
        try:
            return int(str(hand_id).split("_")[-1])
        except Exception:
            return 0

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return default

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
