from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import hypot
from typing import Optional

from .models import FrameAnalysis, HandError, HandState, utc_now_iso


MATCH_STRONG = "strong_match"
MATCH_WEAK = "weak_match"
MATCH_WEAK_CONFLICT = "weak_conflict"
MATCH_NONE = "no_match"


@dataclass(slots=True)
class MatchDecision:
    status: str
    reason: str


class HandStateManager:
    def __init__(self, schema_version: str, stale_timeout_sec: float, close_timeout_sec: float, table_center_max_shift_px: float = 120.0):
        self.schema_version = schema_version
        self.stale_timeout_sec = stale_timeout_sec
        self.close_timeout_sec = close_timeout_sec
        self.table_center_max_shift_px = table_center_max_shift_px
        self.active_hand: Optional[HandState] = None
        self.hand_counter = 0

    def _next_hand_id(self) -> str:
        self.hand_counter += 1
        return f"hand_{self.hand_counter:06d}"

    def _parse_timestamp(self, value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    def _seconds_between(self, older: str, newer: str) -> float:
        return max(0.0, (self._parse_timestamp(newer) - self._parse_timestamp(older)).total_seconds())

    def _sync_snapshot_status(self, hand: HandState) -> None:
        if not hand.render_state_snapshot:
            return
        freshness = "live"
        warnings = list(hand.render_state_snapshot.get("warnings", []))
        warnings = [warning for warning in warnings if warning not in {"State is stale", "State is closed"}]
        if hand.status == "stale":
            freshness = "stale"
            warnings.append("State is stale")
        elif hand.status == "closed":
            freshness = "closed"
            warnings.append("State is closed")
        elif hand.status == "error":
            freshness = "error"
            warnings.append("State is error")
        hand.render_state_snapshot["status"] = "ok" if hand.status in {"active", "stale"} else hand.status
        hand.render_state_snapshot["freshness"] = freshness
        hand.render_state_snapshot["warnings"] = warnings
        hand.render_state_snapshot["updated_at"] = hand.updated_at

    def _normalized_hero_cards(self, cards: list[str]) -> tuple[str, ...]:
        return tuple(sorted(str(card) for card in cards))

    def create_hand(self, analysis: FrameAnalysis) -> HandState:
        now = analysis.timestamp
        hand = HandState(
            schema_version=self.schema_version,
            hand_id=self._next_hand_id(),
            status="active",
            player_count=int(analysis.player_count or 0),
            table_format=str(analysis.table_format or "unknown"),
            created_at=now,
            updated_at=now,
            last_seen_at=now,
            hero_position=str(analysis.hero_position or ""),
            hero_cards=list(analysis.hero_cards),
            occupied_positions=list(analysis.occupied_positions),
            street_state={"current_street": analysis.street, "street_history": [analysis.street]},
            positions=dict(analysis.positions),
            board_cards=list(analysis.board_cards),
            player_states={position: dict(payload) for position, payload in analysis.player_states.items()},
            frames_log=[],
            errors=[],
            artifacts={},
            processing_summary={
                "frames_seen_for_this_hand": 0,
                "successful_frames": 0,
                "failed_frames": 0,
                "hand_closed": False,
            },
            table_center=analysis.table_center,
            table_amount_state=dict(analysis.table_amount_state),
            action_state=dict(analysis.action_inference),
            actions_log=list(analysis.action_inference.get("actions_this_frame", [])),
        )
        self._append_frame_log(hand, analysis, matched_existing=False, processing_status="ok")
        hand.processing_summary["frames_seen_for_this_hand"] += 1
        hand.processing_summary["successful_frames"] += 1
        self.active_hand = hand
        return hand

    def _player_states_changed(self, hand: HandState, analysis: FrameAnalysis) -> bool:
        if set(hand.player_states) != set(analysis.player_states):
            return True
        for position, payload in analysis.player_states.items():
            existing = hand.player_states.get(position, {})
            for key in ("is_fold", "is_all_in", "stack_text_raw", "stack_bb"):
                if existing.get(key) != payload.get(key):
                    return True
        return False

    def _table_center_shift(self, hand: HandState, analysis: FrameAnalysis) -> float:
        if hand.table_center is None or analysis.table_center is None:
            return 0.0
        return hypot(hand.table_center[0] - analysis.table_center[0], hand.table_center[1] - analysis.table_center[1])

    def _position_geometry_conflict(self, hand: HandState, analysis: FrameAnalysis) -> bool:
        if not hand.positions or not analysis.positions:
            return False
        for position in set(hand.positions).intersection(analysis.positions):
            old_center = hand.positions[position].get("center", {})
            new_center = analysis.positions[position].get("center", {})
            if not old_center or not new_center:
                continue
            shift = hypot(old_center.get("x", 0.0) - new_center.get("x", 0.0), old_center.get("y", 0.0) - new_center.get("y", 0.0))
            if shift > self.table_center_max_shift_px:
                return True
        return False

    def compare(self, hand: HandState, analysis: FrameAnalysis) -> MatchDecision:
        if self._normalized_hero_cards(hand.hero_cards) != self._normalized_hero_cards(analysis.hero_cards):
            return MatchDecision(MATCH_NONE, "hero_cards changed")

        conflict_reasons: list[str] = []

        if hand.player_count != analysis.player_count:
            conflict_reasons.append("player_count changed")
        if hand.table_format != analysis.table_format:
            conflict_reasons.append("table_format changed")
        if hand.hero_position != analysis.hero_position:
            conflict_reasons.append("hero_position changed")
        if hand.occupied_positions != analysis.occupied_positions:
            conflict_reasons.append("occupied_positions changed")

        center_shift = self._table_center_shift(hand, analysis)
        if center_shift > self.table_center_max_shift_px:
            conflict_reasons.append(f"table center shifted by {center_shift:.1f}px")
        if self._position_geometry_conflict(hand, analysis):
            conflict_reasons.append("seat geometry changed")

        current_street = hand.street_state.get("current_street", "preflop")
        if not self._street_transition_allowed(current_street, analysis.street):
            conflict_reasons.append(f"street transition {current_street}->{analysis.street} invalid")

        if conflict_reasons:
            return MatchDecision(MATCH_WEAK_CONFLICT, "; ".join(conflict_reasons))

        if self._player_states_changed(hand, analysis):
            return MatchDecision(MATCH_WEAK, "player states changed")
        if current_street == analysis.street:
            return MatchDecision(MATCH_STRONG, "same hero cards confirmed")
        return MatchDecision(MATCH_STRONG, "same hero cards with valid street transition")

    def _street_transition_allowed(self, old: str, new: str) -> bool:
        rank = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
        if old == new:
            return True
        return rank.get(new, -1) == rank.get(old, -1) + 1

    def update_or_create(self, analysis: FrameAnalysis) -> tuple[HandState, MatchDecision, bool]:
        if self.active_hand is None or self.active_hand.status == "closed":
            return self.create_hand(analysis), MatchDecision(MATCH_STRONG, "created new hand"), True
        decision = self.compare(self.active_hand, analysis)
        if decision.status in {MATCH_STRONG, MATCH_WEAK}:
            self._update_hand(self.active_hand, analysis)
            return self.active_hand, decision, False
        if decision.status == MATCH_WEAK_CONFLICT:
            self.active_hand.status = "stale"
            self.active_hand.conflict_state = decision.reason
            self._append_frame_log(self.active_hand, analysis, matched_existing=True, processing_status="warning")
            self._sync_snapshot_status(self.active_hand)
            return self.active_hand, decision, False
        self.active_hand.status = "closed"
        self.active_hand.processing_summary["hand_closed"] = True
        self._sync_snapshot_status(self.active_hand)
        new_hand = self.create_hand(analysis)
        return new_hand, decision, True

    def _update_hand(self, hand: HandState, analysis: FrameAnalysis) -> None:
        hand.status = "active"
        hand.conflict_state = None
        hand.updated_at = analysis.timestamp
        hand.last_seen_at = analysis.timestamp
        hand.positions = dict(analysis.positions)
        hand.table_center = analysis.table_center
        hand.table_amount_state = dict(analysis.table_amount_state)
        hand.action_state = dict(analysis.action_inference)
        hand.actions_log.extend(list(analysis.action_inference.get("actions_this_frame", [])))
        hand.board_cards = list(analysis.board_cards) if analysis.board_cards else hand.board_cards
        hand.player_states = {position: dict(payload) for position, payload in analysis.player_states.items()}
        if analysis.street != hand.street_state.get("current_street"):
            hand.street_state["current_street"] = analysis.street
            hand.street_state.setdefault("street_history", []).append(analysis.street)
        self._append_frame_log(hand, analysis, matched_existing=True, processing_status="ok")
        hand.processing_summary["frames_seen_for_this_hand"] += 1
        hand.processing_summary["successful_frames"] += 1
        self._sync_snapshot_status(hand)

    def register_error(self, hand: Optional[HandState], stage: str, message: str, frame_id: Optional[str], fatal_for_frame: bool = False) -> None:
        if hand is None:
            return
        hand.errors.append(HandError(utc_now_iso(), stage, message, frame_id, fatal_for_frame).to_dict())
        hand.processing_summary["failed_frames"] += 1
        if frame_id is not None:
            hand.frames_log.append({
                "frame_id": frame_id,
                "timestamp": utc_now_iso(),
                "active_hero_found": True,
                "matched_existing_hand": True,
                "assigned_to_hand_id": hand.hand_id,
                "street_detected": hand.street_state.get("current_street", "preflop"),
                "player_state_summary": {},
                "processing_status": "error",
                "error_stage": stage,
                "error_message": message,
            })
        self._sync_snapshot_status(hand)

    def mark_stale_if_needed(self, now_timestamp: str) -> bool:
        if self.active_hand is None:
            return False
        hand = self.active_hand
        age_sec = self._seconds_between(hand.last_seen_at, now_timestamp)
        previous_status = hand.status
        if age_sec >= self.close_timeout_sec and hand.status != "closed":
            hand.status = "closed"
            hand.updated_at = now_timestamp
            hand.processing_summary["hand_closed"] = True
        elif age_sec >= self.stale_timeout_sec and hand.status == "active":
            hand.status = "stale"
            hand.updated_at = now_timestamp
        changed = hand.status != previous_status
        if changed:
            self._sync_snapshot_status(hand)
        return changed

    def _append_frame_log(self, hand: HandState, analysis: FrameAnalysis, matched_existing: bool, processing_status: str) -> None:
        hand.frames_log.append({
            "frame_id": analysis.frame_id,
            "timestamp": analysis.timestamp,
            "active_hero_found": analysis.active_hero_found,
            "matched_existing_hand": matched_existing,
            "assigned_to_hand_id": hand.hand_id,
            "street_detected": analysis.street,
            "player_state_summary": {
                position: {
                    "is_fold": payload.get("is_fold", False),
                    "is_all_in": payload.get("is_all_in", False),
                    "stack_bb": payload.get("stack_bb"),
                }
                for position, payload in analysis.player_states.items()
            },
            "table_amount_summary": dict(analysis.table_amount_state),
            "action_summary": list(analysis.action_inference.get("actions_this_frame", [])),
            "processing_status": processing_status,
        })
