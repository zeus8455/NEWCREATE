from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import hypot
from typing import Optional, Iterable

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
    def __init__(
        self,
        schema_version: str,
        stale_timeout_sec: float,
        close_timeout_sec: float,
        table_center_max_shift_px: float = 120.0,
    ):
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
        hand.render_state_snapshot["status"] = "ok" if hand.status in {"active", "stale", "closed"} else hand.status
        hand.render_state_snapshot["freshness"] = freshness
        hand.render_state_snapshot["warnings"] = warnings
        hand.render_state_snapshot["updated_at"] = hand.updated_at

    def _normalized_hero_cards(self, cards: list[str]) -> tuple[str, ...]:
        return tuple(sorted(str(card) for card in cards))

    def _action_signature(self, action: dict) -> tuple:
        payload = dict(action or {})
        return (
            str(payload.get("position") or ""),
            str(payload.get("street") or ""),
            str(payload.get("action") or ""),
            str(payload.get("semantic_action") or ""),
            str(payload.get("engine_action") or ""),
            payload.get("amount_bb"),
            payload.get("final_contribution_bb"),
            payload.get("current_price_to_call_before"),
            payload.get("current_price_to_call_after"),
            payload.get("raise_level_before_action"),
            payload.get("raise_level_after_action"),
            str(payload.get("opener_pos") or ""),
            str(payload.get("three_bettor_pos") or ""),
            str(payload.get("four_bettor_pos") or ""),
            payload.get("limpers"),
            payload.get("callers_after_open"),
            str(payload.get("call_vs") or ""),
            str(payload.get("spot_family") or ""),
            str(payload.get("open_family") or ""),
            str(payload.get("limp_family") or ""),
            bool(payload.get("legacy_from_forced_blind", False)),
        )

    def _replace_street_actions(self, actions: Iterable[dict], street: str, replacement: Iterable[dict]) -> list[dict]:
        normalized_street = str(street or "").lower()
        preserved = [
            dict(item)
            for item in (actions or [])
            if str((item or {}).get("street", "")).lower() != normalized_street
        ]
        return [*preserved, *self._dedupe_actions(replacement)]

    def _dedupe_actions(self, actions: Iterable[dict]) -> list[dict]:
        seen: set[tuple] = set()
        deduped: list[dict] = []
        for action in actions or []:
            sig = self._action_signature(action)
            if sig in seen:
                continue
            seen.add(sig)
            deduped.append(dict(action))
        return deduped

    def _safe_float(self, value: object, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _format_amount_for_display(self, amount: object) -> str:
        value = self._safe_float(amount, 0.0)
        if abs(value - round(value)) <= 1e-9:
            return f"{value:.1f}"
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        if "." not in text:
            text += ".0"
        return text

    def _max_raise_level_from_actions(self, actions: Iterable[dict]) -> int:
        max_level = 0
        for action in actions or []:
            try:
                max_level = max(max_level, int((action or {}).get("raise_level_after_action") or 0))
            except (TypeError, ValueError):
                continue
        return max_level

    def _call_vs_for_raise_level(self, raise_level: int) -> str:
        if raise_level <= 1:
            return "open_raise"
        if raise_level == 2:
            return "3bet"
        if raise_level == 3:
            return "4bet"
        return "5bet_or_more"

    def _infer_missing_postflop_preflop_closing_calls(
        self,
        reconstructed_preflop: dict,
        incoming_state: dict,
        analysis: Optional[FrameAnalysis],
    ) -> None:
        """Close the immutable preflop ledger when postflop proves who continued.

        Vision can miss the final preflop call frame.  If a hand later reaches
        flop/turn/river, every non-folded/non-all-in player that is still active
        must have paid the final preflop price.  This repair adds an inferred
        preflop CALL to the canonical reconstructed_preflop payload only; it
        never reads postflop bet amounts and therefore cannot contaminate the
        preflop ledger with flop/turn/river chips.
        """
        if not isinstance(reconstructed_preflop, dict) or not reconstructed_preflop:
            return
        if analysis is None:
            return
        street = str(getattr(analysis, "street", "") or incoming_state.get("street") or "").lower()
        if street not in {"flop", "turn", "river"}:
            return

        action_history = [dict(item) for item in list(
            reconstructed_preflop.get("action_history_resolved")
            or reconstructed_preflop.get("action_history")
            or []
        ) if isinstance(item, dict)]
        if not action_history:
            return

        aggressive_actions = {"open_raise", "iso_raise", "3bet", "4bet", "cold_4bet", "5bet_jam"}
        if not any(str(action.get("semantic_action") or "") in aggressive_actions for action in action_history):
            return

        current_price = self._safe_float(reconstructed_preflop.get("current_price_to_call"), 0.0)
        if current_price <= 0.0:
            for action in action_history:
                semantic = str(action.get("semantic_action") or "")
                if semantic in aggressive_actions or semantic == "call":
                    current_price = max(
                        current_price,
                        self._safe_float(action.get("final_contribution_bb"), 0.0),
                        self._safe_float(action.get("amount_bb"), 0.0),
                    )
        if current_price <= 0.0:
            return

        commitments = {
            str(pos): self._safe_float(amount, 0.0)
            for pos, amount in dict(
                reconstructed_preflop.get("resolved_commitments_by_pos")
                or reconstructed_preflop.get("final_contribution_bb_by_pos")
                or {}
            ).items()
        }
        street_commitments = {
            str(pos): self._safe_float(amount, 0.0)
            for pos, amount in dict(
                reconstructed_preflop.get("final_contribution_street_bb_by_pos")
                or commitments
                or {}
            ).items()
        }
        if not commitments:
            return

        player_states = dict(getattr(analysis, "player_states", {}) or {})
        occupied_positions = [str(pos) for pos in list(getattr(analysis, "occupied_positions", []) or [])]
        actor_order = [str(pos) for pos in list(reconstructed_preflop.get("actor_order") or occupied_positions)]
        if not actor_order:
            actor_order = occupied_positions

        existing_call_keys = {
            (
                str(action.get("position") or action.get("pos") or ""),
                round(self._safe_float(action.get("final_contribution_bb") or action.get("amount_bb"), 0.0), 4),
                str(action.get("semantic_action") or ""),
            )
            for action in action_history
        }

        raise_level = self._max_raise_level_from_actions(action_history)
        call_vs = self._call_vs_for_raise_level(raise_level)
        frame_id = getattr(analysis, "frame_id", None)
        timestamp = getattr(analysis, "timestamp", None)
        added_calls: list[dict] = []

        for position in actor_order:
            state = dict(player_states.get(position, {}) or {})
            if bool(state.get("is_fold", False)):
                continue
            if bool(state.get("is_all_in", False)):
                continue
            if not bool(state.get("is_active", True)):
                continue
            committed = self._safe_float(commitments.get(position), 0.0)
            if committed >= current_price - 1e-9:
                continue
            key = (position, round(current_price, 4), "call")
            if key in existing_call_keys:
                commitments[position] = round(current_price, 4)
                street_commitments[position] = round(current_price, 4)
                continue

            event = {
                "order": len(action_history) + len(added_calls) + 1,
                "position": position,
                "pos": position,
                "street": "preflop",
                "final_contribution_bb": round(current_price, 4),
                "amount_bb": round(current_price, 4),
                "semantic_action": "call",
                "engine_action": "call",
                "raise_level_after_action": raise_level,
                "current_price_to_call_after_action": round(current_price, 4),
                "call_vs": call_vs,
                "inferred_postflop_closing_call": True,
                "reason": "player_reached_postflop_with_preflop_commitment_below_final_price",
                "frame_id": frame_id,
                "timestamp": timestamp,
                "action": "CALL",
                "action_display": f"CALL {self._format_amount_for_display(current_price)}",
            }
            added_calls.append(event)
            existing_call_keys.add(key)
            commitments[position] = round(current_price, 4)
            street_commitments[position] = round(current_price, 4)

        if not added_calls:
            return

        merged_history = self._dedupe_actions([*action_history, *added_calls])
        for index, action in enumerate(merged_history, start=1):
            action["order"] = index

        reconstructed_preflop["action_history"] = [dict(item) for item in merged_history]
        reconstructed_preflop["action_history_resolved"] = [dict(item) for item in merged_history]
        reconstructed_preflop["final_contribution_bb_by_pos"] = dict(commitments)
        reconstructed_preflop["final_contribution_street_bb_by_pos"] = dict(street_commitments)
        reconstructed_preflop["resolved_commitments_by_pos"] = dict(commitments)
        reconstructed_preflop["current_price_to_call"] = round(current_price, 4)
        reconstructed_preflop["callers_after_open"] = sum(
            1 for action in merged_history
            if str(action.get("semantic_action") or "") == "call"
            and str(action.get("call_vs") or "") == "open_raise"
        )
        reconstructed_preflop["callers"] = reconstructed_preflop["callers_after_open"]
        reconstructed_preflop["postflop_closing_call_repair_applied"] = True
        notes = list(reconstructed_preflop.get("reconciliation_notes") or [])
        note = "postflop_active_players_closed_missing_preflop_calls"
        if note not in notes:
            notes.append(note)
        reconstructed_preflop["reconciliation_notes"] = notes
        skipped = [str(pos) for pos in list(reconstructed_preflop.get("skipped_positions") or [])]
        reconstructed_preflop["skipped_positions"] = [pos for pos in skipped if pos not in {str(action.get("position")) for action in added_calls}]

        hero_preview = reconstructed_preflop.get("hero_context_preview")
        if isinstance(hero_preview, dict):
            hero_preview["callers"] = reconstructed_preflop["callers"]
            hero_preview["postflop_closing_call_repair_applied"] = True

        projection = reconstructed_preflop.get("preflop_projection")
        if isinstance(projection, dict):
            projection["action_history"] = [dict(item) for item in merged_history]
            projection["action_history_resolved"] = [dict(item) for item in merged_history]
            projection["final_contribution_bb_by_pos"] = dict(commitments)
            projection["final_contribution_street_bb_by_pos"] = dict(street_commitments)
            projection["callers"] = reconstructed_preflop["callers"]
            projection["reconciliation_notes"] = list(notes)
            projection["postflop_closing_call_repair_applied"] = True
            projection_skipped = [str(pos) for pos in list(projection.get("positions_closed_to_action") or [])]
            projection["positions_closed_to_action"] = [pos for pos in projection_skipped if pos not in {str(action.get("position")) for action in added_calls}]
            projection_hero = projection.get("hero_context_preview")
            if isinstance(projection_hero, dict):
                projection_hero["callers"] = reconstructed_preflop["callers"]
                projection_hero["postflop_closing_call_repair_applied"] = True

        incoming_state["reconstructed_preflop"] = reconstructed_preflop
        if isinstance(projection, dict):
            incoming_state["preflop_projection"] = dict(projection)

    def _carry_forward_preflop_state(self, incoming_state: dict, existing_state: dict, analysis: Optional[FrameAnalysis] = None) -> None:
        """Keep the canonical preflop ledger alive across postflop frames.

        Postflop action inference legitimately stores current-street commitments
        such as a river bet in ``final_contribution_bb_by_pos``.  Those fields
        must never be used as a replacement for the resolved preflop ledger.
        When the hand advances flop/turn/river, carry the already-resolved
        preflop payload forward inside action_state so render/solver layers read
        the immutable preflop ledger instead of falling back to postflop money.
        """
        if not isinstance(incoming_state, dict) or not isinstance(existing_state, dict):
            return

        incoming_street = str(incoming_state.get("street") or "").lower()
        if incoming_street == "preflop":
            return

        reconstructed = incoming_state.get("reconstructed_preflop")
        if not isinstance(reconstructed, dict) or not reconstructed:
            previous_reconstructed = existing_state.get("reconstructed_preflop")
            if isinstance(previous_reconstructed, dict) and previous_reconstructed:
                incoming_state["reconstructed_preflop"] = dict(previous_reconstructed)
            elif str(existing_state.get("street") or "").lower() == "preflop":
                # Legacy fallback: older preflop states can themselves be the
                # resolved preflop payload. Only allow this when the previous
                # state explicitly belongs to preflop; never synthesize it from
                # a flop/turn/river amount state.
                incoming_state["reconstructed_preflop"] = {
                    key: value
                    for key, value in dict(existing_state).items()
                    if key
                    in {
                        "street",
                        "source_mode",
                        "hero_position",
                        "node_type",
                        "node_type_preview",
                        "projection_node_type",
                        "advisor_node_type",
                        "advisor_mapping_reason",
                        "opener_pos",
                        "three_bettor_pos",
                        "four_bettor_pos",
                        "limpers",
                        "limpers_count",
                        "callers",
                        "callers_after_open",
                        "action_history",
                        "action_history_resolved",
                        "actor_order",
                        "current_price_to_call",
                        "last_aggressor_position",
                        "final_contribution_bb_by_pos",
                        "final_contribution_street_bb_by_pos",
                        "hero_context_preview",
                        "skipped_positions",
                        "same_hand_identity",
                        "contract_version",
                    }
                }

        previous_projection = existing_state.get("preflop_projection")
        if (not isinstance(incoming_state.get("preflop_projection"), dict) or not incoming_state.get("preflop_projection")) and isinstance(previous_projection, dict) and previous_projection:
            incoming_state["preflop_projection"] = dict(previous_projection)

        reconstructed = incoming_state.get("reconstructed_preflop")
        if isinstance(reconstructed, dict) and reconstructed:
            self._infer_missing_postflop_preflop_closing_calls(reconstructed, incoming_state, analysis)

    def _prepare_action_payloads_for_storage(self, hand: Optional[HandState], analysis: FrameAnalysis) -> tuple[list[dict], list[dict]]:
        incoming_state = dict(getattr(analysis, "action_inference", {}) or {})
        street = str(incoming_state.get("street") or getattr(analysis, "street", "preflop") or "preflop").lower()
        incoming_actions = [dict(item) for item in list(incoming_state.get("actions_this_frame", []) or [])]
        incoming_history = [dict(item) for item in list(incoming_state.get("action_history", []) or [])]
        incoming_resolved = [
            dict(item)
            for item in list(incoming_state.get("action_history_resolved", []) or incoming_history)
        ]

        existing_actions = []
        existing_history = []
        existing_resolved = []
        existing_state = {}
        if hand is not None:
            existing_actions = [dict(item) for item in list(getattr(hand, "actions_log", []) or [])]
            existing_state = dict(getattr(hand, "action_state", {}) or {})
            existing_history = [dict(item) for item in list(existing_state.get("action_history", []) or [])]
            existing_resolved = [
                dict(item)
                for item in list(existing_state.get("action_history_resolved", []) or existing_history)
            ]

        if street == "preflop":
            merged_history = self._dedupe_actions(incoming_resolved)
            existing_signatures = {self._action_signature(item) for item in existing_resolved}
            novel_actions: list[dict] = []
            for action in merged_history:
                sig = self._action_signature(action)
                if sig in existing_signatures:
                    continue
                existing_signatures.add(sig)
                novel_actions.append(dict(action))

            analysis.action_inference["actions_this_frame"] = [dict(item) for item in novel_actions]
            analysis.action_inference["action_history"] = [dict(item) for item in merged_history]
            analysis.action_inference["action_history_resolved"] = [dict(item) for item in merged_history]

            reconstructed = dict(incoming_state.get("reconstructed_preflop") or {})
            reconstructed["action_history"] = [dict(item) for item in merged_history]
            reconstructed["action_history_resolved"] = [dict(item) for item in merged_history]
            if "hero_context_preview" not in reconstructed and incoming_state.get("hero_context_preview") is not None:
                reconstructed["hero_context_preview"] = dict(incoming_state.get("hero_context_preview") or {})
            if "node_type" not in reconstructed and incoming_state.get("node_type_preview") is not None:
                reconstructed["node_type"] = incoming_state.get("node_type_preview")
            analysis.action_inference["reconstructed_preflop"] = reconstructed
            return novel_actions, merged_history

        existing_signatures = {self._action_signature(item) for item in [*existing_actions, *existing_history]}
        novel_actions: list[dict] = []
        for action in incoming_actions:
            sig = self._action_signature(action)
            if sig in existing_signatures:
                continue
            existing_signatures.add(sig)
            novel_actions.append(dict(action))

        merged_history = self._dedupe_actions([*existing_history, *incoming_history, *novel_actions])
        analysis.action_inference["actions_this_frame"] = [dict(item) for item in novel_actions]
        analysis.action_inference["action_history"] = [dict(item) for item in merged_history]
        self._carry_forward_preflop_state(analysis.action_inference, existing_state, analysis)
        return novel_actions, merged_history

    def create_hand(self, analysis: FrameAnalysis) -> HandState:
        now = analysis.timestamp
        novel_actions, merged_history = self._prepare_action_payloads_for_storage(None, analysis)
        analysis.action_inference["action_history"] = [dict(item) for item in merged_history]
        analysis.action_inference["actions_this_frame"] = [dict(item) for item in novel_actions]
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
                "solver_runs": 0,
                "solver_reuse_hits": 0,
            },
            table_center=analysis.table_center,
            table_amount_state=dict(analysis.table_amount_state),
            amount_normalization=dict(analysis.amount_normalization),
            action_state=dict(analysis.action_inference),
            actions_log=self._replace_street_actions([], analysis.street, merged_history if str(analysis.street or "").lower() == "preflop" else novel_actions),
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
            shift = hypot(
                old_center.get("x", 0.0) - new_center.get("x", 0.0),
                old_center.get("y", 0.0) - new_center.get("y", 0.0),
            )
            if shift > self.table_center_max_shift_px:
                return True
        return False

    def compare(self, hand: HandState, analysis: FrameAnalysis) -> MatchDecision:
        # Единственный критерий принадлежности к раздаче: карты HERO.
        if self._normalized_hero_cards(hand.hero_cards) != self._normalized_hero_cards(analysis.hero_cards):
            return MatchDecision(MATCH_NONE, "hero_cards changed")

        # Остальные различия не должны открывать новую раздачу.
        notes: list[str] = []
        if hand.player_count != analysis.player_count:
            notes.append("player_count changed")
        if hand.table_format != analysis.table_format:
            notes.append("table_format changed")
        if hand.hero_position != analysis.hero_position:
            notes.append("hero_position changed")
        if hand.occupied_positions != analysis.occupied_positions:
            notes.append("occupied_positions changed")
        center_shift = self._table_center_shift(hand, analysis)
        if center_shift > self.table_center_max_shift_px:
            notes.append(f"table center shifted by {center_shift:.1f}px")
        if self._position_geometry_conflict(hand, analysis):
            notes.append("seat geometry changed")
        current_street = hand.street_state.get("current_street", "preflop")
        if not self._street_transition_allowed(current_street, analysis.street):
            notes.append(f"street transition {current_street}->{analysis.street} invalid")
        if self._player_states_changed(hand, analysis):
            notes.append("player states changed")
        if notes:
            return MatchDecision(MATCH_WEAK_CONFLICT, "; ".join(notes))
        return MatchDecision(MATCH_STRONG, "same hero cards confirmed")

    def _street_transition_allowed(self, old: str, new: str) -> bool:
        rank = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
        if old == new:
            return True
        return rank.get(new, -1) >= rank.get(old, -1)

    def update_or_create(self, analysis: FrameAnalysis) -> tuple[HandState, MatchDecision, bool]:
        if self.active_hand is None:
            return self.create_hand(analysis), MatchDecision(MATCH_STRONG, "created new hand"), True

        # Даже если hand stale/closed, сначала обязаны сравнить HERO cards.
        decision = self.compare(self.active_hand, analysis)
        if decision.status in {MATCH_STRONG, MATCH_WEAK, MATCH_WEAK_CONFLICT}:
            self.active_hand.conflict_state = decision.reason if decision.status == MATCH_WEAK_CONFLICT else None
            self._update_hand(self.active_hand, analysis)
            self.active_hand.conflict_state = decision.reason if decision.status == MATCH_WEAK_CONFLICT else None
            self._sync_snapshot_status(self.active_hand)
            return self.active_hand, decision, False

        self.active_hand.status = "closed"
        self.active_hand.processing_summary["hand_closed"] = True
        self._sync_snapshot_status(self.active_hand)
        new_hand = self.create_hand(analysis)
        return new_hand, decision, True

    def _update_hand(self, hand: HandState, analysis: FrameAnalysis) -> None:
        hand.status = "active"
        novel_actions, merged_history = self._prepare_action_payloads_for_storage(hand, analysis)
        analysis.action_inference["action_history"] = [dict(item) for item in merged_history]
        analysis.action_inference["actions_this_frame"] = [dict(item) for item in novel_actions]
        hand.conflict_state = None
        hand.updated_at = analysis.timestamp
        hand.last_seen_at = analysis.timestamp
        hand.player_count = int(analysis.player_count or hand.player_count)
        hand.table_format = str(analysis.table_format or hand.table_format)
        hand.hero_position = str(analysis.hero_position or hand.hero_position)
        hand.occupied_positions = list(analysis.occupied_positions)
        hand.positions = dict(analysis.positions)
        hand.table_center = analysis.table_center
        hand.table_amount_state = dict(analysis.table_amount_state)
        hand.amount_normalization = dict(analysis.amount_normalization)
        hand.action_state = dict(analysis.action_inference)
        if str(analysis.street or "").lower() == "preflop":
            hand.actions_log = self._replace_street_actions(hand.actions_log, "preflop", merged_history)
        else:
            # Postflop frames can repair the canonical preflop ledger after it is
            # proven that a player reached postflop.  Keep actions_log in sync
            # with that canonical ledger, otherwise UI/action_annotations may show
            # only the old OPEN while reconstructed_preflop correctly has
            # OPEN -> CALL.
            reconstructed_preflop = analysis.action_inference.get("reconstructed_preflop")
            if isinstance(reconstructed_preflop, dict) and reconstructed_preflop:
                canonical_preflop_history = list(
                    reconstructed_preflop.get("action_history_resolved")
                    or reconstructed_preflop.get("action_history")
                    or []
                )
            else:
                canonical_preflop_history = []

            if canonical_preflop_history:
                synced_actions = self._replace_street_actions(hand.actions_log, "preflop", canonical_preflop_history)
            else:
                synced_actions = list(hand.actions_log)
            hand.actions_log = self._dedupe_actions([*synced_actions, *list(novel_actions)])
        hand.board_cards = list(analysis.board_cards) if analysis.board_cards else hand.board_cards
        hand.player_states = {position: dict(payload) for position, payload in analysis.player_states.items()}
        if analysis.street != hand.street_state.get("current_street"):
            hand.street_state["current_street"] = analysis.street
            history = hand.street_state.setdefault("street_history", [])
            if analysis.street not in history:
                history.append(analysis.street)
        self._append_frame_log(hand, analysis, matched_existing=True, processing_status="ok")
        hand.processing_summary["frames_seen_for_this_hand"] += 1
        hand.processing_summary["successful_frames"] += 1
        self._sync_snapshot_status(hand)

    def register_error(
        self,
        hand: Optional[HandState],
        stage: str,
        message: str,
        frame_id: Optional[str],
        fatal_for_frame: bool = False,
    ) -> None:
        if hand is None:
            return
        hand.errors.append(HandError(utc_now_iso(), stage, message, frame_id, fatal_for_frame).to_dict())
        hand.processing_summary["failed_frames"] += 1
        if frame_id is not None:
            hand.frames_log.append(
                {
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
                }
            )
        self._sync_snapshot_status(hand)

    def mark_stale_if_needed(self, now_timestamp: str) -> bool:
        if self.active_hand is None:
            return False
        hand = self.active_hand
        age_sec = self._seconds_between(hand.last_seen_at, now_timestamp)
        previous_status = hand.status

        # Не закрываем hand автоматически: same HERO cards должны позволять продолжить ту же раздачу.
        if age_sec >= self.stale_timeout_sec and hand.status == "active":
            hand.status = "stale"
            hand.updated_at = now_timestamp

        changed = hand.status != previous_status
        if changed:
            self._sync_snapshot_status(hand)
        return changed

    def _append_frame_log(
        self,
        hand: HandState,
        analysis: FrameAnalysis,
        matched_existing: bool,
        processing_status: str,
    ) -> None:
        hand.frames_log.append(
            {
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
                "amount_normalization_summary": dict(analysis.amount_normalization),
                "action_summary": list(analysis.action_inference.get("actions_this_frame", [])),
                "processing_status": processing_status,
            }
        )
