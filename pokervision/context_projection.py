from __future__ import annotations

"""Official projection layer from PokerVision state to solver contexts.

This module owns the canonical conversion from internal PokerVision runtime
state into the external decision-layer contracts:
- PreflopContext
- PostflopContext

The goal is to keep `solver_bridge.py` thin and to ensure preview,
fingerprint, solver input, and persisted hand/debug payloads all come from one
projection source instead of several ad-hoc builders.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

from .preflop_reconstruction import build_preflop_projection

POSTFLOP_STREETS = ("flop", "turn", "river")

ResolveDecisionTypes = Callable[[], tuple[Any, Any, Any]]
CanonicalContextType = Callable[[Any], str]
Serializer = Callable[[Any], Any]


FULL_RUNOUT_LINE = "full_runout_line"
PARTIAL_RUNTIME_LINE = "partial_runtime_line"


def _count_field(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


@dataclass(slots=True)
class ContextProjector:
    """Build canonical decision-layer contexts from a PokerVision hand."""

    bridge: Any
    resolve_decision_types: ResolveDecisionTypes
    canonical_context_type: CanonicalContextType
    serializer: Serializer

    def build_context(self, analysis, hand, street: Optional[str] = None):
        if hand is None:
            return None
        resolved_street = str(
            street
            or getattr(hand, "street_state", {}).get("current_street")
            or getattr(analysis, "street", None)
            or "preflop"
        ).lower()
        if resolved_street == "preflop":
            return self.build_preflop_context(analysis, hand)
        return self.build_postflop_context(analysis, hand, resolved_street)

    def _resolve_preflop_projection(self, hand, action_state: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        resolved_preflop = dict(
            getattr(hand, "reconstructed_preflop", None)
            or action_state.get("reconstructed_preflop")
            or {}
        )
        candidate_projection = (
            getattr(hand, "preflop_projection", None)
            or resolved_preflop.get("preflop_projection")
            or action_state.get("preflop_projection")
            or {}
        )
        projection = dict(candidate_projection or {})
        if not projection or not (
            projection.get("projection_node_type")
            or projection.get("advisor_node_type")
            or projection.get("action_history_resolved")
            or projection.get("action_history")
        ):
            source_state = dict(resolved_preflop or action_state or {})
            source_state.setdefault("hero_position", getattr(hand, "hero_position", None))
            projection = build_preflop_projection(source_state)
        return projection, resolved_preflop

    def build_preflop_context(self, analysis, hand):
        _, _, PreflopContext = self.resolve_decision_types()
        hero_cards = list(getattr(hand, "hero_cards", None) or getattr(analysis, "hero_cards", None) or [])
        if hand is None or len(hero_cards) != 2:
            return None

        action_state = dict(getattr(hand, "action_state", {}) or {})
        projection, resolved_preflop = self._resolve_preflop_projection(hand, action_state)
        projection_node_type = str(
            projection.get("projection_node_type")
            or projection.get("node_type")
            or resolved_preflop.get("projection_node_type")
            or resolved_preflop.get("node_type")
            or ""
        ).strip()
        advisor_node_type = str(projection.get("advisor_node_type") or projection_node_type or "").strip()
        opener_pos = projection.get("opener_pos")
        three_bettor_pos = projection.get("three_bettor_pos")
        four_bettor_pos = projection.get("advisor_four_bettor_pos") or projection.get("four_bettor_pos")
        limpers = projection.get("limpers")
        callers = projection.get("callers")
        action_history = list(
            projection.get("action_history_resolved")
            or projection.get("action_history")
            or resolved_preflop.get("action_history_resolved")
            or resolved_preflop.get("action_history")
            or []
        )
        projection_source = str(
            projection.get("projection_source")
            or ("preflop_projection" if projection else "reconstructed_preflop_resolved")
        )

        fallback_spot = None
        if not advisor_node_type and not projection_node_type:
            fallback_spot = self.bridge._build_hero_preflop_spot(hand)
            projection_node_type = projection_node_type or fallback_spot.node_type
            advisor_node_type = fallback_spot.node_type
            opener_pos = fallback_spot.opener_pos if opener_pos is None else opener_pos
            three_bettor_pos = fallback_spot.three_bettor_pos if three_bettor_pos is None else three_bettor_pos
            four_bettor_pos = fallback_spot.four_bettor_pos if four_bettor_pos is None else four_bettor_pos
            limpers = fallback_spot.limpers if limpers is None else limpers
            callers = fallback_spot.callers if callers is None else callers
            action_history = action_history or self.bridge._street_actions(hand, "preflop")
            projection_source = "replayed_actions_fallback"

        hero_pos = self.bridge._preflop_pos(hand.hero_position, int(hand.player_count))
        return PreflopContext(
            hero_hand=list(hero_cards),
            hero_pos=hero_pos,
            node_type=advisor_node_type or projection_node_type or "unopened",
            opener_pos=opener_pos,
            three_bettor_pos=three_bettor_pos,
            four_bettor_pos=four_bettor_pos,
            limpers=_count_field(limpers),
            callers=_count_field(callers),
            range_owner="hero",
            action_history=action_history,
            meta={
                "source": "pokervision",
                "hand_id": hand.hand_id,
                "hero_original_position": hand.hero_position,
                "projection_source": projection_source,
                "projection_contract_version": projection.get("contract_version"),
                "projection_node_type": projection_node_type or advisor_node_type or "unopened",
                "advisor_node_type": advisor_node_type or projection_node_type or "unopened",
                "advisor_mapping_reason": projection.get("advisor_mapping_reason"),
                "actions_seen": action_history,
            },
        )

    def build_postflop_context(self, analysis, hand, street: Optional[str] = None):
        _, PostflopContext, _ = self.resolve_decision_types()
        hero_cards = list(getattr(hand, "hero_cards", None) or getattr(analysis, "hero_cards", None) or [])
        if hand is None or len(hero_cards) != 2:
            return None

        resolved_street = str(
            street
            or getattr(hand, "street_state", {}).get("current_street")
            or getattr(analysis, "street", "")
            or ""
        ).lower()
        board = list(getattr(hand, "board_cards", None) or getattr(analysis, "board_cards", None) or [])
        if resolved_street not in POSTFLOP_STREETS or len(board) not in {3, 4, 5}:
            return None

        villain_positions = [
            pos
            for pos in self.bridge._ordered_positions(hand)
            if pos != hand.hero_position and not hand.player_states.get(pos, {}).get("is_fold", False)
        ]
        if not villain_positions:
            return None

        supports_line_builder = self.can_use_postflop_line_builder(board, resolved_street)
        range_build_mode = FULL_RUNOUT_LINE if supports_line_builder else PARTIAL_RUNTIME_LINE
        range_trace_expected = resolved_street in POSTFLOP_STREETS

        line_context = dict(self.bridge._build_postflop_line_context(hand, resolved_street) or {})
        line_context.setdefault("projection_mode", "runtime_partial_board")
        line_context["supports_line_builder"] = supports_line_builder
        line_context["board_card_count"] = len(board)
        line_context["range_build_mode"] = range_build_mode
        line_context["range_trace_expected"] = range_trace_expected

        return PostflopContext(
            hero_hand=list(hero_cards),
            board=list(board),
            pot_before_hero=self.bridge._pot_before_hero(hand),
            to_call=self.bridge._to_call(hand),
            effective_stack=self.bridge._effective_stack(hand),
            hero_position=hand.hero_position,
            villain_positions=list(villain_positions),
            line_context=line_context,
            dead_cards=[],
            street=resolved_street,
            player_count=int(hand.player_count),
            meta={
                "source": "pokervision",
                "hand_id": hand.hand_id,
                "hero_original_position": hand.hero_position,
                "projection_source": "postflop_runtime_projection",
                "hero_in_position": self.bridge._hero_in_position_postflop(hand),
                "line_builder_allowed": supports_line_builder,
                "range_build_mode": range_build_mode,
                "range_trace_expected": range_trace_expected,
            },
        )

    def build_contract_payload(self, context: Any) -> Dict[str, Any]:
        payload = {"context_type": self.canonical_context_type(context)}
        payload.update(self.serializer(context))
        return payload

    def can_use_postflop_line_builder(self, board_cards: Sequence[str], street: str) -> bool:
        expected_by_street = {"flop": 3, "turn": 4, "river": 5}
        resolved_street = str(street or "").lower()
        expected_count = expected_by_street.get(resolved_street)
        if expected_count is None:
            return False
        current_count = len(list(board_cards or []))
        if current_count != expected_count:
            return False
        # Only river satisfies the downstream `board_runout` contract.
        # Flop and turn remain valid PostflopContext spots, but they must use the
        # partial-board runtime path and not the full line-builder path.
        return resolved_street == "river" and current_count == 5
