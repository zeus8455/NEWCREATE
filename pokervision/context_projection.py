from __future__ import annotations

"""Official projection layer from PokerVision state to solver contexts.

This module owns the canonical conversion from internal PokerVision runtime
state into the external decision-layer contracts:
- PreflopContext
- PostflopContext

The goal is to keep `solver_bridge.py` thin and to ensure preview, fingerprint,
solver input, and persisted hand/debug payloads all come from one projection
source instead of several ad-hoc builders.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

POSTFLOP_STREETS = ("flop", "turn", "river")


ResolveDecisionTypes = Callable[[], tuple[Any, Any, Any]]
CanonicalContextType = Callable[[Any], str]
Serializer = Callable[[Any], Any]


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

    def build_preflop_context(self, analysis, hand):
        _, _, PreflopContext = self.resolve_decision_types()
        hero_cards = list(getattr(hand, "hero_cards", None) or getattr(analysis, "hero_cards", None) or [])
        if hand is None or len(hero_cards) != 2:
            return None

        action_state = getattr(hand, "action_state", {}) or {}
        hero_preview = dict(action_state.get("hero_context_preview") or {})
        fallback_spot = None

        node_type = str(hero_preview.get("node_type") or action_state.get("node_type_preview") or "").strip()
        if not node_type:
            fallback_spot = self.bridge._build_hero_preflop_spot(hand)
            node_type = fallback_spot.node_type

        opener_pos = hero_preview.get("opener_pos")
        three_bettor_pos = hero_preview.get("three_bettor_pos")
        four_bettor_pos = hero_preview.get("four_bettor_pos")
        limpers = hero_preview.get("limpers")
        callers = hero_preview.get("callers")

        if fallback_spot is None and (
            opener_pos is None
            and three_bettor_pos is None
            and four_bettor_pos is None
            and limpers is None
            and callers is None
        ):
            fallback_spot = self.bridge._build_hero_preflop_spot(hand)

        if fallback_spot is not None:
            opener_pos = fallback_spot.opener_pos if opener_pos is None else opener_pos
            three_bettor_pos = fallback_spot.three_bettor_pos if three_bettor_pos is None else three_bettor_pos
            four_bettor_pos = fallback_spot.four_bettor_pos if four_bettor_pos is None else four_bettor_pos
            limpers = fallback_spot.limpers if limpers is None else limpers
            callers = fallback_spot.callers if callers is None else callers

        action_history = list(action_state.get("action_history") or self.bridge._street_actions(hand, "preflop"))
        hero_pos = self.bridge._preflop_pos(hand.hero_position, int(hand.player_count))
        return PreflopContext(
            hero_hand=list(hero_cards),
            hero_pos=hero_pos,
            node_type=node_type or "unopened",
            opener_pos=opener_pos,
            three_bettor_pos=three_bettor_pos,
            four_bettor_pos=four_bettor_pos,
            limpers=int(limpers or 0),
            callers=int(callers or 0),
            range_owner="hero",
            action_history=action_history,
            meta={
                "source": "pokervision",
                "hand_id": hand.hand_id,
                "hero_original_position": hand.hero_position,
                "projection_source": "action_state_preview" if hero_preview else "replayed_actions_fallback",
                "actions_seen": self.bridge._street_actions(hand, "preflop"),
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

        line_context = self.bridge._build_postflop_line_context(hand, resolved_street)
        line_context = dict(line_context)
        line_context.setdefault("projection_mode", "runtime_partial_board")
        line_context["supports_line_builder"] = self.can_use_postflop_line_builder(board, resolved_street)
        line_context["board_card_count"] = len(board)

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
                "line_builder_allowed": self.can_use_postflop_line_builder(board, resolved_street),
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
