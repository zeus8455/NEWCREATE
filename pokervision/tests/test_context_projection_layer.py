from __future__ import annotations

from dataclasses import dataclass, field
import unittest

from pokervision.context_projection import ContextProjector


@dataclass(slots=True)
class FakePreflopContext:
    hero_hand: list[str]
    hero_pos: str
    node_type: str
    range_owner: str = "hero"
    opener_pos: str | None = None
    three_bettor_pos: str | None = None
    four_bettor_pos: str | None = None
    limpers: int = 0
    callers: int = 0
    dead_cards: list[str] = field(default_factory=list)
    action_history: list[dict] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class FakePostflopContext:
    hero_hand: list[str]
    board: list[str]
    pot_before_hero: float
    to_call: float = 0.0
    effective_stack: float | None = None
    hero_position: str | None = None
    villain_positions: list[str] = field(default_factory=list)
    line_context: dict[str, object] = field(default_factory=dict)
    dead_cards: list[str] = field(default_factory=list)
    street: str | None = None
    player_count: int | None = None
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class FakeSpot:
    node_type: str
    hero_pos: str
    opener_pos: str | None = None
    three_bettor_pos: str | None = None
    four_bettor_pos: str | None = None
    limpers: int = 0
    callers: int = 0


class FakeBridge:
    def _build_hero_preflop_spot(self, hand):
        return FakeSpot(node_type="facing_open", hero_pos="CO", opener_pos="UTG", callers=1)

    def _street_actions(self, hand, street):
        if street == "preflop":
            return [{"position": "UTG", "action": "OPEN", "amount_bb": 2.5}]
        return [{"position": "MP", "street": street, "action": "BET", "amount_bb": 4.0}]

    def _preflop_pos(self, hero_position, player_count):
        return hero_position

    def _ordered_positions(self, hand):
        return list(hand.occupied_positions)

    def _build_postflop_line_context(self, hand, street):
        return {"prior_aggression": True, "street": street}

    def _pot_before_hero(self, hand):
        return 11.0

    def _to_call(self, hand):
        return 4.0

    def _effective_stack(self, hand):
        return 96.0

    def _hero_in_position_postflop(self, hand):
        return False


class ContextProjectionLayerTests(unittest.TestCase):
    def setUp(self):
        self.projector = ContextProjector(
            bridge=FakeBridge(),
            resolve_decision_types=lambda: (object, FakePostflopContext, FakePreflopContext),
            canonical_context_type=lambda value: type(value).__name__,
            serializer=lambda value: value.__dict__,
        )

    def test_preflop_projection_uses_preview_fields(self):
        analysis = type("A", (), {"street": "preflop", "hero_cards": ["As", "Kh"]})()
        hand = type("H", (), {})()
        hand.hand_id = "hand_pre"
        hand.hero_position = "CO"
        hand.player_count = 6
        hand.hero_cards = ["As", "Kh"]
        hand.action_state = {
            "node_type_preview": "facing_open",
            "hero_context_preview": {
                "hero_pos": "CO",
                "node_type": "facing_open",
                "opener_pos": "UTG",
                "callers": 1,
            },
            "action_history": [{"position": "UTG", "action": "OPEN", "amount_bb": 2.5}],
        }
        context = self.projector.build_preflop_context(analysis, hand)
        self.assertEqual(context.node_type, "facing_open")
        self.assertEqual(context.opener_pos, "UTG")
        self.assertEqual(context.callers, 1)
        self.assertEqual(context.meta["projection_source"], "action_state_preview")

    def test_flop_projection_keeps_partial_board_and_disables_line_builder(self):
        analysis = type("A", (), {"street": "flop", "hero_cards": ["Kh", "8s"], "board_cards": ["Ah", "5d", "4h"]})()
        hand = type("H", (), {})()
        hand.hand_id = "hand_flop"
        hand.hero_position = "SB"
        hand.player_count = 6
        hand.hero_cards = ["Kh", "8s"]
        hand.board_cards = ["Ah", "5d", "4h"]
        hand.street_state = {"current_street": "flop"}
        hand.occupied_positions = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
        hand.player_states = {
            "BTN": {"is_fold": True},
            "SB": {"is_fold": False},
            "BB": {"is_fold": True},
            "UTG": {"is_fold": True},
            "MP": {"is_fold": False},
            "CO": {"is_fold": False},
        }
        context = self.projector.build_postflop_context(analysis, hand, "flop")
        self.assertEqual(context.board, ["Ah", "5d", "4h"])
        self.assertFalse(context.line_context["supports_line_builder"])
        self.assertEqual(context.line_context["projection_mode"], "runtime_partial_board")
        self.assertFalse(context.meta["line_builder_allowed"])

    def test_river_projection_allows_line_builder(self):
        analysis = type("A", (), {"street": "river", "hero_cards": ["Kh", "8s"], "board_cards": ["Ah", "5d", "4h", "7c", "2c"]})()
        hand = type("H", (), {})()
        hand.hand_id = "hand_river"
        hand.hero_position = "CO"
        hand.player_count = 6
        hand.hero_cards = ["Kh", "8s"]
        hand.board_cards = ["Ah", "5d", "4h", "7c", "2c"]
        hand.street_state = {"current_street": "river"}
        hand.occupied_positions = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
        hand.player_states = {
            "BTN": {"is_fold": True},
            "SB": {"is_fold": False},
            "BB": {"is_fold": True},
            "UTG": {"is_fold": True},
            "MP": {"is_fold": True},
            "CO": {"is_fold": False},
        }
        context = self.projector.build_postflop_context(analysis, hand, "river")
        self.assertTrue(context.line_context["supports_line_builder"])
        self.assertTrue(context.meta["line_builder_allowed"])


if __name__ == "__main__":
    unittest.main()
