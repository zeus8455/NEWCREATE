from __future__ import annotations

from dataclasses import dataclass, field
import unittest

from pokervision.solver_bridge import EngineBridge


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
class FakeHeroDecision:
    street: str
    engine_action: str
    amount_to: float | None = None
    size_pct: float | None = None
    actor_name: str = "Hero"
    actor_pos: str | None = None
    reason: str = ""
    confidence: float = 1.0
    source: str = "test"
    preflop: object | None = None
    postflop: object | None = None


class SolverContextProjectionTests(unittest.TestCase):
    def setUp(self):
        import pokervision.solver_bridge as bridge_mod
        self.bridge_mod = bridge_mod
        self.orig_import = bridge_mod._import_decision_types
        self.orig_pre = bridge_mod._solve_hero_preflop
        self.orig_post = bridge_mod._solve_hero_postflop
        bridge_mod._import_decision_types = lambda: (FakeHeroDecision, FakePostflopContext, FakePreflopContext)
        bridge_mod._solve_hero_preflop = lambda context: FakeHeroDecision(
            street="preflop", engine_action="raise", actor_pos=context.hero_pos, preflop={"node_type": context.node_type}
        )
        bridge_mod._solve_hero_postflop = lambda context, **kwargs: FakeHeroDecision(
            street=context.street or "flop", engine_action="check", actor_pos=context.hero_position, postflop={"pot": context.pot_before_hero}
        )

    def tearDown(self):
        self.bridge_mod._import_decision_types = self.orig_import
        self.bridge_mod._solve_hero_preflop = self.orig_pre
        self.bridge_mod._solve_hero_postflop = self.orig_post

    def test_preflop_projection_prefers_action_state_preview(self):
        bridge = EngineBridge(settings=None)
        analysis = type("A", (), {"street": "preflop", "hero_cards": ["As", "Kh"]})()
        hand = type("H", (), {})()
        hand.hand_id = "hand_test"
        hand.hero_position = "CO"
        hand.player_count = 6
        hand.hero_cards = ["As", "Kh"]
        hand.board_cards = []
        hand.action_state = {
            "node_type_preview": "facing_open",
            "hero_context_preview": {
                "hero_pos": "CO",
                "node_type": "facing_open",
                "opener_pos": "UTG",
                "three_bettor_pos": None,
                "four_bettor_pos": None,
                "limpers": 0,
                "callers": 1,
            },
            "action_history": [{"position": "UTG", "action": "OPEN", "amount_bb": 2.5}],
        }
        hand.actions_log = list(hand.action_state["action_history"])
        hand.street_state = {"current_street": "preflop"}
        hand.occupied_positions = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
        hand.player_states = {pos: {"is_fold": False} for pos in hand.occupied_positions}
        payload = bridge.build_recommendation(analysis, hand)
        self.assertEqual(payload["context_type"], "PreflopContext")
        self.assertEqual(payload["advisor_input"]["node_type"], "facing_open")
        self.assertEqual(payload["advisor_input"]["opener_pos"], "UTG")
        self.assertEqual(payload["advisor_input"]["callers"], 1)
        self.assertEqual(payload["advisor_input"]["action_history"][0]["action"], "OPEN")

    def test_postflop_projection_builds_official_contract(self):
        bridge = EngineBridge(settings=None)
        analysis = type("A", (), {"street": "flop", "hero_cards": ["Kh", "8s"], "board_cards": ["Ah", "5d", "4h"]})()
        hand = type("H", (), {})()
        hand.hand_id = "hand_post"
        hand.hero_position = "SB"
        hand.player_count = 6
        hand.hero_cards = ["Kh", "8s"]
        hand.board_cards = ["Ah", "5d", "4h"]
        hand.action_state = {}
        hand.actions_log = [{"position": "MP", "street": "flop", "action": "BET", "amount_bb": 4.0}]
        hand.street_state = {"current_street": "flop"}
        hand.occupied_positions = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
        hand.player_states = {
            "BTN": {"is_fold": True, "stack_bb": 142.5},
            "SB": {"is_fold": False, "stack_bb": 97.0},
            "BB": {"is_fold": True, "stack_bb": 108.0},
            "UTG": {"is_fold": True, "stack_bb": 121.0},
            "MP": {"is_fold": False, "stack_bb": 96.0},
            "CO": {"is_fold": False, "stack_bb": 107.0},
        }
        hand.table_amount_state = {
            "total_pot": {"amount_bb": 11.0},
            "bets_by_position": {"MP": {"amount_bb": 4.0}},
        }
        payload = bridge.build_recommendation(analysis, hand)
        self.assertEqual(payload["context_type"], "PostflopContext")
        self.assertEqual(payload["solver_context"]["pot_before_hero"], 11.0)
        self.assertEqual(payload["solver_context"]["to_call"], 4.0)
        self.assertEqual(payload["solver_context"]["effective_stack"], 96.0)
        self.assertEqual(payload["solver_context"]["villain_positions"], ["MP", "CO"])
        self.assertEqual(payload["solver_context"]["street"], "flop")


if __name__ == "__main__":
    unittest.main()
