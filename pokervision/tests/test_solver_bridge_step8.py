from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
import unittest

from pokervision.models import HandState
from pokervision.pipeline import (
    _apply_reused_solver_payload,
    _apply_solver_payload,
    _mark_solver_reuse_hit,
    _mark_solver_run_attempt,
)
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


class SolverBridgeStep8Tests(unittest.TestCase):
    def setUp(self) -> None:
        import pokervision.solver_bridge as bridge_mod

        self.bridge_mod = bridge_mod
        self.orig_import = bridge_mod._import_decision_types
        self.orig_pre = bridge_mod._solve_hero_preflop
        self.orig_post = bridge_mod._solve_hero_postflop

        bridge_mod._import_decision_types = lambda: (
            FakeHeroDecision,
            FakePostflopContext,
            FakePreflopContext,
        )
        bridge_mod._solve_hero_preflop = lambda context: FakeHeroDecision(
            street="preflop",
            engine_action="raise",
            actor_pos=context.hero_pos,
            reason="preflop:test",
            preflop={
                "node_type": context.node_type,
                "opener_pos": context.opener_pos,
                "three_bettor_pos": context.three_bettor_pos,
                "callers": context.callers,
            },
        )
        bridge_mod._solve_hero_postflop = lambda context, **kwargs: FakeHeroDecision(
            street=context.street or "flop",
            engine_action="call",
            actor_pos=context.hero_position,
            reason="postflop:test",
            postflop={
                "street": context.street,
                "pot_before_hero": context.pot_before_hero,
                "to_call": context.to_call,
                "villain_positions": list(context.villain_positions),
            },
        )

    def tearDown(self) -> None:
        self.bridge_mod._import_decision_types = self.orig_import
        self.bridge_mod._solve_hero_preflop = self.orig_pre
        self.bridge_mod._solve_hero_postflop = self.orig_post

    def _make_preflop_analysis(self) -> SimpleNamespace:
        return SimpleNamespace(
            street="preflop",
            hero_cards=["As", "Kh"],
            board_cards=[],
            solver_context_preview={},
            solver_result={},
            solver_status="not_run",
        )

    def _make_preflop_hand(self) -> SimpleNamespace:
        actions = [
            {"street": "preflop", "position": "CO", "action": "OPEN", "amount_bb": 2.5},
            {"street": "preflop", "position": "BTN", "action": "CALL", "amount_bb": 2.5},
            {
                "street": "preflop",
                "position": "SB",
                "action": "RAISE",
                "semantic_action": "3bet",
                "amount_bb": 9.0,
            },
            {
                "street": "preflop",
                "position": "CO",
                "action": "FOLD",
                "semantic_action": "fold",
                "amount_bb": 0.0,
            },
        ]
        return SimpleNamespace(
            hand_id="hand_preflop_step8",
            player_count=6,
            hero_position="BTN",
            hero_cards=["As", "Kh"],
            board_cards=[],
            occupied_positions=["BTN", "SB", "BB", "UTG", "MP", "CO"],
            player_states={
                "BTN": {"is_fold": False, "is_active": True, "stack_bb": 100.0},
                "SB": {"is_fold": False, "is_active": True, "stack_bb": 92.0},
                "BB": {"is_fold": True, "is_active": False, "stack_bb": 100.0},
                "UTG": {"is_fold": True, "is_active": False, "stack_bb": 100.0},
                "MP": {"is_fold": True, "is_active": False, "stack_bb": 100.0},
                "CO": {"is_fold": True, "is_active": False, "stack_bb": 100.0},
            },
            street_state={"current_street": "preflop"},
            actions_log=list(actions),
            action_state={
                "node_type_preview": "facing_open_callers",
                "hero_context_preview": {
                    "hero_pos": "BTN",
                    "node_type": "facing_open_callers",
                    "opener_pos": "CO",
                    "three_bettor_pos": "SB",
                    "four_bettor_pos": None,
                    "limpers": 0,
                    "callers": 1,
                },
                "action_history": list(actions),
                "last_actions_by_position": {"SB": "RAISE 9.0", "CO": "FOLD"},
            },
            table_amount_state={},
            amount_normalization={},
            processing_summary={},
            solver_context={},
            advisor_input={},
            solver_input={},
            solver_output={},
            engine_result={},
            solver_status="not_run",
            solver_warnings=[],
            solver_errors=[],
            hero_decision_debug={},
            solver_fingerprint="",
            solver_result_reused=False,
            solver_reuse_reason=None,
            solver_run_frame_id=None,
            solver_run_timestamp=None,
        )

    def _make_postflop_analysis(self, street: str, board: list[str]) -> SimpleNamespace:
        return SimpleNamespace(
            street=street,
            hero_cards=["Kh", "8s"],
            board_cards=list(board),
            solver_context_preview={},
            solver_result={},
            solver_status="not_run",
        )

    def _make_postflop_hand(self, street: str, board: list[str]) -> SimpleNamespace:
        contribution = 11.5 if street in {"turn", "river"} else 4.0
        pot = 45.5 if street in {"turn", "river"} else 11.0
        actions = [{
            "street": street,
            "position": "SB",
            "action": "BET",
            "semantic_action": "bet",
            "amount_bb": contribution,
            "final_contribution_street_bb": contribution,
        }]
        return SimpleNamespace(
            hand_id=f"hand_{street}_step8",
            player_count=6,
            hero_position="CO",
            hero_cards=["Kh", "8s"],
            board_cards=list(board),
            occupied_positions=["BTN", "SB", "BB", "UTG", "MP", "CO"],
            player_states={
                "BTN": {"is_fold": True, "is_active": False, "stack_bb": 100.0},
                "SB": {"is_fold": False, "is_active": True, "stack_bb": 82.5},
                "BB": {"is_fold": True, "is_active": False, "stack_bb": 225.0},
                "UTG": {"is_fold": True, "is_active": False, "stack_bb": 136.5},
                "MP": {"is_fold": True, "is_active": False, "stack_bb": 32.5},
                "CO": {"is_fold": False, "is_active": True, "stack_bb": 83.5},
            },
            street_state={"current_street": street},
            actions_log=list(actions),
            action_state={"last_actions_by_position": {"SB": f"BET {contribution}"}},
            table_amount_state={
                "total_pot": {"amount_bb": pot},
                "bets_by_position": {"SB": {"amount_bb": contribution}},
            },
            amount_normalization={
                "street": street,
                "total_pot_bb": pot,
                "hero_position": "CO",
                "active_positions": ["SB", "CO"],
                "final_contribution_street_bb_by_pos": {
                    "BTN": 0.0,
                    "SB": contribution,
                    "BB": 0.0,
                    "UTG": 0.0,
                    "MP": 0.0,
                    "CO": 0.0,
                },
            },
            processing_summary={},
            solver_context={},
            advisor_input={},
            solver_input={},
            solver_output={},
            engine_result={},
            solver_status="not_run",
            solver_warnings=[],
            solver_errors=[],
            hero_decision_debug={},
            solver_fingerprint="",
            solver_result_reused=False,
            solver_reuse_reason=None,
            solver_run_frame_id=None,
            solver_run_timestamp=None,
        )

    def _make_runtime_handstate(self) -> HandState:
        return HandState(
            schema_version="1.1",
            hand_id="hand_runtime_reuse",
            status="active",
            player_count=6,
            table_format="6max",
            created_at="2026-04-09T00:00:00",
            updated_at="2026-04-09T00:00:00",
            last_seen_at="2026-04-09T00:00:00",
            hero_position="BTN",
            hero_cards=["As", "Kh"],
            occupied_positions=["BTN", "SB", "BB", "UTG", "MP", "CO"],
            street_state={"current_street": "preflop", "street_history": ["preflop"]},
            player_states={
                pos: {"is_fold": pos not in {"BTN", "SB"}, "is_active": pos in {"BTN", "SB"}, "stack_bb": 100.0}
                for pos in ["BTN", "SB", "BB", "UTG", "MP", "CO"]
            },
            action_state={
                "node_type_preview": "facing_open_callers",
                "hero_context_preview": {
                    "hero_pos": "BTN",
                    "node_type": "facing_open_callers",
                    "opener_pos": "CO",
                    "three_bettor_pos": "SB",
                    "four_bettor_pos": None,
                    "limpers": 0,
                    "callers": 1,
                },
                "action_history": [
                    {"street": "preflop", "position": "CO", "action": "OPEN", "amount_bb": 2.5},
                    {"street": "preflop", "position": "BTN", "action": "CALL", "amount_bb": 2.5},
                    {"street": "preflop", "position": "SB", "action": "RAISE", "semantic_action": "3bet", "amount_bb": 9.0},
                ],
                "last_actions_by_position": {"SB": "RAISE 9.0"},
            },
            processing_summary={},
        )

    def test_preflop_open_call_3bet_fold_opener_builds_correct_preflop_context(self):
        bridge = EngineBridge(settings=None)
        analysis = self._make_preflop_analysis()
        hand = self._make_preflop_hand()

        context = bridge.build_preflop_context(analysis, hand)

        self.assertIsInstance(context, FakePreflopContext)
        self.assertEqual(context.hero_pos, "BTN")
        self.assertEqual(context.node_type, "facing_open_callers")
        self.assertEqual(context.opener_pos, "CO")
        self.assertEqual(context.three_bettor_pos, "SB")
        self.assertEqual(context.callers, 1)
        self.assertEqual(context.limpers, 0)
        self.assertEqual(context.action_history[-1]["action"], "FOLD")
        self.assertEqual(context.action_history[-1]["position"], "CO")

    def test_preflop_node_type_preview_matches_solver_input_node_type(self):
        bridge = EngineBridge(settings=None)
        analysis = self._make_preflop_analysis()
        hand = self._make_preflop_hand()

        payload = bridge.build_recommendation(analysis, hand)

        self.assertEqual(payload["context_type"], "PreflopContext")
        self.assertEqual(payload["solver_input"]["node_type"], hand.action_state["node_type_preview"])
        self.assertEqual(payload["advisor_input"]["node_type"], hand.action_state["node_type_preview"])
        self.assertEqual(payload["solver_output"]["result"]["preflop"]["node_type"], hand.action_state["node_type_preview"])

    def test_postflop_turn_builds_valid_postflop_context(self):
        bridge = EngineBridge(settings=None)
        analysis = self._make_postflop_analysis("turn", ["Ah", "5d", "4h", "2c"])
        hand = self._make_postflop_hand("turn", ["Ah", "5d", "4h", "2c"])

        context = bridge.build_postflop_context(analysis, hand, "turn")

        self.assertIsInstance(context, FakePostflopContext)
        self.assertEqual(context.street, "turn")
        self.assertEqual(len(context.board), 4)
        self.assertEqual(context.hero_position, "CO")
        self.assertEqual(context.villain_positions, ["SB"])
        self.assertGreater(context.pot_before_hero, 0.0)
        self.assertGreater(context.to_call, 0.0)
        self.assertIsNotNone(context.effective_stack)
        self.assertEqual(context.meta.get("projection_source"), "postflop_runtime_projection")

    def test_postflop_river_builds_valid_postflop_context(self):
        bridge = EngineBridge(settings=None)
        analysis = self._make_postflop_analysis("river", ["Ah", "5d", "4h", "2c", "9s"])
        hand = self._make_postflop_hand("river", ["Ah", "5d", "4h", "2c", "9s"])

        context = bridge.build_postflop_context(analysis, hand, "river")

        self.assertIsInstance(context, FakePostflopContext)
        self.assertEqual(context.street, "river")
        self.assertEqual(len(context.board), 5)
        self.assertEqual(context.hero_position, "CO")
        self.assertEqual(context.villain_positions, ["SB"])
        self.assertGreater(context.pot_before_hero, 0.0)
        self.assertGreater(context.to_call, 0.0)
        self.assertIsNotNone(context.effective_stack)
        self.assertEqual(context.meta.get("projection_source"), "postflop_runtime_projection")

    def test_bridge_does_not_crash_when_solver_raises(self):
        bridge = EngineBridge(settings=None)
        analysis = self._make_postflop_analysis("turn", ["Ah", "5d", "4h", "2c"])
        hand = self._make_postflop_hand("turn", ["Ah", "5d", "4h", "2c"])

        self.bridge_mod._solve_hero_postflop = lambda context, **kwargs: (_ for _ in ()).throw(RuntimeError("range build failed"))

        try:
            payload = bridge.build_recommendation(analysis, hand)
        except Exception as exc:  # pragma: no cover - test must fail loudly if the bridge still crashes
            self.fail(f"bridge crashed instead of returning a controlled payload: {exc}")

        self.assertIsInstance(payload, dict)
        self.assertIn(payload.get("status"), {"ok", "error", "not_run"})

    def test_repeated_frame_reuses_previous_solver_result_when_fingerprint_same(self):
        bridge = EngineBridge(settings=None)
        analysis = self._make_preflop_analysis()
        hand = self._make_runtime_handstate()

        preview_first = bridge.build_recommendation_preview(analysis, hand)
        result_first = bridge.build_recommendation(analysis, hand)

        self.assertEqual(preview_first["status"], "ready")
        self.assertEqual(preview_first["context_type"], "PreflopContext")
        self.assertTrue(preview_first["fingerprint"])

        _mark_solver_run_attempt(hand)
        _apply_solver_payload(analysis, hand, result_first)
        hand.solver_fingerprint = preview_first["fingerprint"]

        preview_second = bridge.build_recommendation_preview(analysis, hand)
        self.assertEqual(preview_second["fingerprint"], preview_first["fingerprint"])
        self.assertEqual(preview_second["solver_input"], preview_first["solver_input"])

        _mark_solver_reuse_hit(hand)
        _apply_reused_solver_payload(analysis, hand, preview_second)

        self.assertEqual(hand.processing_summary["solver_runs"], 1)
        self.assertEqual(hand.processing_summary["solver_reuse_hits"], 1)
        self.assertEqual(hand.solver_status, "reused_previous_solver_result")
        self.assertTrue(hand.solver_result_reused)
        self.assertEqual(hand.solver_reuse_reason, "same_solver_input_fingerprint")
        self.assertEqual(hand.solver_fingerprint, preview_first["fingerprint"])
        self.assertEqual(hand.solver_output, result_first["solver_output"])
        self.assertEqual(hand.engine_result["status"], "reused_previous_solver_result")
        self.assertTrue(hand.engine_result["reused"])


if __name__ == "__main__":
    unittest.main()
