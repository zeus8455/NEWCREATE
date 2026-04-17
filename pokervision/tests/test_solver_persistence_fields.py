from __future__ import annotations

import unittest

from pokervision.models import HandState
from pokervision.pipeline import _apply_solver_payload
from pokervision.render_state import build_render_state


class SolverPersistenceFieldsTests(unittest.TestCase):
    def _make_hand(self) -> HandState:
        return HandState(
            schema_version="1.1",
            hand_id="hand_000001",
            status="active",
            player_count=6,
            table_format="6max",
            created_at="2026-04-09T00:00:00",
            updated_at="2026-04-09T00:00:00",
            last_seen_at="2026-04-09T00:00:00",
            hero_position="CO",
            hero_cards=["Ah", "Kd"],
            occupied_positions=["BTN", "SB", "BB", "UTG", "MP", "CO"],
            street_state={"current_street": "preflop", "street_history": ["preflop"]},
            player_states={pos: {"is_fold": pos not in {"CO", "SB"}, "is_active": pos in {"CO", "SB"}} for pos in ["BTN", "SB", "BB", "UTG", "MP", "CO"]},
            action_state={"node_type_preview": "facing_open", "last_actions_by_position": {}},
            processing_summary={},
        )

    def test_apply_solver_payload_persists_solver_input_and_output(self):
        class DummyAnalysis:
            street = "preflop"
            hero_cards = ["Ah", "Kd"]
            board_cards = []
            solver_context_preview = {}
            solver_result = {}
            solver_status = "not_run"

        hand = self._make_hand()
        payload = {
            "status": "ok",
            "solver_context": {"context_type": "PreflopContext", "node_type": "facing_open"},
            "advisor_input": {"context_type": "PreflopContext", "node_type": "facing_open"},
            "solver_input": {"context_type": "PreflopContext", "node_type": "facing_open"},
            "solver_output": {"result": {"engine_action": "raise"}, "context_type": "PreflopContext"},
            "result": type("R", (), {
                "street": "preflop",
                "engine_action": "raise",
                "amount_to": 8.0,
                "size_pct": None,
                "reason": "preflop:raise",
                "confidence": 0.9,
                "source": "test",
                "actor_name": "Hero",
                "actor_pos": "CO",
                "preflop": None,
                "postflop": None,
            })(),
        }
        _apply_solver_payload(DummyAnalysis(), hand, payload)
        self.assertEqual(hand.solver_input.get("node_type"), "facing_open")
        self.assertEqual(hand.solver_output.get("context_type"), "PreflopContext")
        self.assertEqual(hand.engine_result.get("engine_action"), "raise")

    def test_render_state_exports_recommended_fields(self):
        hand = self._make_hand()
        hand.advisor_input = {"context_type": "PreflopContext", "node_type": "facing_open"}
        hand.solver_input = {"context_type": "PreflopContext", "node_type": "facing_open"}
        hand.solver_output = {"result": {"engine_action": "raise"}}
        hand.engine_result = {"status": "ok", "engine_action": "raise", "amount_to": 8.0, "size_pct": None}
        hand.solver_status = "ok"
        rs = build_render_state(hand, "frame_0001", "2026-04-09T00:00:00")
        self.assertEqual(rs.recommended_action, "RAISE")
        self.assertEqual(rs.recommended_amount_to, 8.0)
        self.assertEqual(rs.node_type, "facing_open")
        self.assertEqual(rs.engine_status, "ok")


if __name__ == "__main__":
    unittest.main()
