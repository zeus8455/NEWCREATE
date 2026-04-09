from __future__ import annotations

import unittest
from types import SimpleNamespace

from pokervision.models import FrameAnalysis, HandState
from pokervision.pipeline import _apply_solver_payload
from pokervision.render_state import build_render_state


class SolverStateFieldsTests(unittest.TestCase):
    def _make_hand(self) -> HandState:
        return HandState(
            schema_version="1.1",
            hand_id="hand_1",
            status="active",
            player_count=6,
            table_format="6max",
            created_at="2026-04-09T10:00:00.000",
            updated_at="2026-04-09T10:00:00.000",
            last_seen_at="2026-04-09T10:00:00.000",
            hero_position="BB",
            hero_cards=["Ah", "Jc"],
            occupied_positions=["BTN", "SB", "BB", "UTG", "MP", "CO"],
            street_state={"current_street": "preflop", "street_history": ["preflop"]},
            player_states={
                "BTN": {"is_fold": True},
                "SB": {"is_fold": True},
                "BB": {"is_fold": False},
                "UTG": {"is_fold": False},
                "MP": {"is_fold": True},
                "CO": {"is_fold": False},
            },
            amount_normalization={
                "total_pot_bb": 3.5,
                "final_contribution_street_bb_by_pos": {"BB": 1.0, "UTG": 2.0},
            },
            table_amount_state={
                "bets_by_position": {
                    "UTG": {"amount_bb": 2.0, "raw_text": "2"},
                    "BB": {"amount_bb": 1.0, "raw_text": "1"},
                }
            },
            action_state={
                "node_type_preview": "facing_open",
                "hero_context_preview": {
                    "hero_pos": "BB",
                    "node_type": "facing_open",
                    "opener_pos": "UTG",
                    "three_bettor_pos": None,
                    "four_bettor_pos": None,
                    "limpers": 0,
                    "callers": 0,
                },
                "action_history": [{"pos": "UTG", "semantic_action": "open_raise", "amount_bb": 2.0}],
                "last_actions_by_position": {"UTG": "OPEN 2.0"},
            },
            actions_log=[{"pos": "UTG", "action": "OPEN", "amount_bb": 2.0}],
        )

    def _make_analysis(self) -> FrameAnalysis:
        return FrameAnalysis(
            frame_id="frame_1",
            timestamp="2026-04-09T10:00:00.000",
            active_hero_found=True,
            street="preflop",
            player_count=6,
            table_format="6max",
            hero_position="BB",
            hero_cards=["Ah", "Jc"],
            occupied_positions=["BTN", "SB", "BB", "UTG", "MP", "CO"],
        )

    def test_apply_solver_payload_populates_normalized_fields(self):
        hand = self._make_hand()
        analysis = self._make_analysis()
        hero_decision = SimpleNamespace(
            street="preflop",
            engine_action="raise",
            amount_to=None,
            size_pct=None,
            actor_name="Hero",
            actor_pos="BB",
            reason="preflop:3bet",
            confidence=0.7,
            source="hero_decision.preflop",
            preflop={"action": "3bet", "hand_class": "AJo"},
            postflop=None,
        )
        payload = {"status": "ok", "result": hero_decision}

        legacy = _apply_solver_payload(analysis, hand, payload)

        self.assertEqual(analysis.solver_status, "ok")
        self.assertEqual(hand.solver_status, "ok")
        self.assertEqual(hand.engine_result["engine_action"], "raise")
        self.assertEqual(hand.advisor_input["context_type"], "PreflopContext")
        self.assertEqual(hand.solver_context["node_type_preview"], "facing_open")
        self.assertEqual(hand.hero_decision_debug["type"], "SimpleNamespace")
        self.assertEqual(legacy["status"], "ok")

    def test_render_state_surfaces_top_level_solver_fields(self):
        hand = self._make_hand()
        analysis = self._make_analysis()
        payload = {
            "status": "ok",
            "result": SimpleNamespace(
                street="preflop",
                engine_action="raise",
                amount_to=None,
                size_pct=None,
                actor_name="Hero",
                actor_pos="BB",
                reason="preflop:3bet",
                confidence=0.7,
                source="hero_decision.preflop",
                preflop={"action": "3bet"},
                postflop=None,
            ),
        }
        _apply_solver_payload(analysis, hand, payload)

        render = build_render_state(hand, "frame_1", "2026-04-09T10:00:00.000")
        data = render.to_dict()

        self.assertEqual(data["solver_status"], "ok")
        self.assertIn("engine_action", data["engine_result"])
        self.assertEqual(data["advisor_input"]["context_type"], "PreflopContext")
        self.assertIn("raw_repr", data["hero_decision_debug"])
        self.assertIn("solver_bridge", data["action_annotations"])


if __name__ == "__main__":
    unittest.main()
