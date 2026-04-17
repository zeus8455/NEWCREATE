from __future__ import annotations

from types import SimpleNamespace
import unittest

from pokervision.action_inference import infer_actions
from pokervision.config import get_default_settings
from pokervision.models import BBox, Detection
from pokervision.table_amount_logic import build_table_amount_state


class PreflopResidualRegressionTests(unittest.TestCase):
    def setUp(self):
        self.settings = get_default_settings()

    def test_prefllop_residual_chips_can_match_folded_position(self):
        positions = {
            "BTN": {"center": {"x": 1272.7590, "y": 1115.0544}},
            "SB": {"center": {"x": 546.1273, "y": 920.8424}},
            "BB": {"center": {"x": 606.9286, "y": 343.0970}},
            "UTG": {"center": {"x": 1275.0746, "y": 243.4777}},
            "CO": {"center": {"x": 1999.8938, "y": 914.3338}},
        }
        player_states = {
            "BTN": {"is_fold": False},
            "SB": {"is_fold": False},
            "BB": {"is_fold": False},
            "UTG": {"is_fold": True},
            "CO": {"is_fold": True},
        }
        residual_region = Detection("Chips", BBox(1705.5256, 779.0468, 1830.2083, 883.0782), 0.96)
        residual_digits = [Detection("2", BBox(44.68, 69.04, 60.59, 95.37), 0.96)]

        result = build_table_amount_state(
            [residual_region],
            {"Chips_0": residual_digits},
            positions,
            player_states,
            table_center=(1275.0, 650.0),
            street="preflop",
            settings=self.settings,
        )

        self.assertTrue(result.ok)
        state = result.meta["table_amount_state"]
        self.assertIn("CO", state["bets_by_position"])
        self.assertEqual(state["bets_by_position"]["CO"]["amount_bb"], 2.0)
        self.assertFalse(state["bets_by_position"]["CO"]["matched_among_live_positions_only"])
        self.assertIn(
            "matched_to_folded_position_as_residual_preflop_contribution",
            state["bets_by_position"]["CO"]["warnings"],
        )

    def test_open_call_3bet_then_opener_fold_is_documented(self):
        analysis = SimpleNamespace(
            street="preflop",
            player_count=5,
            occupied_positions=["BTN", "SB", "BB", "UTG", "CO"],
            hero_position="BTN",
            player_states={
                "BTN": {"is_fold": False},
                "SB": {"is_fold": False},
                "BB": {"is_fold": False},
                "UTG": {"is_fold": True},
                "CO": {"is_fold": True},
            },
            amount_state={
                "final_contribution_bb_by_pos": {"BTN": 2.0, "SB": 0.5, "BB": 9.0, "UTG": 0.0, "CO": 2.0},
                "final_contribution_street_bb_by_pos": {"BTN": 2.0, "SB": 0.5, "BB": 9.0, "UTG": 0.0, "CO": 2.0},
            },
            frame_id="frame_regression",
            timestamp="2026-04-09T08:00:41.485",
        )

        action_state = infer_actions(None, analysis, self.settings)
        semantic_line = [(step["position"], step["semantic_action"]) for step in action_state["action_history"]]

        self.assertEqual(
            semantic_line,
            [("CO", "open_raise"), ("BTN", "call"), ("BB", "3bet"), ("CO", "fold")],
        )
        self.assertEqual(action_state["opener_pos"], "CO")
        self.assertEqual(action_state["three_bettor_pos"], "BB")
        self.assertEqual(action_state["callers_after_open"], 1)
        self.assertEqual(action_state["last_actions_by_position"]["CO"], "FOLD")
        self.assertEqual(
            action_state["historical_terminal_actions"][0]["reason"],
            "folded_with_residual_preflop_contribution_below_final_price",
        )
        self.assertEqual(action_state["historical_terminal_actions"][0]["facing_aggression"], "3bet")

    def test_limp_iso_then_limper_fold_is_documented(self):
        analysis = SimpleNamespace(
            street="preflop",
            player_count=5,
            occupied_positions=["BTN", "SB", "BB", "UTG", "CO"],
            hero_position="BTN",
            player_states={
                "BTN": {"is_fold": False},
                "SB": {"is_fold": False},
                "BB": {"is_fold": False},
                "UTG": {"is_fold": True},
                "CO": {"is_fold": False},
            },
            amount_state={
                "final_contribution_bb_by_pos": {"BTN": 0.0, "SB": 0.5, "BB": 1.0, "UTG": 1.0, "CO": 4.0},
                "final_contribution_street_bb_by_pos": {"BTN": 0.0, "SB": 0.5, "BB": 1.0, "UTG": 1.0, "CO": 4.0},
            },
            frame_id="frame_regression_2",
            timestamp="2026-04-09T08:10:41.485",
        )

        action_state = infer_actions(None, analysis, self.settings)
        semantic_line = [(step["position"], step["semantic_action"]) for step in action_state["action_history"]]

        self.assertEqual(
            semantic_line,
            [("UTG", "limp"), ("CO", "iso_raise"), ("UTG", "fold")],
        )
        self.assertEqual(action_state["last_actions_by_position"]["UTG"], "FOLD")
        self.assertEqual(action_state["historical_terminal_actions"][0]["facing_aggression"], "open_raise")


if __name__ == "__main__":
    unittest.main()
