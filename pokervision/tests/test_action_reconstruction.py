from __future__ import annotations

import types
import unittest

from pokervision.action_inference import infer_actions


class DummySettings:
    infer_checks_without_explicit_evidence = False


class ActionReconstructionTests(unittest.TestCase):
    def _analysis(self, *, player_count, occupied_positions, hero_position, contributions, player_states=None):
        return types.SimpleNamespace(
            street="preflop",
            player_count=player_count,
            occupied_positions=occupied_positions,
            hero_position=hero_position,
            player_states=player_states or {pos: {"is_fold": False} for pos in occupied_positions},
            amount_state={
                "source_mode": "forced_blinds_plus_visible_chips",
                "final_contribution_bb_by_pos": contributions,
                "final_contribution_street_bb_by_pos": contributions,
            },
            amount_normalization={
                "source_mode": "forced_blinds_plus_visible_chips",
                "final_contribution_bb_by_pos": contributions,
                "final_contribution_street_bb_by_pos": contributions,
            },
            frame_id="frame_001",
            timestamp="2026-04-08T12:00:00Z",
            table_amount_state={},
        )

    def test_utg_limp_mp_iso_co_call_sb_3bet(self):
        analysis = self._analysis(
            player_count=6,
            occupied_positions=["BTN", "SB", "BB", "UTG", "MP", "CO"],
            hero_position="MP",
            contributions={
                "UTG": 1.0,
                "MP": 2.5,
                "CO": 2.5,
                "BTN": 0.0,
                "SB": 9.0,
                "BB": 1.0,
            },
        )
        result = infer_actions(None, analysis, DummySettings())
        history = result["action_history"]
        self.assertEqual([step["semantic_action"] for step in history], ["limp", "iso_raise", "call", "3bet"])
        self.assertEqual(history[0]["pos"], "UTG")
        self.assertEqual(history[1]["pos"], "MP")
        self.assertEqual(history[2]["pos"], "CO")
        self.assertEqual(history[3]["pos"], "SB")
        self.assertEqual(result["opener_pos"], "MP")
        self.assertEqual(result["three_bettor_pos"], "SB")
        self.assertEqual(result["callers_after_open"], 1)
        self.assertEqual(result["node_type_preview"], "opener_vs_3bet")

    def test_sb_first_in_limp_bb_check(self):
        analysis = self._analysis(
            player_count=3,
            occupied_positions=["BTN", "SB", "BB"],
            hero_position="BB",
            contributions={
                "BTN": 0.0,
                "SB": 1.0,
                "BB": 1.0,
            },
        )
        result = infer_actions(None, analysis, DummySettings())
        history = result["action_history"]
        self.assertEqual([step["semantic_action"] for step in history], ["limp", "check"])
        self.assertEqual(history[0]["pos"], "SB")
        self.assertEqual(history[1]["pos"], "BB")
        self.assertEqual(result["node_type_preview"], "open_limp_first_in")

    def test_co_open_btn_3bet_sb_cold_4bet(self):
        analysis = self._analysis(
            player_count=6,
            occupied_positions=["BTN", "SB", "BB", "UTG", "MP", "CO"],
            hero_position="SB",
            contributions={
                "UTG": 0.0,
                "MP": 0.0,
                "CO": 2.5,
                "BTN": 8.0,
                "SB": 22.0,
                "BB": 1.0,
            },
        )
        result = infer_actions(None, analysis, DummySettings())
        history = result["action_history"]
        self.assertEqual([step["semantic_action"] for step in history], ["open_raise", "3bet", "4bet"])
        self.assertEqual(history[-1]["spot_family"], "cold_4bet")
        self.assertEqual(result["opener_pos"], "CO")
        self.assertEqual(result["three_bettor_pos"], "BTN")
        self.assertEqual(result["four_bettor_pos"], "SB")
        self.assertEqual(result["node_type_preview"], "cold_4bet")


if __name__ == "__main__":
    unittest.main()
