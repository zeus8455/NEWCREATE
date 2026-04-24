
from __future__ import annotations

import unittest

from pokervision.models import HandState
from pokervision.render_state import build_render_state


class RenderStateStep7ContractTests(unittest.TestCase):
    def _build_hand(self) -> HandState:
        return HandState(
            schema_version="1.1",
            hand_id="hand_000001",
            status="active",
            player_count=6,
            table_format="6max",
            created_at="2026-04-09T17:12:38.945",
            updated_at="2026-04-09T17:14:27.772",
            last_seen_at="2026-04-09T17:14:27.772",
            hero_position="SB",
            hero_cards=["Kh", "8s"],
            occupied_positions=["BTN", "SB", "BB", "UTG", "MP", "CO"],
            street_state={"current_street": "flop", "street_history": ["flop"]},
            board_cards=["4h", "5d", "Ad"],
            player_states={
                "SB": {"stack_bb": 97.0, "stack_text_raw": "97", "is_fold": False, "is_active": True, "warnings": []},
                "MP": {"stack_bb": 96.0, "stack_text_raw": "96", "is_fold": False, "is_active": True, "warnings": []},
                "CO": {"stack_bb": 107.0, "stack_text_raw": "107", "is_fold": False, "is_active": True, "warnings": []},
                "BTN": {"stack_bb": 142.5, "stack_text_raw": "142.5", "is_fold": True, "is_active": False, "warnings": []},
                "BB": {"stack_bb": 108.0, "stack_text_raw": "108", "is_fold": True, "is_active": False, "warnings": []},
                "UTG": {"stack_bb": 121.0, "stack_text_raw": "121", "is_fold": True, "is_active": False, "warnings": []},
            },
            table_amount_state={
                "total_pot": {"amount_bb": 11.0},
                "posted_blinds": {},
                "bets_by_position": {"MP": {"amount_bb": 4.0, "raw_text": "4"}},
                "warnings": [],
                "errors": [],
            },
            amount_normalization={
                "street": "flop",
                "total_pot_bb": 11.0,
                "forced_blinds_by_position": {},
                "visible_bets_by_position": {"MP": 4.0},
                "final_contribution_bb_by_pos": {"MP": 4.0, "SB": 0.0},
                "final_contribution_street_bb_by_pos": {"MP": 4.0, "SB": 0.0},
                "warnings": [],
            },
            action_state={
                "street": "flop",
                "node_type_preview": "",
                "last_actions_by_position": {"MP": "BET 4.0"},
                "final_contribution_bb_by_pos": {"MP": 4.0, "SB": 0.0},
                "final_contribution_street_bb_by_pos": {"MP": 4.0, "SB": 0.0},
                "action_history": [
                    {"street": "preflop", "position": "SB", "semantic_action": "call"},
                    {"street": "flop", "position": "MP", "semantic_action": "bet", "amount_bb": 4.0},
                ],
            },
            actions_log=[
                {"street": "flop", "position": "MP", "semantic_action": "bet", "amount_bb": 4.0},
            ],
            advisor_input={
                "context_type": "PostflopContext",
                "hero_hand": ["Kh", "8s"],
                "board": ["4h", "5d", "Ad"],
                "pot_before_hero": 11.0,
                "to_call": 4.0,
                "effective_stack": 96.0,
                "hero_position": "SB",
                "villain_positions": ["MP", "CO"],
                "line_context": {"projection_mode": "runtime_partial_board", "supports_line_builder": False},
                "street": "flop",
                "player_count": 6,
            },
            solver_input={"context_type": "PostflopContext"},
            solver_output={"context_type": "PostflopContext", "result": {"engine_action": "fold", "reason": "postflop:fold"}},
            engine_result={"status": "reused_previous_solver_result", "engine_action": "fold", "reason": "postflop:fold"},
            solver_context={"context_type": "PostflopContext"},
            solver_status="reused_previous_solver_result",
            solver_warnings=[],
            solver_errors=[],
            hero_decision_debug={"status": "ok", "type": "HeroDecision"},
            solver_fingerprint="abc123",
            solver_result_reused=True,
            solver_reuse_reason="same_solver_input_fingerprint",
            solver_run_frame_id="frame_0040",
            solver_run_timestamp="2026-04-09T17:14:10.701",
        )

    def test_hand_to_dict_exposes_step7_normalized_sections(self) -> None:
        hand = self._build_hand()
        payload = hand.to_dict()
        self.assertIn("amount_state", payload)
        self.assertIn("reconstructed_preflop", payload)
        self.assertIn("reconstructed_postflop", payload)
        self.assertIn("analysis_panel", payload)
        self.assertEqual(payload["analysis_panel"]["recommended_action"], "FOLD")
        self.assertEqual(payload["reconstructed_postflop"]["to_call"], 4.0)

    def test_render_state_exposes_step7_top_level_contract(self) -> None:
        hand = self._build_hand()
        render = build_render_state(hand, "frame_0042", "2026-04-09T17:14:27.772").to_dict()
        self.assertEqual(render["recommended_action"], "FOLD")
        self.assertEqual(render["recommended_amount_to"], None)
        self.assertEqual(render["engine_status"], "reused_previous_solver_result")
        self.assertIn("analysis_panel", render)
        self.assertIn("reconstructed_postflop", render)
        self.assertEqual(render["analysis_panel"]["solver_reused"], True)


if __name__ == "__main__":
    unittest.main()
