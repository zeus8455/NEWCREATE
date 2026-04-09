from __future__ import annotations

import unittest

from pokervision.table_renderer import build_analysis_panel_sections, summarize_action_history, summarize_range_debug


class Step8UiContractTests(unittest.TestCase):
    def test_analysis_panel_sections_use_top_level_solver_contract(self) -> None:
        render_state = {
            "street": "flop",
            "hero_position": "SB",
            "recommended_action": "FOLD",
            "recommended_amount_to": None,
            "recommended_size_pct": None,
            "node_type": "facing_open",
            "engine_status": "ok",
            "solver_status": "computed",
            "solver_result_reused": False,
            "solver_warnings": [],
            "solver_errors": [],
            "advisor_input": {
                "pot_before_hero": 11.0,
                "to_call": 4.0,
                "effective_stack": 96.0,
            },
            "action_annotations": {
                "actions_log": [
                    {
                        "street": "flop",
                        "position": "MP",
                        "semantic_action": "bet",
                        "amount_bb": 4.0,
                    }
                ]
            },
            "solver_output": {
                "result": {
                    "postflop": {
                        "villain_sources": [
                            {
                                "name": "MP",
                                "source_type": "preflop_spot_action_range",
                                "normalized_expr": "AQo AJs KQs",
                            }
                        ]
                    }
                }
            },
            "analysis_panel": {},
        }

        sections = build_analysis_panel_sections(render_state, {"frame_id": "frame_0001"})
        titles = [section["title"] for section in sections]
        self.assertIn("Decision", titles)
        self.assertIn("Context", titles)
        self.assertIn("Action history", titles)
        self.assertIn("Ranges / debug", titles)
        self.assertIn("Solver", titles)

        decision = next(section for section in sections if section["title"] == "Decision")
        self.assertTrue(any("Action: FOLD" in line for line in decision["lines"]))

    def test_action_history_summary_prefers_semantic_action(self) -> None:
        render_state = {
            "action_annotations": {
                "actions_log": [
                    {
                        "street": "turn",
                        "position": "SB",
                        "semantic_action": "bet",
                        "amount_bb": 11.5,
                    }
                ]
            }
        }
        lines = summarize_action_history(render_state)
        self.assertEqual(lines, ["TURN SB: BET 11.5bb"])

    def test_range_debug_summary_uses_analysis_panel_when_present(self) -> None:
        render_state = {
            "analysis_panel": {
                "range_debug": [
                    {
                        "name": "CO",
                        "source_type": "range_after_filter",
                        "normalized_expr": "AK AQ AJ",
                    }
                ]
            }
        }
        lines = summarize_range_debug(render_state)
        self.assertEqual(lines, ["CO [range_after_filter] AK AQ AJ"])


if __name__ == "__main__":
    unittest.main()
