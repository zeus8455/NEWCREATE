from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pokervision import models  # noqa: E402


class ProjectionUiRefreshTests(unittest.TestCase):
    def setUp(self):
        self.action_state = {
            "projection_node_type": "fourbettor_vs_5bet",
            "advisor_node_type": "threebettor_vs_4bet",
            "advisor_mapping_reason": "closest_supported_rereraise_defense_node",
            "action_history_resolved": [
                {
                    "order": 1,
                    "pos": "UTG",
                    "street": "preflop",
                    "semantic_action": "open_raise",
                    "final_contribution_bb": 2.0,
                },
                {
                    "order": 2,
                    "pos": "MP",
                    "street": "preflop",
                    "semantic_action": "3bet",
                    "final_contribution_bb": 6.5,
                },
                {
                    "order": 3,
                    "pos": "UTG",
                    "street": "preflop",
                    "semantic_action": "4bet",
                    "final_contribution_bb": 11.0,
                },
                {
                    "order": 4,
                    "pos": "MP",
                    "street": "preflop",
                    "semantic_action": "5bet_jam",
                    "final_contribution_bb": 29.0,
                },
            ],
        }
        self.advisor_input = {
            "context_type": "preflop",
            "node_type": "threebettor_vs_4bet",
            "meta": {
                "projection_node_type": "fourbettor_vs_5bet",
                "advisor_node_type": "threebettor_vs_4bet",
                "advisor_mapping_reason": "closest_supported_rereraise_defense_node",
            },
        }

    def test_compute_node_type_prefers_projection_node_type(self):
        value = models._compute_node_type(self.action_state, self.advisor_input)
        self.assertEqual(value, "fourbettor_vs_5bet")

    def test_analysis_panel_exposes_projection_and_advisor_nodes(self):
        panel = models.derive_analysis_panel(
            street="preflop",
            hero_position="UTG",
            occupied_positions=["BTN", "SB", "BB", "UTG", "MP", "CO"],
            action_state=self.action_state,
            advisor_input=self.advisor_input,
            solver_input={"context_type": "preflop"},
            solver_output={},
            engine_result={
                "engine_action": "fold",
                "reason": "preflop:fold",
                "confidence": 1.0,
                "status": "ok",
            },
            hero_decision_debug={"preflop": {}},
            solver_status="ok",
            solver_warnings=[],
            solver_errors=[],
            solver_result_reused=False,
            solver_reuse_reason=None,
            solver_fingerprint=None,
        )
        self.assertEqual(panel.get("node_type"), "fourbettor_vs_5bet")
        self.assertEqual(panel.get("projection_node_type"), "fourbettor_vs_5bet")
        self.assertEqual(panel.get("advisor_node_type"), "threebettor_vs_4bet")
        self.assertEqual(panel.get("advisor_mapping_reason"), "closest_supported_rereraise_defense_node")


if __name__ == "__main__":
    unittest.main()
