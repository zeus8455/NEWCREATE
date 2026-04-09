import unittest
from pathlib import Path


class PipelineSolverBridgeHookTests(unittest.TestCase):
    def test_pipeline_calls_solver_bridge_after_hand_update_and_before_render_state(self):
        source = Path(__file__).resolve().parents[1].joinpath("pipeline.py").read_text(encoding="utf-8")
        update_idx = source.index("hand, decision, created_new = self.hand_manager.update_or_create(analysis)")
        solver_idx = source.index("solver_result = build_solver_recommendation(analysis, hand, self.settings)")
        render_idx = source.index("render_state = build_render_state(hand, frame.frame_id, frame.timestamp).to_dict()")
        self.assertLess(update_idx, solver_idx)
        self.assertLess(solver_idx, render_idx)

    def test_render_state_exports_solver_bridge_summary(self):
        source = Path(__file__).resolve().parents[1].joinpath("render_state.py").read_text(encoding="utf-8")
        self.assertIn('"solver_bridge": solver_summary', source)


if __name__ == "__main__":
    unittest.main()
