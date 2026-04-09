from __future__ import annotations

import unittest
from types import SimpleNamespace

from pokervision.models import BBox, Detection
from pokervision.table_amount_logic import build_table_amount_state


class PostflopBlindReinterpretationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = SimpleNamespace(
            table_amount_region_iou_threshold=0.6,
            table_amount_region_center_threshold_px=35.0,
            table_amount_digit_iou_threshold=0.2,
            table_amount_digit_center_threshold_px=10.0,
            table_pot_center_exclusion_radius_px=40.0,
            blind_marker_to_position_max_distance_px=250.0,
            chips_to_position_max_distance_px=450.0,
            chips_ambiguity_margin_px=25.0,
            chips_target_towards_table_center=0.58,
            chips_projection_outside_segment_slack=0.22,
            chips_projection_penalty_scale_px=220.0,
            chips_line_distance_weight=0.85,
            chips_folded_override_margin_px=80.0,
            chips_conflict_iou_threshold=0.3,
            chips_conflict_center_threshold_px=60.0,
        )
        self.positions = {
            "BTN": {"center": {"x": 620.0, "y": 760.0}},
            "SB": {"center": {"x": 240.0, "y": 620.0}},
            "BB": {"center": {"x": 260.0, "y": 260.0}},
        }
        self.player_states = {
            "BTN": {"is_fold": False},
            "SB": {"is_fold": False},
            "BB": {"is_fold": False},
        }
        self.table_center = (450.0, 450.0)

    def _digit(self, label: str, x1: float, y1: float, x2: float, y2: float, confidence: float = 0.99) -> Detection:
        return Detection(label=label, bbox=BBox(x1, y1, x2, y2), confidence=confidence)

    def test_postflop_bb_marker_becomes_chips(self) -> None:
        regions = [
            Detection("BB", BBox(330.0, 345.0, 410.0, 410.0), 0.95),
        ]
        digit_map = {
            "BB_0": [self._digit("4", 8.0, 8.0, 20.0, 28.0)],
        }
        result = build_table_amount_state(
            region_detections=regions,
            digit_detection_map=digit_map,
            positions=self.positions,
            player_states=self.player_states,
            table_center=self.table_center,
            street="flop",
            settings=self.settings,
        )
        self.assertTrue(result.ok)
        state = result.meta["table_amount_state"]
        self.assertEqual(state["posted_blinds"], {})
        self.assertIn("BB", state["bets_by_position"])
        bb_entry = state["bets_by_position"]["BB"]
        self.assertEqual(bb_entry["amount_bb"], 4.0)
        self.assertEqual(bb_entry["reinterpreted_from_blind_marker"], "BB")
        self.assertIn("postflop_blind_marker_reinterpreted_as_chips", bb_entry["warnings"])

    def test_postflop_bb_and_chips_conflict_collapse_to_one_bet(self) -> None:
        regions = [
            Detection("BB", BBox(330.0, 345.0, 410.0, 410.0), 0.95),
            Detection("Chips", BBox(332.0, 347.0, 412.0, 412.0), 0.97),
        ]
        digit_map = {
            "BB_0": [self._digit("4", 8.0, 8.0, 20.0, 28.0)],
            "Chips_1": [self._digit("4", 8.0, 8.0, 20.0, 28.0)],
        }
        result = build_table_amount_state(
            region_detections=regions,
            digit_detection_map=digit_map,
            positions=self.positions,
            player_states=self.player_states,
            table_center=self.table_center,
            street="turn",
            settings=self.settings,
        )
        self.assertTrue(result.ok)
        state = result.meta["table_amount_state"]
        self.assertEqual(state["posted_blinds"], {})
        self.assertEqual(set(state["bets_by_position"].keys()), {"BB"})
        bb_entry = state["bets_by_position"]["BB"]
        self.assertEqual(bb_entry["source"], "Chips")
        self.assertIn("postflop_blind_marker_conflict_resolved_to_chips", bb_entry["warnings"])

    def test_preflop_blind_marker_stays_posted_blind(self) -> None:
        regions = [
            Detection("BB", BBox(310.0, 305.0, 380.0, 365.0), 0.95),
        ]
        digit_map = {
            "BB_0": [self._digit("1", 8.0, 8.0, 16.0, 28.0)],
        }
        result = build_table_amount_state(
            region_detections=regions,
            digit_detection_map=digit_map,
            positions=self.positions,
            player_states=self.player_states,
            table_center=self.table_center,
            street="preflop",
            settings=self.settings,
        )
        self.assertTrue(result.ok)
        state = result.meta["table_amount_state"]
        self.assertIn("BB", state["posted_blinds"])
        self.assertEqual(state["bets_by_position"], {})


if __name__ == "__main__":
    unittest.main()
