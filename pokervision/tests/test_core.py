from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pokervision.action_inference import infer_actions
from pokervision.capture import MockFrameSource
from pokervision.config import get_default_settings
from pokervision.detectors import MockDetectorBackend, build_board_crop, build_hero_crop, build_player_state_crop
from pokervision.hand_state import HandStateManager, MATCH_WEAK_CONFLICT
from pokervision.models import BBox, Detection, FrameAnalysis, HandState
from pokervision.pipeline import PokerVisionPipeline
from pokervision.render_state import build_render_state
from pokervision.storage import StorageManager
from pokervision.table_amount_logic import build_table_amount_state
from pokervision.table_logic import assign_positions, determine_hero_position
from pokervision.validators import (
    determine_street,
    parse_numeric_tokens,
    validate_board_cards,
    validate_hero_cards,
    validate_player_state,
    validate_structure,
)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = Path(tempfile.mkdtemp(prefix="pokervision_test_"))
        self.settings = get_default_settings()
        self.settings.root_dir = self.tempdir
        self.settings.debug_mode = True
        self.storage = StorageManager(self.settings)
        self.detector = MockDetectorBackend(self.settings)
        self.hand_manager = HandStateManager(
            self.settings.schema_version,
            self.settings.hand_stale_timeout_sec,
            self.settings.hand_close_timeout_sec,
        )
        self.pipeline = PokerVisionPipeline(self.settings, self.detector, self.storage, self.hand_manager)
        self.source = MockFrameSource(*self.settings.mock_table_size)

    def tearDown(self):
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def test_structure_and_positions(self):
        frame = self.source.next_frame()
        structure = self.detector.detect_structure(frame)
        v = validate_structure(structure, self.settings.duplicate_iou_threshold, self.settings.duplicate_center_distance_px)
        self.assertTrue(v.ok)
        self.assertEqual(v.meta["player_count"], 6)
        center, positions = assign_positions(v.meta["seats"], v.meta["btn"][0], 6)
        hero = determine_hero_position(positions, self.detector.detect_active_hero(frame), self.settings.seat_match_max_distance_px)
        self.assertIn(hero, positions)
        self.assertEqual(len(positions), 6)

    def test_street_detection(self):
        frame1 = self.source.next_frame()
        street1 = determine_street(self.detector.detect_structure(frame1))
        self.assertEqual(street1.meta["street"], "preflop")
        self.source.next_frame()
        frame3 = self.source.next_frame()
        street3 = determine_street(self.detector.detect_structure(frame3))
        self.assertEqual(street3.meta["street"], "flop")

    def test_hero_and_board_card_validation(self):
        frame1 = self.source.next_frame()
        hero_cards = self.detector.detect_hero_cards(frame1, None)
        hero_val = validate_hero_cards(hero_cards)
        self.assertTrue(hero_val.ok)
        self.assertEqual(hero_val.meta["cards"], ["Ah", "Jd"])

        self.source.next_frame()
        frame3 = self.source.next_frame()
        board_cards = self.detector.detect_board_cards(frame3, None, "flop")
        board_val = validate_board_cards(board_cards, "flop")
        self.assertTrue(board_val.ok)
        self.assertEqual(board_val.meta["cards"], ["8s", "4s", "Kd"])

    def test_strict_crops_follow_structure_bbox(self):
        frame = self.source.next_frame()
        structure = self.detector.detect_structure(frame)
        v = validate_structure(structure, self.settings.duplicate_iou_threshold, self.settings.duplicate_center_distance_px)
        btn = v.meta["btn"][0]
        hero_crop, _ = build_hero_crop(frame.image, btn.bbox, self.settings)
        self.assertEqual(hero_crop.shape[1], int(round(btn.bbox.width)))
        self.assertEqual(hero_crop.shape[0], int(round(btn.bbox.height)))

        board_crop, _ = build_board_crop(frame.image, btn.bbox, self.settings)
        self.assertEqual(board_crop.shape[1], int(round(btn.bbox.width)))
        self.assertEqual(board_crop.shape[0], int(round(btn.bbox.height)))

        player_crop, _ = build_player_state_crop(frame.image, btn.bbox, self.settings)
        self.assertEqual(player_crop.shape[1], int(round(btn.bbox.width)))
        self.assertEqual(player_crop.shape[0], int(round(btn.bbox.height)))

    def test_player_state_parsing(self):
        detections = [
            Detection("1", BBox(10, 50, 20, 62), 0.95),
            Detection("2", BBox(24, 50, 34, 62), 0.95),
            Detection(".", BBox(38, 50, 44, 62), 0.92),
            Detection("5", BBox(48, 50, 58, 62), 0.95),
        ]
        result = validate_player_state(
            "BTN",
            detections,
            crop_height=80,
            lower_band_ratio=0.55,
            iou_threshold=self.settings.player_state_token_iou_threshold,
            center_threshold=self.settings.player_state_token_center_threshold_px,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.meta["player_state"]["stack_text_raw"], "12.5")
        self.assertEqual(result.meta["player_state"]["stack_bb"], 12.5)

    def test_player_state_fold_token_above_lower_band_is_kept(self):
        detections = [
            Detection("fold", BBox(10, 8, 44, 24), 0.95),
            Detection("1", BBox(12, 50, 20, 62), 0.95),
            Detection("0", BBox(24, 50, 34, 62), 0.95),
            Detection("3", BBox(36, 50, 46, 62), 0.95),
        ]
        result = validate_player_state(
            "CO",
            detections,
            crop_height=80,
            lower_band_ratio=0.55,
            iou_threshold=self.settings.player_state_token_iou_threshold,
            center_threshold=self.settings.player_state_token_center_threshold_px,
        )
        self.assertTrue(result.ok)
        self.assertTrue(result.meta["player_state"]["is_fold"])
        self.assertEqual(result.meta["player_state"]["stack_bb"], 103.0)

    def test_overlapping_numeric_conflict_keeps_best_token(self):
        detections = [
            Detection("1", BBox(10, 50, 20, 62), 0.95),
            Detection("8", BBox(24, 50, 36, 62), 0.95),
            Detection("6", BBox(38, 50, 50, 62), 0.90),
            Detection("5", BBox(39, 50, 51, 62), 0.82),
            Detection(".", BBox(54, 56, 58, 62), 0.92),
            Detection("5", BBox(62, 50, 74, 62), 0.95),
        ]
        result = validate_player_state(
            "UTG",
            detections,
            crop_height=80,
            lower_band_ratio=0.55,
            iou_threshold=self.settings.player_state_token_iou_threshold,
            center_threshold=self.settings.player_state_token_center_threshold_px,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.meta["player_state"]["stack_text_raw"], "186.5")
        self.assertEqual(result.meta["player_state"]["stack_bb"], 186.5)

    def test_numeric_amount_parser(self):
        detections = [
            Detection("1", BBox(10, 10, 18, 24), 0.95),
            Detection("2", BBox(24, 10, 32, 24), 0.95),
            Detection(".", BBox(36, 10, 40, 24), 0.93),
            Detection("5", BBox(44, 10, 52, 24), 0.95),
        ]
        val = parse_numeric_tokens(detections, self.settings.table_amount_digit_iou_threshold, self.settings.table_amount_digit_center_threshold_px)
        self.assertTrue(val.ok)
        self.assertEqual(val.meta["raw_text"], "12.5")
        self.assertEqual(val.meta["amount"], 12.5)

    def test_chips_match_uses_live_players_and_betting_lane(self):
        positions = {
            "BTN": {"center": {"x": 1260.0, "y": 240.0}},
            "SB": {"center": {"x": 1600.0, "y": 920.0}},
            "BB": {"center": {"x": 1080.0, "y": 900.0}},
            "UTG": {"center": {"x": 980.0, "y": 260.0}},
            "CO": {"center": {"x": 1880.0, "y": 300.0}},
        }
        player_states = {
            "BTN": {"is_fold": True},
            "SB": {"is_fold": False},
            "BB": {"is_fold": False},
            "UTG": {"is_fold": True},
            "CO": {"is_fold": False},
        }
        region = Detection("Chips", BBox(1747.3135, 441.5223, 1860.9194, 536.8638), 0.96)
        digits = [
            Detection("2", BBox(33.9, 60.8, 48.2, 84.3), 0.92),
            Detection(".", BBox(48.7, 76.9, 55.0, 84.6), 0.77),
            Detection("5", BBox(57.5, 60.7, 71.1, 84.7), 0.96),
        ]
        result = build_table_amount_state(
            [region],
            {"Chips_0": digits},
            positions,
            player_states,
            table_center=(1450.0, 620.0),
            street="preflop",
            settings=self.settings,
        )
        self.assertTrue(result.ok)
        state = result.meta["table_amount_state"]
        self.assertIn("CO", state["bets_by_position"])
        self.assertEqual(state["bets_by_position"]["CO"]["amount_bb"], 2.5)
        self.assertEqual(state["unassigned_chips"], [])

    def test_table_amount_digits_survive_cleaned_region_reordering(self):
        positions = {
            "BTN": {"center": {"x": 760.0, "y": 760.0}},
            "SB": {"center": {"x": 320.0, "y": 600.0}},
            "BB": {"center": {"x": 700.0, "y": 180.0}},
            "UTG": {"center": {"x": 1180.0, "y": 220.0}},
            "MP": {"center": {"x": 1400.0, "y": 560.0}},
            "CO": {"center": {"x": 1160.0, "y": 760.0}},
        }
        player_states = {
            "BTN": {"is_fold": False},
            "SB": {"is_fold": True},
            "BB": {"is_fold": False},
            "UTG": {"is_fold": True},
            "MP": {"is_fold": True},
            "CO": {"is_fold": False},
        }
        chips = Detection("Chips", BBox(772.5, 739.0, 875.9, 832.8), 0.96)
        total_pot = Detection("TotalPot", BBox(1309.7, 490.1, 1391.9, 534.1), 0.96)
        chip_digits = [
            Detection("7", BBox(10, 12, 22, 36), 0.95),
            Detection(".", BBox(24, 28, 28, 36), 0.90),
            Detection("5", BBox(30, 12, 42, 36), 0.96),
        ]
        pot_digits = [
            Detection("2", BBox(8, 8, 20, 32), 0.95),
            Detection("9", BBox(24, 8, 36, 32), 0.95),
        ]
        result = build_table_amount_state(
            [chips, total_pot],
            {
                "Chips_0": chip_digits,
                "TotalPot_1": pot_digits,
            },
            positions,
            player_states,
            table_center=(960.0, 520.0),
            street="river",
            settings=self.settings,
        )
        self.assertTrue(result.ok)
        state = result.meta["table_amount_state"]
        self.assertEqual(state["total_pot"]["amount_bb"], 29.0)
        self.assertEqual(state["bets_by_position"]["BTN"]["amount_bb"], 7.5)


    def test_render_state_keeps_folded_players_in_seat_order(self):
        hand = HandState(
            schema_version=self.settings.schema_version,
            hand_id="hand_000004",
            status="closed",
            player_count=5,
            table_format="5max",
            created_at="2026-04-03T02:50:12.430",
            updated_at="2026-04-03T02:50:20.671",
            last_seen_at="2026-04-03T02:50:20.671",
            hero_position="SB",
            hero_cards=["Td", "2c"],
            occupied_positions=["BTN", "SB", "BB", "UTG", "CO"],
            street_state={"current_street": "preflop", "street_history": ["preflop"]},
            player_states={
                "BTN": {"is_fold": True, "is_active": False},
                "SB": {"is_fold": False, "is_active": True},
                "BB": {"is_fold": False, "is_active": True},
                "UTG": {"is_fold": True, "is_active": False},
                "CO": {"is_fold": False, "is_active": True},
            },
            table_amount_state={"bets_by_position": {"CO": {"amount_bb": 2.5, "raw_text": "2.5"}}},
            action_state={"last_actions_by_position": {"CO": "OPEN 2.5"}},
        )
        render_state = build_render_state(hand, "frame_0040", "2026-04-03T02:50:12.430").to_dict()
        self.assertEqual(render_state["seat_order"], ["SB", "BB", "UTG", "CO", "BTN"])
        self.assertIn("BTN", render_state["players"])
        self.assertIn("UTG", render_state["players"])
        self.assertEqual(render_state["players"]["CO"]["current_bet_bb"], 2.5)
        self.assertEqual(render_state["players"]["CO"]["last_action"], "OPEN 2.5")

    def test_no_false_checks_are_inferred_from_silent_flop_frame(self):
        analysis = SimpleNamespace(
            frame_id="frame_0100",
            timestamp="2026-04-03T05:20:00.000",
            street="flop",
            player_count=6,
            occupied_positions=["BTN", "SB", "BB", "UTG", "MP", "CO"],
            player_states={
                "BTN": {"is_fold": False},
                "SB": {"is_fold": True},
                "BB": {"is_fold": True},
                "UTG": {"is_fold": True},
                "MP": {"is_fold": True},
                "CO": {"is_fold": False},
            },
            table_amount_state={"bets_by_position": {}},
        )
        result = infer_actions(None, analysis, self.settings)
        self.assertEqual(result["actions_this_frame"], [])
        self.assertEqual(result["last_actions_by_position"], {})

    def test_flop_bet_does_not_require_prior_synthetic_checks(self):
        previous = HandState(
            schema_version=self.settings.schema_version,
            hand_id="hand_000003",
            status="active",
            player_count=6,
            table_format="6max",
            created_at="2026-04-03T05:19:50.000",
            updated_at="2026-04-03T05:19:55.000",
            last_seen_at="2026-04-03T05:19:55.000",
            hero_position="BTN",
            hero_cards=["Kd", "9d"],
            occupied_positions=["BTN", "SB", "BB", "UTG", "MP", "CO"],
            street_state={"current_street": "flop", "street_history": ["preflop", "flop"]},
            player_states={
                "BTN": {"is_fold": False},
                "SB": {"is_fold": True},
                "BB": {"is_fold": True},
                "UTG": {"is_fold": True},
                "MP": {"is_fold": True},
                "CO": {"is_fold": False},
            },
            action_state={
                "street": "flop",
                "actor_order": ["CO", "BTN"],
                "street_commitments": {},
                "current_highest_commitment": 0.0,
                "last_aggressor_position": None,
                "acted_positions": [],
                "last_actions_by_position": {},
                "actions_this_frame": [],
            },
        )
        analysis = SimpleNamespace(
            frame_id="frame_0145",
            timestamp="2026-04-03T05:20:34.162",
            street="flop",
            player_count=6,
            occupied_positions=["BTN", "SB", "BB", "UTG", "MP", "CO"],
            player_states=previous.player_states,
            table_amount_state={"bets_by_position": {"CO": {"amount_bb": 2.5}}},
        )
        result = infer_actions(previous, analysis, self.settings)
        self.assertEqual(len(result["actions_this_frame"]), 1)
        self.assertEqual(result["actions_this_frame"][0]["position"], "CO")
        self.assertEqual(result["actions_this_frame"][0]["action"], "BET")
        self.assertEqual(result["last_actions_by_position"], {"CO": "BET 2.5"})

    def test_pipeline_creates_and_updates_hand(self):
        frame1 = self.source.next_frame()
        res1 = self.pipeline.process_frame(frame1)
        self.assertTrue(res1.analysis.active_hero_found)
        self.assertIsNotNone(res1.hand)
        self.assertEqual(res1.hand.hand_id, "hand_000001")
        self.assertEqual(res1.hand.street_state["current_street"], "preflop")
        self.assertIn("BTN", res1.hand.player_states)
        self.assertEqual(res1.hand.player_states["BTN"]["stack_bb"], 25.0)
        self.assertTrue(res1.hand.player_states["BB"]["is_fold"])
        self.assertEqual(res1.hand.table_amount_state["total_pot"]["amount_bb"], 1.5)
        self.assertEqual(res1.hand.table_amount_state["bets_by_position"]["UTG"]["amount_bb"], 3.0)
        self.assertEqual(res1.hand.actions_log[-1]["action"], "OPEN")

        self.source.next_frame()
        frame3 = self.source.next_frame()
        res3 = self.pipeline.process_frame(frame3)
        self.assertEqual(res3.hand.hand_id, "hand_000001")
        self.assertEqual(res3.hand.street_state["current_street"], "flop")
        self.assertEqual(res3.render_state["street"], "flop")
        self.assertIn("current_bet_bb", res3.render_state["players"]["BTN"])
        self.assertEqual(res3.render_state["players"]["BTN"]["last_action"], "BET 2.5")

    def test_matching_new_hand_after_card_change(self):
        frame1 = self.source.next_frame()
        res1 = self.pipeline.process_frame(frame1)
        self.assertEqual(res1.hand.hand_id, "hand_000001")
        for _ in range(7):
            frame = self.source.next_frame()
        res_new = self.pipeline.process_frame(frame)
        self.assertEqual(res_new.hand.hand_id, "hand_000002")
        self.assertEqual(res_new.hand.hero_cards, ["Ks", "Qs"])

    def test_same_hero_cards_do_not_create_new_hand_on_structure_noise(self):
        base = FrameAnalysis(
            frame_id="frame_0200",
            timestamp="2026-04-08T11:00:00.000",
            active_hero_found=True,
            street="preflop",
            player_count=6,
            table_format="6max",
            occupied_positions=["BTN", "SB", "BB", "UTG", "MP", "CO"],
            positions={
                "BTN": {"center": {"x": 900.0, "y": 850.0}},
                "SB": {"center": {"x": 1200.0, "y": 760.0}},
            },
            hero_position="BTN",
            hero_cards=["Ah", "Kd"],
            player_states={
                "BTN": {"is_fold": False, "stack_bb": 25.0},
                "SB": {"is_fold": False, "stack_bb": 25.0},
            },
            table_center=(960.0, 540.0),
            table_amount_state={"bets_by_position": {}},
            action_inference={"actions_this_frame": []},
        )
        hand, decision, created = self.hand_manager.update_or_create(base)
        self.assertTrue(created)
        self.assertEqual(hand.hand_id, "hand_000001")
        self.assertEqual(decision.status, "strong_match")

        noisy = FrameAnalysis(
            frame_id="frame_0201",
            timestamp="2026-04-08T11:00:01.000",
            active_hero_found=True,
            street="preflop",
            player_count=5,
            table_format="5max",
            occupied_positions=["BTN", "SB", "BB", "UTG", "CO"],
            positions={
                "BTN": {"center": {"x": 760.0, "y": 760.0}},
                "SB": {"center": {"x": 1270.0, "y": 690.0}},
            },
            hero_position="SB",
            hero_cards=["Kd", "Ah"],
            player_states={
                "BTN": {"is_fold": False, "stack_bb": 25.0},
                "SB": {"is_fold": False, "stack_bb": 25.0},
            },
            table_center=(1300.0, 900.0),
            table_amount_state={"bets_by_position": {}},
            action_inference={"actions_this_frame": []},
        )
        same_hand, noisy_decision, noisy_created = self.hand_manager.update_or_create(noisy)
        self.assertFalse(noisy_created)
        self.assertEqual(same_hand.hand_id, "hand_000001")
        self.assertEqual(noisy_decision.status, MATCH_WEAK_CONFLICT)
        self.assertEqual(noisy_decision.reason, "player_count changed; table_format changed; hero_position changed; occupied_positions changed; table center shifted by 495.2px; seat geometry changed")

    def test_render_state_written(self):
        frame1 = self.source.next_frame()
        res1 = self.pipeline.process_frame(frame1)
        render_path = self.tempdir / "hands" / res1.hand.hand_id / "render" / "last_render_state.json"
        hand_path = self.tempdir / "hands" / res1.hand.hand_id / "hand.json"
        player_crop_path = self.tempdir / "hands" / res1.hand.hand_id / "crops" / "players" / f"{frame1.frame_id}_BTN.png"
        amount_dir = self.tempdir / "hands" / res1.hand.hand_id / "crops" / "table_amount"
        self.assertTrue(render_path.exists())
        self.assertTrue(hand_path.exists())
        self.assertTrue(player_crop_path.exists())
        self.assertTrue(any(amount_dir.glob(f"{frame1.frame_id}_*.png")))

    def test_register_error_persists_to_hand_json(self):
        frame1 = self.source.next_frame()
        res1 = self.pipeline.process_frame(frame1)
        self.hand_manager.register_error(res1.hand, "unit_test", "something broke", frame1.frame_id, True)
        self.assertEqual(res1.hand.errors[-1]["stage"], "unit_test")
        self.assertEqual(res1.hand.errors[-1]["message"], "something broke")


if __name__ == "__main__":
    unittest.main()
