
from types import SimpleNamespace
import unittest

from pokervision.solver_bridge import EngineBridge


class SolverBridgeExtractionTests(unittest.TestCase):
    def _make_hand(self, actions_log, player_count=5, hero_position="BTN"):
        return SimpleNamespace(
            player_count=player_count,
            hero_position=hero_position,
            occupied_positions=["BTN", "SB", "BB", "UTG", "CO"] if player_count == 5 else ["BTN", "BB"],
            player_states={pos: {"is_fold": False} for pos in (["BTN", "SB", "BB", "UTG", "CO"] if player_count == 5 else ["BTN", "BB"])},
            street_state={"current_street": "preflop"},
            actions_log=actions_log,
            table_amount_state={},
            board_cards=[],
            hand_id="hand_test",
            hero_cards=["As", "Kd"],
        )

    def test_build_hero_spot_open_call_3bet(self):
        bridge = EngineBridge(settings=None)
        hand = self._make_hand([
            {"street": "preflop", "position": "CO", "action": "OPEN", "amount_bb": 2.5},
            {"street": "preflop", "position": "BTN", "action": "CALL", "amount_bb": 2.5},
            {"street": "preflop", "position": "SB", "action": "RAISE", "semantic_action": "3bet", "amount_bb": 9.0},
        ])
        spot = bridge._build_hero_preflop_spot(hand)
        self.assertEqual(spot.node_type, "facing_open")
        self.assertEqual(spot.opener_pos, "CO")
        self.assertEqual(spot.callers, 0)

    def test_build_hero_spot_hu_btn_maps_to_sb(self):
        bridge = EngineBridge(settings=None)
        hand = SimpleNamespace(
            player_count=2,
            hero_position="BTN",
            occupied_positions=["BTN", "BB"],
            player_states={"BTN": {"is_fold": False}, "BB": {"is_fold": False}},
            street_state={"current_street": "preflop"},
            actions_log=[],
            table_amount_state={},
            board_cards=[],
            hand_id="hand_hu",
            hero_cards=["As", "Kd"],
        )
        spot = bridge._build_hero_preflop_spot(hand)
        self.assertEqual(spot.hero_pos, "SB")
        self.assertEqual(spot.node_type, "unopened")


if __name__ == "__main__":
    unittest.main()
