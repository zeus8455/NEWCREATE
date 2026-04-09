from types import SimpleNamespace
import unittest

from pokervision.action_inference import infer_actions


class DummySettings:
    infer_checks_without_explicit_evidence = False


def make_analysis(*, street, hero_cards, hero_position="SB", player_count=6, occupied_positions=None,
                  player_states=None, final_street=None, frame_id="f", timestamp="t"):
    if occupied_positions is None:
        occupied_positions = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
    if player_states is None:
        player_states = {pos: {"is_fold": False} for pos in occupied_positions}
    if final_street is None:
        final_street = {pos: 0.0 for pos in occupied_positions}
    return SimpleNamespace(
        street=street,
        hero_cards=list(hero_cards),
        hero_position=hero_position,
        player_count=player_count,
        occupied_positions=list(occupied_positions),
        player_states=player_states,
        amount_normalization={"final_contribution_street_bb_by_pos": dict(final_street)},
        table_amount_state={"bets_by_position": {}},
        frame_id=frame_id,
        timestamp=timestamp,
    )


def make_previous_hand(*, hero_cards, street="flop", commitments=None, last_actions=None, player_states=None):
    if commitments is None:
        commitments = {"SB": 8.5, "MP": 0.0}
    if last_actions is None:
        last_actions = {"SB": "BET 8.5"}
    if player_states is None:
        player_states = {
            "BTN": {"is_fold": True},
            "SB": {"is_fold": False},
            "BB": {"is_fold": True},
            "UTG": {"is_fold": True},
            "MP": {"is_fold": False},
            "CO": {"is_fold": False},
        }
    return SimpleNamespace(
        hero_cards=list(hero_cards),
        player_states=player_states,
        action_state={
            "street": street,
            "street_commitments": dict(commitments),
            "current_highest_commitment": max(commitments.values()) if commitments else 0.0,
            "acted_positions": [pos for pos, amount in commitments.items() if amount > 0],
            "last_aggressor_position": "SB",
            "last_actions_by_position": dict(last_actions),
        },
    )


class PostflopStateResetTests(unittest.TestCase):
    def test_new_hand_does_not_inherit_previous_postflop_commitments(self):
        previous_hand = make_previous_hand(hero_cards=["Kh", "8s"])
        analysis = make_analysis(
            street="flop",
            hero_cards=["7c", "3c"],
            hero_position="CO",
            player_states={
                "BTN": {"is_fold": True},
                "SB": {"is_fold": False},
                "BB": {"is_fold": True},
                "UTG": {"is_fold": True},
                "MP": {"is_fold": False},
                "CO": {"is_fold": False},
            },
            final_street={"BTN": 0.0, "SB": 0.0, "BB": 0.0, "UTG": 0.0, "MP": 4.0, "CO": 0.0},
        )
        result = infer_actions(previous_hand, analysis, DummySettings())
        self.assertFalse(result["same_hand_identity"])
        self.assertFalse(result["carried_previous_street_state"])
        self.assertEqual(result["current_highest_commitment"], 4.0)
        self.assertEqual(result["last_actions_by_position"], {"MP": "BET 4.0"})
        self.assertEqual(result["actions_this_frame"][0]["semantic_action"], "bet")
        self.assertEqual(result["actions_this_frame"][0]["action"], "BET")

    def test_same_hand_same_street_still_carries_state(self):
        previous_hand = make_previous_hand(hero_cards=["Kh", "8s"])
        analysis = make_analysis(
            street="flop",
            hero_cards=["Kh", "8s"],
            hero_position="SB",
            player_states={
                "BTN": {"is_fold": True},
                "SB": {"is_fold": False},
                "BB": {"is_fold": True},
                "UTG": {"is_fold": True},
                "MP": {"is_fold": False},
                "CO": {"is_fold": False},
            },
            final_street={"BTN": 0.0, "SB": 8.5, "BB": 0.0, "UTG": 0.0, "MP": 8.5, "CO": 0.0},
        )
        result = infer_actions(previous_hand, analysis, DummySettings())
        self.assertTrue(result["same_hand_identity"])
        self.assertTrue(result["carried_previous_street_state"])
        self.assertEqual(result["actions_this_frame"][0]["semantic_action"], "call")
        self.assertEqual(result["last_actions_by_position"]["SB"], "BET 8.5")
        self.assertEqual(result["last_actions_by_position"]["MP"], "CALL 8.5")


if __name__ == "__main__":
    unittest.main()
