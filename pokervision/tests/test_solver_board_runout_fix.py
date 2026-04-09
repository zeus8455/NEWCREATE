
from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

from pokervision.solver_bridge import EngineBridge


@dataclass
class FakePostflopContext:
    hero_hand: list[str]
    board: list[str]
    pot_before_hero: float
    to_call: float
    effective_stack: float | None
    hero_position: str
    villain_positions: list[str]
    line_context: dict
    dead_cards: list[str]
    street: str
    player_count: int
    meta: dict = field(default_factory=dict)


class SolverBridgeBoardRunoutTests(unittest.TestCase):
    def setUp(self):
        self.bridge = EngineBridge(settings=None)
        self.hand = SimpleNamespace(hero_position="SB", player_count=6)
        self.analysis = SimpleNamespace()

    def test_flop_skips_villain_postflop_players_without_full_runout(self):
        context = FakePostflopContext(
            hero_hand=["Kh", "8s"],
            board=["4h", "5d", "Ad"],
            pot_before_hero=11.0,
            to_call=4.0,
            effective_stack=96.0,
            hero_position="SB",
            villain_positions=["MP", "CO"],
            line_context={"prior_aggression": True},
            dead_cards=[],
            street="flop",
            player_count=6,
            meta={"source": "test"},
        )

        calls = []

        def fake_solve(_context, **kwargs):
            calls.append(kwargs)
            return {"decision": "ok"}

        with patch.object(self.bridge, "build_postflop_context", return_value=context), \
             patch.object(self.bridge, "_hero_in_position_postflop", return_value=False), \
             patch.object(self.bridge, "_build_villain_preflop_spots", return_value=[{"name": "MP"}]), \
             patch.object(self.bridge, "_build_villain_postflop_players", return_value=[{"name": "MP"}]), \
             patch("pokervision.solver_bridge._solve_hero_postflop", side_effect=fake_solve):
            payload = self.bridge._build_postflop_recommendation(self.analysis, self.hand, ["Kh", "8s"], "flop")

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(len(calls), 1)
        self.assertIn("villain_preflop_spots", calls[0])
        self.assertNotIn("villain_postflop_players", calls[0])
        self.assertNotIn("warnings", payload)

    def test_river_uses_villain_postflop_players_with_full_runout(self):
        context = FakePostflopContext(
            hero_hand=["Kh", "8s"],
            board=["4h", "5d", "Ad", "7c", "2c"],
            pot_before_hero=45.5,
            to_call=11.5,
            effective_stack=82.5,
            hero_position="CO",
            villain_positions=["SB"],
            line_context={"prior_aggression": True},
            dead_cards=[],
            street="river",
            player_count=6,
            meta={"source": "test"},
        )

        calls = []

        def fake_solve(_context, **kwargs):
            calls.append(kwargs)
            return {"decision": "ok"}

        with patch.object(self.bridge, "build_postflop_context", return_value=context), \
             patch.object(self.bridge, "_hero_in_position_postflop", return_value=True), \
             patch.object(self.bridge, "_build_villain_preflop_spots", return_value=[{"name": "SB"}]), \
             patch.object(self.bridge, "_build_villain_postflop_players", return_value=[{"name": "SB"}]), \
             patch("pokervision.solver_bridge._solve_hero_postflop", side_effect=fake_solve):
            payload = self.bridge._build_postflop_recommendation(self.analysis, self.hand, ["Kh", "8s"], "river")

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(len(calls), 1)
        self.assertIn("villain_postflop_players", calls[0])
        self.assertNotIn("warnings", payload)


if __name__ == "__main__":
    unittest.main()
