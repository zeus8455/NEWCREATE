from __future__ import annotations

import unittest
from dataclasses import dataclass

from pokervision.visualizer import _to_json_safe


@dataclass(slots=True)
class DummyRangeSource:
    name: str
    source_type: str
    weighted_combos: list[tuple[tuple[int, int], float]]


class VisualizerSerializationTests(unittest.TestCase):
    def test_to_json_safe_converts_dataclass_objects(self):
        value = {
            "villain_sources": [
                DummyRangeSource(
                    name="MP",
                    source_type="preflop_spot_action_range",
                    weighted_combos=[((1, 2), 1.0)],
                )
            ]
        }
        safe = _to_json_safe(value)
        self.assertIsInstance(safe, dict)
        self.assertEqual(safe["villain_sources"][0]["name"], "MP")
        self.assertEqual(safe["villain_sources"][0]["weighted_combos"][0][1], 1.0)


if __name__ == "__main__":
    unittest.main()
