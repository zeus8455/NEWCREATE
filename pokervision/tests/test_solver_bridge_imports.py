import sys
import types
import unittest

from pokervision import solver_bridge


class SolverBridgeImportTests(unittest.TestCase):
    def tearDown(self):
        for name in [
            "decision_types",
            "hero_decision",
            "pokervision.decision_types",
            "pokervision.hero_decision",
        ]:
            sys.modules.pop(name, None)

    def test_prefers_package_module_when_present(self):
        package_mod = types.ModuleType("pokervision.decision_types")
        package_mod.HeroDecision = object
        package_mod.PostflopContext = object
        package_mod.PreflopContext = object
        top_mod = types.ModuleType("decision_types")
        top_mod.HeroDecision = int
        top_mod.PostflopContext = int
        top_mod.PreflopContext = int
        sys.modules["pokervision.decision_types"] = package_mod
        sys.modules["decision_types"] = top_mod

        resolved = solver_bridge._import_bridge_module("decision_types")
        self.assertIs(resolved, package_mod)

    def test_falls_back_to_top_level_module(self):
        top_mod = types.ModuleType("decision_types")
        top_mod.HeroDecision = object
        top_mod.PostflopContext = object
        top_mod.PreflopContext = object
        sys.modules["decision_types"] = top_mod

        resolved = solver_bridge._import_bridge_module("decision_types")
        self.assertIs(resolved, top_mod)


if __name__ == "__main__":
    unittest.main()
