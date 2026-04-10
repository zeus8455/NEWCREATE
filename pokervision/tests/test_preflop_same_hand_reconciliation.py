from __future__ import annotations

import types
import unittest

from pokervision.preflop_reconstruction import (
    build_preflop_frame_observation,
    reconstruct_preflop_from_frame,
    reconcile_preflop_with_hand,
)


EPS = 1e-9


class DummySettings:
    infer_checks_without_explicit_evidence = False


CANONICAL_RING = {
    2: ["BTN", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["BTN", "SB", "BB", "CO"],
    5: ["BTN", "SB", "BB", "UTG", "CO"],
    6: ["BTN", "SB", "BB", "UTG", "MP", "CO"],
}

PREFLOP_ORDER = {
    2: ["BTN", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["CO", "BTN", "SB", "BB"],
    5: ["UTG", "CO", "BTN", "SB", "BB"],
    6: ["UTG", "MP", "CO", "BTN", "SB", "BB"],
}


def _safe_float(value, default=0.0):
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _approx_eq(a: float, b: float, eps: float = EPS) -> bool:
    return abs(a - b) <= eps


def _engine_action_for(semantic_action: str) -> str:
    if semantic_action in {"limp", "call"}:
        return "call"
    if semantic_action == "check":
        return "check"
    return "raise"


LEGACY_ACTION_BY_SEMANTIC = {
    "limp": "LIMP",
    "open_raise": "OPEN",
    "iso_raise": "OPEN",
    "call": "CALL",
    "3bet": "RAISE",
    "4bet": "RAISE",
    "cold_4bet": "RAISE",
    "5bet_jam": "RAISE",
    "check": "CHECK",
    "bet": "BET",
    "raise": "RAISE",
    "fold": "FOLD",
}


def _legacy_action_name(semantic_action: str, engine_action: str | None = None) -> str:
    semantic = str(semantic_action or "").lower()
    if semantic in LEGACY_ACTION_BY_SEMANTIC:
        return LEGACY_ACTION_BY_SEMANTIC[semantic]
    engine = str(engine_action or "").upper()
    return engine or semantic.upper()


def _format_amount_for_display(amount):
    value = float(amount)
    if abs(value - round(value)) <= EPS:
        return f"{value:.1f}"
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def _decorate_legacy_action_fields(event):
    out = dict(event)
    action = _legacy_action_name(out.get("semantic_action"), out.get("engine_action"))
    out.setdefault("action", action)
    amount = out.get("final_contribution_street_bb")
    if amount is None:
        amount = out.get("final_contribution_bb")
    if amount is None:
        amount = out.get("amount_bb")
    if amount is None or action in {"CHECK", "FOLD"}:
        out.setdefault("action_display", action)
    else:
        out.setdefault("action_display", f"{action} {_format_amount_for_display(amount)}")
    return out


def _legacy_last_action_display(event):
    decorated = _decorate_legacy_action_fields(event)
    return str(decorated.get("action_display") or decorated.get("action") or "")


def _final_aggression_label(raise_level: int):
    if raise_level <= 0:
        return None
    if raise_level == 1:
        return "open_raise"
    if raise_level == 2:
        return "3bet"
    if raise_level == 3:
        return "4bet"
    return "5bet_or_more"


def _player_is_folded(player_states, position: str) -> bool:
    return bool((player_states or {}).get(position, {}).get("is_fold", False))


def _forced_preflop_blinds(player_count: int, occupied_positions):
    positions = set(occupied_positions)
    if player_count == 2 or ({"BTN", "BB"} <= positions and "SB" not in positions):
        forced = {}
        if "BTN" in positions:
            forced["BTN"] = 0.5
        if "BB" in positions:
            forced["BB"] = 1.0
        return forced
    forced = {}
    if "SB" in positions:
        forced["SB"] = 0.5
    if "BB" in positions:
        forced["BB"] = 1.0
    return forced


def get_street_actor_order(player_count, street, occupied_positions, player_states, contributions=None):
    ring = [pos for pos in CANONICAL_RING.get(player_count, []) if pos in occupied_positions]
    if street == "preflop":
        ordered = [pos for pos in PREFLOP_ORDER.get(player_count, ring) if pos in ring]
        return [
            pos
            for pos in ordered
            if (not _player_is_folded(player_states, pos)) or _safe_float((contributions or {}).get(pos), 0.0) > EPS
        ]
    return ring


def _derive_node_type_preview(*, hero_position, player_count, limpers, opener_pos, three_bettor_pos, four_bettor_pos, callers_after_open, action_history):
    hero_pos = str(hero_position or "")
    if not hero_pos:
        return None
    max_raise_level = 0
    last_aggressive = None
    for step in action_history:
        semantic = str(step.get("semantic_action") or "")
        if semantic in {"open_raise", "iso_raise", "3bet", "4bet", "5bet_jam", "raise"}:
            last_aggressive = str(step.get("position") or step.get("pos") or "") or last_aggressive
            max_raise_level = max(max_raise_level, int(step.get("raise_level_after_action") or 0))
    if max_raise_level >= 4 and four_bettor_pos:
        if hero_pos == four_bettor_pos and last_aggressive and last_aggressive != hero_pos:
            return "fourbettor_vs_5bet"
        if hero_pos == three_bettor_pos and last_aggressive and last_aggressive != hero_pos:
            return "threebettor_vs_4bet"
    if four_bettor_pos:
        if hero_pos == three_bettor_pos:
            return "threebettor_vs_4bet"
        if hero_pos == four_bettor_pos and hero_pos not in {opener_pos, three_bettor_pos}:
            return "cold_4bet"
    if three_bettor_pos:
        if hero_pos == opener_pos:
            return "opener_vs_3bet"
        if opener_pos and hero_pos not in {opener_pos, three_bettor_pos}:
            return "facing_3bet"
    if opener_pos:
        if limpers and hero_pos in limpers:
            return "limper_vs_iso"
        if callers_after_open > 0:
            return "facing_open_callers"
        if hero_pos != opener_pos:
            return "facing_open"
        return "open_raise"
    return "unopened"


def _build_action_step(*, order, position, street, final_contribution_bb, semantic_action, current_price_to_call, raise_level, frame_id, timestamp, extra=None):
    payload = {
        "order": order,
        "position": position,
        "pos": position,
        "street": street,
        "final_contribution_bb": round(final_contribution_bb, 4),
        "amount_bb": round(final_contribution_bb, 4),
        "semantic_action": semantic_action,
        "engine_action": _engine_action_for(semantic_action),
        "raise_level_after_action": raise_level,
        "current_price_to_call_after_action": round(current_price_to_call, 4),
        "frame_id": frame_id,
        "timestamp": timestamp,
    }
    if extra:
        payload.update(extra)
    return _decorate_legacy_action_fields(payload)


def _build_preflop_resolved_ledger(*, hero_position, actor_order, action_history, final_contribution_bb_by_pos, final_contribution_street_bb_by_pos, current_price_to_call, opener_pos, three_bettor_pos, four_bettor_pos, limpers, callers_after_open, node_type_preview, source_mode, skipped_positions, same_hand_identity):
    resolved_history = [_decorate_legacy_action_fields(dict(item)) for item in list(action_history or [])]
    return {
        "street": "preflop",
        "source_mode": source_mode,
        "hero_position": hero_position,
        "node_type": node_type_preview,
        "opener_pos": opener_pos,
        "three_bettor_pos": three_bettor_pos,
        "four_bettor_pos": four_bettor_pos,
        "limpers": list(limpers),
        "limpers_count": len(limpers),
        "callers": callers_after_open,
        "callers_after_open": callers_after_open,
        "action_history": list(resolved_history),
        "action_history_resolved": list(resolved_history),
        "actor_order": list(actor_order),
        "current_price_to_call": round(current_price_to_call, 4),
        "last_aggressor_position": four_bettor_pos or three_bettor_pos or opener_pos,
        "final_contribution_bb_by_pos": {str(pos): round(float(val), 4) for pos, val in dict(final_contribution_bb_by_pos or {}).items()},
        "final_contribution_street_bb_by_pos": {str(pos): round(float(val), 4) for pos, val in dict(final_contribution_street_bb_by_pos or {}).items()},
        "hero_context_preview": {
            "hero_pos": hero_position,
            "node_type": node_type_preview,
            "opener_pos": opener_pos,
            "three_bettor_pos": three_bettor_pos,
            "four_bettor_pos": four_bettor_pos,
            "limpers": len(limpers),
            "callers": callers_after_open,
            "resolved": True,
            "projection_source": "reconstructed_preflop",
        },
        "skipped_positions": list(skipped_positions),
        "same_hand_identity": bool(same_hand_identity),
    }


class PreflopSameHandReconciliationTests(unittest.TestCase):
    def _analysis(self, *, frame_id, hero_cards, player_count=6, occupied_positions=None, hero_position="UTG", contributions=None, player_states=None):
        positions = occupied_positions or ["BTN", "SB", "BB", "UTG", "MP", "CO"]
        return types.SimpleNamespace(
            street="preflop",
            player_count=player_count,
            occupied_positions=list(positions),
            hero_position=hero_position,
            hero_cards=list(hero_cards),
            player_states=player_states or {pos: {"is_fold": False} for pos in positions},
            amount_state={
                "source_mode": "forced_blinds_plus_visible_chips",
                "final_contribution_bb_by_pos": dict(contributions or {}),
                "final_contribution_street_bb_by_pos": dict(contributions or {}),
            },
            amount_normalization={
                "source_mode": "forced_blinds_plus_visible_chips",
                "final_contribution_bb_by_pos": dict(contributions or {}),
                "final_contribution_street_bb_by_pos": dict(contributions or {}),
            },
            frame_id=frame_id,
            timestamp=f"2026-04-10T{frame_id[-4:-2]}:{frame_id[-2:]}:00Z",
            table_amount_state={},
        )

    def _run(self, previous_hand, analysis):
        observation = build_preflop_frame_observation(
            analysis,
            settings=DummySettings(),
            safe_float=_safe_float,
            forced_preflop_blinds=_forced_preflop_blinds,
            get_street_actor_order=get_street_actor_order,
        )
        frame_result = reconstruct_preflop_from_frame(
            observation,
            previous_hand=previous_hand,
            analysis=analysis,
            settings=DummySettings(),
            approx_eq=_approx_eq,
            build_action_step=_build_action_step,
            decorate_legacy_action_fields=_decorate_legacy_action_fields,
            legacy_last_action_display=_legacy_last_action_display,
            final_aggression_label=_final_aggression_label,
            same_hand_identity=lambda prev, current: bool(prev) and tuple(sorted(getattr(prev, "hero_cards", []))) == tuple(sorted(getattr(current, "hero_cards", []))),
            player_is_folded=_player_is_folded,
            eps=EPS,
        )
        return reconcile_preflop_with_hand(
            previous_hand,
            frame_result,
            analysis=analysis,
            settings=DummySettings(),
            safe_float=_safe_float,
            derive_node_type_preview=_derive_node_type_preview,
            build_preflop_resolved_ledger=_build_preflop_resolved_ledger,
            build_action_step=_build_action_step,
        )

    def test_runtime_escalation_open_3bet_4bet_5bet(self):
        a1 = self._analysis(
            frame_id="frame_0001",
            hero_cards=["Ac", "Qs"],
            contributions={"BTN": 0.0, "SB": 0.5, "BB": 1.0, "UTG": 2.0, "MP": 6.5, "CO": 0.0},
        )
        first = self._run(None, a1)
        previous_hand = types.SimpleNamespace(hero_cards=["Ac", "Qs"], action_state=first, reconstructed_preflop=first["reconstructed_preflop"])
        a2 = self._analysis(
            frame_id="frame_0014",
            hero_cards=["Ac", "Qs"],
            contributions={"BTN": 0.0, "SB": 0.5, "BB": 1.0, "UTG": 11.0, "MP": 29.0, "CO": 0.0},
        )
        second = self._run(previous_hand, a2)
        line = [(step["position"], step["semantic_action"], step["final_contribution_bb"]) for step in second["action_history_resolved"]]
        self.assertEqual(
            line,
            [
                ("UTG", "open_raise", 2.0),
                ("MP", "3bet", 6.5),
                ("UTG", "4bet", 11.0),
                ("MP", "5bet_jam", 29.0),
            ],
        )
        self.assertTrue(second["reconciliation"]["applied"])
        self.assertEqual(second["projection_node_type"], "fourbettor_vs_5bet")
        self.assertEqual(second["advisor_node_type"], "threebettor_vs_4bet")
        self.assertEqual(second["advisor_four_bettor_pos"], "MP")

    def test_growth_of_existing_aggressor_does_not_create_new_open(self):
        a1 = self._analysis(
            frame_id="frame_0001",
            hero_cards=["Ah", "Kd"],
            contributions={"BTN": 0.0, "SB": 0.5, "BB": 1.0, "UTG": 2.5, "MP": 8.0, "CO": 0.0},
        )
        first = self._run(None, a1)
        previous_hand = types.SimpleNamespace(hero_cards=["Ah", "Kd"], action_state=first, reconstructed_preflop=first["reconstructed_preflop"])
        a2 = self._analysis(
            frame_id="frame_0002",
            hero_cards=["Ah", "Kd"],
            contributions={"BTN": 0.0, "SB": 0.5, "BB": 1.0, "UTG": 18.0, "MP": 8.0, "CO": 0.0},
        )
        second = self._run(previous_hand, a2)
        actions = [step["semantic_action"] for step in second["action_history_resolved"]]
        self.assertEqual(actions, ["open_raise", "3bet", "4bet"])
        self.assertEqual(second["four_bettor_pos"], "UTG")
        self.assertNotIn(("UTG", "open_raise", 18.0), [(s["position"], s["semantic_action"], s["final_contribution_bb"]) for s in second["action_history_resolved"]])

    def test_same_hand_without_growth_keeps_history(self):
        a1 = self._analysis(
            frame_id="frame_0001",
            hero_cards=["9c", "9d"],
            contributions={"BTN": 0.0, "SB": 0.5, "BB": 1.0, "UTG": 2.0, "MP": 6.0, "CO": 0.0},
        )
        first = self._run(None, a1)
        previous_hand = types.SimpleNamespace(hero_cards=["9c", "9d"], action_state=first, reconstructed_preflop=first["reconstructed_preflop"])
        a2 = self._analysis(
            frame_id="frame_0002",
            hero_cards=["9c", "9d"],
            contributions={"BTN": 0.0, "SB": 0.5, "BB": 1.0, "UTG": 2.0, "MP": 6.0, "CO": 0.0},
        )
        second = self._run(previous_hand, a2)
        self.assertFalse(second["reconciliation"]["applied"])
        self.assertEqual(
            [(step["position"], step["semantic_action"]) for step in second["action_history_resolved"]],
            [("UTG", "open_raise"), ("MP", "3bet")],
        )


if __name__ == "__main__":
    unittest.main()
