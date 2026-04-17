
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _copy_dict(value: Any) -> Dict[str, Any]:
    return deepcopy(value) if isinstance(value, dict) else {}


def _copy_list(value: Any) -> List[Any]:
    return deepcopy(value) if isinstance(value, list) else []


def _trim_float(value: Any) -> Any:
    if isinstance(value, float):
        rounded = round(value, 4)
        if rounded.is_integer():
            return int(rounded)
        return rounded
    return value


def derive_amount_state(
    table_amount_state: Optional[Dict[str, Any]],
    amount_normalization: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    table_amount_state = table_amount_state or {}
    amount_normalization = amount_normalization or {}
    return {
        "street": amount_normalization.get("street"),
        "total_pot_bb": amount_normalization.get("total_pot_bb"),
        "forced_blinds_by_position": _copy_dict(amount_normalization.get("forced_blinds_by_position")),
        "visible_bets_by_position": _copy_dict(amount_normalization.get("visible_bets_by_position")),
        "final_contribution_bb_by_pos": _copy_dict(amount_normalization.get("final_contribution_bb_by_pos")),
        "final_contribution_street_bb_by_pos": _copy_dict(amount_normalization.get("final_contribution_street_bb_by_pos")),
        "posted_blinds": _copy_dict(table_amount_state.get("posted_blinds")),
        "bets_by_position": _copy_dict(table_amount_state.get("bets_by_position")),
        "blind_diagnostics": _copy_dict(amount_normalization.get("blind_diagnostics")),
        "warnings": [
            *[str(item) for item in table_amount_state.get("warnings", [])],
            *[str(item) for item in amount_normalization.get("warnings", [])],
        ],
        "errors": [
            *[str(item) for item in table_amount_state.get("errors", [])],
        ],
    }


def _normalize_count_field(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _resolve_preflop_display_nodes(
    action_state: Optional[Dict[str, Any]],
    advisor_input: Optional[Dict[str, Any]] = None,
) -> tuple[str, str, Optional[str]]:
    action_state = action_state or {}
    advisor_input = advisor_input or {}

    resolved_preflop = action_state.get("reconstructed_preflop")
    if not isinstance(resolved_preflop, dict):
        resolved_preflop = {}
    hero_preview = action_state.get("hero_context_preview")
    if not isinstance(hero_preview, dict):
        hero_preview = {}
    advisor_meta = advisor_input.get("meta")
    if not isinstance(advisor_meta, dict):
        advisor_meta = {}

    projection_node_type = (
        str(resolved_preflop.get("projection_node_type") or "").strip()
        or str(action_state.get("projection_node_type") or "").strip()
        or str(resolved_preflop.get("node_type") or "").strip()
        or str(hero_preview.get("node_type") or "").strip()
        or str(advisor_meta.get("projection_node_type") or "").strip()
        or str(advisor_input.get("projection_node_type") or "").strip()
        or str(action_state.get("node_type_preview") or "").strip()
        or str(advisor_input.get("node_type") or "").strip()
    )
    advisor_node_type = (
        str(resolved_preflop.get("advisor_node_type") or "").strip()
        or str(hero_preview.get("advisor_node_type") or "").strip()
        or str(action_state.get("advisor_node_type") or "").strip()
        or str(advisor_meta.get("advisor_node_type") or "").strip()
        or str(advisor_input.get("node_type") or "").strip()
        or projection_node_type
    )
    advisor_mapping_reason = (
        str(resolved_preflop.get("advisor_mapping_reason") or "").strip()
        or str(hero_preview.get("advisor_mapping_reason") or "").strip()
        or str(action_state.get("advisor_mapping_reason") or "").strip()
        or str(advisor_meta.get("advisor_mapping_reason") or "").strip()
        or None
    )
    return projection_node_type, advisor_node_type, advisor_mapping_reason


def derive_reconstructed_preflop(
    action_state: Optional[Dict[str, Any]],
    *,
    hero_position: Optional[str],
) -> Dict[str, Any]:
    action_state = action_state or {}
    action_history = [
        deepcopy(item)
        for item in (
            action_state.get("action_history_resolved")
            or action_state.get("action_history")
            or []
        )
        if isinstance(item, dict) and str(item.get("street") or "preflop") == "preflop"
    ]
    projection_node_type, advisor_node_type, advisor_mapping_reason = _resolve_preflop_display_nodes(action_state)
    return {
        "street": "preflop",
        "hero_position": hero_position,
        "node_type": projection_node_type,
        "projection_node_type": projection_node_type,
        "advisor_node_type": advisor_node_type,
        "advisor_mapping_reason": advisor_mapping_reason,
        "opener_pos": action_state.get("opener_pos"),
        "three_bettor_pos": action_state.get("three_bettor_pos"),
        "four_bettor_pos": action_state.get("four_bettor_pos"),
        "limpers": _normalize_count_field(action_state.get("limpers", 0)),
        "callers": _normalize_count_field(action_state.get("callers_after_open", 0)),
        "action_history": action_history,
        "final_contribution_bb_by_pos": _copy_dict(action_state.get("final_contribution_bb_by_pos")),
        "same_hand_identity": bool(action_state.get("same_hand_identity", False)),
    }


def derive_reconstructed_postflop(
    *,
    street: str,
    action_state: Optional[Dict[str, Any]],
    advisor_input: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    action_state = action_state or {}
    advisor_input = advisor_input or {}
    action_history = [
        deepcopy(item)
        for item in action_state.get("action_history", [])
        if isinstance(item, dict) and str(item.get("street") or "") not in {"", "preflop"}
    ]
    return {
        "street": street,
        "pot_before_hero": advisor_input.get("pot_before_hero"),
        "to_call": advisor_input.get("to_call"),
        "effective_stack": advisor_input.get("effective_stack"),
        "hero_position": advisor_input.get("hero_position"),
        "villain_positions": _copy_list(advisor_input.get("villain_positions")),
        "line_context": _copy_dict(advisor_input.get("line_context")),
        "action_history": action_history,
        "final_contribution_street_bb_by_pos": _copy_dict(action_state.get("final_contribution_street_bb_by_pos")),
    }


def _extract_engine_values(
    engine_result: Optional[Dict[str, Any]],
    solver_output: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    engine_result = engine_result or {}
    solver_output = solver_output or {}
    result_payload = solver_output.get("result") if isinstance(solver_output, dict) else None
    return {
        "action": engine_result.get("engine_action") or (result_payload or {}).get("engine_action"),
        "amount_to": engine_result.get("amount_to") if "amount_to" in engine_result else (result_payload or {}).get("amount_to"),
        "size_pct": engine_result.get("size_pct") if "size_pct" in engine_result else (result_payload or {}).get("size_pct"),
        "reason": engine_result.get("reason") or (result_payload or {}).get("reason"),
        "confidence": engine_result.get("confidence") if "confidence" in engine_result else (result_payload or {}).get("confidence"),
        "status": engine_result.get("status"),
    }


def _copy_range_trace(value: Any) -> Dict[str, Any]:
    return _copy_dict(value)


def _select_postflop_range_trace(
    advisor_input: Optional[Dict[str, Any]],
    solver_input: Optional[Dict[str, Any]],
    solver_output: Optional[Dict[str, Any]],
    hero_decision_debug: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    candidates = [
        (solver_output or {}).get("postflop_range_trace"),
        (solver_input or {}).get("postflop_range_trace"),
        (advisor_input or {}).get("postflop_range_trace"),
        ((hero_decision_debug or {}).get("postflop") if isinstance(hero_decision_debug, dict) else None),
        (solver_output or {}).get("runtime_range_state"),
        (solver_input or {}).get("runtime_range_state"),
        (advisor_input or {}).get("runtime_range_state"),
    ]
    for item in candidates:
        if isinstance(item, dict) and item:
            return _copy_range_trace(item)
    return {}


def _range_source_combo_count(source: Any) -> Optional[int]:
    if not isinstance(source, dict):
        return None
    weighted = source.get("weighted_combos")
    if isinstance(weighted, list):
        return len(weighted)
    meta = source.get("meta") if isinstance(source.get("meta"), dict) else {}
    value = meta.get("combo_count")
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _range_source_total_weight(source: Any) -> Optional[float]:
    if not isinstance(source, dict):
        return None
    weighted = source.get("weighted_combos")
    if isinstance(weighted, list):
        total = 0.0
        for item in weighted:
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    total += float(item[1])
            except Exception:
                continue
        return round(total, 4)
    meta = source.get("meta") if isinstance(source.get("meta"), dict) else {}
    for key in ("total_weight", "weight_total"):
        value = meta.get(key)
        if value not in (None, ""):
            try:
                return round(float(value), 4)
            except Exception:
                return None
    return None


def _report_combo_count(payload: Any) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    for key in ("combo_count", "kept_combo_count", "final_combo_count"):
        value = payload.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except Exception:
                pass
    weighted = payload.get("weighted_combos")
    if isinstance(weighted, list):
        return len(weighted)
    return None


def _report_total_weight(payload: Any) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    for key in ("total_weight", "kept_total_weight", "final_total_weight"):
        value = payload.get(key)
        if value not in (None, ""):
            try:
                return round(float(value), 4)
            except Exception:
                pass
    weighted = payload.get("weighted_combos")
    if isinstance(weighted, list):
        total = 0.0
        for item in weighted:
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    total += float(item[1])
            except Exception:
                continue
        return round(total, 4)
    return None


def _format_combo_delta(before_count: Optional[int], after_count: Optional[int]) -> str:
    if before_count is None and after_count is None:
        return ""
    if before_count is None:
        return f"→ {after_count}c"
    if after_count is None:
        return f"{before_count}c"
    delta = after_count - before_count
    if delta == 0:
        return f"{before_count}c → {after_count}c"
    sign = "+" if delta > 0 else ""
    return f"{before_count}c → {after_count}c ({sign}{delta})"


def _build_range_trace_summary(trace: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(trace, dict) or not trace:
        return {"route_lines": [], "range_debug": [], "sections": []}

    route_lines: List[str] = []
    sections: List[Dict[str, Any]] = []
    range_debug: List[Dict[str, Any]] = []

    requested_mode = str(trace.get("requested_range_build_mode") or "").strip()
    actual_mode = str(trace.get("range_build_mode") or "").strip()
    payload_kind = str(trace.get("payload_kind") or "").strip()
    resolved_street = str(trace.get("resolved_street") or "").strip()
    contract = str(trace.get("range_contract") or "").strip()
    trace_expected = trace.get("range_trace_expected")

    if requested_mode:
        route_lines.append(f"Requested: {requested_mode}")
    if actual_mode:
        route_lines.append(f"Route: {actual_mode}")
    if payload_kind:
        route_lines.append(f"Payload: {payload_kind}")
    if resolved_street:
        route_lines.append(f"Street: {resolved_street.upper()}")
    if contract:
        route_lines.append(f"Contract: {contract}")
    if trace_expected is not None:
        route_lines.append(f"Trace expected: {'yes' if bool(trace_expected) else 'no'}")
    for warning in [str(item) for item in trace.get("warnings", []) if item not in (None, "")][:2]:
        route_lines.append(f"Warning: {warning}")
    if route_lines:
        sections.append({"title": "Range trace", "lines": route_lines[:8]})

    summary_items = trace.get("villain_sources_summary")
    if not isinstance(summary_items, list):
        summary_items = []
    for item in summary_items:
        if not isinstance(item, dict):
            continue
        actor = str(item.get("name") or item.get("villain_pos") or "Villain")
        source_type = str(item.get("source_type") or item.get("range_source_type") or "range")
        combo_count = item.get("combo_count")
        total_weight = item.get("total_weight")
        line = f"{actor}: {source_type}"
        if combo_count not in (None, ""):
            line += f" | {combo_count}c"
        if total_weight not in (None, ""):
            line += f" | w={total_weight}"
        range_debug.append({
            "name": actor,
            "source_type": source_type,
            "combo_count": combo_count,
            "total_weight": total_weight,
            "normalized_expr": item.get("normalized_expr"),
            "raw_expr": item.get("raw_expr"),
            "expr": item.get("normalized_expr") or item.get("raw_expr") or line,
        })

    reports = trace.get("villain_range_reports") or trace.get("villain_reports") or []
    if not isinstance(reports, list):
        reports = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        actor = str(report.get("name") or report.get("villain_pos") or "Villain")
        source = report.get("range_source") if isinstance(report.get("range_source"), dict) else {}
        final_source = report.get("final_range_source") if isinstance(report.get("final_range_source"), dict) else source
        report_payload = report.get("report") if isinstance(report.get("report"), dict) else {}
        steps = report_payload.get("steps") if isinstance(report_payload.get("steps"), list) else report.get("steps")
        if not isinstance(steps, list):
            steps = []

        start_count = _range_source_combo_count(source)
        final_count = _range_source_combo_count(final_source)
        start_weight = _range_source_total_weight(source)
        final_weight = _range_source_total_weight(final_source)
        villain_lines: List[str] = []
        start_type = str(source.get("source_type") or report.get("source_type") or "range")
        final_type = str(final_source.get("source_type") or start_type or "range")
        base = f"Start: {start_type}"
        if start_count is not None:
            base += f" | {start_count}c"
        if start_weight is not None:
            base += f" | w={start_weight}"
        villain_lines.append(base)
        for step in steps[:4]:
            if not isinstance(step, dict):
                continue
            step_street = str(step.get("street") or resolved_street or "").upper()
            action = str(step.get("action") or step.get("semantic_action") or "action").upper()
            bet_pct = step.get("bet_pct_pot")
            before_source = step.get("range_before_source") if isinstance(step.get("range_before_source"), dict) else {}
            after_source = step.get("range_after_source") if isinstance(step.get("range_after_source"), dict) else {}
            before_report = step.get("range_before") if isinstance(step.get("range_before"), dict) else {}
            after_report = step.get("range_after") if isinstance(step.get("range_after"), dict) else {}
            before_count = _range_source_combo_count(before_source)
            if before_count is None:
                before_count = _report_combo_count(before_report)
            after_count = _range_source_combo_count(after_source)
            if after_count is None:
                after_count = _report_combo_count(after_report)
            detail = f"{step_street} {action}"
            if bet_pct not in (None, ""):
                try:
                    detail += f" {round(float(bet_pct), 1)}%"
                except Exception:
                    detail += f" {bet_pct}%"
            delta = _format_combo_delta(before_count, after_count)
            if delta:
                detail += f" | {delta}"
            villain_lines.append(detail)
        final_line = f"Final: {final_type}"
        if final_count is not None:
            final_line += f" | {final_count}c"
        if final_weight is not None:
            final_line += f" | w={final_weight}"
        villain_lines.append(final_line)
        sections.append({"title": f"Range {actor}", "lines": villain_lines[:8]})

        if not any(str(item.get("name") or item.get("actor") or item.get("position") or "") == actor for item in range_debug):
            range_debug.append({
                "name": actor,
                "source_type": final_type,
                "combo_count": final_count,
                "total_weight": final_weight,
                "normalized_expr": final_source.get("normalized_expr"),
                "raw_expr": final_source.get("raw_expr"),
                "expr": final_source.get("normalized_expr") or final_source.get("raw_expr") or final_line,
            })

    return {
        "route_lines": route_lines,
        "range_debug": range_debug,
        "sections": sections,
    }


def derive_analysis_panel(
    *,
    street: str,
    hero_position: Optional[str],
    occupied_positions: Optional[List[str]],
    action_state: Optional[Dict[str, Any]],
    advisor_input: Optional[Dict[str, Any]],
    solver_input: Optional[Dict[str, Any]],
    solver_output: Optional[Dict[str, Any]],
    engine_result: Optional[Dict[str, Any]],
    hero_decision_debug: Optional[Dict[str, Any]],
    solver_status: str,
    solver_warnings: Optional[List[str]],
    solver_errors: Optional[List[str]],
    solver_result_reused: bool,
    solver_reuse_reason: Optional[str],
    solver_fingerprint: Optional[str],
) -> Dict[str, Any]:
    action_state = action_state or {}
    advisor_input = advisor_input or {}
    solver_input = solver_input or {}
    solver_output = solver_output or {}
    engine_result = engine_result or {}
    hero_decision_debug = hero_decision_debug or {}
    engine_values = _extract_engine_values(engine_result, solver_output)
    villain_positions = (
        advisor_input.get("villain_positions")
        or solver_input.get("villain_positions")
        or []
    )
    projection_node_type, advisor_node_type, advisor_mapping_reason = _resolve_preflop_display_nodes(
        action_state,
        advisor_input,
    )
    display_node_type = projection_node_type or advisor_node_type or ""
    postflop_range_trace = _select_postflop_range_trace(
        advisor_input,
        solver_input,
        solver_output,
        hero_decision_debug,
    )
    range_trace_summary = _build_range_trace_summary(postflop_range_trace)
    range_debug = range_trace_summary.get("range_debug") or _copy_dict((hero_decision_debug.get("postflop") or hero_decision_debug.get("preflop") or {}))
    return {
        "street": street,
        "context_type": advisor_input.get("context_type") or solver_input.get("context_type"),
        "node_type": display_node_type,
        "projection_node_type": projection_node_type,
        "advisor_node_type": advisor_node_type,
        "advisor_mapping_reason": advisor_mapping_reason,
        "hero_position": hero_position,
        "villain_positions": list(villain_positions) if isinstance(villain_positions, list) else [],
        "player_count": len(list(occupied_positions or [])),
        "action_history": _copy_list(
            action_state.get("action_history_resolved") or action_state.get("action_history")
        )[-16:],
        "reconstructed_action_history": _copy_list(
            action_state.get("action_history_resolved") or action_state.get("action_history")
        )[-16:],
        "recommended_action": str(engine_values.get("action") or "").upper(),
        "recommended_amount_to": _trim_float(engine_values.get("amount_to")),
        "recommended_size_pct": _trim_float(engine_values.get("size_pct")),
        "decision_reason": engine_values.get("reason"),
        "decision_confidence": engine_values.get("confidence"),
        "engine_status": engine_result.get("status") or solver_status,
        "solver_status": solver_status,
        "solver_reused": bool(solver_result_reused),
        "solver_reuse_reason": solver_reuse_reason,
        "solver_fingerprint": solver_fingerprint,
        "solver_warnings": [str(item) for item in (solver_warnings or [])],
        "solver_errors": [str(item) for item in (solver_errors or [])],
        "postflop_range_trace": postflop_range_trace,
        "range_trace_lines": _copy_list(range_trace_summary.get("route_lines")),
        "range_debug": deepcopy(range_debug),
        "sections": _copy_list(range_trace_summary.get("sections")),
        "hero_decision_debug": _copy_dict(hero_decision_debug),
    }


def _compute_recommended_action(engine_result: Optional[Dict[str, Any]], solver_output: Optional[Dict[str, Any]]) -> str:
    value = _extract_engine_values(engine_result, solver_output).get("action")
    return str(value).upper() if value else ""


def _compute_recommended_amount_to(engine_result: Optional[Dict[str, Any]], solver_output: Optional[Dict[str, Any]]) -> Optional[float]:
    value = _extract_engine_values(engine_result, solver_output).get("amount_to")
    return None if value is None else float(value)


def _compute_recommended_size_pct(engine_result: Optional[Dict[str, Any]], solver_output: Optional[Dict[str, Any]]) -> Optional[float]:
    value = _extract_engine_values(engine_result, solver_output).get("size_pct")
    return None if value is None else float(value)


def _compute_node_type(action_state: Optional[Dict[str, Any]], advisor_input: Optional[Dict[str, Any]]) -> str:
    projection_node_type, advisor_node_type, _ = _resolve_preflop_display_nodes(action_state, advisor_input)
    value = projection_node_type or advisor_node_type or ""
    return str(value)


def _compute_engine_status(engine_result: Optional[Dict[str, Any]], solver_status: str) -> str:
    engine_result = engine_result or {}
    return str(engine_result.get("status") or solver_status or "not_run")


@dataclass(slots=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class Detection:
    label: str
    bbox: BBox
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return self.bbox.center

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class PlayerToken:
    label: str
    confidence: float
    bbox: Dict[str, float]
    x_sort_key: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlayerState:
    position: str
    is_fold: bool = False
    is_all_in: bool = False
    is_active: bool = True
    stack_text_raw: str = ""
    stack_bb: Optional[float] = None
    tokens: List[PlayerToken] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position,
            "is_fold": self.is_fold,
            "is_all_in": self.is_all_in,
            "is_active": self.is_active,
            "stack_text_raw": self.stack_text_raw,
            "stack_bb": self.stack_bb,
            "tokens": [token.to_dict() for token in self.tokens],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


@dataclass(slots=True)
class TableAmountState:
    total_pot: Dict[str, Any] = field(default_factory=dict)
    posted_blinds: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    bets_by_position: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    unassigned_chips: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_pot": dict(self.total_pot),
            "posted_blinds": {key: dict(value) for key, value in self.posted_blinds.items()},
            "bets_by_position": {key: dict(value) for key, value in self.bets_by_position.items()},
            "unassigned_chips": [dict(item) for item in self.unassigned_chips],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        return ValidationResult(
            ok=self.ok and other.ok,
            errors=[*self.errors, *other.errors],
            warnings=[*self.warnings, *other.warnings],
            meta={**self.meta, **other.meta},
        )


@dataclass(slots=True)
class FrameAnalysis:
    frame_id: str
    timestamp: str
    active_hero_found: bool
    structure_detections: List[Detection] = field(default_factory=list)
    hero_card_detections: List[Detection] = field(default_factory=list)
    board_card_detections: List[Detection] = field(default_factory=list)
    player_state_detections: Dict[str, List[Detection]] = field(default_factory=dict)
    table_amount_region_detections: List[Detection] = field(default_factory=list)
    table_amount_digit_detections: Dict[str, List[Detection]] = field(default_factory=dict)
    street: str = "preflop"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)
    player_count: Optional[int] = None
    table_format: Optional[str] = None
    occupied_positions: List[str] = field(default_factory=list)
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    hero_position: Optional[str] = None
    hero_cards: List[str] = field(default_factory=list)
    board_cards: List[str] = field(default_factory=list)
    player_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    table_center: Optional[tuple[float, float]] = None
    btn_detection: Optional[Detection] = None
    table_amount_state: Dict[str, Any] = field(default_factory=dict)
    amount_normalization: Dict[str, Any] = field(default_factory=dict)
    action_inference: Dict[str, Any] = field(default_factory=dict)
    reconstructed_preflop: Dict[str, Any] = field(default_factory=dict)
    reconstructed_postflop: Dict[str, Any] = field(default_factory=dict)
    advisor_input: Dict[str, Any] = field(default_factory=dict)
    solver_input: Dict[str, Any] = field(default_factory=dict)
    solver_output: Dict[str, Any] = field(default_factory=dict)
    engine_result: Dict[str, Any] = field(default_factory=dict)
    solver_context_preview: Dict[str, Any] = field(default_factory=dict)
    solver_result: Dict[str, Any] = field(default_factory=dict)
    solver_status: str = "not_run"
    solver_warnings: List[str] = field(default_factory=list)
    solver_errors: List[str] = field(default_factory=list)
    hero_decision_debug: Dict[str, Any] = field(default_factory=dict)
    solver_fingerprint_preview: Optional[str] = None
    solver_result_reused: bool = False
    solver_reuse_reason: Optional[str] = None
    processing_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        street = self.street or "preflop"
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "active_hero_found": self.active_hero_found,
            "structure_detections": [d.to_dict() for d in self.structure_detections],
            "hero_card_detections": [d.to_dict() for d in self.hero_card_detections],
            "board_card_detections": [d.to_dict() for d in self.board_card_detections],
            "player_state_detections": {
                position: [det.to_dict() for det in detections]
                for position, detections in self.player_state_detections.items()
            },
            "table_amount_region_detections": [d.to_dict() for d in self.table_amount_region_detections],
            "table_amount_digit_detections": {
                region_id: [det.to_dict() for det in detections]
                for region_id, detections in self.table_amount_digit_detections.items()
            },
            "street": street,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "validation": deepcopy(self.validation),
            "player_count": self.player_count,
            "table_format": self.table_format,
            "occupied_positions": list(self.occupied_positions),
            "positions": deepcopy(self.positions),
            "hero_position": self.hero_position,
            "hero_cards": list(self.hero_cards),
            "board_cards": list(self.board_cards),
            "player_states": deepcopy(self.player_states),
            "table_center": self.table_center,
            "btn_detection": self.btn_detection.to_dict() if self.btn_detection else None,
            "table_amount_state": deepcopy(self.table_amount_state),
            "amount_normalization": deepcopy(self.amount_normalization),
            "amount_state": derive_amount_state(self.table_amount_state, self.amount_normalization),
            "action_inference": deepcopy(self.action_inference),
            "reconstructed_preflop": deepcopy(self.reconstructed_preflop) or derive_reconstructed_preflop(
                self.action_inference,
                hero_position=self.hero_position,
            ),
            "reconstructed_postflop": deepcopy(self.reconstructed_postflop) or derive_reconstructed_postflop(
                street=street,
                action_state=self.action_inference,
                advisor_input=self.advisor_input or self.solver_context_preview,
            ),
            "advisor_input": deepcopy(self.advisor_input),
            "solver_input": deepcopy(self.solver_input),
            "solver_output": deepcopy(self.solver_output),
            "engine_result": deepcopy(self.engine_result),
            "solver_context_preview": deepcopy(self.solver_context_preview),
            "solver_result": deepcopy(self.solver_result),
            "solver_status": self.solver_status,
            "solver_warnings": list(self.solver_warnings),
            "solver_errors": list(self.solver_errors),
            "hero_decision_debug": deepcopy(self.hero_decision_debug),
            "solver_fingerprint_preview": self.solver_fingerprint_preview,
            "solver_result_reused": self.solver_result_reused,
            "solver_reuse_reason": self.solver_reuse_reason,
            "processing_summary": deepcopy(self.processing_summary),
        }


@dataclass(slots=True)
class HandError:
    timestamp: str
    stage: str
    message: str
    frame_id: Optional[str] = None
    fatal_for_frame: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HandState:
    schema_version: str
    hand_id: str
    status: str
    player_count: int
    table_format: str
    created_at: str
    updated_at: str
    last_seen_at: str
    hero_position: str
    hero_cards: List[str]
    occupied_positions: List[str]
    street_state: Dict[str, Any]
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    board_cards: List[str] = field(default_factory=list)
    player_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    frames_log: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    processing_summary: Dict[str, Any] = field(default_factory=dict)
    render_state_snapshot: Dict[str, Any] = field(default_factory=dict)
    conflict_state: Optional[str] = None
    table_center: Optional[tuple[float, float]] = None
    table_amount_state: Dict[str, Any] = field(default_factory=dict)
    amount_normalization: Dict[str, Any] = field(default_factory=dict)
    action_state: Dict[str, Any] = field(default_factory=dict)
    actions_log: List[Dict[str, Any]] = field(default_factory=list)
    reconstructed_preflop: Dict[str, Any] = field(default_factory=dict)
    reconstructed_postflop: Dict[str, Any] = field(default_factory=dict)
    advisor_input: Dict[str, Any] = field(default_factory=dict)
    solver_input: Dict[str, Any] = field(default_factory=dict)
    solver_output: Dict[str, Any] = field(default_factory=dict)
    engine_result: Dict[str, Any] = field(default_factory=dict)
    solver_context: Dict[str, Any] = field(default_factory=dict)
    solver_status: str = "not_run"
    solver_warnings: List[str] = field(default_factory=list)
    solver_errors: List[str] = field(default_factory=list)
    hero_decision_debug: Dict[str, Any] = field(default_factory=dict)
    solver_fingerprint: Optional[str] = None
    solver_result_reused: bool = False
    solver_reuse_reason: Optional[str] = None
    solver_run_frame_id: Optional[str] = None
    solver_run_timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        street = str(self.street_state.get("current_street") or "preflop")
        reconstructed_preflop = deepcopy(self.reconstructed_preflop) or derive_reconstructed_preflop(
            self.action_state,
            hero_position=self.hero_position,
        )
        reconstructed_postflop = deepcopy(self.reconstructed_postflop) or derive_reconstructed_postflop(
            street=street,
            action_state=self.action_state,
            advisor_input=self.advisor_input,
        )
        analysis_panel = derive_analysis_panel(
            street=street,
            hero_position=self.hero_position,
            occupied_positions=self.occupied_positions,
            action_state=self.action_state,
            advisor_input=self.advisor_input,
            solver_input=self.solver_input,
            solver_output=self.solver_output,
            engine_result=self.engine_result,
            hero_decision_debug=self.hero_decision_debug,
            solver_status=self.solver_status,
            solver_warnings=self.solver_warnings,
            solver_errors=self.solver_errors,
            solver_result_reused=self.solver_result_reused,
            solver_reuse_reason=self.solver_reuse_reason,
            solver_fingerprint=self.solver_fingerprint,
        )
        return {
            "schema_version": self.schema_version,
            "hand_id": self.hand_id,
            "status": self.status,
            "player_count": self.player_count,
            "table_format": self.table_format,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_seen_at": self.last_seen_at,
            "hero_position": self.hero_position,
            "hero_cards": list(self.hero_cards),
            "occupied_positions": list(self.occupied_positions),
            "street_state": deepcopy(self.street_state),
            "positions": deepcopy(self.positions),
            "board_cards": list(self.board_cards),
            "player_states": deepcopy(self.player_states),
            "frames_log": deepcopy(self.frames_log),
            "errors": deepcopy(self.errors),
            "artifacts": deepcopy(self.artifacts),
            "processing_summary": deepcopy(self.processing_summary),
            "render_state_snapshot": deepcopy(self.render_state_snapshot),
            "conflict_state": self.conflict_state,
            "table_center": self.table_center,
            "table_amount_state": deepcopy(self.table_amount_state),
            "amount_normalization": deepcopy(self.amount_normalization),
            "amount_state": derive_amount_state(self.table_amount_state, self.amount_normalization),
            "action_state": deepcopy(self.action_state),
            "actions_log": deepcopy(self.actions_log),
            "reconstructed_preflop": reconstructed_preflop,
            "reconstructed_postflop": reconstructed_postflop,
            "advisor_input": deepcopy(self.advisor_input),
            "solver_input": deepcopy(self.solver_input),
            "solver_output": deepcopy(self.solver_output),
            "engine_result": deepcopy(self.engine_result),
            "solver_context": deepcopy(self.solver_context),
            "solver_status": self.solver_status,
            "solver_warnings": list(self.solver_warnings),
            "solver_errors": list(self.solver_errors),
            "hero_decision_debug": deepcopy(self.hero_decision_debug),
            "solver_fingerprint": self.solver_fingerprint,
            "solver_result_reused": self.solver_result_reused,
            "solver_reuse_reason": self.solver_reuse_reason,
            "processing_summary": deepcopy(self.processing_summary),
            "solver_run_frame_id": self.solver_run_frame_id,
            "solver_run_timestamp": self.solver_run_timestamp,
            "analysis_panel": analysis_panel,
        }


@dataclass(slots=True)
class RenderState:
    hand_id: str
    player_count: int
    table_format: str
    street: str
    hero_position: str
    hero_cards: List[str]
    board_cards: List[str]
    players: Dict[str, Dict[str, Any]]
    status: str
    warnings: List[str]
    freshness: str
    source_frame_id: str
    source_timestamp: str
    updated_at: str
    seat_order: List[str] = field(default_factory=list)
    table_amount_state: Dict[str, Any] = field(default_factory=dict)
    amount_normalization: Dict[str, Any] = field(default_factory=dict)
    amount_state: Dict[str, Any] = field(default_factory=dict)
    action_annotations: Dict[str, Any] = field(default_factory=dict)
    reconstructed_preflop: Dict[str, Any] = field(default_factory=dict)
    reconstructed_postflop: Dict[str, Any] = field(default_factory=dict)
    advisor_input: Dict[str, Any] = field(default_factory=dict)
    solver_input: Dict[str, Any] = field(default_factory=dict)
    solver_output: Dict[str, Any] = field(default_factory=dict)
    engine_result: Dict[str, Any] = field(default_factory=dict)
    solver_context: Dict[str, Any] = field(default_factory=dict)
    solver_status: str = "not_run"
    solver_warnings: List[str] = field(default_factory=list)
    solver_errors: List[str] = field(default_factory=list)
    hero_decision_debug: Dict[str, Any] = field(default_factory=dict)
    solver_fingerprint: Optional[str] = None
    solver_result_reused: bool = False
    solver_reuse_reason: Optional[str] = None
    processing_summary: Dict[str, Any] = field(default_factory=dict)
    solver_run_frame_id: Optional[str] = None
    solver_run_timestamp: Optional[str] = None
    recommended_action: str = ""
    recommended_amount_to: Optional[float] = None
    recommended_size_pct: Optional[float] = None
    node_type: str = ""
    engine_status: str = "not_run"
    analysis_panel: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CapturedFrame:
    frame_id: str
    timestamp: str
    image: Any


@dataclass(slots=True)
class PipelineArtifacts:
    raw_frame_path: Optional[str] = None
    overlay_frame_path: Optional[str] = None
    hero_crop_path: Optional[str] = None
    hero_overlay_path: Optional[str] = None
    board_crop_path: Optional[str] = None
    board_overlay_path: Optional[str] = None
    player_crop_paths: Dict[str, str] = field(default_factory=dict)
    player_overlay_paths: Dict[str, str] = field(default_factory=dict)
    table_amount_crop_paths: Dict[str, str] = field(default_factory=dict)
    table_amount_overlay_paths: Dict[str, str] = field(default_factory=dict)
    debug_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_frame_path": self.raw_frame_path,
            "overlay_frame_path": self.overlay_frame_path,
            "hero_crop_path": self.hero_crop_path,
            "hero_overlay_path": self.hero_overlay_path,
            "board_crop_path": self.board_crop_path,
            "board_overlay_path": self.board_overlay_path,
            "player_crop_paths": dict(self.player_crop_paths),
            "player_overlay_paths": dict(self.player_overlay_paths),
            "table_amount_crop_paths": dict(self.table_amount_crop_paths),
            "table_amount_overlay_paths": dict(self.table_amount_overlay_paths),
            "debug_paths": list(self.debug_paths),
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="milliseconds")
