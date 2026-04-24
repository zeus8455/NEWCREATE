from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

try:
    import cv2
    import numpy as np
    from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
except Exception:  # pragma: no cover
    cv2 = np = None
    QtCore = QtGui = QtWidgets = None

from .card_renderer import render_card, render_card_back

CANONICAL_RING = {
    2: ["BTN", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["BTN", "SB", "BB", "CO"],
    5: ["BTN", "SB", "BB", "UTG", "CO"],
    6: ["BTN", "SB", "BB", "UTG", "MP", "CO"],
}

DISPLAY_SLOTS = {
    2: [(0.50, 0.84), (0.50, 0.18)],
    3: [(0.50, 0.84), (0.18, 0.46), (0.82, 0.22)],
    4: [(0.50, 0.84), (0.16, 0.64), (0.18, 0.24), (0.84, 0.28)],
    5: [(0.50, 0.84), (0.16, 0.68), (0.18, 0.28), (0.50, 0.14), (0.84, 0.30)],
    6: [(0.50, 0.84), (0.16, 0.70), (0.18, 0.30), (0.50, 0.13), (0.82, 0.30), (0.84, 0.70)],
}

PANEL_BG = (21, 24, 28)
PANEL_SECTION_BG = (30, 34, 40)
TEXT_PRIMARY = (245, 245, 245)
TEXT_MUTED = (184, 188, 195)
ACCENT = (73, 145, 255)
SUCCESS = (53, 168, 83)
WARNING = (194, 148, 59)
DANGER = (190, 74, 74)


def _coerce_lines(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [f"{key}: {value[key]}" for key in value]
    if isinstance(value, Iterable):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            out.append(str(item))
        return out
    return [str(value)]



def _shorten(text: Any, limit: int = 72) -> str:
    raw = "" if text is None else str(text)
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 1)] + "…"



def _format_amount(value: Any, suffix: str = "") -> str:
    if value in (None, ""):
        return "—"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if number.is_integer():
        text = str(int(number))
    else:
        text = f"{number:.2f}".rstrip("0").rstrip(".")
    return f"{text}{suffix}"



def _history_entries(render_state: dict, limit: int = 8) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    action_annotations = render_state.get("action_annotations", {})
    if isinstance(action_annotations, dict):
        for item in action_annotations.get("actions_log", []) or []:
            if isinstance(item, dict):
                entries.append(item)

    if not entries:
        for block_name in ("reconstructed_preflop", "reconstructed_postflop"):
            block = render_state.get(block_name, {})
            if isinstance(block, dict):
                for item in block.get("action_history", []) or []:
                    if isinstance(item, dict):
                        entries.append(item)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in entries:
        key = (
            item.get("street"),
            item.get("position") or item.get("pos"),
            item.get("semantic_action") or item.get("action"),
            item.get("amount_bb") or item.get("final_contribution_bb") or item.get("final_contribution_street_bb"),
            item.get("timestamp"),
            item.get("frame_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[-limit:]



def summarize_action_history(render_state: dict, limit: int = 8) -> list[str]:
    lines: list[str] = []
    for item in _history_entries(render_state, limit=limit):
        street = str(item.get("street", "")).upper()
        pos = str(item.get("position") or item.get("pos") or "?")
        action = str(item.get("semantic_action") or item.get("action") or item.get("engine_action") or "").upper()
        amount = item.get("amount_bb")
        if amount in (None, "", 0, 0.0):
            amount = item.get("final_contribution_street_bb")
        tail = ""
        if amount not in (None, ""):
            try:
                tail = f" {_format_amount(amount, 'bb')}"
            except Exception:
                tail = f" {amount}"
        lines.append(f"{street} {pos}: {action}{tail}".strip())
    return lines



def _extract_postflop_range_trace(render_state: dict) -> dict:
    analysis_panel = render_state.get("analysis_panel", {})
    if isinstance(analysis_panel, dict):
        trace = analysis_panel.get("postflop_range_trace")
        if isinstance(trace, dict) and trace:
            return trace
    for block_name in ("solver_output", "solver_input", "advisor_input"):
        block = render_state.get(block_name, {})
        if not isinstance(block, dict):
            continue
        for key in ("postflop_range_trace", "runtime_range_state"):
            trace = block.get(key)
            if isinstance(trace, dict) and trace:
                return trace
    return {}


def summarize_range_trace(render_state: dict, limit: int = 12) -> list[str]:
    trace = _extract_postflop_range_trace(render_state)
    if not trace:
        return []

    lines: list[str] = []
    requested = trace.get("requested_range_build_mode")
    route = trace.get("range_build_mode")
    payload_kind = trace.get("payload_kind")
    resolved_street = trace.get("resolved_street")
    if requested:
        lines.append(f"Requested: {requested}")
    if route:
        lines.append(f"Route: {route}")
    if payload_kind:
        lines.append(f"Payload: {payload_kind}")
    if resolved_street:
        lines.append(f"Street: {str(resolved_street).upper()}")

    for item in trace.get("villain_sources_summary", []) or []:
        if not isinstance(item, dict):
            continue
        actor = item.get("name") or item.get("villain_pos") or "Villain"
        source_type = item.get("source_type") or "range"
        combo_count = item.get("combo_count")
        total_weight = item.get("total_weight")
        line = f"{actor}: {source_type}"
        if combo_count not in (None, ""):
            line += f" | {combo_count}c"
        if total_weight not in (None, ""):
            line += f" | w={total_weight}"
        lines.append(line)

    reports = trace.get("villain_range_reports") or trace.get("villain_reports") or []
    for report in reports:
        if len(lines) >= limit:
            break
        if not isinstance(report, dict):
            continue
        actor = str(report.get("name") or report.get("villain_pos") or "Villain")
        payload = report.get("report") if isinstance(report.get("report"), dict) else {}
        steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
        if steps:
            first = steps[0]
            if isinstance(first, dict):
                street = str(first.get("street") or trace.get("resolved_street") or "").upper()
                action = str(first.get("action") or first.get("semantic_action") or "action").upper()
                before = first.get("range_before_source") or {}
                after = first.get("range_after_source") or {}
                before_count = len(before.get("weighted_combos", [])) if isinstance(before, dict) and isinstance(before.get("weighted_combos"), list) else None
                after_count = len(after.get("weighted_combos", [])) if isinstance(after, dict) and isinstance(after.get("weighted_combos"), list) else None
                detail = f"{actor} {street} {action}"
                if before_count is not None or after_count is not None:
                    detail += f" | {before_count if before_count is not None else '?'}c→{after_count if after_count is not None else '?'}c"
                lines.append(detail)

    return lines[:limit]


def summarize_range_debug(render_state: dict, limit: int = 8) -> list[str]:
    lines: list[str] = []
    analysis_panel = render_state.get("analysis_panel", {})
    if isinstance(analysis_panel, dict):
        for item in analysis_panel.get("range_debug", []) or []:
            if isinstance(item, dict):
                actor = item.get("name") or item.get("actor") or item.get("position") or "Villain"
                source_type = item.get("source_type") or item.get("kind") or "range"
                combo_count = item.get("combo_count")
                total_weight = item.get("total_weight")
                expr = item.get("normalized_expr") or item.get("raw_expr") or item.get("expr") or ""
                line = f"{actor} [{source_type}]"
                if combo_count not in (None, ""):
                    line += f" {combo_count}c"
                if total_weight not in (None, ""):
                    line += f" w={total_weight}"
                if expr:
                    line += f" {_shorten(expr, 40)}"
                lines.append(line)

    if not lines:
        lines.extend(summarize_range_trace(render_state, limit=limit))

    if not lines:
        solver_output = render_state.get("solver_output", {})
        result = solver_output.get("result", {}) if isinstance(solver_output, dict) else {}
        postflop = result.get("postflop", {}) if isinstance(result, dict) else {}
        for item in postflop.get("villain_sources", []) or []:
            if isinstance(item, dict):
                actor = item.get("name") or "Villain"
                source_type = item.get("source_type") or "range"
                expr = item.get("normalized_expr") or item.get("raw_expr") or ""
                lines.append(f"{actor} [{source_type}] {_shorten(expr, 56)}")

    return lines[:limit]



def build_analysis_panel_sections(render_state: dict, status: dict | None = None) -> list[dict[str, Any]]:
    status = {} if status is None else dict(status)
    analysis_panel = render_state.get("analysis_panel", {})
    hero_position = render_state.get("hero_position") or "—"
    street = str(render_state.get("street") or "preflop").upper()
    node_type = render_state.get("node_type") or (analysis_panel.get("node_type") if isinstance(analysis_panel, dict) else None) or "—"
    engine_status = render_state.get("engine_status") or status.get("engine_status") or "—"
    recommended = render_state.get("recommended_action") or "—"
    amount_to = render_state.get("recommended_amount_to")
    size_pct = render_state.get("recommended_size_pct")

    decision_lines = [f"Action: {recommended}"]
    if amount_to not in (None, ""):
        decision_lines.append(f"Amount to: {_format_amount(amount_to, 'bb')}")
    if size_pct not in (None, ""):
        decision_lines.append(f"Size: {_format_amount(size_pct, '%')}")

    context_lines = [
        f"Street: {street}",
        f"Hero: {hero_position}",
        f"Node: {node_type}",
        f"Engine: {engine_status}",
    ]

    advisor_input = render_state.get("advisor_input", {})
    if isinstance(advisor_input, dict):
        if advisor_input.get("pot_before_hero") not in (None, ""):
            context_lines.append(f"Pot before hero: {_format_amount(advisor_input.get('pot_before_hero'), 'bb')}")
        if advisor_input.get("to_call") not in (None, ""):
            context_lines.append(f"To call: {_format_amount(advisor_input.get('to_call'), 'bb')}")
        if advisor_input.get("effective_stack") not in (None, ""):
            context_lines.append(f"Effective stack: {_format_amount(advisor_input.get('effective_stack'), 'bb')}")

    solver_lines = []
    solver_status = render_state.get("solver_status") or status.get("solver_status")
    if solver_status:
        solver_lines.append(f"Status: {solver_status}")
    if render_state.get("solver_result_reused"):
        solver_lines.append("Reuse: yes")
    if render_state.get("solver_reuse_reason"):
        solver_lines.append(f"Reuse reason: {_shorten(render_state['solver_reuse_reason'], 56)}")
    if render_state.get("solver_fingerprint"):
        solver_lines.append(f"Fingerprint: {_shorten(render_state['solver_fingerprint'], 44)}")
    for warning in _coerce_lines(render_state.get("solver_warnings"))[:2]:
        solver_lines.append(f"Warning: {_shorten(warning, 56)}")
    for error in _coerce_lines(render_state.get("solver_errors"))[:2]:
        solver_lines.append(f"Error: {_shorten(error, 56)}")

    sections = [
        {"title": "Decision", "lines": decision_lines},
        {"title": "Context", "lines": context_lines},
        {"title": "Action history", "lines": summarize_action_history(render_state) or ["No actions reconstructed"]},
        {"title": "Ranges / debug", "lines": summarize_range_debug(render_state) or ["No range debug"]},
        {"title": "Range trace", "lines": summarize_range_trace(render_state) or ["No range trace"]},
        {"title": "Solver", "lines": solver_lines or ["No solver metadata"]},
    ]

    extra_sections = analysis_panel.get("sections") if isinstance(analysis_panel, dict) else None
    if isinstance(extra_sections, list):
        for section in extra_sections:
            if not isinstance(section, dict):
                continue
            title = str(section.get("title") or "Panel")
            lines = _coerce_lines(section.get("lines"))
            if lines:
                sections.append({"title": title, "lines": lines[:8]})
    return sections


class PokerTableWindow:  # pragma: no cover
    def __init__(self, shared_state, settings):
        if QtWidgets is None:
            raise RuntimeError("PySide6 is not installed; PokerTableWindow unavailable")
        self.shared_state = shared_state
        self.settings = settings
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("PokerVision Table Renderer")
        self.window.resize(1480, 920)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.window.update)
        self.timer.start(settings.ui_refresh_ms)
        self.window.paintEvent = self.paintEvent  # type: ignore[assignment]

    def show(self):
        self.window.show()

    def _ordered_positions(self, render_state: dict) -> List[str]:
        explicit = render_state.get("seat_order") or []
        if explicit:
            return [pos for pos in explicit if pos in render_state.get("players", {})]
        player_count = int(render_state.get("player_count") or 0)
        available = [pos for pos in CANONICAL_RING.get(player_count, []) if pos in render_state.get("players", {})]
        hero_position = render_state.get("hero_position")
        if hero_position in available:
            idx = available.index(hero_position)
            return available[idx:] + available[:idx]
        return available

    def _slot_to_point(self, rect, slot: tuple[float, float]) -> tuple[int, int]:
        return (
            rect.left() + int(slot[0] * rect.width()),
            rect.top() + int(slot[1] * rect.height()),
        )

    def _player_brush(self, player_payload, is_hero):
        if player_payload.get("is_fold"):
            return QtGui.QBrush(QtGui.QColor(62, 62, 62))
        if player_payload.get("is_all_in"):
            return QtGui.QBrush(QtGui.QColor(*DANGER))
        if is_hero:
            return QtGui.QBrush(QtGui.QColor(176, 148, 58))
        return QtGui.QBrush(QtGui.QColor(52, 52, 52))

    def _draw_badge(self, painter, x: int, y: int, text: str, bg: QtGui.QColor, width: int = 36):
        painter.setPen(QtGui.QPen(QtGui.QColor(*TEXT_PRIMARY), 1))
        painter.setBrush(QtGui.QBrush(bg))
        painter.drawRoundedRect(x, y, width, 18, 8, 8)
        painter.drawText(QtCore.QRect(x, y, width, 18), QtCore.Qt.AlignmentFlag.AlignCenter, text)

    def _pixmap_from_bgr(self, image: "np.ndarray"):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        return QtGui.QPixmap.fromImage(qimg.copy())

    def _draw_card_pixmap(self, painter, image: "np.ndarray", x: int, y: int):
        painter.drawPixmap(x, y, self._pixmap_from_bgr(image))

    def _player_card_y(self, rect, py: int, box_y: int, box_h: int, card_h: int) -> int:
        if py >= rect.top() + int(rect.height() * 0.58):
            return box_y - card_h - 16
        return box_y + box_h + 12

    def _draw_player(self, painter, table_rect, pos_name: str, payload: dict, center_xy: Tuple[int, int], blind_text: str | None = None):
        px, py = center_xy
        box_w, box_h = 178, 106
        box_x = px - box_w // 2
        box_y = py - box_h // 2
        is_hero = payload.get("is_hero", False)

        painter.setBrush(self._player_brush(payload, is_hero))
        painter.setPen(QtGui.QPen(QtGui.QColor(*TEXT_PRIMARY), 2))
        painter.drawRoundedRect(box_x, box_y, box_w, box_h, 14, 14)

        painter.setPen(QtGui.QColor(*TEXT_PRIMARY))
        painter.drawText(box_x + 10, box_y + 20, pos_name)
        if is_hero:
            self._draw_badge(painter, box_x + box_w - 86, box_y + 8, "HERO", QtGui.QColor(120, 88, 10), width=46)
        if payload.get("is_button"):
            self._draw_badge(painter, box_x + box_w - 36, box_y + 8, "D", QtGui.QColor(70, 110, 170), width=24)
        if blind_text:
            self._draw_badge(painter, box_x + 10, box_y - 20, blind_text, QtGui.QColor(95, 115, 55), width=max(34, 10 + len(blind_text) * 6))

        stack_text = "—"
        if payload.get("stack_bb") is not None:
            stack_text = f"{float(payload['stack_bb']):.1f} BB"
        elif payload.get("stack_text_raw"):
            stack_text = f"{payload['stack_text_raw']} BB"
        painter.drawText(box_x + 10, box_y + 42, f"Stack: {stack_text}")

        bet_text = "—"
        if payload.get("current_bet_bb") is not None:
            bet_text = f"{float(payload['current_bet_bb']):.1f} BB"
        elif payload.get("current_bet_raw"):
            bet_text = f"{payload['current_bet_raw']} BB"
        painter.drawText(box_x + 10, box_y + 62, f"Bet: {bet_text}")

        status_parts: list[str] = []
        if payload.get("is_fold"):
            status_parts.append("FOLD")
        if payload.get("is_all_in"):
            status_parts.append("ALL-IN")
        if payload.get("last_action"):
            status_parts.append(str(payload["last_action"]))
        warnings = payload.get("state_warnings", [])
        if warnings:
            status_parts.append(str(warnings[0]))
        status_text = " | ".join(status_parts)[:42]
        if status_text:
            painter.drawText(box_x + 10, box_y + 84, status_text)

        if payload.get("show_card_backs") and not payload.get("is_fold"):
            back = render_card_back()
            card_y = self._player_card_y(table_rect, py, box_y, box_h, back.shape[0])
            self._draw_card_pixmap(painter, back, box_x + 58, card_y)
            self._draw_card_pixmap(painter, back, box_x + 98, card_y)
        return {"box_x": box_x, "box_y": box_y, "box_w": box_w, "box_h": box_h, "center_x": px, "center_y": py}

    def _draw_action_banner(self, painter, render_state: dict, table_rect) -> None:
        recommended = str(render_state.get("recommended_action") or "WAIT")
        node_type = str(render_state.get("node_type") or "—")
        amount_to = render_state.get("recommended_amount_to")
        size_pct = render_state.get("recommended_size_pct")
        engine_status = str(render_state.get("engine_status") or "—")

        banner = QtCore.QRect(table_rect.left(), 20, min(520, table_rect.width() - 20), 84)
        painter.setPen(QtGui.QPen(QtGui.QColor(*TEXT_PRIMARY), 1))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(18, 22, 26, 232)))
        painter.drawRoundedRect(banner, 16, 16)

        painter.setPen(QtGui.QColor(*TEXT_PRIMARY))
        headline_font = painter.font()
        headline_font.setPointSize(22)
        headline_font.setBold(True)
        painter.setFont(headline_font)
        painter.drawText(banner.adjusted(14, 8, -14, -36), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, recommended)

        body_font = painter.font()
        body_font.setPointSize(10)
        body_font.setBold(False)
        painter.setFont(body_font)
        tail = []
        if amount_to not in (None, ""):
            tail.append(f"amount_to={_format_amount(amount_to, 'bb')}")
        if size_pct not in (None, ""):
            tail.append(f"size={_format_amount(size_pct, '%')}")
        tail_text = " | ".join(tail)
        painter.drawText(banner.adjusted(14, 40, -14, -14), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, f"node_type={node_type} | engine={engine_status}{(' | ' + tail_text) if tail_text else ''}")

    def _draw_side_panel(self, painter, panel_rect, render_state: dict, status: dict) -> None:
        painter.setPen(QtGui.QPen(QtGui.QColor(50, 54, 61), 1))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(*PANEL_BG)))
        painter.drawRoundedRect(panel_rect, 18, 18)

        title_font = painter.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QtGui.QColor(*TEXT_PRIMARY))
        painter.drawText(panel_rect.adjusted(16, 12, -16, -12), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop, "Analysis panel")

        body_font = painter.font()
        body_font.setPointSize(9)
        body_font.setBold(False)
        sections = build_analysis_panel_sections(render_state, status)
        y = panel_rect.top() + 42
        for section in sections:
            lines = section.get("lines", [])
            if not lines:
                continue
            block_h = 36 + min(len(lines), 8) * 16
            section_rect = QtCore.QRect(panel_rect.left() + 12, y, panel_rect.width() - 24, block_h)
            painter.setPen(QtGui.QPen(QtGui.QColor(56, 61, 70), 1))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(*PANEL_SECTION_BG)))
            painter.drawRoundedRect(section_rect, 12, 12)

            painter.setFont(title_font)
            painter.setPen(QtGui.QColor(*ACCENT))
            painter.drawText(section_rect.adjusted(12, 8, -12, -8), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop, str(section.get("title") or "Section"))

            painter.setFont(body_font)
            painter.setPen(QtGui.QColor(*TEXT_PRIMARY))
            line_y = section_rect.top() + 28
            for line in lines[:8]:
                text_rect = QtCore.QRect(section_rect.left() + 12, line_y, section_rect.width() - 24, 15)
                painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, _shorten(line, 66))
                line_y += 16
            y += block_h + 10
            if y > panel_rect.bottom() - 60:
                break

    def paintEvent(self, event):
        painter = QtGui.QPainter(self.window)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = self.window.rect()
            painter.fillRect(rect, QtGui.QColor(19, 24, 19))

            panel_width = 390
            panel_rect = QtCore.QRect(rect.width() - panel_width - 20, 20, panel_width, rect.height() - 40)
            table_outer = QtCore.QRect(20, 20, rect.width() - panel_width - 60, rect.height() - 40)
            table_rect = QtCore.QRect(table_outer.left() + 20, table_outer.top() + 80, table_outer.width() - 40, table_outer.height() - 110)

            painter.setBrush(QtGui.QBrush(QtGui.QColor(24, 36, 24)))
            painter.setPen(QtGui.QPen(QtGui.QColor(40, 52, 40), 2))
            painter.drawRoundedRect(table_outer, 18, 18)

            painter.setBrush(QtGui.QBrush(QtGui.QColor(30, 92, 36)))
            painter.setPen(QtGui.QPen(QtGui.QColor(220, 220, 220), 3))
            painter.drawEllipse(table_rect)

            _, render_state, status = self.shared_state.snapshot()
            if not render_state:
                painter.setPen(QtGui.QColor(*TEXT_PRIMARY))
                painter.drawText(table_outer, QtCore.Qt.AlignmentFlag.AlignCenter, "Waiting for render_state")
                return

            self._draw_action_banner(painter, render_state, table_outer)
            self._draw_side_panel(painter, panel_rect, render_state, status)

            painter.setPen(QtGui.QColor(*TEXT_PRIMARY))
            meta_font = painter.font()
            meta_font.setPointSize(10)
            painter.setFont(meta_font)
            painter.drawText(table_outer.left() + 12, table_outer.bottom() - 14, f"Hand: {render_state.get('hand_id')} | Street: {render_state.get('street')} | Status: {render_state.get('status')} / {render_state.get('freshness')}")

            table_amount_state = render_state.get("table_amount_state", {}) if isinstance(render_state.get("table_amount_state", {}), dict) else {}
            total_pot = table_amount_state.get("total_pot", {}) if isinstance(table_amount_state, dict) else {}
            if total_pot.get("amount_bb") is not None:
                painter.drawText(table_outer.right() - 220, table_outer.top() + 60, f"Pot: {float(total_pot['amount_bb']):.1f} BB")

            blind_labels: Dict[str, str] = {}
            posted_blinds = table_amount_state.get("posted_blinds", {}) if isinstance(table_amount_state, dict) else {}
            for blind_name, payload in posted_blinds.items():
                pos = payload.get("matched_position") or blind_name
                amount = payload.get("amount_bb")
                blind_labels[pos] = blind_name if amount in (None, "") else f"{blind_name} {_format_amount(amount)}"

            player_count = int(render_state.get("player_count") or 0)
            ordered_positions = self._ordered_positions(render_state)
            slots = DISPLAY_SLOTS.get(player_count, DISPLAY_SLOTS[6])
            player_boxes = {}
            for pos_name, slot in zip(ordered_positions, slots):
                payload = render_state.get("players", {}).get(pos_name, {})
                center_xy = self._slot_to_point(table_rect, slot)
                player_boxes[pos_name] = self._draw_player(painter, table_rect, pos_name, payload, center_xy, blind_labels.get(pos_name))

            board_cards = render_state.get("board_cards", [])
            board_w = len(board_cards) * 82 - (8 if board_cards else 0)
            bx = table_rect.center().x() - board_w // 2
            by = table_rect.center().y() - 54
            for idx, card in enumerate(board_cards):
                self._draw_card_pixmap(painter, render_card(card), bx + idx * 82, by)

            hero_position = render_state.get("hero_position")
            hero_cards = render_state.get("hero_cards", [])
            hero_payload = render_state.get("players", {}).get(hero_position, {})
            hero_box = player_boxes.get(hero_position)
            if hero_cards and hero_box and not hero_payload.get("is_fold", False):
                card_h = 104
                total_w = len(hero_cards) * 82 - 8
                hx = hero_box["center_x"] - total_w // 2
                hy = hero_box["box_y"] - card_h - 18
                hy = max(table_outer.top() + 110, hy)
                for idx, card in enumerate(hero_cards):
                    self._draw_card_pixmap(painter, render_card(card), hx + idx * 82, hy)

            if getattr(self.settings, "show_last_action_labels", True):
                actions = summarize_action_history(render_state, limit=4)
                y = table_outer.bottom() - 92
                painter.setPen(QtGui.QColor(*TEXT_MUTED))
                for action_text in actions:
                    painter.drawText(table_outer.left() + 16, y, action_text)
                    y += 18
        finally:
            if painter.isActive():
                painter.end()
