from __future__ import annotations

from typing import Dict, List, Tuple

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


class PokerTableWindow:  # pragma: no cover
    def __init__(self, shared_state, settings):
        if QtWidgets is None:
            raise RuntimeError("PySide6 is not installed; PokerTableWindow unavailable")
        self.shared_state = shared_state
        self.settings = settings
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("PokerVision Table Renderer")
        self.window.resize(1220, 860)
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
        player_count = int(render_state["player_count"])
        available = [pos for pos in CANONICAL_RING[player_count] if pos in render_state.get("players", {})]
        hero_position = render_state.get("hero_position")
        if hero_position in available:
            idx = available.index(hero_position)
            return available[idx:] + available[:idx]
        return available

    def _player_brush(self, player_payload, is_hero):
        if player_payload.get("is_fold"):
            return QtGui.QBrush(QtGui.QColor(62, 62, 62))
        if player_payload.get("is_all_in"):
            return QtGui.QBrush(QtGui.QColor(118, 48, 48))
        if is_hero:
            return QtGui.QBrush(QtGui.QColor(176, 148, 58))
        return QtGui.QBrush(QtGui.QColor(52, 52, 52))

    def _draw_badge(self, painter, x: int, y: int, text: str, bg: QtGui.QColor, width: int = 36):
        painter.setPen(QtGui.QPen(QtGui.QColor(250, 250, 250), 1))
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
        if py >= int(rect.height() * 0.58):
            return box_y - card_h - 16
        return box_y + box_h + 12

    def _draw_player(self, painter, rect, pos_name: str, payload: dict, center_xy: Tuple[int, int], blind_text: str | None = None):
        px, py = center_xy
        box_w, box_h = 178, 106
        box_x = px - box_w // 2
        box_y = py - box_h // 2
        is_hero = payload.get("is_hero", False)

        painter.setBrush(self._player_brush(payload, is_hero))
        painter.setPen(QtGui.QPen(QtGui.QColor(245, 245, 245), 2))
        painter.drawRoundedRect(box_x, box_y, box_w, box_h, 14, 14)

        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.drawText(box_x + 10, box_y + 20, pos_name)
        if is_hero:
            self._draw_badge(painter, box_x + box_w - 86, box_y + 8, "HERO", QtGui.QColor(120, 88, 10), width=46)
        if payload.get("is_button"):
            self._draw_badge(painter, box_x + box_w - 36, box_y + 8, "D", QtGui.QColor(70, 110, 170), width=24)
        if blind_text:
            self._draw_badge(painter, box_x + 10, box_y - 20, blind_text, QtGui.QColor(95, 115, 55), width=max(34, 10 + len(blind_text) * 6))

        stack_text = "—"
        if payload.get("stack_bb") is not None:
            stack_text = f"{payload['stack_bb']:.1f} BB"
        elif payload.get("stack_text_raw"):
            stack_text = f"{payload['stack_text_raw']} BB"
        painter.drawText(box_x + 10, box_y + 42, f"Stack: {stack_text}")

        bet_text = "—"
        if payload.get("current_bet_bb") is not None:
            bet_text = f"{payload['current_bet_bb']:.1f} BB"
        elif payload.get("current_bet_raw"):
            bet_text = f"{payload['current_bet_raw']} BB"
        painter.drawText(box_x + 10, box_y + 62, f"Bet: {bet_text}")

        status_parts: List[str] = []
        if payload.get("is_fold"):
            status_parts.append("FOLD")
        if payload.get("is_all_in"):
            status_parts.append("ALL-IN")
        if payload.get("last_action"):
            status_parts.append(str(payload["last_action"]))
        warnings = payload.get("state_warnings", [])
        if warnings:
            status_parts.append(warnings[0])
        status_text = " | ".join(status_parts)[:42]
        if status_text:
            painter.drawText(box_x + 10, box_y + 84, status_text)

        if payload.get("show_card_backs") and not payload.get("is_fold"):
            back = render_card_back()
            card_y = self._player_card_y(rect, py, box_y, box_h, back.shape[0])
            self._draw_card_pixmap(painter, back, box_x + 58, card_y)
            self._draw_card_pixmap(painter, back, box_x + 98, card_y)
        return {"box_x": box_x, "box_y": box_y, "box_w": box_w, "box_h": box_h, "center_x": px, "center_y": py}

    def paintEvent(self, event):
        painter = QtGui.QPainter(self.window)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = self.window.rect()
            painter.fillRect(rect, QtGui.QColor(24, 44, 24))

            table_rect = QtCore.QRect(120, 110, rect.width() - 240, rect.height() - 240)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(30, 92, 36)))
            painter.setPen(QtGui.QPen(QtGui.QColor(220, 220, 220), 3))
            painter.drawEllipse(table_rect)

            _, render_state, _ = self.shared_state.snapshot()
            if not render_state:
                painter.setPen(QtGui.QColor(240, 240, 240))
                painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, "Waiting for render_state")
                return

            painter.setPen(QtGui.QColor(255, 255, 255))
            painter.drawText(20, 28, f"Hand: {render_state['hand_id']}")
            painter.drawText(20, 50, f"Street: {render_state['street']}")
            painter.drawText(20, 72, f"Status: {render_state['status']} / {render_state['freshness']}")
            painter.drawText(20, 94, f"Players: {render_state['player_count']} ({render_state['table_format']})")
            if render_state.get("warnings"):
                painter.drawText(20, 116, f"Warning: {render_state['warnings'][0]}")

            table_amount_state = render_state.get("table_amount_state", {}) if isinstance(render_state.get("table_amount_state", {}), dict) else {}
            total_pot = table_amount_state.get("total_pot", {}) if isinstance(table_amount_state, dict) else {}
            if total_pot.get("amount_bb") is not None:
                painter.drawText(rect.width() - 210, 28, f"Pot: {float(total_pot['amount_bb']):.1f} BB")

            blind_labels: Dict[str, str] = {}
            posted_blinds = table_amount_state.get("posted_blinds", {}) if isinstance(table_amount_state, dict) else {}
            for blind_name, payload in posted_blinds.items():
                pos = payload.get("matched_position") or blind_name
                amount = payload.get("amount_bb")
                if amount is not None:
                    blind_labels[pos] = f"{blind_name} {float(amount):g}"

            player_count = int(render_state["player_count"])
            ordered_positions = self._ordered_positions(render_state)
            slots = DISPLAY_SLOTS[player_count]
            player_boxes = {}
            for pos_name, slot in zip(ordered_positions, slots):
                payload = render_state["players"].get(pos_name, {})
                center_xy = (int(slot[0] * rect.width()), int(slot[1] * rect.height()))
                player_boxes[pos_name] = self._draw_player(painter, rect, pos_name, payload, center_xy, blind_labels.get(pos_name))

            board_cards = render_state.get("board_cards", [])
            board_w = len(board_cards) * 82 - (8 if board_cards else 0)
            bx = rect.width() // 2 - board_w // 2
            by = rect.height() // 2 - 54
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
                hy = max(145, hy)
                for idx, card in enumerate(hero_cards):
                    self._draw_card_pixmap(painter, render_card(card), hx + idx * 82, hy)

            if getattr(self.settings, "show_last_action_labels", True):
                actions = render_state.get("action_annotations", {}).get("actions_log", [])[-4:]
                y = rect.height() - 70
                for action in actions:
                    action_text = f"{action.get('street', '').upper()} {action.get('position', '')}: {action.get('action', '')}"
                    if action.get("amount_bb") not in (None, 0, 0.0):
                        action_text += f" {float(action['amount_bb']):.1f}"
                    painter.drawText(20, y, action_text)
                    y += 18
        finally:
            if painter.isActive():
                painter.end()
