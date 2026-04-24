from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .config import Settings
from .models import BBox, CapturedFrame, Detection

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None


PLAYER_STATE_TOKEN_LABELS = {"fold", ".", "all-in", *[str(i) for i in range(10)]}
TABLE_AMOUNT_REGION_LABELS = {"Chips", "SB", "BB", "TotalPot"}
TABLE_AMOUNT_DIGIT_LABELS = {".", *[str(i) for i in range(10)]}


class DetectorBackend:
    def detect_active_hero(self, frame: CapturedFrame) -> List[Detection]:
        raise NotImplementedError

    def detect_structure(self, frame: CapturedFrame) -> List[Detection]:
        raise NotImplementedError

    def detect_player_state(self, frame: CapturedFrame, player_bbox: BBox | None) -> List[Detection]:
        raise NotImplementedError

    def detect_hero_cards(self, frame: CapturedFrame, hero_bbox: BBox | None) -> List[Detection]:
        raise NotImplementedError

    def detect_board_cards(self, frame: CapturedFrame, board_bbox: BBox | None, street: str) -> List[Detection]:
        raise NotImplementedError

    def detect_table_amount_regions(self, frame: CapturedFrame) -> List[Detection]:
        raise NotImplementedError

    def detect_table_amount_digits(self, frame: CapturedFrame, amount_bbox: BBox | None) -> List[Detection]:
        raise NotImplementedError


def _resolve_model_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_file():
        return str(path)
    if path.is_dir():
        candidates = [path / "best.pt", path / "last.pt"]
        candidates.extend(sorted(path.glob("*.pt")))
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return str(candidate)
    return str(path)


def _clip_bbox(image: np.ndarray, bbox: BBox) -> Tuple[int, int, int, int]:
    h, w = image.shape[:2]
    x1 = max(0, min(w, int(round(bbox.x1))))
    y1 = max(0, min(h, int(round(bbox.y1))))
    x2 = max(0, min(w, int(round(bbox.x2))))
    y2 = max(0, min(h, int(round(bbox.y2))))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def build_exact_crop(image: np.ndarray, bbox: BBox) -> tuple[np.ndarray, tuple[int, int]]:
    x1, y1, x2, y2 = _clip_bbox(image, bbox)
    return image[y1:y2, x1:x2].copy(), (x1, y1)


def build_hero_crop(image: np.ndarray, bbox: BBox, settings: Settings) -> tuple[np.ndarray, tuple[int, int]]:
    if settings.strict_hero_crop_to_structure_bbox:
        return build_exact_crop(image, bbox)
    pad_x = bbox.width * settings.hero_crop_pad_x_ratio
    pad_top = bbox.height * settings.hero_crop_pad_top_ratio
    pad_bottom = bbox.height * settings.hero_crop_pad_bottom_ratio
    expanded = BBox(bbox.x1 - pad_x, bbox.y1 - pad_top, bbox.x2 + pad_x, bbox.y2 + pad_bottom)
    return build_exact_crop(image, expanded)


def build_board_crop(image: np.ndarray, bbox: BBox, settings: Settings) -> tuple[np.ndarray, tuple[int, int]]:
    if settings.strict_board_crop_to_marker_bbox:
        return build_exact_crop(image, bbox)
    width = max(bbox.width * settings.board_crop_pad_x_ratio, settings.board_min_crop_width_px)
    height = max(bbox.height * settings.board_crop_pad_y_ratio, settings.board_min_crop_height_px)
    cx, cy = bbox.center
    expanded = BBox(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
    return build_exact_crop(image, expanded)


def build_player_state_crop(image: np.ndarray, bbox: BBox, settings: Settings) -> tuple[np.ndarray, tuple[int, int]]:
    if settings.strict_player_state_crop_to_structure_bbox:
        return build_exact_crop(image, bbox)
    pad_x = bbox.width * settings.player_state_crop_pad_x_ratio
    pad_top = bbox.height * settings.player_state_crop_pad_top_ratio
    pad_bottom = bbox.height * settings.player_state_crop_pad_bottom_ratio
    expanded = BBox(bbox.x1 - pad_x, bbox.y1 - pad_top, bbox.x2 + pad_x, bbox.y2 + pad_bottom)
    return build_exact_crop(image, expanded)


def build_table_amount_crop(image: np.ndarray, bbox: BBox, settings: Settings) -> tuple[np.ndarray, tuple[int, int]]:
    return build_exact_crop(image, bbox)


@dataclass(slots=True)
class MockDetectorBackend(DetectorBackend):
    settings: Settings

    def _stage(self, frame_id: str) -> tuple[int, str, int]:
        index = int(frame_id.split("_")[-1])
        phase = (index - 1) % 8
        if phase in {0, 1}:
            return 1, "preflop", phase
        if phase in {2, 3}:
            return 1, "flop", phase
        if phase in {4, 5}:
            return 1, "turn", phase
        if phase == 6:
            return 1, "river", phase
        return 2, "preflop", phase

    def detect_active_hero(self, frame: CapturedFrame) -> List[Detection]:
        return [Detection("ActiveHero", BBox(520, 550, 760, 700), 0.98)]

    def detect_structure(self, frame: CapturedFrame) -> List[Detection]:
        _, street, _ = self._stage(frame.frame_id)
        detections = [
            Detection("BTN", BBox(600, 580, 660, 640), 0.99),
            Detection("player_seat", BBox(980, 480, 1080, 560), 0.96),
            Detection("player_seat", BBox(980, 220, 1080, 300), 0.96),
            Detection("player_seat", BBox(610, 90, 710, 170), 0.96),
            Detection("player_seat", BBox(210, 190, 310, 270), 0.96),
            Detection("player_seat", BBox(170, 500, 270, 580), 0.96),
        ]
        if street == "flop":
            detections.append(Detection("Flop", BBox(560, 300, 640, 360), 0.91))
        elif street == "turn":
            detections.append(Detection("Turn", BBox(560, 300, 640, 360), 0.91))
        elif street == "river":
            detections.append(Detection("River", BBox(560, 300, 640, 360), 0.91))
        return detections

    def _mock_player_tokens(self, bbox: BBox) -> List[tuple[str, BBox, float]]:
        cx, cy = bbox.center
        width = max(80.0, bbox.width)
        height = max(70.0, bbox.height)
        start_x = 10.0
        token_w = 16.0
        gap = 6.0
        y1 = max(0.0, height * 0.63)
        y2 = min(height - 2.0, y1 + 18.0)

        if 560 <= cx <= 700 and cy > 540:
            labels = ["2", "5", ".", "0"]
        elif cx > 900 and cy > 420:
            labels = ["1", "8", ".", "5"]
        elif cx > 900 and cy <= 420:
            labels = ["all-in"]
        elif 560 <= cx <= 760 and cy < 220:
            labels = ["4", "2"]
        elif cx < 400 and cy < 350:
            labels = ["fold"]
        else:
            labels = ["3", "0", ".", "2"]

        tokens: List[tuple[str, BBox, float]] = []
        for idx, label in enumerate(labels):
            x1 = start_x + idx * (token_w + gap)
            token_width = 12.0 if label == "." else (32.0 if label == "all-in" else 16.0)
            x2 = min(width - 4.0, x1 + token_width)
            tokens.append((label, BBox(x1, y1, x2, y2), 0.91 - idx * 0.01))
        return tokens

    def detect_player_state(self, frame: CapturedFrame, player_bbox: BBox | None) -> List[Detection]:
        if player_bbox is None:
            return []
        _, origin = build_player_state_crop(frame.image, player_bbox, self.settings)
        ox, oy = origin
        return [Detection(label, BBox(bbox.x1 + ox, bbox.y1 + oy, bbox.x2 + ox, bbox.y2 + oy), conf) for label, bbox, conf in self._mock_player_tokens(player_bbox)]

    def detect_hero_cards(self, frame: CapturedFrame, hero_bbox: BBox | None) -> List[Detection]:
        hand_no, _, _ = self._stage(frame.frame_id)
        cards = [("A_hearts", BBox(20, 20, 60, 90), 0.95), ("J_diamonds", BBox(65, 20, 105, 90), 0.93)] if hand_no == 1 else [("K_spades", BBox(20, 20, 60, 90), 0.95), ("Q_spades", BBox(65, 20, 105, 90), 0.93)]
        if hero_bbox is None:
            return [Detection(label, bbox, conf) for label, bbox, conf in cards]
        _, origin = build_hero_crop(frame.image, hero_bbox, self.settings)
        ox, oy = origin
        return [Detection(label, BBox(bbox.x1 + ox, bbox.y1 + oy, bbox.x2 + ox, bbox.y2 + oy), conf) for label, bbox, conf in cards]

    def detect_board_cards(self, frame: CapturedFrame, board_bbox: BBox | None, street: str) -> List[Detection]:
        if street == "preflop":
            return []
        board = [("8_spades", BBox(20, 10, 60, 80), 0.94), ("4_spades", BBox(68, 10, 108, 80), 0.93), ("K_diamonds", BBox(116, 10, 156, 80), 0.95)]
        if street == "turn":
            board.append(("2_clubs", BBox(164, 10, 204, 80), 0.92))
        elif street == "river":
            board.extend([("2_clubs", BBox(164, 10, 204, 80), 0.92), ("T_hearts", BBox(212, 10, 252, 80), 0.90)])
        if board_bbox is None:
            return [Detection(label, bbox, conf) for label, bbox, conf in board]
        _, origin = build_board_crop(frame.image, board_bbox, self.settings)
        ox, oy = origin
        return [Detection(label, BBox(bbox.x1 + ox, bbox.y1 + oy, bbox.x2 + ox, bbox.y2 + oy), conf) for label, bbox, conf in board]

    def _mock_table_amount_regions(self, frame_id: str) -> List[tuple[str, BBox, str]]:
        hand_no, street, phase = self._stage(frame_id)
        if hand_no == 1 and street == "preflop":
            if phase == 0:
                return [
                    ("SB", BBox(1000, 505, 1065, 540), "0.5"),
                    ("BB", BBox(1000, 240, 1065, 275), "1"),
                    ("TotalPot", BBox(560, 340, 650, 372), "1.5"),
                    ("Chips", BBox(610, 165, 690, 198), "3"),
                ]
            return [
                ("SB", BBox(1000, 505, 1065, 540), "0.5"),
                ("BB", BBox(1000, 240, 1065, 275), "1"),
                ("TotalPot", BBox(560, 340, 650, 372), "4.5"),
                ("Chips", BBox(610, 165, 690, 198), "3"),
            ]
        if hand_no == 1 and street == "flop":
            return [
                ("TotalPot", BBox(560, 340, 650, 372), "7.5"),
                ("Chips", BBox(610, 620, 690, 652), "2.5"),
            ]
        if hand_no == 1 and street == "turn":
            return [
                ("TotalPot", BBox(560, 340, 650, 372), "10.0"),
                ("Chips", BBox(610, 620, 690, 652), "4"),
            ]
        if hand_no == 1 and street == "river":
            return [
                ("TotalPot", BBox(560, 340, 650, 372), "18.0"),
            ]
        return [
            ("SB", BBox(1000, 505, 1065, 540), "0.5"),
            ("BB", BBox(1000, 240, 1065, 275), "1"),
            ("TotalPot", BBox(560, 340, 650, 372), "1.5"),
            ("Chips", BBox(210, 530, 290, 562), "1"),
        ]

    def detect_table_amount_regions(self, frame: CapturedFrame) -> List[Detection]:
        return [Detection(label, bbox, 0.94 - idx * 0.01) for idx, (label, bbox, _) in enumerate(self._mock_table_amount_regions(frame.frame_id))]

    def detect_table_amount_digits(self, frame: CapturedFrame, amount_bbox: BBox | None) -> List[Detection]:
        if amount_bbox is None:
            return []
        match_text = None
        match_bbox = None
        for _, bbox, text in self._mock_table_amount_regions(frame.frame_id):
            if abs(bbox.x1 - amount_bbox.x1) < 1e-6 and abs(bbox.y1 - amount_bbox.y1) < 1e-6 and abs(bbox.x2 - amount_bbox.x2) < 1e-6 and abs(bbox.y2 - amount_bbox.y2) < 1e-6:
                match_text = text
                match_bbox = bbox
                break
        if match_text is None or match_bbox is None:
            return []
        _, origin = build_table_amount_crop(frame.image, amount_bbox, self.settings)
        ox, oy = origin
        start_x = 10.0
        gap = 6.0
        out: List[Detection] = []
        for idx, ch in enumerate(match_text):
            width = 10.0 if ch == "." else 16.0
            x1 = start_x + idx * (16.0 + gap)
            x2 = x1 + width
            y1 = 8.0
            y2 = 26.0
            out.append(Detection(ch, BBox(x1 + ox, y1 + oy, x2 + ox, y2 + oy), 0.95 - idx * 0.01))
        return out


class YoloDetectorBackend(DetectorBackend):
    def __init__(self, settings: Settings):
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed; real YOLO backend is unavailable")
        self.settings = settings
        self._active = YOLO(_resolve_model_path(settings.active_hero_model_path))
        self._structure = YOLO(_resolve_model_path(settings.table_structure_model_path))
        self._player_state = YOLO(_resolve_model_path(settings.player_state_model_path))
        self._hero_cards = YOLO(_resolve_model_path(settings.hero_cards_model_path))
        self._board_cards = YOLO(_resolve_model_path(settings.board_cards_model_path))
        self._table_amount = YOLO(_resolve_model_path(settings.table_amount_model_path))
        self._table_amount_digits = YOLO(_resolve_model_path(settings.table_amount_digits_model_path))

    def _predict(self, model, image: np.ndarray, conf: float, offset: tuple[int, int] = (0, 0)) -> List[Detection]:  # pragma: no cover
        results = model.predict(image, conf=conf, verbose=False)
        detections: List[Detection] = []
        ox, oy = offset
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                detections.append(Detection(label=str(names[cls_id]), bbox=BBox(x1 + ox, y1 + oy, x2 + ox, y2 + oy), confidence=float(box.conf[0])))
        return detections

    def detect_active_hero(self, frame: CapturedFrame) -> List[Detection]:  # pragma: no cover
        return self._predict(self._active, frame.image, self.settings.active_hero_conf)

    def detect_structure(self, frame: CapturedFrame) -> List[Detection]:  # pragma: no cover
        return self._predict(self._structure, frame.image, self.settings.table_conf)

    def detect_player_state(self, frame: CapturedFrame, player_bbox: BBox | None) -> List[Detection]:  # pragma: no cover
        if player_bbox is None:
            return []
        crop, origin = build_player_state_crop(frame.image, player_bbox, self.settings)
        detections = self._predict(self._player_state, crop, self.settings.player_state_conf, offset=origin)
        return [det for det in detections if det.label.lower() in PLAYER_STATE_TOKEN_LABELS]

    def detect_hero_cards(self, frame: CapturedFrame, hero_bbox: BBox | None) -> List[Detection]:  # pragma: no cover
        if hero_bbox is None:
            return []
        crop, origin = build_hero_crop(frame.image, hero_bbox, self.settings)
        return self._predict(self._hero_cards, crop, self.settings.card_conf, offset=origin)

    def detect_board_cards(self, frame: CapturedFrame, board_bbox: BBox | None, street: str) -> List[Detection]:  # pragma: no cover
        if street == "preflop" or board_bbox is None:
            return []
        crop, origin = build_board_crop(frame.image, board_bbox, self.settings)
        return self._predict(self._board_cards, crop, self.settings.card_conf, offset=origin)

    def detect_table_amount_regions(self, frame: CapturedFrame) -> List[Detection]:  # pragma: no cover
        detections = self._predict(self._table_amount, frame.image, self.settings.table_amount_conf)
        return [det for det in detections if det.label in TABLE_AMOUNT_REGION_LABELS]

    def detect_table_amount_digits(self, frame: CapturedFrame, amount_bbox: BBox | None) -> List[Detection]:  # pragma: no cover
        if amount_bbox is None:
            return []
        crop, origin = build_table_amount_crop(frame.image, amount_bbox, self.settings)
        detections = self._predict(self._table_amount_digits, crop, self.settings.table_amount_digits_conf, offset=origin)
        return [det for det in detections if det.label in TABLE_AMOUNT_DIGIT_LABELS]
