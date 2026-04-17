from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


CARD_BG = (246, 246, 246)
CARD_BORDER = (48, 48, 48)
RANK_TEXT = (35, 35, 35)
SUIT_COLOR_BGR = {
    "s": (20, 20, 20),
    "h": (40, 40, 220),
    "d": (220, 120, 30),
    "c": (40, 170, 40),
}


def _rank_text(card: str) -> str:
    rank = card[0]
    return "10" if rank == "T" else rank


def _draw_heart(img: np.ndarray, center: tuple[int, int], size: int, color: tuple[int, int, int]) -> None:
    cx, cy = center
    radius = max(2, size // 4)
    left = (cx - radius, cy - radius // 2)
    right = (cx + radius, cy - radius // 2)
    cv2.circle(img, left, radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, right, radius, color, -1, lineType=cv2.LINE_AA)
    triangle = np.array([
        [cx - radius * 2, cy],
        [cx + radius * 2, cy],
        [cx, cy + radius * 3],
    ], dtype=np.int32)
    cv2.fillConvexPoly(img, triangle, color, lineType=cv2.LINE_AA)


def _draw_diamond(img: np.ndarray, center: tuple[int, int], size: int, color: tuple[int, int, int]) -> None:
    cx, cy = center
    half = max(3, size // 2)
    diamond = np.array([
        [cx, cy - half],
        [cx + half, cy],
        [cx, cy + half],
        [cx - half, cy],
    ], dtype=np.int32)
    cv2.fillConvexPoly(img, diamond, color, lineType=cv2.LINE_AA)


def _draw_club(img: np.ndarray, center: tuple[int, int], size: int, color: tuple[int, int, int]) -> None:
    cx, cy = center
    radius = max(2, size // 4)
    cv2.circle(img, (cx, cy - radius), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (cx - radius, cy + radius // 2), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (cx + radius, cy + radius // 2), radius, color, -1, lineType=cv2.LINE_AA)
    stem_w = max(2, radius // 2)
    stem_h = max(6, radius * 2)
    cv2.rectangle(img, (cx - stem_w, cy + radius), (cx + stem_w, cy + radius + stem_h), color, -1)
    base = np.array([
        [cx - radius * 2, cy + radius + stem_h],
        [cx + radius * 2, cy + radius + stem_h],
        [cx, cy + radius + stem_h // 2],
    ], dtype=np.int32)
    cv2.fillConvexPoly(img, base, color, lineType=cv2.LINE_AA)


def _draw_spade(img: np.ndarray, center: tuple[int, int], size: int, color: tuple[int, int, int]) -> None:
    cx, cy = center
    radius = max(2, size // 4)
    cv2.circle(img, (cx - radius, cy), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (cx + radius, cy), radius, color, -1, lineType=cv2.LINE_AA)
    triangle = np.array([
        [cx - radius * 2, cy + radius // 2],
        [cx + radius * 2, cy + radius // 2],
        [cx, cy - radius * 3],
    ], dtype=np.int32)
    cv2.fillConvexPoly(img, triangle, color, lineType=cv2.LINE_AA)
    stem_w = max(2, radius // 2)
    stem_h = max(6, radius * 2)
    cv2.rectangle(img, (cx - stem_w, cy + radius), (cx + stem_w, cy + radius + stem_h), color, -1)
    base = np.array([
        [cx - radius * 2, cy + radius + stem_h],
        [cx + radius * 2, cy + radius + stem_h],
        [cx, cy + radius + stem_h // 2],
    ], dtype=np.int32)
    cv2.fillConvexPoly(img, base, color, lineType=cv2.LINE_AA)


def _draw_suit(img: np.ndarray, suit: str, center: tuple[int, int], size: int) -> None:
    color = SUIT_COLOR_BGR[suit]
    if suit == "h":
        _draw_heart(img, center, size, color)
    elif suit == "d":
        _draw_diamond(img, center, size, color)
    elif suit == "c":
        _draw_club(img, center, size, color)
    else:
        _draw_spade(img, center, size, color)


def render_card(card: str, size: Tuple[int, int] = (74, 104)) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), CARD_BG, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), CARD_BORDER, 2)

    rank = _rank_text(card)
    suit = card[1]
    suit_color = SUIT_COLOR_BGR[suit]

    cv2.putText(img, rank, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, suit_color, 2, lineType=cv2.LINE_AA)
    _draw_suit(img, suit, (18, 36), 12)

    _draw_suit(img, suit, (w // 2, h // 2 + 4), 28)

    rank_size, _ = cv2.getTextSize(rank, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 1)
    cv2.putText(
        img,
        rank,
        (w - rank_size[0] - 8, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        suit_color,
        1,
        lineType=cv2.LINE_AA,
    )
    _draw_suit(img, suit, (w - 16, h - 28), 10)
    return img


def render_card_back(size: Tuple[int, int] = (36, 52)) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), 76, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (205, 205, 205), 2)
    cv2.rectangle(img, (5, 5), (w - 6, h - 6), (155, 155, 155), 1)
    for offset in range(-h, w + h, 8):
        cv2.line(img, (offset, 0), (offset + h, h), (118, 118, 118), 1, lineType=cv2.LINE_AA)
        cv2.line(img, (offset, h), (offset + h, 0), (96, 96, 96), 1, lineType=cv2.LINE_AA)
    return img
