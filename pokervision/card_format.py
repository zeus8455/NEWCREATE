from __future__ import annotations

RANK_MAP = {
    "A": "A",
    "K": "K",
    "Q": "Q",
    "J": "J",
    "10": "T",
    "T": "T",
    "9": "9",
    "8": "8",
    "7": "7",
    "6": "6",
    "5": "5",
    "4": "4",
    "3": "3",
    "2": "2",
}

SUIT_MAP = {
    "spades": "s",
    "hearts": "h",
    "diamonds": "d",
    "clubs": "c",
}

DISPLAY_SUIT = {
    "s": "♠",
    "h": "♥",
    "d": "♦",
    "c": "♣",
}

SUIT_COLOR_NAME = {
    "s": "black",
    "h": "red",
    "d": "blue",
    "c": "green",
}


def detector_label_to_internal(label: str) -> str:
    try:
        rank, suit = label.split("_", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid detector label: {label}") from exc
    if rank not in RANK_MAP or suit not in SUIT_MAP:
        raise ValueError(f"Unsupported detector label: {label}")
    return f"{RANK_MAP[rank]}{SUIT_MAP[suit]}"


def internal_to_display(card: str) -> str:
    if len(card) != 2:
        raise ValueError(f"Invalid internal card: {card}")
    rank, suit = card[0], card[1]
    rank_display = "10" if rank == "T" else rank
    return f"{rank_display}{DISPLAY_SUIT[suit]}"


def suit_color_name(card: str) -> str:
    return SUIT_COLOR_NAME[card[1]]
