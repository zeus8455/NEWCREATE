from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ModelPaths:
    active_hero: str = r"C:\PokerAI\AI_detect\Active_Hero\weights\best.pt"
    players_seat: str = r"C:\PokerAI\AI_detect\Player_Seat\weights\best.pt"
    all_cards: str = r"C:\PokerAI\AI_detect\All_Cards\weights\best.pt"
    player_state: str = r"C:\PokerAI\AI_detect\ChipsFold_from_scratch_train\weights\best.pt"
    table_amounts: str = r"C:\PokerAI\AI_detect\TotalPot_chips_SB_BB\weights\best.pt"
    digits: str = r"C:\PokerAI\AI_detect\0_9_Chips\weights\best.pt"


@dataclass(slots=True)
class Thresholds:
    active_hero_conf: float = 0.35
    players_seat_conf: float = 0.30
    all_cards_conf: float = 0.28
    player_state_conf: float = 0.30
    table_amounts_conf: float = 0.30
    digits_conf: float = 0.30
    duplicate_iou: float = 0.55
    seat_duplicate_iou: float = 0.45
    btn_conflict_iou: float = 0.30
    street_duplicate_iou: float = 0.35
    card_duplicate_iou: float = 0.30
    card_center_distance_px: int = 28


@dataclass(slots=True)
class CropSettings:
    image_ext: str = "png"


@dataclass(slots=True)
class RuntimeSettings:
    capture_interval_seconds: float = 1.0
    post_cycle_pause_seconds: float = 3.0
    ui_refresh_ms: int = 400
    frame_debounce_ms: int = 250
    save_raw_frame: bool = True
    save_overlay_frame: bool = True
    use_mock_detectors: bool = False
    monitor_index: int = 1
    mock_table_size: tuple[int, int] = (1920, 1080)


@dataclass(slots=True)
class OutputSettings:
    base_dir: Path = field(default_factory=lambda: Path(r"C:\PokerAI\NEWCREATE"))
    state_json_name: str = "current_hand_state.json"
    last_frame_json_name: str = "last_frame_analysis.json"


@dataclass(slots=True)
class AppConfig:
    model_paths: ModelPaths = field(default_factory=ModelPaths)
    thresholds: Thresholds = field(default_factory=Thresholds)
    crops: CropSettings = field(default_factory=CropSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

    schema_version: str = "1.1"
    debug_mode: bool = False
    hand_stale_timeout_sec: float = 30.0
    hand_close_timeout_sec: float = 90.0
    normalize_short_stack_to_40bb: bool = False
    short_stack_min_inclusive_bb: float = 0.0
    short_stack_max_exclusive_bb: float = 40.0
    short_stack_forced_value_bb: float = 40.0

    @property
    def state_json_path(self) -> Path:
        return self.output.base_dir / self.output.state_json_name

    @property
    def last_frame_json_path(self) -> Path:
        return self.output.base_dir / self.output.last_frame_json_name

    # launcher compatibility aliases
    @property
    def ui_refresh_ms(self) -> int:
        return self.runtime.ui_refresh_ms

    @ui_refresh_ms.setter
    def ui_refresh_ms(self, value: int) -> None:
        self.runtime.ui_refresh_ms = int(value)

    @property
    def frame_debounce_ms(self) -> int:
        return self.runtime.frame_debounce_ms

    @frame_debounce_ms.setter
    def frame_debounce_ms(self, value: int) -> None:
        self.runtime.frame_debounce_ms = int(value)

    @property
    def monitor_index(self) -> int:
        return self.runtime.monitor_index

    @monitor_index.setter
    def monitor_index(self, value: int) -> None:
        self.runtime.monitor_index = int(value)

    @property
    def mock_table_size(self) -> tuple[int, int]:
        return self.runtime.mock_table_size

    @mock_table_size.setter
    def mock_table_size(self, value: tuple[int, int]) -> None:
        self.runtime.mock_table_size = tuple(value)

    @property
    def base_dir(self) -> Path:
        return self.output.base_dir

    @base_dir.setter
    def base_dir(self, value: str | Path) -> None:
        self.output.base_dir = Path(value)


POSITION_LABELS_BY_COUNT: dict[int, list[str]] = {
    2: ["BTN", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["BTN", "SB", "BB", "CO"],
    5: ["BTN", "SB", "BB", "UTG", "CO"],
    6: ["BTN", "SB", "BB", "UTG", "MP", "CO"],
}

ACTIVE_HERO_CLASS_ALIASES = {"activehero", "active_hero", "hero_active", "active-hero"}
PLAYER_SEAT_CLASS_ALIASES = {"player_seat", "player-seat", "seat", "playerseat"}
BTN_CLASS_ALIASES = {"btn", "dealer_button", "button"}
STREET_CLASS_ALIASES = {
    "flop": {"flop"},
    "turn": {"turn"},
    "river": {"river"},
}
SUIT_ALIASES = {
    "spades": "s",
    "spade": "s",
    "s": "s",
    "hearts": "h",
    "heart": "h",
    "h": "h",
    "diamonds": "d",
    "diamond": "d",
    "d": "d",
    "clubs": "c",
    "club": "c",
    "c": "c",
}
RANK_ALIASES = {
    "a": "A",
    "ace": "A",
    "k": "K",
    "king": "K",
    "q": "Q",
    "queen": "Q",
    "j": "J",
    "jack": "J",
    "t": "T",
    "10": "T",
    "ten": "T",
    "9": "9",
    "8": "8",
    "7": "7",
    "6": "6",
    "5": "5",
    "4": "4",
    "3": "3",
    "2": "2",
}


def get_default_settings() -> AppConfig:
    return AppConfig()
