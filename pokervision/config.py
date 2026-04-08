from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Settings:
    # ---- identity / runtime ----
    schema_version: str = "1.1"
    debug_mode: bool = True
    monitor_index: int = 1
    mock_table_size: tuple[int, int] = (1280, 720)
    frame_debounce_ms: int = 250
    ui_refresh_ms: int = 400
    hand_stale_timeout_sec: float = 12.0
    hand_close_timeout_sec: float = 30.0
    keep_temp_on_exit: bool = False
    save_debug_on_error: bool = True

    # ---- short stack normalization used by launcher ----
    normalize_short_stack_to_40bb: bool = False
    short_stack_min_inclusive_bb: float = 0.0
    short_stack_max_exclusive_bb: float = 40.0
    short_stack_forced_value_bb: float = 40.0

    # ---- model paths ----
    active_hero_model_path: str = r"C:\PokerAI\AI_detect\Active_Hero\weights\best.pt"
    table_structure_model_path: str = r"C:\PokerAI\AI_detect\Player_Seat\weights\best.pt"
    player_state_model_path: str = r"C:\PokerAI\AI_detect\ChipsFold_from_scratch_train\weights\best.pt"
    hero_cards_model_path: str = r"C:\PokerAI\AI_detect\All_Cards\weights\best.pt"
    board_cards_model_path: str = r"C:\PokerAI\AI_detect\All_Cards\weights\best.pt"
    table_amount_model_path: str = r"C:\PokerAI\AI_detect\TotalPot_chips_SB_BB\weights\best.pt"
    table_amount_digits_model_path: str = r"C:\PokerAI\AI_detect\0_9_Chips\weights\best.pt"

    # ---- confidences ----
    active_hero_conf: float = 0.35
    table_conf: float = 0.30
    player_state_conf: float = 0.30
    card_conf: float = 0.28
    table_amount_conf: float = 0.28
    table_amount_digits_conf: float = 0.28

    # ---- generic duplicate / matching thresholds ----
    duplicate_iou_threshold: float = 0.55
    duplicate_center_distance_px: int = 28
    seat_match_max_distance_px: float = 260.0
    blind_marker_to_position_max_distance_px: float = 240.0
    chips_to_position_max_distance_px: float = 320.0
    chips_ambiguity_margin_px: float = 28.0
    table_pot_center_exclusion_radius_px: float = 95.0

    # ---- player-state token normalization ----
    player_state_lower_band_ratio: float = 0.58
    player_state_token_iou_threshold: float = 0.60
    player_state_token_center_threshold_px: float = 14.0

    # ---- table amount token normalization ----
    table_amount_region_iou_threshold: float = 0.55
    table_amount_region_center_threshold_px: float = 26.0
    table_amount_digit_iou_threshold: float = 0.30
    table_amount_digit_center_threshold_px: float = 12.0

    # ---- exact crop rules ----
    strict_hero_crop_to_structure_bbox: bool = True
    strict_board_crop_to_marker_bbox: bool = True
    strict_player_state_crop_to_structure_bbox: bool = True

    # ---- optional crop expansion fallback ----
    hero_crop_pad_x_ratio: float = 0.10
    hero_crop_pad_top_ratio: float = 0.00
    hero_crop_pad_bottom_ratio: float = 0.00
    board_crop_pad_x_ratio: float = 3.6
    board_crop_pad_y_ratio: float = 2.2
    board_min_crop_width_px: int = 220
    board_min_crop_height_px: int = 96
    player_state_crop_pad_x_ratio: float = 0.05
    player_state_crop_pad_top_ratio: float = 0.00
    player_state_crop_pad_bottom_ratio: float = 0.00

    # ---- chips geometry scoring ----
    chips_target_towards_table_center: float = 0.58
    chips_projection_outside_segment_slack: float = 0.22
    chips_projection_penalty_scale_px: float = 220.0
    chips_line_distance_weight: float = 0.85

    # ---- pipeline switches ----
    action_reconstruction_enabled: bool = True
    normal_mode_save_repeated_frames: bool = False
    max_retry_per_stage: int = 1

    # ---- output ----
    root_dir: Path = field(default_factory=lambda: Path(r"C:\PokerAI\NEWCREATE"))
    state_json_name: str = "current_hand_state.json"
    last_frame_json_name: str = "last_frame_analysis.json"

    def hands_dir(self) -> Path:
        return self.root_dir / "hands"

    def logs_dir(self) -> Path:
        return self.root_dir / "logs"

    def temp_dir(self) -> Path:
        return self.root_dir / "temp"

    @property
    def state_json_path(self) -> Path:
        return self.root_dir / self.state_json_name

    @property
    def last_frame_json_path(self) -> Path:
        return self.root_dir / self.last_frame_json_name


# backwards/forwards compatibility
AppConfig = Settings


def get_default_settings() -> Settings:
    return Settings()


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
    "spades": "s", "spade": "s", "s": "s",
    "hearts": "h", "heart": "h", "h": "h",
    "diamonds": "d", "diamond": "d", "d": "d",
    "clubs": "c", "club": "c", "c": "c",
}
RANK_ALIASES = {
    "a": "A", "ace": "A",
    "k": "K", "king": "K",
    "q": "Q", "queen": "Q",
    "j": "J", "jack": "J",
    "t": "T", "10": "T", "ten": "T",
    "9": "9", "8": "8", "7": "7", "6": "6", "5": "5", "4": "4", "3": "3", "2": "2",
}
