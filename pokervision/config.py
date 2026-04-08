from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass(slots=True)
class Settings:
    schema_version: str = "1.1"

    active_hero_model_path: str = r"C:\PokerAI\AI_detect\Active_Hero\weights"
    table_structure_model_path: str = r"C:\PokerAI\AI_detect\Player_Seat\weights"
    hero_cards_model_path: str = r"C:\PokerAI\AI_detect\All_Cards\weights"
    board_cards_model_path: str = r"C:\PokerAI\AI_detect\All_Cards\weights"
    player_state_model_path: str = r"C:\PokerAI\AI_detect\ChipsFold_from_scratch_train\weights"
    table_amount_model_path: str = r"C:\PokerAI\AI_detect\TotalPot_chips_SB_BB\weights"
    table_amount_digits_model_path: str = r"C:\PokerAI\AI_detect\0_9_Chips\weights"

    root_dir: Path = Path(r"C:\PokerAI\PokerVision\PokerVision_DataSafeFiles")
    hands_dir_name: str = "hands"
    logs_dir_name: str = "logs"
    temp_dir_name: str = "temp"

    active_hero_conf: float = 0.50
    table_conf: float = 0.40
    card_conf: float = 0.35
    player_state_conf: float = 0.30
    table_amount_conf: float = 0.35
    table_amount_digits_conf: float = 0.30
    duplicate_iou_threshold: float = 0.65
    nms_iou_threshold: float = 0.45
    duplicate_center_distance_px: float = 32.0
    seat_match_max_distance_px: float = 95.0
    table_center_max_shift_px: float = 120.0
    frame_debounce_ms: int = 200
    hand_stale_timeout_sec: float = 3.5
    hand_close_timeout_sec: float = 8.0
    max_retry_per_stage: int = 1
    frame_queue_size: int = 2

    strict_hero_crop_to_structure_bbox: bool = True
    strict_board_crop_to_marker_bbox: bool = True
    strict_player_state_crop_to_structure_bbox: bool = True
    strict_table_amount_crop_to_region_bbox: bool = True
    hero_crop_pad_x_ratio: float = 0.65
    hero_crop_pad_top_ratio: float = 1.35
    hero_crop_pad_bottom_ratio: float = 0.40
    board_crop_pad_x_ratio: float = 2.20
    board_crop_pad_y_ratio: float = 1.20
    player_state_crop_pad_x_ratio: float = 0.20
    player_state_crop_pad_top_ratio: float = 0.10
    player_state_crop_pad_bottom_ratio: float = 0.20
    player_state_lower_band_ratio: float = 0.55
    player_state_token_iou_threshold: float = 0.50
    player_state_token_center_threshold_px: float = 18.0
    board_min_crop_width_px: int = 220
    board_min_crop_height_px: int = 90
    monitor_index: int = 1

    table_amount_region_iou_threshold: float = 0.55
    table_amount_region_center_threshold_px: float = 24.0
    table_amount_digit_iou_threshold: float = 0.50
    table_amount_digit_center_threshold_px: float = 14.0
    chips_to_position_max_distance_px: float = 240.0
    chips_ambiguity_margin_px: float = 28.0
    table_pot_center_exclusion_radius_px: float = 160.0
    blind_marker_to_position_max_distance_px: float = 200.0
    chips_target_towards_table_center: float = 0.55
    chips_line_distance_weight: float = 0.60
    chips_projection_outside_segment_slack: float = 0.22
    chips_projection_penalty_scale_px: float = 220.0
    action_state_stability_frames: int = 1
    action_reconstruction_enabled: bool = True
    infer_checks_without_explicit_evidence: bool = False
    show_bets_in_renderer: bool = True
    show_pot_in_renderer: bool = True
    show_last_action_labels: bool = True

    normalize_short_stack_to_40bb: bool = False
    short_stack_min_inclusive_bb: float = 0.0
    short_stack_max_exclusive_bb: float = 20.0
    short_stack_forced_value_bb: float = 40.0

    debug_mode: bool = True
    save_debug_on_error: bool = True
    normal_mode_save_repeated_frames: bool = False
    keep_temp_on_exit: bool = False
    keep_render_snapshots: bool = False

    ui_refresh_ms: int = 100
    use_pyside6: bool = True

    mock_table_size: tuple[int, int] = (1280, 720)
    mock_player_count: int = 6

    position_layouts: Dict[int, Dict[str, tuple[float, float]]] = field(default_factory=lambda: {
        2: {"BTN": (0.50, 0.82), "BB": (0.50, 0.18)},
        3: {"BTN": (0.50, 0.82), "SB": (0.82, 0.50), "BB": (0.18, 0.50)},
        4: {"BTN": (0.50, 0.82), "SB": (0.84, 0.60), "BB": (0.78, 0.25), "CO": (0.20, 0.40)},
        5: {"BTN": (0.50, 0.82), "SB": (0.86, 0.63), "BB": (0.82, 0.28), "UTG": (0.50, 0.14), "CO": (0.18, 0.45)},
        6: {"BTN": (0.50, 0.84), "SB": (0.86, 0.66), "BB": (0.82, 0.30), "UTG": (0.52, 0.14), "MP": (0.20, 0.26), "CO": (0.15, 0.62)},
    })

    def hands_dir(self) -> Path:
        return self.root_dir / self.hands_dir_name

    def logs_dir(self) -> Path:
        return self.root_dir / self.logs_dir_name

    def temp_dir(self) -> Path:
        return self.root_dir / self.temp_dir_name


def get_default_settings() -> Settings:
    return Settings()
