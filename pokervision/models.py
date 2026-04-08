from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


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
    action_inference: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
            "street": self.street,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "validation": self.validation,
            "player_count": self.player_count,
            "table_format": self.table_format,
            "occupied_positions": list(self.occupied_positions),
            "positions": self.positions,
            "hero_position": self.hero_position,
            "hero_cards": list(self.hero_cards),
            "board_cards": list(self.board_cards),
            "player_states": self.player_states,
            "table_center": self.table_center,
            "btn_detection": self.btn_detection.to_dict() if self.btn_detection else None,
            "table_amount_state": dict(self.table_amount_state),
            "action_inference": dict(self.action_inference),
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
    action_state: Dict[str, Any] = field(default_factory=dict)
    actions_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
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
            "street_state": self.street_state,
            "positions": self.positions,
            "board_cards": list(self.board_cards),
            "player_states": self.player_states,
            "frames_log": self.frames_log,
            "errors": self.errors,
            "artifacts": self.artifacts,
            "processing_summary": self.processing_summary,
            "render_state_snapshot": self.render_state_snapshot,
            "conflict_state": self.conflict_state,
            "table_center": self.table_center,
            "table_amount_state": dict(self.table_amount_state),
            "action_state": dict(self.action_state),
            "actions_log": list(self.actions_log),
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
    action_annotations: Dict[str, Any] = field(default_factory=dict)

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
    return datetime.utcnow().isoformat(timespec="milliseconds")
