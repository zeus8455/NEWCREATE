from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Optional

import cv2
import numpy as np

from .action_inference import infer_actions
from .amount_normalization import normalize_amount_contributions
from .config import Settings
from .detectors import (
    build_board_crop,
    build_hero_crop,
    build_player_state_crop,
    build_table_amount_crop,
)
from .hand_state import HandStateManager, MATCH_STRONG, MATCH_WEAK
from .json_manager import save_hand_json
from .models import BBox, Detection, FrameAnalysis, HandState
from .render_state import build_render_state
from .solver_bridge import build_recommendation as build_solver_recommendation
from .storage import StorageManager
from .table_amount_logic import build_table_amount_state
from .table_logic import assign_positions, determine_hero_position
from .validators import (
    determine_street,
    validate_board_cards,
    validate_hero_cards,
    validate_player_state,
    validate_structure,
)


@dataclass(slots=True)
class PipelineResult:
    analysis: FrameAnalysis
    hand: Optional[HandState]
    render_state: Optional[dict]


def _draw_detections(image: np.ndarray, detections: list[Detection], color=(0, 255, 255)) -> np.ndarray:
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, det.label, (x1, max(12, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def _region_key(det: Detection) -> str:
    bbox = det.bbox
    return f"{det.label}@{round(bbox.x1, 1)}:{round(bbox.y1, 1)}:{round(bbox.x2, 1)}:{round(bbox.y2, 1)}"


def _localize(global_detections: list[Detection], origin: tuple[int, int]) -> list[Detection]:
    ox, oy = origin
    localized: list[Detection] = []
    for d in global_detections:
        localized.append(
            Detection(
                d.label,
                BBox(d.bbox.x1 - ox, d.bbox.y1 - oy, d.bbox.x2 - ox, d.bbox.y2 - oy),
                d.confidence,
            )
        )
    return localized


def _safe_jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_jsonable(v, depth=depth + 1) for v in value]
    if is_dataclass(value):
        return {f.name: _safe_jsonable(getattr(value, f.name), depth=depth + 1) for f in fields(value)}
    if hasattr(value, "__dict__"):
        return {
            str(k): _safe_jsonable(v, depth=depth + 1)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return repr(value)


def _build_legacy_solver_summary(payload: Any, status: str, errors: list[str]) -> dict[str, Any]:
    result_obj = payload.get("result") if isinstance(payload, dict) and "result" in payload else payload
    summary = {
        "status": status,
        "result": {
            "type": type(result_obj).__name__ if result_obj is not None else "NoneType",
            "raw_repr": repr(result_obj) if result_obj is not None else "",
        },
    }
    if errors:
        summary["errors"] = list(errors)
    return summary


def _build_solver_context_preview_fallback(analysis: FrameAnalysis, hand: HandState) -> dict[str, Any]:
    action_state = hand.action_state or {}
    preview: dict[str, Any] = {
        "street": analysis.street,
        "hero_position": hand.hero_position,
        "hero_cards": list(hand.hero_cards),
        "board_cards": list(hand.board_cards),
        "player_count": hand.player_count,
        "table_format": hand.table_format,
        "node_type_preview": action_state.get("node_type_preview"),
        "hero_context_preview": _safe_jsonable(action_state.get("hero_context_preview", {})),
        "action_history": _safe_jsonable(action_state.get("action_history", [])),
    }
    if analysis.street != "preflop":
        amount_norm = hand.amount_normalization or {}
        active_positions = [
            pos
            for pos, state in (hand.player_states or {}).items()
            if pos != hand.hero_position and not bool(state.get("is_fold", False))
        ]
        preview.update(
            {
                "pot_before_hero": amount_norm.get("total_pot_bb"),
                "villain_positions": active_positions,
            }
        )
    return preview


def _build_advisor_input_fallback(analysis: FrameAnalysis, hand: HandState, solver_context: dict[str, Any]) -> dict[str, Any]:
    action_state = hand.action_state or {}
    amount_norm = hand.amount_normalization or {}
    if analysis.street == "preflop":
        hero_preview = action_state.get("hero_context_preview", {}) or {}
        return {
            "context_type": "PreflopContext",
            "hero_hand": list(hand.hero_cards),
            "hero_pos": hand.hero_position,
            "node_type": action_state.get("node_type_preview"),
            "opener_pos": hero_preview.get("opener_pos") or action_state.get("opener_pos"),
            "three_bettor_pos": hero_preview.get("three_bettor_pos") or action_state.get("three_bettor_pos"),
            "four_bettor_pos": hero_preview.get("four_bettor_pos") or action_state.get("four_bettor_pos"),
            "limpers": hero_preview.get("limpers", action_state.get("limpers", 0)),
            "callers": hero_preview.get("callers", action_state.get("callers_after_open", 0)),
            "action_history": _safe_jsonable(action_state.get("action_history", [])),
            "meta": _safe_jsonable(hero_preview.get("meta", {})),
        }
    return {
        "context_type": "PostflopContext",
        "hero_hand": list(hand.hero_cards),
        "board": list(hand.board_cards),
        "pot_before_hero": amount_norm.get("total_pot_bb"),
        "to_call": solver_context.get("to_call", 0.0),
        "effective_stack": solver_context.get("effective_stack"),
        "hero_position": hand.hero_position,
        "villain_positions": list(solver_context.get("villain_positions", [])),
        "street": analysis.street,
        "player_count": hand.player_count,
        "line_context": _safe_jsonable(action_state.get("line_context", {})),
    }


def _get_solver_context_payload(analysis: FrameAnalysis, hand: HandState, payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("solver_context"), dict):
        return _safe_jsonable(payload.get("solver_context"))
    return _build_solver_context_preview_fallback(analysis, hand)


def _get_advisor_input_payload(analysis: FrameAnalysis, hand: HandState, payload: Any, solver_context: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("advisor_input"), dict):
        return _safe_jsonable(payload.get("advisor_input"))
    return _build_advisor_input_fallback(analysis, hand, solver_context)

def _build_engine_result_from_payload(payload: Any, status: str) -> dict[str, Any]:
    result_obj = payload.get("result") if isinstance(payload, dict) and "result" in payload else payload
    if result_obj is None:
        return {"status": status}
    return {
        "status": status,
        "decision_type": type(result_obj).__name__,
        "street": getattr(result_obj, "street", None),
        "engine_action": getattr(result_obj, "engine_action", None),
        "amount_to": getattr(result_obj, "amount_to", None),
        "size_pct": getattr(result_obj, "size_pct", None),
        "reason": getattr(result_obj, "reason", None),
        "confidence": getattr(result_obj, "confidence", None),
        "source": getattr(result_obj, "source", None),
        "actor_name": getattr(result_obj, "actor_name", None),
        "actor_pos": getattr(result_obj, "actor_pos", None),
    }


def _build_hero_decision_debug_from_payload(payload: Any, status: str, errors: list[str]) -> dict[str, Any]:
    result_obj = payload.get("result") if isinstance(payload, dict) and "result" in payload else payload
    debug = {
        "status": status,
        "type": type(result_obj).__name__ if result_obj is not None else "NoneType",
        "raw_repr": repr(result_obj) if result_obj is not None else "",
        "source": getattr(result_obj, "source", None) if result_obj is not None else None,
        "confidence": getattr(result_obj, "confidence", None) if result_obj is not None else None,
        "reason": getattr(result_obj, "reason", None) if result_obj is not None else None,
        "preflop": _safe_jsonable(getattr(result_obj, "preflop", None)) if result_obj is not None else None,
        "postflop": _safe_jsonable(getattr(result_obj, "postflop", None)) if result_obj is not None else None,
    }
    if errors:
        debug["errors"] = list(errors)
    return debug


def _apply_solver_payload(analysis: FrameAnalysis, hand: HandState, payload: Any) -> dict[str, Any]:
    """Populate stable solver fields on analysis/hand and return a legacy summary.

    CRITICAL INVARIANT:
    Solver output must live in normalized model fields, not only inside
    processing_summary/action_annotations. Keep the legacy summary for backwards
    compatibility, but treat HandState/RenderState solver fields as the source of truth.
    """
    status = "ok"
    errors: list[str] = []
    if isinstance(payload, dict):
        status = str(payload.get("status") or "ok")
        if payload.get("errors"):
            errors.extend([str(item) for item in payload.get("errors", [])])
        if payload.get("error"):
            errors.append(str(payload.get("error")))
    elif payload is None:
        status = "not_run"

    solver_context = _get_solver_context_payload(analysis, hand, payload)
    advisor_input = _get_advisor_input_payload(analysis, hand, payload, solver_context)
    if isinstance(payload, dict) and payload.get("warnings"):
        errors.extend([str(item) for item in payload.get("warnings", [])])
    engine_result = _build_engine_result_from_payload(payload, status)
    hero_debug = _build_hero_decision_debug_from_payload(payload, status, errors)
    legacy_summary = _build_legacy_solver_summary(payload, status, errors)

    analysis.solver_context_preview = dict(solver_context)
    analysis.solver_result = dict(engine_result)
    analysis.solver_status = status

    hand.solver_context = dict(solver_context)
    hand.advisor_input = dict(advisor_input)
    hand.engine_result = dict(engine_result)
    hand.solver_status = status
    hand.solver_errors = list(errors)
    hand.hero_decision_debug = dict(hero_debug)

    hand.processing_summary["solver_bridge"] = dict(legacy_summary)
    return legacy_summary


class PokerVisionPipeline:
    def __init__(self, settings: Settings, detector_backend, storage: StorageManager, hand_manager: HandStateManager):
        self.settings = settings
        self.detector_backend = detector_backend
        self.storage = storage
        self.hand_manager = hand_manager

    def process_frame(self, frame) -> PipelineResult:
        self.hand_manager.mark_stale_if_needed(frame.timestamp)
        active = self.detector_backend.detect_active_hero(frame)
        overlay = _draw_detections(frame.image, active, color=(0, 255, 255))
        if not active:
            analysis = FrameAnalysis(frame.frame_id, frame.timestamp, False)
            return PipelineResult(
                analysis=analysis,
                hand=self.hand_manager.active_hand,
                render_state=(self.hand_manager.active_hand.render_state_snapshot or None)
                if self.hand_manager.active_hand
                else None,
            )

        structure_dets = self.detector_backend.detect_structure(frame)
        structure_val = validate_structure(
            structure_dets,
            self.settings.duplicate_iou_threshold,
            self.settings.duplicate_center_distance_px,
        )
        overlay = _draw_detections(overlay, structure_dets, color=(0, 200, 0))
        analysis = FrameAnalysis(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            active_hero_found=True,
            structure_detections=structure_dets,
            validation={"structure_ok": structure_val.ok},
        )
        if not structure_val.ok:
            analysis.errors.extend(structure_val.errors)
            self.storage.save_failure_artifacts(
                "structure",
                frame.frame_id,
                frame.image,
                overlay_frame=overlay,
                debug_images=[overlay],
            )
            self.hand_manager.register_error(
                self.hand_manager.active_hand,
                "structure",
                "; ".join(structure_val.errors),
                frame.frame_id,
                True,
            )
            return PipelineResult(analysis=analysis, hand=self.hand_manager.active_hand, render_state=None)

        cleaned = structure_val.meta["cleaned"]
        btn = structure_val.meta["btn"][0]
        seats = structure_val.meta["seats"]
        player_count = int(structure_val.meta["player_count"])
        table_format = str(structure_val.meta["table_format"])
        analysis.player_count = player_count
        analysis.table_format = table_format
        analysis.btn_detection = btn

        street_val = determine_street(cleaned)
        if not street_val.ok:
            analysis.errors.extend(street_val.errors)
            self.storage.save_failure_artifacts(
                "street",
                frame.frame_id,
                frame.image,
                overlay_frame=overlay,
                debug_images=[overlay],
            )
            self.hand_manager.register_error(
                self.hand_manager.active_hand,
                "street",
                "; ".join(street_val.errors),
                frame.frame_id,
                True,
            )
            return PipelineResult(analysis=analysis, hand=self.hand_manager.active_hand, render_state=None)

        street = str(street_val.meta["street"])
        analysis.street = street
        table_center, positions = assign_positions(seats, btn, player_count)
        hero_position = determine_hero_position(positions, active, self.settings.seat_match_max_distance_px)
        positions[hero_position]["is_hero"] = True
        occupied_positions = list(positions.keys())
        analysis.positions = positions
        analysis.hero_position = hero_position
        analysis.occupied_positions = occupied_positions
        analysis.table_center = table_center

        player_crops: dict[str, np.ndarray] = {}
        player_overlays: dict[str, np.ndarray] = {}
        player_validation: dict[str, bool] = {}
        for position, payload in positions.items():
            player_bbox = BBox(**payload["bbox"])
            player_crop, player_origin = build_player_state_crop(frame.image, player_bbox, self.settings)
            player_global_dets = self.detector_backend.detect_player_state(frame, player_bbox)
            player_local_dets = _localize(player_global_dets, player_origin)
            player_overlay = _draw_detections(player_crop, player_local_dets, color=(150, 220, 255))
            player_crops[position] = player_crop
            player_overlays[position] = player_overlay
            analysis.player_state_detections[position] = player_global_dets
            player_val = validate_player_state(
                position,
                player_local_dets,
                crop_height=player_crop.shape[0],
                lower_band_ratio=self.settings.player_state_lower_band_ratio,
                iou_threshold=self.settings.player_state_token_iou_threshold,
                center_threshold=self.settings.player_state_token_center_threshold_px,
            )
            player_validation[position] = player_val.ok
            analysis.player_states[position] = player_val.meta["player_state"]
            if player_val.errors:
                analysis.warnings.extend([f"{position}: {error}" for error in player_val.errors])
            if player_val.warnings:
                analysis.warnings.extend([f"{position}: {warning}" for warning in player_val.warnings])
        analysis.validation["player_states_ok"] = player_validation

        table_amount_regions = self.detector_backend.detect_table_amount_regions(frame)
        analysis.table_amount_region_detections = table_amount_regions
        overlay = _draw_detections(overlay, table_amount_regions, color=(255, 100, 255))
        table_amount_crops: dict[str, np.ndarray] = {}
        table_amount_overlays: dict[str, np.ndarray] = {}
        table_amount_digit_map: dict[str, list[Detection]] = {}
        for idx, region_det in enumerate(table_amount_regions):
            region_id = f"{region_det.label}_{idx}"
            region_key = _region_key(region_det)
            region_crop, region_origin = build_table_amount_crop(frame.image, region_det.bbox, self.settings)
            digit_global = self.detector_backend.detect_table_amount_digits(frame, region_det.bbox)
            digit_local = _localize(digit_global, region_origin)
            region_overlay = _draw_detections(region_crop, digit_local, color=(255, 180, 0))
            table_amount_crops[region_id] = region_crop
            table_amount_overlays[region_id] = region_overlay
            table_amount_digit_map[region_id] = digit_local
            table_amount_digit_map[region_key] = digit_local
            analysis.table_amount_digit_detections[region_id] = digit_global
            analysis.table_amount_digit_detections[region_key] = digit_global

        table_amount_val = build_table_amount_state(
            table_amount_regions,
            table_amount_digit_map,
            positions,
            analysis.player_states,
            table_center,
            street,
            self.settings,
        )
        analysis.table_amount_state = table_amount_val.meta["table_amount_state"]
        if table_amount_val.warnings:
            analysis.warnings.extend(table_amount_val.warnings)
        if table_amount_val.errors:
            analysis.warnings.extend(table_amount_val.errors)
        analysis.validation["table_amount_ok"] = table_amount_val.ok

        analysis.amount_normalization = normalize_amount_contributions(
            analysis.table_amount_state,
            street,
            int(analysis.player_count or 0),
            list(analysis.occupied_positions),
            analysis.hero_position,
            analysis.player_states,
        )
        if analysis.amount_normalization.get("warnings"):
            analysis.warnings.extend(list(analysis.amount_normalization.get("warnings", [])))

        if self.settings.action_reconstruction_enabled:
            analysis.action_inference = infer_actions(self.hand_manager.active_hand, analysis, self.settings)
        else:
            analysis.action_inference = {}

        hero_bbox = BBox(**positions[hero_position]["bbox"])
        hero_crop, hero_origin = build_hero_crop(frame.image, hero_bbox, self.settings)
        hero_card_dets = self.detector_backend.detect_hero_cards(frame, hero_bbox)
        hero_local_dets = _localize(hero_card_dets, hero_origin)
        hero_overlay = _draw_detections(hero_crop, hero_local_dets, color=(255, 255, 0))
        analysis.hero_card_detections = hero_card_dets
        hero_val = validate_hero_cards(hero_local_dets)
        analysis.validation["hero_cards_ok"] = hero_val.ok
        if hero_val.ok:
            analysis.hero_cards = hero_val.meta["cards"]
        else:
            analysis.errors.extend(hero_val.errors)
            self.storage.save_failure_artifacts(
                "hero_cards",
                frame.frame_id,
                frame.image,
                overlay_frame=overlay,
                hero_crop=hero_crop,
                hero_overlay=hero_overlay,
                player_crops=player_crops,
                player_overlays=player_overlays,
                table_amount_crops=table_amount_crops,
                table_amount_overlays=table_amount_overlays,
                debug_images=[overlay, hero_overlay],
            )
            self.hand_manager.register_error(
                self.hand_manager.active_hand,
                "hero_cards",
                "; ".join(hero_val.errors),
                frame.frame_id,
                True,
            )
            return PipelineResult(analysis=analysis, hand=self.hand_manager.active_hand, render_state=None)

        board_crop = None
        board_overlay = None
        if street != "preflop":
            marker_label = street.capitalize()
            marker = next((d for d in cleaned if d.label == marker_label), None)
            if marker is None:
                analysis.errors.append(f"Missing {marker_label} marker")
                self.storage.save_failure_artifacts(
                    "board_marker",
                    frame.frame_id,
                    frame.image,
                    overlay_frame=overlay,
                    hero_crop=hero_crop,
                    hero_overlay=hero_overlay,
                    player_crops=player_crops,
                    player_overlays=player_overlays,
                    table_amount_crops=table_amount_crops,
                    table_amount_overlays=table_amount_overlays,
                    debug_images=[overlay, hero_overlay],
                )
                self.hand_manager.register_error(
                    self.hand_manager.active_hand,
                    "board_marker",
                    f"Missing {marker_label} marker",
                    frame.frame_id,
                    True,
                )
                return PipelineResult(analysis=analysis, hand=self.hand_manager.active_hand, render_state=None)

            board_bbox = marker.bbox
            board_crop, board_origin = build_board_crop(frame.image, board_bbox, self.settings)
            board_global_dets = self.detector_backend.detect_board_cards(frame, board_bbox, street)
            board_local_dets = _localize(board_global_dets, board_origin)
            board_overlay = _draw_detections(board_crop, board_local_dets, color=(255, 150, 0))
            analysis.board_card_detections = board_global_dets
            board_val = validate_board_cards(board_local_dets, street)
            analysis.validation["board_cards_ok"] = board_val.ok
            if board_val.ok:
                analysis.board_cards = board_val.meta["cards"]
            else:
                retry_ok = False
                if self.settings.max_retry_per_stage > 0:
                    expanded_marker = BBox(board_bbox.x1 - 20, board_bbox.y1 - 10, board_bbox.x2 + 20, board_bbox.y2 + 10)
                    retry_global = self.detector_backend.detect_board_cards(frame, expanded_marker, street)
                    retry_crop, retry_origin = build_board_crop(frame.image, expanded_marker, self.settings)
                    retry_local = _localize(retry_global, retry_origin)
                    retry_val = validate_board_cards(retry_local, street)
                    if retry_val.ok:
                        board_crop = retry_crop
                        board_overlay = _draw_detections(retry_crop, retry_local, color=(255, 150, 0))
                        analysis.board_card_detections = retry_global
                        analysis.board_cards = retry_val.meta["cards"]
                        retry_ok = True
                if not retry_ok:
                    analysis.errors.extend(board_val.errors)
                    self.storage.save_failure_artifacts(
                        "board_cards",
                        frame.frame_id,
                        frame.image,
                        overlay_frame=overlay,
                        hero_crop=hero_crop,
                        hero_overlay=hero_overlay,
                        board_crop=board_crop,
                        board_overlay=board_overlay,
                        player_crops=player_crops,
                        player_overlays=player_overlays,
                        table_amount_crops=table_amount_crops,
                        table_amount_overlays=table_amount_overlays,
                        debug_images=[overlay, hero_overlay] + ([board_overlay] if board_overlay is not None else []),
                    )
                    self.hand_manager.register_error(
                        self.hand_manager.active_hand,
                        "board_cards",
                        "; ".join(board_val.errors),
                        frame.frame_id,
                        True,
                    )
                    return PipelineResult(analysis=analysis, hand=self.hand_manager.active_hand, render_state=None)

        previous_street = self.hand_manager.active_hand.street_state.get("current_street") if self.hand_manager.active_hand else None
        hand, decision, created_new = self.hand_manager.update_or_create(analysis)

        # CRITICAL INVARIANT:
        # The main pipeline must invoke the solver bridge after HandState is updated and
        # before RenderState/hand.json are built. Do not move this logic back into the
        # external launcher, otherwise main-pipeline semantics and persisted state drift again.
        solver_result: Any = None
        try:
            solver_result = build_solver_recommendation(analysis, hand, self.settings)
        except Exception as exc:  # pragma: no cover - defensive safety for runtime integration
            solver_result = {
                "status": "error",
                "error": str(exc),
                "result": None,
            }
        legacy_solver_summary = _apply_solver_payload(analysis, hand, solver_result)

        repeated_same_street = (
            not created_new
            and previous_street == analysis.street
            and decision.status in {MATCH_STRONG, MATCH_WEAK}
        )
        save_only_raw = repeated_same_street and not self.settings.normal_mode_save_repeated_frames
        artifacts = self.storage.save_pipeline_artifacts(
            hand.hand_id,
            frame.frame_id,
            frame.image,
            overlay_frame=overlay,
            hero_crop=hero_crop,
            hero_overlay=hero_overlay,
            board_crop=board_crop,
            board_overlay=board_overlay,
            player_crops=player_crops,
            player_overlays=player_overlays,
            table_amount_crops=table_amount_crops,
            table_amount_overlays=table_amount_overlays,
            debug_images=[
                img
                for img in [overlay, hero_overlay, board_overlay, *player_overlays.values(), *table_amount_overlays.values()]
                if img is not None
            ],
            save_only_raw=save_only_raw,
        )
        hand.artifacts = artifacts.to_dict()
        hand.processing_summary["solver_bridge"] = dict(legacy_solver_summary)

        render_state = build_render_state(hand, frame.frame_id, frame.timestamp).to_dict()
        hand.render_state_snapshot = render_state
        self.storage.save_render_state(hand.hand_id, render_state)
        save_hand_json(self.storage.hand_dir(hand.hand_id) / "hand.json", hand)
        return PipelineResult(analysis=analysis, hand=hand, render_state=render_state)
