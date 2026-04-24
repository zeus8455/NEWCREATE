from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .config import Settings
from .models import PipelineArtifacts


def _hand_number_suffix(hand_id: str) -> str:
    raw = str(hand_id or "").strip()
    tail = raw.split("_")[-1] if raw else ""
    return tail or raw or "unknown"


SLOT_IDS = tuple(f"table_{index:02d}" for index in range(1, 7))
DEFAULT_SLOT_ID = SLOT_IDS[0]
SLOT_SUBDIRS = ("hands", "temp", "render", "logs")


def _normalize_slot_id(slot_id: Optional[str]) -> str:
    candidate = str(slot_id or DEFAULT_SLOT_ID).strip()
    if candidate not in SLOT_IDS:
        raise KeyError(f"Unknown slot_id: {candidate}")
    return candidate


def resolve_slot_paths(base_root: Path | str, slot_id: str) -> dict[str, Path]:
    normalized = _normalize_slot_id(slot_id)
    root = Path(base_root).expanduser().resolve()
    slot_root = root / "tables" / normalized
    return {
        "slot_root": slot_root,
        "hands": slot_root / "hands",
        "temp": slot_root / "temp",
        "render": slot_root / "render",
        "logs": slot_root / "logs",
    }


class StorageManager:
    def __init__(self, settings: Settings, slot_id: str = DEFAULT_SLOT_ID):
        self.settings = settings
        self.slot_id = _normalize_slot_id(slot_id)
        self._slot_paths = resolve_slot_paths(self.settings.root_dir, self.slot_id)
        self._ensure_root_structure()

    def _ensure_root_structure(self) -> None:
        root = Path(self.settings.root_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        tables_root = root / "tables"
        tables_root.mkdir(parents=True, exist_ok=True)

        for slot_id in SLOT_IDS:
            paths = resolve_slot_paths(root, slot_id)
            paths["slot_root"].mkdir(parents=True, exist_ok=True)
            paths["hands"].mkdir(parents=True, exist_ok=True)
            paths["logs"].mkdir(parents=True, exist_ok=True)
            temp_dir = paths["temp"]
            if temp_dir.exists() and not self.settings.keep_temp_on_exit:
                shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir.mkdir(parents=True, exist_ok=True)
            paths["render"].mkdir(parents=True, exist_ok=True)

        self._slot_paths = resolve_slot_paths(root, self.slot_id)

    def set_slot_id(self, slot_id: str) -> None:
        self.slot_id = _normalize_slot_id(slot_id)
        self._slot_paths = resolve_slot_paths(self.settings.root_dir, self.slot_id)
        self._slot_paths["slot_root"].mkdir(parents=True, exist_ok=True)
        for name in SLOT_SUBDIRS:
            self._slot_paths[name].mkdir(parents=True, exist_ok=True)

    def slot_paths(self, slot_id: Optional[str] = None) -> dict[str, Path]:
        normalized = _normalize_slot_id(slot_id or self.slot_id)
        if normalized == self.slot_id:
            return dict(self._slot_paths)
        return resolve_slot_paths(self.settings.root_dir, normalized)

    def hand_dir(self, hand_id: str) -> Path:
        path = self._slot_paths["hands"] / hand_id
        for sub in [
            path / "raw_frames",
            path / "overlays",
            path / "crops" / "hero",
            path / "crops" / "board",
            path / "crops" / "players",
            path / "crops" / "table_amount",
            path / "debug",
            path / "render",
        ]:
            sub.mkdir(parents=True, exist_ok=True)
        return path

    def failure_dir(self, stage: str, frame_id: str) -> Path:
        path = self._slot_paths["temp"] / "failed_frames" / stage / frame_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_image(self, path: Path, image: np.ndarray) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
        return str(path)

    def save_json(self, path: Path, payload: dict[str, Any]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return str(path)

    def save_render_state(self, hand_id: str, render_state: dict) -> str:
        suffix = _hand_number_suffix(hand_id)
        path = self.hand_dir(hand_id) / "render" / f"last_render_state_{suffix}.json"
        return self.save_json(path, render_state)

    def save_pipeline_artifacts(
        self,
        hand_id: str,
        frame_id: str,
        raw_frame: np.ndarray,
        overlay_frame: Optional[np.ndarray] = None,
        hero_crop: Optional[np.ndarray] = None,
        hero_overlay: Optional[np.ndarray] = None,
        board_crop: Optional[np.ndarray] = None,
        board_overlay: Optional[np.ndarray] = None,
        player_crops: Optional[dict[str, np.ndarray]] = None,
        player_overlays: Optional[dict[str, np.ndarray]] = None,
        table_amount_crops: Optional[dict[str, np.ndarray]] = None,
        table_amount_overlays: Optional[dict[str, np.ndarray]] = None,
        debug_images: Optional[list[np.ndarray]] = None,
        save_only_raw: bool = False,
    ) -> PipelineArtifacts:
        hand_dir = self.hand_dir(hand_id)
        artifacts = PipelineArtifacts()
        artifacts.raw_frame_path = self.save_image(hand_dir / "raw_frames" / f"{frame_id}.png", raw_frame)
        if overlay_frame is not None and not save_only_raw:
            artifacts.overlay_frame_path = self.save_image(hand_dir / "overlays" / f"{frame_id}_overlay.png", overlay_frame)
        if hero_crop is not None and not save_only_raw:
            artifacts.hero_crop_path = self.save_image(hand_dir / "crops" / "hero" / f"{frame_id}_hero.png", hero_crop)
        if hero_overlay is not None and not save_only_raw:
            artifacts.hero_overlay_path = self.save_image(hand_dir / "crops" / "hero" / f"{frame_id}_hero_overlay.png", hero_overlay)
        if board_crop is not None and not save_only_raw:
            artifacts.board_crop_path = self.save_image(hand_dir / "crops" / "board" / f"{frame_id}_board.png", board_crop)
        if board_overlay is not None and not save_only_raw:
            artifacts.board_overlay_path = self.save_image(hand_dir / "crops" / "board" / f"{frame_id}_board_overlay.png", board_overlay)
        if player_crops and not save_only_raw:
            for position, image in player_crops.items():
                artifacts.player_crop_paths[position] = self.save_image(hand_dir / "crops" / "players" / f"{frame_id}_{position}.png", image)
        if player_overlays and not save_only_raw:
            for position, image in player_overlays.items():
                artifacts.player_overlay_paths[position] = self.save_image(hand_dir / "crops" / "players" / f"{frame_id}_{position}_overlay.png", image)
        if table_amount_crops and not save_only_raw:
            for region_id, image in table_amount_crops.items():
                artifacts.table_amount_crop_paths[region_id] = self.save_image(hand_dir / "crops" / "table_amount" / f"{frame_id}_{region_id}.png", image)
        if table_amount_overlays and not save_only_raw:
            for region_id, image in table_amount_overlays.items():
                artifacts.table_amount_overlay_paths[region_id] = self.save_image(hand_dir / "crops" / "table_amount" / f"{frame_id}_{region_id}_overlay.png", image)
        if debug_images and self.settings.debug_mode and not save_only_raw:
            for idx, image in enumerate(debug_images):
                artifacts.debug_paths.append(self.save_image(hand_dir / "debug" / f"{frame_id}_debug_{idx}.png", image))
        return artifacts

    def save_failure_artifacts(
        self,
        stage: str,
        frame_id: str,
        raw_frame: np.ndarray,
        overlay_frame: Optional[np.ndarray] = None,
        hero_crop: Optional[np.ndarray] = None,
        hero_overlay: Optional[np.ndarray] = None,
        board_crop: Optional[np.ndarray] = None,
        board_overlay: Optional[np.ndarray] = None,
        player_crops: Optional[dict[str, np.ndarray]] = None,
        player_overlays: Optional[dict[str, np.ndarray]] = None,
        table_amount_crops: Optional[dict[str, np.ndarray]] = None,
        table_amount_overlays: Optional[dict[str, np.ndarray]] = None,
        debug_images: Optional[list[np.ndarray]] = None,
        extra_images: Optional[dict[str, np.ndarray]] = None,
    ) -> PipelineArtifacts:
        out_dir = self.failure_dir(stage, frame_id)
        artifacts = PipelineArtifacts()
        artifacts.raw_frame_path = self.save_image(out_dir / f"{frame_id}.png", raw_frame)
        if overlay_frame is not None:
            artifacts.overlay_frame_path = self.save_image(out_dir / f"{frame_id}_overlay.png", overlay_frame)
        if hero_crop is not None:
            artifacts.hero_crop_path = self.save_image(out_dir / f"{frame_id}_hero.png", hero_crop)
        if hero_overlay is not None:
            artifacts.hero_overlay_path = self.save_image(out_dir / f"{frame_id}_hero_overlay.png", hero_overlay)
        if board_crop is not None:
            artifacts.board_crop_path = self.save_image(out_dir / f"{frame_id}_board.png", board_crop)
        if board_overlay is not None:
            artifacts.board_overlay_path = self.save_image(out_dir / f"{frame_id}_board_overlay.png", board_overlay)
        if player_crops:
            for position, image in player_crops.items():
                artifacts.player_crop_paths[position] = self.save_image(out_dir / f"{frame_id}_{position}.png", image)
        if player_overlays:
            for position, image in player_overlays.items():
                artifacts.player_overlay_paths[position] = self.save_image(out_dir / f"{frame_id}_{position}_overlay.png", image)
        if table_amount_crops:
            for region_id, image in table_amount_crops.items():
                artifacts.table_amount_crop_paths[region_id] = self.save_image(out_dir / f"{frame_id}_{region_id}.png", image)
        if table_amount_overlays:
            for region_id, image in table_amount_overlays.items():
                artifacts.table_amount_overlay_paths[region_id] = self.save_image(out_dir / f"{frame_id}_{region_id}_overlay.png", image)
        if extra_images:
            for name, image in extra_images.items():
                safe_name = str(name).strip().replace(" ", "_") or "extra"
                artifacts.extra_paths[safe_name] = self.save_image(out_dir / f"{frame_id}_{safe_name}.png", image)
        if debug_images and self.settings.save_debug_on_error:
            for idx, image in enumerate(debug_images):
                artifacts.debug_paths.append(self.save_image(out_dir / f"{frame_id}_debug_{idx}.png", image))
        return artifacts

    def save_failure_state(
        self,
        stage: str,
        frame_id: str,
        failed_frame_payload: dict[str, Any],
        ui_error_state_payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, str]:
        out_dir = self.failure_dir(stage, frame_id)
        failed_frame_path = str(out_dir / "failed_frame.json")
        ui_error_state_path = str(out_dir / "ui_error_state.json")

        failed_frame_payload = dict(failed_frame_payload or {})
        failed_frame_payload["failed_frame_json_path"] = failed_frame_path
        failed_frame_payload["ui_error_state_path"] = ui_error_state_path if ui_error_state_payload is not None else None
        self.save_json(Path(failed_frame_path), failed_frame_payload)

        paths = {"failed_frame_json_path": failed_frame_path}
        if ui_error_state_payload is not None:
            ui_error_state_payload = dict(ui_error_state_payload or {})
            ui_error_state_payload["failed_frame_json_path"] = failed_frame_path
            ui_error_state_payload["ui_error_state_path"] = ui_error_state_path
            self.save_json(Path(ui_error_state_path), ui_error_state_payload)
            paths["ui_error_state_path"] = ui_error_state_path
        return paths
