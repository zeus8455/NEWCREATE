from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from pokervision.capture import MockFrameSource, ScreenFrameSource
    from pokervision.config import get_default_settings
    from pokervision.detectors import MockDetectorBackend, YoloDetectorBackend
    from pokervision.hand_state import HandStateManager
    from pokervision.pipeline import PokerVisionPipeline
    from pokervision.solver_runtime import shutdown_solver_runtime
    from pokervision.storage import StorageManager
    from pokervision.ui_bridge import SharedState
else:
    from .capture import MockFrameSource, ScreenFrameSource
    from .config import get_default_settings
    from .detectors import MockDetectorBackend, YoloDetectorBackend
    from .hand_state import HandStateManager
    from .pipeline import PokerVisionPipeline
    from .solver_runtime import shutdown_solver_runtime
    from .storage import StorageManager
    from .ui_bridge import SharedState


def run_headless(args) -> int:
    settings = get_default_settings()
    settings.debug_mode = True
    source = MockFrameSource(*settings.mock_table_size) if args.mock else ScreenFrameSource(settings.monitor_index)
    detector = MockDetectorBackend(settings) if args.mock else YoloDetectorBackend(settings)
    storage = StorageManager(settings)
    hand_manager = HandStateManager(settings.schema_version, settings.hand_stale_timeout_sec, settings.hand_close_timeout_sec)
    pipeline = PokerVisionPipeline(settings, detector, storage, hand_manager)
    iterations = args.iterations or 8

    try:
        for _ in range(iterations):
            frame = source.next_frame()
            result = pipeline.process_frame(frame)
            print(
                {
                    "frame_id": result.analysis.frame_id,
                    "active_hero": result.analysis.active_hero_found,
                    "street": result.analysis.street,
                    "hero": result.analysis.hero_position,
                    "hero_cards": result.analysis.hero_cards,
                    "board": result.analysis.board_cards,
                    "errors": result.analysis.errors,
                    "hand_id": result.hand.hand_id if result.hand else None,
                }
            )
        return 0
    finally:
        shutdown_solver_runtime(wait=False, cancel_futures=True)


def run_with_ui(args) -> int:  # pragma: no cover
    try:
        from PySide6 import QtWidgets

        if __package__ in {None, ""}:
            from pokervision.visualizer import DebugMonitorWindow
            from pokervision.table_renderer import PokerTableWindow
        else:
            from .visualizer import DebugMonitorWindow
            from .table_renderer import PokerTableWindow
    except Exception as exc:
        print(f"UI mode is unavailable: {exc}")
        return 1

    settings = get_default_settings()
    source = MockFrameSource(*settings.mock_table_size) if args.mock else ScreenFrameSource(settings.monitor_index)
    detector = MockDetectorBackend(settings) if args.mock else YoloDetectorBackend(settings)
    storage = StorageManager(settings)
    hand_manager = HandStateManager(settings.schema_version, settings.hand_stale_timeout_sec, settings.hand_close_timeout_sec)
    pipeline = PokerVisionPipeline(settings, detector, storage, hand_manager)

    shared = SharedState()
    stop_event = threading.Event()

    def worker():
        while not stop_event.is_set():
            frame = source.next_frame()
            shared.update_frame(frame.image)
            result = pipeline.process_frame(frame)
            if result.render_state:
                shared.update_render_state(result.render_state)
            shared.update_status(
                {
                    "frame_id": result.analysis.frame_id,
                    "street": result.analysis.street,
                    "errors": result.analysis.errors,
                }
            )
            time.sleep(settings.frame_debounce_ms / 1000.0)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    debug_window = DebugMonitorWindow(shared, settings.ui_refresh_ms)
    table_window = PokerTableWindow(shared, settings)
    debug_window.show()
    table_window.show()
    try:
        code = app.exec()
    finally:
        stop_event.set()
        shutdown_solver_runtime(wait=False, cancel_futures=True)
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PokerVision")
    parser.add_argument("--mock", action="store_true", help="Use mock frame source and mock detectors")
    parser.add_argument("--real", action="store_true", help="Use real screen capture and YOLO detectors")
    parser.add_argument("--headless", action="store_true", help="Run without UI")
    parser.add_argument("--iterations", type=int, default=0, help="Headless iteration count")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.real:
        args.mock = False
    elif not args.mock:
        args.mock = True
    if args.headless:
        return run_headless(args)
    return run_with_ui(args)


if __name__ == "__main__":
    raise SystemExit(main())
