from __future__ import annotations

"""Dedicated solver runtime layer for CPU-heavy recommendation jobs.

This module owns the warm ProcessPoolExecutor used by PokerVision for solver
execution. The pipeline stays an orchestrator, `context_projection.py` stays the
canonical context source, and `solver_bridge.py` remains the adapter that turns
hand state into solver jobs.

Step 1 goal:
- introduce a distinct runtime layer
- keep a warm process pool alive
- allow solver_bridge to submit postflop recommendation jobs into that pool
- keep preview/fingerprint building local and unchanged

Later optimization steps can plug parallel Monte Carlo and parallel villain-range
building *inside* the worker process without changing the pipeline contract.
"""

from dataclasses import dataclass
from concurrent.futures import Future, ProcessPoolExecutor, TimeoutError
from multiprocessing import get_context
from threading import Lock
from typing import Any, Optional
import importlib
import os


@dataclass(slots=True)
class SolverRuntimeJob:
    """Serializable solver job payload sent to worker processes."""

    analysis: Any
    hand: Any
    settings: Any = None


@dataclass(slots=True)
class SolverRuntimeResult:
    """Envelope returned by worker processes."""

    payload: Any


def _import_solver_bridge_module():
    """Import solver_bridge lazily inside worker processes."""

    candidates: list[str] = []
    if __package__:
        candidates.append(f"{__package__}.solver_bridge")
    candidates.append("pokervision.solver_bridge")
    candidates.append("solver_bridge")

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except Exception as exc:  # pragma: no cover - exercised at runtime
            last_error = exc
    if last_error is None:  # pragma: no cover
        raise ImportError("Could not import solver_bridge")
    raise last_error


def _warm_worker() -> str:
    """Force worker process start and module import once."""

    _import_solver_bridge_module()
    return "ready"


def _solve_postflop_job(job: SolverRuntimeJob) -> SolverRuntimeResult:
    """Execute one recommendation job inside a worker process."""

    module = _import_solver_bridge_module()
    bridge = module.EngineBridge(settings=job.settings, enable_runtime=False)
    payload = bridge._build_recommendation_inline(job.analysis, job.hand)
    if isinstance(payload, dict):
        processing_summary = payload.get("processing_summary")
        if isinstance(processing_summary, dict):
            solver_bridge = processing_summary.get("solver_bridge")
            if isinstance(solver_bridge, dict):
                updated = dict(solver_bridge)
                updated["runtime_used"] = True
                processing_summary = dict(processing_summary)
                processing_summary["solver_bridge"] = updated
                payload = dict(payload)
                payload["processing_summary"] = processing_summary
    return SolverRuntimeResult(payload=payload)


class SolverRuntime:
    """Warm process-pool owner used by solver_bridge."""

    def __init__(self, settings: Any = None) -> None:
        self.settings = settings
        self.enabled = bool(self._setting("solver_runtime_enabled", True))
        self._executor: Optional[ProcessPoolExecutor] = None
        self._lock = Lock()
        self._started = False

    def _setting(self, name: str, default: Any) -> Any:
        if self.settings is None:
            return default
        return getattr(self.settings, name, default)

    def _worker_count(self) -> int:
        configured = int(self._setting("solver_runtime_workers", 1) or 1)
        cpu_count = max(1, os.cpu_count() or 1)
        return max(1, min(configured, cpu_count))

    def _start_method(self) -> str:
        method = str(self._setting("solver_runtime_start_method", "spawn") or "spawn").strip().lower()
        return method or "spawn"

    def _timeout_sec(self) -> float:
        value = self._setting("solver_runtime_timeout_sec", 30.0)
        try:
            return max(0.1, float(value))
        except (TypeError, ValueError):
            return 30.0

    def _warm_start_enabled(self) -> bool:
        return bool(self._setting("solver_runtime_warm_start", True))

    def is_available(self) -> bool:
        return self.enabled

    def ensure_started(self) -> None:
        if not self.enabled or self._started:
            return
        with self._lock:
            if self._started:
                return
            context = get_context(self._start_method())
            self._executor = ProcessPoolExecutor(
                max_workers=self._worker_count(),
                mp_context=context,
            )
            if self._warm_start_enabled() and self._executor is not None:
                warmup_futures = [self._executor.submit(_warm_worker) for _ in range(self._worker_count())]
                for future in warmup_futures:
                    future.result(timeout=self._timeout_sec())
            self._started = True

    def submit_postflop_recommendation(self, analysis: Any, hand: Any) -> Future:
        if not self.enabled:
            raise RuntimeError("solver runtime is disabled")
        self.ensure_started()
        if self._executor is None:  # pragma: no cover - defensive
            raise RuntimeError("solver runtime executor is unavailable")
        job = SolverRuntimeJob(analysis=analysis, hand=hand, settings=self.settings)
        return self._executor.submit(_solve_postflop_job, job)

    def compute_postflop_recommendation(self, analysis: Any, hand: Any) -> Any:
        future = self.submit_postflop_recommendation(analysis, hand)
        try:
            result = future.result(timeout=self._timeout_sec())
        except TimeoutError:
            future.cancel()
            raise TimeoutError(
                f"solver runtime timed out after {self._timeout_sec():.1f}s"
            )
        return result.payload

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        with self._lock:
            executor = self._executor
            self._executor = None
            self._started = False
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=cancel_futures)


_RUNTIME_SINGLETON: Optional[SolverRuntime] = None
_RUNTIME_LOCK = Lock()


def get_solver_runtime(settings: Any = None) -> Optional[SolverRuntime]:
    """Return the singleton runtime for the current process."""

    enabled = True if settings is None else bool(getattr(settings, "solver_runtime_enabled", True))
    if not enabled:
        return None

    global _RUNTIME_SINGLETON
    with _RUNTIME_LOCK:
        if _RUNTIME_SINGLETON is None:
            _RUNTIME_SINGLETON = SolverRuntime(settings=settings)
        elif settings is not None and _RUNTIME_SINGLETON.settings is None:
            _RUNTIME_SINGLETON.settings = settings
        return _RUNTIME_SINGLETON


def shutdown_solver_runtime(*, wait: bool = True, cancel_futures: bool = False) -> None:
    global _RUNTIME_SINGLETON
    with _RUNTIME_LOCK:
        runtime = _RUNTIME_SINGLETON
        _RUNTIME_SINGLETON = None
    if runtime is not None:
        runtime.shutdown(wait=wait, cancel_futures=cancel_futures)


__all__ = [
    "SolverRuntime",
    "SolverRuntimeJob",
    "SolverRuntimeResult",
    "get_solver_runtime",
    "shutdown_solver_runtime",
]
