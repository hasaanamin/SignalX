"""Feature analysis worker with bounded queue, explicit lifecycle, and quality gating."""

from __future__ import annotations

import collections
import enum
import queue
import threading
from collections.abc import Callable
from typing import Any

from signal_processing.capture.sidecar import CaptureSidecar
from signal_processing.context import AnalysisConfig, SignalContext
from signal_processing.core.contracts import AnalysisResult
from signal_processing.features import FeatureAggregator, build_default_feature_registry


class WorkerState(enum.Enum):
    """Lifecycle states for FeatureAnalysisWorker."""

    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


_SENTINEL = object()


class FeatureAnalysisWorker:
    """Consumes completed event captures, applies quality gates, and extracts feature vectors.

    Supports synchronous mode for deterministic testing or an asynchronous background worker
    thread with a bounded queue and bounded result delivery to prevent slow feature extraction
    from blocking real-time acquisition or unbounded RAM accumulation.
    """

    def __init__(
        self,
        max_queue_size: int = 32,
        max_results: int = 128,
        skip_incomplete_quality: bool = True,
        synchronous: bool = True,
        analysis_config: AnalysisConfig | None = None,
        result_sink: Callable[[CaptureSidecar], None] | None = None,
    ) -> None:
        if isinstance(max_queue_size, bool) or not isinstance(max_queue_size, int) or max_queue_size < 1:
            raise ValueError(f"max_queue_size must be an integer >= 1, got {max_queue_size!r}")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise ValueError(f"max_results must be an integer >= 1, got {max_results!r}")

        self._max_queue_size = int(max_queue_size)
        self._max_results = int(max_results)
        self._skip_incomplete_quality = bool(skip_incomplete_quality)
        self._synchronous = bool(synchronous)
        self._analysis_config = analysis_config
        self._result_sink = result_sink
        self._result_sinks: list[Callable[[CaptureSidecar], None]] = (
            [result_sink] if result_sink is not None else []
        )

        self._state = WorkerState.OPEN
        self._current_generation: int = 0
        self._aggregator = FeatureAggregator(build_default_feature_registry())
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=self._max_queue_size)
        self._result_queue: queue.Queue[CaptureSidecar] = queue.Queue(maxsize=self._max_results)
        self._dropped_overload_count: int = 0
        self._dropped_result_overload_count: int = 0
        self._cancelled_count: int = 0
        self._completed_count: int = 0
        self._failed_count: int = 0
        self._discarded_old_generation_count: int = 0
        self._sink_error_count: int = 0
        self._processed_results: collections.deque[CaptureSidecar] = collections.deque(maxlen=self._max_results)
        self._lock = threading.Lock()
        self._sentinel_enqueued: bool = False

        self._thread: threading.Thread | None = None

        if not self._synchronous:
            self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="FeatureAnalysisWorker")
            self._thread.start()

    @property
    def state(self) -> WorkerState:
        """Current lifecycle state of the worker."""
        with self._lock:
            return self._state

    @property
    def current_generation(self) -> int:
        with self._lock:
            return self._current_generation

    @property
    def _generation(self) -> int:
        """Compatibility property for generation counter."""
        return self.current_generation

    @property
    def dropped_overload_count(self) -> int:
        """Count of captures dropped because the worker input queue was full."""
        with self._lock:
            return self._dropped_overload_count

    @property
    def dropped_result_overload_count(self) -> int:
        """Count of results dropped because the result queue was full and undrained."""
        with self._lock:
            return self._dropped_result_overload_count

    @property
    def cancelled_count(self) -> int:
        """Count of submitted items cancelled during worker cancel/reset."""
        with self._lock:
            return self._cancelled_count

    @property
    def completed_count(self) -> int:
        with self._lock:
            return self._completed_count

    @property
    def failed_count(self) -> int:
        with self._lock:
            return self._failed_count

    @property
    def discarded_old_generation_count(self) -> int:
        with self._lock:
            return self._discarded_old_generation_count

    @property
    def sink_error_count(self) -> int:
        with self._lock:
            return self._sink_error_count

    @property
    def processed_results(self) -> list[CaptureSidecar]:
        """List snapshot of bounded processed sidecars."""
        with self._lock:
            return list(self._processed_results)

    def _record_result_locked(self, result: CaptureSidecar, generation: int) -> bool:
        """Store result in bounded deque and result queue under self._lock."""
        if generation < self._current_generation:
            self._discarded_old_generation_count += 1
            return False

        self._processed_results.append(result)
        try:
            self._result_queue.put_nowait(result)
        except queue.Full:
            self._dropped_result_overload_count += 1

        if result.feature_status == "failed":
            self._failed_count += 1
        elif result.feature_status in ("ok", "degraded"):
            self._completed_count += 1

        return True

    def add_result_sink(self, sink: Callable[[CaptureSidecar], None]) -> None:
        """Register a callback to receive completed event sidecars."""
        if not callable(sink):
            raise TypeError(f"sink must be callable, got {type(sink).__name__}")
        with self._lock:
            self._result_sinks.append(sink)

    def _deliver_sink(self, result: CaptureSidecar) -> None:
        """Invoke external result sink callback strictly outside lifecycle locks."""
        with self._lock:
            sinks_snapshot = list(self._result_sinks)

        for sink in sinks_snapshot:
            try:
                sink(result)
            except Exception:
                with self._lock:
                    self._sink_error_count += 1

    def get_result(self, block: bool = True, timeout: float | None = None) -> CaptureSidecar | None:
        """Retrieve the next processed result from the result delivery queue."""
        try:
            return self._result_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def _process_one(self, sidecar: CaptureSidecar) -> CaptureSidecar:
        # Quality Gate
        if self._skip_incomplete_quality and not sidecar.event.complete:
            reason = f"Capture incomplete ({sidecar.termination_reason}, dropped={sidecar.known_lost_samples})"
            return sidecar.with_features(features=None, feature_status="skipped_quality", feature_reason=reason)

        if not sidecar.noise_valid:
            reason = f"Uncalibrated or invalid noise reference ({sidecar.noise_provenance})"
            if self._skip_incomplete_quality:
                return sidecar.with_features(features=None, feature_status="skipped_quality", feature_reason=reason)

        try:
            context = SignalContext(event=sidecar.event, config=self._analysis_config)
            analysis_result: AnalysisResult = self._aggregator.extract_analysis_result(context)
            if not sidecar.event.complete:
                status = "degraded"
                reason = f"Degraded: incomplete capture extent ({sidecar.termination_reason}, dropped={sidecar.known_lost_samples})"
            elif not sidecar.noise_valid:
                status = "degraded"
                reason = f"Degraded noise reference ({sidecar.noise_provenance})"
            else:
                status = "ok"
                reason = None

            return sidecar.with_features(
                features=analysis_result.feature_vector,
                feature_status=status,
                feature_reason=reason,
                analysis_result=analysis_result,
            )
        except Exception as exc:
            return sidecar.with_features(
                features=None,
                feature_status="failed",
                feature_reason=f"Feature extraction failed: {type(exc).__name__}: {exc}",
            )

    def submit(self, sidecar: CaptureSidecar) -> CaptureSidecar:
        """Submit a captured event sidecar for feature analysis.

        Raises:
            RuntimeError: If worker is in CLOSING or CLOSED state.
        """
        to_deliver: CaptureSidecar | None = None
        result_to_return: CaptureSidecar | None = None

        with self._lock:
            if self._state != WorkerState.OPEN:
                raise RuntimeError(f"Cannot submit to FeatureAnalysisWorker in state {self._state.value}")

            gen = self._current_generation

            if self._synchronous:
                result = self._process_one(sidecar)
                if self._record_result_locked(result, gen):
                    to_deliver = result
                result_to_return = result
            else:
                try:
                    self._queue.put_nowait((gen, sidecar))
                    result_to_return = sidecar
                except queue.Full:
                    self._dropped_overload_count += 1
                    dropped_result = sidecar.with_features(
                        features=None,
                        feature_status="dropped_overload",
                        feature_reason="Feature worker queue full",
                    )
                    if self._record_result_locked(dropped_result, gen):
                        to_deliver = dropped_result
                    result_to_return = dropped_result

        if to_deliver is not None:
            self._deliver_sink(to_deliver)

        assert result_to_return is not None
        return result_to_return

    def _worker_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                with self._lock:
                    if self._state != WorkerState.OPEN and self._queue.empty():
                        break
                continue

            to_deliver: CaptureSidecar | None = None
            try:
                if item is _SENTINEL:
                    break

                gen, sidecar = item
                with self._lock:
                    if gen < self._current_generation:
                        self._discarded_old_generation_count += 1
                        continue

                result = self._process_one(sidecar)
                with self._lock:
                    if self._record_result_locked(result, gen):
                        to_deliver = result
            finally:
                self._queue.task_done()

            if to_deliver is not None:
                self._deliver_sink(to_deliver)

    def cancel(self) -> None:
        """Cancel and drain all pending unstarted jobs from the queue."""
        with self._lock:
            cancelled = 0
            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    if item is not _SENTINEL:
                        cancelled += 1
                    self._queue.task_done()
                except queue.Empty:
                    break
            self._cancelled_count += cancelled

    def advance_generation(self) -> int:
        """Advance generation to invalidate old in-flight captures across stream discontinuities."""
        with self._lock:
            self._current_generation += 1
            return self._current_generation

    def shutdown(self, drain: bool = True, timeout: float = 5.0) -> None:
        """Orderly shutdown of worker thread using a single monotonic deadline.

        Args:
            drain: If True, waits for queued jobs to be processed before stopping.
                If False, cancels pending jobs immediately.
            timeout: Maximum seconds to wait for worker thread termination.

        Raises:
            TimeoutError: If thread does not terminate within the timeout.
        """
        import time
        deadline = time.monotonic() + max(0.0, float(timeout))

        with self._lock:
            if self._state == WorkerState.CLOSED:
                return
            self._state = WorkerState.CLOSING
            enqueued = self._sentinel_enqueued

        if not self._synchronous and self._thread is not None and self._thread.is_alive():
            if not drain:
                self.cancel()

            # Insert sentinel using remaining time on monotonic deadline
            if not enqueued:
                rem_sentinel = max(0.0, deadline - time.monotonic())
                try:
                    self._queue.put(_SENTINEL, timeout=rem_sentinel)
                    with self._lock:
                        self._sentinel_enqueued = True
                except queue.Full:
                    pass

            rem_join = max(0.0, deadline - time.monotonic())
            self._thread.join(timeout=rem_join)
            if self._thread.is_alive():
                raise TimeoutError(f"FeatureAnalysisWorker shutdown timed out after {timeout} seconds")

        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except (queue.Empty, ValueError):
                    break
            self._state = WorkerState.CLOSED

    def reset(self) -> None:
        """Reset internal queue, results, and counters, advancing generation to invalidate in-flight jobs."""
        with self._lock:
            self._current_generation += 1
            self._sentinel_enqueued = False
            cancelled = 0
            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    if item is not _SENTINEL:
                        cancelled += 1
                    self._queue.task_done()
                except queue.Empty:
                    break
            while not self._result_queue.empty():
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    break
            self._cancelled_count += cancelled
            self._dropped_overload_count = 0
            self._dropped_result_overload_count = 0
            self._completed_count = 0
            self._failed_count = 0
            self._discarded_old_generation_count = 0
            self._sink_error_count = 0
            self._processed_results.clear()
