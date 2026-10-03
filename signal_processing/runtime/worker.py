"""Feature analysis worker with bounded queue, explicit lifecycle, and quality gating."""

from __future__ import annotations

import collections
import dataclasses
import enum
import queue
import threading
import time
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
        analysis_function: Callable[[SignalContext], AnalysisResult] | None = None,
        result_delivery: str = "polling",
    ) -> None:
        if isinstance(max_queue_size, bool) or not isinstance(max_queue_size, int) or max_queue_size < 1:
            raise ValueError(f"max_queue_size must be an integer >= 1, got {max_queue_size!r}")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise ValueError(f"max_results must be an integer >= 1, got {max_results!r}")

        self._max_queue_size = int(max_queue_size)
        self._max_results = int(max_results)
        if result_delivery not in ('polling', 'callbacks'):
            raise ValueError("result_delivery must be polling or callbacks")
        self._result_delivery = result_delivery
        self._polling_reserved = 0
        self._skip_incomplete_quality = bool(skip_incomplete_quality)
        self._synchronous = bool(synchronous)
        self._analysis_config = analysis_config
        self._result_sink = result_sink
        self._analysis_function = analysis_function
        self._cancellation_operations = 0
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
        self.analysis_seconds = 0.0
        self.sink_seconds = 0.0
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
    def cancellation_operations(self) -> int:
        """Lifetime explicit cancellation barriers, not reset with per-generation counters."""
        with self._lock:
            return self._cancellation_operations

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
        if self._result_delivery == 'polling':
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

    def _deliver_sink(self, result: CaptureSidecar, *args: Any, **kwargs: Any) -> None:
        """Invoke external result sink callback strictly outside lifecycle locks."""
        generation = args[0] if args else result.processing_generation
        with self._lock:
            if generation is not None and generation < self._current_generation:
                self._discarded_old_generation_count += 1
                return
            sinks_snapshot = list(self._result_sinks)

        for sink in sinks_snapshot:
            # Admission is atomic with reset. An admitted callback may finish after
            # reset; callbacks not yet admitted are suppressed, including reentrant reset.
            with self._lock:
                if generation is not None and generation != self._current_generation:
                    self._discarded_old_generation_count += 1
                    return
            try:
                started = time.perf_counter()
                sink(result)
            except Exception:
                with self._lock:
                    self._sink_error_count += 1
            finally:
                with self._lock:
                    self.sink_seconds += time.perf_counter() - started

    def get_result(self, block: bool = True, timeout: float | None = None) -> CaptureSidecar | None:
        """Retrieve the next processed result from the result delivery queue."""
        if self._result_delivery != 'polling':
            raise RuntimeError("This worker declares callbacks, not polling, as its delivery channel")
        import time
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else max(0, deadline - time.monotonic())
            try:
                result = self._result_queue.get(block=block, timeout=remaining)
            except queue.Empty:
                return None
            with self._lock:
                generation = getattr(result, "processing_generation", None)
                if generation is None or generation == self._current_generation:
                    if not self._synchronous:
                        self._polling_reserved -= 1
                    return result
                self._discarded_old_generation_count += 1

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
            from signal_processing.core.contracts import CaptureQuality

            if not sidecar.event.complete:
                if sidecar.termination_reason == "eof":
                    warns = ("Incomplete event capture: terminated at EOF",)
                elif sidecar.unknown_loss:
                    warns = ("Incomplete event capture: terminated by unknown loss / overflow",)
                elif sidecar.known_lost_samples > 0:
                    warns = (
                        f"Incomplete event capture: terminated by discontinuity with {sidecar.known_lost_samples} known lost samples",
                    )
                else:
                    warns = (f"Incomplete event capture: {sidecar.termination_reason}",)
            else:
                warns = ()
            cap_quality = CaptureQuality(
                complete=sidecar.event.complete,
                dropped_samples=sidecar.event.dropped_samples,
                is_continuous=sidecar.event.dropped_samples == 0,
                warnings=warns,
                known_lost_samples=sidecar.known_lost_samples,
                unknown_loss=sidecar.unknown_loss,
                termination_reason=sidecar.termination_reason,
                capture_epoch=sidecar.stream_epoch,
                noise_valid=sidecar.noise_valid,
                noise_provenance=sidecar.noise_provenance,
                boundary_uncertainty_samples=sidecar.boundary_uncertainty_samples,
            )
            context = SignalContext(
                event=sidecar.event, config=self._analysis_config, capture_quality=cap_quality
            )
            analysis_result: AnalysisResult = (self._analysis_function(context) if self._analysis_function is not None
                                               else self._aggregator.extract_analysis_result(context))
            if not isinstance(analysis_result, AnalysisResult):
                raise TypeError("analysis_function must return AnalysisResult")
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
            if self._result_delivery == 'callbacks' and not self._result_sinks:
                raise RuntimeError("Register the required callback before submitting work")

            gen = self._current_generation
            sidecar = dataclasses.replace(sidecar, features=None, analysis_result=None,
                                          feature_status='pending', feature_reason=None, processing_generation=gen)

            if not self._synchronous:
                try:
                    if self._result_delivery == 'polling' and self._polling_reserved >= self._max_results:
                        raise queue.Full
                    self._queue.put_nowait((gen, sidecar))
                    if self._result_delivery == 'polling':
                        self._polling_reserved += 1
                    result_to_return = dataclasses.replace(sidecar, processing_generation=gen)
                except queue.Full:
                    self._dropped_overload_count += 1
                    dropped_result = sidecar.with_features(
                        features=None,
                        feature_status="dropped_overload",
                        feature_reason="Feature worker input or reserved result capacity full; job not accepted",
                    )
                    dropped_result = dataclasses.replace(dropped_result, processing_generation=gen)
                    # Rejection is returned directly to submitter; do not consume
                    # a result slot reserved for an already accepted async job.
                    self._processed_results.append(dropped_result)
                    to_deliver = dropped_result
                    result_to_return = dropped_result

        if self._synchronous:
            started = time.perf_counter()
            result = self._safe_process_one(sidecar)
            result = dataclasses.replace(result, processing_generation=gen)
            with self._lock:
                self.analysis_seconds += time.perf_counter() - started
                if self._record_result_locked(result, gen):
                    to_deliver = result
            result_to_return = result

        if to_deliver is not None:
            self._deliver_sink(to_deliver)

        assert result_to_return is not None
        return result_to_return

    def _safe_process_one(self, sidecar: CaptureSidecar) -> CaptureSidecar:
        """A calculation failure is a terminal result, never a dead worker thread."""
        try:
            return self._process_one(sidecar)
        except Exception as exc:
            return sidecar.with_features(features=None, feature_status="failed",
                feature_reason=f"Analysis failed: {type(exc).__name__}: {exc}")

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
            gen: int | None = None
            try:
                if item is _SENTINEL:
                    with self._lock:
                        self._sentinel_enqueued = False
                    break

                gen, sidecar = item
                with self._lock:
                    if gen < self._current_generation:
                        self._discarded_old_generation_count += 1
                        continue

                started = time.perf_counter()
                result = self._safe_process_one(sidecar)
                self.analysis_seconds += time.perf_counter() - started
                result = dataclasses.replace(result, processing_generation=gen)
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
            self._cancellation_operations += 1
            cancelled = 0
            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    if item is not _SENTINEL:
                        cancelled += 1
                        if self._result_delivery == 'polling' and item[0] == self._current_generation:
                            self._polling_reserved -= 1
                    else:
                        self._sentinel_enqueued = False
                    self._queue.task_done()
                except queue.Empty:
                    break
            self._cancelled_count += cancelled

    def advance_generation(self) -> int:
        """Explicit cancellation barrier; ordinary raw acquisition gaps must not call this."""
        with self._lock:
            self._cancellation_operations += 1
            self._current_generation += 1
            self._polling_reserved = 0
            while not self._result_queue.empty():
                self._result_queue.get_nowait()
                self._discarded_old_generation_count += 1
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
            self._cancellation_operations += 1
            self._current_generation += 1
            self._polling_reserved = 0
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
