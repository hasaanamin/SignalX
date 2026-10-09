"""Feature analysis worker with bounded queue, explicit lifecycle, and quality gating."""

from __future__ import annotations

import collections
import dataclasses
import enum
import math
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
        analysis_backend: str = "thread",
        process_timeout: float = 30.0,
        process_workers: int = 4,
        result_order: str = 'submission',
    ) -> None:
        if isinstance(max_queue_size, bool) or not isinstance(max_queue_size, int) or max_queue_size < 1:
            raise ValueError(f"max_queue_size must be an integer >= 1, got {max_queue_size!r}")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise ValueError(f"max_results must be an integer >= 1, got {max_results!r}")

        if analysis_backend not in ('thread', 'process', 'process_pool'):
            raise ValueError('analysis_backend must be thread, process or process_pool')
        if analysis_backend != 'thread' and (synchronous or analysis_function is not None):
            raise ValueError('Process backend requires asynchronous default analysis; custom callbacks use thread backend')
        if type(process_workers) is not int or not 1<=process_workers<=4:
            raise ValueError('process_workers must be an integer from 1 to 4')
        if result_order not in ('submission','completion'):
            raise ValueError('result_order must be submission or completion')
        if result_order!='submission' and analysis_backend!='process_pool':
            raise ValueError('Completion-order delivery requires process_pool backend')
        self._result_order=result_order
        self._max_queue_size = int(max_queue_size)
        self._max_results = int(max_results)
        if result_delivery not in ('polling', 'callbacks'):
            raise ValueError("result_delivery must be polling or callbacks")
        self._result_delivery = result_delivery
        self._polling_reserved = 0
        self._skip_incomplete_quality = bool(skip_incomplete_quality)
        self._synchronous = bool(synchronous)
        self._analysis_config = analysis_config if analysis_config is not None else AnalysisConfig()
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
        self.parent_worker_cpu_seconds = 0.0
        self._processed_results: collections.deque[CaptureSidecar] = collections.deque(maxlen=self._max_results)
        self._lock = threading.Lock()
        self._completion = threading.Condition(self._lock)
        # At most max_queue_size concurrent synchronous callers; each remains
        # admitted through result storage AND out-of-lock callback delivery.
        self._sync_threads: set[int] = set()
        self._async_outstanding = 0  # Accepted jobs, never control sentinels.
        self._sentinel_enqueued: bool = False

        self._thread: threading.Thread | None = None
        self._process_backend = None
        self._backend_cleanup_error = None
        self._parallel_processes = analysis_backend == 'process_pool'
        self._wake_receiver = self._wake_sender = None
        self._readiness = None
        self._wake_pending = False
        self.wakeup_errors = 0
        if analysis_backend == 'process':
            from signal_processing.runtime.process_analysis import ProcessAnalysisBackend
            self._process_backend = ProcessAnalysisBackend(timeout=process_timeout)
            self._analysis_function = self._process_backend
        elif self._parallel_processes:
            from signal_processing.runtime.process_analysis import ParallelProcessAnalysisBackend
            try:
                import multiprocessing, sys
                if sys.platform == 'win32':
                    from signal_processing.runtime._readiness import WindowsReadiness
                    self._readiness = WindowsReadiness()
                else:
                    self._wake_receiver,self._wake_sender=multiprocessing.Pipe(duplex=False)
                self._process_backend = ParallelProcessAnalysisBackend(timeout=process_timeout,
                    workers=process_workers, readiness=self._readiness)
                self._analysis_function = self._process_backend
            except BaseException:
                if self._process_backend is not None:self._process_backend.close()
                if self._readiness is not None:self._readiness.close()
                if self._wake_receiver is not None:self._wake_receiver.close();self._wake_sender.close()
                raise

        if not self._synchronous:
            self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="FeatureAnalysisWorker")
            try:self._thread.start()
            except BaseException:
                if self._process_backend is not None:self._process_backend.close()
                if self._wake_receiver is not None:self._wake_receiver.close();self._wake_sender.close()
                if self._readiness is not None:self._readiness.close()
                self._state=WorkerState.CLOSED
                raise

    def _wake_locked(self):
        """Coalesce input/lifecycle readiness to ONE bounded internal byte.

        Called under admission lock. The sole receiver consumes that byte under
        the same lock before clearing the flag; an empty-to-one write cannot
        fill this pipe. No external callback or IQ is transported here.
        """
        if self._readiness is not None:
            try:self._readiness.set()
            except OSError:self.wakeup_errors+=1
        elif self._wake_sender is not None and not self._wake_pending:
            try:
                self._wake_sender.send_bytes(b'x')
                self._wake_pending=True
            except OSError:
                self.wakeup_errors+=1  # Affected runs cannot claim clean recovery.

    def _drain_wake(self):
        with self._lock:
            if self._readiness is not None:
                self._readiness.clear()
            if self._wake_pending:
                self._wake_receiver.recv_bytes(1)
                self._wake_pending=False

    @property
    def backend_pid(self) -> int | None:
        """Child PID for diagnostics; None for the compatibility thread backend."""
        return self._process_backend.pid if self._process_backend is not None else None

    @property
    def backend_pids(self) -> tuple[int, ...]:
        """Persistent child identities for owned-process diagnostics/placement."""
        if self._process_backend is None:
            return ()
        return getattr(self._process_backend,'pids',(self._process_backend.pid,))

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

    @property
    def unfinished_work(self) -> int:
        """Accepted queued/running work, including required callback delivery.

        Backend teardown is represented by CLOSING until cleanup completes, not
        an extra fictitious analysis job. CLOSED and zero work prove both drained.
        """
        with self._lock:
            return len(self._sync_threads) if self._synchronous else self._async_outstanding

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
                    self._completion.notify_all()
                    return result
                self._discarded_old_generation_count += 1

    def _process_one(self, sidecar: CaptureSidecar, *, prepare_only=False):
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
            if prepare_only:
                return context
            analysis_result: AnalysisResult = (self._analysis_function(context) if self._analysis_function is not None
                                               else self._aggregator.extract_analysis_result(context))
            if not isinstance(analysis_result, AnalysisResult):
                raise TypeError("analysis_function must return AnalysisResult")
            if self._analysis_function is not None:
                self._validate_backend_result(analysis_result, context, cap_quality,
                                              policy_verified=self._process_backend is not None)
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

    @staticmethod
    def _validate_backend_result(result, context, quality, *, policy_verified=False):
        """Validate observable attribution, not arbitrary backend arithmetic.

        Custom feature sets/schema and extractor overrides are supported. Context
        transforms, sample domain, policy content identity and capture provenance
        must still match. No coordinates beyond those in these records are inferred.
        """
        from signal_processing.core.contracts import ExtractionPolicy
        vector, policy = result.feature_vector, result.policy
        if vector.event_id != context.event_id:
            raise ValueError('Backend result event identity differs from admitted event')
        if result.capture_quality != quality:
            raise ValueError('Backend capture quality/provenance differs from admitted context')
        if policy.schema_version != vector.feature_schema_version or set(policy.feature_names) != set(vector.features):
            raise ValueError('Backend feature schema/names contradict its policy')
        if not policy_verified:
            ExtractionPolicy.from_dict(policy.to_dict())
        cfg, n = context.config, context.signal_num_samples
        welch = cfg.psd_mode == 'welch'
        nperseg = (cfg.psd_nperseg if cfg.psd_nperseg is not None else min(n, 1024)) if welch else n
        expected = dict(sample_rate_hz=context.sample_rate_hz, observation_samples=n,
            psd_mode=cfg.psd_mode, nfft=cfg.nfft if cfg.nfft is not None else nperseg,
            psd_nperseg=nperseg,
            psd_noverlap=(cfg.psd_noverlap if cfg.psd_noverlap is not None else nperseg//2) if welch else 0,
            spectral_window=cfg.psd_welch_window if welch else cfg.spectral_window,
            centered_spectrum=cfg.centered_spectrum, power_floor_dbfs=cfg.power_floor_dbfs,
            psd_floor_dbfs_per_hz=cfg.psd_floor_dbfs_per_hz,
            phase_amplitude_threshold=cfg.phase_amplitude_threshold,
            autocorrelation_normalized=cfg.autocorrelation_normalized,
            autocorrelation_max_lag_samples=cfg.autocorrelation_max_lag_samples if cfg.autocorrelation_max_lag_samples is not None else max(0, n-1))
        for name, value in expected.items():
            if getattr(policy, name) != value:
                raise ValueError(f'Backend policy {name} differs from admitted context')

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

            if self._synchronous:
                ident = threading.get_ident()
                if ident in self._sync_threads or len(self._sync_threads) >= self._max_queue_size:
                    raise RuntimeError('Synchronous admission capacity reached or recursive submit; work not accepted')
                self._sync_threads.add(ident)

            if not self._synchronous:
                try:
                    # Parallel service does not enlarge total accepted storage:
                    # queue capacity + one legacy running-job lease, including
                    # ordered results waiting for a required sink.
                    if self._parallel_processes and self._async_outstanding >= self._max_queue_size+1:
                        raise queue.Full
                    if self._result_delivery == 'polling' and self._polling_reserved >= self._max_results:
                        raise queue.Full
                    self._queue.put_nowait((gen, sidecar))
                    self._async_outstanding += 1
                    self._wake_locked()
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
                    result_to_return = dropped_result

        if self._synchronous:
            try:
                started = time.perf_counter()
                result = self._safe_process_one(sidecar)
                result = dataclasses.replace(result, processing_generation=gen)
                with self._lock:
                    self.analysis_seconds += time.perf_counter() - started
                    deliver = self._record_result_locked(result, gen)
                if deliver:
                    self._deliver_sink(result)
                return result
            finally:
                with self._completion:
                    self._sync_threads.remove(ident)
                    if self._state == WorkerState.CLOSING and not self._sync_threads:
                        self._state = WorkerState.CLOSED
                    self._completion.notify_all()

        if to_deliver is not None:
            self._deliver_sink(to_deliver)

        assert result_to_return is not None
        return result_to_return

    def wait_for_capacity(self, timeout: float) -> bool:
        """Optional bounded downstream backpressure, never called by submit().

        Intended for a separate completion pump, NOT the IQ acquisition caller.
        This is a readiness observation, not a multi-producer reservation. An
        absent consumer cannot create unlimited lossless storage: upstream finite
        buffers eventually fill, with explicit source loss/overload accounting.
        """
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError('Capacity timeout must be finite and nonnegative')
        deadline = time.monotonic()+float(timeout)
        with self._completion:
            while self._state == WorkerState.OPEN:
                if (not self._queue.full()
                        and (not self._parallel_processes or self._async_outstanding < self._max_queue_size+1)
                        and (self._result_delivery != 'polling' or self._polling_reserved < self._max_results)):
                    return True
                if threading.current_thread() is self._thread or threading.get_ident() in self._sync_threads:
                    raise RuntimeError('A callback cannot wait for its own delivery capacity')
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    return False
                self._completion.wait(remaining)
            return False

    def _safe_process_one(self, sidecar: CaptureSidecar) -> CaptureSidecar:
        """A calculation failure is a terminal result, never a dead worker thread."""
        try:
            return self._process_one(sidecar)
        except Exception as exc:
            return sidecar.with_features(features=None, feature_status="failed",
                feature_reason=f"Analysis failed: {type(exc).__name__}: {exc}")

    def _worker_loop(self) -> None:
        cpu_started = time.thread_time()
        try:
            if self._parallel_processes:
                self._run_parallel_jobs()
            else:
                self._run_jobs()
        finally:
            self.parent_worker_cpu_seconds = time.thread_time()-cpu_started
            # CLOSED includes child cleanup; timeout callers observe CLOSING
            # until analysis, callbacks and process termination actually finish.
            try:
                if self._process_backend is not None:
                    self._process_backend.close()
            except Exception as exc:
                self._backend_cleanup_error = exc
            with self._completion:
                self._state = WorkerState.CLOSING if self._backend_cleanup_error else WorkerState.CLOSED
                if self._wake_receiver is not None:
                    self._wake_receiver.close();self._wake_sender.close()
                    self._wake_receiver=self._wake_sender=None
                if self._readiness is not None:
                    self._readiness.close()
                    self._readiness=None
                self._completion.notify_all()

    def _run_parallel_jobs(self) -> None:
        """Four persistent servers; bounded ordered completion, serial sinks.

        No executor/future queue exists. Every transaction belongs to the same
        accepted-work ledger as queued work, through completion of its sinks.
        """
        pending = collections.deque()
        available = collections.deque(self._process_backend._backends)
        stopping = False
        while not stopping or pending:
                self._drain_wake()
                while not stopping and available and len(pending) < len(self._process_backend._backends):
                    try:
                        item = self._queue.get_nowait() if pending else self._queue.get(timeout=0.1)
                    except queue.Empty:
                        with self._lock:
                            stopping = self._state != WorkerState.OPEN and self._queue.empty()
                        break
                    if item is _SENTINEL:
                        self._queue.task_done()
                        with self._lock:
                            self._sentinel_enqueued = False
                        stopping = True
                        break
                    gen, sidecar = item
                    backend = available.popleft()
                    context = None
                    result = None
                    try:
                        context = self._process_one(sidecar, prepare_only=True)
                        result = context if isinstance(context, CaptureSidecar) else None
                        if result is None:
                            backend.begin(context)
                    except Exception as exc:
                        result = sidecar.with_features(features=None, feature_status='failed',
                            feature_reason=f'Analysis failed: {type(exc).__name__}: {exc}')
                    pending.append((gen, time.perf_counter(), backend, sidecar, context, result))
                if not pending:
                    continue
                selected=0
                if self._result_order=='completion':
                    # Bounded scan (<=4). Independent immutable event jobs may
                    # finish out of order only under this explicit opt-in. Fast
                    # servers can then accept work without waiting for a slow
                    # predecessor. Sinks themselves remain strictly serial.
                    for i,item in enumerate(pending):
                        try:ready=item[5] is not None or item[2].ready()
                        except Exception:ready=True  # Controlled failure below.
                        if ready:
                            selected=i
                            break
                gen, started, backend, sidecar, context, result = pending[selected]
                if result is None:
                    try:
                        if not backend.ready():
                            # Wake on ANY eligible completed server or new input,
                            # not a polling timer tied to the oldest transaction.
                            if self._readiness is not None:
                                self._readiness.wait(.05)
                            else:
                                from multiprocessing.connection import wait
                                connections=([item[2]._connection for item in pending if item[5] is None]
                                    if self._result_order=='completion' else [backend._connection])
                                wait([*connections,self._wake_receiver],timeout=.05)
                            continue
                    except Exception as exc:
                        result = sidecar.with_features(features=None, feature_status='failed',
                            feature_reason=f'Analysis transport failed: {type(exc).__name__}: {exc}')
                del pending[selected]
                try:
                    if result is None:
                        try:
                            analysis = backend.finish()
                            self._validate_backend_result(analysis, context, context.capture_quality, policy_verified=True)
                            degraded = not sidecar.event.complete or not sidecar.noise_valid
                            result = sidecar.with_features(features=analysis.feature_vector,
                                analysis_result=analysis, feature_status='degraded' if degraded else 'ok',
                                feature_reason=(f'Degraded capture: {sidecar.termination_reason}; noise_valid={sidecar.noise_valid}'
                                                if degraded else None))
                        except Exception as exc:
                            result = sidecar.with_features(features=None, feature_status='failed',
                                feature_reason=f'Analysis failed: {type(exc).__name__}: {exc}')
                    result = dataclasses.replace(result, processing_generation=gen)
                    with self._lock:
                        self.analysis_seconds += time.perf_counter()-started
                        deliver = self._record_result_locked(result, gen)
                    if deliver:
                        self._deliver_sink(result)
                finally:
                    available.append(backend)
                    self._queue.task_done()
                    with self._completion:
                        self._async_outstanding -= 1
                        self._completion.notify_all()

    def _run_jobs(self) -> None:
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
                if to_deliver is not None:
                    self._deliver_sink(to_deliver)
            finally:
                # A dequeued job is outstanding through required delivery.
                # Callbacks remain outside lifecycle locks and can reset/close.
                self._queue.task_done()
                if item is not _SENTINEL:
                    with self._completion:
                        self._async_outstanding -= 1
                        self._completion.notify_all()

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
                        self._async_outstanding -= 1
                        if self._result_delivery == 'polling' and item[0] == self._current_generation:
                            self._polling_reserved -= 1
                    else:
                        self._sentinel_enqueued = False
                    self._queue.task_done()
                except queue.Empty:
                    break
            self._cancelled_count += cancelled
            self._wake_locked()
            self._completion.notify_all()

    def advance_generation(self) -> int:
        """Explicit cancellation barrier; ordinary raw acquisition gaps must not call this."""
        with self._lock:
            self._cancellation_operations += 1
            self._current_generation += 1
            self._wake_locked()
            self._polling_reserved = 0
            while not self._result_queue.empty():
                self._result_queue.get_nowait()
                self._discarded_old_generation_count += 1
            self._completion.notify_all()
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
            self._wake_locked()
            enqueued = self._sentinel_enqueued
            reentrant = threading.get_ident() in self._sync_threads or threading.current_thread() is self._thread

        # A callback cannot synchronously drain itself. Closing is deferred;
        # completion of the outer admission/worker loop performs the transition.
        if reentrant:
            if not drain:
                self.cancel()
            return

        if self._synchronous:
            with self._completion:
                while self._sync_threads:
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('Synchronous analysis/delivery drain timed out; worker remains CLOSING')
                    self._completion.wait(remaining)
                self._state = WorkerState.CLOSED
            return

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

        if self._process_backend is not None:
            self._process_backend.close()
            self._backend_cleanup_error = None
        with self._lock:
            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    if item is not _SENTINEL:
                        self._async_outstanding -= 1
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
                        self._async_outstanding -= 1
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
            self._wake_locked()
            self._completion.notify_all()
