"""One persistent spawn worker for default features; no unbounded executor queue.

Only the owning FeatureAnalysisWorker thread calls this backend. One request can
be in flight; admission, polling reservations, cancellation and sinks stay in the
parent worker. Payloads are explicit records, never pickled SignalContext caches
or mapping proxies. IQ is copied at this boundary, not shared/recycled unsafely.
"""
from __future__ import annotations

import dataclasses
import math
import multiprocessing
import pickle
import queue
import time

import numpy as np

from signal_processing.context import AnalysisConfig, SignalContext
from signal_processing.core.contracts import (
    AnalysisResult, CaptureQuality, ExtractionPolicy, FeatureValue, FeatureVector,
    SignalEvent, StreamInfo,
)

_PROTOCOL = 'rf-default-features-v1'
MAX_PROCESS_EVENT_SAMPLES = 4096
MAX_REQUEST_BYTES = 48*1024
MAX_RESPONSE_BYTES = 48*1024


def _request(context):
    event = context.event
    fields = {f.name: getattr(event, f.name) for f in dataclasses.fields(event)
              if f.name not in ('iq', 'stream')}
    return (fields, dataclasses.asdict(event.stream), event.iq.tobytes(),
            dataclasses.asdict(context.config), dataclasses.asdict(context.capture_quality))


def _context(record):
    fields, stream, iq, config, quality = record
    event = SignalEvent(**fields, stream=StreamInfo(**stream),
                        iq=np.frombuffer(iq, dtype=np.complex64))
    return SignalContext(event, AnalysisConfig(**config), CaptureQuality(**quality))


def _result_record(result):
    vector = result.feature_vector
    return (vector.event_id, vector.feature_schema_version,
            {k: (v.value, v.unit, v.valid, v.reason, v.interpretation)
             for k, v in vector.features.items()},
            result.policy.to_dict(), dataclasses.asdict(result.capture_quality))


def _result(record):
    event_id, schema, values, policy, quality = record
    return AnalysisResult(FeatureVector(event_id, schema,
        {k: FeatureValue(*v) for k, v in values.items()}),
        ExtractionPolicy.from_dict(policy), CaptureQuality(**quality))


def _serve(connection, readiness=None):
    """Spawn-safe entry point. No caller functions or open application resources."""
    try:
        from signal_processing.features import FeatureAggregator, build_default_feature_registry
        aggregator = FeatureAggregator(build_default_feature_registry())
        # Readiness includes a real disposable analysis, so lazy FFT/statistics
        # initialization is not charged to an already running IQ source. The
        # context/result are discarded; no event waveform/results are cached.
        rng = np.random.default_rng(9317)
        iq = ((rng.normal(size=256)+1j*rng.normal(size=256))*.01+.1).astype(np.complex64)
        event = SignalEvent('backend-readiness',StreamInfo('readiness','internal',1e6),
            iq,0,256,0,256,0)
        aggregator.extract_analysis_result(SignalContext(event))
        del event,iq,rng
        connection.send(('ready', None))
        while True:
            request = connection.recv()
            if request is None:
                break
            version, token, request = request
            if version != _PROTOCOL or type(token) is not int or token < 1:
                raise ValueError('Invalid feature transport request header')
            try:
                started = time.perf_counter()
                cpu_started = time.process_time()
                context = _context(request)
                result = aggregator.extract_analysis_result(context)
                response = ('ok', (_result_record(result), time.perf_counter()-started,
                                   time.process_time()-cpu_started))
            except Exception as exc:
                response = ('error', f'{type(exc).__name__}: {exc}')
            payload = pickle.dumps((_PROTOCOL, token, response), protocol=5)
            if len(payload) > MAX_RESPONSE_BYTES:
                payload = pickle.dumps((_PROTOCOL, token,
                    ('error', 'Result exceeds bounded 48 KiB transport envelope')), protocol=5)
            connection.send_bytes(payload)
            if readiness is not None:
                readiness.set()
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()
        if readiness is not None:
            try: readiness.set()
            finally: readiness.close()


class ProcessAnalysisBackend:
    """Bounded request/reply backend, made ready before public admission starts."""

    def __init__(self, *, timeout=30.0, startup_timeout=15.0, readiness=None):
        for value in (timeout, startup_timeout):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Process timeouts must be positive finite seconds')
        self.timeout = timeout
        self._closed = False
        self._broken = False
        self._pid = None
        self.child_analysis_seconds = 0.0
        self.child_cpu_seconds = 0.0
        self.roundtrip_seconds = 0.0
        self._token = 0
        ctx = multiprocessing.get_context('spawn')
        self._connection, child = ctx.Pipe()
        self._process = ctx.Process(target=_serve, args=(child, readiness), name='RF-default-features', daemon=True)
        started = time.monotonic()
        try:
            self._process.start()
            self._pid = self._process.pid
            child.close()
            if not self._connection.poll(startup_timeout):
                raise TimeoutError('Feature process readiness timed out')
            if self._connection.recv() != ('ready', None):
                raise RuntimeError('Invalid feature process readiness response')
        except BaseException:
            child.close()
            self.close()
            raise
        self.startup_seconds = time.monotonic() - started

    @property
    def pid(self):
        return self._pid

    def __call__(self, context):
        self.begin(context)
        return self.finish()

    def begin(self, context):
        """Start one transaction on an idle server; owned by the dispatcher."""
        if self._broken or self._closed or not self._process.is_alive():
            raise RuntimeError('Feature process unavailable; recreate worker to recover')
        if getattr(self, '_pending_since', None) is not None:
            raise RuntimeError('Feature process already has an outstanding transaction')
        if context.event.num_samples > MAX_PROCESS_EVENT_SAMPLES:
            raise ValueError('Process feature backend supports at most 4096 captured IQ samples; use thread backend for larger events')
        self._token += 1
        payload=pickle.dumps((_PROTOCOL,self._token,_request(context)),protocol=5)
        if len(payload)>MAX_REQUEST_BYTES:
            raise ValueError('Process feature request metadata exceeds bounded 48 KiB transport envelope')
        self._pending_since = time.perf_counter()
        try:
            self._connection.send_bytes(payload)
        except (EOFError, BrokenPipeError, OSError):
            self._broken = True
            self.close()
            raise

    def ready(self):
        return (self._connection.poll(0) or not self._process.is_alive()
                or time.perf_counter()-self._pending_since >= self.timeout)

    def finish(self):
        try:
            remaining = max(0., self.timeout-(time.perf_counter()-self._pending_since))
            if not self._connection.poll(remaining):
                raise TimeoutError('Feature process calculation timed out')
            version, token, response = self._connection.recv()
            if version != _PROTOCOL or token != self._token:
                self._broken = True
                self.close()
                raise RuntimeError('Foreign or obsolete feature transport response')
            status, record = response
            if status == 'error':
                raise ValueError('Feature process analysis failed: ' + record)
            if status != 'ok':
                raise RuntimeError('Malformed feature process response')
            record, child_seconds, child_cpu_seconds = record
            result = _result(record)
            self.child_analysis_seconds += child_seconds
            self.child_cpu_seconds += child_cpu_seconds
            self.roundtrip_seconds += time.perf_counter()-self._pending_since
            return result
        except (EOFError, BrokenPipeError, OSError, TimeoutError):
            self._broken = True
            self.close()
            raise
        finally:
            self._pending_since = None

    def close(self):
        if self._closed:
            return
        if self._process.pid is not None:
            if self._process.is_alive() and not self._broken:
                try:
                    self._connection.send(None)
                except (EOFError, BrokenPipeError, OSError):
                    pass
                self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            if self._process.is_alive():
                raise TimeoutError('Feature process cleanup incomplete')
        self._connection.close()
        self._process.close()
        self._closed = True


class ParallelProcessAnalysisBackend:
    """Four persistent spawn servers; callers and jobs remain bounded by worker.

    Each pipe has one exclusive RPC owner. No shared mutable IQ or unbounded
    executor input queue crosses this boundary. External sinks stay in parent.
    """

    def __init__(self, *, timeout=30.0, workers=4, readiness=None):
        if type(workers) is not int or not 1<=workers<=4:
            raise ValueError('Persistent feature server count must be an integer from 1 to 4')
        self._backends = []
        self._available = queue.Queue(maxsize=workers)
        try:
            for _ in range(workers):
                backend = ProcessAnalysisBackend(timeout=timeout, readiness=readiness)
                self._backends.append(backend)
                self._available.put_nowait(backend)
        except BaseException:
            self.close()
            raise

    @property
    def pid(self):
        return self._backends[0].pid

    @property
    def pids(self):
        return tuple(b.pid for b in self._backends)

    @property
    def child_analysis_seconds(self):
        return sum(b.child_analysis_seconds for b in self._backends)

    @property
    def child_cpu_seconds(self):
        return sum(b.child_cpu_seconds for b in self._backends)

    @property
    def roundtrip_seconds(self):
        return sum(b.roundtrip_seconds for b in self._backends)

    def __call__(self, context):
        backend = self._available.get()
        try:
            return backend(context)
        finally:
            self._available.put_nowait(backend)

    def close(self):
        errors = []
        for backend in self._backends:
            try:
                backend.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
