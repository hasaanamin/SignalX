"""Explicit spawn-isolated capture with parent-side feature delivery.

One synchronous, bounded block RPC preserves the existing return/exception
ownership contract. It is not an unbounded acquisition queue. Raw detection,
adaptation, ring and assembly live together in the child; arbitrary sinks and
feature callbacks stay in the parent. No writable IQ view crosses the boundary.
"""
from __future__ import annotations

import dataclasses
import multiprocessing
import threading
import time

import numpy as np

from signal_processing.capture.sidecar import CaptureSidecar
from signal_processing.core import IQBlock, SignalEvent, StreamInfo
from signal_processing.detection import GaussianNoiseEstimator, AdaptiveRobustNoiseEstimator
from .pipeline import StreamingCapturePipeline, PipelineState
from .worker import FeatureAnalysisWorker

_VERSION = 'rf-capture-v1'


def _encode_sidecar(sc):
    fields = {f.name: getattr(sc, f.name) for f in dataclasses.fields(sc)
              if f.name not in ('event', 'features', 'analysis_result')}
    event = {f.name: getattr(sc.event, f.name) for f in dataclasses.fields(sc.event)
             if f.name not in ('stream', 'iq')}
    return fields, event, dataclasses.asdict(sc.event.stream), sc.event.iq.tobytes()


def _decode_sidecar(record):
    fields, event, stream, iq = record
    return CaptureSidecar(**fields, event=SignalEvent(**event, stream=StreamInfo(**stream),
                          iq=np.frombuffer(iq, dtype=np.complex64)))


def _snapshot(pipe):
    return dict(state=pipe.state.value, consumed_end_sample=pipe.consumed_end_sample,
        received_end_sample=pipe.received_end_sample, **pipe.retention_snapshot())


def _serve(connection, stream, noise_mode, noise_options, options):
    pipe = None
    try:
        estimator = (GaussianNoiseEstimator if noise_mode == 'fixed' else AdaptiveRobustNoiseEstimator)(**noise_options)
        pipe = StreamingCapturePipeline(StreamInfo(**stream), noise_estimator=estimator,
                                        enable_feature_analysis=False, **options)
        connection.send((_VERSION, 0, None, [], _snapshot(pipe)))
        while True:
            version, token, operation, payload = connection.recv()
            if version != _VERSION:
                raise ValueError('Unknown capture transport protocol')
            error = None
            events = []
            try:
                if operation == 'block':
                    metadata, raw = payload
                    metadata['stream'] = StreamInfo(**metadata['stream'])
                    events = pipe.process_block(IQBlock(**metadata, iq=np.frombuffer(raw, np.complex64)))
                elif operation == 'flush':
                    events = pipe.flush()
                elif operation == 'reset':
                    pipe.reset()
                elif operation == 'close':
                    pipe.shutdown()
                else:
                    raise ValueError('Unknown capture operation')
            except Exception as exc:
                events = list(getattr(exc, 'completed_captures', ()))
                # Preserve the actual type/message without pickling immutable
                # ndarray/mapping-backed captures inside exception attributes.
                error = type(exc)(*exc.args)
            connection.send((_VERSION, token, error, list(map(_encode_sidecar, events)), _snapshot(pipe)))
            if operation == 'close':
                break
    except (EOFError, BrokenPipeError, OSError):
        pass
    except Exception as exc:
        try:
            connection.send((_VERSION, 0, exc, [], None))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if pipe is not None:
            pipe.shutdown()
        connection.close()


class IsolatedCapturePipeline:
    """Positioned capture API with an explicit separately owned interpreter.

    Configuration dictionaries are constructor inputs, not mutable live tuning.
    Input blocks are capped at 4096 samples; larger transport blocks must be
    sliced at the caller. Custom estimators remain on StreamingCapturePipeline.
    An unacknowledged backend death makes the consumed horizon unknown; it must
    never be relabelled as a known RF gap or silently retried.
    """

    def __init__(self, stream_info, *, noise_mode='fixed', noise_options=None,
                 pipeline_options=None, feature_options=None, enable_feature_analysis=True,
                 timeout=5.0):
        if noise_mode not in ('fixed', 'adaptive'):
            raise ValueError('Isolated capture supports fixed Gaussian or adaptive robust estimators')
        if isinstance(timeout, bool) or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be positive finite seconds')
        self._timeout = float(timeout)
        self._lock = threading.Lock()
        self._token = 0
        self._closed = self._broken = False
        self._snapshot = None
        self.consumed_horizon_known = True
        self._enable_features = enable_feature_analysis
        options = dict(pipeline_options or {})
        forbidden = {'noise_estimator', 'enable_feature_analysis', 'analysis_function', 'analysis_backend', 'stream_info'}
        if forbidden.intersection(options):
            raise ValueError('Use explicit isolated configuration boundaries')
        self.worker = FeatureAnalysisWorker(**(feature_options or {}))
        ctx = multiprocessing.get_context('spawn')
        self._connection, child = ctx.Pipe()
        self._process = ctx.Process(target=_serve, args=(child, dataclasses.asdict(stream_info),
            noise_mode, dict(noise_options or {}), options), name='RF-capture', daemon=True)
        try:
            self._process.start()
            child.close()
            if not self._connection.poll(20):
                raise TimeoutError('Capture process startup timed out')
            version, token, error, events, snapshot = self._connection.recv()
            if error is not None:
                raise error
            if (version, token) != (_VERSION, 0) or events:
                raise RuntimeError('Invalid capture readiness')
            self._snapshot = snapshot
        except BaseException:
            child.close()
            if self._process.pid is not None and self._process.is_alive():
                self._process.terminate()
                self._process.join(5)
            self._connection.close()
            self._process.close()
            self.worker.shutdown(timeout=5)
            raise

    @property
    def state(self):
        return PipelineState.FAULTED if self._broken else PipelineState(self._snapshot['state'])

    @property
    def consumed_end_sample(self):
        return self._snapshot['consumed_end_sample']

    @property
    def received_end_sample(self):
        return self._snapshot['received_end_sample']

    @property
    def capture_pid(self):
        return self._process.pid

    def retention_snapshot(self):
        return {k:self._snapshot[k] for k in ('retained_noise_samples', 'retained_history_samples')}

    def _rpc(self, operation, payload=None, timeout=None):
        if self._closed or self._broken:
            raise RuntimeError('Capture process unavailable; recreate pipeline')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Concurrent capture operations are not supported')
        try:
            self._token += 1
            self._connection.send((_VERSION, self._token, operation, payload))
            if not self._connection.poll(self._timeout if timeout is None else timeout):
                raise TimeoutError('Capture RPC timed out; commit horizon unacknowledged')
            version, token, error, records, snapshot = self._connection.recv()
            if (version, token) != (_VERSION, self._token):
                raise RuntimeError('Foreign capture response')
            self._snapshot = snapshot
            captures = tuple(map(_decode_sidecar, records))
            if error is not None:
                error.completed_captures = captures
                error.consumed_end_sample = self.consumed_end_sample
                error.received_end_sample = self.received_end_sample
                raise error
            return list(captures)
        except (EOFError, BrokenPipeError, OSError, TimeoutError):
            self._broken = True
            self.consumed_horizon_known = False
            raise
        finally:
            self._lock.release()

    def _run(self, operation, payload=None):
        try:
            captures = self._rpc(operation, payload)
        except Exception as exc:
            # Raw recovery remains independent of analysis. Ordinary successful
            # captures are dispatched below; fault captures are owned by the
            # original typed exception, not an undiscoverable local list.
            raise
        if not self._enable_features:
            return captures
        outputs = []
        try:
            for sc in captures:
                outputs.append(self.worker.submit(sc))
        except Exception as exc:
            exc.completed_captures = tuple(captures)
            exc.consumed_end_sample = self.consumed_end_sample
            exc.received_end_sample = self.received_end_sample
            raise
        return outputs

    def process_block(self, block):
        if not isinstance(block, IQBlock) or block.num_samples > 4096:
            raise ValueError('Expected validated IQBlock of at most 4096 samples')
        metadata = {f.name:getattr(block,f.name) for f in dataclasses.fields(block) if f.name not in ('stream','iq')}
        metadata['stream'] = dataclasses.asdict(block.stream)
        return self._run('block', (metadata,block.iq.tobytes()))

    def flush(self):
        return self._run('flush')

    def reset(self):
        self._rpc('reset')
        self.worker.reset()

    def shutdown(self, drain=True, timeout=5.0):
        if self._closed:
            return
        deadline = time.monotonic()+timeout
        self.worker.shutdown(drain=drain,timeout=max(0,deadline-time.monotonic()))
        if not self._broken:
            self._rpc('close', timeout=max(0,deadline-time.monotonic()))
        if self._broken and self._process.is_alive():
            self._process.terminate()
        self._process.join(max(0,deadline-time.monotonic()))
        if self._process.is_alive():
            raise TimeoutError('Capture process cleanup incomplete')
        self._connection.close()
        self._process.close()
        self._closed = True
