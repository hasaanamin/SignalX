"""Read-only attribution of genuine production paths (not release qualification).

Stage timers are inclusive and nested: do not add parent and child durations.
Profiles/timers perturb timing; scheduled capture-only is a diagnostic control,
not a claim that the feature-enabled end-to-end configuration passes.
"""
import argparse
import collections
import cProfile
import functools
import json
import os
from pathlib import Path
import platform
import pstats
import queue
import sys
import time

import numpy as np

from signal_processing.core import IQBlock, StreamInfo


class Timers:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.values = collections.defaultdict(lambda: dict(calls=0, wall_s=0., thread_cpu_s=0., maximum_wall_s=0.))
        self.originals = []

    def wrap(self, owner, name, label=None):
        if not self.enabled:
            return
        original = getattr(owner, name)
        key = label or f'{owner.__name__}.{name}'
        @functools.wraps(original)
        def measured(*args, **kwargs):
            wall, cpu = time.perf_counter(), time.thread_time()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter()-wall
                entry = self.values[key]
                entry['calls'] += 1
                entry['wall_s'] += elapsed
                entry['thread_cpu_s'] += time.thread_time()-cpu
                entry['maximum_wall_s'] = max(entry['maximum_wall_s'], elapsed)
        self.originals.append((owner, name, original))
        setattr(owner, name, measured)

    def close(self):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)

    def concatenate(self):
        original = np.concatenate
        def measured(arrays, *args, **kwargs):
            caller = sys._getframe(1).f_code
            key = f'concatenate:{Path(caller.co_filename).name}:{caller.co_name}'
            start = time.perf_counter()
            result = original(arrays, *args, **kwargs)
            entry = self.values[key]
            elapsed = time.perf_counter()-start
            entry['calls'] += 1
            entry['wall_s'] += elapsed
            entry['maximum_wall_s'] = max(entry['maximum_wall_s'], elapsed)
            entry['output_bytes'] = entry.get('output_bytes', 0)+result.nbytes
            return result
        self.originals.append((np, 'concatenate', original))
        np.concatenate = measured


def lte(args, timers):
    from signal_processing.lte import LTEBroadcastReceiver, LTESyncConfig
    from signal_processing.lte.receiver import LTESynchronizer
    from signal_processing.lte.channel import LTEChannel
    from signal_processing.lte import broadcast, pbch
    stream = StreamInfo('attribution', 'real-encoded-file', args.sample_rate, 2.14e9)
    receiver = LTEBroadcastReceiver(stream, LTESyncConfig.low_snr(args.sample_rate), numba_threads=1)
    if args.allocations:
        timers.concatenate()
    for owner, names in (
        (LTEChannel, ('process',)),
        (LTESynchronizer, ('_search', '_tracking_candidates', '_blind_candidates', '_bounded_scores', '_decode_integrated')),
        (LTEBroadcastReceiver, ('_search',)),
        (pbch.PBCHDecoder, ('decode',)),
        (broadcast, ('extract_slot',)),
        (pbch, ('recover_bch', 'slot_soft_bits')),
    ):
        for name in names:
            timers.wrap(owner, name)
    counts = collections.Counter()
    n = round(args.seconds*args.sample_rate)
    profiler = cProfile.Profile() if args.profile else None
    start, cpu = time.perf_counter(), time.thread_time()
    if profiler:
        profiler.enable()
    try:
        with args.input.open('rb') as source:
            for seq, position in enumerate(range(0, n, 16384)):
                x = np.fromfile(source, '<c8', min(16384, n-position))
                if not len(x):
                    raise ValueError('Input shorter than requested extent')
                batch = receiver.process_block(IQBlock(stream, seq, position, None, x))
                counts['mibs'] += len(batch.mibs)
                counts['sync'] += len(batch.synchronization)
            batch = receiver.flush()
            counts['mibs'] += len(batch.mibs)
            counts['sync'] += len(batch.synchronization)
    finally:
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(args.output.with_suffix('.prof')))
            pstats.Stats(profiler).sort_stats('cumulative').print_stats(35)
    wall, cpu = time.perf_counter()-start, time.thread_time()-cpu
    return dict(scope='contiguous offline LTE; true samples, no source loss or generic features',
                samples=n, wall_s=wall, thread_cpu_s=cpu, input_duration_s=args.seconds,
                samples_per_wall_second=n/wall, results=dict(counts), receiver=receiver.snapshot())


def adaptive(args, timers):
    from signal_processing.runtime import StreamingCapturePipeline
    from signal_processing.detection import AdaptiveRobustNoiseEstimator
    from signal_processing.detection.energy_detector import EnergyDetector
    from benchmarks.process_source import _ScheduledIQ
    stream = StreamInfo('attribution', 'scheduled-synthetic', 1e6)
    estimator = AdaptiveRobustNoiseEstimator(window_size=256, initial_noise_power_linear=1e-4,
        time_constant_s=.5, adaptation_window_size=1024)
    pipe = StreamingCapturePipeline(stream, noise_estimator=estimator, enable_feature_analysis=False,
        window_size=256, hop_size=64, ring_capacity=4096, start_persistence=2, end_persistence=2,
        pre_trigger_samples=256, pre_signal_margin_samples=0, post_signal_samples=0,
        min_post_trigger_samples=0, max_event_samples=4096, fixed_quiet_batch=True)
    for owner, names in (
        (StreamingCapturePipeline, ('_process_full_band_batch', '_process_detection_chunk', '_sync_pending_age')),
        (AdaptiveRobustNoiseEstimator, ('get_estimate', 'update_samples', 'advance_stream_samples', 'set_pending_samples')),
        (EnergyDetector, ('_prepare_full_band', '_commit_full_band', '_linear_thresholds')),
    ):
        for name in names:
            timers.wrap(owner, name)
    generator = _ScheduledIQ(1024)
    n = round(args.seconds*1e6)
    # Input preparation is deliberately outside each processing CPU/wall sample.
    profiler = cProfile.Profile() if args.profile else None
    cpu = wall = 0.
    captures = 0
    block_times = []  # Bounded by explicitly requested diagnostic extent.
    try:
        for seq, position in enumerate(range(0, n, 1024)):
            block = IQBlock(stream, seq, position, None, generator.generate(position, min(1024, n-position)))
            start, start_cpu = time.perf_counter(), time.thread_time()
            if profiler:
                profiler.enable()
            try:
                emitted = pipe.process_block(block)
            finally:
                if profiler:
                    profiler.disable()
            dt = time.perf_counter()-start
            wall += dt
            cpu += time.thread_time()-start_cpu
            block_times.append(dt*1000)
            captures += len(emitted)
        captures += len(pipe.flush())
    finally:
        pipe.shutdown()
        if profiler:
            profiler.dump_stats(str(args.output.with_suffix('.prof')))
            pstats.Stats(profiler).sort_stats('cumulative').print_stats(40)
    return dict(scope='offline adaptive production pipeline, same IQ; no features/transport wait',
                samples=n, wall_s=wall, thread_cpu_s=cpu, captures=captures,
                processing_ms_percentiles=np.percentile(block_times, [50, 95, 99, 100]).tolist(),
                samples_per_wall_second=n/wall, snapshot=pipe.retention_snapshot())


def capture_only(args):
    from signal_processing.runtime.capture_stream import CaptureStreamProcess
    from benchmarks.process_source import ProcessScheduledSource
    stream = StreamInfo('bounded-acquisition', 'scheduled-synthetic', 1e6)
    source = ProcessScheduledSource(stream, round(args.seconds*1e6), 1024, 16)
    capture = None
    counts = collections.Counter()
    wall = cpu = 0.
    latencies = []
    try:
        capture = CaptureStreamProcess(source._queue, stream, noise_mode='adaptive',
            noise_options=dict(window_size=256, initial_noise_power_linear=1e-4, time_constant_s=.5,
                               adaptation_window_size=1024),
            pipeline_options=dict(window_size=256, hop_size=64, ring_capacity=4096,
                start_persistence=2, end_persistence=2, pre_trigger_samples=256,
                pre_signal_margin_samples=0, post_signal_samples=0, min_post_trigger_samples=0,
                max_event_samples=4096, fixed_quiet_batch=True))
        started = time.perf_counter()
        source.start(started)
        while not capture.done:
            try:
                record = capture.get(.1)
            except queue.Empty:
                if time.perf_counter()-started > args.seconds+15:
                    raise TimeoutError('Diagnostic source/capture did not finish')
                continue
            try:
                if record['error']:
                    raise RuntimeError(record['error'])
                counts['captures'] += len(record['captures'])
                if record['sequence'] >= 0:
                    counts['samples'] += record['num_samples']
                    counts['blocks'] += 1
                    delay = time.perf_counter()-record['arrival']
                    counts['deadline_misses'] += delay > .001024
                    latencies.append(delay*1000)
                    wall += record['processing_seconds']
                    cpu += record['cpu_seconds']
            finally:
                capture.task_done()
        source.join(5)
        report = source.finish()
        return dict(scope='scheduled adaptive capture-only diagnostic; NOT end-to-end feature acceptance',
            seconds=args.seconds, counts=dict(counts), processing_wall_s=wall, processing_thread_cpu_s=cpu,
            acquisition_fraction=counts['samples']/round(args.seconds*1e6), source=report,
            latency_ms_percentiles=np.percentile(latencies, [50, 95, 99, 100]).tolist(),
            drained=capture.drained)
    finally:
        if capture is not None:
            capture.close()
        source.close()


def _profile_capture(path, *args):
    # A spawn child imports the unmodified production module. Profiling is a
    # benchmark wrapper, never an alternate capture implementation.
    from signal_processing.runtime.capture_stream import _stream_capture
    profiler = cProfile.Profile()
    try:
        profiler.runcall(_stream_capture, *args)
    finally:
        profiler.dump_stats(path)


def full_profile(args):
    from signal_processing.runtime import capture_stream, FeatureAnalysisWorker
    from benchmarks.stream_isolated import run
    original_capture = capture_stream._stream_capture
    original_worker = FeatureAnalysisWorker._worker_loop
    capture_stream._stream_capture = functools.partial(_profile_capture,
        str(args.output.with_suffix('.capture.prof').resolve()))
    def profile_worker(worker):
        profiler = cProfile.Profile()
        try:
            profiler.runcall(original_worker, worker)
        finally:
            profiler.dump_stats(str(args.output.with_suffix('.worker.prof')))
    FeatureAnalysisWorker._worker_loop = profile_worker
    try:
        result = run(args.seconds, adaptive=True, profile='positive_snr', result_order='completion')
    finally:
        capture_stream._stream_capture = original_capture
        FeatureAnalysisWorker._worker_loop = original_worker
    result['scope'] = 'INSTRUMENTED full adaptive run; call attribution, NOT timing qualification'
    for suffix in ('.capture.prof', '.worker.prof'):
        print(suffix)
        pstats.Stats(str(args.output.with_suffix(suffix))).sort_stats('cumulative').print_stats(40)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('lte', 'adaptive', 'capture_only', 'full_profile'))
    parser.add_argument('--seconds', type=float, default=1.)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--sample-rate', type=float, choices=(1_920_000., 2_000_000.), default=2_000_000.)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--no-stage-timers', action='store_true')
    parser.add_argument('--allocations', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 30 or args.output.exists():
        raise ValueError('Use a new output and duration in (0,30]')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    timers = Timers(not args.no_stage_timers)
    try:
        result = (globals()[args.mode](args) if args.mode in ('capture_only', 'full_profile')
                  else globals()[args.mode](args, timers))
        result.update(stages_inclusive=dict(timers.values), cprofile=args.profile,
                      timing_note='Nested stage wall times overlap; Windows short-call thread CPU is quantized and must not be used for stage shares.',
                      python=platform.python_version(), openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2))
    finally:
        timers.close()


if __name__ == '__main__':
    main()
