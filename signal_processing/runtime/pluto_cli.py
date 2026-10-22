"""Run adaptive preprocessing or LTE PBCH/MIB decoding from a receive-only Pluto."""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import sys
import threading
import time
from contextlib import ExitStack

import numpy as np


def run(mode, factory, *, initial_noise_power=None, result_sink=None,
        max_latency_ms=100., performance_qos=True, priority_class='high', dsp_scheduling='priority'):
    """Finite live run; result_sink receives captures or LTE broadcast batches.

    Every capture reaches the sink, with feature analysis disabled. Sink work
    and flush are timed. This reports observed host/DMA acquisition success;
    Pluto's standard IQ scan has no sample counter to prove RF continuity.
    """
    from ._performance import performance_scope, realtime_scope
    from ._publication import OrderedPublisher
    from .process_source_runner import ProcessSourceRunner

    if mode not in ('adaptive', 'lte'):
        raise ValueError('Mode must be adaptive or lte')
    if not math.isfinite(max_latency_ms) or max_latency_ms <= 0:
        raise ValueError('Supply a positive maximum latency')
    if priority_class not in ('normal', 'high'):
        raise ValueError('Process priority must be normal or high')
    if dsp_scheduling not in ('priority', 'mmcss'):
        raise ValueError('DSP scheduling must be priority or mmcss')
    if mode == 'adaptive' and (isinstance(initial_noise_power, bool)
            or not isinstance(initial_noise_power, (int, float))
            or not math.isfinite(initial_noise_power) or initial_noise_power <= 0):
        raise ValueError('Adaptive needs a measured positive quiet power in normalized IQ units')
    stream = factory.stream
    runner = ProcessSourceRunner(factory, stream, capacity=16,
        max_block_samples=factory.block_size, performance_qos=performance_qos)
    errors = []
    processed = captures = incomplete = observations = mibs = 0
    latency = collections.deque(maxlen=4096)
    maximum = 0.
    last_received = eof_tail = None
    latency_lock = threading.Lock()
    started = None
    scheduling = realtime = None
    receiver_state = execution = None
    publisher = receiver = None

    def record(anchor):
        nonlocal maximum
        if anchor is not None:
            delay = (time.perf_counter_ns()-anchor)/1e6
            with latency_lock:
                latency.append(delay)
                maximum = max(maximum, delay)

    def publish(item):
        results, anchor = item
        if result_sink is not None:
            for result in results if mode == 'adaptive' else (results,):
                result_sink(result)
        record(anchor)

    def emit(results, anchor):
        nonlocal captures, incomplete, observations, mibs
        if mode == 'adaptive':
            captures += len(results)
            incomplete += sum(bool(x.known_lost_samples or x.unknown_loss) for x in results)
            nonempty = bool(results)
        else:
            observations += len(results.synchronization)
            mibs += len(results.mibs)
            nonempty = bool(results.synchronization or results.mibs)
        if nonempty:
            publisher.submit((results, anchor))
        else:
            record(anchor)

    try:
        with ExitStack() as resources:
            scheduling = resources.enter_context(performance_scope(performance_qos))
            if mode == 'adaptive':
                from signal_processing.detection import POSITIVE_SNR_IID_1920
                receiver = POSITIVE_SNR_IID_1920.create_pipeline(stream, initial_noise_power,
                    enable_feature_analysis=False)
                resources.callback(receiver.shutdown)
                execution = {'feature_analysis': False, 'profile_id': POSITIVE_SNR_IID_1920.profile_id}
            else:
                from signal_processing.lte import LTEBroadcastReceiver, LTESyncConfig
                receiver = LTEBroadcastReceiver(stream, LTESyncConfig.low_snr(stream.sample_rate_hz),
                    sync_backend='auto', decode_cadence='bch_block')
                execution = receiver.execution
            if sys.platform == 'win32' and priority_class == 'high':
                import psutil
                process = psutil.Process()
                previous = process.nice()
                resources.callback(process.nice, previous)
                process.nice(psutil.HIGH_PRIORITY_CLASS)
            publisher = OrderedPublisher(publish)
            # Cleanup registration order matters: stop acquisition, finish
            # actual output, then restore process/thread policy.
            resources.callback(publisher.finish)
            runner.start(start_paused=True)
            resources.callback(runner.shutdown, drain=False, timeout=5.)
            if sys.platform == 'win32' and priority_class == 'high':
                psutil.Process(runner.pid).nice(psutil.HIGH_PRIORITY_CLASS)
            realtime = resources.enter_context(realtime_scope(performance_qos and dsp_scheduling == 'mmcss'))
            import gc
            gc.collect()
            started = time.perf_counter()
            runner.begin()
            while (lease := runner.get(timeout=5.)) is not None:
                try:
                    last_received = lease.envelope.received_monotonic_ns
                    results = (receiver.process_block(lease.envelope.block) if mode == 'adaptive'
                               else receiver.process_envelope(lease.envelope))
                    emit(results, last_received)
                    processed += lease.envelope.block.num_samples
                except BaseException:
                    runner.ack(lease, outcome='failed')
                    raise
                runner.ack(lease)
            emit(receiver.flush(), last_received)
            publisher.finish()
            if mode == 'lte':
                receiver_state = receiver.snapshot()
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
    finally:
        # Also cover failed initialization before cleanup registration.
        try:
            runner.shutdown(drain=False, timeout=5.)
        except Exception as exc:
            errors.append(f'acquisition cleanup: {type(exc).__name__}: {exc}')
        if last_received is not None:
            eof_tail = (time.perf_counter_ns()-last_received)/1e6
    acquisition = runner.snapshot()
    try:
        source = runner.source_snapshot()
    except Exception as exc:
        errors.append(f'source accounting: {type(exc).__name__}: {exc}')
        source = None
    publication = publisher.snapshot() if publisher is not None else None
    acquisition_passed = bool(source is not None and source.finished and not errors
        and not source.unknown_loss and source.hardware_overflow_events == source.transport_errors == 0
        and source.returned_samples == processed == source.requested_samples
        and acquisition.state == 'CLOSED' and acquisition.accounting_balanced
        and acquisition.processed_samples == processed
        and acquisition.queue_dropped_samples == acquisition.failed_samples == 0
        and acquisition.cancelled_samples == acquisition.unaccepted_cancelled_samples == 0
        and acquisition.outstanding_deliveries == incomplete == 0)
    tail_p99 = float(np.percentile(tuple(latency), 99)) if latency else None
    latency_passed = bool(latency and maximum < max_latency_ms
        and tail_p99 <= max_latency_ms/2 and eof_tail is not None and eof_tail < max_latency_ms)
    return dict(mode=mode, sample_rate_hz=stream.sample_rate_hz,
        processed_samples=processed, captures=captures, incomplete_captures=incomplete,
        observations=observations, decoded_mibs=mibs, source=dataclasses.asdict(source) if source else None,
        acquisition=dataclasses.asdict(acquisition), publication=publication,
        acquisition_passed=acquisition_passed, latency_passed=latency_passed,
        passed=acquisition_passed and latency_passed, hardware_qualified=False,
        hardware_qualification_note='DMA overflow flag monitored; no hardware sample counter. Validate continuity with an independent known signal on the target Pluto/driver.',
        latency_scope='Successful libiio refill to completed result sink, including ordered output flush; excludes RF integration and time already spent in device/driver buffers.',
        latency_tail_blocks=4096, latency_ms_p99=tail_p99, maximum_latency_ms=maximum,
        eof_tail_latency_ms=eof_tail, elapsed_seconds=time.perf_counter()-started if started else None,
        performance_qos=scheduling, realtime_scheduling=realtime, execution=execution,
        dsp_scheduling=dsp_scheduling, receiver=receiver_state, errors=errors)


def main(argv=None):
    from signal_processing.core import StreamInfo
    from signal_processing.sources.pluto_source import PlutoSourceFactory
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('adaptive', 'lte'))
    parser.add_argument('--uri', required=True)
    parser.add_argument('--center-frequency', type=int, required=True)
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--block-size', type=int, default=16384)
    parser.add_argument('--rf-bandwidth', type=int, default=1_500_000)
    parser.add_argument('--gain-db', type=float, default=30.)
    parser.add_argument('--initial-noise-power', type=float)
    parser.add_argument('--require-mib', action='store_true', help='LTE: fail if no CRC-valid MIB is decoded')
    parser.add_argument('--max-latency-ms', type=float, default=100.)
    parser.add_argument('--cpu-qos', choices=('performance', 'system'), default='performance')
    parser.add_argument('--priority-class', choices=('normal', 'high'), default='high')
    parser.add_argument('--dsp-scheduling', choices=('priority', 'mmcss'), default='priority')
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 86400:
        parser.error('seconds must be in (0, 86400]')
    if not 1024 <= args.block_size <= 65536:
        parser.error('block-size must be in [1024, 65536]')
    if args.mode == 'adaptive' and (args.initial_noise_power is None or not math.isfinite(args.initial_noise_power) or args.initial_noise_power <= 0):
        parser.error('adaptive requires positive --initial-noise-power, measured with the same manual gain and normalized IQ')
    if args.require_mib and args.mode != 'lte':
        parser.error('--require-mib requires LTE mode')
    if not math.isfinite(args.max_latency_ms) or args.max_latency_ms <= 0:
        parser.error('max-latency-ms must be finite and positive')
    stream = StreamInfo('pluto-rx', args.uri, 1_920_000., args.center_frequency)
    try:
        factory = PlutoSourceFactory(args.uri, stream, args.block_size,
            math.ceil(args.seconds*1_920_000/args.block_size), args.rf_bandwidth, args.gain_db)
    except ValueError as exc:
        parser.error(str(exc))

    def output(value):
        if args.mode == 'adaptive':
            # Captures contain IQ; avoid unbounded JSON arrays. This CLI emits
            # capture metadata. Python callers can persist each raw capture in
            # result_sink, whose full cost is included in the same latency gate.
            record = dict(type='capture', event_id=value.event.event_id,
                start_sample=value.event.event_start_sample, end_sample=value.event.event_end_sample,
                known_lost_samples=value.known_lost_samples, unknown_loss=value.unknown_loss)
            print(json.dumps(record, allow_nan=False))
        else:
            for result in value.synchronization:
                print(json.dumps(dict(type='observation', **result.to_dict()), allow_nan=False))
            for result in value.mibs:
                print(json.dumps(dict(type='mib', **result.to_dict()), allow_nan=False))
        sys.stdout.flush()

    result = run(args.mode, factory, initial_noise_power=args.initial_noise_power,
        result_sink=output, max_latency_ms=args.max_latency_ms,
        performance_qos=args.cpu_qos == 'performance', priority_class=args.priority_class,
        dsp_scheduling=args.dsp_scheduling)
    print(json.dumps(dict(type='summary', **result), allow_nan=False), flush=True)
    if not result['passed']:
        return 1
    return 2 if args.require_mib and result['decoded_mibs'] == 0 else 0


if __name__ == '__main__':
    raise SystemExit(main())
