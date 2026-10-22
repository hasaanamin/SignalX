"""Receive LTE sync and optional PBCH/MIB from raw complex64, without truth labels."""
import argparse
import collections
import json
import math
from pathlib import Path
import sys
import time

from signal_processing.core import StreamInfo
from signal_processing.runtime import SourceRunner,ProcessSourceRunner,ReplayFileFactory
from signal_processing.sources import RecordedFileIQSource, ReplayIQSource, SourceMetadata


def _main(argv=None, *, scheduling, reservation):
    import numpy as np
    from .receiver import LTESynchronizer
    from .config import LTESyncConfig
    from .broadcast import LTEBroadcastReceiver
    from .mib import LTEBroadcastBatch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Little-endian complex64 IQ")
    parser.add_argument("--sample-rate", type=float, choices=(1920000., 2000000.), default=2000000.)
    parser.add_argument("--center-frequency", type=float)
    parser.add_argument("--block-size", type=int, default=16384)
    parser.add_argument('--file-backend',choices=('buffered','mmap'),default='buffered')
    parser.add_argument("--origin", type=int, default=0)
    parser.add_argument("--scheduled", action="store_true", help="Independent paced acquisition; otherwise lossless offline replay")
    parser.add_argument("--require-lock", action="store_true")
    parser.add_argument("--decode-mib", action="store_true", help="Decode CRC-valid FDD/normal-CP PBCH/MIB")
    parser.add_argument("--require-mib", action="store_true", help="Enable MIB decoding and exit 2 if no MIB is recovered")
    parser.add_argument("--profile", choices=("compatibility", "low_snr"), default="compatibility",
                        help="Low-SNR profile integrates four half-frames; increases dwell/memory/CPU")
    parser.add_argument("--enforce-acquisition", action="store_true")
    parser.add_argument('--source-backend',choices=('process','thread'),default='process',
                        help='Scheduled acquisition: bounded spawn process (default) or thread compatibility')
    parser.add_argument('--sync-backend',choices=('auto','numpy','numba'),default='auto')
    parser.add_argument('--dsp-cpus',type=int,nargs='+')
    parser.add_argument('--source-cpus',type=int,nargs='+')
    parser.add_argument('--cpu-qos',choices=('performance','system'),default='performance',
                        help='Scoped Windows process performance scheduling during acquisition')
    parser.add_argument('--priority-class',choices=('normal','above-normal','high'),default=None,
                        help='Windows process priority; defaults to high with enforced process acquisition, normal otherwise')
    parser.add_argument('--dsp-scheduling',choices=('priority','mmcss'),default='priority',
                        help='DSP uses process priority by default; optional MMCSS has a CPU quota')
    parser.add_argument('--dsp-wait',choices=('spin','block'),default=None,
                        help='Enforced performance acquisition defaults to spin on the reserved DSP core; block reduces idle CPU use')
    parser.add_argument('--dsp-smt',choices=('reserve','shared'),default='reserve',
                        help='Spin mode reserves the selected core\'s SMT sibling; shared uses less CPU')
    parser.add_argument('--decode-cadence',choices=('every_frame','bch_block'),default='every_frame')
    parser.add_argument('--max-latency-ms',type=float,
                        help='With --scheduled, enforce maximum block-availability to result-publication latency')
    args = parser.parse_args(argv)
    if args.dsp_wait is None:
        args.dsp_wait = ('spin' if args.scheduled and args.enforce_acquisition
                         and args.cpu_qos == 'performance' and args.source_backend == 'process' else 'block')
    if args.dsp_wait == 'spin' and (not args.scheduled or args.source_backend != 'process'):
        parser.error('--dsp-wait spin requires scheduled process acquisition')
    if args.priority_class is None:
        args.priority_class = ("high" if sys.platform == "win32" and args.scheduled
                               and args.enforce_acquisition and args.source_backend == "process" else "normal")
    placement = original_affinity = original_priority = None
    if args.dsp_cpus is not None or args.source_cpus is not None or args.priority_class != 'normal':
        if not args.scheduled or args.source_backend != 'process':
            parser.error('CPU placement requires scheduled process acquisition')
        import psutil
        placement = psutil.Process()
        original_affinity = placement.cpu_affinity()
        original_priority = placement.nice()
        if args.priority_class != 'normal' and sys.platform != 'win32':
            parser.error('Explicit priority classes require Windows')
        for cpus in (args.dsp_cpus,args.source_cpus):
            if cpus is not None and (not cpus or not set(cpus).issubset(original_affinity)):
                parser.error('CPU placement must use available logical CPUs')
    if not 1 <= args.block_size <= 262144 or args.origin < 0:
        parser.error("Invalid bounded block size/origin")
    if args.max_latency_ms is not None and (not args.scheduled or not math.isfinite(args.max_latency_ms) or args.max_latency_ms<=0):
        parser.error('--max-latency-ms requires scheduled input and a finite positive bound')
    stream = StreamInfo("lte-recording", str(args.input.resolve()), args.sample_rate, args.center_frequency)
    config = LTESyncConfig.low_snr(args.sample_rate) if args.profile == "low_snr" else LTESyncConfig(input_sample_rate_hz=args.sample_rate)
    receiver = (LTEBroadcastReceiver(stream,config,sync_backend=args.sync_backend,decode_cadence=args.decode_cadence)
                if args.decode_mib or args.require_mib else LTESynchronizer(stream,config,sync_backend=args.sync_backend))
    recording = RecordedFileIQSource(args.input, stream, args.block_size, start_sample=args.origin,
                                     read_backend=args.file_backend)
    replay = ReplayIQSource(recording, scheduled=args.scheduled, lateness_budget_s=.01,
                            metadata=SourceMetadata(), owns_source=True)
    runner = SourceRunner(replay, max_block_samples=args.block_size) if args.scheduled else None
    if args.scheduled and args.source_backend=='process':
        runner=ProcessSourceRunner(ReplayFileFactory(str(args.input.resolve()),stream,args.block_size,
            True,.01,args.origin,read_backend=args.file_backend),stream,capacity=16,max_block_samples=args.block_size,
            performance_qos=scheduling['requested'],acquisition_cpus=args.source_cpus)
    observations = confirmed = processed = 0
    mib_count = 0
    error = None
    latency=collections.deque(maxlen=4096)
    maximum_latency=0.
    dsp_latency=collections.deque(maxlen=4096)
    maximum_dsp_latency=0.
    maximum_dsp_latency_block = None
    slow_dsp_blocks = collections.deque(maxlen=32)
    import threading
    latency_lock = threading.Lock()
    last_release = eof_tail_latency_ms = None

    def publish_mibs(results):
        nonlocal mib_count
        for result in results:
            mib_count += 1
            print(json.dumps({"type": "mib", **result.to_dict()}, allow_nan=False))

    def publish(results):
        nonlocal observations, confirmed
        if isinstance(results, LTEBroadcastBatch):
            publish_mibs(results.mibs)
            results = results.synchronization
        for result in results:
            observations += 1
            confirmed += result.state == "SYNCHRONIZED"
            print(json.dumps({"type": "observation", **result.to_dict()}, allow_nan=False))

    def record_latency(anchor):
        nonlocal maximum_latency
        if anchor is not None:
            delay=max(0.,(time.perf_counter_ns()-anchor)/1e6)
            with latency_lock:
                latency.append(delay)
                maximum_latency=max(maximum_latency,delay)

    def publication(item):
        results,anchor = item
        publish(results)
        sys.stdout.flush()
        record_latency(anchor)

    from signal_processing.runtime._publication import OrderedPublisher
    publisher = OrderedPublisher(publication)
    def emit(results,anchor):
        nonempty = bool(results.synchronization or results.mibs) if isinstance(results,LTEBroadcastBatch) else bool(results)
        if nonempty:
            publisher.submit((results,anchor))
        else:
            record_latency(anchor)
    from contextlib import ExitStack
    from signal_processing.runtime._performance import realtime_scope
    realtime_resources = ExitStack()
    realtime_policy = None

    # JIT/import graphs belong to initialization, before any replay clock runs.
    # Leave automatic collection enabled during acquisition.
    import gc
    collection_started = time.perf_counter()
    startup_collected = gc.collect()
    startup_collection_seconds = time.perf_counter()-collection_started
    try:
        realtime_policy = realtime_resources.enter_context(realtime_scope(
            scheduling['requested'] and args.scheduled and args.dsp_scheduling == 'mmcss'))
        if runner:
            if placement is not None:
                runner.start(start_paused=True)
                if args.dsp_cpus is not None:
                    placement.cpu_affinity(args.dsp_cpus)
                if args.priority_class != 'normal':
                    priority = (psutil.HIGH_PRIORITY_CLASS if args.priority_class == 'high'
                                else psutil.ABOVE_NORMAL_PRIORITY_CLASS)
                    placement.nice(priority)
                    psutil.Process(runner.pid).nice(priority)
                runner.begin()
            else:
                runner.start()
            wait_options = {'spin':args.dsp_wait == 'spin'} if isinstance(runner,ProcessSourceRunner) else {}
            while (lease := runner.get(timeout=5,**wait_options)) is not None:
                last_release = lease.envelope.scheduled_monotonic_ns
                try:
                    before_cpu = time.process_time_ns()
                    before_wall = time.perf_counter_ns()
                    results=receiver.process_envelope(lease.envelope)
                    after_wall = time.perf_counter_ns()
                    after_cpu = time.process_time_ns()
                    delay=(after_wall-lease.envelope.scheduled_monotonic_ns)/1e6
                    dsp_latency.append(delay)
                    if delay > maximum_dsp_latency or delay > 50:
                        timing = dict(sequence=lease.envelope.block.sequence_number,
                            start_sample=lease.envelope.block.start_sample,latency_ms=delay,
                            processing_wall_ms=(after_wall-before_wall)/1e6,
                            processing_cpu_ms=(after_cpu-before_cpu)/1e6,
                            before_processing_ms=(before_wall-last_release)/1e6,
                            source_lateness_ms=(lease.envelope.received_monotonic_ns-last_release)/1e6)
                        if delay > maximum_dsp_latency:
                            maximum_dsp_latency_block = timing
                        if delay > 50:
                            slow_dsp_blocks.append(timing)
                    maximum_dsp_latency=max(maximum_dsp_latency,delay)
                    emit(results,lease.envelope.scheduled_monotonic_ns)
                except Exception:
                    runner.ack(lease, outcome="failed")
                    raise
                runner.ack(lease)
                processed += lease.envelope.block.num_samples
        else:
            while (envelope := replay.read_envelope()) is not None:
                emit(receiver.process_envelope(envelope),None)
                processed += envelope.block.num_samples
        emit(receiver.flush(),None)
        publisher.finish()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        # Real completed prefix results survive a later receiver exception.
        completed=LTEBroadcastBatch(tuple(getattr(exc, "completed_observations", ())),tuple(getattr(exc, "completed_mibs", ())))
        if completed.synchronization or completed.mibs:
            try:
                publisher.submit((completed,last_release))
            except Exception as publication_error:
                error += f'; {publication_error}'
    finally:
        try:
            if runner:
                runner.shutdown(drain=False)
            replay.close()
        except Exception as exc:
            error = error or f"cleanup: {type(exc).__name__}: {exc}"
        finally:
            try:
                publisher.finish()
            except Exception as exc:
                error=error or f'publication cleanup: {type(exc).__name__}: {exc}'
            finally:
                try:
                    realtime_resources.close()
                except Exception as exc:
                    error=error or f'scheduling cleanup: {type(exc).__name__}: {exc}'
                finally:
                    try:
                        reservation.close()
                    except Exception as exc:
                        error=error or f'core cleanup: {type(exc).__name__}: {exc}'
                    if placement is not None:
                        try:
                            placement.cpu_affinity(original_affinity)
                        finally:
                            placement.nice(original_priority)
    if last_release is not None:
        eof_tail_latency_ms = max(0.,(time.perf_counter_ns()-last_release)/1e6)
    replay_state = replay.snapshot()
    if isinstance(runner,ProcessSourceRunner):
        try:
            actual=runner.source_snapshot()
            if actual is not None: replay_state=actual
        except Exception as exc:
            error=error or f'source accounting: {type(exc).__name__}: {exc}'
    runner_state = runner.snapshot() if runner else None
    acquisition_passed = (error is None and replay_state.finished
        and processed == replay_state.total_samples and replay_state.expired_samples == 0
        and (runner_state is None or runner_state.accounting_balanced
             and runner_state.queue_dropped_samples == runner_state.outstanding_deliveries == 0))
    print(json.dumps({"type": "summary", "observations": observations, "confirmed": confirmed, "decoded_mibs": mib_count,
        "processed_samples": processed, "total_samples": replay_state.total_samples,
        "replay_expired_samples": replay_state.expired_samples,
        "queue_dropped_samples": runner_state.queue_dropped_samples if runner_state else 0,
        'source_backend':args.source_backend if args.scheduled else 'offline',
        'processing_cpus':args.dsp_cpus,'acquisition_cpus':args.source_cpus,
        'cpu_qos':args.cpu_qos,'performance_qos_applied':scheduling['applied'],
        'preferred_cpus':scheduling['preferred_cpus'],
        'thread_qos_applied':scheduling['thread_qos_applied'],
        'thread_relative_priority':scheduling['thread_relative_priority'],
        'smt_reservation':{key:reservation[key] for key in ('requested','applied','cpus','reason','stopped','affinity_restored','error')},
        'realtime_scheduling':realtime_policy,
        'acquisition_scheduling':runner.source_execution() if isinstance(runner,ProcessSourceRunner) else None,
        'priority_class':args.priority_class,
        'dsp_scheduling':args.dsp_scheduling,
        'dsp_wait':args.dsp_wait,
        'file_backend':args.file_backend,
        'startup_gc_collected':startup_collected,
        'startup_gc_seconds':startup_collection_seconds,
        'execution':receiver.execution,
        'publication':publisher.snapshot(),
        'dsp_latency_ms_p99':float(np.percentile(tuple(dsp_latency),99)) if dsp_latency else None,
        'maximum_dsp_latency_ms':maximum_dsp_latency if dsp_latency else None,
        'maximum_dsp_latency_block':maximum_dsp_latency_block,
        'slow_dsp_blocks':list(slow_dsp_blocks),
        'latency_ms_p99':float(np.percentile(latency,99)) if latency else None,
        'maximum_latency_ms':maximum_latency if latency else None,
        'eof_tail_latency_ms':eof_tail_latency_ms,
        'latency_passed':(bool(latency) and maximum_latency<args.max_latency_ms
                          and eof_tail_latency_ms is not None and eof_tail_latency_ms<args.max_latency_ms
                          if args.max_latency_ms is not None else None),
        'latency_scope':'Block availability to completed ordered stdout publication, INCLUDING bounded writer wait and flush. RF integration/confirmation dwell is separate.',
        "acquisition_passed": acquisition_passed, "receiver": receiver.snapshot(), "error": error}, allow_nan=False))
    if error:
        print(error, file=sys.stderr)
        return 1
    if args.enforce_acquisition and not acquisition_passed:
        return 1
    if args.max_latency_ms is not None and (not acquisition_passed or not latency or maximum_latency>=args.max_latency_ms
            or eof_tail_latency_ms is None or eof_tail_latency_ms>=args.max_latency_ms):
        return 1
    return 2 if (args.require_lock and confirmed == 0) or (args.require_mib and mib_count == 0) else 0


def main(argv=None):
    # Select policy before FFT/JIT planning too. The full parser below still
    # validates every option; the context restores policy even on parser exits.
    from signal_processing.runtime._performance import performance_scope
    from signal_processing.runtime._core_reservation import core_reservation
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--cpu-qos',choices=('performance','system'),default='performance')
    selector.add_argument('--dsp-cpus',type=int,nargs='+')
    selector.add_argument('--source-cpus',type=int,nargs='+')
    selector.add_argument('--scheduled',action='store_true')
    selector.add_argument('--enforce-acquisition',action='store_true')
    selector.add_argument('--source-backend',choices=('process','thread'),default='process')
    selector.add_argument('--dsp-wait',choices=('spin','block'),default=None)
    selector.add_argument('--dsp-smt',choices=('reserve','shared'),default='reserve')
    options,_ = selector.parse_known_args(argv)
    with performance_scope(options.cpu_qos == 'performance',prefer_fast_cpus=options.dsp_cpus is None and options.source_cpus is None) as scheduling:
        spin = options.dsp_wait == 'spin' or options.dsp_wait is None and options.enforce_acquisition
        enabled = (scheduling['applied'] and options.scheduled and options.source_backend == 'process'
                   and spin and options.dsp_smt == 'reserve')
        with core_reservation(scheduling['preferred_core_cpus'],enabled=enabled) as reservation:
            return _main(argv,scheduling=scheduling,reservation=reservation)


if __name__ == "__main__":
    raise SystemExit(main())
