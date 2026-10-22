"""Bounded adaptive preprocessing/capture with independent scheduled acquisition."""

import argparse
import collections
import dataclasses
import json
import math
import time
from pathlib import Path


def _run(
    path,
    *,
    sample_rate_hz=1_920_000.0,
    block_size=16384,
    initial_noise_power,
    capture_sink=None,
    processing_cpus=None,
    acquisition_cpus=None,
    queue_capacity=128,
    priority_class="normal",
    realtime_scheduling=True,
    dsp_scheduling="priority",
    file_backend="buffered",
    dsp_wait=None,
):
    """Process every admitted sample; deliver captures without feature jobs.

    Scheduled input is read by an independent spawn process into 128 finite
    slots. DSP and capture delivery have one owner, avoiding a second IQ/result
    serialization boundary. A sink runs on this owner and its cost is included.
    Real-time callers must bound their sink work and keep selected IQ bounded.
    """
    import numpy as np

    from signal_processing.core import StreamInfo
    from signal_processing.detection import POSITIVE_SNR_IID, POSITIVE_SNR_IID_1920
    from signal_processing.runtime import ProcessSourceRunner, ReplayFileFactory

    if sample_rate_hz not in (1_000_000.0, 1_920_000.0):
        raise ValueError("Adaptive replay supports 1 MS/s or 1.92 MS/s")
    if type(block_size) is not int or not 1 <= block_size <= 262144:
        raise ValueError("block_size must be an integer in [1, 262144]")
    if (
        type(queue_capacity) is not int
        or queue_capacity < 1
        or queue_capacity * (block_size * 8 + 8200) > 256 * 1024 * 1024
    ):
        raise ValueError("Acquisition pool must have positive capacity and use at most 256 MiB")
    if (
        isinstance(initial_noise_power, bool)
        or not math.isfinite(initial_noise_power)
        or initial_noise_power <= 0
    ):
        raise ValueError("Supply a finite positive measured quiet noise power")
    import sys
    if priority_class not in ("normal","above-normal","high"):
        raise ValueError("Unsupported process priority class")
    if dsp_scheduling not in ('priority','mmcss'):
        raise ValueError('DSP scheduling must be priority or mmcss')
    if dsp_wait is None:
        dsp_wait = 'spin' if realtime_scheduling else 'block'
    if dsp_wait not in ('spin','block'):
        raise ValueError('DSP wait must be spin or block')
    if file_backend not in ('buffered','mmap'):
        raise ValueError('File backend must be buffered or mmap')
    if priority_class != "normal" and sys.platform != "win32":
        raise ValueError("Explicit priority classes require Windows")
    placement = None
    original_priority = None
    if processing_cpus is not None or acquisition_cpus is not None or priority_class != "normal":
        import psutil

        placement = psutil.Process()
        allowed = placement.cpu_affinity()
        original_priority = placement.nice()
        for cpus in (processing_cpus, acquisition_cpus):
            if cpus is not None and (
                not cpus or any(type(cpu) is not int or cpu not in allowed for cpu in cpus)
            ):
                raise ValueError("CPU placement must be a nonempty subset of the current affinity")
    initialized = time.perf_counter()
    path = Path(path).resolve()
    stream = StreamInfo("adaptive-recording", str(path), sample_rate_hz)
    profile = POSITIVE_SNR_IID_1920 if sample_rate_hz == 1_920_000.0 else POSITIVE_SNR_IID
    pipe = profile.create_pipeline(stream, initial_noise_power, enable_feature_analysis=False)
    factory = ReplayFileFactory(str(path), stream, block_size, True, 0.01,read_backend=file_backend)
    runner = ProcessSourceRunner(
        factory, stream, capacity=queue_capacity, max_block_samples=block_size,performance_qos=realtime_scheduling,
        acquisition_cpus=acquisition_cpus
    )
    init_seconds = None
    latency = collections.deque(maxlen=65536)
    processing = collections.deque(maxlen=65536)
    maximum = 0.0
    captures = incomplete = 0
    last_release = eof_latency_ms = None
    errors = []
    from contextlib import ExitStack
    from ._performance import realtime_scope
    realtime_resources = ExitStack()
    realtime_policy = None

    def deliver(values):
        nonlocal captures, incomplete
        for value in values:
            captures += 1
            incomplete += bool(value.known_lost_samples or value.unknown_loss)
            if capture_sink is not None:
                capture_sink(value)

    started = time.perf_counter()
    try:
        runner.start(start_paused=True)
        if placement is not None:
            if processing_cpus is not None:
                placement.cpu_affinity(list(processing_cpus))
            if priority_class != "normal":
                priority = (psutil.HIGH_PRIORITY_CLASS if priority_class == "high"
                            else psutil.ABOVE_NORMAL_PRIORITY_CLASS)
                placement.nice(priority)
                psutil.Process(runner.pid).nice(priority)
        import gc
        gc.collect()
        init_seconds = time.perf_counter() - initialized
        realtime_policy = realtime_resources.enter_context(realtime_scope(
            realtime_scheduling and dsp_scheduling == 'mmcss'))
        started = time.perf_counter()
        runner.begin()
        while (lease := runner.get(timeout=5,spin=dsp_wait == 'spin')) is not None:
            last_release = lease.envelope.scheduled_monotonic_ns
            before = time.perf_counter()
            try:
                deliver(pipe.process_block(lease.envelope.block))
                processing.append((time.perf_counter() - before) * 1000)
                anchor = lease.envelope.scheduled_monotonic_ns
                if anchor is not None:
                    delay = max(0.0, (time.perf_counter_ns() - anchor) / 1e6)
                    latency.append(delay)
                    maximum = max(maximum, delay)
            except BaseException:
                runner.ack(lease, outcome="failed")
                raise
            runner.ack(lease)
        deliver(pipe.flush())
        if last_release is not None:
            eof_latency_ms = max(0.,(time.perf_counter_ns()-last_release)/1e6)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            runner.shutdown(drain=False, timeout=5)
        except Exception as exc:
            errors.append(f"cleanup: {type(exc).__name__}: {exc}")
        try:
            pipe.shutdown()
        finally:
            try:
                realtime_resources.close()
            except Exception as exc:
                errors.append(f'scheduling cleanup: {type(exc).__name__}: {exc}')
            finally:
                if placement is not None:
                    try:
                        placement.cpu_affinity(allowed)
                    finally:
                        placement.nice(original_priority)
    source, acquisition = runner.source_snapshot(), runner.snapshot()
    lossless = bool(
        source is not None
        and source.finished
        and not errors
        and acquisition.state == "CLOSED"
        and acquisition.accounting_balanced
        and acquisition.processed_samples == source.total_samples
        and not source.expired_samples
        and not acquisition.queue_dropped_samples
        and not acquisition.failed_samples
        and not acquisition.cancelled_samples
        and not acquisition.unaccepted_cancelled_samples
        and not acquisition.outstanding_deliveries
    )
    return dict(
        sample_rate_hz=sample_rate_hz,
        block_size=block_size,
        queue_capacity=queue_capacity,
        acquisition_pool_bytes=queue_capacity * (block_size * 8 + 8200),
        acquisition_buffer_seconds=queue_capacity * block_size / sample_rate_hz,
        scheduled=True,
        initialization_seconds=init_seconds,
        elapsed_seconds=time.perf_counter() - started,
        processing_cpus=processing_cpus,
        acquisition_cpus=acquisition_cpus,
        priority_class=priority_class,
        dsp_scheduling=dsp_scheduling,
        dsp_wait=dsp_wait,
        file_backend=file_backend,
        profile_id=profile.profile_id,
        captures=captures,
        captures_with_input_loss=incomplete,
        processed_samples=acquisition.processed_samples,
        total_samples=source.total_samples if source else None,
        acquisition_passed=lossless,
        scheduled_lossless_passed=lossless,
        latency_ms_percentiles=np.percentile(latency, [50, 95, 99]).tolist() if latency else [],
        maximum_latency_ms=maximum,
        eof_tail_latency_ms=eof_latency_ms,
        _last_release_ns=last_release,
        _acquisition_started=started,
        processing_ms_percentiles=np.percentile(processing, [50, 95, 99, 100]).tolist()
        if processing
        else [],
        latency_scope="Block availability through preprocessing, capture assembly and capture sink",
        features="disabled; every admitted IQ sample traverses detection/noise/segmentation/capture",
        runner=dataclasses.asdict(acquisition),
        source=dataclasses.asdict(source) if source else None,
        retained=pipe.retention_snapshot(),
        errors=errors,
        realtime_scheduling=realtime_policy,
        acquisition_scheduling=runner.source_execution(),
    )


def run(path, *, sample_rate_hz=1_920_000.0,block_size=16384,initial_noise_power,
        capture_sink=None,processing_cpus=None,acquisition_cpus=None,queue_capacity=128,
        performance_qos=True,priority_class="normal",dsp_scheduling="priority",file_backend="buffered",dsp_wait=None,dsp_smt='reserve'):
    """Acquire and preprocess with scoped process performance scheduling."""
    from ._performance import performance_scope
    from ._core_reservation import core_reservation
    if dsp_smt not in ('reserve','shared'):
        raise ValueError('DSP SMT policy must be reserve or shared')
    with performance_scope(performance_qos,prefer_fast_cpus=processing_cpus is None and acquisition_cpus is None) as scheduling:
        enabled = performance_qos and dsp_smt == 'reserve' and dsp_wait in (None,'spin')
        with core_reservation(scheduling['preferred_core_cpus'],enabled=enabled) as reservation:
            result = _run(path,sample_rate_hz=sample_rate_hz,block_size=block_size,
                initial_noise_power=initial_noise_power,capture_sink=capture_sink,
                processing_cpus=processing_cpus,acquisition_cpus=acquisition_cpus,
                queue_capacity=queue_capacity,priority_class=priority_class,realtime_scheduling=performance_qos,
                dsp_scheduling=dsp_scheduling,file_backend=file_backend,dsp_wait=dsp_wait)
    result['performance_qos'] = scheduling
    result['smt_reservation'] = reservation
    anchor = result.pop('_last_release_ns')
    result['elapsed_seconds'] = time.perf_counter()-result.pop('_acquisition_started')
    if anchor is not None:
        result['eof_tail_latency_ms'] = max(0.,(time.perf_counter_ns()-anchor)/1e6)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Little-endian complex64 IQ")
    parser.add_argument(
        "--sample-rate", type=float, choices=(1_000_000.0, 1_920_000.0), default=1_920_000.0
    )
    parser.add_argument("--block-size", type=int, default=16384)
    parser.add_argument(
        "--queue-capacity",
        type=int,
        default=128,
        help="Finite input slots; default covers 1.09 s at 1.92 MS/s (about 17 MiB)",
    )
    parser.add_argument(
        "--initial-noise-power",
        type=float,
        required=True,
        help="Measured quiet IQ mean power, linear",
    )
    parser.add_argument("--enforce-acquisition", action="store_true")
    parser.add_argument(
        "--dsp-cpus",
        type=int,
        nargs="+",
        help="Optional explicit processing CPU placement (requires psutil)",
    )
    parser.add_argument(
        "--source-cpus",
        type=int,
        nargs="+",
        help="Optional independent acquisition CPU placement (requires psutil)",
    )
    parser.add_argument("--cpu-qos",choices=("performance","system"),default="performance")
    parser.add_argument("--priority-class",choices=("normal","above-normal","high"),default=None)
    parser.add_argument('--dsp-scheduling',choices=('priority','mmcss'),default='priority')
    parser.add_argument('--file-backend',choices=('buffered','mmap'),default='buffered')
    parser.add_argument('--dsp-wait',choices=('spin','block'),default=None,
                        help='Performance mode defaults to spin on the reserved DSP core; block reduces idle CPU use')
    parser.add_argument('--dsp-smt',choices=('reserve','shared'),default='reserve',
                        help='Spin mode reserves the selected core\'s SMT sibling; shared uses less CPU')
    args = parser.parse_args(argv)
    if args.priority_class is None:
        import sys
        args.priority_class = "high" if args.enforce_acquisition and sys.platform == "win32" else "normal"
    result = run(
        args.input,
        sample_rate_hz=args.sample_rate,
        block_size=args.block_size,
        initial_noise_power=args.initial_noise_power,
        processing_cpus=args.dsp_cpus,
        acquisition_cpus=args.source_cpus,
        queue_capacity=args.queue_capacity,
        performance_qos=args.cpu_qos == "performance",
        priority_class=args.priority_class,
        dsp_scheduling=args.dsp_scheduling,
        file_backend=args.file_backend,
        dsp_wait=args.dsp_wait,
        dsp_smt=args.dsp_smt,
    )
    print(json.dumps(result, allow_nan=False))
    return int(
        bool(result["errors"]) or args.enforce_acquisition and not result["acquisition_passed"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
