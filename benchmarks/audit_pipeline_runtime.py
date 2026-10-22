"""Read-only production qualification harness: no production fixes or fake results.

Run timing commands sequentially, without pytest/tracing concurrently. Generates
bounded-memory synthetic encoded LTE files offline; truth is for scoring only.
LTE and generic capture/features remain separate production paths.
"""
import argparse
import collections
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import platform
import time

# Keep spawn bootstrap acquisition-only. Heavy DSP/evaluator imports belong in
# their parent execution functions, not in every file-reader interpreter.


def environment():
    import numpy as np
    import scipy
    import os
    import psutil
    return dict(platform=platform.platform(), python=platform.python_version(),
                process_cpu_affinity=psutil.Process().cpu_affinity(),
                numpy=np.__version__, scipy=scipy.__version__,
                memory_available_bytes=psutil.virtual_memory().available,
                openblas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
                logical_cpus=psutil.cpu_count(), physical_cpus=psutil.cpu_count(logical=False))


def prepare(path, seconds, snr=-5., *, sample_rate_hz=2_000_000., max_seconds=30.):
    from functools import lru_cache
    import numpy as np
    from scipy.signal import resample_poly
    from tests.lte_pbch_fixture import slot_grids
    total = round(seconds*sample_rate_hz)
    if sample_rate_hz not in (1_920_000., 2_000_000.):
        raise ValueError('Fixture rate must be genuine native 1.92 or resampled 2 MS/s')
    if not .04 <= seconds <= max_seconds <= 600 or path.exists():
        raise ValueError(f"Use a new file and a duration in [.04,{max_seconds}]")
    reference = Path(__file__).resolve().parents[1]/"tests/fixtures/lte_sync_sequences.npz"
    with np.load(reference, allow_pickle=False) as ref:
        pss, sss = ref["pss"][42 % 3], ref["sss"][42]
    data_rng, noise_rng = np.random.default_rng(20261019), np.random.default_rng(1151)
    carriers = np.r_[np.arange(-36, 0), np.arange(1, 37)] % 128
    sync = np.r_[np.arange(-31, 0), np.arange(1, 32)] % 128

    @lru_cache(maxsize=1024)
    def broadcast_grid(phase):
        # Test fixture SFN is modulo 256 in its eight encoded bits, with four
        # quarter phases. Cache only these bounded OFFLINE transmit constants;
        # receiver code never sees this table. Random data/noise are not cached.
        _, grid = slot_grids(42, 2, index=4, quarter=phase % 4, frame=phase)
        grid.flags.writeable = False
        return grid

    def native(frame):
        grid = np.zeros((140, 128), complex)
        grid[:, carriers] = ((2*data_rng.integers(0, 2, (140, 72))-1)
                             +1j*(2*data_rng.integers(0, 2, (140, 72))-1))/np.sqrt(2)
        pilots = broadcast_grid(frame % 1024)
        grid[7:14] = 0
        grid[7:14, carriers] = pilots[0]+(.6+.3j)*pilots[1]
        for symbol in (5, 6, 75, 76):
            grid[symbol] = 0
            grid[symbol, sync] = pss if symbol in (6, 76) else sss[int(symbol == 75)]
        useful = np.fft.ifft(grid, axis=1)*np.sqrt(128)
        return np.concatenate([np.r_[row[-(10 if l % 7 == 0 else 9):], row]
                               for l, row in enumerate(useful)])

    frame_length = round(sample_rate_hz/100)
    frames = math.ceil(max(0, total-431)/frame_length)
    # Independent standard SciPy zero-phase resampling with 48 input samples
    # of real overlap on either side. Frame length 19200 and overlap 48 are
    # divisible by down=24; central output is equivalent to full-file resampling.
    previous = np.zeros(48, complex)
    current, following = native(0), native(1)
    emitted = 0
    # Mean ensemble power over the declared central-six-RB OFDM resource map.
    # Noise is generated only offline; the receiver never gets this value.
    sigma = math.sqrt((9744.8/(140*128))/10**(snr/10)/2)
    started = time.perf_counter()
    with path.open("wb") as output:
        prefix = sigma*(noise_rng.normal(size=431)+1j*noise_rng.normal(size=431))
        prefix.astype("<c8").tofile(output)
        emitted += len(prefix)
        for frame in range(frames):
            padded = np.r_[previous, current, following[:48]]
            signal = current if sample_rate_hz == 1_920_000. else resample_poly(padded, 25, 24)[50:20050]
            n = min(frame_length, total-emitted)
            signal = signal[:n]*np.exp(2j*np.pi*12345*(emitted+np.arange(n))/sample_rate_hz)
            signal += sigma*(noise_rng.normal(size=n)+1j*noise_rng.normal(size=n))
            signal.astype("<c8").tofile(output)
            emitted += n
            previous, current = current[-48:].copy(), following
            following = native(frame+2)
    assert path.stat().st_size == total*8
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            digest.update(chunk)
    return dict(scope="offline bounded-memory encoded LTE test fixture", samples=total,
                sample_rate_hz=sample_rate_hz, onset=431, pci=42, ports=2, index=4, cfo=12345,
                snr_db=snr, nominal_input_noise_power=2*sigma*sigma,
                sha256=digest.hexdigest(), preparation_seconds=time.perf_counter()-started,
                source="frozen external coding/sequences + independent test resource mapping",
                environment=environment())


def lte(path, profile, *, scheduled=True, bch_backend='auto', numba_threads=1, sample_rate_hz=2_000_000., scoring_backend='precomputed', fft_workers=1, sync_backend='numpy', decode_cadence='every_frame', source_backend='thread', pin_primary_host=False, progress_seconds=0, priority_class="normal",dsp_wait='block',core_owner=None):
    import numpy as np
    from signal_processing.core import StreamInfo
    from signal_processing.lte import LTEBroadcastReceiver,LTESyncConfig,LTEBroadcastBatch
    from signal_processing.runtime import SourceRunner
    from signal_processing.sources import RecordedFileIQSource,ReplayIQSource,SourceMetadata
    from tests.lte_pbch_fixture import frame_fixture
    if scoring_backend not in ('inline', 'precomputed'):
        raise ValueError('Scoring backend must be inline or precomputed')
    if dsp_wait not in ('spin','block') or (dsp_wait == 'spin' and source_backend != 'process'):
        raise ValueError('Spin wait requires the isolated source backend')
    if (pin_primary_host or priority_class != "normal") and source_backend != 'process':
        raise ValueError('Host placement experiment requires the isolated source backend')
    initialization_started = time.perf_counter()
    stream = StreamInfo("runtime-audit", "encoded-synthetic-file", sample_rate_hz, 2.14e9)
    config = LTESyncConfig.low_snr(sample_rate_hz) if profile == "low_snr" else LTESyncConfig(input_sample_rate_hz=sample_rate_hz)
    receiver = LTEBroadcastReceiver(stream, config, bch_backend=bch_backend, numba_threads=numba_threads, fft_workers=fft_workers,
                                    measure_timing=True,sync_backend=sync_backend,decode_cadence=decode_cadence)
    replay = ReplayIQSource(RecordedFileIQSource(path, stream, 16384),
        scheduled=scheduled, lateness_budget_s=.01, metadata=SourceMetadata(), owns_source=True)
    runner = SourceRunner(replay, capacity=16, max_block_samples=16384, owns_source=True)
    if source_backend == 'process':
        from signal_processing.runtime.process_source_runner import ProcessSourceRunner, ReplayFileFactory
        runner = ProcessSourceRunner(ReplayFileFactory(str(path.resolve()),stream,scheduled=scheduled),stream,
                                     capacity=16,max_block_samples=16384,performance_qos=priority_class == 'high',
                                     acquisition_cpus=[2,3] if pin_primary_host else None)
    elif source_backend != 'thread':
        raise ValueError('source_backend must be thread or process')
    lag, processing = collections.deque(maxlen=4096), collections.deque(maxlen=4096)
    maximum = 0.
    results = collections.Counter()
    frame_length = round(sample_rate_hz/100)
    frames = math.ceil(max(0, replay.total_samples-431)/frame_length)
    seen = np.zeros(frames, bool)  # finite test extent, no retained event IQ
    before_validation = time.perf_counter()
    # Offline truth belongs to the evaluator, NEVER to the receiver. Construct
    # all finite expected payloads before the independently advancing source
    # starts, rather than reopening encoder fixtures on its consumer thread.
    # Every returned MIB is still checked, with identical frame/CRC/port/timing
    # assertions. Initialization and this cost remain reported separately.
    expected_payloads = (tuple(tuple(map(int, frame_fixture(4, 2, f)[0])) for f in range(0, frames, 4))
                         if scoring_backend == 'precomputed' else None)
    validation_setup_seconds = time.perf_counter()-before_validation
    initialization = time.perf_counter()-initialization_started
    failures = []
    sync_timing_errors = collections.deque(maxlen=4096)
    sync_frequency_errors = collections.deque(maxlen=4096)
    scoring_seconds = 0.
    receiver_processing = collections.deque(maxlen=4096)
    first_blocks = []
    last_input_anchor=None
    flush_seconds=eof_tail_latency_ms=None
    started = time.perf_counter()
    progress_step = round(progress_seconds*sample_rate_hz)
    next_progress = progress_step

    def observe(batch):
        results["sync_observations"] += len(batch.synchronization)
        results["confirmed_observations"] += sum(o.state == "SYNCHRONIZED" for o in batch.synchronization)
        results["wrong_confirmed_pci"] += sum(o.pci != 42 for o in batch.synchronization if o.state == "SYNCHRONIZED")
        for observation in batch.synchronization:
            if observation.state == 'SYNCHRONIZED':
                delta = float(observation.input_frame_start)-431
                error = abs(delta-round(delta/frame_length)*frame_length)
                sync_timing_errors.append(error)
                sync_frequency_errors.append(abs(observation.frequency_offset_hz-12345.))
                results['sync_outside_timing_tolerance'] += error > 2
        for mib in batch.mibs:
            results["mibs"] += 1
            frame = round(float(mib.input_frame_start-431)/frame_length)
            error = abs(float(mib.input_frame_start-431-frame*frame_length))
            if not 0 <= frame < frames:
                results["foreign_frame_results"] += 1
                continue
            expected = (expected_payloads[frame//4] if expected_payloads is not None
                        else tuple(map(int, frame_fixture(4, 2, frame)[0])))
            valid = (mib.pci == 42 and mib.antenna_ports == 2 and mib.frame_quarter == frame % 4
                     and mib.mib.payload_bits == expected and error <= 2)
            if not valid:
                results["wrong_mibs_or_timing"] += 1
            elif seen[frame]:
                results["duplicate_mibs"] += 1
            else:
                seen[frame] = True
                results["verified_mibs"] += 1

    source_start = time.perf_counter()
    import gc
    gc.collect()
    original_affinity = original_priority = None
    if pin_primary_host or priority_class != "normal":
        import psutil
        parent=psutil.Process()
        original_affinity,original_priority=parent.cpu_affinity(),parent.nice()
        if pin_primary_host and not set(range(12)).issubset(original_affinity):
            raise ValueError('This placement requires the declared 12-logical-CPU Windows target')
        runner.start(start_paused=True)
        # Owned processes only; no host power-plan/timer/global priority change.
        producer=psutil.Process(runner.pid)
        if pin_primary_host:
            parent.cpu_affinity([0,1])
            producer.cpu_affinity([2,3])
        priority = (psutil.HIGH_PRIORITY_CLASS if priority_class == "high"
                    else psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        parent.nice(priority)
        producer.nice(priority)
        runner.begin()
    else:
        runner.start()
    source_startup_seconds = time.perf_counter()-source_start
    try:
        wait_options = {'spin':dsp_wait == 'spin'} if source_backend == 'process' else {}
        while (lease := runner.get(timeout=5,**wait_options)) is not None:
            last_input_anchor=lease.envelope.scheduled_monotonic_ns
            try:
                begin = time.perf_counter()
                batch = receiver.process_envelope(lease.envelope)
                score_start = time.perf_counter()
                receiver_processing.append((score_start-begin)*1000)
                observe(batch)
                scoring_seconds += time.perf_counter()-score_start
                processing.append((time.perf_counter()-begin)*1000)
                anchor = lease.envelope.scheduled_monotonic_ns
                delay = max(0., (time.perf_counter_ns()-anchor)/1e6) if anchor is not None else 0.
                maximum = max(maximum, delay)
                lag.append(delay)
                if len(first_blocks) < 128:
                    first_blocks.append((lease.envelope.block.sequence_number, delay, processing[-1]))
            except Exception:
                runner.ack(lease, outcome="failed")
                raise
            runner.ack(lease)
            if progress_step and lease.envelope.block.end_sample >= next_progress:
                state = runner.snapshot()
                print(json.dumps(dict(lte_processed_samples=state.processed_samples,
                    queue_dropped_samples=state.queue_dropped_samples,
                    verified_mibs=results['verified_mibs'])),flush=True)
                next_progress += progress_step
        flush_started=time.perf_counter()
        observe(receiver.flush())
        flush_seconds=time.perf_counter()-flush_started
        if last_input_anchor is not None:
            eof_tail_latency_ms=max(0.,(time.perf_counter_ns()-last_input_anchor)/1e6)
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        observe(LTEBroadcastBatch(getattr(exc, "completed_observations", ()), getattr(exc, "completed_mibs", ())))
    finally:
        try:
            runner.shutdown(drain=False, timeout=5)
        except Exception as exc:
            failures.append(f"cleanup: {type(exc).__name__}: {exc}")
        try:
            replay.close()
        except Exception as exc:
            failures.append(f"source cleanup: {type(exc).__name__}: {exc}")
        if original_affinity is not None:
            parent.cpu_affinity(original_affinity);parent.nice(original_priority)
    if core_owner is not None:
        core_owner.close()
    if last_input_anchor is not None:
        eof_tail_latency_ms=max(0.,(time.perf_counter_ns()-last_input_anchor)/1e6)
    elapsed = time.perf_counter()-started
    r = runner.snapshot()
    s = runner.source_snapshot() if source_backend == 'process' else replay.snapshot()
    replay.close()
    balanced = (r.accounting_balanced and s.total_samples == s.read_samples+s.unread_samples
        and s.read_samples == s.returned_samples+s.expired_samples+s.pending_samples
        and s.returned_samples == r.read_samples)
    acquisition = (balanced and r.processed_samples == s.total_samples and not s.expired_samples
        and not r.queue_dropped_samples and not r.failed_samples and not r.cancelled_samples
        and not r.unaccepted_cancelled_samples and not r.outstanding_deliveries and not failures)
    lag_passed = bool(lag) and float(np.percentile(lag, 99)) <= 50 and maximum <= 100
    protocol = (results["verified_mibs"] == frames and not any(results[k] for k in
                ("wrong_confirmed_pci", "wrong_mibs_or_timing", "foreign_frame_results", "duplicate_mibs")))
    correct_results = results["verified_mibs"] > 0 and not any(results[k] for k in
                ("wrong_confirmed_pci", "wrong_mibs_or_timing", "foreign_frame_results", "duplicate_mibs"))
    blocks = math.ceil(frames/4)
    verified_blocks = sum(bool(np.any(seen[start:start+4])) for start in range(0, frames, 4))
    processing_eligible = s.total_samples-s.expired_samples
    processing_acquisition = (balanced and r.processed_samples == processing_eligible and processing_eligible > 0
        and not r.queue_dropped_samples and not r.failed_samples and not r.cancelled_samples
        and not r.unaccepted_cancelled_samples and not r.outstanding_deliveries and not failures)
    # One successful MIB per 40-ms BCH block is reported separately from the
    # stricter legacy every-10-ms-frame gate; neither erases failed frame CRCs.
    decoding_blocks = correct_results and verified_blocks == blocks
    sync_correct = results['confirmed_observations'] > 0 and not results['wrong_confirmed_pci'] and not results['sync_outside_timing_tolerance']
    return dict(scope="file source/runner + LTE synchronization/PBCH/MIB; no generic features",
        first_blocks_sequence_latency_processing_ms=first_blocks,
        profile=profile, scheduled=scheduled, sample_rate_hz=sample_rate_hz, block_size=16384,
        source_backend=source_backend,
        pin_primary_host=pin_primary_host,
        priority_class=priority_class,
        acquisition_scheduling=runner.source_execution() if source_backend == 'process' else None,
        source_startup_seconds=source_startup_seconds,
        sub10ms_lossless_passed=bool(scheduled and acquisition and maximum < 10
            and eof_tail_latency_ms is not None and eof_tail_latency_ms < 10
            and sync_correct and decoding_blocks and r.state=='CLOSED'),
        flush_seconds=flush_seconds,eof_tail_latency_ms=eof_tail_latency_ms,
        last_input_scheduled_monotonic_ns=last_input_anchor,
        execution=receiver.execution, initialization_seconds=initialization,
        scoring_backend=scoring_backend, validation_setup_seconds=validation_setup_seconds,
        scoring_seconds=scoring_seconds,
        receiver_processing_ms_percentiles=np.percentile(receiver_processing, [50,95,99,100]).tolist() if receiver_processing else [],
        stage_timing=receiver.stage_timing,
        synchronization_correct=bool(sync_correct),
        sync_timing_error_samples_percentiles=np.percentile(sync_timing_errors,[50,95,99,100]).tolist() if sync_timing_errors else [],
        sync_frequency_error_hz_percentiles=np.percentile(sync_frequency_errors,[50,95,99,100]).tolist() if sync_frequency_errors else [],
        queue_capacity=16, lateness_budget_s=.01, dsp_wait=dsp_wait, environment=environment(),
        elapsed_seconds=elapsed, consumed_samples=r.processed_samples, total_samples=s.total_samples,
        acquisition_fraction=r.processed_samples/s.total_samples,
        sample_accounting_balanced=balanced, acquisition_passed=acquisition,
        latency_ms_percentiles=np.percentile(lag, [50, 95, 99]).tolist() if lag else [],
        max_latency_ms=maximum, latency_scope="percentiles last <=4096 blocks; max whole run",
        processing_ms_percentiles=np.percentile(processing, [50, 95, 99, 100]).tolist() if processing else [],
        latency_gate_passed=lag_passed, latency_limits_ms=dict(p99=50, maximum=100),
        results=dict(results), expected_pbch_frames=frames, missing_verified_mibs=frames-int(seen.sum()),
        expected_bch_blocks=blocks, verified_bch_blocks=verified_blocks, missing_bch_blocks=blocks-verified_blocks,
        protocol_results_correct=correct_results, protocol_all_frames_passed=protocol,
        decoding_bch_blocks_passed=bool(decoding_blocks),
        processing_eligible_samples=processing_eligible,
        processing_acquisition_passed=bool(processing_acquisition),
        realtime_decoding_passed=bool(scheduled and processing_acquisition and lag_passed and sync_correct and decoding_blocks and r.state=='CLOSED'),
        realtime_decoding_scope='Excludes ONLY synthetic source expiry. Requires zero acquisition queue loss, p99<=50/max<=100 ms including synchronization AND PBCH/MIB, correct sync timing/PCI, at least one verified MIB in every 40-ms BCH block, and cleanup. Legacy every-frame gate remains unchanged.',
        acceptance_scope="strict lossless samples/latency AND every PBCH frame; frame CRC failures are distinct from queue/job loss",
        runner=dataclasses.asdict(r), replay=dataclasses.asdict(s),
        receiver=receiver.snapshot(), errors=failures,
        passed=acquisition and (lag_passed or not scheduled) and protocol and r.state == "CLOSED")


def failing_sink(_):
    raise RuntimeError("Injected required sink failure")


def failing_analysis(_):
    raise RuntimeError("Injected real analysis failure")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "lte", "generic", "failure", "pause", "recovery"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--profile", choices=("compatibility", "low_snr"), default="low_snr")
    parser.add_argument("--unpaced", action="store_true")
    parser.add_argument("--bch-backend", choices=('auto', 'reference', 'numba'), default='auto')
    parser.add_argument("--numba-threads", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument('--sample-rate', type=float, choices=(1_920_000., 2_000_000.), default=2_000_000.)
    parser.add_argument('--scoring-backend', choices=('precomputed', 'inline'), default='precomputed')
    parser.add_argument('--fft-workers', type=int, choices=(1,2,3,4), default=1)
    parser.add_argument('--lte-acceptance-scope',choices=('legacy','realtime_decoding','sub10ms'),default='legacy')
    parser.add_argument('--sync-backend',choices=('auto','numpy','numba'),default='numpy')
    parser.add_argument('--decode-cadence',choices=('every_frame','bch_block'),default='every_frame')
    parser.add_argument('--source-backend',choices=('thread','process'),default='thread')
    parser.add_argument('--pin-primary-host',action='store_true')
    parser.add_argument("--failure-kind", choices=("sink", "analysis"), default="sink")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior observations")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        result = prepare(args.input, args.seconds, sample_rate_hz=args.sample_rate)
    elif args.mode == "lte":
        result = lte(args.input, args.profile, scheduled=not args.unpaced,
                     bch_backend=args.bch_backend, numba_threads=args.numba_threads, sample_rate_hz=args.sample_rate,
                     scoring_backend=args.scoring_backend, fft_workers=args.fft_workers,sync_backend=args.sync_backend,decode_cadence=args.decode_cadence,source_backend=args.source_backend,pin_primary_host=args.pin_primary_host)
        result['selected_acceptance_scope']=args.lte_acceptance_scope
    else:
        from benchmarks.sustained_evaluation import bounded_acquisition
        kw = dict(sample_rate_hz=2e6, block_size=16384, queue_capacity=16,
                  features=True, analysis_backend="process_pool", producer_backend="process",
                  fixed_quiet_batch=True)
        if args.mode in ("failure", "recovery"):
            kw.update(sample_rate_hz=10000., block_size=1024, analysis_backend="thread",
                      producer_backend="thread")
        if args.mode == "failure":
            kw["result_sink" if args.failure_kind == "sink" else "analysis_function"] = (
                failing_sink if args.failure_kind == "sink" else failing_analysis)
        if args.mode == "recovery":
            kw["slow_sink_seconds"] = .5
        if args.mode == "pause":
            kw["consumer_pause_seconds"] = .5
        result = bounded_acquisition(args.seconds, **kw)
        result["environment"] = environment()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf8")
    print(json.dumps(result, indent=2, allow_nan=False))
    acceptance = (result.get('sub10ms_lossless_passed') if args.mode=='lte' and args.lte_acceptance_scope=='sub10ms' else
                  result.get('realtime_decoding_passed') if args.mode=='lte' and args.lte_acceptance_scope=='realtime_decoding'
                  else result.get('passed'))
    return int(acceptance is False)


if __name__ == "__main__":
    raise SystemExit(main())
