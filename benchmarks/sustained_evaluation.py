"""Bounded acquisition, recovery and controlled detection evaluation.

Wall-clock tests measure this host, not an SDR hardware guarantee. Memory tracing
is optional because it changes scheduling. Statistical bounds are conditional on
the explicitly recorded independent-trial / Poisson assumptions.
"""
from __future__ import annotations

import math
import collections
import queue
import threading
import time
import tracemalloc

import numpy as np
from scipy import signal
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta, chi2

from signal_processing.core import IQBlock, StreamInfo
from signal_processing.detection import GaussianNoiseEstimator
from signal_processing.runtime import StreamingCapturePipeline


def match_intervals(truth, detected, minimum_iou=0.2):
    """Maximum-cardinality one-to-one matching, then maximize total IoU."""
    if not truth or not detected:
        return [], list(range(len(truth))), list(range(len(detected)))
    scores = np.zeros((len(truth), len(detected)))
    for i, (a, b) in enumerate(truth):
        for j, (c, d) in enumerate(detected):
            intersection = max(0, min(b, d) - max(a, c))
            scores[i, j] = intersection / (b - a + d - c - intersection)
    eligible = scores >= minimum_iou
    cost = -(eligible * (min(scores.shape) + 1) + scores * eligible)
    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if eligible[i, j]]
    return (pairs, [i for i in range(len(truth)) if i not in {p[0] for p in pairs}],
            [j for j in range(len(detected)) if j not in {p[1] for p in pairs}])


def _pipeline(stream, *, features=False, asynchronous=False, queue_size=4, analysis_function=None):
    return StreamingCapturePipeline(
        stream, ring_capacity=4096, window_size=64, hop_size=32,
        noise_estimator=GaussianNoiseEstimator(window_size=64, calibrated_noise_power_linear=1e-4),
        start_persistence=2, end_persistence=2, pre_trigger_samples=0,
        pre_signal_margin_samples=0, post_signal_samples=0, min_post_trigger_samples=0,
        max_event_samples=4096, enable_feature_analysis=features,
        synchronous_analysis=not asynchronous, max_worker_queue_size=queue_size,
        analysis_function=analysis_function,
        result_delivery='callbacks' if features and asynchronous else 'polling',
    )


def bounded_acquisition(duration_seconds=5.0, *, sample_rate_hz=1e6, block_size=1024,
                        queue_capacity=16, features=False, slow_sink_seconds=0.0,
                        consumer_pause_seconds=0.0, trace_memory=False,
                        result_sink=None, configure_pipeline=None, analysis_function=None):
    """Independent arrival thread with drop-newest overload and truthful loss counts.

    A callback admitted before reset may complete after reset. The slow sink runs
    in the feature worker; acquisition never waits for it. Recovery drains the
    bounded acquisition queue after production and reports trailing losses too.
    """
    if duration_seconds <= 0 or sample_rate_hz <= 0 or block_size < 1 or queue_capacity < 1:
        raise ValueError("Duration, sample rate, block size and queue capacity must be positive")
    stream = StreamInfo("bounded-acquisition", "scheduled-synthetic", sample_rate_hz)
    pipe = _pipeline(stream, features=features, asynchronous=features, analysis_function=analysis_function)
    # Callback delivery is mandatory when features=True; polling is explicitly
    # optional. Do not retain recording-sized IQ or a growing event ledger.
    ledger_lock = threading.Lock()
    pending = {}
    recent = collections.deque(maxlen=4096)
    recent_set = set()
    delivery = dict(terminal_results=0, duplicate_results=0, failed_results=0,
                    overloaded_results=0, ledger_overflow=0, raw_captures=0,
                    accepted_callback_results=0, immediate_unaccepted_results=0)
    def correlate(sc, sign):
        key = (sc.stream_epoch, sc.event_id)
        with ledger_lock:
            if sign == -1:
                delivery['terminal_results'] += 1
                if sc.feature_status == 'dropped_overload':
                    delivery['immediate_unaccepted_results'] += 1
                else:
                    delivery['accepted_callback_results'] += 1
                delivery['failed_results'] += sc.feature_status == 'failed'
                delivery['overloaded_results'] += sc.feature_status == 'dropped_overload'
                if key in recent_set:
                    delivery['duplicate_results'] += 1
                if len(recent) == recent.maxlen:
                    recent_set.discard(recent[0])
                recent.append(key); recent_set.add(key)
            else:
                delivery['raw_captures'] += 1
            if key not in pending and len(pending) >= 4096:
                delivery['ledger_overflow'] += 1
                return
            delta = pending.get(key, 0) + sign
            if delta:
                pending[key] = delta
            else:
                pending.pop(key, None)
    def observe_raw(sc):
        correlate(sc, 1)
        # Unaccepted jobs have one terminal outcome returned to acquisition,
        # not a callback on that thread and not a reserved polling result.
        if sc.feature_status == 'dropped_overload':
            correlate(sc, -1)
    if features:
        pipe.worker.add_result_sink(lambda sc: correlate(sc, -1))
    if result_sink is not None:
        pipe.worker.add_result_sink(result_sink)
    if configure_pipeline is not None:
        configure_pipeline(pipe)
    if slow_sink_seconds:
        pipe.worker.add_result_sink(lambda _: time.sleep(slow_sink_seconds))
    arrivals = queue.Queue(maxsize=queue_capacity)
    done, stop = threading.Event(), threading.Event()
    stats = dict(produced_samples=0, admitted_samples=0, consumed_samples=0, dropped_samples=0,
                 producer_deadline_dropped_samples=0, queue_dropped_samples=0,
                 trailing_dropped_samples=0, acquisition_queue_peak=0, worker_queue_peak=0,
                 blocks_processed=0, deadline_misses=0, max_backlog_ms=0.0,
                 max_producer_lateness_ms=0.0, emitted_events=0, consumer_failed_samples=0,
                 consumer_processing_cpu_seconds=0.0, producer_preparation_seconds=0.0,
                 producer_wait_seconds=0.0, producer_queue_seconds=0.0, errors=[])
    latencies = collections.deque(maxlen=65536)
    processing_times = collections.deque(maxlen=65536)
    memory = []
    total = int(duration_seconds * sample_rate_hz)
    interval = block_size / sample_rate_hz
    started = time.perf_counter()
    if trace_memory:
        tracemalloc.start()

    def produce():
        cpu_started = time.thread_time()
        rng = np.random.default_rng(730)
        lost = 0
        try:
            for seq, start in enumerate(range(0, total, block_size)):
                if stop.is_set():
                    break
                n = min(block_size, total - start)
                arrival = started + (start + n) / sample_rate_hz
                # A late software producer cannot manufacture timely hardware
                # arrivals retrospectively. Expired ticks are explicit losses.
                if time.perf_counter() > arrival + interval:
                    stats["produced_samples"] += n
                    stats["dropped_samples"] += n
                    stats["producer_deadline_dropped_samples"] += n
                    lost += n
                    continue
                # Allocate only one transport block. No recording-sized array.
                preparation_start = time.perf_counter()
                iq = (rng.normal(size=n) + 1j * rng.normal(size=n)) * math.sqrt(1e-4 / 2)
                indices = start + np.arange(n)
                iq[(indices % 4096) < 256] += 0.1
                stats['producer_preparation_seconds'] += time.perf_counter()-preparation_start
                wait_start = time.perf_counter()
                while True:
                    remaining = arrival - time.perf_counter()
                    if remaining <= 0 or stop.wait(remaining):
                        break
                if stop.is_set():
                    break
                now = time.perf_counter()  # Recheck after every sleep, including overshoot.
                stats['producer_wait_seconds'] += now-wait_start
                stats["max_producer_lateness_ms"] = max(stats["max_producer_lateness_ms"], 1000 * max(0, now - arrival))
                stats["produced_samples"] += n
                if now > arrival + interval:
                    lost += n
                    stats["dropped_samples"] += n
                    stats["producer_deadline_dropped_samples"] += n
                    continue
                queue_start = time.perf_counter()
                block = IQBlock(stream, seq, start, None, iq.astype(np.complex64), dropped_samples_before=lost)
                try:
                    arrivals.put_nowait((arrival, block))
                    stats['admitted_samples'] += n
                    stats["acquisition_queue_peak"] = max(stats["acquisition_queue_peak"], arrivals.qsize())
                    lost = 0
                except queue.Full:
                    lost += n
                    stats["dropped_samples"] += n
                    stats["queue_dropped_samples"] += n
                stats['producer_queue_seconds'] += time.perf_counter()-queue_start
        except Exception as exc:
            stats["errors"].append(f"producer: {type(exc).__name__}: {exc}")
        finally:
            stats['producer_cpu_seconds'] = time.thread_time()-cpu_started
            stats["trailing_dropped_samples"] = lost
            stats["producer_finished_seconds"] = time.perf_counter() - started
            done.set()

    producer = threading.Thread(target=produce, name="bounded-IQ-producer", daemon=True)
    producer.start()
    paused = False
    try:
        while not done.is_set() or not arrivals.empty():
            try:
                arrival, block = arrivals.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                if consumer_pause_seconds and not paused and block.start_sample >= total // 3:
                    time.sleep(consumer_pause_seconds)
                    paused = True
                t0 = time.perf_counter()
                cpu0 = time.thread_time()
                try:
                    events = pipe.process_block(block)
                except Exception as exc:
                    committed = max(0, min(block.num_samples, pipe.consumed_end_sample-block.start_sample))
                    stats['consumed_samples'] += committed
                    stats['consumer_failed_samples'] += block.num_samples-committed
                    for sc in getattr(exc, 'completed_captures', ()):
                        if features:
                            observe_raw(sc)
                    raise
                processing_times.append(1000*(time.perf_counter()-t0))
                stats['consumer_processing_cpu_seconds'] += time.thread_time()-cpu0
                for sc in events:
                    if features:
                        observe_raw(sc)
                elapsed = time.perf_counter() - arrival
                latencies.append(1000*elapsed)
                stats["consumed_samples"] += block.num_samples
                stats["blocks_processed"] += 1
                stats["emitted_events"] += len(events)
                stats["deadline_misses"] += int(elapsed > interval)
                stats["max_backlog_ms"] = max(stats["max_backlog_ms"], 1000 * elapsed)
                stats["worker_queue_peak"] = max(stats["worker_queue_peak"], pipe.worker._queue.qsize())
                if trace_memory and (not memory or block.start_sample >= memory[-1][0] + sample_rate_hz / 2):
                    # Keep a bounded diagnostic history, too.
                    memory.append((block.start_sample, *tracemalloc.get_traced_memory()))
                    if len(memory) > 128:
                        memory.pop(0)
            finally:
                arrivals.task_done()
        tail = pipe.flush()
        stats["emitted_events"] += len(tail)
        for sc in tail:
            if features:
                observe_raw(sc)
    except Exception as exc:
        stats["errors"].append(f"consumer: {type(exc).__name__}: {exc}")
        stop.set()
    finally:
        stop.set()
        producer.join(timeout=2)
        undelivered = 0
        while True:
            try:
                _, item = arrivals.get_nowait()
                undelivered += item.num_samples
                arrivals.task_done()
            except queue.Empty:
                break
        try:
            pipe.shutdown(timeout=max(5, 6 * slow_sink_seconds))
        except Exception as exc:
            stats["errors"].append(f"shutdown: {type(exc).__name__}: {exc}")
        if trace_memory:
            tracemalloc.stop()
    stats.update(scheduled_samples=total, sample_rate_hz=sample_rate_hz, block_size=block_size,
                 duration_seconds=duration_seconds, elapsed_seconds=time.perf_counter() - started,
                 unproduced_samples=total-stats['produced_samples'], undelivered_admitted_samples=undelivered,
                 latency_ms_percentiles=np.percentile(latencies, [50, 95, 99, 100]).tolist() if latencies else [],
                 processing_ms_percentiles=np.percentile(processing_times, [50, 95, 99, 100]).tolist() if processing_times else [],
                 queue_capacity=queue_capacity, acquisition_queue_remaining=arrivals.qsize(),
                 acquisition_unfinished=arrivals.unfinished_tasks,
                 producer_alive=producer.is_alive(), overload_drops=pipe.worker.dropped_overload_count,
                 result_overload_drops=pipe.worker.dropped_result_overload_count,
                 memory_samples=memory, tracing_enabled=trace_memory,
                 retained_noise_samples=len(pipe._pending_noise_buf),
                 retained_history_samples=sum(len(x) for x in pipe._pending_noise_iq),
                 worker_state=pipe.worker.state.value,
                 worker_unfinished=pipe.worker._queue.unfinished_tasks,
                 worker_completed=pipe.worker.completed_count,
                 worker_discarded_generation=pipe.worker.discarded_old_generation_count,
                 worker_failed=pipe.worker.failed_count,
                 worker_sink_errors=pipe.worker.sink_error_count, worker_cancelled=pipe.worker.cancelled_count,
                 cancellation_operations=pipe.worker.cancellation_operations,
                 worker_analysis_seconds=pipe.worker.analysis_seconds, worker_sink_seconds=pipe.worker.sink_seconds,
                 retained_worker_results=len(pipe.worker.processed_results))
    stats['producer_failed_samples'] = (stats['produced_samples']-stats['admitted_samples']
        - stats['producer_deadline_dropped_samples']-stats['queue_dropped_samples'])
    stats['dropped_samples'] += (stats['unproduced_samples'] + undelivered + stats['consumer_failed_samples']
                                 + stats['producer_failed_samples'])
    stats["recovered"] = (not producer.is_alive() and arrivals.empty() and arrivals.unfinished_tasks == 0
                          and pipe.worker.state.value == 'CLOSED' and pipe.worker._queue.unfinished_tasks == 0)
    stats["sample_accounting_balanced"] = total == stats["consumed_samples"] + stats["dropped_samples"]
    stats['delivery_scope'] = 'accepted_callbacks_and_immediate_unaccepted_returns; polling_queue_disabled' if features else 'capture_return'
    stats['delivery'] = dict(delivery, unresolved_identities=len(pending))
    stats['acquisition_passed'] = (not stats['errors'] and stats['sample_accounting_balanced'] and
                                  stats['deadline_misses'] == 0 and stats['dropped_samples'] == 0)
    stats['capture_accounting_passed'] = not stats['errors'] and not delivery['ledger_overflow']
    stats['analysis_delivery_passed'] = (not features or (
        not pending and delivery['raw_captures'] == delivery['terminal_results']
        and not any(delivery[k] for k in ('duplicate_results', 'failed_results', 'overloaded_results', 'ledger_overflow'))
        and not any(stats[k] for k in ('worker_failed', 'worker_sink_errors', 'worker_cancelled',
                                      'worker_discarded_generation', 'overload_drops', 'cancellation_operations'))))
    stats['passed'] = all(stats[k] for k in ('acquisition_passed', 'capture_accounting_passed', 'analysis_delivery_passed', 'recovered'))
    return stats


def detection_sweep(trials=3, noise_exposure_seconds=1.0):
    """Multiple truths, duration/frequency/bandwidth/noise sweeps and unmatched events."""
    if trials < 1 or noise_exposure_seconds <= 0:
        raise ValueError("trials and exposure must be positive")
    fs = 1e6
    stream = StreamInfo("sweep", "controlled-synthetic", fs)
    rows = []
    scenarios = [("cw", 0., 0.), ("cw", 200_000., 0.),
                 ("bandlimited_noise", 30_000., 10_000.), ("bandlimited_noise", -100_000., 100_000.)]
    for duration in (64, 256, 2048):
        for modulation, frequency, bandwidth in scenarios:
            for noise_model in ("white", "ar1_0.8"):
                for snr in (-10., 0., 10.):
                    hits = misses = extras = all_matched = 0
                    onset, ending = [], []
                    for trial in range(trials):
                        rng = np.random.default_rng(1500 + len(rows) * trials + trial)
                        n = 8192
                        noise = (rng.normal(size=n + 512) + 1j * rng.normal(size=n + 512)) * math.sqrt(1e-4 / 2)
                        if noise_model != "white":
                            noise = signal.lfilter([math.sqrt(1 - .8**2)], [1, -.8], noise)
                        iq = noise[512:]
                        truth = [(1000, 1000 + duration), (4500, 4500 + duration)]
                        for start, end in truth:
                            if modulation == "cw":
                                wave = np.ones(duration, complex)
                            else:
                                sos = signal.butter(4, bandwidth / fs, output="sos")
                                raw = rng.normal(size=duration + 4096) + 1j * rng.normal(size=duration + 4096)
                                wave = signal.sosfilt(sos, raw)[4096:]
                                wave /= math.sqrt(float(np.mean(abs(wave)**2)))
                            iq[start:end] += math.sqrt(1e-4 * 10**(snr / 10)) * wave * np.exp(2j*np.pi*frequency*np.arange(duration)/fs)
                        pipe = _pipeline(stream)
                        events = []
                        for seq, start in enumerate(range(0, n, 512)):
                            events.extend(pipe.process_block(IQBlock(stream, seq, start, None, iq[start:start+512].astype(np.complex64))))
                        events.extend(pipe.flush())
                        pipe.shutdown()
                        spans = [(s.event.detected_start_sample, s.event.detected_end_sample) for s in events]
                        pairs, lost, false = match_intervals(truth, spans)
                        hits += len(pairs); misses += len(lost); extras += len(false)
                        all_matched += not lost
                        for i, j in pairs:
                            onset.append(spans[j][0] - truth[i][0]); ending.append(spans[j][1] - truth[i][1])
                    lo = float(beta.ppf(.025, all_matched, trials-all_matched+1)) if all_matched else 0.
                    hi = float(beta.ppf(.975, all_matched+1, trials-all_matched)) if all_matched < trials else 1.
                    rows.append(dict(duration_samples=duration, modulation=modulation, frequency_hz=frequency,
                                     bandwidth_hz=bandwidth, noise_model=noise_model, snr_db=snr,
                                     matched_truths=hits, missed_truths=misses, unmatched_detections=extras,
                                     independent_trials=trials, all_truths_detected_trials=all_matched,
                                     trial_success_95pct_interval=[lo, hi],
                                     mean_onset_error_samples=float(np.mean(onset)) if onset else None,
                                     mean_end_error_samples=float(np.mean(ending)) if ending else None))
    pipe = _pipeline(stream)
    rng = np.random.default_rng(9001)
    total = int(fs * noise_exposure_seconds)
    false_count = 0
    for seq, start in enumerate(range(0, total, 1024)):
        n = min(1024, total-start)
        iq = ((rng.normal(size=n) + 1j*rng.normal(size=n))*math.sqrt(1e-4/2)).astype(np.complex64)
        false_count += len(pipe.process_block(IQBlock(stream, seq, start, None, iq)))
    false_count += len(pipe.flush()); pipe.shutdown()
    return dict(scenarios=rows, false_events=false_count, noise_exposure_seconds=total/fs,
                false_events_per_second=false_count/(total/fs),
                poisson_95pct_upper_events_per_second=float(chi2.ppf(.95, 2*(false_count+1))/2/(total/fs)),
                uncertainty_assumptions="Trial intervals use independent seeded trials. Poisson event-rate bound is conditional; clustered receiver events need a different model.")
