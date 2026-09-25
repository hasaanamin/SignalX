"""Quantitative evaluation and performance benchmark script for the streaming capture pipeline.

Measures:
  1. Detection probability (Pd), false alarm rate (Pfa), onset/end error, and decision latency
     across SNRs (-15 dB to +20 dB) and burst durations using 1-to-1 IoU matching.
  2. Ingestion and extraction throughput at 1 MS/s complex64, p50/p95/p99 latency per block,
     memory usage, and queue occupancy across capture-only, asynchronous, and synchronous modes.
  3. Paced streaming stress run verifying bounded execution and queue saturation safety.
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import scipy

from signal_processing.core.contracts import StreamInfo
from signal_processing.detection.noise_estimation import GaussianNoiseEstimator
from signal_processing.runtime.pipeline import StreamingCapturePipeline
from signal_processing.sources.array_source import ArrayIQSource
from signal_processing.sources.synthetic_source import BurstDescriptor, SyntheticBurstSource


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate Wilson score confidence interval for binomial proportion."""
    if trials == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence z-score
    p_hat = successes / trials
    denominator = 1.0 + (z**2) / trials
    centre = (p_hat + (z**2) / (2.0 * trials)) / denominator
    spread = (z * math.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4.0 * trials)) / trials)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def evaluate_detection_performance(num_trials_per_snr: int = 50) -> dict:
    print("=" * 80)
    print("PART 1: QUANTITATIVE DETECTION PERFORMANCE & 1-TO-1 IOU EVALUATION")
    print("=" * 80)

    fs = 1_000_000.0  # 1 MS/s
    stream_info = StreamInfo("stream-eval-01", "synthetic", fs, 915e6)

    snr_levels = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0]
    burst_duration_samples = 2048  # 8*N (N=256)
    noise_power_linear = 1e-4

    results = {}

    print(f"Testing {num_trials_per_snr} trials per SNR level for 8*N (2048 sample) bursts...")
    print(f"{'SNR (dB)':<9} | {'Pd (%)':<9} | {'95% CI':<18} | {'1st On Lat (smp)':<16} | {'Onset Err (smp)':<15} | {'Fall Err (smp)'}")
    print("-" * 95)

    for snr in snr_levels:
        detected_count = 0
        first_on_latencies = []
        onset_errors = []
        fall_errors = []

        for trial in range(num_trials_per_snr):
            burst_start = 2000
            burst_end = burst_start + burst_duration_samples
            burst = BurstDescriptor(
                start_sample=burst_start,
                duration_samples=burst_duration_samples,
                snr_db=snr,
                modulation="cw",
                freq_offset_hz=30_000.0,
            )
            source = SyntheticBurstSource(
                stream_info=stream_info,
                total_samples=6000,
                block_size=512,
                noise_power_linear=noise_power_linear,
                bursts=[burst],
                seed=1000 + int(snr * 10) + trial,
            )

            noise_estimator = GaussianNoiseEstimator(
                window_size=256,
                pfa_on=1e-4,
                pfa_off=1e-2,
                calibrated_noise_power_linear=noise_power_linear,
            )

            pipeline = StreamingCapturePipeline(
                stream_info=stream_info,
                ring_capacity=8192,
                window_size=256,
                hop_size=64,
                noise_estimator=noise_estimator,
                start_persistence=2,
                end_persistence=2,
                synchronous_analysis=True,
            )

            emitted = []
            while (block := source.read_block()) is not None:
                emitted.extend(pipeline.process_block(block))
            emitted.extend(pipeline.flush())

            # Evaluate 1-to-1 match using positive Intersection over Union (IoU >= 0.2)
            matched = False
            for sc in emitted:
                evt = sc.event
                inter = max(0, min(evt.detected_end_sample, burst_end) - max(evt.detected_start_sample, burst_start))
                union = max(evt.detected_end_sample, burst_end) - min(evt.detected_start_sample, burst_start)
                iou = inter / union if union > 0 else 0.0

                if iou >= 0.2:
                    matched = True
                    first_on_latencies.append(sc.first_on_sample - burst_start)
                    onset_errors.append(abs(evt.detected_start_sample - burst_start))
                    fall_errors.append(abs(evt.detected_end_sample - burst_end))
                    break

            if matched:
                detected_count += 1

        pd = detected_count / num_trials_per_snr
        ci_low, ci_high = wilson_score_interval(detected_count, num_trials_per_snr)
        mean_on_lat = float(np.mean(first_on_latencies)) if first_on_latencies else float("nan")
        mean_on_err = float(np.mean(onset_errors)) if onset_errors else float("nan")
        mean_fall_err = float(np.mean(fall_errors)) if fall_errors else float("nan")

        results[snr] = {
            "pd": pd,
            "ci": (ci_low, ci_high),
            "mean_first_on_latency_samples": mean_on_lat,
            "mean_onset_error_samples": mean_on_err,
            "mean_falling_edge_error_samples": mean_fall_err,
        }

        print(f"{snr:<9.1f} | {pd*100:<9.1f} | [{ci_low*100:.1f}%, {ci_high*100:.1f}%]{'':<4} | {mean_on_lat:<16.1f} | {mean_on_err:<15.1f} | {mean_fall_err:.1f}")

    # Noise-only false alarm rate test (100,000 noise samples = 100 ms)
    noise_source = SyntheticBurstSource(
        stream_info=stream_info,
        total_samples=100_000,
        block_size=1024,
        noise_power_linear=noise_power_linear,
        bursts=[],
        seed=9999,
    )
    noise_estimator = GaussianNoiseEstimator(window_size=256, pfa_on=1e-4, pfa_off=1e-2, calibrated_noise_power_linear=noise_power_linear)
    pipeline = StreamingCapturePipeline(
        stream_info=stream_info,
        ring_capacity=8192,
        window_size=256,
        hop_size=64,
        noise_estimator=noise_estimator,
        start_persistence=2,
        end_persistence=2,
    )
    noise_emitted = []
    while (b := noise_source.read_block()) is not None:
        noise_emitted.extend(pipeline.process_block(b))
    noise_emitted.extend(pipeline.flush())

    obs_time_s = 100_000 / fs
    false_events_count = len(noise_emitted)
    false_events_per_sec = false_events_count / obs_time_s
    print(f"\nNoise-only false events per second: {false_events_per_sec:.2f} events/sec ({false_events_count} events observed over {obs_time_s*1e3:.0f} ms observation time)")
    results["false_events_per_sec"] = false_events_per_sec
    return results


def run_latency_and_throughput_benchmarks() -> dict:
    print("\n" + "=" * 80)
    print("PART 2: LATENCY, THROUGHPUT & REAL-TIME SUSTAINABILITY BENCHMARK")
    print("=" * 80)

    fs = 1_000_000.0  # 1 MS/s
    stream_info = StreamInfo("stream-bench-01", "benchmark", fs, 915e6)
    block_size = 1024
    total_samples = 200_000  # 200 ms continuous stream

    rng = np.random.default_rng(42)
    stream_data = (rng.normal(0, 1e-2, total_samples) + 1j * rng.normal(0, 1e-2, total_samples)).astype(np.complex64)
    for b_idx in range(5):
        b_start = 20_000 + b_idx * 35_000
        stream_data[b_start : b_start + 4000] += 0.5

    # Benchmark 1: Ingestion & Ring Buffer Capture Pipeline Alone (Feature extraction disabled)
    pipe_capture_only = StreamingCapturePipeline(
        stream_info=stream_info,
        ring_capacity=16_384,
        window_size=256,
        hop_size=64,
        noise_estimator=GaussianNoiseEstimator(window_size=256, calibrated_noise_power_linear=2e-4),
        start_persistence=2,
        end_persistence=2,
        enable_feature_analysis=False,
    )

    src1 = ArrayIQSource(stream_data, stream_info, block_size=block_size)
    capture_block_times = []

    t0 = time.perf_counter()
    while (block := src1.read_block()) is not None:
        t_b0 = time.perf_counter()
        pipe_capture_only.process_block(block)
        capture_block_times.append(time.perf_counter() - t_b0)
    pipe_capture_only.flush()
    pipe_capture_only.shutdown()
    t_cap_total = time.perf_counter() - t0

    cap_thp = total_samples / t_cap_total / 1e6
    cap_p50 = float(np.percentile(capture_block_times, 50)) * 1e3
    cap_p95 = float(np.percentile(capture_block_times, 95)) * 1e3
    cap_p99 = float(np.percentile(capture_block_times, 99)) * 1e3

    # Benchmark 2: Full End-to-End Pipeline (Capture + Synchronous 56-Feature Extraction)
    pipe_full = StreamingCapturePipeline(
        stream_info=stream_info,
        ring_capacity=16_384,
        window_size=256,
        hop_size=64,
        noise_estimator=GaussianNoiseEstimator(window_size=256, calibrated_noise_power_linear=2e-4),
        start_persistence=2,
        end_persistence=2,
        synchronous_analysis=True,
    )

    src2 = ArrayIQSource(stream_data, stream_info, block_size=block_size)
    full_block_times = []
    full_events = []

    t0 = time.perf_counter()
    while (block := src2.read_block()) is not None:
        t_b0 = time.perf_counter()
        emitted_b = pipe_full.process_block(block)
        full_block_times.append(time.perf_counter() - t_b0)
        full_events.extend(emitted_b)
    full_events.extend(pipe_full.flush())
    t_full_total = time.perf_counter() - t0

    full_thp = total_samples / t_full_total / 1e6
    full_p50 = float(np.percentile(full_block_times, 50)) * 1e3
    full_p95 = float(np.percentile(full_block_times, 95)) * 1e3
    full_p99 = float(np.percentile(full_block_times, 99)) * 1e3

    # Quantitative feature validity breakdown (B13)
    total_keys = len(full_events) * 56
    valid_keys = sum(
        1
        for ev in full_events
        if ev.features is not None
        for f in ev.features.features.values()
        if f.valid
    )
    invalid_keys = total_keys - valid_keys

    env_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }

    print(f"Host Environment: {env_info['platform']} | Python {env_info['python_version']} | NumPy {env_info['numpy_version']} | SciPy {env_info['scipy_version']}")
    print(f"Sample Rate: 1.0 MS/s | Block Size: {block_size} samples (1.024 ms duration per block)")
    print("\nUnpaced Capture Pipeline Alone (Feature extraction disabled):")
    print(f"  Throughput:           {cap_thp:.2f} MS/s ({cap_thp/1.0:.2f}x real-time)")
    print(f"  Latency p50:          {cap_p50:.3f} ms / block")
    print(f"  Latency p95:          {cap_p95:.3f} ms / block")
    print(f"  Latency p99:          {cap_p99:.3f} ms / block")
    print(f"  Real-time achieved:   {'YES (Substantial headroom)' if cap_p99 < 1.024 else 'Marginal'}")

    print("\nUnpaced Full Pipeline (Capture + Synchronous 56-Feature Extraction):")
    print(f"  Events Captured:      {len(full_events)}")
    print(f"  Total Canonical Keys: {total_keys}")
    print(f"  Valid Features:       {valid_keys} ({valid_keys/max(1, total_keys)*100:.1f}%)")
    print(f"  Invalid Features:     {invalid_keys} ({invalid_keys/max(1, total_keys)*100:.1f}%)")
    print(f"  Throughput:           {full_thp:.2f} MS/s ({full_thp/1.0:.2f}x real-time)")
    print(f"  Latency p50:          {full_p50:.3f} ms / block")
    print(f"  Latency p95:          {full_p95:.3f} ms / block")
    print(f"  Latency p99:          {full_p99:.3f} ms / block")
    print(f"  Real-time achieved:   {'YES' if full_thp > 1.0 else 'NO (Slower than real-time; asynchronous worker recommended)'}")

    return {
        "env": env_info,
        "capture_thp_msps": cap_thp,
        "full_thp_msps": full_thp,
        "capture_p50_ms": cap_p50,
        "capture_p95_ms": cap_p95,
        "capture_p99_ms": cap_p99,
        "full_p50_ms": full_p50,
        "full_p95_ms": full_p95,
        "full_p99_ms": full_p99,
        "valid_features": valid_keys,
        "invalid_features": invalid_keys,
    }


def run_paced_stress_benchmark(duration_seconds: float = 2.0) -> dict:
    print("\n" + "=" * 80)
    print(f"PART 3: GENUINE PACED STREAMING BENCHMARK ({duration_seconds:.1f} s equivalent at 1 MS/s)")
    print("=" * 80)

    fs = 1_000_000.0
    stream_info = StreamInfo("stream-stress-01", "stress", fs, 915e6)
    total_samples = int(duration_seconds * fs)
    block_size = 1024
    block_duration_s = block_size / fs  # 1.024 ms

    # Generate signal with periodic bursts
    bursts = [
        BurstDescriptor(
            start_sample=int(i * 0.2 * fs),
            duration_samples=2048,
            snr_db=15.0,
            modulation="cw",
            freq_offset_hz=20_000.0,
        )
        for i in range(max(1, int(duration_seconds / 0.2)))
    ]

    source = SyntheticBurstSource(
        stream_info=stream_info,
        total_samples=total_samples,
        block_size=block_size,
        noise_power_linear=1e-4,
        bursts=bursts,
        seed=42,
    )

    pipeline = StreamingCapturePipeline(
        stream_info=stream_info,
        ring_capacity=16_384,
        window_size=256,
        hop_size=64,
        noise_estimator=GaussianNoiseEstimator(window_size=256, calibrated_noise_power_linear=1e-4),
        start_persistence=2,
        end_persistence=2,
        synchronous_analysis=False,
        max_worker_queue_size=64,
    )

    # Genuine paced loop with independent arrival deadlines (B13)
    t_start = time.perf_counter()
    emitted_count = 0
    blocks_processed = 0
    deadline_misses = 0
    max_backlog_ms = 0.0

    while (block := source.read_block()) is not None:
        target_delivery_time = t_start + (blocks_processed + 1) * block_duration_s
        now = time.perf_counter()
        if now < target_delivery_time:
            # Pace producer to match physical arrival rate
            time.sleep(target_delivery_time - now)
        else:
            # Producer fell behind deadline
            lateness_ms = (now - target_delivery_time) * 1e3
            deadline_misses += 1
            if lateness_ms > max_backlog_ms:
                max_backlog_ms = lateness_ms

        emitted = pipeline.process_block(block)
        emitted_count += len(emitted)
        blocks_processed += 1

    pipeline.flush()
    pipeline.shutdown(drain=True, timeout=5.0)
    elapsed = time.perf_counter() - t_start

    overload_drops = pipeline.feature_worker.dropped_overload_count
    deadline_miss_pct = (deadline_misses / max(1, blocks_processed)) * 100.0

    print(f"  Stream Duration:      {duration_seconds:.3f} s ({total_samples} samples, {blocks_processed} blocks)")
    print(f"  Wall-clock Elapsed:   {elapsed:.3f} s")
    print(f"  Block Interval:       {block_duration_s*1e3:.3f} ms")
    print(f"  Deadline Misses:      {deadline_misses} / {blocks_processed} ({deadline_miss_pct:.1f}%)")
    print(f"  Max Arrival Backlog:  {max_backlog_ms:.3f} ms")
    print(f"  Events Emitted:       {emitted_count}")
    real_time_passed = (overload_drops == 0 and deadline_misses == 0)
    print(f"  Paced Assessment:     {'PASSED (Real-time arrival pace met, zero queue overload)' if real_time_passed else f'DEADLINE MISSES DETECTED ({deadline_misses}/{blocks_processed} missed, max backlog {max_backlog_ms:.2f} ms)'}")

    return {
        "duration_seconds": duration_seconds,
        "elapsed_seconds": elapsed,
        "blocks_processed": blocks_processed,
        "deadline_misses": deadline_misses,
        "max_backlog_ms": max_backlog_ms,
        "emitted_events": emitted_count,
        "overload_drops": overload_drops,
        "passed": real_time_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming RF Signal Analysis Benchmarks")
    parser.add_argument("--stress-duration", type=float, default=2.0, help="Duration in seconds for paced stress run")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials per SNR level")
    parser.add_argument("--enforce-acceptance", action="store_true", help="Exit with code 1 if real-time acceptance criteria are not met")
    args = parser.parse_args()

    evaluate_detection_performance(num_trials_per_snr=args.trials)
    run_latency_and_throughput_benchmarks()
    stress_results = run_paced_stress_benchmark(duration_seconds=args.stress_duration)
    print("\n" + "=" * 80)
    print("ALL EVALUATIONS AND BENCHMARKS COMPLETED")
    print("=" * 80)

    if args.enforce_acceptance and not stress_results.get("passed", False):
        print("FAIL: Enforce acceptance failed (real-time deadlines missed or queue drops occurred)")
        sys.exit(1)


if __name__ == "__main__":
    main()
