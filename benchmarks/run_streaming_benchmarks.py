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


def evaluate_detection_performance(num_trials_per_snr: int = 3, noise_exposure_seconds: float = 1.0) -> dict:
    from benchmarks.sustained_evaluation import detection_sweep
    result = detection_sweep(num_trials_per_snr, noise_exposure_seconds)
    print(f"Detection sweep: {len(result['scenarios'])} scenarios, {num_trials_per_snr} independent trials each")
    print(f"Noise exposure: {result['noise_exposure_seconds']:.3f}s; false events: {result['false_events']}")
    print(result["uncertainty_assumptions"])
    return result


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


def run_paced_stress_benchmark(duration_seconds: float = 5.0, **kwargs) -> dict:
    from benchmarks.sustained_evaluation import bounded_acquisition
    result = bounded_acquisition(duration_seconds, **kwargs)
    print(f"Independent bounded acquisition: {result['blocks_processed']} blocks; "
          f"deadline misses={result['deadline_misses']}; "
          f"lost samples={result['dropped_samples']}; "
          f"queue peak={result['acquisition_queue_peak']}/{result['queue_capacity']}; "
          f"recovered={result['recovered']}")
    print("PASS" if result["passed"] else "FAIL: host did not meet acquisition deadlines/loss criteria")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming RF Signal Analysis Benchmarks")
    parser.add_argument("--stress-duration", type=float, default=2.0, help="Duration in seconds for paced stress run")
    parser.add_argument("--trials", type=int, default=3, help="Independent trials per sweep scenario")
    parser.add_argument("--noise-exposure", type=float, default=1.0)
    parser.add_argument("--skip-detection", action="store_true")
    parser.add_argument("--queue-capacity", type=int, default=16)
    parser.add_argument("--trace-memory", action="store_true")
    parser.add_argument("--slow-sink", type=float, default=0.0)
    parser.add_argument("--consumer-pause", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enforce-acceptance", action="store_true", help="Exit with code 1 if real-time acceptance criteria are not met")
    args = parser.parse_args()

    detection = None if args.skip_detection else evaluate_detection_performance(args.trials, args.noise_exposure)
    throughput = run_latency_and_throughput_benchmarks()
    stress_results = run_paced_stress_benchmark(
        duration_seconds=args.stress_duration, queue_capacity=args.queue_capacity,
        features=True, slow_sink_seconds=args.slow_sink,
        consumer_pause_seconds=args.consumer_pause, trace_memory=args.trace_memory,
    )
    if args.output:
        import json
        args.output.write_text(json.dumps(dict(detection=detection, throughput=throughput,
                                              acquisition=stress_results), indent=2), encoding="utf-8")
    print("\n" + "=" * 80)
    print("ALL EVALUATIONS AND BENCHMARKS COMPLETED")
    print("=" * 80)

    if args.enforce_acceptance and not stress_results.get("passed", False):
        print("FAIL: Enforce acceptance failed (real-time deadlines missed or queue drops occurred)")
        sys.exit(1)


if __name__ == "__main__":
    main()
