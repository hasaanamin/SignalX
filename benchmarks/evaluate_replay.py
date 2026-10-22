"""Measure finite positioned replay; no LTE decoder or RF hardware is implied.

Use a precomputed raw complex64 recording via --input. The optional generated
fixture is a seeded noise stream for acquisition tests only; generation occurs
before scheduling begins. --noise-power enables the real existing capture and
56-feature pipeline with an explicitly supplied fixed reference.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np

from signal_processing.core import StreamInfo
from signal_processing.detection import GaussianNoiseEstimator
from signal_processing.runtime import SourceRunner, StreamingCapturePipeline
from signal_processing.sources import ArrayIQSource, RecordedFileIQSource, ReplayIQSource, SourceMetadata


def evaluate(*, path=None, seconds=1., sample_rate=2_000_000., block_size=16384,
             capacity=16, lateness_budget=.01, consumer_delay=0., scheduled=True,
             noise_power=None):
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(consumer_delay) or consumer_delay < 0:
        raise ValueError("seconds must be positive and consumer_delay nonnegative")
    stream = StreamInfo("replay-evaluation", "file" if path else "generated-noise", sample_rate)
    prepared = time.perf_counter()
    if path:
        underlying = RecordedFileIQSource(path, stream, block_size=block_size)
    else:
        n = round(seconds*sample_rate)
        if not 1 <= n <= 20_000_000:
            raise ValueError("Generated test fixture is capped at 20 million samples; use --input for longer recordings")
        rng = np.random.default_rng(731)
        iq = (rng.normal(0, .003, n)+1j*rng.normal(0, .003, n)).astype(np.complex64)
        underlying = ArrayIQSource(iq, stream, block_size=block_size)
        del iq
    source = ReplayIQSource(underlying, scheduled=scheduled,
        lateness_budget_s=lateness_budget, metadata=SourceMetadata(), owns_source=True)
    pipeline = None
    if noise_power is not None:
        pipeline = StreamingCapturePipeline(stream, window_size=64, hop_size=32,
            ring_capacity=4096, pre_trigger_samples=128,
            noise_estimator=GaussianNoiseEstimator(window_size=64, calibrated_noise_power_linear=noise_power),
            enable_feature_analysis=True, synchronous_analysis=True)
    preparation_seconds = time.perf_counter()-prepared
    runner = SourceRunner(source, capacity=capacity, max_block_samples=block_size, owns_source=True)
    captures = feature_failures = blocks = 0
    errors = []
    latencies = []  # Finite cap; percentiles are a bounded latest-window diagnostic.
    max_latency = 0.
    started = time.perf_counter()
    runner.start()
    try:
        while (lease := runner.get(timeout=5)) is not None:
            try:
                if consumer_delay:
                    time.sleep(consumer_delay)
                if pipeline:
                    events = pipeline.process_block(lease.envelope.block)
                    captures += len(events)
                    feature_failures += sum(e.feature_status != "ok" for e in events)
                    while pipeline.feature_worker.get_result(block=False) is not None:
                        pass  # Drain the declared polling channel.
                if lease.envelope.scheduled_monotonic_ns is not None:
                    latency = max(0, time.perf_counter_ns()-lease.envelope.scheduled_monotonic_ns)/1e6
                    max_latency = max(max_latency, latency)
                    latencies.append(latency)
                    if len(latencies) > 4096:
                        del latencies[:2048]
                blocks += 1
            except Exception:
                runner.ack(lease, outcome="failed")
                raise
            else:
                runner.ack(lease)
        if pipeline:
            events = pipeline.flush()
            captures += len(events)
            feature_failures += sum(e.feature_status != "ok" for e in events)
            while pipeline.feature_worker.get_result(block=False) is not None:
                pass
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            runner.shutdown(drain=False, timeout=5)
        except Exception as exc:
            errors.append(f"source cleanup: {type(exc).__name__}: {exc}")
        if pipeline:
            try:
                pipeline.shutdown()
            except Exception as exc:
                errors.append(f"pipeline cleanup: {type(exc).__name__}: {exc}")
    elapsed = time.perf_counter()-started
    r, s = runner.snapshot(), source.snapshot()
    accounting = (r.accounting_balanced
        and s.read_samples == s.returned_samples+s.expired_samples+s.pending_samples
        and s.total_samples == s.read_samples+s.unread_samples
        and s.returned_samples == r.read_samples)
    passed = (not errors and accounting and s.finished and s.expired_samples == 0
              and r.queue_dropped_samples == r.failed_samples == r.cancelled_samples == 0
              and r.unaccepted_cancelled_samples == r.outstanding_deliveries == feature_failures == 0
              and r.state == "CLOSED" and r.terminal_reason == "eof")
    return {"passed": passed, "scope": "capture/features" if pipeline else "acquisition only",
            "sample_rate_hz": sample_rate, "block_size": block_size, "capacity": capacity,
            "scheduled": scheduled, "lateness_budget_s": lateness_budget,
            "consumer_delay_s": consumer_delay, "preparation_seconds": preparation_seconds,
            "elapsed_seconds": elapsed, "processed_samples_per_second": r.processed_samples/elapsed,
            "sample_accounting_balanced": accounting, "blocks": blocks,
            "captures": captures, "feature_failures": feature_failures,
            "latency_ms": ({"p50": float(np.percentile(latencies, 50)),
                            "p99": float(np.percentile(latencies, 99)), "max": max_latency} if latencies else None),
            "latency_scope": "percentiles: latest bounded window up to 4096 delivered blocks; max: whole run",
            "runner": dataclasses.asdict(r), "replay": dataclasses.asdict(s), "errors": errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Raw little-endian complex64 recording")
    parser.add_argument("--seconds", type=float, default=1.)
    parser.add_argument("--sample-rate", type=float, default=2_000_000.)
    parser.add_argument("--block-size", type=int, default=16384)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--lateness-budget", type=float, default=.01)
    parser.add_argument("--consumer-delay", type=float, default=0.)
    parser.add_argument("--unpaced", action="store_true")
    parser.add_argument("--noise-power", type=float, help="Enable actual pipeline using this external fixed noise reference")
    parser.add_argument("--enforce-acceptance", action="store_true")
    args = parser.parse_args(argv)
    result = evaluate(path=args.input, seconds=args.seconds, sample_rate=args.sample_rate,
        block_size=args.block_size, capacity=args.capacity, lateness_budget=args.lateness_budget,
        consumer_delay=args.consumer_delay, scheduled=not args.unpaced, noise_power=args.noise_power)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if args.enforce_acceptance and not result["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
