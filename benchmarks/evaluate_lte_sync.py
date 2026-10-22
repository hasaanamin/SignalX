"""Measure LTE synchronization using independent reference-sequence OFDM IQ.

This fixture contains standards-correct FDD PSS/SSS but no encoded PBCH/MIB.
Expected PCI/timing/CFO are used only by this scorer, never by the receiver.
"""
import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta

from signal_processing.core import StreamInfo
from signal_processing.lte import LTESynchronizer
from signal_processing.sources import ArrayIQSource, ReplayIQSource, SourceMetadata
from signal_processing.runtime import SourceRunner
from tests.lte_fixture import waveform


def characterize(trials=100, snr_db=0., seed=20261019):
    rng = np.random.default_rng(seed)
    successes = wrong = 0
    timing_errors, frequency_errors = [], []
    started = time.perf_counter()
    for trial in range(trials):
        pci = int(rng.integers(0, 504))
        cfo = float(rng.uniform(-30000, 30000))
        onset = int(rng.integers(0, 10000))
        iq = waveform(pci, frames=4, cfo=cfo, snr_db=snr_db, onset=onset,
                      seed=int(rng.integers(0, 2**32)))
        stream = StreamInfo(f"lte-{trial}", "independent-sync-fixture", 2e6)
        receiver = LTESynchronizer(stream)
        observations = []
        source = ArrayIQSource(iq, stream, block_size=16384)
        while (block := source.read_block()) is not None:
            observations.extend(receiver.process_block(block))
        observations.extend(receiver.flush())
        confirmed = [o for o in observations if o.state == "SYNCHRONIZED"]
        wrong += int(any(o.pci != pci for o in confirmed))
        matched = [o for o in confirmed if o.pci == pci]
        if matched:
            o = matched[-1]
            error = float((o.input_frame_start-onset+10000) % 20000-10000)
            frequency_error = o.frequency_offset_hz-cfo
            timing_errors.append(error)
            frequency_errors.append(frequency_error)
            successes += int(abs(error) <= 2 and abs(frequency_error) <= 1500)
    lower = float(beta.ppf(.05, successes, trials-successes+1)) if successes else 0.
    def summary(values):
        return {"p50_absolute": float(np.percentile(np.abs(values), 50)),
                "p95_absolute": float(np.percentile(np.abs(values), 95)),
                "max_absolute": float(np.max(np.abs(values)))} if values else None
    return {"scope": "FDD normal-CP synchronization; independent sequence fixture; no PBCH/hardware",
            "trials": trials, "snr_db": snr_db, "snr_domain": "mean active central-72-carrier waveform power / iid input noise power",
            "successes": successes, "wrong_confirmed_trials": wrong,
            "one_sided_95_lower_success": lower,
            "mixed_cell_gate_passed": trials >= 400 and lower >= .9 and wrong == 0,
            "gate_scope": "mixture of random PCI/CFO/onset at this SNR, NOT a per-cell confidence certificate",
            "success_definition": "correct confirmed PCI, frame error <=2 raw samples, CFO error <=1500 Hz within 40 ms signal",
            "frame_error_input_samples": summary(timing_errors),
            "frequency_error_hz": summary(frequency_errors),
            "elapsed_seconds": time.perf_counter()-started, "seed": seed}


def scheduled(seconds=1.):
    prepared = time.perf_counter()
    stream = StreamInfo("scheduled-lte", "independent-sync-fixture", 2e6)
    iq = waveform(42, frames=math.ceil(seconds*100), cfo=13200., snr_db=0., onset=431)
    receiver = LTESynchronizer(stream)
    replay = ReplayIQSource(ArrayIQSource(iq, stream, 16384), scheduled=True,
                             lateness_budget_s=.01, metadata=SourceMetadata())
    runner = SourceRunner(replay, capacity=16, max_block_samples=16384)
    preparation_seconds = time.perf_counter()-prepared
    seen = confirmed = wrong = 0
    errors = []
    max_latency = 0.
    latency = []
    started = time.perf_counter()
    runner.start()
    try:
        while (lease := runner.get(timeout=5)) is not None:
            try:
                observations = receiver.process_envelope(lease.envelope)
                seen += len(observations)
                confirmed += sum(o.state == "SYNCHRONIZED" for o in observations)
                wrong += sum(o.pci != 42 for o in observations if o.state == "SYNCHRONIZED")
                delay = (time.perf_counter_ns()-lease.envelope.scheduled_monotonic_ns)/1e6
                max_latency = max(max_latency, delay)
                latency.append(delay)
                if len(latency) > 4096:
                    del latency[:2048]
            except Exception:
                runner.ack(lease, outcome="failed")
                raise
            runner.ack(lease)
        final = receiver.flush()
        seen += len(final)
        confirmed += sum(o.state == "SYNCHRONIZED" for o in final)
        wrong += sum(o.pci != 42 for o in final if o.state == "SYNCHRONIZED")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        runner.shutdown(drain=False)
    elapsed = time.perf_counter()-started
    r, s = runner.snapshot(), replay.snapshot()
    p99 = float(np.percentile(latency, 99)) if latency else None
    lag_passed = p99 is not None and p99 <= 50. and max_latency <= 100.
    passed = (not errors and s.finished and s.returned_samples == s.total_samples
              and s.expired_samples == r.queue_dropped_samples == r.outstanding_deliveries == 0
              and r.failed_samples == r.cancelled_samples == r.unaccepted_cancelled_samples == 0
              and r.accounting_balanced and r.read_samples == s.returned_samples
              and r.terminal_reason == "eof" and r.state == "CLOSED" and confirmed > 0 and wrong == 0
              and lag_passed)
    return {"scope": "2 MS/s acquisition plus LTE sync only; no generic feature work/PBCH",
            "passed": passed, "preparation_seconds": preparation_seconds,
            "elapsed_seconds": elapsed, "observations": seen, "confirmed": confirmed,
            "wrong_confirmed": wrong, "processing_lag_passed": lag_passed,
            "processing_lag_limits_ms": {"p99": 50., "max": 100.},
            "latency_ms": {"p99": p99,
            "max": max_latency}, "latency_scope": "p99 bounded last 4096 blocks; max whole run",
            "runner": dataclasses.asdict(r), "replay": dataclasses.asdict(s),
            "receiver": receiver.snapshot(), "errors": errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--snr", type=float, default=0.)
    parser.add_argument("--scheduled-seconds", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enforce-acceptance", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.trials <= 10000 or not math.isfinite(args.snr):
        parser.error("Invalid trial/SNR bounds")
    if args.scheduled_seconds is not None and not 0 < args.scheduled_seconds <= 30:
        parser.error("Scheduled fixture is capped at 30 seconds")
    result = scheduled(args.scheduled_seconds) if args.scheduled_seconds is not None else characterize(args.trials, args.snr)
    content = json.dumps(result, indent=2, allow_nan=False)
    print(content)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content+"\n", encoding="utf8")
    return 1 if args.enforce_acceptance and not result.get("passed", result.get("mixed_cell_gate_passed")) else 0


if __name__ == "__main__":
    raise SystemExit(main())
