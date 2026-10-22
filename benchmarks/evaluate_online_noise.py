"""Offline changing-noise characterization; never supplies truth to detection.

No scheduled-acquisition pass is inferred from this unpaced experiment.
Run: python -m benchmarks.evaluate_online_noise
"""
from __future__ import annotations
import json
import time

import numpy as np

from signal_processing.core import IQBlock, StreamInfo
from signal_processing.detection import OnlineNoiseConfig
from signal_processing.runtime.pipeline import StreamingCapturePipeline
from signal_processing.ui.smoke_runner import match_activity


def run(model, changing):
    rng = np.random.default_rng(90210)
    n = 262144
    t = np.arange(n)/n
    rho = (.2+.5*t if changing else np.full(n, .8)) if model == "ar1" else np.zeros(n)
    power = 1e-4*(1+t if changing else np.ones(n))
    innovations = (rng.normal(size=n)+1j*rng.normal(size=n))*np.sqrt(power*(1-rho**2)/2)
    x = np.empty(n, np.complex64)
    previous = 0j
    for i in range(n):
        previous = rho[i]*previous+innovations[i]
        x[i] = previous
    truths = [(2048, 2304)] + [(i, i+512) for i in range(20000, n-5000, 12000)]
    for a, b in truths:
        x[a:b] += np.sqrt(10*power[a:b])*np.exp(.4j*np.arange(b-a))
    config = OnlineNoiseConfig(model=model, time_constant_s=.005)
    stream = StreamInfo("online-characterization", "synthetic", 1e6)
    p = StreamingCapturePipeline(stream, online_noise=config,
        window_size=256, hop_size=64, ring_capacity=16384,
        pre_trigger_samples=256, pre_signal_margin_samples=128, post_signal_samples=256,
        min_post_trigger_samples=256, max_event_samples=16384,
        synchronous_analysis=True, skip_incomplete_quality=False)
    captures = []
    started = time.perf_counter()
    try:
        for seq, i in enumerate(range(0, n, 1024)):
            captures.extend(p.process_block(IQBlock(stream, seq, i, None, x[i:i+1024])))
        captures.extend(p.flush())
        elapsed = time.perf_counter()-started
        matches = match_activity(truths, [(c.event.detected_start_sample, c.event.detected_end_sample) for c in captures], .2)
        outcomes = {}
        for c in captures:
            outcomes[c.feature_status] = outcomes.get(c.feature_status, 0)+1
        return dict(model=model, changing=changing, samples=n, elapsed_seconds=elapsed,
            unpaced_samples_per_second=n/elapsed, scheduled_qualification=False,
            truths=len(truths), matched=len(matches), missed=len(truths)-len(matches),
            extras=len(captures)-len(matches), matches=matches,
            startup_truth_included=True, final_expected_rho=float(rho[-1]),
            final_expected_innovation=float(power[-1]*(1-rho[-1]**2)),
            tracker=p.noise_estimator.diagnostics(), retention=p.retention_snapshot(),
            outcomes=outcomes, exact_iq=all(np.array_equal(c.event.iq, x[c.event.event_start_sample:c.event.event_end_sample]) for c in captures),
            default_schema=all(c.features is not None and len(c.features.features)==56 for c in captures))
    finally:
        p.shutdown()


def main():
    observations = [run(model, changing) for model in ("gaussian", "ar1") for changing in (False, True)]
    print(json.dumps(observations, indent=2))


if __name__ == "__main__":
    main()
