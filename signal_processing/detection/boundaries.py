"""Bounded received-IQ boundary refinement; never crops the raw capture."""
from dataclasses import replace

import numpy as np


def refine_ar1_capture(sidecar, model, search_samples=256):
    """Fixed-variance likelihood edge scan on each side of a coarse event.

    This is a point estimate, not a confidence certificate. No unavailable samples
    or truth coordinates enter the scan. Coarse landmarks and raw IQ are preserved.
    """
    event = sidecar.event
    base = event.event_start_sample
    left, right = event.detected_start_sample-base, event.detected_end_sample-base
    if not event.complete or right-left < 2:
        return sidecar
    x = event.iq.astype(np.complex128)-model.mean
    y = x[1:]-model.coefficient*x[:-1]
    powers = y.real*y.real+y.imag*y.imag
    # Index i in powers describes raw sample i+1. The unavailable predecessor
    # at capture index zero is excluded rather than zero padded.
    lo, hi = max(1,left), min(len(x),right)
    noise = model.innovation_power
    signal = float(np.mean(powers[lo-1:hi-1]))
    if not np.isfinite(signal) or signal <= noise:
        return sidecar
    llr = np.log(noise/signal) + powers*(1/noise-1/signal)
    if not np.all(np.isfinite(llr)):
        return sidecar
    start_min = max(1,left-search_samples)
    start_max = min(right-1,left+search_samples,event.trigger_sample-base)
    end_min = max(start_max+1,right-2*search_samples,event.trigger_sample-base+1)
    end_max = min(len(x),right+search_samples)
    if start_min>start_max or end_min>end_max:
        return sidecar
    # Only short local likelihood arrays are accumulated. No subtraction of
    # a large earlier raw-power prefix is used for a small quiet-window sum.
    start_scores = np.cumsum(llr[start_min-1:start_max][::-1])[::-1]
    start = start_min+int(np.argmax(start_scores))
    end_scores = np.cumsum(llr[end_min-1:end_max-1])
    end = end_min if not len(end_scores) or float(np.max(end_scores))<=0 else end_min+1+int(np.argmax(end_scores))
    if end <= start:
        return sidecar
    refined = replace(event, detected_start_sample=base+start, detected_end_sample=base+end)
    return replace(sidecar,event=refined,coarse_detected_start_sample=event.detected_start_sample,
                   coarse_detected_end_sample=event.detected_end_sample,
                   boundary_method="ar1_local_variance_edges_v1")
