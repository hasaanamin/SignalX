# Acquisition acceleration: implemented contracts and limitations

19 October 2026, working candidate **2.4.0rc1**. Not a frozen/lossless release.
Diagnosis, plan and live results: [repair plan](ACQUISITION_LOSS_REPAIR_PLAN_20261019.md).

## What is accelerated

1. `StreamingCapturePipeline(..., fixed_quiet_batch=True)` now handles large
   full-band transport blocks as successive **<=4,096-sample, hop-aligned** pure
   numerical preparations. Each chunk commits chronologically before the next
   is prepared. It retains causal adaptive updates, exact raw retention and the
   typed fault/completed-output recovery contract. It does not batch SOS/AR1 or
   online-noise paths whose state rules differ. Legacy default remains False.
2. LTE blind search transforms only the received correlation regions needed by
   each repeated observation, not the entire retained history. Direct window
   energy remains nonnegative reduction, with no subtractive running sum.
   `search_backend='reference'` retains the full-history numerical reference.
3. Optional Numba compiles **exact all-64-initial-state tail-biting BCH**. The
   default one-thread kernel has destination-major contiguous metrics; no fast
   math, terminated-trellis approximation, reduced hypotheses or truth input.
4. Process-feature transport caches at most **32 deeply immutable policies**,
   keyed by the entire canonical record. Edited same-ID content is revalidated.
   Event IQ, vectors, values, contexts and quality are never cached across jobs.
5. On aligned x86-64 Windows with 64-KiB duplex pipes, a single-writer completion
   token avoids repeated kernel polling. Replies are capped at 48 KiB and the
   token is published only after the full reply write. Tokens do not replace
   message/version/event/policy validation or ownership. Other platforms/smaller
   pipes keep the polling path. Death checks remain bounded at 50 ms, timeouts
   remain enforced, and pending work is tracked through required delivery.

## Numba is not automatic dynamic thread scaling

Install explicitly in the supported environment:

```powershell
py -3.14 -m pip install -e ".[acceleration]"
```

This host uses Numba 0.67.0 / llvmlite 0.49.0, Python 3.14.7, NumPy 2.5.2,
SciPy 1.18.0. The base package can fall back to the reference without Numba.
An explicit unavailable `numba` request fails rather than pretending to accelerate.
JIT/cache readiness completes in the receiver constructor, **before acquisition**;
cold compilation can take several seconds and is not a dropped-input workaround.
The cache must be writable or configured through Numba's documented cache setting.

```python
from signal_processing.lte import LTEBroadcastReceiver, LTESyncConfig

receiver = LTEBroadcastReceiver(stream, LTESyncConfig.low_snr(),
    bch_backend="auto", numba_threads=1, search_backend="bounded")
print(receiver.execution)  # Detached, actual selected backend diagnostics.
```

`bch_backend` is `auto`, `reference` or `numba`. `numba_threads` is a strict
integer 1..4. One uses native `nogil` execution; 2..4 use `parallel=True/prange`
over independent initial states, not temporal stream decisions. On this host
the parallel layer is **OpenMP**, not TBB dynamic chunk scheduling. The caller's
thread-local mask is restored; no global start method, BLAS setting or Python
switch interval is changed. The pool capacity may exceed the active mask.
Concurrent native callers have private work arrays and tested mask restoration.

One thread remains default because the contiguous kernel was faster than its
2/4-thread alternative in the measured small-job workload. The implementation
does **not** automatically increase thread count with queue depth. The generic
feature worker still uses bounded Python threads/persistent spawn processes;
its ten default extractors were not replaced with Numba stubs.

## Backpressure and supported execution choices

The direct pipeline returns an immediate unaccepted feature-overload outcome;
it never invokes a slow overload sink on acquisition. The separate existing
`CaptureStreamProcess` completion pump can wait for feature capacity outside
source/capture execution, but bounded backpressure can still fill upstream
queues. Neither policy invents missing IQ or creates unlimited job storage.

Persistent feature pools can use `result_order='completion'` when callers accept
out-of-order analysis of independently identified events. Required sinks remain
serial and outside locks. Submission order remains the compatibility default.

Current measurements still show upstream queue losses and/or feature overload
at the tested event-dense 1/2 MS/s workloads. Acceleration is **not** a lossless
real-time certification. LTE synchronization/PBCH is still separate from generic
transient capture/56-feature integration. Pluto/hardware RF, classification,
LTE data-channel decoding and arbitrary receiver-noise qualification remain
outside these changes.
