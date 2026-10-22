# Why LTE 2 MS/s and adaptive 1 MS/s still lose acquisition

Diagnosis date: 19 October 2026. Active package: 2.4.0rc1. Primary host: Windows 11 build 26200, Core i5-1235U, 10 physical/12 logical CPUs. Python 3.14.7, NumPy 2.5.2, SciPy 1.18.0, Numba 0.67.0.

**Conclusion:** these are different bottlenecks. LTE has insufficient continuous processing throughput even without a scheduled source or generic feature extraction. Adaptive capture has sufficient isolated average throughput, but its small-block processing, IPC, feature workload and scheduling interference leave inadequate real-time headroom. Neither is currently lossless or qualified to freeze.

No production implementation was changed in this diagnosis. Added only a live attribution harness and this report. No wheel/ZIP was rebuilt, no release gate was reclassified, and no full test suite was rerun. Existing dirty source and historical evidence were preserved.

## 1. Existing scheduled results: separate the loss mechanisms

These are the previously executed final 30-second runs, not new reruns:

| Path | Scheduled / consumed | Acquisition | Producer expiry | Input queue loss | Processing p50 / p99 | Input block budget |
|---|---|---:|---:|---:|---:|---:|
| LTE low-SNR profile, 2 MS/s, 16,384 samples | 60,000,000 / 25,919,488 | 43.1991% | 0 | 34,080,512 | 11.659 / 158.634 ms | 8.192 ms |
| Adaptive positive-SNR profile, 1 MS/s, 1,024 samples | 30,000,000 / 26,607,616 | 88.6921% | 361,472 | 3,030,912 | 0.694 / 3.135 ms | 1.024 ms |

Sources: `output/acquisition_fix_20261019/lte_final_simd_1.json` and `adaptive_1ms_final.json`.

For LTE, the producer delivered every scheduled input sample to SourceRunner, but its bounded queue rejected 56.8% because the receiver could not consume fast enough. This is not a producer-expiry accounting artifact. Even the median processing block exceeds its arrival interval.

For adaptive capture, queue loss is 10.1030% of the original schedule and producer expiry is another 1.2049%. Ignoring expiry does not remove the 3,030,912 samples lost when the input queue filled. Input queue capacity is 16 blocks, approximately 16.384 ms of incoming IQ; LTE's 16 blocks represent 131.072 ms. These are finite jitter buffers, not additional processing capacity.

An end-to-end block deadline includes queue waiting and delivery, not just the processing call. Therefore a median processing time below 1.024 ms does not contradict nearly every adaptive block arriving late downstream.

## 2. Fresh LTE attribution: computation, not an I/O-only problem

The current real LTE receiver consumed the first four million complex64 samples of the existing encoded test recording without a SourceRunner queue or scheduled dropping. Numba readiness completed before timing. Nothing supplied runtime truth or noise power to the receiver.

| Offline control, two seconds of input | Processing wall time | Calling-thread CPU | Throughput | Results |
|---|---:|---:|---:|---|
| Default BLAS environment, no stage timers/cProfile | 3.352 s | 3.266 s | 1.193 MS/s | 196 MIBs, 397 synchronization observations |
| Explicit BLAS threads=1, no stage timers/cProfile | 3.406 s | 3.313 s | 1.175 MS/s | Same 196 MIBs and 397 observations |

No discontinuities or lock losses occurred in either control. The exact same receiver mathematics cannot service a sustained 2 MS/s stream on these measured runs. Approximately 97% of its wall time was accounted as calling-thread CPU, so ordinary I/O waiting or Windows wake-up jitter cannot explain this deficit alone. These are short diagnostic controls, not sustained performance qualifications.

The BLAS control does not establish a meaningful speed improvement; timing varies with host state. A one-thread BLAS setting alone did not close the throughput gap. No conclusion about the exact default BLAS runtime thread count is inferred from an unset environment variable.

### Measured work per second of real input

The following disjoint grouping comes from a separate one-second stage-instrumented run, total 1.753 s wall. Timers perturb execution. Parent/child timers were subtracted where necessary to avoid double counting; the individual JSON stage fields themselves are inclusive.

| Work | Wall time | Share of instrumented total |
|---|---:|---:|
| PSS/SSS synchronization, tracking and frequency refinement | 0.557 s | 31.8% |
| PBCH decoding, including soft demapping and BCH recovery | 0.410 s | 23.4% |
| Causal 2 MHz → 1.92 MHz channel preparation | 0.259 s | 14.7% |
| Broadcast history concatenations alone | 0.210 s | 12.0% |
| PBCH slot extraction / received CP-based frequency refinement | 0.063 s | 3.6% |
| Remaining buffer copies, bookkeeping, validation and overhead | 0.254 s | 14.5% |

Concrete causes in active source:

1. **Synchronization remains substantial after lock.** `lte/receiver.py:_tracking_candidates` evaluates the original 27 PSS/frequency branches around predicted timing, including four-repeat integration. `_decode_integrated` still performs 31-point and 11-point frequency refinement and four-repeat SSS work. This input second produced 204 integrated decodes and 458 tracking searches. Tracking is much cheaper than full blind FFT search, but is not free.
2. **PBCH still tests a large exact hypothesis set.** `lte/pbch.py:PBCHDecoder.decode` checks three antenna configurations and four quarter hypotheses for a single frame, then permitted multi-frame combinations if needed. The observed 100 PBCH frame attempts invoked `recover_bch` 1,254 times. A fully unsuccessful four-frame search can invoke it 30 times. Native Viterbi accelerates one part of each attempt; rate recovery, soft extraction, CRC and hypothesis orchestration remain.
3. **History is bounded but repeatedly moved.** `lte/broadcast.py:_search` concatenates the retained broadcast IQ with newly available IQ and then copies the retained suffix. Its 469 concatenations allocated 490,859,520 output bytes for just one second of input. This is allocated/copied volume, not retained memory or an IQ leak. It excludes the subsequent suffix copies. The receiver's own input concatenations added 91,149,008 bytes. Reducing queue loss does not eliminate this unnecessary memory traffic.
4. **Resampling is real work.** `lte/channel.py:process` calls the causal 1,201-tap polyphase filter, converts input to complex128 and copies the released output/history. Its time is significant, although it is not the largest stage.
5. **Queue drops create a recovery penalty.** The scheduled run encountered 658 discontinuities, 79 lock losses and 555 blind searches. Each genuine missing interval resets cross-gap synchronization and BCH combining. The continuous two-second control had no discontinuities and only four startup blind searches. This establishes a reacquisition feedback mechanism, not an exact percentage of time attributable to it. Retaining lock across missing IQ would violate the continuity contract.

The NumPy/Numba paths are already native where selected; the receiver is nevertheless one ordered caller executing these stages serially. Numba currently accelerates exact BCH, not the entire LTE receive chain. Its default kernel is one thread and diagnostics explicitly report `dynamic_queue_scaling=False`. Existing kernel measurements favor serial SIMD for this small trellis; merely enabling more threads is not a demonstrated solution.

## 3. Fresh adaptive attribution: fast in isolation, inadequate headroom in context

All controls use the same seeded IQ generation, N=256/H=64, M=1,024 adaptive statistic, 256-sample pre-trigger, 4,096-sample ring, and chronological pipeline batching. No adaptation was disabled to obtain the isolated result. Feature-off runs are explicitly diagnostic, not substitutes for the required full pipeline.

| Fresh control | Acquisition / throughput | Other observations |
|---|---|---|
| Offline adaptive pipeline, 2M samples, no cProfile/stage timers | 3.207 MS/s; 0.624 s summed processing wall | 489 actual captures; processing p99 1.072 ms |
| Scheduled capture-only, 10 s | 96.9485%; 168,960 expiry + 136,192 queue samples lost | 2,379 raw captures; diagnostic teardown issue noted below |
| Scheduled capture + real four-process features, 10 s | 91.0400%; 103,424 expiry + 792,576 queue samples lost | 2,234 terminal results: 2,045 ok, 189 quality skips; no missing/duplicate outcomes; recovered |
| Scheduled capture-only, 5 s, corrected diagnostic join | 99.0374%; 48,128 expiry, zero queue loss | 1,215 raw captures; queue peak 7, clean drain, no source errors |

The runs were sequential, not simultaneous, and differ in duration/load. They demonstrate sensitivity and isolate relevant costs; they do not establish a repeatable percentage advantage or a lossless acceptance pass. Host boost, placement and other applications were uncontrolled.

### What actually consumes the adaptive budget

- **Many small state/object operations, not just median noise arithmetic.** The profiled two-million-sample continuous input made 5,889 detector commits, 7,843 received-slice constructions, 9,821 noise-estimate calls, 13,239 threshold lookups, 9,797 pending-age synchronizations and 15,686 pending-count setter calls. Pure window power was 0.086 s of profiled cumulative time; adaptive `update_samples` was 0.137 s; obtaining estimates was 0.114 s. These timings are profiler rankings, not uninstrumented production budgets. The calculation must remain causal, but repeated immutable envelope creation and redundant state queries are optimization candidates.
- **Adaptive boundaries limit the fixed-reference optimization.** A reference can change at its next complete authorized M statistic, and candidates/active IQ must be excluded. Current batching deliberately stops at these boundaries. Treating the entire block as fixed noise to get a higher throughput would be a correctness regression.
- **Per-block transport and reconstruction remain.** `runtime/capture_stream.py` sends one completion record per 1,024-sample input, even if no capture completed. That is approximately 977 messages per second, plus source transport, sidecar encoding/decoding and feature request/result messages. A separately instrumented three-second full run spent 0.215 s in capture-child `connection.send` for 1,484 records, including 0.103 s pickling. Pipeline processing was 2.366 s of that approximately three-second child profile. Thus IPC matters, but is not the only bottleneck.
- **Feature processing is not free merely because its queue drains.** In the existing 30-second run, feature children used 42.469 aggregate CPU seconds, parent feature coordination used 12.000 thread-CPU seconds, capture used 24.563, and the producer used 14.047. Feature CPU is parallel across four children: do not add it to latency as though it were serial. Nonetheless it competes for CPU/cache/memory and power on this hybrid laptop. There are roughly 244 burst opportunities/second in this deliberately event-dense workload.
- **Required-sink or capacity waiting is not the dominant observed cause.** The fresh full ten-second run spent only 0.049 s waiting for feature capacity; the old thirty-second run spent 0.149 s. Both delivered every accepted capture outcome without overload or sink failure. This differs from saying that every IQ sample was captured. Feature work and parent reconstruction can still slow the rest of the system without a persistently saturated feature input queue.
- **Tail latency exceeds available buffering.** Full-run processing p99 was 3.053 ms versus a 1.024 ms interval. A short burst of slow blocks/scheduling pauses fills the finite input queue even when median work is faster. The source additionally enforces an independently advancing arrival schedule and post-sleep expiry; source expiry and queue rejection remain distinct losses.

Capture calling-thread CPU per processed block was approximately 0.406 ms in the first capture-only control versus 0.909 ms with features enabled. This is evidence of workload-dependent service cost, not proof of a particular E-core placement, throttling event or cache-miss mechanism. Those require hardware/scheduler counters. It is not defensible to attribute the whole difference to Windows or to a Python GIL shared across separate processes: those interpreters do not share one GIL. Parent completion and feature-coordination threads do share their own interpreter and may interfere.

### Measurement boundary that matters

`capture_stream._stream_capture` stamps `finished` after pipeline processing, **before** building/encoding the completion message, acquiring its bounded completion slot and calling `connection.send`. Its reported processing wall time therefore omits these subsequent costs. Its thread-CPU sample includes message construction but still precedes semaphore wait/send. End-to-end latency includes them.

Consequently the existing processing percentile is useful but cannot prove total service time fits the arrival budget. The independent source accounting and downstream deadlines remain the appropriate loss/latency evidence.

## 4. Priority of a future repair

This section identifies candidates, not implemented or qualified fixes.

1. **LTE: replace repeated full-history concatenation/copying with bounded owned storage.** Preserve exact received indices, four-repeat history, retrospective PBCH extraction and failure/reset ownership. Measure both copied bytes and latency.
2. **LTE: batch/compile the complete exact hypothesis workload, not just Viterbi.** Hoist immutable rate-recovery/scrambling/CRC/template work; evaluate identical hypotheses with fewer Python/native crossings. Investigate a cheaper verified tracking refinement path with explicit loss-of-lock fallback. Do not drop quarter/antenna alternatives or alter low-SNR support silently.
3. **Adaptive: reduce repeated hop metadata/reference envelopes and pending-state work.** Establish trusted internal inputs only after public preflight. Reuse immutable configuration and unchanged-reference calculations without bypassing age, adaptation, evidence or causal freeze checks.
4. **Adaptive: reduce empty per-block completion traffic and parent object reconstruction.** A bounded coalesced progress-notification option could retain the original input blocks/windows and deliver completed captures promptly, with truthful fault horizons and exact acknowledgement. This is not permission to enlarge the acquisition queue, change source scheduling or retain unbounded work.
5. **Measure process placement, GIL handoffs and power behavior after reducing work.** Merely increasing feature processes can worsen contention; merely enlarging the input queue cannot fix LTE's mean service deficit. A Linux comparison can test scheduler/IPC effects but cannot be presumed to remove the measured serial LTE computation.

Acceptance must still include the full real feature workload and previous numerical/continuity/lifecycle gates. No sensitivity reduction, lost-sample reclassification, fixed-reference substitution for adaptation or suppressed deadlines is justified by these observations.

## 5. Evidence and reproduction

New live harness: `benchmarks/attribute_processing_limits.py`. All successful diagnostics exited 0; its exit means the observation completed, **not** that acquisition passed. The existing full scheduled CLI exited 1 because its acquisition gate failed. No production tests were relabelled as passed.

Artifacts: `output/processing_attribution_20261019/` contains JSON, complete logs and cProfile files:

- `lte_clean`, `lte_blas1_clean`: minimally instrumented continuous throughput controls.
- `lte_default_timers`, `lte_blas1_timers`, `lte_allocations`: inclusive stage and allocation observations.
- `lte_default_profile.prof`: current source hotspots, with startup/JIT outside profiling.
- `adaptive_clean`, `adaptive_clean_profile`: actual adaptive arithmetic/state path and call counts.
- `adaptive_capture_only_10s`, `adaptive_full_10s`, `adaptive_capture_only_joined_5s`: independent scheduled controls.
- `adaptive_full_profile.capture.prof` and `.worker.prof`: actual three-second multi-process workload under profiling. The heavily instrumented run acquired only about 50.6%; it is attribution, not a normal performance result. Parent cProfile under Python 3.14 also captures other parent-thread activity, so its overlapping wait times are not exact per-thread CPU attribution. Child process profiling is separate.

Diagnostic limitations are preserved rather than hidden:

- An initial profile save failed because the new output directory was absent; the harness now creates it before execution. No successful measurement was inferred from that failed invocation.
- The first ten-second capture-only helper called source `finish()` without joining the already-finished producer. That helper triggered its forced-termination warning. Its JSON retains the error; it is not clean shutdown evidence. The helper now joins first. The separate five-second control completed with no source errors and a clean capture drain. Counts from the first run are used only for acquisition/processing attribution.
- Short-call Windows `thread_time` readings are quantized. Do not use accumulated per-stage thread-CPU fields as precise percentages; stage wall times and the longer whole-run CPU intervals are used above.
- Timers/cProfile perturb execution. No traced/profiled result is presented as unperturbed sustained acceptance. No hardware radio, Linux, thermal or CPU-core-placement claim was measured here.

Commands, from repository root (use a new output filename; existing evidence is not overwritten):

```powershell
py -3.14 -m benchmarks.attribute_processing_limits lte --seconds 2 --input output/pipeline_audit_20261019/lte_30s.cf32 --no-stage-timers --output output/new_lte_control.json
py -3.14 -m benchmarks.attribute_processing_limits lte --seconds 1 --input output/pipeline_audit_20261019/lte_30s.cf32 --allocations --output output/new_lte_allocations.json
py -3.14 -m benchmarks.attribute_processing_limits adaptive --seconds 2 --no-stage-timers --profile --output output/new_adaptive_profile.json
py -3.14 -m benchmarks.attribute_processing_limits capture_only --seconds 5 --output output/new_capture_control.json
py -3.14 -m benchmarks.stream_isolated --seconds 10 --adaptive --profile positive_snr --result-order completion --feature-processes 4 --output output/new_full_adaptive.json
py -3.14 -m benchmarks.attribute_processing_limits full_profile --seconds 3 --output output/new_full_profile.json
```

**Release status unchanged: BLOCKED.** LTE remains compute-limited below 2 MS/s on this measured path; adaptive 1 MS/s still loses samples in its complete workload. All-outcome delivery and successful cleanup do not imply lossless acquisition.
