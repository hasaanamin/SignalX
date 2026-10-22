# On-demand features and real-time LTE: working candidate

Status: **BLOCKED for a lossless real-time freeze**. Version remains
`2.4.0rc1`; this is not another final release. Actual final observations and
functional/build results are recorded below when qualification completes.

## Work boundaries

All received IQ still traverses metadata/clock validation, causal noise and
threshold processing, detection, segmentation, circular storage and exact
received-sample capture. Features were already **event-level**, never evaluated
once per individual IQ sample. The previous dense-stream stress workload
nevertheless requested all 56 scalars for every emitted event (roughly 244
events/s). That is a different workload from a UI/classification consumer.

New explicit `feature_analysis_mode='on_demand'` returns raw captures without
automatically submitting feature jobs. `request_features(capture,purpose=...)`
selects an immutable capture for the existing asynchronous worker. Existing
automatic behavior remains the default. Capture-only behavior remains available.
Offline dataset UI/CSV jobs are not silently changed to omit event features.

`FeatureRequestGate` can also wrap a separate `FeatureAnalysisWorker`, including
the parent-side worker used with `CaptureStreamProcess`. It retains no IQ or
capture history, only scalar counters and a UI admission clock:

- `purpose='ui'`: at most 60 requests/s (a configurable smaller ceiling is
  allowed). Throttling returns `skipped` immediately, with a stated reason.
- `purpose='classification'`: bypasses the UI rate limit, not finite worker
  capacity. It does not implement or run a classifier.
- Accepted requests return `pending`; their terminal results use the worker's
  declared callback or polling channel. The real ten-family/56-key calculation,
  provenance validation, quality gating and reset/generation contracts apply.
- Capacity rejection returns `dropped_overload` immediately. Unaccepted requests
  never call external sinks on the acquisition caller. Neither request method
  waits for capacity. Required sinks still participate in bounded shutdown.
- The gate requires an asynchronous worker. It owns no threads, processes or
  extra notification queue. The worker owns those resources and must be closed.
- A 60-Hz UI should render the latest completed result; this ceiling is **not**
  a guarantee of 60 complete 56-feature computations or new events each second.
  Repeated requests for the same event are deliberate new work, not new RF events.
  An application retaining captures for later selection must bound its history.
- Classification can tolerate seconds, but arbitrary model runtime is not a
  feature-worker throughput guarantee. Select explicit queue/time budgets and
  handle rejection. The process feature backend currently limits an event to
  4,096 IQ samples; the compatibility thread backend supports larger events.

Example (insert into a program's existing guarded main when using processes):

```python
pipe = StreamingCapturePipeline(
    stream_info=stream, noise_estimator=estimator,
    synchronous_analysis=False, feature_analysis_mode='on_demand',
    analysis_backend='process_pool', process_workers=1,
    result_delivery='callbacks',
    # Supply the detector/retention configuration required by your profile.
)
pipe.worker.add_result_sink(on_features)  # not called by process_block
captures = pipe.process_block(block)      # every IQ sample is preprocessed
if captures and user_requests_features:
    immediate = pipe.request_features(captures[-1], purpose='ui')
# Feed later input normally. At EOF: deliver flush() captures, then shutdown().
```

Raw gaps do not cancel assembled jobs. Explicit reset/cancel remains separate.
No lifecycle lock is held across analysis or sinks. Snapshot counters are
lifetime request counts, not an alternative unfinished-job ledger.

## LTE includes actual decoding

LTE timing always includes channel preparation, PSS/SSS detection and
synchronization, PBCH extraction/channel estimation, real BCH decoding/CRC and
MIB construction. It does not merely time synchronization. Generic 56-feature
analysis was already absent from this path, so removing it cannot by itself
repair LTE latency.

The original bounded/reference FFT and NumPy calculations remain available.
`sync_backend='numba'` explicitly selects pre-warmed, serial, nopython/nogil
direct tracking/SSS kernels. There is no fast math, running-subtraction power,
dynamic thread team or signal/result cache. `sync_backend='numpy'` remains the
compatibility default; `auto` allows an optional-dependency fallback. Compilation
and configuration FFT readiness finish before acquisition starts and are
reported as initialization, not hidden initial input loss.

The NumPy FFT-batching and incremental-correlation-cache experiments did not
improve scheduled operation and were removed. Their failed logs remain present.

`decode_cadence='every_frame'` preserves per-radio-frame attempts/publication.
The additive `decode_cadence='bch_block'` mode attempts undecoded PBCH frames,
including existing chronological soft combining. **Only after a genuine CRC-
and semantic-valid MIB identifies the quarter** does it avoid the remaining
repetitions of that same 40-ms BCH block. It resumes on the next codeword.
No predicted/synthetic MIB is emitted. Gaps/reset erase this knowledge; at most
`max_cells` scalar block-end/last-considered records are retained. This mode has
a distinct broadcast profile identity, reports avoided repetitions, and does
not claim to decode every redundant 10-ms frame. A failed codeword is not marked
decoded merely to save work. Tested scope remains FDD/normal CP/central six RB,
not LTE user-data-channel demodulation or SDR validation.

The protocol's 40-ms scrambling boundary is specified in
[ETSI/3GPP TS 36.211, section 6.6.1](https://www.etsi.org/deliver/etsi_ts/136200_136299/136211/16.02.00_60/ts_136211v160200p.pdf).
The native kernels use the documented
[Numba nopython/nogil mechanism](https://numba.readthedocs.io/en/stable/user/jit.html#nogil);
these references do not certify this implementation's timing.

## Acceptance is explicit, not redefined after a failure

`benchmarks.stream_isolated` retains `all_events` by default. Its new explicit
`--feature-mode none` times capture/preprocessing without feature jobs;
`--feature-mode on_demand` samples UI requests at up to 60/s and makes a
classification feature request every two input seconds. It never waits for
feature capacity in on-demand mode. Raw captures, unrequested captures, accepted
requests, terminal identities, overloads and recovery are counted separately.
Unrequested work is not a missing accepted job. A failure/overload on requested
work still fails delivery acceptance.

`--acceptance-scope processing` excludes **only** synthetic producer expiry as
requested; acquisition-queue loss, incorrect accounting, missing/failed jobs
and unfinished cleanup still fail. Scheduled totals/expiry remain visible.
It does not claim a one-block hard deadline or hardware losslessness. Actual
commit latency is separate from grouped scalar telemetry receipt latency.

The LTE benchmark retains its legacy lossless/every-frame gate. The explicit
`--lte-acceptance-scope realtime_decoding` instead requires source-eligible IQ
without acquisition-queue loss, correct PCI/timing, at least one actual verified
MIB in **every** 40-ms codeword block, p99 <=50 ms / maximum <=100 ms including
decoding, and closed/empty recovery. Every-frame CRC misses and intentionally
avoided repetitions remain visible. This scope also excludes only synthetic
expiry, not receiver/queue losses. Both scopes currently fail measured timing.

Primary host: Windows 11 / Core i5-1235U, Python 3.14.7; pinned observed
dependencies are in `requirements-acquisition-observed.txt`. Queues and IQ
block sizes are unchanged in the target tests. The fixed LTE file is genuine
native 1.92 MS/s, not relabelled 2 MS/s. No hardware receiver is connected.

## Reproducible commands

Run each timing command separately, without pytest/profiling/tracing alongside:

```powershell
py -3.14 -m benchmarks.stream_isolated --seconds 30 --adaptive --profile positive_snr --feature-mode none --progress-group 4 --acceptance-scope processing --output output/NEW/adaptive_none.json
py -3.14 -m benchmarks.stream_isolated --seconds 30 --adaptive --profile positive_snr --feature-mode on_demand --feature-processes 1 --result-order completion --progress-group 4 --acceptance-scope processing --output output/NEW/adaptive_requests.json
py -3.14 -m benchmarks.audit_pipeline_runtime lte --input output/native_acquisition_fix_20261020/lte_native_30s.cf32 --sample-rate 1920000 --sync-backend numba --decode-cadence bch_block --lte-acceptance-scope realtime_decoding --output output/NEW/lte.json
```

The LTE IQ file is an external generated fixture, not a required wheel payload.
Recreate it using `benchmarks.audit_pipeline_runtime prepare --seconds 30
--sample-rate 1920000 --input PATH --output NEW_METADATA_JSON`. Truth is used
only by the evaluator, never given to the receiver.

Focused checks: `py -3.14 -m pytest tests/regressions/test_on_demand_features.py -q`.
They ingest real captured/independently encoded IQ, use actual aggregation and
bounded public thread gates, compare native dot products with independent
`math.fsum` oracles at scales 1e-20/1/1e20, compare reference outputs, and check
codeword updates, quarter offsets, corruption and gap cache invalidation.

Final combined functional/build command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File benchmarks/qualify_scheduling_candidate.ps1 -OutputRoot output/on_demand_20261020/qualification -NumericalRoot output/native_acquisition_fix_20261020/numerical
```

This runs the source suite, explicitly routed current checks, genuine numerical
equivalents, bounded-memory/fault/recovery checks and a fresh isolated wheel
build/install. Installed checks include this change's public APIs/oracles.
Suite counts overlap. A successful functional runner does not turn failed
scheduled timing into acceptance.

## Observations

See `output/on_demand_20261020/` for complete logs, including failed experiments.

### Scheduled adaptive acquisition, 30 seconds each

Same 1 MS/s, 1,024-sample blocks, N=256/H=64 positive-SNR adaptive profile,
4,096 ring and 16-block input capacity. All percentages below exclude only
synthetic producer expiry; none excludes acquisition-queue loss.

| Mode | Consumed / source-eligible samples | Processing fraction | Queue loss | Requested / terminal features | Gate |
|---|---:|---:|---:|---:|---|
| Preprocessing/capture only | 26,944,384 / 27,181,952 | 99.1260% | 237,568 | 0 / 0 | FAIL |
| On demand, normal priority, one feature process | 25,970,560 / 26,101,632 | 99.4978% | 131,072 | 1,564 / 1,564 | FAIL |
| On demand, explicit above-normal acquisition processes | 29,309,824 / 29,782,912 | 98.4115% | 473,088 | 1,574 / 1,574 | FAIL |

Normal-priority on-demand: 6,690 raw captures, 5,126 intentionally unrequested
UI jobs, 14 classification-feature requests; zero feature-capacity rejections,
duplicates or unresolved jobs. Processing p50/p99/max was 0.745/1.914/5.678 ms
per block versus 1.024 ms interarrival. Actual commit p99/max was 20.024/35.761
ms. Worker recovery was clean and unfinished work zero. Source expiry remained
3,898,368 samples, separately visible; actual scheduled acquisition was 86.57%,
not 99.50%. The latter is solely the explicitly requested eligible-input view.

The priority experiment improved source scheduling but increased queue loss;
it is not adopted as a default. There is no evidence for a lossless freeze.

### LTE: decoding is correct in exercised offline cases, not scheduled-qualified

Controlled one-second real IQ comparison: NumPy took 1.3263 s, optional native
tracking/SSS 1.1872 s, both yielding 100 genuine MIBs. The optional block cadence
with one BLAS thread took 0.9703 s and yielded 25 MIBs, one per codeword. These
are unpaced diagnostics, **not** proofs of scheduled acceptance. A seed-2233
native -5-dB four-frame parity fixture has three decodable frames in both
reference and new paths; that failure is retained, not hidden by changing seed.

The 30-second native 1.92-MS/s, -5-dB/CFO12,345-Hz encoded-file run with explicit
native synchronization, block decoding and one BLAS thread consumed only
9,365,504 / 57,600,000 samples (16.2596%). Source expiry was zero; 48,234,496
samples were lost at the acquisition queue. Of 750 codewords, only 151 had a
verified MIB. Published results had no wrong PCI/MIB/timing or duplicate result,
but missing data/codewords are real failures. p99/max total latency was
1,689.582/1,825.494 ms against 50/100 ms. Synchronization consumed 28.933 s;
PBCH extraction/decoding consumed 0.940 s. Cleanup closed with no outstanding
deliveries. Enforcement exited 1. **Saving redundant decoding did not cure
blind acquisition/reacquisition tail cost.**

All intermediate scheduled failures are retained: initial NumPy batching,
native synchronization, incremental caching, and block decoding. Batching and
incremental caching were removed when they did not help. An additional final
NumPy/every-frame compatibility diagnostic also failed (12.3449% acquired;
p99/max 2,849.956/3,049.277 ms); a WMI hardware-inventory call overlapped part
of that diagnostic, so it is not an isolated performance qualification or a
controlled claim of regression versus the earlier reports. The explicit native
run above had no heavy test suite running alongside it. These results do not
reproduce the earlier near-lossless LTE observation. OS/host variability versus
code-path cost needs controlled attribution; no unmeasured thermal, power-plan
or Linux explanation is asserted as the cause.

The remaining concrete bottleneck is the blind-search/reacquisition path's
worst-case cost, which can exceed the finite queue's 136.53-ms input coverage
and trigger a feedback loop of loss, reset and expensive reacquisition. A
separately measured acquisition/DSP process boundary and exact blind-search
kernel work are practical next investigations, not implemented/performance-
certified fixes in this change. No enlarged queue, relaxed latency, missing
decoder or stubs are used to claim a pass.

### Final functional/build verification

| Check | Actual result |
|---|---|
| Supplied source suite after combined production edits | 1,515 passed, one deliberate arithmetic warning, zero failures/errors/skips; 277.98 s |
| Current acceptance, explicit `SIGNAL_PROCESSING_SRC` | 165 passed, zero failures/errors/skips; 14.26 s |
| New focused request/native/cadence checks | 18 passed, included in source/installed suites; not an extra unique suite count |
| Independent numerical/lifecycle scripts | All three executed to completion with exit 0 and passing observations, at original tolerances; genuine current equivalents, not old saved answers |
| Isolated installed wheel | 515 passed, zero failures/errors/skips; 227.98 s; all 120 production modules byte-identical and imports verified outside checkout |
| Required failing sink, 10 kS/s / 1 s | All 10,000 samples processed, 3 real sink failures, enforcement exit 1, recovery true, zero unresolved/unfinished work |
| Slow sink, 10 kS/s / 3 s | All 30,000 samples processed, delivery/recovery pass, enforcement exit 0, zero unresolved/unfinished work; not the 1-MS/s target |
| Traced quiet LTE retention | 4,096,000 samples; retained IQ histories constant at 29,100/62,400 samples; live trace 44,680 ->55,637 bytes, RSS 185,274,368 ->185,225,216 bytes |
| RSS instrument sanity | Touched 128 MiB changed RSS 104,796,160 ->239,017,984 bytes; responsive |

After the full source run, a benchmark-only guard was added so **unpaced** runs
cannot report `realtime_decoding_passed=True`. The two source harness cases were
rerun (exit 0); installed verification includes their stronger assertions. No
runtime production module changed after the full source gate. All counts
overlap and must not be added as unique tests. Frozen original checks remain
unchanged; their already documented API/statistical migrations are preserved.

The actual functional runner's commands/exit codes are in
`qualification/execution.json`; the extra source guard check is recorded in
`qualification/benchmark_guard.log` and `.xml`. The prior G/F/H and available
A/B/E functional gates remain exercised in the combined suites; **E14 timing
remains failed**, not converted to a pass by these counts. Three successful
30-second runs/ten-minute lossless soak are not claimed while shorter gates fail.

Development wheel (not a frozen release):
`output/on_demand_20261020/qualification/wheels/signal_processing-2.4.0rc1-py3-none-any.whl`.
SHA-256: `05fadb4c41bce6cf24997954c13c60724724fc3ebc14bbf7dfc1d3c2ad8461fc`.
Its 120 module members and CRC were checked; exact current production hashes are
in `qualification/production_manifest.csv`. Old archives/builds were not edited.

**Outcome:** feature work is explicitly decoupled and the new mechanisms pass
functional/numerical/installed checks, but adaptive lossless acquisition and LTE
real-time latency are unresolved. No accepted/frozen ZIP or `FROZEN_PASS` is
produced. Next work must target the measured acquisition/reacquisition/scheduling
bottleneck, not omit decoding or quietly relax the gate.
