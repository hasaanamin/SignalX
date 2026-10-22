# End-to-end pipeline audit — 19 October 2026

**Baseline observations before the subsequent repair request.** Later production
changes and follow-up results belong to `ACQUISITION_LOSS_REPAIR_PLAN_20261019.md`
and its repair evidence; these original measurements have not been overwritten.

Active working-folder package: **2.4.0rc1**, not a frozen release. This task
tests the implementation; it does not repair production code. Only a runtime
audit harness and its genuine encoded-IQ positive-control test were added.

## Scope and host

Windows 11 build 26200, Intel Core i5-1235U (10 physical / 12 logical cores),
8.25 GB installed RAM; Python 3.14.7, NumPy 2.5.2, SciPy 1.18.0. Available RAM
was approximately 0.8–1.3 GB during observations. No affinity, priority, global
scheduling or BLAS setting was changed. Normal host activity was not disabled.
The pre-existing `benchmarks.stream_isolated` entry point sets
`OPENBLAS_NUM_THREADS=1` if unset; the standalone runtime harness does not. No new
global BLAS setting was introduced during this audit. These results apply to this host/run, not a controlled comparison with previous
measurements. Timing runs were sequential, without pytest or memory tracing
running alongside them. Persistent workers were initialized before acquisition;
startup time is separately recorded.

There are **two separate production paths**:

- Generic transient detector, noise estimator, segmenter, circular retention,
  exact raw captures and bounded asynchronous 56-feature analysis.
- Positioned replay, LTE resampling, PSS/SSS and PBCH/MIB receiver.

LTE-aligned captures are not currently routed into the generic feature worker.
Testing both paths does not establish that missing integration. Synthetic LTE
IQ is not a hardware or complete eNodeB test.

## Scheduled generic acquisition — failures

| Configuration, 30 seconds | Scheduled | Consumed | Acquired | Source expiry | Acquisition queue loss | Feature overload |
|---|---:|---:|---:|---:|---:|---:|
| Adaptive 1 MS/s, 1,024/block, one feature process | 30,000,000 | 15,219,584 | 50.7319% | 692,224 | 14,088,192 | 0 |
| Adaptive 1 MS/s, 1,024/block, four feature processes | 30,000,000 | 25,654,144 | 85.5138% | 616,448 | 3,729,408 | 0 |
| Fixed reference 2 MS/s, 16,384/block, process-pool features | 60,000,000 | 9,437,184 | 15.7286% | 0 | 50,562,816 | 1,442 |

These are different explicit configurations, not interchangeable throughput
claims. All acquisition queues reached their capacity of 16 blocks. Source
expiry is **not** the only cause: most losses here are queue overflow. The
83,712 trailing samples in the 2 MS/s report are a **subset** of queue loss,
not an additional loss to add to it. Absolute end coordinates reaching the
scheduled end do not mean all intervening IQ was received.

| Configuration | Raw captures | Terminal outcomes | Successful analyses | Quality-skipped / overload | Unresolved / duplicate | Deadline misses | Terminal latency p99 / max |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 MS/s, one process | 3,701 | 3,701 | 2,171 | 1,530 / 0 | 0 / 0 | 14,863 / 14,863 blocks | 176.25 / 254.93 ms |
| 1 MS/s, four processes | 6,294 | 6,294 | 5,396 | 898 / 0 | 0 / 0 | 25,052 / 25,053 blocks | 56.63 / 74.53 ms |
| 2 MS/s | 2,304 | 2,304 | 862 accepted terminal callbacks; success/quality-skip breakdown not recorded | Not separately recorded / 1,442 | 0 / 0 | 576 / 576 blocks | Not measured by this harness |

The first two runs had no failed analyses, sink errors, cancellation or generation
discards. All three runs drained acquisition and worker tasks and retained only
128 result-history entries, with no pending noise/IQ history at EOF. Accounted
terminal rejection/skipping is not successful feature extraction or lossless IQ.

Measured attribution: single-process feature delivery backpressure totaled
18.24 seconds; four processes reduced it to 0.167 seconds, but the capture stage
still used 24.70 CPU-seconds in the four-process run and overflow persisted.
At 2 MS/s the consumer process used 28.83 CPU-seconds (including its threads);
block processing p50/p99 was 48.57/113.71 ms against an 8.192 ms arrival interval.
The reported process-feature round-trip sum was 105.52 seconds (overlapping
jobs; not elapsed wall time), versus 4.95 child feature CPU-seconds. This is
evidence of scheduling/transport/parent-path overhead, not proof that the
feature arithmetic itself consumed 105 seconds. Detailed profiling would be
needed before selecting production fixes.

## Evidence and reproducibility

All new observations/logs are in `output/pipeline_audit_20261019/`. Original
previous evidence remains unchanged. JSON observations are produced by live
execution, not read as expected runtime answers.

Run timing commands sequentially:

```powershell
py -3.14 -m benchmarks.stream_isolated --seconds 30 --adaptive --profile positive_snr --feature-processes 1 --output output/pipeline_audit_20261019/generic_1ms_serial.json
py -3.14 -m benchmarks.stream_isolated --seconds 30 --adaptive --profile positive_snr --feature-processes 4 --output output/pipeline_audit_20261019/generic_1ms_pool.json
py -3.14 -m benchmarks.audit_pipeline_runtime generic --seconds 30 --output output/pipeline_audit_20261019/generic_2ms.json
```

These three runs returned exit **1**. The isolated benchmark's relaxed
`candidate_passed` and strict `strict_passed` were both false. No 10-minute
lossless soak or freeze sign-off is warranted by these failures.

The newly added harness initially mishandled an absent scheduled timestamp in
an **unpaced** positive control. `harness.log` preserves that test failure. Only
the harness was corrected to treat unpaced latency separately; the genuine
80-ms receiver/encoded-IQ test then passed (`harness_corrected.log`). This was
not a production receiver repair and is not counted as a receiver defect.

## Scheduled replay and LTE — failures

Same finite 30-second, 60,000,000-sample raw complex64 file, 2 MS/s,
16,384 samples/block, 16-block queue, 10-ms source lateness budget. Prepared
offline using independent test mapping and frozen external PSS/SSS/BCH vectors;
two TX ports, PCI 42, signed carrier offset +12,345 Hz, -5 dB nominal mean
waveform/noise SNR. The runtime receives IQ/metadata, not truth or noise power.
Different SFNs/payloads are checked against frame identity, not just a CRC count.
The finite fixture SHA-256 and generation/environment details are recorded in
`lte_fixture.json`. It is synthetic central-six-RB FDD/normal-CP data, not a full
RF acquisition or arbitrary propagation test.

| Path | Consumed / 60,000,000 | Acquired | Source expiry | Acquisition queue loss | Latency p99 / maximum |
|---|---:|---:|---:|---:|---|
| Source-only replay control | 59,983,616 | 99.9727% | 16,384 | 0 | 2.00 / 4.47 ms for returned blocks |
| Compatibility LTE + PBCH | 10,387,456 | 17.3124% | 16,384 | 49,596,160 | 1,166.62 / 1,220.45 ms |
| Low-SNR LTE + PBCH | 8,568,832 | 14.2814% | 32,768 | 51,398,400 | 1,910.67 / 2,046.45 ms |

All three commands returned **1**, correctly failing strict acceptance. Source
control maximum lateness was 10.98 ms, slightly beyond its 10-ms expiry budget.
Its returned-block latency excludes the expired block and is not evidence that
no scheduling loss occurred. Both LTE queues filled to 16; source control peaked
at one. All paths had balanced sample accounting, no failed/cancelled input
jobs, no outstanding leases after shutdown, and no exceptions.

The low-SNR receiver returned **178 correct, nonduplicate MIBs**, covering 76
of 750 complete four-frame BCH opportunities. Compatibility returned four
correct, nonduplicate MIBs. Neither returned a wrong confirmed PCI or wrong
payload/frame/port/timing attribution in this run. Missing frame decodes are
**not** labelled feature-job drops: they combine unreceived RF input, repeated
reacquisition after genuine queue-loss gaps, and conditional decoding success.
No all-frame or complete-opportunity claim is supported by these runs.
The harness's all-frame decode diagnostic is stricter than a probability-of-
detecting-at-least-one-MIB claim. Acquisition and latency already fail
independently of that stricter diagnostic.

Low-SNR LTE processing p50/p99 was 11.75/386.41 ms per input block; compatibility
was 45.77/86.55 ms, versus 8.192 ms arrivals. Low-SNR timing has a mixture of cheap
buffering blocks and expensive searches. This is not uniform per-block service.
Both are substantially slower than the source-only control. The positive-control
test processes genuine 80-ms high-SNR IQ unpaced: eight exact MIBs with no loss.
Offline correctness therefore does not transfer to scheduled 2 MS/s performance.

## Fault injection and recovery

| Case | Outcome | Recovery |
|---|---|---|
| Required sink raises, 2 s at 10 kS/s | 5 sink errors; acquisition passes, delivery/aggregate fail; exit 1 | Closed, zero unfinished/unresolved/duplicate outcomes |
| Analysis raises, 2 s at 10 kS/s | 5 failed results; acquisition passes, delivery/aggregate fail; exit 1 | Closed, zero unfinished/unresolved/duplicate outcomes |
| 500-ms consumer pause, 3 s at 2 MS/s | 3,952,000 queue-lost samples; 447 immediate feature overloads; exit 1 | 500/500 terminal outcomes, queues drained, zero unresolved/duplicate outcomes |
| 500-ms required sink, 3 s at 10 kS/s | 30,000/30,000 samples; 8/8 terminal callbacks, no errors/overload/deadline misses; exit 0 | Shutdown drains in 4.16 s total elapsed, no unfinished tasks |

The lower-rate slow-sink pass is a recovery control, **not** a substituted target
or a 1/2 MS/s performance qualification. Immediate overload rejection and
accepted callback delivery are counted as separate observations of their
respective events, without duplicating terminal identities. These injected
failures do not disappear because cleanup succeeds.

## Correctness and regression execution

- Entire active source suite: **1,431 passed**, zero failed/errors/skipped,
  212.56 seconds. One deliberate overflow warning comes from computing the
  unsafe comparison in `test_large_complex64_no_power_overflow`, not from the
  production power calculation.
- Current independent acceptance scripts: **165 passed**, zero
  failed/errors/skipped, 14.88 seconds, explicitly routed using
  `SIGNAL_PROCESSING_SRC` to this working folder.

These suites overlap; they are not 1,596 unique tests. The source suite includes
previous A/B/E/F/G/H regressions, exact pre-trigger/raw-byte retention across
wrap/gap/EOF, nonzero/large origins, adaptive chronology, fault-output recovery,
metadata/timestamp atomic rejection and retry, process-backend death/reset,
shutdown deadlines/reentrancy, feature-policy provenance and source lifecycle.
The numerical assertions are case evidence, not a proof for arbitrary RF input.
