# Acquisition repair results — 19 October 2026

**Status: BLOCKED for lossless real-time release.** Targeted production fixes
are implemented in the active working folder, not just in an archive. Numerical
and lifecycle regression checks pass, but the scheduled full pipeline still
loses samples and/or overloads feature analysis. No FROZEN_PASS is claimed.

- [Diagnosis, implementation plan and next required work](ACQUISITION_LOSS_REPAIR_PLAN_20261019.md)
- [Execution contracts, Numba controls and usage](ACQUISITION_ACCELERATION.md)
- [Unmodified pre-repair measurements](PIPELINE_END_TO_END_AUDIT_RESULTS.md)

## Changes and attribution

| Finding | Production repair | Live evidence |
|---|---|---|
| 16,384-sample input disables the <=4,096 fast path and causes 512 hop calls/block | Successive bounded hop-aligned preparations, chronological commits | Profiled detection-chunk calls 16,384 → 647 over 32 blocks; pipeline time 1.708 → 0.151 s under profiling |
| Expensive exact BCH tail-biting work in Python/NumPy temporaries | Optional cached Numba, exact all-64-start-state trellis; contiguous destination-major one-thread layout | Median kernel 1,365.95 → 188.05 microseconds in 500 calls; real MIB and quantized/random tie checks agree with reference |
| Blind search repeatedly transforms unused retained history | FFT only required received regions per repetition | Full-history FFT and independent direct dot/energy oracles agree at original tolerances for 1/4/8 repeats |
| Repeated immutable policy reconstruction per feature result | 32-entry full-content policy cache | Same-ID edited policy rejected; no event/vector/IQ/quality reuse |
| Windows kernel readiness scans on every feature transaction | Owned-pipe aligned completion token under explicit architecture/capacity guards; polling fallback retained | Process, death, timeout/retry, reset, blocked delivery and provenance tests pass |

Profiling is for attribution, not real-time acceptance. Numba compilation is
completed before acquisition and startup is recorded separately. No thresholds,
supported-PFA assumptions, CRC hypotheses, feature families, schema or queues
were reduced to gain throughput.

## Actual scheduled results

All measurements below are **one 30-second run per listed configuration**,
sequentially on this Windows 11 / i5-1235U host, without pytest/tracing running
alongside timing. They are not three passing qualification repeats. Queue
capacity stays 16. The event-dense generic waveform retains 256-sample bursts
every 4,096 samples; LTE uses the same independently encoded finite file before
and after. See the baseline report for exact source preparation/hash details.

| Path/configuration | Scheduled samples | Consumed | Acquisition | Expiry | Acquisition queue loss | Analysis outcome |
|---|---:|---:|---:|---:|---:|---|
| Pre-repair direct 2 MS/s, N64/H32, 16,384/block | 60,000,000 | 9,437,184 | 15.7286% | 0 | 50,562,816 | 1,442 overloads / 2,304 captures |
| Repaired direct 2 MS/s, same configuration | 60,000,000 | 59,279,104 | **98.7985%** | 196,608 | 524,288 | **6,820 overloads / 14,479 captures**; 7,659 accepted terminal callbacks |
| Repaired isolated capture, 2 MS/s, N64/H32, completion-order features | 60,000,000 | 41,254,912 | 68.7582% | 0 | 18,745,088 | 10,077/10,077 captures analysed OK; backpressure loses input instead |
| Pre-repair adaptive 1 MS/s, N256/H64, pre-trigger256, four feature processes | 30,000,000 | 25,654,144 | 85.5138% | 616,448 | 3,729,408 | 5,396 OK, 898 quality-skipped |
| Repaired adaptive 1 MS/s, same RF settings, completion-order features | 30,000,000 | 26,607,616 | **88.6921%** | 361,472 | 3,030,912 | 5,921 OK, 642 quality-skipped; zero feature overload |
| Pre-repair low-SNR LTE/PBCH, 2 MS/s, 16,384/block | 60,000,000 | 8,568,832 | 14.2814% | 32,768 | 51,398,400 | 178 verified MIBs |
| Repaired low-SNR LTE/PBCH, bounded FFT + one native thread | 60,000,000 | 25,919,488 | **43.1991%** | 0 | 34,080,512 | 669 verified MIBs |

The two generic configurations have different detector windows/reference modes;
do not treat them as controlled sample-rate scaling. Completion ordering and
transport/cache changes are a combined treatment, not an isolated speedup claim
for one line of code. Available memory varied approximately 0.8–1.7 GB of 8.25 GB;
normal host activity/power state was not held constant. No new affinity, priority
or Python switch-interval tuning was applied. The pre-existing isolated benchmark
sets an unset `OPENBLAS_NUM_THREADS` to 1; direct/LTE harnesses do not do so.

The final direct 2-MS/s run had 2,692 deadline misses; block p50/p99 was
6.22/16.98 ms versus 8.192-ms arrivals, and maximum backlog was 237.69 ms.
The adaptive 1-MS/s run missed 25,984 deadlines. LTE's block p50/p99 was
11.66/158.63 ms; latency p99/max was 653.04/705.20 ms. All strict timing gates
failed and commands returned **1**. A >90% IQ statistic alone is not a complete
pipeline pass while feature jobs are overloaded.

All final listed runs had balanced sample/terminal accounting and recovered:
zero unresolved or duplicate terminal identities, zero unexpected failed
analysis/sink/generation-discard outcomes, and zero unfinished work at closure.
An immediate overload or quality skip is an accounted terminal outcome, **not**
successful feature extraction. All 669 returned LTE MIBs matched payload,
PCI/ports/frame/timing; no wrong/duplicate MIB was reported. Missing frame CRC
results are not called feature-job drops and cannot be used as an independent
Pd estimate while input is being lost.

Post-repair failure injection also exercised the real process pool: at 10 kS/s,
10,000/10,000 samples and 3/3 actual analyses completed, but the required sink
raised three times. Both acceptance flags failed and CLI exit was **1**, with
zero duplicate/unresolved outcomes and successful zero-unfinished cleanup.
See `failing_sink_final.json`; delivery failures are not hidden by balanced IQ.

Intermediate measurements remain available, including initial chunk-only
99.7542% acquisition with 7,636 feature overloads, isolated submission-order
62.7507%, native-before-bounded-search 16.2748%, and bounded-search/four-thread
33.0957%. They are not substituted for the final results or hidden.

## Numba / multithreading answer

Before repair, Numba 0.67.0 was installed but unused. It is now an optional
**actual native BCH backend**, not a dependency-name-only claim. Default is one
`nogil` native thread because its contiguous kernel is faster for this small
job. Explicit 2–4 threads use `parallel=True/prange`; the observed layer is
OpenMP. Masks are restored and concurrent callers are tested. There is **no
automatic queue-aware thread-count scaling** or TBB dynamic chunk scheduling.

The generic ten-family feature worker remains bounded Python coordination and
persistent spawn processes. More Python threads cannot by themselves remove its
GIL/object/dispatch cost. The plan identifies the remaining larger service-cost
and coordination repair; these changes do not claim it has been implemented.

## Final functional and numerical gate

| Execution | Actual outcome |
|---|---|
| Complete active source suite | **1,454 passed**, zero failures/errors/skips, 235.47 s; one deliberate overflow warning in the unsafe arithmetic comparison test |
| Current independent acceptance suite | **165 passed**, zero failures/errors/skips, 13.47 s |
| Fresh-wheel isolated tests, including new native/large-block/process tests | **213 passed**, zero failures/errors/skips, 105.53 s |
| Periodogram / Welch / direct autocorrelation | 72 / 64 / 10 comparisons pass |
| Randomized ring wrap/oversize/read/snapshot oracle | 600 cases pass |
| Detector direct powers / absolute windows | 425 logical windows agree over each of seven partitions |
| Ten feature families | 48 scalar, 24 general joint-cumulant, two exact bandwidth comparisons pass; 56-key A/B/A isolation passes |
| Streamed vs batch empirical calibration | 10,873 windows, maximum relative difference 2.10e-14; training-only support remains False |
| Configured SOS continuity | 187 windows per partition, maximum dB difference 0 across four partitions |
| EOF / reset / replay | Exact IQ and terminal behavior pass |

Suite counts overlap and are **not** added as unique tests. Original numerical
tolerances are unchanged. The numerical scripts are current equivalents with
training-only support False, not a claim that an obsolete frozen support assertion
was repaired by changing its mathematics. No skip/xfail hides a live failure.
All original source/evidence files and prior observations remain in place.

## Previous-issue regression mapping after combined edits

Each functional row below passed in the final source gate, not just before
later production edits. Timing is explicitly separate.

| Area | Concrete current checks / retained mechanism | Outcome |
|---|---|---|
| H01 | `tests/regressions/test_h_metadata.py`: exact atomic metadata rejection, residuals, corrected retry | Pass |
| G01 / F07 | `test_g_revision.py`, `test_f_revision.py`: physical SNR, zero-excess validity, paired power scaling | Pass |
| G02 / G03 | G revisions, acquisition-stage2 and new large-block tests; direct power and calibration oracles | Pass |
| G04 / E08 | G revisions, process-analysis, acquisition-stage2: synchronous admission, timeout/retry, reset/reentrant sinks, backend death | Pass |
| G05 / G07 | G revisions and H metadata: frozen warmup, original transport/timestamp anchors, atomic retry | Pass |
| G06 / G08 | G revisions, process-analysis and new native/transport tests: attribution, canonical policy, immediate overload, reservations | Pass |
| F01 / F02 / E05 | F revisions and new later-chunk failure: gap-surviving jobs, typed error, exact bounded completed outputs/horizons, no duplicate flush | Pass |
| F03 / F04 / E11 | F revisions: held-out support binding, strict fixed statistic/threshold/domain/actual-N applicability | Pass |
| F05 / E02 / E03 | F revisions, sample-retention and new large-block tests: direct continuity, absolute origins, exact EOF/gap/pre-trigger IQ | Pass |
| E01 / E04 / E12 | Examples2 repairs, acquisition-stage2 and buffer tests: bounded state, pending/frozen age, atomic coordinates/finalize | Pass |
| E06 / E07 | Detection/regression suites: configured Gaussian bound, domain/Fs/gain, settling/budgets, actual-N cases | Pass |
| E09 / E10 / E13 | Bandwidth/aggregation/quality suites and F/G revisions: coverage semantics, immutable identity, structured quality | Pass |
| F06 / E14 | Benchmark regression checks: sample/result identities, failures affect acceptance, matching and recovery | Accounting passes; **scheduled target fails** |
| F08 | Fresh isolated wheel and 117 module hashes; genuine checkout checks retained | Pass for this verification build, not a frozen source-release claim |
| Available historical A/B/E | Entire `tests/regressions` plus current routed `checks`; original positive numerical oracles | Pass for available current cases; no invented expectations for absent historical artifacts |

## Memory and resources

Fresh traced diagnostic, not a timing run: 4,096,000 quiet native-rate samples,
checkpoints after 409,600 / 2,048,000 / 4,096,000. Current traced allocation was
5,093,916 / 5,095,461 / 5,096,942 bytes: **3,026-byte growth** after caches settled,
not recording-sized IQ retention. RSS was 177,860,608 bytes at all checkpoints.
The instrument responded to a touched 128-MiB control: 105,275,392 → 239,497,216
bytes, so these RSS readings pass the responsiveness check. Then 64 genuine
dense MIBs decoded; reset left zero broadcast samples/pending frames/BCH frames.

Generic runs exercised thousands of real analyses beyond the 128-result-history
cap; histories remained capped, pending IQ/noise histories were zero after EOF,
and queues/jobs drained. New policy state is capped at 32 configurations; native
and FFT caches are bounded. This is finite retention evidence, not an indefinite
hardware or ten-minute lossless soak qualification.

## Distribution and reproducibility

Verified wheel (development candidate, **not** a lossless release):

`output/acquisition_fix_20261019/wheels/signal_processing-2.4.0rc1-py3-none-any.whl`

SHA-256: `22f744dffd40063ad3d3bccbb86a98ab7f0efe92ddeceef340647fd5fac51a22`.
Built from a separate `build_source/` copy containing only current production
Python files and build metadata, no stale `build/lib` or native cache artifacts.
Installed into `output/acquisition_fix_20261019/installed/` with `--no-deps`;
`-I` verification checks actual import origins and byte hashes of **117 modules**.
The wheel is pure Python with an optional external Numba/LLVM dependency; it is
not represented as a precompiled platform-specific native extension.

Dependency observations: `requirements-acquisition-observed.txt`. Source hashes:
`output/acquisition_fix_20261019/production_manifest.csv`. Logs, all intermediate
failures, profiles and JSON observations are under that output directory.
The initial eight new-test failures were a nonexistent helper-field access
(`NoiseEstimate.noise_power_linear`), corrected to the actual contract. A
profiling attempt also hit Python 3.14's one-cProfile-monitoring-tool limit;
the harness was corrected. Those logs are retained, not labelled production
defects or successful regressions. Frozen production regression assertions were
not weakened.

Example commands (choose new output paths to preserve observations):

```powershell
$env:SIGNAL_PROCESSING_SRC = (Get-Location).Path
py -3.14 -m pytest -q tests --junitxml=output/acquisition_fix_20261019/source_final.xml
py -3.14 -m pytest -q checks --junitxml=output/acquisition_fix_20261019/acceptance_final.xml
py -3.14 output/acquisition_fix_20261019/numerical/signal_processing2_positive_validation.py
py -3.14 output/acquisition_fix_20261019/numerical/signal_processing2_feature_oracles.py
py -3.14 output/acquisition_fix_20261019/numerical/signal_processing2_filtered_and_lifecycle.py
py -3.14 -I benchmarks/verify_lte_install.py output/acquisition_fix_20261019/installed .
py -3.14 -m benchmarks.benchmark_bch_native --iterations 500
py -3.14 -m benchmarks.audit_pipeline_runtime generic --seconds 30 --output output/acquisition_fix_20261019/generic_2ms_final.json
py -3.14 -m benchmarks.stream_isolated --seconds 30 --adaptive --profile positive_snr --feature-processes 4 --result-order completion --output output/acquisition_fix_20261019/adaptive_1ms_final.json
py -3.14 -m benchmarks.audit_pipeline_runtime lte --input output/pipeline_audit_20261019/lte_30s.cf32 --profile low_snr --bch-backend numba --numba-threads 1 --output output/acquisition_fix_20261019/lte_final_simd_1.json
py -3.14 -m benchmarks.check_lte_broadcast_retention --profile low_snr --output output/acquisition_fix_20261019/retention_final.json
py -3.14 -m benchmarks.stream_isolated --seconds 1 --sample-rate 10000 --block-size 1024 --feature-processes 1 --fail-sink --output output/acquisition_fix_20261019/failing_sink_final.json
```

Tests/numerical/memory/build/install commands completed with exit 0. The scheduled
commands listed above returned 1, as required by their failing acceptance scope.
No Git commit/tag/push or frozen-release ZIP was produced. A ten-minute lossless
soak and three passing qualification repeats are **not** claimed after these
failed gates. Hardware/LTE-data/classification and arbitrary-noise guarantees
remain unverified/out of scope.
