# Adaptive acquisition and LTE scheduling repairs — working candidate

Primary host: Windows 11 build 26200, Core i5-1235U, Python 3.14.7.
Dependencies: `requirements-acquisition-observed.txt`. Version remains 2.4.0rc1.
This changes the active `signal_processing/` source, not an old uploaded ZIP.

## Status and measurement scope

**BLOCKED, not a frozen or universally lossless release.** Functional repairs and
mean-throughput improvements are not proof of scheduled lossless operation.
The fresh logs, failed experiments and observations are retained in
`output/scheduling_fix_20261020/`. Older evidence remains unchanged.

As requested, processing acceptance can exclude synthetic producer deadline
expiry. Its denominator is scheduled samples minus those expired samples;
acquisition-queue loss is NOT excluded. Delivery errors, missing/duplicate
outcomes and incomplete cleanup still fail. The original strict/candidate
gates remain available and unchanged. An actual SDR can still lose data through
device, USB, driver or application buffering; hardware operation is unverified.

## Production changes

- Full-band adaptive idle transactions use the existing exact fixed-M statistic
  on a detached preview. Every threshold uses only already authorized samples.
  A possible ON decision abandons the preview without committing it. Only a
  confirmed event-free prefix is committed together. Policies, filtering,
  expiry, custom estimators, rearm and unsupported window alignment fall back.
  The ordinary single-update path avoids unnecessary preview overhead.
- Short ON/OFF persistence prefixes share raw/detector commitment while retaining
  every logical decision and landmark. They cannot cross confirmation,
  termination, retention, expiry or cap horizons. Working mathematical features,
  threshold support and pre-trigger contracts are not replaced.
- Capture progress optionally groups at most four scalar observations per reply
  (`progress_group=4`). Captures, faults and EOF publish immediately. Every
  constituent block keeps its real coordinates, committed count and processing
  times. No IQ blocks or detector windows are merged; no input queue is enlarged.
  Default `progress_group=1` retains per-block compatibility. Independent input
  production is required for grouped telemetry. Parent receipt latency includes
  grouping delay; actual child completion latency is separately reported.
- An optional bounded shared scalar completion channel is available for
  comparison. Full capture/error records remain on the pipe; publication occurs
  before a large pipe write so the reader can drain it. Pipe remains the default:
  the shared channel alone did not solve overload.
- LTE Gold sequences advance the exact GF(2) recurrence in 31-bit words rather
  than thousands of scalar NumPy calls. Configuration-only FFT preparation occurs
  before input starts. FFT worker count is explicit; one remains the default.
- Default serial-native PBCH processing batches all existing port/quarter
  hypotheses into one Numba call, preserving chronological rate recovery,
  normalization, exact tail-biting decoding, CRC masks and ambiguity rejection.
  Closed free-start survivor paths strengthen the rigorous pruning bound;
  unsupported approximations/omitted hypotheses are not used. Reference and
  explicitly threaded implementations remain available.
- CP pairs are gathered from the same at-most-20 complete received slots and
  promoted before products. Batched FFT and fixed configuration arrays avoid
  repeated per-symbol/per-frame allocation/dispatch. No future unreceived IQ,
  synthetic truth or previous event's feature cache is used.

## Observed limits and rejected approaches

Final grouped-progress adaptive run (same 30 seconds, 1 MS/s, 1,024-sample
input, N=256/H=64, M=1,024, ring 4,096, input capacity 16 and four real feature
processes): **28,064,640/29,666,176 source-eligible samples = 94.6015%**.
Queue loss was **1,601,536**; producer expiry **333,824** is excluded from that
fraction and remains separately recorded. There were **6,900/6,900** terminal
capture outcomes: **6,546 ok, 354 quality-skipped**, no failed/overloaded,
unresolved/duplicate outcomes or unfinished work; recovery succeeded. Skips are
not successful feature calculations. Processing-scope enforcement correctly
returned **exit 1** because queue loss remains. See `adaptive_grouped_30s.json`.

True per-block child-commit latency p50/p99/max was **13.686/27.811/37.699 ms**;
block-processing p50/p99/max **0.778/2.893/6.284 ms**. Terminal-result latency
p99/max was **42.598/55.814 ms**. Input rate is not met losslessly by this
configuration. The small improvement over the preceding run is not a controlled
cross-host speed guarantee or a reason to hide losses.

Before grouped telemetry, the final automatic-placement adaptive run consumed
27,407,232 of 29,643,648 source-eligible samples: **92.4557%**, with 2,236,416
queue-dropped samples. All **6,732** emitted captures had terminal outcomes;
delivery/recovery passed with no unfinished work. Synthetic expiry was 356,352,
reported separately. See `adaptive_final_auto_30s.json`.

Mixed processor placement plus completion-order delivery reached **99.3089%**
source-eligible processing in one different configured run, but still dropped
202,752 samples at the acquisition queue. Other affinity/QoS layouts were worse.
These are configuration-specific observations, not a repeatable lossless claim.
Pinned efficiency-core workers exhibited substantial feature backpressure.
Default ordering/affinity are not silently changed to a favorable experiment.

LTE's contiguous one-second diagnostic improved from 0.97484 s to **0.80688 s**,
returning the same **100 MIB / 197 synchronization** observations. It is not a
scheduled acceptance result. In the final automatic-placement 30-second run,
57,567,232/57,600,000 samples were consumed (**99.9431%**), with 32,768 queue
losses; p99/max latency **128.524/302.942 ms** failed the unchanged 50/100-ms gate.
CRC/frame misses are distinct from sample loss. See `lte_final_auto_30s.json`.

An earlier batched run consumed every sample, but p99 56.539 ms still failed.
Parallel FFT batches, explicit core pinning and reusable FFT workspaces did not
establish a latency fix. Workspace reuse was removed; single-worker FFT remains
the default. Failed observations are retained, not replaced with a passing run.
Rare backlog/drop episodes cause expensive reacquisition and amplify delay.

## Reproduction

Use a new output name: benchmark tools refuse to overwrite prior observations.

```powershell
py -3.14 -m benchmarks.stream_isolated --adaptive --profile positive_snr --seconds 30 --result-order completion --progress-group 4 --acceptance-scope processing --output output/new_adaptive.json
py -3.14 -m benchmarks.audit_pipeline_runtime lte --profile low_snr --sample-rate 1920000 --input output/native_acquisition_fix_20261020/lte_native_30s.cf32 --output output/new_lte.json
py -3.14 -m pytest -q tests
$env:SIGNAL_PROCESSING_SRC = (Get-Location).Path
py -3.14 -m pytest -q checks
```

The LTE file is an external generated test artifact, not included as a runtime
answer. If absent, generate it with the `prepare --seconds 30 --sample-rate
1920000` mode of `benchmarks.audit_pipeline_runtime`. The receiver receives real
positioned IQ, not the evaluator's truth labels. Native 1.92 MS/s must be genuine;
do not relabel 2 MS/s IQ. The 2 MS/s converter path remains supported separately.

## Regression contracts

Final combined source verification: **1,497 passed**, one deliberate arithmetic
warning, no failures/errors/skips, 269.57 s. Current explicitly routed acceptance:
**165 passed**, 13.05 s. These overlap. The final source suite includes F-series
83, G-series 123, H metadata 30 and new scheduling 31 cases; these are subsets,
not additional unique tests. Installed verification is recorded separately below.

All three independent current-equivalent scripts completed against active
source: periodogram/Welch, direct autocorrelation, randomized ring/window,
scalar/joint-cumulant, bandwidth and A/B/A oracles, plus streamed calibration,
filter continuity and EOF/reset. Original frozen scripts remain in `checks/`;
the documented training-only support assertion migration is not called an
unchanged original-script pass. Fresh copies/observations are under
`qualification/numerical/`. Reproduce with `SIGNAL_PROCESSING_SRC` explicit.

The separate traced probe processed **4,096,000** quiet LTE samples with bounded
29,100/62,400-sample histories. Current traced allocation grew **6,841 bytes**
after the first checkpoint; RSS settled within **12,288 bytes**. The touched
128-MiB control increased RSS by 134,221,824 bytes, so that instrument responded.
All **64** dense-stream MIBs returned and reset left zero retained IQ/soft frames.
These timings are traced diagnostics, not unperturbed throughput measurements.

At 10 kS/s, the failing-sink injection consumed all 10,000 samples and recorded
**3 real sink errors**, enforcement exit **1**, clean recovery and zero unfinished
or unresolved work. A separate 0.1-s slow-sink control consumed all 30,000 samples
and delivered **8/8** outcomes with processing-scope exit **0**. Neither is a
1 MS/s acquisition pass.

Fresh isolated wheel verification: **497 passed**, 183.54 s, including the current
F/G/H and new scheduling cases. All **119** production modules match active
source, wheel entries and isolated installation byte-for-byte; actual loaded
origins are asserted outside the checkout. No duplicate entries, stale build/lib,
Python bytecode or Numba cache binaries are in the wheel. Wheel SHA-256:
`0e12d300a8d893492482c505ea637fb22c77055882aeb2020c0680cabbac46fd`.

The functional/build runner exited 0. This is not the scheduled-performance
gate: both final full-rate benchmark commands correctly exited 1. No production
edit followed combined verification. Clean source, candidate wheel, evidence
and checksums are generated by `benchmarks/package_scheduling_candidate.py`
from this manifest, not from a historical nested build tree.

### Previous-issue gate after combined edits

The available A/B cases in `test_audit_repairs_a01_a20.py` and
`test_audit_repairs_b01_b14.py` passed, alongside the C/D/reaudit/current checks
actually present in tests/checks. No expectation is invented for an absent case.
F01–F08, G01–G08 and H01 retain the explicit contracts checked by the public
source/installed suites. Here `E` names refer to
`tests/regressions/test_examples2_repairs.py`, augmented by F/G/H and new tests.

| Earlier area | Concrete live evidence | Outcome |
|---|---|---|
| E01 | `test_quiet_state_is_bounded_over_many_ring_wraps`, finite histories/results, new retention probe | PASS for exercised bounds |
| E02 | `test_absolute_admission_translation_and_replay`, new exact snapshot/capture tests at four origins | PASS |
| E03 | `test_gap_evaluates_complete_old_window_without_splicing`, chronological EOF and grouped tail tests | PASS |
| E04 | `test_startup_expiry_is_live_without_training`, adaptive frozen/pending age and causal snapshot comparisons | PASS |
| E05 | `test_midblock_fault_exposes_consumed_prefix_and_requires_reset`, F02 completed-output recovery, H01 retry | PASS |
| E06 | `test_configured_window_sizes_and_single_sample`, existing conservative configured-model tests | PASS tested configurations; no universal noise claim |
| E07 | Domain/scale/actual-clock/settling/budget tests in E, F04/F05 and H01 | PASS |
| E08 | Reentrant reset/epoch tests, G04 drain, real process death/timeout/overload/EOF checks | PASS |
| E09 | `test_heavy_tails_preserve_observed_width_but_require_physical_support`, independent bandwidth formulas | PASS |
| E10 | Effective-control identities, immutable canonical round trips, backend attribution | PASS |
| E11 | Training-only unsupported policy, held-out bound/evidence binding and strict applied-threshold tests | PASS under declared assumptions |
| E12 | Atomic invalid coordinates, recoverable failed finalization, constant-size warmup statistics | PASS |
| E13 | Structured envelope loss/calibration/epoch quality and G07/H01 timing trust | PASS |
| E14 | One-to-one matching, extra detections, explicit overload/failure and recovery accounting | Functional PASS; scheduled acquisition/latency FAIL |

Reproduction after extracting the paired source/evidence artifacts:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File benchmarks/qualify_scheduling_candidate.ps1 -OutputRoot output/new_qualification -NumericalRoot qualification/numerical
py -3.14 -I benchmarks/verify_lte_install.py output/new_qualification/installed .
py -3.14 benchmarks/package_scheduling_candidate.py --verification output/new_qualification
```

The packaging tool deliberately labels this a **BLOCKED candidate**, not a
qualified final release. Evidence ZIP paths are `observations/` for stage/timing
records and `qualification/` for final executed checks. Historical build copies
and raw datasets are not bundled as current source. See `FINAL_RELEASE_LOCK.md`.

`tests/regressions/test_scheduling_repairs.py` tests independent bit recurrence,
exact adaptive/reference snapshots and capture bytes at 0/1/1000/10^12 origins,
awkward statistic boundaries, real spawn/large records, EOF/gaps, child death,
grouped progress parity, all encoded port masks/quarters, and direct physical
CP/FFT oracles across representable scales. Existing G/F/H, numerical and
retention checks must remain passing; counts overlap across suites.

This stage does not establish lossless SDR acquisition, a hard real-time OS
guarantee, complete LTE data-channel demodulation, hardware validation or a frozen
foundation. Remaining queue/tail-latency failures are release blockers, not
hidden by successful terminal-result accounting or by excluding source expiry.

## Remaining limiting stages

The adaptive ingress path still spends serial Python time constructing and
committing decision/capture state around transition windows. Final per-block
processing p99 exceeds the 1.024-ms arrival interval. Feature coordination also
becomes limiting under some placements: mixed pinned runs waited 5–10 seconds
for bounded feature capacity, whereas the final automatic/grouped run waited
only 0.181 seconds. It would be incorrect to attribute every current queue drop
to feature math or every producer wall-time interval to RNG CPU.

LTE startup and gap reacquisition remain bursty. The final maximum receiver-call
time was **142.825 ms**, while 16 queued 16,384-sample blocks at 1.92 MS/s cover
about **136.5 ms**. A sufficiently long pause/backlog exceeds that budget, and a
gap then forces additional blind search. Current measurements do not isolate
all long wall-time spikes into DSP CPU versus OS/preemption/page-fault time.
Resampling is absent in the measured native path, so removing it is not a
remaining explanation or a complete latency fix. No larger queue, lower rate,
reduced hypothesis set, feature stub or changed acceptance limit was used.
