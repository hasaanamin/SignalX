# Native-rate LTE and acquisition-overhead repairs

20 October 2026. Working candidate **2.4.0rc1 — BLOCKED, not frozen**.
Primary host: Windows 11 build 26200, Core i5-1235U, 10 physical/12 logical CPUs.
Python 3.14.7 / NumPy 2.5.2 / SciPy 1.18.0 / Numba 0.67.0. Dependencies were not upgraded.

## Implemented production changes

- **Explicit native LTE profile:** `LTESyncConfig.native(low_snr=True)` selects genuine 1.92 MS/s input. `LTEChannel` uses its identity path: no resampling filter. The actual source metadata must match. The existing 2 MS/s conversion remains available; no recording is silently relabelled. Native profile identity now truthfully identifies the identity preparation path; the previous 2 MS/s profile identity is unchanged.
- **Fixed-capacity mirrored histories:** synchronization and broadcast storage write each new sample twice, allowing contiguous borrowed views across wrap. They no longer concatenate/copy the whole retained history at each search horizon. Capacity covers the existing retention, search horizon and bounded input preparation. Reset reuses storage. Borrowed views stay inside the ordered receiver; public results never reference recycled IQ.
- **Bounded immutable-configuration work:** cache profile hashes and at most 32 CFO template pairs per receiver; precompute fixed smoothing denominators, conjugate templates and SSS index permutations. Caches contain no received IQ, signal correlations, feature results or truth labels. Existing statistics, search hypotheses, filtering and thresholds are unchanged.
- **Exact BCH bound pruning:** free-start Viterbi supplies an upper bound for every possible closed-path end state. One feasible closed path supplies a lower bound. Only starts with upper bound strictly below that feasible score are omitted. All ties remain eligible and original predecessor/first-argmax rules remain. When more than 32 states survive, the previous full 64-start SIMD kernel is used. This is a certified equivalence optimization, not approximate tail-biting or reduced antenna/quarter search. Explicit parallel 2–4-thread operation and the NumPy reference remain available; the default is still one native thread, not dynamic thread scaling.
- **Adaptive reference bookkeeping:** reuse an immutable reference already obtained immediately before an internal batch commit. Never reuse it across adaptation, commits or external callbacks. Causal admission/freeze/statistic/age boundaries are preserved.
- **Compact empty progress:** capture processes encode empty OPEN progress in a fixed 92-byte scalar record rather than nested pickled dictionaries. Captures, faults and EOF still use complete records. Per-block notification frequency, input sizes, capacities, acknowledgement ownership and public `get()` records are unchanged. `progress_protocol='pickle'` is the comparison path. Horizons outside signed-int64 encoding fall back to full Python-integer records rather than truncating or failing after commitment.

## Reproducible measurements

All unperturbed timing commands were sequential, without pytest/tracing running concurrently. These are finite software tests, not hardware or universal real-time guarantees. Host thermal/power/background state was not controlled tightly enough to infer small differences.

### LTE compute controls

Same actual native-rate fixture, same low-SNR receiver profile and exact returned results:

| Source revision/path | Two seconds of input: processing wall | Throughput | MIB / sync observations |
|---|---:|---:|---:|
| Prior source, explicitly imported from the preserved earlier build copy | 4.096 s | 0.937 MS/s | 199 / 397 |
| Mirrored histories/configuration optimization, before BCH bound | 2.287 s | 1.679 MS/s | 199 / 397 |
| Combined repairs including exact BCH bound | 1.609 s | 2.387 MS/s | 199 / 397 |

This is a same-file native comparison. Do not equate it with the older 2 MS/s scheduled workload: the native source rate is explicitly 4% lower and its independently generated waveform is different.

A stage-instrumented native control retained correct results, performed no resampling, and made no receiver/broadcast history concatenations. Its one-second input still needed about 0.950 s including timer overhead; remaining work includes tracking/refinement and exact PBCH hypotheses. Profiling is not acceptance timing.

### Scheduled native LTE: acquisition passed, aggregate failed

Thirty seconds, 1.92 MS/s, 16,384 samples/block, queue capacity 16, real synchronization/PBCH/MIB, low-SNR four-half-frame integration, encoded synthetic waveform at -5 dB, CFO 12,345 Hz:

- Scheduled, admitted and consumed: **57,600,000 samples (100%)**.
- Producer expiry, acquisition queue drops, failed/cancelled input jobs: **zero**; balanced accounting and clean closure.
- Correct MIBs: **2,903 / 3,000 frames**, with no incorrect or duplicate MIBs. All **750 / 750 BCH blocks** yielded at least one verified MIB. Frame CRC/decoding misses are not missing acquired IQ or silently discarded analysis jobs.
- End-to-end latency p50/p95/p99: **8.56 / 51.34 / 98.99 ms**, maximum **112.25 ms**. This **fails** the existing p99 ≤50 ms / maximum ≤100 ms limits.
- Receiver processing p50/p95/p99: **6.73 / 12.03 / 15.84 ms**, maximum **85.21 ms**.
- Aggregate enforcement exit: **1**, correctly, because latency and the benchmark's every-frame decoding requirement did not pass.

Only one corrected 30-second native schedule was run here. Three accepted runs and a ten-minute accepted soak have not been established. The larger-bandwidth LTE data receiver, real receiver noise and SDR acquisition remain unverified/downstream.

### Important evaluator correction, separately identified

The old evaluator called the independent `frame_fixture()` oracle for every MIB inside the timed consumer loop. A live measurement took 1.365 s for 1,000 such calls. This test work is not production LTE processing but consumed its real scheduling headroom.

The evaluator now constructs finite expected 24-bit payloads per BCH block **before starting the source**, reports that initialization cost, then still verifies every actual MIB's payload, identity, antenna ports, frame quarter and timing during acquisition. The source schedule, queue, latency gates, received samples and protocol checks are unchanged; no truth is passed into the receiver. `--scoring-backend inline` retains the original execution layout.

The combined production repairs with the old inline scoring acquired only **37.43%** in its recorded 30-second native run. That failure is preserved. The corrected run's 100% acquisition must not be attributed solely to production optimization, nor presented as an unchanged historical benchmark pass. The corrected run spent 1.153 s preparing validation payloads before acquisition and 0.102 s scoring during acquisition; total reported initialization was 2.201 s.

### Adaptive 1 MS/s: still unresolved

Unchanged target: N=256/H=64, M=1,024 adaptive statistic, 256-sample pre-trigger, ring 4,096, 1,024-sample source blocks, input queue 16, all ten feature families and four persistent feature processes:

- Final thirty-second run after the signed-int64 fallback guard: **27,310,976 / 30,000,000 samples (91.03659%)**.
- Producer expiry **288,768**, input queue loss **2,400,256**; balanced accounting. Ignoring expiry would not remove these input queue losses.
- **6,723 / 6,723** capture outcomes: 6,222 ok and 501 explicitly quality-skipped; no unresolved results, feature overload, analysis/sink errors or unfinished work; clean recovery.
- Processing p50/p99 **0.710 / 3.168 ms** versus a 1.024 ms block arrival interval. All 26,671 delivered blocks missed that end-to-end deadline. Capture-delivery p99/max were 30.93/45.70 ms.
- Existing candidate/strict enforcement both failed; exit **1**. The earlier four-process run in this implementation stage acquired **88.77056%**, with 351,232 expiry and 3,017,600 queue losses; that observation is preserved as `adaptive_compact_30s.json`. The pre-repair four-process run was 88.6921%. Variation across these runs is not evidence of a reliable adaptive throughput repair.

Additional real-feature ten-second controls: two feature processes acquired **92.35072%**, with 646,144 queue losses and 1.898 s capacity backpressure; three acquired **85.68448%**, with 1,289,216 queue losses. Both recovered and accounted for terminal results, but both failed the acquisition gate. These differing-duration controls are not proof of an optimal worker count, and the default was not silently changed to a favorable configuration.

Compact progress and reference reuse are implemented and regression-checked, but do not solve adaptive scheduling/contention. Next work remains reducing the adaptive hop/commit/object cost and parent result coordination under the actual concurrent workload, with service-time accounting that includes serialization and send. Larger queues or suppressing producer expiry would not resolve the measured input queue losses.

## Verification and artifacts

Fresh results under `output/native_acquisition_fix_20261020/`:

| Verification | Actual outcome |
|---|---|
| Complete source suite | **1,465 passed**, one deliberate arithmetic-warning fixture, no failures/errors/skips; 225.49 s |
| Current explicitly routed acceptance | **165 passed**, no failures/errors/skips; 13.93 s |
| Final focused checks after the coordinate fallback guard, including the added boundary case | **18 passed**; 33.09 s |
| Isolated installed-wheel suite | **230 passed**, no failures/errors/skips; 134.36 s |
| Independent numerical scripts | All three completed successfully, with live observations; original tolerances |
| Quiet/event-dense retention | Bounded ownership, responsive RSS control, 64 genuine dense-stream MIBs, empty reset state |
| Required failing-sink injection | 10,000/10,000 input samples, 3 terminal results and 3 reported sink errors; acceptance correctly false, clean recovery |
| Low-rate slow-sink recovery control | 30,000/30,000 samples at 10 kS/s over 3 s, 8/8 terminal outcomes, 0 unfinished/unresolved work, clean recovery with a 0.5 s sink; not a 1 MS/s throughput qualification |

Counts overlap and must not be added as unique tests. The full source run began before the final compact coordinate-boundary guard; its normal paths are unchanged. The final focused and installed runs exercise that final guard and the newly added boundary case. No production edit was made after installed verification.

The independent oracles rechecked 72 periodogram, 64 Welch, 10 autocorrelation, 600 ring and 425 detector-window comparisons across seven partitions; 48 scalar and 24 joint-cumulant comparisons; two exact bandwidth constructions; all ten feature families and 56-key A/B/A isolation. The streamed calibration fixture retained 10,873 actual observations (not a support threshold), maximum relative streamed/batch difference 2.73e-15, training-only support False. Filter continuity and exact EOF/reset replay also passed. These are fresh current-equivalent scripts, not a claim that an obsolete frozen supported-PFA assertion was changed into a passing original.

Memory: after legitimate caches/storage filled, quiet traced allocations changed from 3,638,242 to 3,643,310 bytes across 409,600 to 4,096,000 received samples (+5,068 bytes); RSS changed by one 4,096-byte page. Touching 128 MiB increased reported RSS by approximately 128 MiB, so the instrument was responsive. Fixed history capacities and template-cache bounds are independently checked. The traced run is not timing evidence.

The frozen/public regression coverage remains explicit:

| Earlier areas | Live coverage and outcome |
|---|---|
| H01 and G05/G07; metadata, freeze and timing admission | `test_h_metadata.py`, `test_g_revision.py`, current G checks, native rejected-metadata snapshots: passed |
| G01–G03/F07; physical SNR/power and partition stability | Source G regressions/current public checks plus direct numerical scripts: passed |
| G04/G06/G08 and F01/F02/F05; jobs, provenance, gaps/fault recovery, backpressure | Full source/current checks; installed lifecycle/capture tests; new real spawned progress parity/acknowledgement checks: passed |
| F03/F04 and E06/E07/E11; evidence/statistic/domain binding | Source F regressions, policy/quality tests and current F acceptance: passed; no support requirements relaxed |
| Available A/B/E storage, numerical, feature, event-isolation and lifecycle cases | Complete `tests` and current `checks`, independent oracles, bounded-retention run: passed for available current equivalents; no invented missing historical assertions |
| E14/F06 acquisition evaluation | Accounting, matching and explicit error reporting retained; native acquisition passed one run, but latency/every-frame and adaptive scheduling gates remain failed |
| F08 distribution | Corrected clean build, 118-module source/wheel/install parity and isolated imports: passed; failed staging attempt preserved separately |

An initial packaging-staging command used a path helper unavailable in this machine's PowerShell/.NET version. Pip produced an empty wheel, and the isolated import check correctly failed before any passing installation claim. That failed artifact is quarantined in `FAILED_empty_wheel_DO_NOT_USE`; its failed logs are preserved. Staging was corrected, checked for the expected module count and rebuilt in a separate clean directory.

Verified **candidate** wheel (not a frozen release):

`output/native_acquisition_fix_20261020/verified_wheels/signal_processing-2.4.0rc1-py3-none-any.whl`

SHA-256: `8ecca46f8012cbfae56d822566a2368f03fd44467d6cfb14f6362edd04e24450`.

All **118** production Python modules are byte-identical between active source, wheel and isolated installation. Wheel entry checks found no duplicate entries, stale build/lib tree, Python bytecode or Numba cache files. `production_manifest.csv`, `wheel_parity.log`, `verified_build.log`, `verified_install.log` and `verified_installed.log` identify the actual verified artifact. The wheel is pure Python; its optional native backend requires the pinned compatible Numba environment and completes compilation/readiness before acquisition.

The focused regressions cover direct-byte history oracles over wraps, bounded/reused storage, genuine native IQ without any resampler call, exact frozen-vector MIBs, strict metadata rejection, bounded template equivalence, 240 random/quantized/extreme-scale exact BCH comparisons, real spawned compact/pickle progress parity across gaps with adaptive and fixed noise, acknowledgements, and large-coordinate fallback. Existing suites retain EOF/gap/fault/pre-trigger/lifecycle and ten-family numerical checks.

Example usage:

```python
from signal_processing.core import StreamInfo
from signal_processing.lte import LTEBroadcastReceiver, LTESyncConfig

stream = StreamInfo('pluto-input', 'receiver', 1_920_000., center_frequency_hz=2.14e9)
receiver = LTEBroadcastReceiver(stream, LTESyncConfig.native(low_snr=True))
# Feed real positioned complex64 IQBlocks with this actual rate; flush at EOF.
```

This configures the software contract, not a hardware driver. Verify the radio's actual read-back sample rate before acquisition. This central-six-resource-block path provides FDD/normal-CP synchronization and PBCH/MIB, not complete wideband LTE demodulation.

Reproduction, repository root:

```powershell
py -3.14 -m benchmarks.audit_pipeline_runtime prepare --seconds 30 --sample-rate 1920000 --input output/new_native.cf32 --output output/new_native_fixture.json
py -3.14 -m benchmarks.audit_pipeline_runtime lte --sample-rate 1920000 --input output/new_native.cf32 --output output/new_native_schedule.json
py -3.14 -m benchmarks.stream_isolated --seconds 30 --adaptive --profile positive_snr --result-order completion --feature-processes 4 --output output/new_adaptive_schedule.json
$env:SIGNAL_PROCESSING_SRC = (Get-Location).Path
py -3.14 -m pytest -q tests
py -3.14 -m pytest -q checks
py -3.14 -I benchmarks/verify_lte_install.py output/native_acquisition_fix_20261020/verified_installed .
```

The fixture is independently generated offline and never supplied as a runtime answer. Its current 57,600,000-sample native file SHA-256 is `aa40248a5bf5b832c0bb5faba026fab264983cee3319f1296391822a28a8c1e3`. Saved JSON files are observations, not algorithm inputs. Old source, checks and observations were not edited.

**No FROZEN_PASS, release tag or production-readiness claim is warranted.** This is a tested implementation stage: native LTE acquisition improved substantially, while latency, every-frame decoding and adaptive acquisition remain explicit failing gates.
