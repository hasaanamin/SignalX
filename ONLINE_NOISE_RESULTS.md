# Online noise implementation results — 13 October 2026

Status: implemented streaming candidate, **NOT real-time-qualified or frozen**.
Subsequent default-only change: at the user's request, the detector UI/CLI now
defaults to `ar1_spectral`, and the HDF5 tab again defaults to seeded adaptive
noise. Online modes remain opt-in. The implementation-time results and wheel
below precede that default restoration; they are not a newly rebuilt release.
This is work in the active `signal_processing` folder. Legacy archives and frozen
checks were not changed. No GitHub push or release publication was performed.

## Production changes

- `detection/online_noise.py`: bounded blind Gaussian/complex-AR1 frame estimator,
  robust startup checks, gated updates of mean/correlation/innovation power,
  sample-clock smoothing/expiry, immutable snapshots and model/config identities.
- `runtime/pipeline.py`: explicit `online_noise=OnlineNoiseConfig(...)`, causal
  post-decision updates, no externally designated warmup, gap/reset relearning,
  existing typed fault/recovery and exact raw/pre-trigger capture preserved.
- Online detector re-whitens raw N+1 windows under one model revision. Physical
  ON/OFF thresholds follow the innovation estimate; no fixed Pfa certificate is
  reused. It currently uses a single full-band window, not the spectral bank.
- Detector UI/CLI defaults to `online_ar1`; `online_gaussian` also available.
  Online execution ignores prefix/SNR/ideal-noise-power calibration labels and
  scores all received samples, including startup. Gray shading shows learning.
- HDF5 UI has a blind-online option. Its synthetic stream-building/normalization
  is still an evaluation harness, not hardware input; generator reference power
  is no longer supplied to the estimator when online mode is selected.

See [public contracts and limitations](ONLINE_NOISE_CONTRACT.md). Restart an
already-running UI to load the new implementation. Legacy programmatic defaults
remain available for reproducibility.

## Fresh verification

Environment: Windows 11 / Core i5-1235U; Python 3.14.7, NumPy 2.5.2, SciPy 1.18.0.

| Check | Actual final result |
|---|---|
| Combined supplied source suite | 1245 passed, one intentional arithmetic warning; exit 0 |
| Prior/current independent acceptance | 165 passed; exit 0 |
| Focused online + dataset/HDF5 tests | 47 passed; exit 0 |
| Installed online/AR1/UI + prior acceptance | 227 passed; exit 0 |
| Installed import/module verification | All imported production modules under isolated installed target; 101 source modules byte-identical |
| Independent numerical scripts | Positive validation, feature oracles, filter/lifecycle scripts all completed; exit 0 |
| Hidden Tk control test | Both online profiles selectable, N/H controls correct; exit 0 |
| Ruff F checks / diff whitespace | Passed |

Counts overlap: do not add them as unique tests. The 29 new online tests cover
Gaussian/real and complex AR1, powers and thresholds, direct-sum window oracles,
one-sample/irregular/oversize partitioning, origin 10^12, scale changes, freeze,
zero/tonal uncertainty, contaminated startup, raw bytes/pre-trigger retention,
causality, gaps, metadata atomicity, reset, expiry, bounded storage and policy
rejection. Additional UI tests remove prefix annotations/change labelled noise
power/SNR, and forbid constructing seeded estimators in HDF5 online mode.

Initial new-test failures were test-helper mistakes (calling a method as a
property) and assumptions about the subsequently configurable update cadence;
the original per-frame cadence assertions remain exercised with update_frames=1,
and a separate test verifies the default four-frame schedule.
Two isolated-install harness attempts had missing sibling probe/benchmark imports
and an overbroad module-name prefix check. Those logs remain in output. Harness
routing was corrected without changing acceptance assertions; final run completed
227 tests and verified actual installed production origins.

## Changing-noise characterization

`python -m benchmarks.evaluate_online_noise` executes four 262144-sample streams:
Gaussian fixed/drifting and AR1 fixed/drifting, 21 truths/cell, including one
startup burst. Post-startup bursts are 512 samples at +10 dB. This experiment uses
a 5-ms tracking time constant (default is 20 ms), N/H=256/64 and real synchronous
features. No truth, reference power or rho is passed to the pipeline.

| Case | Matches / truths | Extras | Final innovation estimate | Final generating value |
|---|---:|---:|---:|---:|
| Fixed Gaussian | 20/21 | 0 | 9.9413e-5 | 1.0000e-4 |
| Gaussian power ramp | 20/21 | 0 | 1.9547e-4 | approximately 2.0000e-4 |
| Fixed AR1 rho=.8 | 20/21 | 0 | 3.5790e-5 | 3.6000e-5 |
| AR1 rho .2→.7 and power ramp | 20/21 | 0 | 1.0449e-4 | approximately 1.0200e-4 |

The four startup misses remain counted: total 80/84, not 84/84. All 80 post-startup
bursts matched and generated actual 56-feature results; capture IQ matched exact
source slices. Final rho was approximately .7968 for fixed .8 and .6772 for the
ramp ending at .7, showing expected tracking lag. Candidate-fit residual checks
do not certify the smoothed model's instantaneous distribution under drift.

Unpaced pipeline+feature rates were approximately 0.642, 0.342, 0.283 and 0.296
MS/s respectively. These finite, host-dependent runs are not independently paced
acquisition tests and do not establish 1 MS/s performance. Profiling identified
repeated model fitting and per-hop Python processing as costs; fitting was reduced
from every frame to every four frames with sample-clock smoothing adjusted.

## Actual dataset characterization

`python -m benchmarks.evaluate_online_dataset` selected repetition zero of all
eight scenarios, both noise groups, and 0/+5/+10 dB: 48 records. The SAME online
AR1 profile is used for white and AR1 data; labels select/score cells only.

| Noise / SNR | Matches / truths | Extras |
|---|---:|---:|
| White / 0 dB | 11/12 | 0 |
| White / +5 dB | 11/12 | 1 |
| White / +10 dB | 11/12 | 1 |
| AR1 / 0 dB | 8/12 | 13 |
| AR1 / +5 dB | 11/12 | 1 |
| AR1 / +10 dB | 11/12 | 1 |
| Total | **63/72 (87.5%)** | **17** |

All 48 completed, no analysis failures, all captured IQ exact. These results do
not satisfy a >90% general detection requirement. Short-burst sensitivity and
fragmentation remain weaker than the legacy fixed-calibration spectral bank.
This dataset run preceded only the additive first-learning-horizon diagnostic;
no detection calculations changed afterward. Final combined/installed tests
followed all production edits. No full 480-record online qualification is claimed.

## Traced bounded-storage check

`python -m benchmarks.check_online_retention` processed 2097152 samples.

| Received | Traced current bytes | Traced peak bytes | Estimator retained samples | Detector residual | Unfinished jobs |
|---:|---:|---:|---:|---:|---:|
| 524288 | 156998 | 1097734 | 8192 | 256 | 0 |
| 1048576 | 158769 | 1105970 | 8576 | 256 | 0 |
| 2097152 | 160265 | 1107498 | 8576 | 256 | 0 |

Pending-noise/history counters stayed zero. This is traced allocation/ownership
evidence, not RSS or unperturbed timing evidence. Diagnostics snapshots themselves
account for some small checkpoint growth. Existing worker/result bounds remain.

## Evidence and verification wheel

Fresh logs/observations: [output/online_noise](output/online_noise/), including
`supplied_final.log`, `acceptance.log`, `installed_verified.log`, numerical logs,
`changing_noise_final.json`, `dataset.jsonl`, `retention.json`, and `ui.log`.

Isolated build copied active package, README and pyproject into
`output/online_noise/build_input`, ran `pip wheel --no-deps --no-build-isolation`,
then `pip install --no-deps --target output/online_noise/installed`.

```powershell
py -3.14 -I benchmarks/verify_online_install.py output/online_noise/installed .
```

Verification wheel (NOT a frozen release):
`output/online_noise/wheels/signal_processing-2.4.0rc1-py3-none-any.whl`

SHA-256: `3b19881d29d6d1457cc94d892f5494f6168b9af6c1741499c38a8f5c231a986e`.

## Still unresolved

1. Lossless 1 MS/s scheduled processing and hardware acquisition are unqualified;
   current new-path unpaced measurements already fall short of that target.
2. No universal blind noise/signal separation; persistent Gaussian-like signals,
   high occupancy and startup bursts remain limitations.
3. Abrupt noise jumps can trigger/freeze learning and eventually expire the
   reference. They are not automatically relabelled as noise.
4. General >90% detection, short-burst/boundary performance and applicable held-out
   Pfa certification are not established for the new dynamic statistic.

The implementation removes dependence on supplied calibration metadata and adds
online adaptation. It must not be described as a fully robust, frozen, real-time
release on the strength of passing numerical/unit tests.
