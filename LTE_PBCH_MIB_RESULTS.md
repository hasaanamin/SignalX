# PBCH/MIB implementation and verification

19 October 2026. Active source: `signal_processing/`, version **2.4.0rc1**.
Implemented in the working folder, not by modifying an older input ZIP.
This is a tested development addition, **not a frozen/hardware/real-time release**.

## Implemented outcome

`LTEBroadcastReceiver` adds real OFDM/CRS/PBCH/BCH processing after existing
PSS/SSS synchronization. It returns CRC-valid MIB payloads, downlink bandwidth,
PHICH settings, full SFN, transmit-port hypothesis, exact rational input timing,
raw epoch and source provenance. The existing synchronizer API remains unchanged.
Normal-CP FDD, one receive channel and 1/2/4 transmit ports are implemented.
Four chronological same-block frames may be soft-combined. See
[the public contract and commands](LTE_PBCH_MIB.md).

New production files: `lte/bch.py`, `mib.py`, `pbch.py`, `broadcast.py`.
The synchronizer has narrow output/snapshot hooks under its existing single-owner
gate so collection, reset and fault recovery cannot race. CLI adds `--decode-mib`
and `--require-mib`. No detector UI defaults, generic DSP formulas, noise policies
or generic feature schemas were changed by this addition.

## Final combined verification

Windows 11 build 26200, Intel Core i5-1235U; Python 3.14.7, NumPy 2.5.2,
SciPy 1.18.0. No dependency upgrade was made.

| Check | Actual final result |
|---|---|
| Entire source `tests/` | **1,407 passed**, one existing deliberate arithmetic warning, no failures/errors/skips; 237.77 s; exit 0 |
| Existing independent `checks/` | **165 passed**, no failures/errors/skips; 21.10 s; exit 0 |
| Isolated installed replay/LTE checks | **157 passed**, no failures/errors/skips; 45.53 s; exit 0 |
| New PBCH-focused checks | **46**, included in the source and installed totals above |
| Wheel/source parity | All **116** production Python modules byte-identical; installed import origins verified |
| Installed module CLI smoke | Real two-port, +7.5-kHz CFO recording: two MIBs recovered; `--require-mib` exit 0 |
| External coding reference | 90 unchanged-srsRAN-C BCH vectors: exact tail-biting recovery and rate-matching agreement; independent CRC oracle passed |
| Positive numerical/lifecycle oracles | All three scripts completed with exit 0; details below |
| Static checks | Ruff F checks and `git diff --check` pass |

These suites overlap; counts must not be added as unique tests. Test durations
are loaded-host verification times, not processing-throughput claims. The warning
is the existing test deliberately demonstrating unsafe complex64 scalar squaring;
production safe-arithmetic assertions pass.

Public tests cover all four scrambling quarters, all six bandwidth codes, CRS
frequency shifts, antenna CRC masks, positive/negative CFO, both input rates,
origins including 10^12, one-sample/irregular/large partitions, zero/invalid grids,
extreme numerical scales, source provenance, serialization and tampering, real
corrupt-PBCH/valid-sync input, bounds, reset/replay, known/unknown gaps, late
processing faults and concurrent lifecycle rejection. A deterministic noisy-grid
case fails on every single frame alone, then recovers the exact independently
encoded payload by combining all four frames.

Initial focused testing identified a test-helper mistake: prepared input horizons
were compared as if they were logical window coordinates. Actual transport-call
horizons legitimately differ with partition size. Current checks preserve exact
decoded fields/timing while separately checking truthful preparation progress.
No mathematical tolerance was relaxed to hide a signal-processing failure.

The first broad characterization also exposed a real duplicate-output defect:
half-frame tracking could refine a frame estimate by one sample, which was treated
as a new frame. The integrated receiver now associates frame identity/combining
with the 10-ms lattice within configured timing tolerance, while extracting IQ
at the measured position. Both exact failing stimuli are permanent live tests.
The initial report/log remains under `output/lte_pbch_20261019/characterization.*`;
the repaired same-seed execution is `characterization_final.*`.

## Finite synthetic characterization

Same final production revision as the combined tests and wheel. Seed 20261019;
ten independent 40-ms recordings per SNR/port cell, random PCI/payload/CFO/onset,
2 MS/s input, iid complex Gaussian noise. Runtime receives only IQ and stream
configuration. All waveform/coding truth is confined to the offline scorer.
SNR is signal/noise power over the complete synthetic waveform, not an asserted
PBCH-per-RE SINR. This is not a full independent eNodeB or a hardware recording.

| Input SNR | TX ports | Recordings yielding correct MIB / 10 | Correct frames / 40 | Wrong or duplicate MIBs |
|---|---:|---:|---:|---:|
| +10 dB | 1 | 10 | 40 | 0 |
| +10 dB | 2 | 10 | 40 | 0 |
| +10 dB | 4 | 10 | 40 | 0 |
| 0 dB | 1 | 10 | 36 | 0 |
| 0 dB | 2 | 10 | 39 | 0 |
| 0 dB | 4 | 10 | 39 | 0 |
| -5 dB | 1 | 1 | 1 | 0 |
| -5 dB | 2 | 2 | 3 | 0 |
| -5 dB | 4 | 0 | 0 | 0 |

All verified frame positions were within 0.625 raw samples of the independently
known position. Counts include all misses. Ten successes out of ten have only a
74.11% one-sided 95% detection lower bound per cell, so these runs do **not**
qualify a universal reliable minimum SNR or a >90% per-cell guarantee. At -5 dB
this configuration performs poorly. Synchronization misses, CRC failures, channel
estimation and finite dwell can all limit recovery; CRC-valid MIB decoding does
not imply SIB1/user-data recovery.

## Independent numerical preservation

Fresh copied scripts, explicitly routed to the active source with
`SIGNAL_PROCESSING_SRC`, not old JSON:

- Periodogram versus SciPy: 72 comparisons; Welch: 64; autocorrelation: 10.
- Randomized ring wrap/read/oversize/snapshot oracle: 600 cases.
- Detector power/window partition oracle: 425 measurements; 56-key A/B/A feature
  isolation and phase validity/immutability checks passed.
- Ten feature families: 48 direct scalar and 24 general joint-cumulant comparisons;
  two exact bandwidth comparisons. Maximum joint-cumulant absolute error
  4.440892098500626e-15, maximum scalar error 1.7763568394002505e-15.
- Streamed/batch empirical calibration: 10,873 observations, training-only support
  remains False. Configured-filter partition continuity and EOF/reset/replay pass;
  maximum filter-power difference in this run was 0 dB.

Original tolerances/current documented support semantics were retained. These are
positive numerical/lifecycle checks, not arbitrary receiver-noise certification.

## Retention evidence

Separate diagnostic, not a timing benchmark:

| Quiet samples received | Current traced bytes | Native broadcast IQ retained |
|---:|---:|---:|
| 409,600 | 4,210,570 | 24,000 |
| 2,048,000 | 4,212,119 | 24,000 |
| 4,096,000 | 4,213,600 | 24,000 |

Post-warmup traced growth was 3,030 bytes, including the small checkpoint reports;
no accumulating IQ/result history was observed. Quiet runs retained no BCH frames
or pending targets. A separate dense synthetic sequence returned 64 genuine MIBs,
retained four BCH frames at the cap, and reset released raw/soft-frame state.
This is bounded-storage evidence, not an indefinite-run proof.

RSS responsiveness control: touching 134,217,728 bytes changed measured RSS from
105,107,456 to 239,329,280 bytes. RSS responded; allocation tracing and explicit
ownership counters are recorded separately in `retention.json`.

## Distribution and reproducibility

Fresh isolated build:

`output/lte_pbch_20261019/wheels/signal_processing-2.4.0rc1-py3-none-any.whl`

SHA-256: `471a5524bdf5b9c4781b6e7f537ef7622bc4120e57ac5d8c05c21b6adcc92959`.
This pure-Python verification wheel is not a frozen release. No external C source,
reference DLL or test NPZ is included as a production dependency. The source
fixture JSON records the external C and numeric-vector hashes.

Commands (PowerShell, repository root; isolated target must be unused):

```powershell
py -3.14 -m pytest -q tests --junitxml=output/lte_pbch_20261019/supplied.xml
$env:SIGNAL_PROCESSING_SRC = (Get-Location).Path
py -3.14 -m pytest -q checks --junitxml=output/lte_pbch_20261019/acceptance.xml
py -3.14 -m benchmarks.evaluate_lte_pbch --trials 10 --output output/lte_pbch_20261019/characterization_final.json
py -3.14 -m benchmarks.check_lte_broadcast_retention
# build_input contains fresh copies of pyproject.toml, README and active .py files.
py -3.14 -m pip wheel --no-deps --no-build-isolation output/lte_pbch_20261019/build_input -w output/lte_pbch_20261019/wheels
py -3.14 -m pip install --no-deps --target output/lte_pbch_20261019/installed output/lte_pbch_20261019/wheels/signal_processing-2.4.0rc1-py3-none-any.whl
py -3.14 -I benchmarks/verify_lte_install.py output/lte_pbch_20261019/installed .
```

The verifier copies source/LTE tests and numeric fixtures outside the checkout,
inserts only the installed production target first, checks all module hashes,
and asserts every loaded production import resolves to that target. Repository
benchmarks are exposed only for test support, not as production imports.
Retention uses installed `psutil` 7.2.2 as a diagnostic-only dependency.
`installed_cli.log` additionally records an actual isolated `runpy` execution of
the installed `signal_processing.lte` module entry point, with `--require-mib`.

All logs/XML, independent script copies/results, initial/final characterization,
retention JSON, isolated build/install and verification wheel are under
`output/lte_pbch_20261019/`. Original historical evidence was not overwritten.

## Still unqualified / next integration

PBCH/MIB is implemented and verified under the above software conditions.
Scheduled acquisition with this added decoder has **not** been qualified; the
earlier synchronization-only scheduling results do not transfer to its added CPU
work. Existing real-time release blockers remain unresolved. No new 30-second
or soak timing pass is claimed. Pluto/real RF, severe fading/Doppler, TDD/extended
CP, certified false-lock exposure and general low-SNR operation remain unverified.
Protocol-aligned exact raw captures, generic feature dispatch and LTE UI display
are subsequent integration work. Classification, SIB1, LTE user data and 5G are
not implemented by this change. No Git commit/push/tag or release freeze was made.
