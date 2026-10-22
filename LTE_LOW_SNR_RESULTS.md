# LTE low-SNR improvement results

19 October 2026. Active package: `signal_processing/`, development candidate
**2.4.0rc1**. Implemented in the working folder. No frozen-release or hardware
sign-off is claimed. [Usage and operating contract](LTE_LOW_SNR.md).

## Implemented change and diagnosis

The preceding 40-ms synthetic sweep recovered a MIB in only 3/30 recordings
at -5 dB. Live stage attribution showed that most did not reach PBCH: a single
noisy PSS/SSS observation was the main acquisition bottleneck. After repeated
sync was added, the same cases reached PBCH but two four-port cases still failed.
The single-slot residual-CFO estimate was too noisy. A further 0-dB corner case
exposed Zadoff-Chu delay/frequency ambiguity: cross-frequency peak suppression
discarded the real timing before SSS could disambiguate it.

Production repairs:

- An immutable opt-in `LTESyncConfig.low_snr()` profile combines four PSS and
  phase/channel-corrected SSS observations over consecutive received half-frames.
- Bounded per-frequency PSS candidates survive until SSS validation; no truth
  positions, PCI, noise variance or SNR are supplied to runtime search.
- Retained real earlier PBCH slots are decoded once acquisition is confirmed.
  Frame identity, epoch, exact rational coordinates and duplicate prevention
  remain intact.
- PBCH residual CFO averages up to 20 complete received slots within fixed
  chronological processing horizons; no cross-gap or unprocessed suffix input.
- Integration controls are keyword-only, preserving all legacy positional
  constructor arguments. Compatibility is still the default and retains its
  original profile ID and numerical path.

No generic energy-detector sensitivity, feature formulas, AR1 UI defaults or
adaptive-noise calibration was changed by this LTE-specific repair. Added
source tests exercise the actual receiver, not fabricated successful results.

## Same-fixture comparison

Original seed 20261019, four frames (40 ms active waveform), ten recordings per
SNR/antenna cell, 2 MS/s input. This preserves the original random PCI, payload,
signed CFO, onset and noise realizations. SNR is defined from the mean active
synthetic waveform power before short onset padding, exactly as before.

| SNR | Prior compatibility: recordings with correct MIB | New low-SNR profile | New correct frames / 120 | Wrong/duplicate MIBs |
|---|---:|---:|---:|---:|
| +10 dB | 30/30 | 30/30 | 120 | 0 |
| 0 dB | 30/30 | 30/30 | 120 | 0 |
| -5 dB | 3/30 | 30/30 | 111 | 0 |

At -5 dB, per-port frame results were 40/40, 38/40 and 33/40 for one, two and
four TX ports respectively. A successful recording means at least one correctly
attributed CRC-valid MIB, not that every frame decoded. The preceding baseline
results remain in `output/lte_pbch_20261019/characterization_final.json`.
Current live runs are `sync_improved.json`, `cfo_improved.json` and
`final_original_cases.json` under this task's output directory. They record the
stages of the repair; intermediate misses are not hidden.

The old production source and original test helper were copied before editing to
`output/lte_low_snr_20261019/baseline_source/`. Frozen external sequence/BCH NPZ
vectors were not modified. Three representative four-frame waveforms, including
noise and CFO, were freshly checked byte-identical to the pre-edit generator.

## Fresh-seed generalization and confidence

No parameter tuning used these held-out results. Independent trials are complete
recordings; multiple frames in one recording are not counted as independent.

| Fresh scenario | TX ports | Correct-MIB recordings | Wrong/duplicate MIBs |
|---|---:|---:|---:|
| -5 dB, 40 ms, seed 20261020 | 1 | 50/50 | 0 |
| -5 dB, 40 ms, seed 20261020 | 2 | 50/50 | 0 |
| -5 dB, 40 ms, seed 20261020 | 4 | 50/50 | 0 |
| -5 dB, 80 ms, random BCH phase, seed 20261021 | 1 | 20/20 | 0 |
| -5 dB, 80 ms, random BCH phase, seed 20261021 | 2 | 20/20 | 0 |
| -5 dB, 80 ms, random BCH phase, seed 20261021 | 4 | 20/20 | 0 |
| -10 dB, 80 ms, random BCH phase, seed 20261022 | 1 | 0/10 | 0 |
| -10 dB, 80 ms, random BCH phase, seed 20261022 | 2 | 0/10 | 0 |
| -10 dB, 80 ms, random BCH phase, seed 20261022 | 4 | 0/10 | 0 |

For each 50/50 cell, the exact one-sided 95% Clopper–Pearson lower bound is
**0.9418449208830277**. This supports a >90% one-MIB-per-record probability for
that declared synthetic mixture: iid complex Gaussian noise, fixed port gains,
random PCI/payload/onset/CFO in ±20 kHz, and the original BCH-phase-zero start.
It is not a per-PCI, arbitrary-channel or hardware guarantee. The random-phase
20/20 cells have only an 86.09% lower bound and are characterization, not a
separate >90% per-mode qualification. -10 dB remains unreliable in this profile.

All verified frame coordinates stayed within 0.666667 raw samples across these
runs; the original <=2-sample acceptance limit was retained. All payload bits,
antenna masks, full SFN/quarter, PCI, frame identity and duplicates were scored,
not just output counts. General low-SNR or rare-event reliability is not proved.

Longer fixtures now advance SFN and re-encode actual new BCH blocks instead of
repeating an impossible constant SFN indefinitely. Their independent test-only
scalar encoder matches all 90 original compiled-srsRAN C vectors exactly.
The original four-frame block still uses frozen external codewords. Neither
generator nor ground truth is imported into production decoding.

## Signal-absent controls and retained memory

66 independent finite controls: 25 white-noise, 25 AR(1)=0.8 noise, ten unrelated
OFDM, five tones and one silence record. Exposure totals 5.282155 seconds.
There were **zero confirmed cells and zero MIBs**. This is a smoke/control
exposure, not a rare-false-alarm certificate. Colored-noise negative controls do
not qualify reception of weak LTE signals in colored receiver noise.

Separate traced diagnostic (not timing acceptance):

| Quiet samples received | Current traced bytes | Broadcast IQ retained |
|---:|---:|---:|
| 409,600 | 30,241,546 | 62,400 native samples |
| 2,048,000 | 30,243,091 | 62,400 native samples |
| 4,096,000 | 30,244,572 | 62,400 native samples |

Warm-cache traced growth was 3,026 bytes, including the small checkpoint reports.
64 dense genuine broadcast results completed, soft history stayed capped at four
frames and reset released retained raw/soft state. RSS responsiveness passed:
touching 134,217,728 bytes changed RSS from 104,665,088 to 238,886,912 bytes.
No indefinite-memory or ten-minute-soak claim follows from this finite probe.

## Final source/distribution verification

Windows 11 build 26200 / Intel Core i5-1235U; Python 3.14.7, NumPy 2.5.2,
SciPy 1.18.0. No dependencies upgraded. Timing of these loaded-host quality
runs is not an acquisition-throughput measurement.

Final verification outcomes under `output/lte_low_snr_20261019/`:

| Final combined check | Actual result |
|---|---|
| Source `tests/` | **1,430 passed**, one pre-existing deliberate arithmetic warning; 261.38 s; exit 0 |
| Current independent `checks/` | **165 passed**; 17.33 s; exit 0 |
| Isolated installed replay/LTE checks | **180 passed**; 76.60 s; exit 0 |
| Production module parity | **116** byte-identical modules; installed import origins checked |
| Installed CLI | Real four-port -5 dB IQ, +12.345-kHz CFO: `--profile low_snr --require-mib` exit 0 |
| Static validation | Ruff F checks and `git diff --check` pass |

No failures, errors or skips in these final suites. The new 23 low-SNR checks are
included in source/installed totals, not additional unique tests. The warning
is the existing test intentionally demonstrating unsafe complex64 squaring.
Initial combined runs before the constructor-only compatibility correction
passed 1,429 source tests, 165 current independent acceptance checks and 179
installed tests; those logs remain separate. Suites overlap and must not be
summed as unique tests.

Fresh independent numerical scripts also completed: 72 periodogram, 64 Welch,
ten autocorrelation, 600 ring, 425 detector-window/partition comparisons; ten
families with 48 scalar, 24 joint-cumulant and two bandwidth comparisons; 56-key
A/B/A event isolation; 10,873-window streamed/batch training-only calibration;
configured-filter continuity and exact EOF/reset replay. Original tolerances and
training-only supported-PFA=False semantics were preserved.

Characterization preceded the final constructor-signature-only compatibility
adjustment; DSP/profile values and identities are unchanged by that adjustment.
Final public-source and isolated-wheel tests include the weak-signal, arbitrary
phase, direct-sum, metadata/gap and legacy-constructor cases after that adjustment.

Final isolated verification wheel:

`output/lte_low_snr_20261019/final_wheels/signal_processing-2.4.0rc1-py3-none-any.whl`

SHA-256: `ad6df26980c17c9097d98cd1688e4656f9bf34a86a47af72e38fdfa99f2c7e45`.
All 116 production modules were compared byte-for-byte with active source.
The installed verifier asserts actual import origins, copies tests/fixtures
outside the checkout and executes the real public APIs there. Earlier wheels
remain historical verification artifacts, not this final deliverable.

## Reproduce and remaining limits

See [the contract](LTE_LOW_SNR.md) for source tests, same-fixture/fresh-seed
sweeps, negative controls and retention commands. Final combined/build commands:

```powershell
py -3.14 -m pytest -q tests --junitxml=output/lte_low_snr_20261019/supplied_final.xml
$env:SIGNAL_PROCESSING_SRC = (Get-Location).Path
py -3.14 -m pytest -q checks --junitxml=output/lte_low_snr_20261019/acceptance_final.xml
# final_build_input contains fresh active .py files, README and pyproject.
py -3.14 -m pip wheel --no-deps --no-build-isolation output/lte_low_snr_20261019/final_build_input -w output/lte_low_snr_20261019/final_wheels
py -3.14 -m pip install --no-deps --target output/lte_low_snr_20261019/final_installed output/lte_low_snr_20261019/final_wheels/signal_processing-2.4.0rc1-py3-none-any.whl
py -3.14 -I benchmarks/verify_lte_install.py output/lte_low_snr_20261019/final_installed .
```

This delivers improved LTE protocol acquisition/MIB recovery, not a new generic
short-burst energy-detector guarantee. It costs longer dwell, larger bounded
buffers and more CPU; no scheduled real-time acceptance gate was run or passed
for the new profile. Existing acquisition-rate limitations remain. Real Pluto
RF/noise, severe fading/interference/drift, weak colored-noise reception, TDD,
extended CP, SIB1/user data and generic LTE-aligned capture/feature integration
remain unqualified or pending. No commit/push/tag/freeze was made.
