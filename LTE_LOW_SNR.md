# LTE low-SNR acquisition profile

The active 2.4.0rc1 development candidate has an **opt-in repeated-observation
LTE profile**. It improves PSS/SSS acquisition and PBCH/MIB recovery; it does not
change the generic transient detector, dataset UI defaults or adaptive-noise
policy. See [measured results](LTE_LOW_SNR_RESULTS.md) for the actual test scope.

```powershell
py -3.14 -m signal_processing.lte --input recording.cf32 --sample-rate 2000000 --profile low_snr --require-mib
```

```python
from signal_processing.lte import LTEBroadcastReceiver, LTESyncConfig
receiver = LTEBroadcastReceiver(stream, LTESyncConfig.low_snr(stream.sample_rate_hz))
# Same process_block/process_envelope/flush and immutable batch/result API.
```

`--profile compatibility` remains the default, with its original configuration
identity and single-observation calculations. Both 2 MS/s and native 1.92 MS/s
are supported. The receiver still receives only IQ and public configuration:
it does not read dataset SNR, truth positions, noise power, PCI or SFN.

## What changes

- Four PSS observations spaced 5 ms apart are combined noncoherently: sum of
  squared matched correlations divided by their actual summed window energy.
  This avoids assuming carrier-phase coherence over those 5-ms intervals.
- Fine CFO is selected jointly from those four PSS observations. SSS evidence
  is channel-corrected separately at each PSS/SSS pair, aligned for its alternating
  half-frame sequence, then coherently accumulated across hypotheses.
- Per-frequency-branch PSS peaks survive into bounded SSS validation. The old
  broad cross-frequency peak suppression could eliminate the real timing when
  a Zadoff-Chu delay/frequency ambiguity produced a nearby stronger false peak.
- After repeated cell confirmation, retained earlier real PBCH slots can be
  decoded retrospectively. This is delayed decoding, not an earlier decision
  using future IQ, and does not emit invented synchronization observations.
- Residual CFO for PBCH uses up to 20 complete already-received OFDM slots,
  rather than a single noisy slot. The observation window is bounded and tied
  to fixed processing horizons, not transport block size. No future, missing
  or next-epoch samples contribute.

Profile controls: four half-frames; normalized PSS threshold 0.10; combined SSS
threshold 0.20; best/second SSS ratio 1.5; two consistent integrated decisions;
eight-half-frame loss timeout. These thresholds describe different integrated
statistics from the compatibility profile's 0.22/0.35 thresholds: this is not
just reducing the original thresholds. Exact profile/algorithm identity changes
accordingly. Overlapping integrations are **not independent statistical trials**;
the confirmation counter does not certify a false-alarm probability.

## Tradeoffs and limits

Four observations need at least 15 ms between the first and last usable PSS,
plus a further confirmation, unknown arrival phase, processing chunks and compute
time. Budget approximately 20–30 ms or more for acquisition, not an instantaneous
short-burst detector. A 40-ms record need not contain all four quarters of the
same BCH block if recording starts at an arbitrary phase. An 80-ms or longer
observation is a safer replay/receiver budget for that case; actual compute and
RF conditions can require longer.

This is LTE FDD/normal CP, central six RB, one RX and 1/2/4 TX hypotheses.
Its tested setting is a persistent cell with bounded/approximately stable CFO
and iid complex Gaussian noise. Channel correction is local to each sync pair,
but PBCH CFO averaging assumes sufficiently stable offset over its received
window. Severe frequency selectivity, fast oscillator drift/Doppler, strong
interferers, TDD/extended CP, unknown receiver distortions and real hardware
remain unqualified. No universal minimum SNR or arbitrary-noise certificate is
implied. CRC is error detection, not authentication.

Additional low-profile retained native IQ is bounded to 29,100 synchronization
samples and 62,400 broadcast samples after each processing step; BCH history is
still four soft frames per bounded cell decoder. There are also bounded input
chunks and two FFT-kernel cache entries. Compared with compatibility this needs
more memory, FFT/candidate work and latency. No scheduled real-time throughput
gate is claimed for it. Existing package acquisition limitations remain separate
from improved decoding probability.

Exact metadata/timestamp rejection, corrected retry, gap/EOF finalization, reset,
source provenance and exception-completed-prefix ownership are unchanged. Gaps
clear integration and combining history. A short suffix cannot be padded into
additional sync repetitions. Prepared horizons describe actual input preparation;
logical sample origins and numerical outputs remain partition-invariant.

## Reproduce

```powershell
py -3.14 -m pytest -q tests/test_lte_low_snr.py tests/test_lte_sync.py tests/test_lte_pbch.py
# Original fixed cases (before/after use the same fixtures and seed).
py -3.14 -m benchmarks.evaluate_lte_pbch --profile low_snr --output output/lte_low_snr_20261019/final_original_cases.json
# Fresh recordings, not used to select the implementation.
py -3.14 -m benchmarks.evaluate_lte_pbch --profile low_snr --snr -5 --trials 50 --seed 20261020 --output output/lte_low_snr_20261019/heldout_minus5.json
py -3.14 -m benchmarks.evaluate_lte_pbch --profile low_snr --snr -5 --frames 8 --random-quarter --trials 20 --seed 20261021 --output output/lte_low_snr_20261019/random_phase_minus5.json
py -3.14 -m benchmarks.evaluate_lte_low_snr_negatives
py -3.14 -m benchmarks.check_lte_broadcast_retention --profile low_snr --output output/lte_low_snr_20261019/retention.json
```

The independent synthetic transmitter still uses unchanged external BCH vectors
for its original block. Longer/arbitrary-phase fixtures advance the actual SFN
and re-encode new blocks with a test-only scalar encoder independently verified
against all 90 original compiled-C vectors. Test truth is used only for scoring.
This is not a complete independent eNodeB or measured RF dataset.

Repeated observation/averaging and CP frequency correction are also used in
[MathWorks' LTE cell-search reference](https://www.mathworks.com/help/wireless-hdl/ug/lte-hdl-cell-search.html).
The actual algorithms, bounds, thresholds and capability here require their own
tests; that reference does not certify this implementation.
