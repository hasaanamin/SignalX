# LTE synchronization: 2 MS/s input, 1.92 MS/s processing

Implemented 19 October 2026, development version 2.4.0rc1. This adds continuous
FDD/normal-CP PSS/SSS synchronization to the positioned IQ interface. It does not
yet decode PBCH/MIB, recover user data or provide hardware qualification.

## Running on an IQ recording

```powershell
# Raw little-endian complex64, actual recording rate 2 MS/s.
py -3.14 -m signal_processing.lte --input recording.cf32 --sample-rate 2000000 --center-frequency 2140000000 --require-lock

# Independent acquisition schedule; explicit loss accounting and enforcement.
py -3.14 -m signal_processing.lte --input recording.cf32 --sample-rate 2000000 --scheduled --enforce-acquisition --require-lock
```

The installed entry point is `rf-lte-sync`. Without `--scheduled`, replay is
lossless offline processing and has no real-time claim. JSON-lines output contains
immutable observations followed by a summary. Exit 0 means the requested run
completed; `--require-lock` returns 2 when no cell was confirmed. Runtime failures
and failed `--enforce-acquisition` return nonzero. No PCI, SNR, quiet prefix, noise
power or expected frame position is passed to the receiver.

Existing recordings must keep their real sample rates. A command-line rate
change does not transform the file. The generic replay benchmark now defaults
to 2 MS/s; dataset metadata/defaults still describe their original recordings.

## Public API

```python
from signal_processing.core import StreamInfo
from signal_processing.lte import LTESynchronizer, LTESyncConfig

stream = StreamInfo("rx-session-1", "recording", 2_000_000, 2_140_000_000)
receiver = LTESynchronizer(stream, LTESyncConfig())
for envelope in application_input:
    observations = receiver.process_envelope(envelope)
    # CANDIDATE is one validated PSS/SSS pair; SYNCHRONIZED requires repetition.
    for observation in observations:
        consume(observation.to_dict())
consume(receiver.flush())
```

Applications may instead call `process_block(IQBlock)` when extra source
provenance is unavailable. Do not mix the two APIs within a run. Envelope mode
preserves source epochs, gain/scaling/timestamp-origin metadata and source
configuration identity in every observation. Receiver epoch, source epoch and
feature-worker cancellation generation have different meanings.

One ordered owner calls the receiver. Reentrant/concurrent calls reject. Config
is an immutable versioned record; replacement requires a new receiver. Exact
stream metadata, sequence/sample position, input size, gain/epoch and timestamps
are checked before state mutation. An initially unspecified RF centre can bind
on the first valid block. Contradictory timestamps reject with 1 ns tolerance
against a fixed epoch anchor, preventing accumulated per-packet rounding drift.
Correcting a rejected block permits retry. Native 1.92 MS/s input is also
supported; other rates reject rather than being relabelled.

## Computation and chronology

At 2 MS/s the receiver converts by 24/25 using a causal 1,201-tap Kaiser FIR at
the interpolated rate, 700 kHz cutoff and explicit startup support. The filter
group delay is 600 interpolated ticks = 25 raw input samples = 24 processing
samples. For processing index m, the signal-time coordinate in the original
input epoch is `(25*m - 600)/24`. The full support/settling interval is 48
processing samples. Synchronization excludes incomplete initial support.

Rational `Fraction` coordinates retain nonzero/10^12 origins without float
rounding. JSON uses numerator/denominator pairs. A frame estimate can precede
the received recording; `frame_start_received=False` explicitly identifies that
case. This is an estimated timing anchor, not a claim that missing IQ was captured
or a UTC timestamp. Receiver outputs are quantized by the 1.92 MS/s timing grid;
no finer multipath timing accuracy is certified.

The channel keeps a bounded raw tail aligned to the rational phase. It does not
reset at transport boundaries or pad zeros at EOF. Fixed internal 4,096-sample
search chunks make logical decisions independent of transport partitioning.
Pure filter preparation can advance beyond the last searched window;
`prepared_input_end` and the processed-search horizon are reported separately.
Per-observation prepared horizons may differ across transport partitions; the
cell identity, correlation measurements and mapped signal timing do not.

PSS search uses all three standard sequences and a bounded frequency grid,
initially -30 to +30 kHz with 7.5 kHz coarse spacing. FFT correlations use direct
nonnegative window-energy reductions, avoiding prefix-difference cancellation.
Candidate frequency is refined by matched-filter search. The preceding SSS is
frequency corrected and equalized with a smoothed PSS channel estimate, then
compared with all 168 groups and both half-frame hypotheses for that sector.

Defaults require normalized PSS score >=0.22, SSS score >=0.35 and best/second
SSS score >=1.3. Two consistent observations establish SYNCHRONIZED. Candidate
confirmation counts are capped at the configured requirement. Locks expire
without refresh; the strongest confirmed candidate is exposed in the snapshot.
These are detector design controls, not certified Pfa thresholds. Their full
configuration and numerical semantics affect profile identity.

After repeated confirmation, the default tracking path evaluates the same PSS/CFO
branches around the cell's predicted next half-frame boundary instead of searching
the entire chunk again. This prediction uses only prior observations. A missing
peak falls back to blind search; expiry and gaps also reacquire. `tracking=False`
retains the full blind-search reference. The tracked profile follows an acquired
cell; it does not promise continual discovery of a newly stronger asynchronous
cell while the current one remains locked. Timing/CFO/metric equivalence is tested
against the blind reference. SSS hypothesis caching contains fixed mathematical
sequences only, bounded to three sector banks, never received waveform data.

## Gaps, EOF and failures

- Known sample gaps require accurate absolute positions and loss counts. Unknown
  overflow must remain flagged; no missing IQ is manufactured.
- A declared gap evaluates every eligible complete old-prefix search window
  using the same finalization as EOF, then resets filter/search/confirmation
  state and advances the receiver epoch. Previously returned observations remain
  valid immutable records of their epoch; they are not retracted or reissued.
- EOF searches the final real partial chunk once. It does not invent a future
  filter tail or complete a truncated synchronization symbol with zeros.
- Unexpected processing errors leave FAULTED. The original typed exception
  carries `completed_observations`, `received_input_end` and
  `prepared_input_end`. Completed observations inside a nested search are
  preserved. Reset is required; flush does not retry or duplicate faulted work.
- The raw receive horizon is distinct from filtered preparation and searched
  windows. A prepared suffix is not represented as successfully synchronized.

## Bounds and interpretation

Default limits: 262,144 raw samples per input call (larger inputs must be
partitioned), 16,384-sample preparation subdivisions, eight PSS candidates per
search chunk, four retained cell candidates, two cached numerical FFT-kernel
sizes and at most 100 retained raw resampler samples. Between calls, processing
residual/overlap stays below 4,396 complex samples. The previous validated input
block is retained for continuity, bounded by the input cap. Temporary arrays
scale with the fixed search/preparation sizes; no IQ recording history grows
with stream length. Returned observations belong to the caller.

The receiver currently targets a single dominant FDD cell with normal CP and
the central synchronization region. TDD, extended CP, arbitrary interference,
large clock drift, severe multipath and multi-cell separation are not qualified.
It uses the synchronization sequences independently of the energy detector;
there is no adaptive-noise pretraining requirement or energy ON gate here.

The package's ten generic feature families remain available through the
existing capture pipeline. This LTE stage does not yet request frame-aligned
raw captures or dispatch cellular observations to those workers. The additive
[PBCH/MIB receiver](LTE_PBCH_MIB.md) now decodes broadcast configuration;
protocol-aligned raw capture remains a subsequent stage.

## Independent reference and tests

`tests/fixtures/lte_sync_sequences.npz` contains all 504 PCI / both half-frame
SSS vectors generated by compiling unchanged srsRAN C at pinned commit
`bef8680d5f9714f3e040e6f9cbc88d7888439b6d`. The companion JSON records source
and fixture hashes. Three PSS vectors use an independent scalar equation oracle.
The test-only MSVC ABI/headers under `benchmarks/lte_reference/` contain no LTE
sequence algorithm. Downloaded reference source retains its AGPL notice under
`output/`; neither it nor the reference DLL is a runtime wheel dependency.

```powershell
# Optional regeneration; requires existing MSVC and network, not used at runtime.
py -3.14 -m benchmarks.generate_lte_reference

# Ordinary checks require only the frozen reference vectors and NumPy/SciPy.
py -3.14 -m pytest -q tests/test_lte_sync.py
py -3.14 -m benchmarks.evaluate_lte_sync --trials 400 --snr 0 --output output/lte_sync_20261019/sync_0db_400.json --enforce-acceptance
py -3.14 -m benchmarks.evaluate_lte_sync --scheduled-seconds 1 --enforce-acceptance
```

The evaluation waveform independently maps those sequences into FDD normal-CP
OFDM with random payload subcarriers. It has no encoded PBCH/MIB or full LTE
transmitter/channel stack. The receiver receives only IQ and permitted stream
configuration. Truth is used only by scoring. Trial-level confidence for a
random mixture is not a separate per-cell/per-CFO certificate; full planned
receiver qualification and hardware-noise validation remain additional work.

Primary mathematical reference: [ETSI TS 136 211 v16.6.0, section 6.11](https://www.etsi.org/deliver/etsi_ts/136200_136299/136211/16.06.00_60/ts_136211v160600p.pdf).
Independent sequence reference: [pinned srsRAN generator](https://github.com/srsran/srsRAN_4G/blob/bef8680d5f9714f3e040e6f9cbc88d7888439b6d/lib/src/phy/sync/gen_sss.c).
