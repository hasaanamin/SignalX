# 2 MS/s replay and LTE synchronization results

19 October 2026. Active development candidate: 2.4.0rc1. **LTE synchronization
is implemented; lossless sustained end-to-end release qualification is pending.**

## What changed

The replay benchmark now defaults to 2 MS/s. Existing dataset metadata was not
relabelled. The new public `signal_processing.lte` package accepts positioned
2 MS/s or native 1.92 MS/s IQ, performs bounded causal rate conversion where
required, searches all PSS/SSS cell identities, estimates frequency offset and
confirms repeated FDD/normal-CP synchronization. It preserves original input
sample coordinates as rational values and source provenance in observations.

The raw-recording CLI is `python -m signal_processing.lte` / `rf-lte-sync`.
No expected PCI, timing, SNR label or noise-only prefix is supplied to the
receiver. PBCH/MIB decoding, frame-aligned feature integration and a real Pluto
source are subsequent stages. See [usage and contracts](LTE_SYNCHRONIZATION.md).

## Functional and numerical verification

| Check | Final result |
|---|---|
| Entire supplied suite | **1,361 passed**, one deliberate arithmetic warning, no failures/errors/skips; exit 0 |
| Current independent acceptance | **165 passed**, no failures/errors/skips; exit 0 |
| Isolated installed replay/LTE public checks | **111 passed**; exit 0 |
| Production parity | All **112 Python modules** byte-identical to active source; installed imports verified |
| LTE reference vectors | All 504 PCI / both SSS half frames match compiled original pinned srsRAN output; three PSS sequences match an independent scalar equation oracle |
| LTE OFDM/channel math | Direct DFT control and direct/batch rational-filter controls pass; partition and exact coordinate checks pass |
| Lifecycle/boundaries | Atomic metadata/time rejection and retry; gap/EOF residuals; partial-error observation recovery; expiry/reset; bounded retained state; raw CLI pass/no-lock paths pass |
| Tracking reference | Timing and PSS/SSS/CFO statistics match blind-search reference at original tolerances in the tested differential fixtures |

There are 57 new LTE tests. The installed 111 checks contain those plus the 54
replay/source checks; all are also included in the 1,361 suite count. Suites
overlap and must not be added as unique tests.

The three existing independent numerical scripts also completed with exit 0:

- 72 periodogram and 64 Welch comparisons against SciPy; 10 direct autocorrelation
  comparisons; 600 randomized ring cases; 425 detector windows per tested
  partition; 56-key A/B/A event isolation.
- 48 direct scalar and 24 general joint-cumulant comparisons; two exact
  known-spectrum bandwidth comparisons across all ten families.
- 10,873-window streamed/batch empirical statistics with training-only support
  explicitly False; configured-filter continuity and exact EOF/reset replay.

Original scripts were copied byte-for-byte into a separate output directory
before execution, with explicit source routing. Their generated JSON did not
overwrite historical observations in `checks/`.

## Detection and timing accuracy characterization

The fixture independently maps reference PSS/SSS sequences into normal-CP FDD
OFDM with random payload carriers. It is **not a complete encoded LTE transmitter**:
there is no PBCH/MIB or protocol stack in the fixture. All expected values stay
in the scorer. Each trial uses separately generated noise, PCI, CFO and onset;
multiple observations inside a recording are not counted as independent trials.

The 0 dB mixed-condition test used 400 recordings, 40 ms of signal each, random
PCI 0–503, random frequency offset within +/-30 kHz and random onset up to 5 ms.
SNR means active central-72-carrier waveform power divided by total iid complex
Gaussian input-noise power. Success requires confirmed correct PCI, frame error
<=2 raw input samples and frequency error <=1,500 Hz.

| Test | Joint successes | Wrong confirmed-cell trials | Qualification interpretation |
|---|---:|---:|---|
| 0 dB, 400 trials, final tracked profile | **396/400 (99%)** | 0 | One-sided exact 95% lower success bound **97.726%** for this random mixture |
| -5 dB, 50 trials | 6/50 (12%) | 0 | Weak-SNR characterization; unreliable |
| -10 dB, 50 trials | 0/50 | 0 | Weak-SNR characterization; unreliable |

The final 0 dB run's matched-frame absolute timing error was p50 0.292, p95
0.542 and maximum 0.708 raw samples. Frequency-error p50/p95/max was
401.6/1,160.5/2,047.2 Hz. The joint success criterion retains its frequency-error
limit; observations outside it are not counted successful just because PCI agrees.

The pre-tracking implementation achieved 397/400 under the same seeded mixture;
that earlier observation is preserved. The tracked implementation achieved
396/400 and passed the predeclared mixture gate. Neither result is a confidence
certificate for every PCI/CFO cell, arbitrary interference or real hardware.
The final frequency-grid boundary validation was also hardened; a live check
verifies that the qualified default grid remains exactly unchanged.

## Scheduled measurements and remaining timing failure

Windows 11 build 26200, Intel Core i5-1235U, Python 3.14.7, NumPy 2.5.2,
SciPy 1.18.0. Raw rate 2 MS/s, transport blocks 16,384, capacity 16,
replay expiry budget 10 ms. Fixtures are prepared before the independent replay
clock starts, and preparation is reported separately. Final timing runs were
performed after heavy checks completed. These are short software tests, not a
30-second/ten-minute or hardware qualification campaign.

| Scope | Scheduled samples | Processed | Expired | Queue drops | Outcome |
|---|---:|---:|---:|---:|---|
| Source only, 3 s | 6,000,000 | 6,000,000 (100%) | 0 | 0 | Pass, exit 0 |
| Existing synchronous generic DSP, 1 s | 2,000,000 | 737,280 (36.864%) | 1,262,720 | 0 | **Fail**, exit 1 |
| LTE sync before tracking, 1 s signal | 2,000,431 | 573,440 | 0 | 1,426,991 | **Fail**, exit 1 |
| LTE sync with tracking, 1 s signal | 2,000,431 | 2,000,431 (100%) | 0 | 0 | Pass, exit 0 |
| Final LTE sync, 3 s signal | 6,000,431 | 5,984,047 (**99.727%**) | **16,384** | 0 | **Strict no-loss gate fails**, exit 1 |

The additional 431 samples in LTE runs are the actual fixture's initial onset
interval; the schedule includes them. Every run balanced its samples and closed
with zero outstanding deliveries. Generic DSP, source-only and LTE-sync scopes
are distinct; none of their rates is a universal throughput claim.

The final three-second run had p99/max delivery lag **49.797/54.911 ms**, within
the separately declared 50/100 ms lag limits. Peak outstanding was seven. One
producer wake was **10.5928 ms late**, exceeding the unchanged 10 ms expiry
budget, so the block was explicitly lost. There was no acquisition-queue
overflow. The receiver processed the declared discontinuity, reacquired PCI 42
and finished with 596 confirmed updates, zero wrong confirmed IDs and zero
unfinished work. The failure remains visible; the expiry policy was not relaxed.

This exceeds the earlier >90% acquisition aspiration in this finite test, but
does **not** satisfy the stricter zero-loss freeze gate. Hardware integration
and sustained processing remain unqualified. Precisely attributing the remaining
wake delay requires source/OS scheduling measurements or a process-isolated
source path; replay cannot establish what Pluto USB/DMA will deliver.

### Measured optimization

Profiling identified repeated broad PSS/CFO searches as the receive bottleneck.
The initial receiver also rebuilt the same SSS basis thousands of times during
initialization. The final code caches only fixed mathematical hypothesis data
and, after repeated real confirmation, searches around predicted PSS times.
Loss/expiry returns to blind search; a full-search reference remains selectable.

The one-second post-change run used four blind and 465 tracking search chunks,
processed all samples, produced 199 confirmed updates and had no incorrect cell
IDs. The final three-second run used nine blind and 1,394 tracking searches,
including reacquisition after its one lost input block. No feature family,
false-alarm policy, source deadline or queue capacity was removed to obtain this.
The LTE scheduled path does not yet run the generic 56-feature pipeline.

## Reproduction and artifacts

```powershell
py -3.14 -m benchmarks.evaluate_replay --sample-rate 2000000 --seconds 3 --enforce-acceptance
py -3.14 -m benchmarks.evaluate_replay --sample-rate 2000000 --seconds 1 --noise-power 0.000018 --enforce-acceptance
py -3.14 -m benchmarks.evaluate_lte_sync --trials 400 --snr 0 --output output/lte_sync_20261019/sync_0db_400_final.json --enforce-acceptance
py -3.14 -m benchmarks.evaluate_lte_sync --trials 50 --snr -5
py -3.14 -m benchmarks.evaluate_lte_sync --trials 50 --snr -10
py -3.14 -m benchmarks.evaluate_lte_sync --scheduled-seconds 3 --enforce-acceptance
py -3.14 -m pytest -q tests
py -3.14 -I benchmarks/verify_lte_install.py output/lte_sync_20261019/installed .
```

Actual logs/JSON/XML, the profiling record, isolated build input and installed
verification target are under `output/lte_sync_20261019/`. The complete current
acceptance command remains in README. Independent oracle copies and outputs
are under that evidence directory's `oracles/` with explicit
`SIGNAL_PROCESSING_SRC` set to the active project root.

Fresh verification wheel:
`output/lte_sync_20261019/wheels/signal_processing-2.4.0rc1-py3-none-any.whl`

SHA-256:
`98a25728011daeee0704feaa845a3025120d6fad5fe31fc3f38abd27b024032a`

The wheel contains Python production code; the compiled external reference DLL
is only a validation tool outside the package. Source and installed tests use
the same mathematical records and verify actual import origins.

Next work: PBCH/MIB recovery and protocol capture/feature integration, alongside
remaining source-scheduling and existing generic-DSP throughput work. No claim
of live Pluto operation, full LTE channel decoding, low-SNR universality or a
frozen release is made.
