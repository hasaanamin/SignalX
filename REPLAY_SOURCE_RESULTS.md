# Replay/source implementation results — 19 October 2026

Status: **source interface implemented and tested; LTE receiver and hardware
qualification pending.** Active package remains development version 2.4.0rc1.
No release tag, GitHub push or frozen-release sign-off was performed.

## Implemented behavior

- `sources/streaming.py`: immutable acquisition provenance, cancellable positioned
  source protocol, explicit EOF/timeout/cancel/disconnect outcomes and a bounded
  versioned envelope codec preserving metadata through spawn IPC.
- `sources/replay_source.py`: finite array/file replay, independent sample-clock
  pacing, post-wait expiry checks, pending-data timeout retry, reset epochs,
  cancellation and complete returned/expired/pending/unread accounting.
- `runtime/source_runner.py`: one acquisition thread, finite capacity shared by
  queued and checked-out blocks, explicit acknowledgements, drop-newest loss
  propagation, original failure causes and truthful drain timeout/cleanup retry.
- Array/file timestamp conversion uses the existing rational helper. This fixes
  a one-nanosecond tie-rounding difference observed at sample 123 at 1.92 MS/s.
- `benchmarks/evaluate_replay.py`: source-only or existing real DSP diagnostic,
  honest sample balances, failure injection and enforced acceptance exit status.
- `benchmarks/verify_replay_install.py`: isolated import/module parity and public
  test execution, including a real Windows spawn metadata round trip.

The first paced development run used Event.wait for schedule sleeps and expired
674,816 of 1,920,000 samples. Its wait was changed to Python's high-resolution
sleep, polling cancellation without changing global Windows timer settings.
The final runs below use that corrected implementation and an explicit 10 ms
lateness budget. No production edits occurred after the final combined tests.

Usage and ownership contracts: [REPLAY_SOURCE_INTERFACE.md](REPLAY_SOURCE_INTERFACE.md).

## Test results on the combined implementation

| Suite | Actual result | Exit |
|---|---|---:|
| Entire `tests/` suite | 1,304 passed; one deliberate large-complex64 arithmetic warning; no failures/errors/skips | 0 |
| Current independent acceptance suites | 165 passed; no failures/errors/skips | 0 |
| Focused replay/runner/evaluation tests | 54 passed | 0 |
| Same 54 public tests against isolated installed wheel | 54 passed | 0 |
| Production module parity | All 104 Python modules byte-identical to final source; imports resolved to installed target | 0 |
| Ruff undefined/unused-symbol checks and Git whitespace check | Passed | 0 |

The 54 new tests are included in the 1,304 supplied-suite count. The acceptance
suites also overlap existing coverage. These counts are not additive unique tests.

The live integration test passed positioned IQ through the runner into actual
detection, pre-trigger capture and feature computation. It compared exact raw
bytes, coordinates and all 56 feature values/validities with direct pipeline
execution. This is generic DSP verification, not LTE synchronization evidence.

Lifecycle tests cover absent consumers, capacity including held deliveries,
known plus unknown loss propagation, trailing drops, duplicate/foreign acks,
source failure and preserved prefix, cancel races, drain timeout/retry, cleanup
failure/retry, metadata rejection and reset epochs. Memory evidence here is the
bounded ownership/capacity checks; no new RSS or long-soak claim is made.

## Isolated timing diagnostics

Host: Windows 11 build 26200, Intel Core i5-1235U, Python 3.14.7,
NumPy 2.5.2, SciPy 1.18.0. Reader backend: one Python thread. All workloads use
precomputed seeded generic noise, 1.92 MS/s and 16,384-sample blocks. Fixture
preparation is timed separately and completes before the replay clock starts.
These are short diagnostics, not three-run/ten-minute or hardware qualification.

| Run | Scheduled | Processed | Replay expired | Queue dropped | Outstanding at close | Gate / CLI exit |
|---|---:|---:|---:|---:|---:|---|
| Source-only, 3 s, capacity 16 | 5,760,000 | 5,760,000 (100%) | 0 | 0 | 0 | Pass / 0 |
| Slow consumer, 1 s, 40 ms/block, capacity 2 | 1,920,000 | 425,984 | 0 | 1,494,016 | 0 | Expected failure / 1 |
| Existing synchronous DSP, 1 s, capacity 16 | 1,920,000 | 1,346,560 (70.13%) | 573,440 | 0 | 0 | **Failure / 1** |

All sample balances were complete. Slow-consumer recovery closed cleanly; queue
peak was exactly two and trailing queue loss remained visible. The synchronous
DSP diagnostic used an explicitly supplied fixed noise reference of 0.000018,
N=64/H=32, ring 4,096 and actual default features. One capture completed with no
feature failure; the input was noise, so this is not a detection-recall result.

Source-only elapsed time was 3.0021112 s; observed processing rate 1.91865 MS/s
(all scheduled samples delivered, with startup/final overhead in elapsed time).
Delivery latency p50/p99/max was 0.967/2.085/4.891 ms. Peak outstanding was one.
DSP diagnostic p99/max delivery latency was 33.809/36.467 ms. Adding the existing
DSP path failed to keep input service timely in this configuration; further
profiling is needed to attribute computation and reader preemption precisely.
The reader thread does not establish end-to-end 1.92 MS/s capability.

## Reproducible commands and evidence

Evidence directory: `output/replay_source_20261019/` (local generated output,
ignored by Git). Logs include actual failures as well as passes:

- `supplied.log`, `supplied.xml`: complete suite, 101.86 s.
- `acceptance.log`, `acceptance.xml`: current 165-check suite, 8.65 s.
- `build.log`, `install.log`, `installed.log`, `installed_tests.xml`: wheel and
  installed verification; installed 54 tests completed in 3.87 s.
- `scheduled_acquisition.json`, `slow_consumer.json`, `scheduled_pipeline.json`:
  the three final timing observations above.

```powershell
py -3.14 -m pytest -q tests --junitxml=output/replay_source_20261019/supplied.xml

$env:SIGNAL_PROCESSING_SRC = (Get-Location).Path
py -3.14 -m pytest -q checks/test_signal_processing2_acceptance.py checks/test_new_mechanism_acceptance.py checks/test_prior_repairs_against_signal_processing2.py checks/test_examples2_acceptance.py checks/test_f_public_acceptance.py checks/test_g_public_acceptance.py

py -3.14 -m benchmarks.evaluate_replay --seconds 3 --enforce-acceptance
py -3.14 -m benchmarks.evaluate_replay --seconds 1 --consumer-delay 0.04 --capacity 2 --enforce-acceptance
py -3.14 -m benchmarks.evaluate_replay --seconds 1 --noise-power 0.000018 --enforce-acceptance

# Build input was a separate copy of the final production package plus
# pyproject.toml and README.md, outside the active package/build tree.
py -3.14 -m pip wheel output/replay_source_20261019/build_input --no-deps --no-build-isolation --wheel-dir output/replay_source_20261019/wheels
py -3.14 -m pip install --no-deps --target output/replay_source_20261019/installed output/replay_source_20261019/wheels/signal_processing-2.4.0rc1-py3-none-any.whl
py -3.14 -I benchmarks/verify_replay_install.py output/replay_source_20261019/installed .
```

Verification wheel (not a frozen release):
`output/replay_source_20261019/wheels/signal_processing-2.4.0rc1-py3-none-any.whl`.
SHA-256:
`f14eace5f1e8209203870e579950d37a22769c7e5a10516e7ea70c7035a5f436`.

## Pending work

The source-facing replay/provenance/ownership portion is ready for the next LTE
implementation stages. A native Pluto adapter, independent actual LTE fixtures,
PSS/SSS synchronization, PBCH/MIB decoding, protocol-aligned captures and cellular
UI outputs are not implemented by this change. The legacy process transport is
unchanged; future LTE IPC must deliberately use the full envelope codec.

The measured synchronous DSP timing failure remains unresolved. No hardware,
long-duration real-time, LTE-detection or frozen-release claim follows from the
passing source-interface tests.
