# Release lock: BLOCKED

Candidate: **2.4.0rc1**, active source, Windows 11 / Core i5-1235U.
This is not `FROZEN_PASS`; no release tag or external publication is authorized
by these results. Do not replace the failed timing observations with unit-test
counts. Source/distribution identity is recorded in the qualification manifest
and candidate `ARTIFACTS.json`; checksums identify, not certify, the code.

## Actual gates after on-demand features / LTE decoding changes

Latest contracts and full observations:
`ON_DEMAND_FEATURES_AND_REALTIME_LTE_20261020.md`.
Latest executed evidence: `output/on_demand_20261020/qualification/`.
No accepted release ZIP is issued while the required timing gates fail.

| Gate | Status / evidence |
|---|---|
| Exact atomic stream/timestamp admission | PASS: H-series public cases, 30 source tests; corrected retry preserved |
| Full source suite after combined production edits | PASS: 1,515 tests, one deliberate arithmetic-warning fixture, no failures/errors/skips |
| Current explicitly routed acceptance | PASS: 165 tests; `SIGNAL_PROCESSING_SRC` points to active source |
| G01–G08 public regression mechanisms | PASS: 123 source cases in `test_g_revision.py` |
| F01–F08 public regression mechanisms | PASS: 83 source cases in `test_f_revision.py`; benchmark failures still enforce failure |
| A/B/E available regressions | PASS for available functional cases/current equivalents in tests/checks; E14 scheduled performance remains FAIL |
| New scheduling/numerical/transport controls | PASS: 31 focused cases included in the full source suite |
| On-demand features / native sync / BCH cadence | PASS: 18 new focused cases, included in source and installed suites; real IQ/aggregation and independent direct-sum oracles |
| Independent numerical scripts | PASS: three genuinely executed current-equivalent scripts at original tolerances, not saved JSON answers |
| Bounded memory / resources | PASS for exercised finite probes: 4,096,000 quiet LTE samples, 64 real dense-stream MIBs, empty reset; responsive touched-128-MiB RSS control |
| Required-sink failure injection | Expected failure observed: 3 sink errors, enforcement exit 1, clean recovery and zero unfinished/unresolved work |
| Slow-sink recovery | PASS at explicitly lower 10 kS/s control workload; not a 1 MS/s timing qualification |
| Adaptive 1 MS/s processing, excluding producer expiry | **FAIL**: 99.4978% eligible samples consumed, 131,072 acquisition-queue losses, normal-priority on-demand run; all 1,564 requested feature jobs terminal, no feature overload |
| LTE native 1.92 MS/s including PBCH/MIB | **FAIL**: optional native/block-cadence, one-BLAS-thread run acquired 16.2596%; p99/max 1,689.582/1,825.494 ms against 50/100 ms, 48,234,496 queue losses. Only 151/750 BCH blocks verified; correct returned results cannot excuse missing data |
| Required repeated lossless runs / ten-minute soak | NOT QUALIFIED: shorter actual runs already fail mandatory gates; no lossless soak claim |
| Broad detection / SDR hardware qualification | Not requalified by this scheduling task; real receiver acquisition remains unverified |
| Isolated final wheel / import and module parity | PASS: 515 installed tests; all 120 production modules byte-identical to source, imports outside checkout; wheel CRC checked |

Counts overlap; do not add them as unique tests. The frozen historical helpers
are unchanged under `checks/`. Their documented API/statistical migrations are
not silently reverted: training-only calibration is uncertified, observed-excess
bandwidth is not physical coverage, unavailable SNR is explicit, timestamp
anchors are sample-index consistent, and async unaccepted overload is immediate.
Current numerical equivalents are copied byte-identically from the earlier
verified copies and executed freshly. The obsolete original support assertion
is not described as a completed original numerical-script pass.

An additional benchmark guard prevents an unpaced functional run from reporting
real-time qualification. Its two source harness cases were rerun successfully
after that guard; the 515-case installed run includes the stronger assertions.
No runtime production module changed after the combined source gate. The
function/build runner exited 0; that is not a timing pass. Both normal 30-second
on-demand acquisition and required LTE timing enforcement returned 1.

## Supported interface, not a timing sign-off

Use immutable positioned complex64 IQ with exact stream metadata and declared
loss, and actual read-back receiver sample rate. Native LTE requires genuine
1.92 MS/s; do not relabel 2 MS/s data. Preserve causal authorized calibration;
display floors and dataset truth do not supply noise. Respect pre-trigger/ring
requirements and feature validity, including SNR availability and bandwidth
coverage semantics. Handle overload, typed faults with completed-prefix recovery,
EOF and bounded shutdown explicitly.

Default callback order and per-block progress remain compatible. Optional
completion-order analysis and grouped scalar telemetry are explicit controls;
arbitrary sinks stay outside lifecycle locks and off the acquisition caller.
Input capacity remains 16 in measured runs. The on-demand measurement uses one
persistent feature process, retaining real ten-family/56-key analysis only for
requested captures; legacy automatic per-event analysis remains available. No global OS power-plan or start-
method change is made. Affinity/QoS experiments are identified separately and
do not establish portable real-time behavior.

The primary deployment environment is pinned in
`requirements-acquisition-observed.txt`, not qualified on every supported Python
or OS. LTE synchronization/PBCH/MIB is not complete LTE data-channel demodulation,
5G, classification, or SDR hardware validation.

## Reproduction and change rules

See `ON_DEMAND_FEATURES_AND_REALTIME_LTE_20261020.md`. Functional/build runner:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File benchmarks/qualify_scheduling_candidate.ps1 -OutputRoot output/new_qualification -NumericalRoot PATH_TO_GENUINE_CURRENT_NUMERICAL_SCRIPTS
```

The evidence archive supplies those runnable scripts under
`qualification/numerical/`; unchanged originals are in the source archive's
`checks/`. Raw LTE test IQ is an explicitly external generated fixture; the
report gives its generation command. Do not copy stored observations into a
baseline or use them as runtime answers.

The next practical requirement is additional measured reduction of adaptive
hop/object/coordination cost and LTE worst-case startup/reacquisition scheduling
cost, or a separately qualified bounded execution backend. Excluding producer
expiry does not excuse queue loss. Requalify production changes, distributions,
sample/result accounting and actual deadline gates before changing this status.
No untested edit, favorable isolated run, or passing test count can unlock it.
