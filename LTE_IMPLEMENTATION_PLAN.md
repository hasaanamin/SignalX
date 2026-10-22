# SignalX: LTE receiver implementation plan without SDR hardware

Date: 19 October 2026

Status: **PARTIALLY IMPLEMENTED — source/replay, synchronization and PBCH/MIB
are implemented; integrated capture/UI and full qualification remain pending.**

## 1. Outcome and scope

Build a streaming, receive-only LTE downlink acquisition, synchronization and
MIB receiver using the existing IQ foundation. Until Pluto is available, run
the real receiver on independently generated LTE IQ and attributable recordings
through the same positioned-source interface planned for hardware.

The first release profile is deliberately specific:

| Property | Initial profile |
|---|---|
| Radio mode | LTE FDD, normal cyclic prefix, one receive channel |
| Carrier | One configured carrier centre; no truth-provided cell identity or timing |
| Input / receiver processing rate | 2 MS/s input; causal 24/25 conversion to 1.92 MS/s |
| Observation | Central six resource blocks, not a complete wider channel |
| Implemented transmit-port hypotheses | One, two and four CRS ports; synthetic verification, no hardware qualification |
| Required outputs | PCI, sample-based frame timing, frequency-offset estimate, CRC-validated MIB, acquisition/continuity quality, exact raw captures |
| Optional existing outputs | Generic 56-key features, PSD and transient monitoring |
| Host | Existing Windows 11/Core i5-1235U; qualify actual installed dependency versions |
| Exclusions | TDD/extended-CP qualification, SIB1, PDSCH user-data decoding, uplink, MIMO data reception, carrier aggregation, 5G and transmission |

This central-band receiver can search for a cell on a wider LTE carrier; it does
not preserve that carrier's full bandwidth. MIB recovery does not imply SIB1 or
user-data recovery. Unsupported inputs may produce no lock: the receiver cannot
always determine the cause from received IQ alone.

No claim of continuous hardware reception is permitted until hardware tests pass.
The absence of an SDR does not prevent implementing the receiver, its contracts,
replay integration, failure handling, UI and software performance tests.

## 2. Existing code to retain and actual gaps

The current production package is `signal_processing/`, version `2.4.0rc1`.
There are pre-existing uncommitted online-noise and UI-default changes. Preserve
them; this plan does not authorize resetting them or creating a release tag.

| Existing component | Reuse / required work |
|---|---|
| `core/contracts.py`, `stream_compatibility.py`, `timebase.py` | Preserve immutable IQ, exact metadata admission, half-open intervals and rational sample/time conversion. Add LTE-specific records separately. |
| `sources/array_source.py`, `file_source.py` | Reuse for deterministic replay; add bounded scheduled replay rather than a second receiver implementation. |
| `buffering/ring_buffer.py` | Reuse exact storage/read semantics; configure retention in milliseconds. |
| `capture/assembler.py` | Retain transient semantics. Add a protocol-aligned interval assembler instead of faking energy-trigger transitions. |
| `runtime/pipeline.py` | Keep existing transient pipeline. Add a sibling LTE orchestrator sharing proven primitives. |
| `runtime/shared_transport.py`, `capture_stream.py` | Existing bounded process transport is useful, but currently does not preserve a complete source metadata envelope. Extend/version it for timing quality, unknown loss and configuration epoch. |
| `runtime/worker.py`, `process_analysis.py` | Preserve terminal outcomes, reset and backpressure. The process backend currently rejects events over 4,096 samples; solve explicitly for longer LTE captures. |
| `context/`, `transforms/`, `features/` | Retain the ten families and 56-key schema. No cross-capture caches. |
| `detection/` | Keep activity monitoring optional. Neither a quiet-prefix energy detector nor blind AR1 learning is the required LTE acquisition gate. |
| `ui/` | Keep dataset tools; add a separate LTE replay/live panel and typed LTE results. |

PSS/SSS synchronization and PBCH/MIB now exist under `signal_processing/lte/`.
See `LTE_SYNCHRONIZATION.md` and `LTE_PBCH_MIB.md` for actual implemented scope.
No production Pluto source exists. Generic OFDM examples are not LTE reference signals.

## 3. Proposed architecture and ownership

```text
IQSource: file / array / scheduled replay / future Pluto
    -> bounded source runner and validated acquisition envelope
    -> raw circular storage (one authoritative receive coordinate system)
       -> stateful channel preparation, if needed
          -> continuous PSS/CFO/SSS acquisition and tracking
             -> OFDM/CRS/channel estimation/PBCH/MIB
             -> bounded LTE result channel
       -> protocol interval requests -> immutable raw IQ capture
          -> existing bounded generic feature workers / optional storage
       -> optional independent energy monitor

UI consumes bounded summaries/results; it never owns the acquisition clock.
```

The decoder must not wait for energy ON. A weak LTE synchronization signal can
be detectable by correlation even when an energy detector does not trigger.

Use one ordered LTE state owner initially. Batch numerical correlations/FFTs,
but commit tracking, capture requests and results in chronological order. The
raw ring retains uncorrected input; frequency correction, equalization and any
resampling belong to explicitly identified derived streams. Generic raw-IQ
features must not silently receive whitened/equalized IQ instead.

### Public contracts to introduce

- `LTEReceiverConfig`: immutable, versioned and canonically identified; rate,
  duplex/CP scope, search range, candidate limit, thresholds, confirmation rules,
  timing/filter semantics, retention and queue limits.
- `LTECellObservation`: PCI, source/processing epoch, frame-start sample,
  fractional timing estimate where applicable, frequency offset in Hz, search
  metrics, status and bounded diagnostic reasons.
- `LTEMIBResult`: decoded fields, CRC outcome, transmit-port hypothesis,
  frame-number provenance, contributing sample ranges and configuration ID.
- `LTECaptureRequest/Result`: stable identity, requested/actual raw interval,
  alignment reference, missing-prefix/suffix reasons and existing quality.
- `ReceiverSnapshot`: bounded counters/latencies, lock state, drops, expiry,
  retained samples and unfinished work. It must not retain waveform history.

New records need serialization round trips and strict validation. Preserve the
existing feature schema; add cellular records rather than reinterpreting its
56 scalar keys. Sample timing is not UTC timing. Transport arrival time is a
separate diagnostic, not a fabricated RF timestamp.

## 4. Work packages, in implementation order

### P0 — Freeze the development baseline and reference methodology

1. Record source manifest, dirty changes, Python/dependencies and existing test
   results in a dated development-evidence directory; do not rename it a release.
2. Pin a concrete edition of LTE physical/coding specifications and versions of
   external reference generators/decoders. Start with classic FDD/normal-CP
   behavior, not every later LTE extension.
3. Use an independent LTE waveform generator and independently verified decoded
   reference outputs. Prefer existing srsRAN tooling or an available licensed
   LTE Toolbox installation. Do not assume MATLAB is installed or licensed.
4. Review dependency licensing/platform availability before integrating native
   code. srsRAN is AGPL/commercial licensed; do not copy it into this repository
   without an explicit compatible distribution decision. It can be evaluated
   as an external reference with its provenance recorded.
5. If no independent generator is available, create a standards-based fixture
   generator but require cross-validation before marking receiver correctness
   qualified. Sharing sequence/coding helpers with the decoder is not an
   independent oracle. Do not fabricate a pass while this prerequisite is absent.

Deliverables: baseline manifest, reference-method note, fixture manifest/schema.

### P1 — Source-neutral streaming and hardware adapter seam

Progress, 19 October 2026: the replay source, cancellable source protocol,
bounded reader/acknowledgement runner and versioned envelope codec are now
implemented. See [the source-interface contract](REPLAY_SOURCE_INTERFACE.md).
The Pluto adapter and its real-device qualification remain pending; replay
transport tests do not supply the independent LTE waveform qualification in P0.

Proposed files: `sources/replay_source.py`, `runtime/source_runner.py`,
`sources/pluto_source.py` (optional dependency, hardware-unverified).

- Reuse `IQSource.read_block()` for IQ, but distinguish finite EOF from timeout,
  disconnected device and cancellation through documented typed outcomes.
- Keep a hardware source's restart semantics separate from rewinding a file.
  Do not silently reuse sample epoch zero on receiver reconnection.
- Implement independent scheduled replay at 1.92 MS/s, using precomputed real
  LTE fixtures so waveform-generation CPU is separately measured. The simulated
  source timeline advances independently; late data cannot be hidden by slowing
  that clock. Preserve the old 1 MS/s benchmark unchanged for comparisons.
- Carry configuration, capture epoch, known/unknown loss and timestamp trust
  through any process boundary. Never assign a guessed size to unknown loss.
- Use bounded input/result channels and explicit overflow terminal outcomes.
  No arbitrary sink, display or feature function runs on the source reader.
- Prepare an injectable libiio/pyadi-iio adapter: configuration/readback, manual
  gain, scaling, timeout, buffer destruction and stop/reconnect. Mock only the
  device transport to test this wrapper; run genuine DSP on supplied IQ.
- Do not import hardware dependencies for ordinary replay/package use. Probe
  Windows/Python compatibility in an isolated environment before pinning an SDR
  optional extra. The currently used Python 3.14 environment is not evidence
  of compatibility with a future native libiio installation.

Gate: partitions/origins/retry/EOF/gaps remain truthful; source cancellation
terminates within its contract; bounded counters balance under overload. Adapter
tests are explicitly transport-mocked, never labelled hardware validation.

### P2 — LTE primitives and channel preparation

Progress, 19 October 2026: standard PSS/SSS, normal-CP mapping and a bounded
causal 2 MS/s -> 1.92 MS/s channel adapter are implemented. Native 1.92 MS/s
remains supported. Exact rational timing preserves original sample coordinates.

Proposed files under `signal_processing/lte/`: `config.py`, `contracts.py`,
`sequences.py`, `ofdm.py`, `channel.py`.

- Implement PSS and SSS sequences and LTE resource mapping from the pinned spec.
- Implement normal-CP OFDM extraction at 1.92 MS/s (128-point FFT), explicit DC
  handling, carrier indexing and symbol/slot/frame mapping.
- Start with input already at 1.92 MS/s and centred on the carrier. Reject
  unsupported input rates before mutation instead of silently resampling.
- Add stateful frequency translation and rational resampling only as a separate
  tested adapter when required. Account for filter delay, settling, rational
  source-coordinate mapping and fractional timing; never splice through gaps.
- Specify DC/IQ impairment handling without an arbitrary notch that removes
  useful central subcarriers. Preserve original input for capture.

Gate: all 504 PCI sequence combinations and independent OFDM/resource vectors;
all applicable CP positions; scale/phase/CFO controls; ordinary and extreme
numeric inputs; direct mathematical comparisons at declared tolerances.

### P3 — Continuous acquisition and tracking

Progress, 19 October 2026: `LTESynchronizer` and the raw-recording CLI implement
continuous FDD/normal-CP acquisition and confirmed-cell tracking. See
[the synchronization contract](LTE_SYNCHRONIZATION.md). PBCH/MIB remains P4;
the reference fixture is sync-bearing OFDM, not a full encoded LTE transmitter.

Proposed files: `lte/synchronization.py`, `lte/receiver.py`.

- Search all three PSS identities, retain a bounded candidate list, then verify
  SSS, half-frame/frame alignment and consistency across repeated observations.
- Search a configured frequency-offset range; combine coarse/integer-offset
  hypotheses with fine correction. CP-based estimation alone has ambiguity and
  is not enough for a broad acquisition range.
- Use normalized correlation metrics, explicit confirmation rules and false-lock
  evaluation. Do not silently reuse energy-detector Gamma Pfa certificates for
  synchronization or equate per-hypothesis thresholds with overall false alarms.
- Implement SEARCHING -> CANDIDATE -> SYNCHRONIZED -> MIB_VALID states, with
  tracking failure/reacquisition and typed fault/stop outcomes. One bad PBCH
  decode need not destroy otherwise valid timing; specify expiry rules.
- Start with the strongest confirmed cell, retaining a small bounded candidate
  set. Characterize near/far and cochannel collisions; do not promise universal
  multi-cell separation. Searching frequency hypotheses must have a finite cost.
- Correct carrier drift and track sample timing over long runs. Gaps invalidate
  affected tracking/combining, but not immutable completed captures.

Gate: genuine IQ recovers correct PCI/timing/CFO across block boundaries and
origin translations; no truth, SNR labels or injected onset reaches the receiver.
Test missed/false lock, loss/reacquisition, empty-result silence and unsupported
waveforms, not just successful detections.

### P4 — PBCH/MIB decoder

Proposed files: `lte/channel_estimation.py`, `lte/pbch.py`, `lte/coding.py`.

- Generate cell-specific reference signals, estimate the channel for supported
  transmit-port hypotheses, extract the PBCH resources and perform appropriate
  equalization/transmit-diversity processing.
- Implement soft demodulation, descrambling, rate recovery, tail-biting
  convolutional decoding and antenna-port CRC-mask handling.
- Handle the PBCH 40 ms structure and unknown initial frame phase. Bound
  combining history and clear it on incompatible cell/epoch/timing changes.
- Recover MIB fields and frame-number interpretation using actual decoded bits
  and timing. Never use the fixture's expected MIB to select a hypothesis.
- Confirm repeated consistent CRC-valid results before announcing stable MIB
  acquisition. Report failures and ambiguity explicitly; one CRC is not an
  unlimited statistical guarantee under arbitrary search.
- Keep raw CRC success distinct from confirmed-cell status. Return only the
  supported fields; later-release optional bits need explicit version semantics.

Gate: independent coding/resource vectors, one/two-port fixtures, all possible
starting PBCH phases, SFN wrap, corrupted bits, noise-only and false-candidate
controls; exact expected MIB and correct submitted stream/cell attribution.

### P5 — Protocol-aligned capture and analysis integration

Proposed files: `capture/interval_assembler.py`, `runtime/lte_pipeline.py`.

- Add interval capture requests referenced to received raw samples. Use a shared
  ring owner; avoid two uncoordinated pipelines duplicating every incoming block.
- Start with a 192,000-sample ring (100 ms at 1.92 MS/s, 1,536,000 IQ bytes).
  Validate the retention requirement against measured decision latency plus
  pre-trigger need. Do not promise old samples if the search outlives retention.
- Start with a 20 ms capture cap (38,400 samples) and four in-flight captures.
  Stage-copy requested prefixes before wraparound, then collect only received
  suffixes. Include all temporary copies and queued IQ in the memory budget.
- Use time-based configuration converted to exact sample counts. Longer windows
  must be deliberate configurations, not unbounded recordings.
- Keep full raw captures separate from optional generic feature windows. The
  default LTE feature policy may select explicitly labelled 4,096-sample windows
  for the existing process backend; retain parent capture ID and subrange. Do not
  claim these are full-capture statistics. Full-capture process analysis needs a
  separately tested bounded payload/lease extension, not just a larger constant.
- Define capture frequency (initial lock, loss/reacquisition, explicit request,
  bounded periodic observation), not one analysis job for every OFDM symbol.
- Preserve exception.completed_captures ownership, terminal-result accounting,
  callback/reset semantics and exact IQ bytes under failure.

Gate: raw byte equality across wrap, large blocks, delayed triggers, partial
EOF/gap captures and resets; no duplicate output or stale cell attribution.

### P6 — UI, measurement semantics and delivery

Proposed file: `ui/lte_panel.py`, with a separate headless CLI entry point.

- Select replay input now; expose a disabled/unavailable hardware connection
  state until an optional adapter and device are present.
- Display actual sample rate/tuning, SEARCHING/LOCK status, PCI, CFO, relative
  frame timing, MIB/CRC status, clipping and continuity uncertainty.
- Present queue/latency/drop/unfinished counters and capture/feature outcomes.
  Throttle plotting independently (for example 5 Hz); never drop required results
  simply to keep plots smooth. Bound UI history by record and byte limits.
- Do not relabel generic excess SNR as LTE SINR. Add reference-based LTE
  measurements only with explicit definitions and tests. Narrow central-band
  observations are not unqualified full-channel RSRQ or bandwidth measurements.
- Export typed result/capture metadata and optional bounded IQ recordings with
  their actual frequency coverage. UI replay does not normalize from truth labels
  or insert quiet training guards.

Gate: headless and UI paths use the same receiver; exports round-trip identity,
units, unavailable reasons and coverage; slow/failing UI cannot block acquisition.

### P7 — Combined software qualification and packaging

Run the complete existing G/F/E and available historical/numerical regression
gate after combined production changes. Preserve frozen originals and documented
migrations; do not claim old dataset recall as LTE performance.

Build/install outside the checkout, verify module parity/import origins, then
run the LTE vectors, replay, lifecycle and representative existing feature tests.
Pin the tested environment, reference-tool versions and fixture hashes. Package
only redistributable fixtures with explicit licences/provenance.

Deliver `LTE_SOFTWARE_QUALIFICATION.md` with independent rows for numerical,
receiver, source-adapter, scheduled throughput, lifecycle and hardware status.
Hardware remains **NOT TESTED — SDR unavailable**, not a green skipped gate.

## 5. Predeclared offline evaluation and performance gates

The following are proposed engineering targets, not previously measured results
or existing university requirements. Freeze them in evaluation configuration
before implementation tuning; report failures without changing the test envelope.

### Correctness and receiver evaluation

- Independent clean sequence/resource/coding vectors must pass completely.
- Use actual LTE downlink waveform fixtures, including central-band extraction
  from wider carriers; not generic TorchSig OFDM relabelled as LTE.
- Record generator, seed, bandwidth, transmit ports, channel taps, impairments,
  sample scaling and hashes. Keep expected fields in a separate scorer-only file.
- Baseline supported-test grid: one cell, FDD/normal CP, one/two transmit ports,
  iid noise, 0/+5/+10 dB measured central-band SNR, CFO -30/0/+30 kHz, random
  initial sample/frame/PBCH phase. For each ports/SNR/CFO cell use at least 400
  independently generated 200 ms opportunities. Randomize PCI; separately cover
  all PCIs deterministically. These limits are a proposed acquisition envelope,
  not a claim about all Pluto units or all RF bands.
- Success requires correct PCI and CRC-valid correct MIB within 200 ms; require
  the one-sided exact 95% lower success bound >=0.90 in every claimed cell.
  Noiseless timing must agree to one input sample. Publish timing/CFO errors in
  impaired cases alongside success; failed attempts remain in the denominator.
- Evaluate -5/-10 dB, intermediate CFOs, timing drift, multipath within/outside CP,
  DC/IQ imbalance, gain changes, clipping and cochannel near/far separately.
  Mark them characterized unless predeclared gates pass. Do not redefine SNR
  after observing results or count correlated frames as independent trials.
- Negative controls: noise, tones and unrelated OFDM. Report false candidates,
  false locks, CRC passes and confirmed false MIBs separately. Proposed false-MIB
  target: one-sided 95% upper rate <=1 per hour under an explicitly stated Poisson
  model, using enough exposure (zero events needs at least about three hours).
  Include independent-record confirmation probabilities without assuming
  arbitrary correlated RF interference is Poisson.
- Repeat each deterministic waveform across one-sample, irregular, 1,024,
  16,384 and oversize blocks and origins 0/1/1000/10^12. Compare logical outputs
  and raw captures, not host wall-clock arrival times.

### Software real-time target

- Independent scheduled replay, 1.92 MS/s, 16,384-sample transport blocks and
  16 input slots. A block represents 8.533 ms; slots cover approximately 136.5 ms
  of input, not permission to accumulate unlimited latency.
- Baseline search/CFO scope above; sustained LTE occupancy; genuine MIB decoding
  and real optional feature jobs at the declared capture cadence (initially at
  most 10 periodic captures/s). State counts and byte caps for every queue.
- Three isolated >=30-second runs and one >=10-minute soak on the declared host.
  Require zero schedule expiry/input loss/overload/unresolved/duplicate results,
  bounded backlog, balanced accounting and complete cleanup. Proposed steady-state
  processing-lag budget: p99 <=50 ms and max <=100 ms measured from the latest
  required input sample becoming available; report acquisition time separately.
- Time from hardware/software buffer fill, startup readiness and algorithmic
  observation time must not be disguised as compute latency. Report per-stage
  CPU and wall time, p50/p95/p99/max, queue peaks and feature/sink costs.
- Profile before optimizing. Prefer batched FFT/correlation and reduced repeated
  allocation; compile/native acceleration only if measured hot paths justify it.
  Preserve a numerically verified reference path. Persistent spawn processes must
  be ready before acquisition, with no per-event process creation.
- Fault runs: slow/blocked/failing sink, full input/output, backend death,
  disconnect emulation, known/unknown gaps, reset races, timeout/retry and EOF.
  Required stages must report failures honestly while recovery remains separately
  measurable. Test polling and callback terminal ownership.
- Separate traced allocation/RSS-responsive memory runs from timing. Verify
  retained counts plateau through quiet, continuous LTE and dense-event workloads.

The new 1.92 MS/s profile does not retroactively pass the old 1 MS/s transient
benchmark. Both have distinct workloads and capability statements.

## 6. What must wait for Pluto

| Hardware gate | Why offline work cannot replace it |
|---|---|
| Discovery, native driver/firmware compatibility and readback | Requires actual device and installed transport stack |
| Sustained 1.92 MS/s USB/DMA continuity | A replay scheduler does not exercise USB, firmware or RF sampling |
| Gain, clipping, IQ/DC errors and clock drift | Simulations cover hypotheses, not this receiver's measured behavior |
| Real carrier acquisition and repeated MIB validation | Must observe actual RF with known tuning and compare independently |
| Overflow visibility and unknown-loss semantics | Verify what this firmware/driver really reports, not assumed counters |
| Antenna/front-end response and received sensitivity | Cannot derive from software tests or normalized dataset SNR |

On arrival: receive-only discovery -> configuration/readback -> controlled
continuity test -> known carrier acquisition -> repeated MIB validation ->
integrated three-run/soak tests -> unplug/reconnect/gap recovery. Do not connect a
transmitter directly to RX without a correctly designed attenuation arrangement.
Document actual RF and host configuration. No real-time-hardware release sign-off
before these checks pass.

## 7. Work deliberately deferred

After central-band MIB qualification, add TDD/extended-CP support as
explicit profiles; then band scanning, full-channel acquisition and SIB1 if
required. A decoded MIB can identify a wider advertised channel but does not
increase captured bandwidth. Full-channel throughput must be separately measured
against Pluto's USB/front-end limits. Uplink, user-data stacks and 5G remain out
of this implementation plan.

## 8. Implementation checkpoints and stop conditions

| Order | Deliverable | Ready to advance when |
|---|---|---|
| 1 | P0/P1 contracts, provenance and replay source | Independent fixture path exists; source/queue tests pass |
| 2 | P2 numerical LTE primitives | Independent vector and coordinate tests pass |
| 3 | P3 acquisition/tracking | PCI/timing/CFO and false-lock evidence recorded |
| 4 | P4 real PBCH/MIB decoding | Exact independently expected MIBs recovered with valid CRC |
| 5 | P5/P6 integrated captures, workers and UI | Exact IQ, attribution, bounded delivery and UI parity pass |
| 6 | P7 software qualification/artifacts | Combined regression, installed tests and declared software gates pass |
| 7 | Hardware validation | Actual Pluto gates pass; cannot complete now |

If independent fixtures or a compatible reference tool are unavailable, report
that blocker instead of validating the receiver solely against its own generator.
If scheduled processing fails, retain its measurements and optimize the limiting
stage before claiming software readiness. Missing SDR does not justify leaving
independent software defects unresolved.

P0/P1 replay, P2/P3 synchronization and the first P4 PBCH/MIB implementation are
now present. See their result reports for exact verification and remaining
qualification limits. Next: protocol-aligned capture and feature integration.
No unrelated dependency upgrade or change to the restored detector UI default.

## References

- LTE sequence/resource/OFDM definitions: [ETSI TS 136 211 V16.6.0](https://www.etsi.org/deliver/etsi_ts/136200_136299/136211/16.06.00_60/ts_136211v160600p.pdf).
- BCH coding/CRC/rate recovery: [ETSI TS 136 212 V18.0.0](https://www.etsi.org/deliver/etsi_TS/136200_136299/136212/18.00.00_60/ts_136212v180000p.pdf). Align editions in P0 before implementing; these links are starting references, not a final dependency lock.
- Practical 1.92 MS/s cell search: [srsRAN LTE example](https://github.com/srsran/srsRAN_4G/blob/master/lib/examples/cell_search.c).
- Reference receiver stages: [LTE cell search, MIB and SIB1 recovery](https://www.mathworks.com/help/lte/ug/cell-search-mib-and-sib1-recovery.html). MATLAB is an optional reference, not an assumed runtime dependency.
- Open-source integration decision: [srsRAN licensing information](https://github.com/srsran/srsRAN_4G/wiki/Licensing-and-contributions).
- Future source interface: [ADI Pluto example](https://github.com/analogdevicesinc/pyadi-iio/blob/main/examples/pluto.py), [ADI receive-buffer implementation](https://github.com/analogdevicesinc/pyadi-iio/blob/main/adi/rx_tx.py), [libiio streaming example](https://analogdevicesinc.github.io/system-level/solutions/platforms/pluto/software/c_install/).
- Hardware scope: [official Pluto hardware specifications](https://wiki.analog.com/university/tools/pluto/hacking/hardware).

References define primitives and examples. They do not certify this package,
this host's timing, arbitrary RF noise, or a hardware device not yet tested.
