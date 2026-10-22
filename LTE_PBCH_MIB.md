# LTE PBCH/MIB decoding

Implemented in the active `signal_processing/lte/` package, development version
2.4.0rc1. This is not a frozen release or hardware qualification.
An opt-in [low-SNR profile](LTE_LOW_SNR.md) now adds repeated synchronization
evidence and received-CP averaging; the compatibility behavior below stays default.

## Public usage

```python
from signal_processing.core import StreamInfo
from signal_processing.lte import LTEBroadcastReceiver

stream = StreamInfo("rx", "recording", 2_000_000., 2.14e9)
receiver = LTEBroadcastReceiver(stream)
for block in positioned_source:  # your ordered IQBlock iterator
    batch = receiver.process_block(block)
    for observation in batch.synchronization:
        print(observation.pci, observation.state)
    for result in batch.mibs:
        print(result.pci, result.system_frame_number,
              result.mib.downlink_bandwidth_hz, result.antenna_ports)
final_batch = receiver.flush()  # process its outputs too
```

`process_envelope(SourceEnvelope)` additionally preserves source/gain/configuration
provenance. Do not mix envelope and plain-block admission within one run.
`LTESynchronizer` retains its existing tuple-returning API; the additive
`LTEBroadcastReceiver` returns an immutable `LTEBroadcastBatch` with tuples
`synchronization` and `mibs`. Input is received IQ plus public configuration:
no cell ID, frame position, SNR, ideal noise power or truth labels are supplied
to the streaming receiver.

```powershell
py -3.14 -m signal_processing.lte --input recording.cf32 --sample-rate 2000000 --decode-mib
# A normal run without a MIB can succeed; explicitly require one when needed:
py -3.14 -m signal_processing.lte --input recording.cf32 --require-mib
```

Raw files are little-endian complex64. Native 1.92 MS/s is also supported.
`--require-mib` returns 2 if no CRC-valid MIB is found, 1 for a processing error,
and 0 on success. `--scheduled --enforce-acquisition` retains independent replay
and its truthful no-loss gate; decoding success does not override source loss.
JSON-lines output includes `type: "mib"` records and a final `decoded_mibs` count.

## Processing and scope

Supported implementation: LTE FDD, normal cyclic prefix, one receive channel,
central six resource blocks, 1/2/4 transmitting CRS ports. The existing causal
2 MS/s to 1.92 MS/s preparation and PSS/SSS/CFO tracking remain authoritative.
After repeated synchronization, the receiver retains enough recent native IQ
to recover subframe-zero PBCH even when confirmation arrives in subframe five.

The decoder performs:

1. Received slot extraction, CP-based residual CFO correction and 128-point FFT.
2. Cell-specific reference generation and frequency/time channel interpolation.
3. PBCH RE extraction, reserving all four CRS ports as required by the mapping.
4. Single-port equalization or two-/four-port SFBC recovery and soft QPSK bits.
5. Four possible 40-ms scrambling phases, convolutional rate recovery, exact
   64-start-state tail-biting Viterbi decoding and CRC-16 antenna-mask checking.
6. Classic MIB parsing, rejecting reserved bandwidth encodings. Ambiguous
   CRC-valid configurations are rejected rather than selected arbitrarily.

Up to four consecutive received frames can be combined, only inside the
hypothesized BCH block. A nonconsecutive frame clears combining history. Grid
extraction uses the measured timing; frame identity/combining uses the existing
10-ms frame lattice within the synchronization timing tolerance. A refined
half-frame timing estimate does not publish the same radio frame a second time.
Grid
normalization is a common numerical scale, not a measured noise floor or SNR.
Frame weights are equal after normalization; this is not an optimal likelihood
model for arbitrary changing channels/noise. No adaptive energy gate or quiet
training prefix is used for this protocol receiver.

`LTEMIB` preserves all 24 bits. Parsed fields are advertised downlink RB count
and nominal bandwidth, PHICH duration/resources, and the eight SFN MSBs.
`LTEPBCHObservation.system_frame_number` combines those with the recovered
two-bit frame phase. Antenna ports are inferred from the CRC mask, not a MIB
payload field. Trailing release-specific/spare bits are preserved but not
interpreted. `to_dict`/`from_dict` preserve exact rational input coordinates,
epoch, profile and source attribution, and reject inconsistent derived fields.

CRC is error detection, not cryptographic authentication or a universal
false-lock certificate. Failed CRC/semantics yield no MIB and increment attempt
counters. A returned MIB is associated with its actual received frame; repeated
frames can broadcast the same payload and are not duplicate transient events.

## Ownership, bounds and faults

One ordered caller owns a receiver. Admission uses the same exact stream,
sequence, positioned-IQ, timestamp (1 ns) and source-provenance checks as
synchronization. Predictable rejection is before state mutation and corrected
retry is allowed. Real processing faults remain FAULTED and require reset.

Each call owns its immutable returned outputs. A later failure attaches genuine
completed-prefix MIBs to `exception.completed_mibs`, alongside the existing
`completed_observations`, received and prepared horizons where provided. Do not
expect flush to republish those results. EOF is idempotent; no fabricated tail
samples or partial-statistic certificates are used. A raw gap finalizes eligible
old-prefix work, then clears raw and combining state; old completed results keep
their original epoch. Reset clears retained input/decoders and begins a new epoch.

Additional persistent raw history is at most 24,000 complex128 native samples
after a search step (plus one bounded search chunk temporarily). At most
`max_cells` decoders hold four frames of three 480-soft-bit hypotheses each.
Calls are bounded by `LTESyncConfig.max_input_samples`; the receiver has no
unbounded output history. A caller is responsible for bounded downstream storage.
Snapshots expose PBCH attempts/successes, retained samples/frames and pending
targets. Counters reset with the raw receiver epoch. `prepared_input_end` is the
actual transport/preparation horizon, which can vary across transport partitions;
decoded windows, fields and exact coordinates must not.

The lower-level `PBCHDecoder(pci).decode(grid, frame_start)` accepts a finite
complex `(5,72)` array: symbols l=0..4 of slot 1, ordered subcarriers -36..-1,
+1..+36, already CFO-corrected at 15-kHz spacing. Frame positions are chronological
native sample indices. It returns `(LTEMIB, ports, frame_mod4, combined_frames)`
or `None`. It is a single-owner numerical decoder, not another positioned source.

## Verification and limitations

Run current source checks:

```powershell
py -3.14 -m pytest -q tests/test_lte_pbch.py tests/test_lte_sync.py
py -3.14 -m benchmarks.evaluate_lte_pbch --trials 10 --output output/lte_pbch_20261019/characterization.json
```

`tests/fixtures/lte_bch_reference.npz` contains 90 independently encoded BCH
vectors. `benchmarks.generate_pbch_reference` downloads pinned unchanged srsRAN
convolutional encoder/rate matcher C, compiles it with an existing MSVC toolchain,
and records URLs/hashes in the companion fixture JSON. CRC is independently
checked using `binascii.crc_hqx`. The production wheel contains neither that
external encoder/DLL nor the test generator. External reference notices stay
under ignored `output/`; frozen numeric fixtures need no compiler at test time.

Synthetic IQ assembly uses independent reference sequences/codewords, separately
implemented Gold generation/resource mapping, CP/OFDM, signed CFO and noise.
It is not an independently complete commercial LTE transmitter, real receiver
recording or RF hardware validation. Exact CRC-valid payloads, all scrambling
quarters/antenna masks, four-frame noisy combining, partition/origin invariance,
gaps/fault recovery, bounds, CLI and installed imports are checked live.

See [fresh results](LTE_PBCH_MIB_RESULTS.md) for actual counts/limitations. This
stage does not establish a low-SNR threshold, certified noise false-alarm rate,
scheduled real-time performance, TDD, extended CP, severe multipath/Doppler,
multi-cell separation, SIB1, user traffic or Pluto operation. The MIB can advertise
20 MHz while this receiver observes only the centre; it does not acquire the full
carrier. Protocol-aligned raw captures and integration with generic feature/UI
workers remain separate work. Existing generic detection/capture/feature APIs
and defaults are unchanged.

Mathematical references: [TS 36.211](https://www.etsi.org/deliver/etsi_ts/136200_136299/136211/17.04.00_60/ts_136211v170400p.pdf)
(§§6.3.4.3, 6.6, 6.10.1, 7.2), [TS 36.212](https://www.etsi.org/deliver/etsi_TS/136200_136299/136212/18.00.00_60/ts_136212v180000p.pdf)
(BCH/CRC/convolutional rate matching), and the pinned external reference recorded
in `tests/fixtures/lte_bch_reference.json`. References are not certification.
