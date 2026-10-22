# Replay and source interface

Implemented 19 October 2026. This is the source-interface portion of
[the LTE plan](LTE_IMPLEMENTATION_PLAN.md). LTE synchronization/PBCH decoding and
Pluto hardware validation remain pending.

## Public interfaces

`ReplayIQSource` wraps the existing finite `ArrayIQSource` or
`RecordedFileIQSource`. Unpaced replay returns original positioned IQ as fast as
the caller reads. Scheduled replay releases a block when its last sample would
have arrived on an independent monotonic clock. There is no waveform generation,
noise calibration, truth-based scheduling or label normalization in the source.

```python
from signal_processing.core import StreamInfo
from signal_processing.sources import RecordedFileIQSource, ReplayIQSource
from signal_processing.runtime import SourceRunner

stream = StreamInfo("lte-session-1", "recording", 1_920_000, 2_140_000_000)
recording = RecordedFileIQSource("capture.cf32", stream, block_size=16384)
replay = ReplayIQSource(recording, scheduled=True, lateness_budget_s=0.01,
                        owns_source=True)
runner = SourceRunner(replay, capacity=16, max_block_samples=16384,
                      owns_source=True).start()
try:
    while (delivery := runner.get(timeout=2)) is not None:
        try:
            block = delivery.envelope.block
            # Feed a receiver/pipeline here. Persist envelope provenance alongside
            # results if required; IQBlock alone does not contain all provenance.
            consume(block)
        except Exception:
            runner.ack(delivery, outcome="failed")
            raise
        else:
            runner.ack(delivery, outcome="processed")
finally:
    runner.shutdown(drain=False, timeout=5)
```

`consume` is application code in the consumer thread; no sink, feature extraction
or UI function runs on the source reader. For lossless offline analysis without
an acquisition clock, use `ReplayIQSource(..., scheduled=False).read_block()`
directly in the processing loop. An unpaced source inside a drop-newest runner
can intentionally outrun processing.

The raw file is little-endian complex64, interleaved float32 I/Q. Set the actual
sample rate and tuning metadata. Relabelling a recording does not resample it.
The existing file source supports explicit big endian when constructed directly.
It memory-maps recordings; replay does not load the whole file into RAM.

## Time, continuity and provenance

- `read_envelope(timeout=...)` returns `SourceEnvelope`, or `None` only at finite
  EOF. `SourceReadTimeout` is retryable, `SourceCancelled` is explicit cancellation
  and `SourceDisconnected` is transport failure. Pending data survives a timeout.
  Cancellation interrupts schedule waits within 50 ms; an already executing
  finite file read must finish before its cancellation check. Timeouts do not
  forcibly interrupt operating-system disk I/O.
- `read_block()` is the existing IQSource-compatible convenience API; it omits
  access to the extra envelope fields. Both APIs use one cursor. Concurrent
  reading/reset rejects with `SourceBusy`.
- A replay clock starts on the first read, before preparation of that first
  block. Every subsequent release is calculated from its absolute sample offset
  and the same initial monotonic anchor. A slow consumer cannot move this clock.
- The lateness budget is explicit. After every wait, expired whole blocks are
  counted and skipped. Later reads do not replay them as if they arrived on time.
  `ReplaySnapshot` reports total/read/returned/expired/pending/unread samples,
  maximum lateness and trailing loss. EOF cannot shrink the declared recording.
- Recording/device timestamps stay in `IQBlock.timestamp_ns`. Host availability
  is `received_monotonic_ns`; scheduled availability is `scheduled_monotonic_ns`.
  A host arrival is never promoted to a receiver clock anchor. Array/file timestamp
  conversion now uses the existing rational sample-time helper, including ties
  at 1.92 MS/s; it does not accumulate per-packet rounding.
- `SourceMetadata` explicitly identifies timestamp origin (`unknown`, `recording`,
  `receiver`), source/configuration epochs, gain mode/value and applied ADC scale.
  Unknown timestamp origin requires an absent IQ timestamp. Declaring recording
  or receiver origin identifies provenance, not independently certified accuracy.
- `source_epoch` increments when replay is explicitly reset. Hardware cannot
  rewind RF; the future hardware adapter must restart with a new source epoch.
  Source epoch is separate from capture-ring epoch and worker cancellation
  generation. `configuration_id` hashes stream metadata and gain/scale/configuration
  epoch; source restart alone leaves that configuration identity unchanged.
- SourceRunner rejects mid-run epoch/configuration changes. Stop, drain/cancel
  admitted work, and construct a runner/receiver for the new configuration. It
  does not silently reinterpret new rates or tuner settings. This is additive;
  the existing pipeline retains its own exact admission and timestamp checks.
- `dropped_samples_before` is a known extent; `overflow=True` preserves overflow
  uncertainty, including when its additional extent is unknown. The source/runner
  never invents samples or an unknown-loss count. A receiver must handle the
  actual markers before combining window/filter state.

## Bounds, outcomes and shutdown

`SourceRunner` has one reader thread and a finite number of reservations. Queued
and checked-out deliveries share `capacity`. A consumer cannot evade the limit
by repeatedly calling `get()` without acknowledging. Every accepted delivery has
one terminal acknowledgement: processed, failed or cancelled. Copied, foreign and
double acknowledgements reject before changing counters.

When capacity is full, the incoming block is unaccepted and counted under
`queue_dropped_samples`. Its known loss and overflow uncertainty accumulate into
the next admitted block. This call path invokes no external notification sink.
Trailing loss remains in the final snapshot when there is no later block.

An input block that finishes after shutdown was requested is counted separately
as `unaccepted_cancelled_samples`, not as queue saturation. A source/decoder
contract error fails the run and does not admit that invalid block. Reader errors
are raised as `SourceRunnerFailed` with the underlying exception as the cause,
after already queued valid input is retrieved. Earlier valid leases remain
acknowledgeable. EOF means no further deliveries, not completion of held leases.

`shutdown(drain=True)` stops new reads and waits for accepted work. Another
consumer must finish queued/held work while it drains. `drain=False` cancels
queued deliveries; a checked-out delivery still requires an acknowledgement.
The timeout is one monotonic budget across reader join and work drain. On timeout
the runner remains CLOSING. Complete/release held work and retry shutdown. Owned
source cleanup failures also prevent CLOSED and can be retried. Source-provided
cancel/close must themselves follow a bounded contract; Python cannot forcibly
terminate an arbitrary blocking device callback. A future adapter must configure
native read timeouts/cancellation accordingly. Callbacks never hold runner locks.

Snapshots maintain the following separate balances:

```text
replay total = read + unread
replay read = returned + expired + pending
runner valid read = admitted + queue dropped + unaccepted cancelled
admitted = processed + failed + cancelled + queued + checked out
```

Queue loss and replay expiry are distinct. Original declared RF loss is additional
source metadata, not another copy of recording samples in these balances.
Runner snapshots count validated input; a rejected malformed packet is identified
by the source failure, not classified as successfully received/processed IQ.

With capacity C and max block length B, runner-owned accepted IQ is at most C*B
samples. Allow additional bounded blocks for source preparation, previous-block
validation and temporary loss-marker reconstruction. Replay itself owns at most
a pending block and its previous validated block, plus its underlying source.
ArrayIQSource deliberately owns the complete finite input; use a file source for
large recordings. Consumer-retained completed data is the consumer's responsibility.
No outcome list, recording history or per-block thread is added by the runner.

## Future device interface and process transport

`PositionedIQSource` is the structural protocol for the future Pluto adapter:
`stream_info`, `read_envelope(timeout)`, `cancel()` and `close()`. Reset/rewind is
intentionally absent from this hardware-facing contract. Actual tuning/gain
readback, transport scaling and hardware loss visibility must be implemented and
measured when adding the adapter. There is currently no `adi.Pluto` dependency
or claimed hardware implementation in this change.

`encode_envelope` / `decode_envelope` provide the versioned `SXI1` wire format:
bounded JSON metadata and little-endian complex64 payload, with a caller-specified
sample cap and an 8,192-byte header cap. They preserve the full envelope through
spawn IPC and reconstruct validated immutable IQ without pickling event contexts.
Use transport `recv_bytes` size limits as in the spawn regression. The format
does not itself establish authenticity or hardware continuity.

The legacy `SharedIQTransport` and compatibility capture process are unchanged.
They retain their older tuple format. Use the new envelope codec for a future
LTE process boundary; do not pass it through the old tuple conversion and assume
the extra metadata survives. Production replay currently uses one reader thread.

## Running and verifying

```powershell
py -3.14 -m pytest -q tests/test_replay_source.py tests/test_source_runner.py tests/test_replay_evaluation.py

# Seeded noise fixture; measures acquisition only, not LTE decoding.
py -3.14 -m benchmarks.evaluate_replay --seconds 1 --enforce-acceptance

# Same source contract with an actual raw recording.
py -3.14 -m benchmarks.evaluate_replay --input capture.cf32 --sample-rate 1920000 --enforce-acceptance

# Deliberately slow consumer: enforcement should fail if input is lost.
py -3.14 -m benchmarks.evaluate_replay --seconds 1 --consumer-delay 0.04 --capacity 2 --enforce-acceptance

# Optional existing capture/feature pipeline with an explicit external reference.
# This does not invoke an LTE decoder or certify the supplied noise reference.
py -3.14 -m benchmarks.evaluate_replay --seconds 1 --noise-power 0.000018 --enforce-acceptance

# Run from an installed verification wheel, with no production checkout import.
py -3.14 -I benchmarks/verify_replay_install.py PATH_TO_INSTALLED_TARGET .
```

The benchmark records input/expiry/queue/terminal balances, startup preparation,
elapsed rate, queue peak and bounded-window latency percentiles (with a whole-run
maximum). Its optional DSP path uses actual synchronous capture/features and drains
the polling channel. It is diagnostic; existing independently scheduled acceptance
criteria are unchanged. Passing a short source-only run does not qualify cellular
DSP, a long soak, or USB hardware reception.

Since the LTE synchronization stage was added, the replay benchmark's default
rate is 2 MS/s. Specify `--sample-rate 1920000` to reproduce the earlier source
configuration. Explicit recording metadata and existing datasets are unchanged.
