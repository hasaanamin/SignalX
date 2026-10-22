"""Independent time, exact bytes, continuity and cancellation tests for replay."""
from __future__ import annotations

import dataclasses
import threading

import numpy as np
import pytest

from signal_processing.core import IQBlock, StreamInfo
from signal_processing.sources import (
    ArrayIQSource, RecordedFileIQSource, ReplayIQSource, SourceBusy,
    SourceCancelled, SourceClosed, SourceEnvelope, SourceMetadata,
    SourceReadTimeout, decode_envelope, encode_envelope,
)


class ManualClock:
    """Controls time only; samples and source computations remain production code."""
    def __init__(self):
        self.now = 1_000_000_000
        self.overshoot = 0

    def now_ns(self):
        return self.now

    def wait_until_ns(self, deadline, cancellation):
        self.now = max(self.now, deadline) + self.overshoot
        self.overshoot = 0


def replay(iq, *, rate=1_920_000, block=16, origin=0, **kwargs):
    return ReplayIQSource(ArrayIQSource(iq, StreamInfo("lte", "fixture", rate, 915e6),
                                       block_size=block, start_sample=origin,
                                       start_timestamp_ns=1_000_000_000), **kwargs)


def test_file_factory_cold_read_finishes_before_clock_without_losing_first_block(tmp_path,monkeypatch):
    from signal_processing.runtime import ReplayFileFactory
    import signal_processing.sources.replay_source as replay_module
    iq = np.arange(48,dtype=np.complex64)
    path = tmp_path/'cold.cf32'
    iq.tofile(path)
    clock = ManualClock()
    original_read = RecordedFileIQSource.read_block
    cold = True
    def read(recording):
        nonlocal cold
        if cold:
            clock.now += 100_000_000  # A cold read exceeds the real 10-ms budget.
            cold = False
        return original_read(recording)
    monkeypatch.setattr(RecordedFileIQSource,'read_block',read)
    monkeypatch.setattr(replay_module,'MonotonicReplayClock',lambda: clock)
    source = ReplayFileFactory(str(path),StreamInfo('cold','file',1_920_000.),16)()
    try:
        assert source.snapshot().read_samples == 0
        values = []
        while (value := source.read_envelope()) is not None:
            values.append(value)
        np.testing.assert_array_equal(np.concatenate([v.block.iq for v in values]),iq)
        assert values[0].block.sequence_number == 0
        assert values[0].scheduled_monotonic_ns == 1_100_000_000+round(16*1e9/1_920_000)
        assert source.snapshot().expired_samples == 0
    finally:
        source.close()


def test_file_factory_closes_recording_when_warm_read_fails(tmp_path,monkeypatch):
    from signal_processing.runtime import ReplayFileFactory
    path = tmp_path/'bad.cf32'
    np.ones(16,np.complex64).tofile(path)
    recordings = []
    def fail(recording):
        recordings.append(recording)
        raise ValueError('read failed')
    monkeypatch.setattr(RecordedFileIQSource,'read_block',fail)
    with pytest.raises(ValueError,match='read failed'):
        ReplayFileFactory(str(path),StreamInfo('bad','file',1_920_000.))()
    assert len(recordings) == 1 and recordings[0].closed and recordings[0]._file.closed


@pytest.mark.parametrize("block", [1, 8, 16, 29, 1024])
@pytest.mark.parametrize("origin", [0, 1, 1000, 10**12])
def test_replay_preserves_exact_samples_positions_and_rational_schedule(block, origin):
    iq = (np.arange(129)+1j*np.arange(129)[::-1]).astype(np.complex64)
    clock = ManualClock()
    source = replay(iq, block=block, origin=origin, clock=clock, scheduled=True)
    chunks = []
    while (envelope := source.read_envelope()) is not None:
        b = envelope.block
        chunks.append(b.iq)
        end = b.end_sample-origin
        assert envelope.scheduled_monotonic_ns == 1_000_000_000 + round(end*10**9/1_920_000)
        # Existing source anchors are recording metadata, not host arrival time.
        assert b.timestamp_ns == 1_000_000_000 + round((b.start_sample-origin)*10**9/1_920_000)
        assert envelope.metadata.timestamp_origin == "recording"
        with pytest.raises(ValueError):
            b.iq.setflags(write=True)
    assert np.concatenate(chunks).tobytes() == iq.tobytes()
    snap = source.snapshot()
    assert snap.finished and snap.returned_samples == 129
    assert snap.read_samples == snap.returned_samples + snap.expired_samples + snap.pending_samples
    assert snap.expired_samples == snap.unread_samples == 0


def test_source_input_copy_and_replay_reset_epoch():
    iq = np.ones(32, np.complex64)
    source = replay(iq)
    iq[:] = 2
    first = source.read_envelope()
    source.cancel()
    with pytest.raises(SourceCancelled):
        source.read_block()
    source.reset()
    second = source.read_envelope()
    assert first.block.iq.tobytes() == second.block.iq.tobytes() == np.ones(16, np.complex64).tobytes()
    assert second.block.start_sample == first.block.start_sample
    assert second.metadata.source_epoch == first.metadata.source_epoch+1
    assert second.configuration_id == first.configuration_id


def test_timeout_retains_pending_block_for_correct_retry():
    source = replay(np.ones(16, np.complex64), rate=16, clock=ManualClock(), scheduled=True)
    with pytest.raises(SourceReadTimeout):
        source.read_envelope(timeout=.1)
    assert source.snapshot().pending_samples == 16
    result = source.read_envelope(timeout=1)
    assert result.block.start_sample == 0
    assert source.snapshot().returned_samples == 16
    assert source.read_block() is None


def test_independent_schedule_and_post_sleep_expiry_cannot_hide_slow_consumer():
    clock = ManualClock()
    source = replay(np.ones(64, np.complex64), rate=16, scheduled=True,
                    lateness_budget_s=.05, clock=clock)
    first = source.read_envelope()
    assert first.block.start_sample == 0
    clock.now += 2_100_000_000  # No read while the independent radio clock advances.
    last = source.read_envelope()
    assert last.block.start_sample == 48
    assert last.block.dropped_samples_before == 32
    assert last.scheduled_monotonic_ns == 5_000_000_000
    assert source.snapshot().expired_samples == 32
    assert source.read_block() is None

    clock = ManualClock()
    clock.overshoot = 51_000_000
    source = replay(np.ones(16, np.complex64), rate=16, scheduled=True,
                    lateness_budget_s=.05, clock=clock)
    assert source.read_block() is None
    assert source.snapshot().trailing_lost_samples == 16
    assert source.snapshot().expired_samples == 16


def test_cancel_wait_busy_reset_and_close_have_bounded_real_thread_completion():
    class GatedClock:
        def now_ns(self):
            return 1_000_000_000

        def wait_until_ns(self, deadline, cancellation):
            entered.set()
            assert cancellation.wait(2)

    entered = threading.Event()
    errors = []
    source = replay(np.ones(16, np.complex64), rate=1, scheduled=True, clock=GatedClock())

    def read():
        try:
            source.read_block()
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=read)
    thread.start()
    try:
        assert entered.wait(2)
        with pytest.raises(SourceBusy):
            source.reset()
        with pytest.raises(SourceBusy):
            source.read_block()
        source.cancel()
        thread.join(2)
        assert not thread.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], SourceCancelled)
        source.close()
        with pytest.raises(SourceClosed):
            source.read_block()
    finally:
        source.cancel()
        thread.join(2)


def test_raw_file_replay_empty_file_and_owned_close(tmp_path):
    path = tmp_path / "recording.cf32"
    iq = np.array([1+2j, -3+4j, 5-6j], np.complex64)
    iq.tofile(path)
    file_source = RecordedFileIQSource(path, StreamInfo("s", "file", 1.92e6), block_size=2)
    with ReplayIQSource(file_source, owns_source=True) as source:
        assert source.read_block().iq.tobytes() == iq[:2].tobytes()
        assert source.read_block().iq.tobytes() == iq[2:].tobytes()
        assert source.read_block() is None
    assert file_source.closed
    empty = tmp_path / "empty.cf32"
    empty.touch()
    with ReplayIQSource(RecordedFileIQSource(empty, file_source.stream_info), owns_source=True) as source:
        assert source.read_block() is None
        assert source.snapshot().total_samples == 0


@pytest.mark.parametrize("kwargs", [{"lateness_budget_s": -1}, {"lateness_budget_s": float("nan")},
                                   {"lateness_budget_s": True}, {"scheduled": 1}, {"owns_source": 1}])
def test_replay_constructor_rejects_invalid_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        replay(np.ones(16, np.complex64), **kwargs)


def test_wire_envelope_preserves_all_acquisition_metadata_and_unknown_loss():
    b = IQBlock(StreamInfo("s", "device", 1.92e6, 2.1e9, "rx0"), 11, 10**12,
                1_234_567_890, np.array([1+2j, 3-4j], np.complex64),
                dropped_samples_before=23, overflow=True)
    meta = SourceMetadata(7, 2, "receiver", "manual", 35, 1/2048)
    original = SourceEnvelope(b, meta, 7_000_000_000, 6_999_000_000)
    result = decode_envelope(encode_envelope(original, max_samples=16), max_samples=16)
    assert result.block.stream == b.stream
    assert result.block.iq.tobytes() == b.iq.tobytes()
    for name in ("sequence_number", "start_sample", "timestamp_ns", "dropped_samples_before", "overflow"):
        assert getattr(result.block, name) == getattr(b, name)
    assert result.metadata == meta
    assert result.received_monotonic_ns == original.received_monotonic_ns
    assert result.scheduled_monotonic_ns == original.scheduled_monotonic_ns
    assert result.configuration_id == original.configuration_id
    assert not result.block.iq.flags.writeable
    assert dataclasses.replace(original, metadata=dataclasses.replace(meta, gain_db=36)).configuration_id != original.configuration_id


@pytest.mark.parametrize("payload", [b"", b"bad"*100, b"SXI1\xff\xff\xff\xffx"])
def test_malformed_wire_data_rejected(payload):
    with pytest.raises(ValueError):
        decode_envelope(payload, max_samples=16)


def test_wire_caps_and_metadata_validation():
    b = IQBlock(StreamInfo("s", "f", 1e6), 0, 0, None, np.ones(16, np.complex64))
    e = SourceEnvelope(b)
    with pytest.raises(ValueError):
        encode_envelope(e, max_samples=8)
    with pytest.raises(ValueError):
        decode_envelope(encode_envelope(e, max_samples=16), max_samples=8)
    for kwargs in ({"source_epoch": True}, {"source_epoch": .5}, {"gain_db": float("nan")},
                   {"adc_scale": 0}, {"timestamp_origin": "host_arrival"}):
        with pytest.raises(ValueError):
            SourceMetadata(**kwargs)
    with pytest.raises(ValueError):
        SourceEnvelope(dataclasses.replace(b, timestamp_ns=12))
