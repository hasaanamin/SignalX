"""Public bounded source admission, terminal ownership and real pipeline tests."""
from __future__ import annotations

import dataclasses
import multiprocessing
import threading
import time

import numpy as np
import pytest

from signal_processing.core import IQBlock, StreamInfo
from signal_processing.detection import GaussianNoiseEstimator
from signal_processing.runtime import SourceRunner, SourceRunnerFailed, StreamingCapturePipeline
from signal_processing.sources import (
    ArrayIQSource, ReplayIQSource, SourceCancelled, SourceDisconnected,
    SourceEnvelope, SourceMetadata, SourceReadTimeout, decode_envelope, encode_envelope,
)


STREAM = StreamInfo("s", "test", 1_920_000, 915e6)


class GatedSource:
    """Public transport gate; each read returns real positioned IQ or an exception."""
    stream_info = STREAM

    def __init__(self):
        import queue
        self.items = queue.Queue(maxsize=16)
        self.entered = threading.Event()
        self.returned = threading.Event()
        self.cancelled = threading.Event()
        self.closed = False

    def read_envelope(self, timeout=None):
        import queue
        self.entered.set()
        if self.cancelled.is_set():
            raise SourceCancelled("Transport cancelled")
        try:
            item = self.items.get(timeout=timeout)
        except queue.Empty:
            raise SourceReadTimeout("No transport data") from None
        if isinstance(item, Exception):
            raise item
        self.returned.set()
        return item

    def cancel(self):
        self.cancelled.set()

    def close(self):
        self.closed = True


def envelope(seq=0, start=0, *, lost=0, unknown=False, metadata=None):
    b = IQBlock(STREAM, seq, start, None, np.full(16, seq+1, np.complex64), lost, unknown)
    return SourceEnvelope(b, metadata or SourceMetadata())


def wait_snapshot(runner, condition, timeout=3):
    # Bounded polling of observable counters, never used as a synchronization oracle
    # for callback completion; it only waits for a requested accounting checkpoint.
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        snap = runner.snapshot()
        if condition(snap):
            return snap
        threading.Event().wait(.002)
    raise AssertionError(f"Source runner did not reach checkpoint: {runner.snapshot()}")


def test_absent_consumer_retention_bounded_and_trailing_loss_visible():
    source = ReplayIQSource(ArrayIQSource(np.ones(1600, np.complex64), STREAM, 16))
    runner = SourceRunner(source, capacity=2, max_block_samples=16).start()
    try:
        snap = wait_snapshot(runner, lambda s: s.terminal_reason == "eof")
        assert snap.admitted_samples == 32
        assert snap.queue_dropped_samples == snap.trailing_lost_samples == 1568
        assert snap.peak_outstanding == 2
        assert snap.accounting_balanced
        for _ in range(2):
            delivery = runner.get(timeout=1)
            runner.ack(delivery)
        assert runner.get(timeout=1) is None
        runner.shutdown()
        assert runner.snapshot().outstanding_deliveries == 0
        assert runner.snapshot().accounting_balanced
    finally:
        runner.shutdown(drain=False)


def test_queue_loss_and_unknown_transport_loss_survive_to_next_accepted_block():
    source = GatedSource()
    runner = SourceRunner(source, capacity=1, max_block_samples=16).start()
    try:
        source.items.put(envelope())
        held = runner.get(timeout=2)
        # A checked-out lease still owns the capacity reservation.
        source.items.put(envelope(1, 16, unknown=True))
        wait_snapshot(runner, lambda s: s.queue_dropped_samples == 16)
        source.items.put(envelope(2, 39, lost=7))
        wait_snapshot(runner, lambda s: s.queue_dropped_samples == 32)
        runner.ack(held)
        source.items.put(envelope(3, 55))
        next_delivery = runner.get(timeout=2)
        b = next_delivery.envelope.block
        assert b.start_sample == 55 and b.dropped_samples_before == 39
        assert b.overflow is True
        assert b.iq.tobytes() == np.full(16, 4, np.complex64).tobytes()
        runner.ack(next_delivery)
        source.items.put(None)
        assert runner.get(timeout=2) is None
        runner.shutdown()
        snap = runner.snapshot()
        assert snap.accounting_balanced and snap.trailing_lost_samples == 0
    finally:
        runner.shutdown(drain=False)


def test_shutdown_timeout_keeps_closing_until_real_lease_is_completed():
    source = GatedSource()
    runner = SourceRunner(source, capacity=2).start()
    try:
        source.items.put(envelope())
        held = runner.get(timeout=2)
        with pytest.raises(TimeoutError):
            runner.shutdown(timeout=.02)
        assert runner.snapshot().state == "CLOSING"
        assert runner.snapshot().outstanding_deliveries == 1
        runner.ack(held)
        runner.shutdown(timeout=2)
        assert runner.snapshot().state == "CLOSED"
        assert runner.snapshot().accounting_balanced
    finally:
        if runner.snapshot().outstanding_deliveries:
            runner.ack(held, outcome="cancelled")
        runner.shutdown(drain=False)


def test_source_failure_preserves_accepted_prefix_and_original_cause():
    source = GatedSource()
    runner = SourceRunner(source, capacity=2).start()
    try:
        source.items.put(envelope())
        source.items.put(SourceDisconnected("USB went away"))
        first = runner.get(timeout=2)
        runner.ack(first, outcome="failed")
        with pytest.raises(SourceRunnerFailed) as exc:
            runner.get(timeout=2)
        assert isinstance(exc.value.__cause__, SourceDisconnected)
        runner.shutdown()
        snap = runner.snapshot()
        assert snap.terminal_reason == "failed" and snap.failed_samples == 16
        assert snap.accounting_balanced
    finally:
        runner.shutdown(drain=False)


def test_timeout_not_eof_and_cancel_owned_source_cleanup():
    source = GatedSource()
    runner = SourceRunner(source, owns_source=True).start()
    assert source.entered.wait(2)
    with pytest.raises(SourceReadTimeout):
        runner.get(timeout=.01)
    runner.shutdown(timeout=2)
    assert source.closed
    assert runner.get(timeout=0) is None
    assert runner.snapshot().terminal_reason == "cancelled"


def test_ack_identity_and_duplicate_rejection_are_atomic():
    source = GatedSource()
    runner = SourceRunner(source).start()
    try:
        source.items.put(envelope())
        lease = runner.get(timeout=2)
        before = runner.snapshot().processed_samples
        with pytest.raises(ValueError):
            runner.ack(dataclasses.replace(lease))
        assert runner.snapshot().processed_samples == before
        runner.ack(lease)
        with pytest.raises(ValueError):
            runner.ack(lease)
        assert runner.snapshot().processed_samples == 16
    finally:
        runner.shutdown(drain=False)


def test_abort_cancels_queued_but_never_pretends_held_work_completed():
    source = GatedSource()
    runner = SourceRunner(source, capacity=2).start()
    try:
        source.items.put(envelope())
        held = runner.get(timeout=2)
        source.items.put(envelope(1, 16))
        wait_snapshot(runner, lambda s: s.admitted_samples == 32)
        with pytest.raises(TimeoutError):
            runner.shutdown(drain=False, timeout=.02)
        snap = runner.snapshot()
        assert snap.cancelled_samples == 16 and snap.in_flight_samples == 16
        assert snap.queued_samples == 0 and snap.accounting_balanced
        runner.ack(held, outcome="cancelled")
        runner.shutdown(timeout=2)
        assert runner.snapshot().cancelled_samples == 32
    finally:
        if runner.snapshot().in_flight_samples:
            runner.ack(held, outcome="cancelled")
        runner.shutdown(drain=False)


def test_cleanup_failure_requires_retry_and_does_not_claim_closed():
    class BadClose(GatedSource):
        allow_close = False

        def close(self):
            if not self.allow_close:
                raise OSError("close failed")
            super().close()

    source = BadClose()
    runner = SourceRunner(source, owns_source=True).start()
    assert source.entered.wait(2)
    with pytest.raises(SourceRunnerFailed, match="cleanup failed"):
        runner.shutdown(timeout=2)
    assert runner.snapshot().state == "CLOSING"
    source.allow_close = True
    runner.shutdown(timeout=2)
    assert source.closed and runner.snapshot().state == "CLOSED"
    assert runner.snapshot().terminal_reason == "failed"


def test_block_finishing_during_cancel_is_counted_separately_from_queue_overload():
    class FinishingRead(GatedSource):
        def read_envelope(self, timeout=None):
            self.entered.set()
            assert self.cancelled.wait(2)
            return envelope()

    source = FinishingRead()
    runner = SourceRunner(source).start()
    assert source.entered.wait(2)
    runner.shutdown(timeout=2)
    snap = runner.snapshot()
    assert snap.admitted_samples == snap.queue_dropped_samples == 0
    assert snap.unaccepted_cancelled_samples == 16 and snap.accounting_balanced


def test_start_validation_and_shutdown_before_start():
    source = GatedSource()
    for kwargs in ({"capacity": True}, {"capacity": 0}, {"max_block_samples": 0}):
        with pytest.raises(ValueError):
            SourceRunner(source, **kwargs)
    runner = SourceRunner(source, owns_source=True)
    with pytest.raises(Exception, match="Start the source runner"):
        runner.get(timeout=0)
    runner.shutdown()
    assert source.closed and runner.snapshot().state == "CLOSED"


@pytest.mark.parametrize("change", ["rate", "epoch", "overlap", "oversize"])
def test_source_contract_violation_is_typed_failure_without_accepting_bad_block(change):
    source = GatedSource()
    runner = SourceRunner(source, max_block_samples=16).start()
    try:
        source.items.put(envelope())
        first = runner.get(timeout=2)
        runner.ack(first)
        bad = envelope(1, 16)
        if change == "rate":
            bad = dataclasses.replace(bad, block=dataclasses.replace(bad.block,
                stream=dataclasses.replace(STREAM, sample_rate_hz=STREAM.sample_rate_hz+.0005)))
        elif change == "epoch":
            bad = dataclasses.replace(bad, metadata=SourceMetadata(source_epoch=1))
        elif change == "overlap":
            bad = dataclasses.replace(bad, block=dataclasses.replace(bad.block, start_sample=8))
        else:
            bad = dataclasses.replace(bad, block=dataclasses.replace(bad.block, iq=np.ones(17, np.complex64)))
        source.items.put(bad)
        with pytest.raises(SourceRunnerFailed):
            runner.get(timeout=2)
        assert runner.snapshot().admitted_samples == 16
    finally:
        runner.shutdown(drain=False)


def _wire_receiver(connection):
    """Spawn entry point: actual explicit envelope serialization in both directions."""
    try:
        wire = connection.recv_bytes(8192+8+16*8)
        e = decode_envelope(wire, max_samples=16)
        connection.send_bytes(encode_envelope(e, max_samples=16))
    finally:
        connection.close()


def test_spawn_roundtrip_does_not_lose_timestamp_unknown_loss_gain_or_epochs():
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_wire_receiver, args=(child,))
    proc.start()
    child.close()
    original = envelope(4, 10**12, lost=19, unknown=True,
                        metadata=SourceMetadata(9, 3, "unknown", "manual", 24, 1/2048))
    original = dataclasses.replace(original, block=dataclasses.replace(original.block, timestamp_ns=123456),
                                   metadata=dataclasses.replace(original.metadata, timestamp_origin="receiver"))
    try:
        parent.send_bytes(encode_envelope(original, max_samples=16))
        assert parent.poll(10)
        restored = decode_envelope(parent.recv_bytes(8192+8+16*8), max_samples=16)
        proc.join(5)
        assert proc.exitcode == 0
        assert restored.configuration_id == original.configuration_id
        assert restored.block.timestamp_ns == 123456
        assert restored.block.overflow and restored.block.dropped_samples_before == 19
        assert restored.metadata == original.metadata
        assert restored.block.iq.tobytes() == original.block.iq.tobytes()
    finally:
        parent.close()
        if proc.is_alive():
            proc.terminate()
            proc.join(2)


def test_real_detection_capture_features_through_source_runner_matches_direct():
    rng = np.random.default_rng(774)
    iq = (rng.normal(0, .003, 2048)+1j*rng.normal(0, .003, 2048)).astype(np.complex64)
    iq[768:1280] += .2

    def pipeline():
        return StreamingCapturePipeline(STREAM, window_size=64, hop_size=16,
            ring_capacity=4096, pre_trigger_samples=128, pre_signal_margin_samples=0,
            post_signal_samples=32, min_post_trigger_samples=32,
            start_persistence=1, end_persistence=1,
            noise_estimator=GaussianNoiseEstimator(window_size=64, calibrated_noise_power_linear=1.8e-5),
            synchronous_analysis=True)

    direct = pipeline()
    through = pipeline()
    expected = []
    actual = []
    runner = SourceRunner(ReplayIQSource(ArrayIQSource(iq, STREAM, 128, start_sample=10**12)),
                          capacity=32, max_block_samples=128).start()
    try:
        source = ArrayIQSource(iq, STREAM, 128, start_sample=10**12)
        while (b := source.read_block()) is not None:
            expected.extend(direct.process_block(b))
        expected.extend(direct.flush())
        while (lease := runner.get(timeout=2)) is not None:
            actual.extend(through.process_block(lease.envelope.block))
            runner.ack(lease)
        actual.extend(through.flush())
        runner.shutdown()
        assert len(actual) == len(expected) == 1
        a, e = actual[0], expected[0]
        assert a.event.event_start_sample == e.event.event_start_sample
        assert a.event.event_end_sample == e.event.event_end_sample
        assert a.event.iq.tobytes() == e.event.iq.tobytes()
        start = a.event.event_start_sample-10**12
        assert a.event.iq.tobytes() == iq[start:start+a.event.num_samples].tobytes()
        assert a.feature_status == e.feature_status == "ok"
        assert len(a.analysis_result.feature_vector.features) == 56
        for key, value in a.analysis_result.feature_vector.features.items():
            other = e.analysis_result.feature_vector.features[key]
            assert value == other
        assert runner.snapshot().accounting_balanced
    finally:
        runner.shutdown(drain=False)
        direct.shutdown()
        through.shutdown()
