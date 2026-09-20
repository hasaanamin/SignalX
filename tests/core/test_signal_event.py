"""Unit tests for SignalEvent contract."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from signal_processing.core import ContractValidationError, SignalEvent, StreamInfo


@pytest.fixture
def sample_stream() -> StreamInfo:
    return StreamInfo(
        stream_id="stream-01",
        source_id="synthetic-source",
        sample_rate_hz=1_000_000.0,  # 1 MHz -> 1 us per sample
        center_frequency_hz=915_000_000.0,
    )


def test_signal_event_valid_pre_signal_post(sample_stream: StreamInfo) -> None:
    """Test valid SignalEvent with pre-trigger, detected signal, and post-trigger margins."""
    # Event layout:
    # event_start=1000, detected_start=1200, trigger=1300, detected_end=1700, event_end=1900
    # pre-trigger samples = 1200 - 1000 = 200
    # signal samples = 1700 - 1200 = 500
    # post-trigger samples = 1900 - 1700 = 200
    # total samples = 900
    total_samples = 900
    iq_data = np.arange(total_samples, dtype=np.complex64)

    event = SignalEvent(
        event_id="ev-001",
        stream=sample_stream,
        iq=iq_data,
        event_start_sample=1000,
        event_end_sample=1900,
        detected_start_sample=1200,
        detected_end_sample=1700,
        trigger_sample=1300,
        timestamp_ns=100_000_000,
        dropped_samples=0,
        complete=True,
    )

    # Derived sample counts
    assert event.num_samples == 900
    assert event.pre_signal_samples == 200
    assert event.pre_trigger_samples == 300
    assert event.signal_samples == 500
    assert event.post_signal_samples == 200
    assert event.post_trigger_samples == 600
    assert (
        event.pre_signal_samples + event.signal_samples + event.post_signal_samples
        == event.num_samples
    )

    # Offsets
    assert event.signal_start_offset == 200
    assert event.signal_end_offset == 700

    # Durations (at 1 MHz: 1 us per sample)
    assert event.duration_s == pytest.approx(900 / 1e6)
    assert event.pre_signal_duration_s == pytest.approx(200 / 1e6)
    assert event.pre_trigger_duration_s == pytest.approx(300 / 1e6)
    assert event.signal_duration_s == pytest.approx(500 / 1e6)
    assert event.post_signal_duration_s == pytest.approx(200 / 1e6)
    assert event.post_trigger_duration_s == pytest.approx(600 / 1e6)


    # Signal IQ slice
    sig_iq = event.signal_iq
    assert len(sig_iq) == 500
    # IQ was arange(900), signal is slice [200:700]
    np.testing.assert_array_equal(sig_iq, np.arange(200, 700, dtype=np.complex64))
    assert not sig_iq.flags.writeable


def test_signal_event_read_only_iq(sample_stream: StreamInfo) -> None:
    """Test that event.iq and event.signal_iq cannot be modified."""
    iq_data = np.ones(100, dtype=np.complex64)
    event = SignalEvent(
        event_id="ev-002",
        stream=sample_stream,
        iq=iq_data,
        event_start_sample=0,
        event_end_sample=100,
        detected_start_sample=10,
        detected_end_sample=90,
        trigger_sample=50,
    )

    # 1. Stored array is non-writeable
    assert not event.iq.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        event.iq[0] = 5.0

    # 2. Slice is non-writeable
    assert not event.signal_iq.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        event.signal_iq[0] = 5.0

    # 3. Caller modifying original array does not mutate event
    iq_data[0] = 999.0
    assert event.iq[0] == 1.0


def test_signal_event_immutability(sample_stream: StreamInfo) -> None:
    """Test that SignalEvent is frozen."""
    event = SignalEvent(
        event_id="ev-003",
        stream=sample_stream,
        iq=np.ones(10, dtype=np.complex64),
        event_start_sample=0,
        event_end_sample=10,
        detected_start_sample=2,
        detected_end_sample=8,
        trigger_sample=5,
    )
    with pytest.raises(FrozenInstanceError):
        event.event_id = "ev-changed"  # type: ignore[misc]


def test_signal_event_iq_length_mismatch(sample_stream: StreamInfo) -> None:
    """Test rejection when len(iq) != event_end - event_start."""
    iq_data = np.ones(50, dtype=np.complex64)  # 50 samples
    with pytest.raises(ContractValidationError, match="length"):
        SignalEvent(
            event_id="ev-004",
            stream=sample_stream,
            iq=iq_data,
            event_start_sample=0,
            event_end_sample=60,  # declares 60 samples
            detected_start_sample=10,
            detected_end_sample=40,
            trigger_sample=20,
        )


@pytest.mark.parametrize(
    ("ev_start", "det_start", "trig", "det_end", "ev_end"),
    [
        (100, 50, 120, 150, 200),   # detected_start < event_start
        (100, 120, 110, 150, 200),  # trigger < detected_start
        (100, 120, 160, 150, 200),  # trigger > detected_end
        (100, 120, 130, 210, 200),  # detected_end > event_end
        (100, 150, 130, 120, 200),  # detected_end < detected_start
    ],
)
def test_signal_event_invalid_index_ordering(
    sample_stream: StreamInfo,
    ev_start: int,
    det_start: int,
    trig: int,
    det_end: int,
    ev_end: int,
) -> None:
    """Test rejection of violated ordering invariants."""
    n_samples = max(1, ev_end - ev_start)
    iq_data = np.ones(n_samples, dtype=np.complex64)

    with pytest.raises(ContractValidationError):
        SignalEvent(
            event_id="ev-bad-order",
            stream=sample_stream,
            iq=iq_data,
            event_start_sample=ev_start,
            event_end_sample=ev_end,
            detected_start_sample=det_start,
            detected_end_sample=det_end,
            trigger_sample=trig,
        )


def test_signal_event_negative_start(sample_stream: StreamInfo) -> None:
    """Test rejection of negative event_start_sample."""
    with pytest.raises(ContractValidationError, match="event_start_sample"):
        SignalEvent(
            event_id="ev-neg",
            stream=sample_stream,
            iq=np.ones(10, dtype=np.complex64),
            event_start_sample=-5,
            event_end_sample=5,
            detected_start_sample=-2,
            detected_end_sample=2,
            trigger_sample=0,
        )


def test_signal_event_negative_dropped_samples(sample_stream: StreamInfo) -> None:
    """Test rejection of negative dropped_samples."""
    with pytest.raises(ContractValidationError, match="dropped_samples"):
        SignalEvent(
            event_id="ev-neg-dropped",
            stream=sample_stream,
            iq=np.ones(10, dtype=np.complex64),
            event_start_sample=0,
            event_end_sample=10,
            detected_start_sample=2,
            detected_end_sample=8,
            trigger_sample=5,
            dropped_samples=-1,
        )


def test_signal_event_blank_event_id(sample_stream: StreamInfo) -> None:
    """Test rejection of blank event_id."""
    with pytest.raises(ContractValidationError, match="event_id"):
        SignalEvent(
            event_id="   ",
            stream=sample_stream,
            iq=np.ones(10, dtype=np.complex64),
            event_start_sample=0,
            event_end_sample=10,
            detected_start_sample=2,
            detected_end_sample=8,
            trigger_sample=5,
        )


def test_signal_event_invalid_stream_type() -> None:
    """Test rejection when stream is not a StreamInfo instance."""
    with pytest.raises(ContractValidationError, match="StreamInfo"):
        SignalEvent(
            event_id="ev-stream-type",
            stream="not_a_stream",  # type: ignore[arg-type]
            iq=np.ones(10, dtype=np.complex64),
            event_start_sample=0,
            event_end_sample=10,
            detected_start_sample=2,
            detected_end_sample=8,
            trigger_sample=5,
        )


def test_signal_event_invalid_complete_and_timestamp(sample_stream: StreamInfo) -> None:
    """Test rejection when complete is non-bool or timestamp_ns is non-int."""
    with pytest.raises(ContractValidationError, match="complete"):
        SignalEvent(
            event_id="ev-complete",
            stream=sample_stream,
            iq=np.ones(10, dtype=np.complex64),
            event_start_sample=0,
            event_end_sample=10,
            detected_start_sample=2,
            detected_end_sample=8,
            trigger_sample=5,
            complete="yes",  # type: ignore[arg-type]
        )

    with pytest.raises(ContractValidationError, match="timestamp_ns"):
        SignalEvent(
            event_id="ev-ts",
            stream=sample_stream,
            iq=np.ones(10, dtype=np.complex64),
            event_start_sample=0,
            event_end_sample=10,
            detected_start_sample=2,
            detected_end_sample=8,
            trigger_sample=5,
            timestamp_ns=True,  # type: ignore[arg-type]
        )


def test_signal_event_non_integer_sample_index(sample_stream: StreamInfo) -> None:
    """Test rejection when sample index is a float or boolean."""
    with pytest.raises(ContractValidationError, match="must be an integer"):
        SignalEvent(
            event_id="ev-non-int",
            stream=sample_stream,
            iq=np.ones(10, dtype=np.complex64),
            event_start_sample=0.5,  # type: ignore[arg-type]
            event_end_sample=10,
            detected_start_sample=2,
            detected_end_sample=8,
            trigger_sample=5,
        )


def test_signal_event_completeness_invariant(sample_stream: StreamInfo) -> None:
    """Test completeness flag rules with dropped samples."""
    # 1. dropped=0, complete=True -> valid
    ev1 = SignalEvent(
        event_id="ev-drop-0-comp-true",
        stream=sample_stream,
        iq=np.ones(10, dtype=np.complex64),
        event_start_sample=0,
        event_end_sample=10,
        detected_start_sample=2,
        detected_end_sample=8,
        trigger_sample=5,
        dropped_samples=0,
        complete=True,
    )
    assert ev1.complete is True

    # 2. dropped=0, complete=False -> valid (e.g. truncated capture)
    ev2 = SignalEvent(
        event_id="ev-drop-0-comp-false",
        stream=sample_stream,
        iq=np.ones(10, dtype=np.complex64),
        event_start_sample=0,
        event_end_sample=10,
        detected_start_sample=2,
        detected_end_sample=8,
        trigger_sample=5,
        dropped_samples=0,
        complete=False,
    )
    assert ev2.complete is False

    # 3. dropped>0, complete=False -> valid
    ev3 = SignalEvent(
        event_id="ev-drop-pos-comp-false",
        stream=sample_stream,
        iq=np.ones(10, dtype=np.complex64),
        event_start_sample=0,
        event_end_sample=10,
        detected_start_sample=2,
        detected_end_sample=8,
        trigger_sample=5,
        dropped_samples=15,
        complete=False,
    )
    assert ev3.dropped_samples == 15
    assert ev3.complete is False

    # 4. dropped>0, complete=True -> reject!
    with pytest.raises(ContractValidationError, match="dropped_samples"):
        SignalEvent(
            event_id="ev-drop-pos-comp-true",
            stream=sample_stream,
            iq=np.ones(10, dtype=np.complex64),
            event_start_sample=0,
            event_end_sample=10,
            detected_start_sample=2,
            detected_end_sample=8,
            trigger_sample=5,
            dropped_samples=15,
            complete=True,
        )


def test_signal_event_zero_length_detected_signal_rejected(sample_stream: StreamInfo) -> None:
    """Test rejection of zero-length signal intervals: detected_start == detected_end."""
    with pytest.raises(ContractValidationError, match="Detected signal interval must be non-empty"):
        SignalEvent(
            event_id="ev-zero-sig",
            stream=sample_stream,
            iq=np.ones(10, dtype=np.complex64),
            event_start_sample=0,
            event_end_sample=10,
            detected_start_sample=5,
            detected_end_sample=5,  # zero length!
            trigger_sample=5,
        )


def test_signal_event_trigger_at_detected_end_rejected(sample_stream: StreamInfo) -> None:
    """Test rejection when trigger_sample == detected_end_sample (exclusive boundary)."""
    with pytest.raises(ContractValidationError):
        SignalEvent(
            event_id="ev-trig-at-end",
            stream=sample_stream,
            iq=np.ones(10, dtype=np.complex64),
            event_start_sample=0,
            event_end_sample=10,
            detected_start_sample=2,
            detected_end_sample=8,
            trigger_sample=8,  # trigger must be strictly < detected_end!
        )


def test_signal_event_pre_post_partitions(sample_stream: StreamInfo) -> None:
    """Test exact specification example for pre-signal/pre-trigger and post-signal/post-trigger."""
    total_samples = 900
    iq_data = np.zeros(total_samples, dtype=np.complex64)

    event = SignalEvent(
        event_id="ev-spec-example",
        stream=sample_stream,
        iq=iq_data,
        event_start_sample=1000,
        event_end_sample=1900,
        detected_start_sample=1200,
        detected_end_sample=1700,
        trigger_sample=1300,
    )

    # Verification of values from spec
    assert event.pre_signal_samples == 200
    assert event.pre_trigger_samples == 300
    assert event.signal_samples == 500
    assert event.post_signal_samples == 200
    assert event.post_trigger_samples == 600

    # Non-overlapping partition verification
    assert (
        event.pre_signal_samples + event.signal_samples + event.post_signal_samples
        == event.num_samples
    )

    # Durations
    assert event.pre_signal_duration_s == pytest.approx(200 / 1e6)
    assert event.pre_trigger_duration_s == pytest.approx(300 / 1e6)
    assert event.signal_duration_s == pytest.approx(500 / 1e6)
    assert event.post_signal_duration_s == pytest.approx(200 / 1e6)
    assert event.post_trigger_duration_s == pytest.approx(600 / 1e6)



