"""Unit tests for IQBlock contract."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from signal_processing.core import ContractValidationError, IQBlock, StreamInfo


@pytest.fixture
def sample_stream() -> StreamInfo:
    return StreamInfo(
        stream_id="stream-01",
        source_id="synthetic-source",
        sample_rate_hz=1_000_000.0,  # 1 MHz -> 1 us per sample
        center_frequency_hz=915_000_000.0,
    )


def test_iq_block_valid(sample_stream: StreamInfo) -> None:
    """Test valid IQBlock construction and derived properties."""
    iq_data = np.array([1.0 + 2.0j, 3.0 - 4.0j, -0.5 + 0.5j], dtype=np.complex64)
    block = IQBlock(
        stream=sample_stream,
        sequence_number=0,
        start_sample=1000,
        timestamp_ns=1_000_000_000,
        iq=iq_data,
        dropped_samples_before=0,
        overflow=False,
    )

    assert block.stream == sample_stream
    assert block.sequence_number == 0
    assert block.start_sample == 1000
    assert block.timestamp_ns == 1_000_000_000
    assert block.num_samples == 3
    assert block.end_sample == 1003
    assert block.duration_s == pytest.approx(3 / 1e6)
    assert block.dropped_samples_before == 0
    assert not block.overflow


def test_iq_block_sample_offset_to_time_s(sample_stream: StreamInfo) -> None:
    """Test sample_offset_to_time_s helper calculation."""
    iq_data = np.ones(100, dtype=np.complex64)
    block = IQBlock(
        stream=sample_stream,
        sequence_number=0,
        start_sample=0,
        timestamp_ns=None,
        iq=iq_data,
    )

    assert block.sample_offset_to_time_s(0) == 0.0
    assert block.sample_offset_to_time_s(500) == pytest.approx(500 / 1e6)
    assert block.sample_offset_to_time_s(1e6) == pytest.approx(1.0)

    # Invalid offsets
    with pytest.raises(ContractValidationError):
        block.sample_offset_to_time_s(float("nan"))
    with pytest.raises(ContractValidationError):
        block.sample_offset_to_time_s(True)


def test_iq_block_immutability(sample_stream: StreamInfo) -> None:
    """Test that IQBlock fields cannot be reassigned."""
    block = IQBlock(
        stream=sample_stream,
        sequence_number=0,
        start_sample=0,
        timestamp_ns=None,
        iq=np.ones(10, dtype=np.complex64),
    )
    with pytest.raises(FrozenInstanceError):
        block.sequence_number = 1  # type: ignore[misc]


def test_iq_block_read_only_storage(sample_stream: StreamInfo) -> None:
    """Test that IQ array is defensive against caller mutation and stored read-only."""
    caller_array = np.array([1.0 + 1.0j, 2.0 + 2.0j], dtype=np.complex64)
    block = IQBlock(
        stream=sample_stream,
        sequence_number=0,
        start_sample=0,
        timestamp_ns=None,
        iq=caller_array,
    )

    # 1. Stored array is marked non-writeable
    assert not block.iq.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        block.iq[0] = 99.0 + 99.0j

    # 2. Mutating caller's original array does not mutate block's data
    caller_array[0] = -50.0 - 50.0j
    assert block.iq[0] == 1.0 + 1.0j


def test_iq_block_negative_start_sample(sample_stream: StreamInfo) -> None:
    """Test rejection of negative start_sample."""
    with pytest.raises(ContractValidationError, match="start_sample"):
        IQBlock(
            stream=sample_stream,
            sequence_number=0,
            start_sample=-1,
            timestamp_ns=None,
            iq=np.ones(10, dtype=np.complex64),
        )


def test_iq_block_negative_sequence_number(sample_stream: StreamInfo) -> None:
    """Test rejection of negative sequence_number."""
    with pytest.raises(ContractValidationError, match="sequence_number"):
        IQBlock(
            stream=sample_stream,
            sequence_number=-1,
            start_sample=0,
            timestamp_ns=None,
            iq=np.ones(10, dtype=np.complex64),
        )


def test_iq_block_negative_dropped_samples_before(sample_stream: StreamInfo) -> None:
    """Test rejection of negative dropped_samples_before."""
    with pytest.raises(ContractValidationError, match="dropped_samples_before"):
        IQBlock(
            stream=sample_stream,
            sequence_number=0,
            start_sample=0,
            timestamp_ns=None,
            iq=np.ones(10, dtype=np.complex64),
            dropped_samples_before=-5,
        )


@pytest.mark.parametrize(
    "bad_iq",
    [
        np.ones((10, 2), dtype=np.complex64),       # 2D
        np.ones((5, 2, 2), dtype=np.complex64),    # 3D
        np.array(1.0 + 1.0j, dtype=np.complex64),  # 0D scalar
    ],
)
def test_iq_block_invalid_dimensions(sample_stream: StreamInfo, bad_iq: np.ndarray) -> None:
    """Test rejection of non-1D IQ arrays."""
    with pytest.raises(ContractValidationError, match="1-dimensional"):
        IQBlock(
            stream=sample_stream,
            sequence_number=0,
            start_sample=0,
            timestamp_ns=None,
            iq=bad_iq,
        )


def test_iq_block_empty_array(sample_stream: StreamInfo) -> None:
    """Test rejection of empty IQ array."""
    with pytest.raises(ContractValidationError, match="empty"):
        IQBlock(
            stream=sample_stream,
            sequence_number=0,
            start_sample=0,
            timestamp_ns=None,
            iq=np.array([], dtype=np.complex64),
        )


@pytest.mark.parametrize(
    "wrong_dtype_array",
    [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([1 + 2j, 3 + 4j], dtype=np.complex128),
        np.array([1, 2], dtype=np.int32),
    ],
)
def test_iq_block_wrong_dtype(sample_stream: StreamInfo, wrong_dtype_array: np.ndarray) -> None:
    """Test rejection of non-complex64 arrays."""
    with pytest.raises(ContractValidationError, match="np.complex64"):
        IQBlock(
            stream=sample_stream,
            sequence_number=0,
            start_sample=0,
            timestamp_ns=None,
            iq=wrong_dtype_array,
        )


@pytest.mark.parametrize(
    "nan_iq",
    [
        np.array([np.nan + 1.0j], dtype=np.complex64),
        np.array([1.0 + np.nan * 1.0j], dtype=np.complex64),
        np.array([np.nan + np.nan * 1.0j], dtype=np.complex64),
    ],
)
def test_iq_block_nan_iq(sample_stream: StreamInfo, nan_iq: np.ndarray) -> None:
    """Test rejection of NaN in IQ array."""
    with pytest.raises(ContractValidationError, match="non-finite"):
        IQBlock(
            stream=sample_stream,
            sequence_number=0,
            start_sample=0,
            timestamp_ns=None,
            iq=nan_iq,
        )


@pytest.mark.parametrize(
    "inf_iq",
    [
        np.array([np.inf + 0.0j], dtype=np.complex64),
        np.array([0.0 - np.inf * 1.0j], dtype=np.complex64),
    ],
)
def test_iq_block_inf_iq(sample_stream: StreamInfo, inf_iq: np.ndarray) -> None:
    """Test rejection of Inf in IQ array."""
    with pytest.raises(ContractValidationError, match="non-finite"):
        IQBlock(
            stream=sample_stream,
            sequence_number=0,
            start_sample=0,
            timestamp_ns=None,
            iq=inf_iq,
        )


@pytest.mark.parametrize("invalid_ts", [True, False, 1.5, "12345"])
def test_iq_block_invalid_timestamp_ns(sample_stream: StreamInfo, invalid_ts: object) -> None:
    """Test rejection of invalid timestamp_ns types."""
    with pytest.raises(ContractValidationError, match="timestamp_ns"):
        IQBlock(
            stream=sample_stream,
            sequence_number=0,
            start_sample=0,
            timestamp_ns=invalid_ts,  # type: ignore[arg-type]
            iq=np.ones(10, dtype=np.complex64),
        )


def test_iq_block_invalid_stream_type() -> None:
    """Test rejection when stream is not a StreamInfo instance."""
    with pytest.raises(ContractValidationError, match="StreamInfo"):
        IQBlock(
            stream="not_a_stream",  # type: ignore[arg-type]
            sequence_number=0,
            start_sample=0,
            timestamp_ns=None,
            iq=np.ones(10, dtype=np.complex64),
        )


def test_iq_block_invalid_overflow_type(sample_stream: StreamInfo) -> None:
    """Test rejection when overflow is not boolean."""
    with pytest.raises(ContractValidationError, match="overflow"):
        IQBlock(
            stream=sample_stream,
            sequence_number=0,
            start_sample=0,
            timestamp_ns=None,
            iq=np.ones(10, dtype=np.complex64),
            overflow="yes",  # type: ignore[arg-type]
        )


def test_iq_block_from_convertible_sequence(sample_stream: StreamInfo) -> None:
    """Test valid construction from a sequence convertible to complex64 array."""
    list_iq = [np.complex64(1.0 + 2.0j), np.complex64(3.0 + 4.0j)]
    block = IQBlock(
        stream=sample_stream,
        sequence_number=0,
        start_sample=0,
        timestamp_ns=None,
        iq=list_iq,  # type: ignore[arg-type]
    )
    assert block.num_samples == 2
    assert block.iq.dtype == np.complex64


def test_iq_block_from_unconvertible_iq(sample_stream: StreamInfo) -> None:
    """Test rejection when iq cannot be converted to complex64 numpy array."""
    with pytest.raises(ContractValidationError):
        IQBlock(
            stream=sample_stream,
            sequence_number=0,
            start_sample=0,
            timestamp_ns=None,
            iq=[object()],  # type: ignore[arg-type]
        )


