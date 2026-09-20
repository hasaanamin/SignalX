"""Unit tests for StreamInfo contract."""

from dataclasses import FrozenInstanceError

import pytest

from signal_processing.core import ContractValidationError, StreamInfo


def test_stream_info_valid() -> None:
    """Test valid construction with all default and non-default parameters."""
    info = StreamInfo(
        stream_id="stream-01",
        source_id="synthetic-sine",
        sample_rate_hz=1e6,
        center_frequency_hz=915e6,
        channel_id="ch0",
        dtype="complex64",
    )
    assert info.stream_id == "stream-01"
    assert info.source_id == "synthetic-sine"
    assert info.sample_rate_hz == 1e6
    assert info.center_frequency_hz == 915e6
    assert info.channel_id == "ch0"
    assert info.dtype == "complex64"


def test_stream_info_defaults() -> None:
    """Test valid construction with minimal parameters."""
    info = StreamInfo(
        stream_id="stream-02",
        source_id="file-source",
        sample_rate_hz=2.4e6,
    )
    assert info.center_frequency_hz is None
    assert info.channel_id is None
    assert info.dtype == "complex64"


def test_stream_info_immutability() -> None:
    """Test that StreamInfo is strictly immutable."""
    info = StreamInfo(
        stream_id="stream-01",
        source_id="source-01",
        sample_rate_hz=1e6,
    )
    with pytest.raises(FrozenInstanceError):
        info.sample_rate_hz = 2e6  # type: ignore[misc]


@pytest.mark.parametrize("blank_id", ["", "   ", "\t\n"])
def test_stream_info_blank_stream_id(blank_id: str) -> None:
    """Test that blank stream_id raises ContractValidationError."""
    with pytest.raises(ContractValidationError, match="stream_id"):
        StreamInfo(
            stream_id=blank_id,
            source_id="source-01",
            sample_rate_hz=1e6,
        )


@pytest.mark.parametrize("blank_id", ["", "   ", "\t\n"])
def test_stream_info_blank_source_id(blank_id: str) -> None:
    """Test that blank source_id raises ContractValidationError."""
    with pytest.raises(ContractValidationError, match="source_id"):
        StreamInfo(
            stream_id="stream-01",
            source_id=blank_id,
            sample_rate_hz=1e6,
        )


@pytest.mark.parametrize("invalid_rate", [0.0, -1.0, -1e6])
def test_stream_info_zero_or_negative_sample_rate(invalid_rate: float) -> None:
    """Test that zero or negative sample rate raises ContractValidationError."""
    with pytest.raises(ContractValidationError, match="sample_rate_hz"):
        StreamInfo(
            stream_id="stream-01",
            source_id="source-01",
            sample_rate_hz=invalid_rate,
        )


@pytest.mark.parametrize("invalid_rate", [float("nan"), float("inf"), float("-inf"), True, "1000"])
def test_stream_info_infinite_nan_or_non_numeric_sample_rate(invalid_rate: object) -> None:
    """Test that non-finite or non-numeric sample rate raises ContractValidationError."""
    with pytest.raises(ContractValidationError, match="sample_rate_hz"):
        StreamInfo(
            stream_id="stream-01",
            source_id="source-01",
            sample_rate_hz=invalid_rate,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_cf", [float("nan"), float("inf"), float("-inf"), True, "915M"])
def test_stream_info_invalid_center_frequency(invalid_cf: object) -> None:
    """Test that non-finite or non-numeric center_frequency_hz raises ContractValidationError."""
    with pytest.raises(ContractValidationError, match="center_frequency_hz"):
        StreamInfo(
            stream_id="stream-01",
            source_id="source-01",
            sample_rate_hz=1e6,
            center_frequency_hz=invalid_cf,  # type: ignore[arg-type]
        )


def test_stream_info_blank_channel_id() -> None:
    """Test that blank channel_id raises ContractValidationError if provided."""
    with pytest.raises(ContractValidationError, match="channel_id"):
        StreamInfo(
            stream_id="stream-01",
            source_id="source-01",
            sample_rate_hz=1e6,
            channel_id="   ",
        )


def test_stream_info_unsupported_dtype() -> None:
    """Test that dtype other than complex64 raises ContractValidationError."""
    with pytest.raises(ContractValidationError, match="Unsupported dtype"):
        StreamInfo(
            stream_id="stream-01",
            source_id="source-01",
            sample_rate_hz=1e6,
            dtype="int16",
        )
