"""Unit tests for stream consistency validation helpers."""

import numpy as np
import pytest

from signal_processing.core import (
    IQBlock,
    StreamConsistencyError,
    StreamInfo,
    validate_consecutive_blocks,
)


@pytest.fixture
def stream_a() -> StreamInfo:
    return StreamInfo(
        stream_id="stream-alpha",
        source_id="synthetic-source",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )


def test_validate_consecutive_blocks_perfect_continuity(stream_a: StreamInfo) -> None:
    """Test validation passes with exact consecutive samples and sequential blocks."""
    block1 = IQBlock(
        stream=stream_a,
        sequence_number=0,
        start_sample=0,
        timestamp_ns=0,
        iq=np.ones(1000, dtype=np.complex64),
    )
    block2 = IQBlock(
        stream=stream_a,
        sequence_number=1,
        start_sample=1000,
        timestamp_ns=1_000_000,
        iq=np.ones(1000, dtype=np.complex64),
        dropped_samples_before=0,
    )

    # Should not raise any exception
    validate_consecutive_blocks(block1, block2)


def test_validate_consecutive_blocks_legal_declared_dropped_samples(
    stream_a: StreamInfo,
) -> None:
    """Test validation passes when sample gap matches declared dropped_samples_before."""
    block1 = IQBlock(
        stream=stream_a,
        sequence_number=0,
        start_sample=0,
        timestamp_ns=0,
        iq=np.ones(1000, dtype=np.complex64),  # ends at 1000
    )
    block2 = IQBlock(
        stream=stream_a,
        sequence_number=1,
        start_sample=1500,  # gap of 500
        timestamp_ns=1_500_000,
        iq=np.ones(1000, dtype=np.complex64),
        dropped_samples_before=500,  # declares 500 dropped
    )

    # Should pass because 1000 + 500 == 1500
    validate_consecutive_blocks(block1, block2)


def test_validate_consecutive_blocks_unexplained_sample_gap(stream_a: StreamInfo) -> None:
    """Test rejection when sample index jumps without declaring dropped samples."""
    block1 = IQBlock(
        stream=stream_a,
        sequence_number=0,
        start_sample=0,
        timestamp_ns=0,
        iq=np.ones(1000, dtype=np.complex64),  # ends at 1000
    )
    block2 = IQBlock(
        stream=stream_a,
        sequence_number=1,
        start_sample=1500,  # gap of 500
        timestamp_ns=1_500_000,
        iq=np.ones(1000, dtype=np.complex64),
        dropped_samples_before=0,  # undeclared gap!
    )

    with pytest.raises(StreamConsistencyError, match="Unexplained sample discontinuity"):
        validate_consecutive_blocks(block1, block2)


def test_validate_consecutive_blocks_sample_overlap(stream_a: StreamInfo) -> None:
    """Test rejection when current block start overlaps with previous block range."""
    block1 = IQBlock(
        stream=stream_a,
        sequence_number=0,
        start_sample=0,
        timestamp_ns=0,
        iq=np.ones(1000, dtype=np.complex64),  # ends at 1000
    )
    block2 = IQBlock(
        stream=stream_a,
        sequence_number=1,
        start_sample=800,  # overlaps 200 samples
        timestamp_ns=800_000,
        iq=np.ones(1000, dtype=np.complex64),
    )

    with pytest.raises(StreamConsistencyError, match="overlap"):
        validate_consecutive_blocks(block1, block2)


def test_validate_consecutive_blocks_duplicate_sequence(stream_a: StreamInfo) -> None:
    """Test rejection of duplicate block sequence numbers."""
    block1 = IQBlock(
        stream=stream_a,
        sequence_number=5,
        start_sample=5000,
        timestamp_ns=None,
        iq=np.ones(1000, dtype=np.complex64),
    )
    block2 = IQBlock(
        stream=stream_a,
        sequence_number=5,  # duplicate!
        start_sample=6000,
        timestamp_ns=None,
        iq=np.ones(1000, dtype=np.complex64),
    )

    with pytest.raises(StreamConsistencyError, match="Duplicate or out-of-order"):
        validate_consecutive_blocks(block1, block2)


def test_validate_consecutive_blocks_out_of_order_sequence(stream_a: StreamInfo) -> None:
    """Test rejection of out-of-order sequence numbers."""
    block1 = IQBlock(
        stream=stream_a,
        sequence_number=5,
        start_sample=5000,
        timestamp_ns=None,
        iq=np.ones(1000, dtype=np.complex64),
    )
    block2 = IQBlock(
        stream=stream_a,
        sequence_number=4,  # backwards!
        start_sample=6000,
        timestamp_ns=None,
        iq=np.ones(1000, dtype=np.complex64),
    )

    with pytest.raises(StreamConsistencyError, match="Duplicate or out-of-order"):
        validate_consecutive_blocks(block1, block2)


def test_validate_consecutive_blocks_sequence_gap(stream_a: StreamInfo) -> None:
    """Test rejection of sequence number jump (e.g. 5 -> 7)."""
    block1 = IQBlock(
        stream=stream_a,
        sequence_number=5,
        start_sample=5000,
        timestamp_ns=None,
        iq=np.ones(1000, dtype=np.complex64),
    )
    block2 = IQBlock(
        stream=stream_a,
        sequence_number=7,  # gap! expected 6
        start_sample=6000,
        timestamp_ns=None,
        iq=np.ones(1000, dtype=np.complex64),
    )

    with pytest.raises(StreamConsistencyError, match="Block sequence number gap"):
        validate_consecutive_blocks(block1, block2)


def test_validate_consecutive_blocks_stream_mismatch(stream_a: StreamInfo) -> None:
    """Test rejection when blocks originate from different stream IDs."""
    stream_b = StreamInfo(
        stream_id="stream-beta",
        source_id="synthetic-source",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    block1 = IQBlock(
        stream=stream_a,
        sequence_number=0,
        start_sample=0,
        timestamp_ns=None,
        iq=np.ones(100, dtype=np.complex64),
    )
    block2 = IQBlock(
        stream=stream_b,
        sequence_number=1,
        start_sample=100,
        timestamp_ns=None,
        iq=np.ones(100, dtype=np.complex64),
    )

    with pytest.raises(StreamConsistencyError, match="Stream ID mismatch"):
        validate_consecutive_blocks(block1, block2)


def test_validate_consecutive_blocks_sample_rate_mismatch(stream_a: StreamInfo) -> None:
    """Test rejection when sample rates differ."""
    stream_diff_sr = StreamInfo(
        stream_id="stream-alpha",
        source_id="synthetic-source",
        sample_rate_hz=2_000_000.0,  # 2 MHz vs 1 MHz
        center_frequency_hz=915_000_000.0,
    )
    block1 = IQBlock(
        stream=stream_a,
        sequence_number=0,
        start_sample=0,
        timestamp_ns=None,
        iq=np.ones(100, dtype=np.complex64),
    )
    block2 = IQBlock(
        stream=stream_diff_sr,
        sequence_number=1,
        start_sample=100,
        timestamp_ns=None,
        iq=np.ones(100, dtype=np.complex64),
    )

    with pytest.raises(StreamConsistencyError, match="Sample rate mismatch"):
        validate_consecutive_blocks(block1, block2)


def test_validate_consecutive_blocks_center_frequency_mismatch(stream_a: StreamInfo) -> None:
    """Test rejection when center frequencies are incompatible."""
    stream_diff_cf = StreamInfo(
        stream_id="stream-alpha",
        source_id="synthetic-source",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=916_000_000.0,  # 916 MHz vs 915 MHz
    )
    block1 = IQBlock(
        stream=stream_a,
        sequence_number=0,
        start_sample=0,
        timestamp_ns=None,
        iq=np.ones(100, dtype=np.complex64),
    )
    block2 = IQBlock(
        stream=stream_diff_cf,
        sequence_number=1,
        start_sample=100,
        timestamp_ns=None,
        iq=np.ones(100, dtype=np.complex64),
    )

    with pytest.raises(StreamConsistencyError, match="Center frequency mismatch"):
        validate_consecutive_blocks(block1, block2)


def test_validate_consecutive_blocks_non_iqblock_types(stream_a: StreamInfo) -> None:
    """Test rejection when arguments are not IQBlock instances."""
    block1 = IQBlock(
        stream=stream_a,
        sequence_number=0,
        start_sample=0,
        timestamp_ns=None,
        iq=np.ones(100, dtype=np.complex64),
    )
    with pytest.raises(StreamConsistencyError, match="IQBlock instances"):
        validate_consecutive_blocks(block1, "not_a_block")  # type: ignore[arg-type]

    with pytest.raises(StreamConsistencyError, match="IQBlock instances"):
        validate_consecutive_blocks("not_a_block", block1)  # type: ignore[arg-type]


def test_validate_consecutive_blocks_center_frequency_presence_mismatch(
    stream_a: StreamInfo,
) -> None:
    """Test rejection when one block specifies center frequency and the other does not."""
    stream_no_cf = StreamInfo(
        stream_id="stream-alpha",
        source_id="synthetic-source",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=None,
    )
    block_with_cf = IQBlock(
        stream=stream_a,
        sequence_number=0,
        start_sample=0,
        timestamp_ns=None,
        iq=np.ones(100, dtype=np.complex64),
    )
    block_no_cf = IQBlock(
        stream=stream_no_cf,
        sequence_number=1,
        start_sample=100,
        timestamp_ns=None,
        iq=np.ones(100, dtype=np.complex64),
    )

    with pytest.raises(StreamConsistencyError, match="Incompatible center frequency configuration"):
        validate_consecutive_blocks(block_with_cf, block_no_cf)


def test_validate_consecutive_blocks_source_id_mismatch() -> None:
    """Test rejection when blocks have same stream_id but differing source_id."""
    stream1 = StreamInfo("stream-01", "source-A", 1e6)
    stream2 = StreamInfo("stream-01", "source-B", 1e6)
    b1 = IQBlock(stream=stream1, sequence_number=0, start_sample=0, timestamp_ns=None, iq=np.ones(10, dtype=np.complex64))
    b2 = IQBlock(stream=stream2, sequence_number=1, start_sample=10, timestamp_ns=None, iq=np.ones(10, dtype=np.complex64))

    with pytest.raises(StreamConsistencyError, match="Source ID mismatch"):
        validate_consecutive_blocks(b1, b2)


def test_validate_consecutive_blocks_channel_id_mismatch() -> None:
    """Test rejection when blocks have differing channel_id."""
    stream1 = StreamInfo("stream-01", "source-A", 1e6, channel_id="ch0")
    stream2 = StreamInfo("stream-01", "source-A", 1e6, channel_id="ch1")
    b1 = IQBlock(stream=stream1, sequence_number=0, start_sample=0, timestamp_ns=None, iq=np.ones(10, dtype=np.complex64))
    b2 = IQBlock(stream=stream2, sequence_number=1, start_sample=10, timestamp_ns=None, iq=np.ones(10, dtype=np.complex64))

    with pytest.raises(StreamConsistencyError, match="Channel ID mismatch"):
        validate_consecutive_blocks(b1, b2)


def test_validate_consecutive_blocks_channel_id_none_presence_mismatch() -> None:
    """Test rejection when one block specifies channel_id and the other has channel_id=None."""
    stream1 = StreamInfo("stream-01", "source-A", 1e6, channel_id="ch0")
    stream2 = StreamInfo("stream-01", "source-A", 1e6, channel_id=None)
    b1 = IQBlock(stream=stream1, sequence_number=0, start_sample=0, timestamp_ns=None, iq=np.ones(10, dtype=np.complex64))
    b2 = IQBlock(stream=stream2, sequence_number=1, start_sample=10, timestamp_ns=None, iq=np.ones(10, dtype=np.complex64))

    with pytest.raises(StreamConsistencyError, match="Channel ID mismatch"):
        validate_consecutive_blocks(b1, b2)


