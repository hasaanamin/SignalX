"""Behavioral tests for CircularIQBuffer."""

import numpy as np
import pytest

from signal_processing.buffering.exceptions import (
    BufferContinuityError,
    BufferOverwrittenError,
    BufferRangeError,
    BufferUnavailableError,
)
from signal_processing.buffering.ring_buffer import CircularIQBuffer


class TestCircularIQBuffer:
    """Rigorous tests for circular buffer oracle, wrapping, and immutability."""

    def test_randomized_oracle_comparison(self) -> None:
        """Compare CircularIQBuffer operations against an independent retained-history oracle."""
        rng = np.random.default_rng(12345)
        capacity = 500  # Non-power-of-two capacity
        buf = CircularIQBuffer(capacity=capacity)

        # Independent oracle: dictionary mapping absolute sample index -> sample value
        oracle: dict[int, complex] = {}
        curr_sample = 0

        # Perform 50 randomized appends and reads
        for _ in range(50):
            chunk_len = int(rng.integers(10, 300))
            chunk = (rng.normal(size=chunk_len) + 1j * rng.normal(size=chunk_len)).astype(np.complex64)

            buf.write(curr_sample, chunk)
            for idx, val in enumerate(chunk):
                oracle[curr_sample + idx] = complex(val)

            curr_sample += chunk_len
            min_retained = max(0, curr_sample - capacity)

            # Query available range
            oldest, next_write = buf.available_range()
            assert oldest == min_retained
            assert next_write == curr_sample

            # Perform valid random range reads
            if next_write - oldest > 10:
                r_start = int(rng.integers(oldest, next_write - 5))
                r_end = int(rng.integers(r_start + 1, next_write + 1))
                read_data = buf.read_range(r_start, r_end)

                expected = np.array([oracle[s] for s in range(r_start, r_end)], dtype=np.complex64)
                np.testing.assert_array_equal(read_data, expected)

    def test_writes_larger_than_capacity(self) -> None:
        """Writing a chunk larger than capacity retains the final capacity samples."""
        capacity = 128
        buf = CircularIQBuffer(capacity=capacity)

        large_chunk = np.arange(500, dtype=np.float32).astype(np.complex64)
        buf.write(0, large_chunk)

        assert buf.oldest_sample == 500 - 128
        assert buf.next_write_sample == 500
        assert buf.occupancy == 128

        read_data = buf.read_range(500 - 128, 500)
        expected = large_chunk[-128:]
        np.testing.assert_array_equal(read_data, expected)

    def test_overwritten_range_error(self) -> None:
        """Attempting to read already overwritten samples raises BufferOverwrittenError."""
        buf = CircularIQBuffer(capacity=100)
        buf.write(0, np.ones(250, dtype=np.complex64))

        # Oldest is 150
        assert buf.oldest_sample == 150
        with pytest.raises(BufferOverwrittenError, match="overwritten"):
            buf.read_range(100, 160)

    def test_future_range_unavailable_error(self) -> None:
        """Attempting to read future unwritten samples raises BufferUnavailableError."""
        buf = CircularIQBuffer(capacity=100)
        buf.write(0, np.ones(50, dtype=np.complex64))

        with pytest.raises(BufferUnavailableError, match="not yet arrived"):
            buf.read_range(10, 60)

    def test_invalid_range_order(self) -> None:
        """start_sample > end_sample raises BufferRangeError."""
        buf = CircularIQBuffer(capacity=100)
        buf.write(0, np.ones(50, dtype=np.complex64))

        with pytest.raises(BufferRangeError, match="start_sample.*>.*end_sample"):
            buf.read_range(30, 20)

    def test_continuity_error_on_gap_without_reset(self) -> None:
        """Writing with a sample coordinate gap without reset raises BufferContinuityError."""
        buf = CircularIQBuffer(capacity=100)
        buf.write(0, np.ones(50, dtype=np.complex64))

        with pytest.raises(BufferContinuityError, match="Write continuity violated"):
            buf.write(60, np.ones(10, dtype=np.complex64))

    def test_snapshot_independence_after_many_wraps(self) -> None:
        """Read snapshots must remain completely unchanged when the ring wraps arbitrarily many times."""
        capacity = 64
        buf = CircularIQBuffer(capacity=capacity)

        # Write initial block and take snapshot
        initial = (np.arange(32) + 10j).astype(np.complex64)
        buf.write(0, initial)
        snapshot = buf.read_range(0, 32)
        expected_copy = snapshot.copy()

        # Wrap buffer 20 times with completely new data
        for k in range(1, 21):
            buf.write(32 * k, (np.full(32, -999.0 - 999.0j)).astype(np.complex64))

        # Assert snapshot taken at generation start was not mutated
        np.testing.assert_array_equal(snapshot, expected_copy)
        # Verify snapshot cannot be made writeable
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
            snapshot.setflags(write=True)
