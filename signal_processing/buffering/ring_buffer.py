"""Preallocated circular IQ buffer with absolute stream coordinate indexing."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from signal_processing.buffering.exceptions import (
    BufferContinuityError,
    BufferOverwrittenError,
    BufferRangeError,
    BufferUnavailableError,
)
from signal_processing.core.contracts import _validate_and_freeze_iq_array


class CircularIQBuffer:
    """Fixed-capacity preallocated circular complex64 array.

    Maintains absolute chronological stream coordinates across continuous wraparounds.
    Provides slice-based writing without intermediate Python lists or full-history allocations.
    Reads return independent frozen/detached copies to ensure snapshots cannot be mutated
    by future buffer wraparounds or writes.
    """

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)) or capacity < 1:
            raise ValueError(f"Circular buffer capacity must be an integer >= 1, got {capacity!r}")
        self._capacity = int(capacity)
        self._buffer = np.zeros(self._capacity, dtype=np.complex64)
        self._oldest_sample: int = 0
        self._next_write_sample: int = 0
        self._stream_generation: int = 0
        self._initialized: bool = False
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        """Physical buffer capacity in samples."""
        return self._capacity

    @property
    def occupancy(self) -> int:
        """Current number of active samples retained in the buffer."""
        with self._lock:
            if not self._initialized:
                return 0
            return min(self._next_write_sample - self._oldest_sample, self._capacity)

    @property
    def oldest_sample(self) -> int:
        """Absolute stream index of the oldest sample currently retained."""
        with self._lock:
            return self._oldest_sample

    @property
    def next_write_sample(self) -> int:
        """Absolute stream index of the next sample to be written."""
        with self._lock:
            return self._next_write_sample

    @property
    def total_samples_written(self) -> int:
        """Total samples written to ring buffer."""
        with self._lock:
            return self._next_write_sample

    @property
    def stream_generation(self) -> int:
        """Current stream continuity epoch / generation counter."""
        with self._lock:
            return self._stream_generation

    def available_range(self) -> tuple[int, int]:
        """Return half-open interval [oldest_sample, next_write_sample) of currently held data."""
        with self._lock:
            return (self._oldest_sample, self._next_write_sample)

    def write(self, start_sample: int, iq: Any) -> None:
        """Append a contiguous chunk of IQ samples starting at start_sample.

        Args:
            start_sample: Absolute stream index of the first sample in iq.
            iq: 1D complex64 array of samples.

        Raises:
            BufferContinuityError: If start_sample does not match next_write_sample
                in an ongoing continuous generation.
            ValueError: If iq is malformed or not complex64.
        """
        if isinstance(start_sample, bool) or not isinstance(start_sample, (int, np.integer)) or int(start_sample) < 0:
            raise ValueError(f"start_sample must be an integer >= 0, got {start_sample!r}")
        arr = np.asarray(iq)
        if arr.dtype != np.complex64 or arr.ndim != 1:
            raise ValueError(
                f"IQ data must be a 1D np.complex64 array, got shape {arr.shape} dtype {arr.dtype}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("IQ data must contain only finite values (no NaN or Inf)")
        n = len(arr)
        if n == 0:
            return

        with self._lock:
            if not self._initialized:
                self._oldest_sample = int(start_sample)
                self._next_write_sample = int(start_sample)
                self._initialized = True
            elif start_sample != self._next_write_sample:
                raise BufferContinuityError(
                    f"Write continuity violated: expected start_sample {self._next_write_sample}, "
                    f"got {start_sample} (generation {self._stream_generation})"
                )

            if n >= self._capacity:
                # Chunk is at least as large as capacity: retain the final capacity samples
                keep = arr[n - self._capacity :]
                offset = start_sample + n - self._capacity
                phys_start = offset % self._capacity
                first = self._capacity - phys_start
                if first == self._capacity:
                    self._buffer[:] = keep
                else:
                    self._buffer[phys_start:] = keep[:first]
                    self._buffer[:phys_start] = keep[first:]
                self._next_write_sample = int(start_sample + n)
                self._oldest_sample = self._next_write_sample - self._capacity
            else:
                # Ordinary chunk smaller than capacity: 1 or 2 slice assignments
                phys_start = start_sample % self._capacity
                if phys_start + n <= self._capacity:
                    self._buffer[phys_start : phys_start + n] = arr
                else:
                    first = self._capacity - phys_start
                    self._buffer[phys_start:] = arr[:first]
                    self._buffer[: n - first] = arr[first:]
                self._next_write_sample = int(start_sample + n)
                self._oldest_sample = max(self._oldest_sample, self._next_write_sample - self._capacity)

    def read_range(self, start_sample: int, end_sample: int) -> np.ndarray:
        """Extract a chronological span [start_sample, end_sample) as an independent copy.

        Args:
            start_sample: Absolute stream index of the first requested sample (inclusive).
            end_sample: Absolute stream index of the final requested sample (exclusive).

        Returns:
            1D np.complex64 array of length (end_sample - start_sample).

        Raises:
            BufferRangeError: If start_sample > end_sample.
            BufferOverwrittenError: If start_sample < oldest_sample.
            BufferUnavailableError: If end_sample > next_write_sample.
        """
        start = int(start_sample)
        end = int(end_sample)
        if start > end:
            raise BufferRangeError(f"Invalid range: start_sample ({start}) > end_sample ({end})")
        if start == end:
            return np.empty(0, dtype=np.complex64)

        with self._lock:
            if not self._initialized:
                raise BufferUnavailableError("Buffer is empty; no data has been written")
            if start < self._oldest_sample:
                raise BufferOverwrittenError(
                    f"Requested start_sample {start} has been overwritten. "
                    f"Oldest available sample is {self._oldest_sample}"
                )
            if end > self._next_write_sample:
                raise BufferUnavailableError(
                    f"Requested end_sample {end} has not yet arrived. "
                    f"Next write sample is {self._next_write_sample}"
                )

            n = end - start
            phys_start = start % self._capacity
            out = np.empty(n, dtype=np.complex64)
            if phys_start + n <= self._capacity:
                out[:] = self._buffer[phys_start : phys_start + n]
            else:
                first = self._capacity - phys_start
                out[:first] = self._buffer[phys_start:]
                out[first:] = self._buffer[: n - first]

            return _validate_and_freeze_iq_array(out)

    def reset(self, start_sample: int = 0) -> None:
        """Reset buffer state and advance to a new stream generation epoch.

        Args:
            start_sample: Initial sample coordinate for the new generation.
        """
        with self._lock:
            self._stream_generation += 1
            self._oldest_sample = int(start_sample)
            self._next_write_sample = int(start_sample)
            self._initialized = False
