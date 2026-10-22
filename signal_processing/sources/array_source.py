"""In-memory array streaming IQ source."""

from __future__ import annotations

from typing import Any

import numpy as np

from signal_processing.core.contracts import IQBlock, StreamInfo, _validate_and_freeze_iq_array
from signal_processing.core.timebase import sample_offset_ns


class ArrayIQSource:
    """Streaming IQ source yielding blocks from an in-memory complex64 array."""

    def __init__(
        self,
        iq_data: Any,
        stream_info: StreamInfo,
        block_size: int = 1024,
        start_sample: int = 0,
        start_timestamp_ns: int | None = None,
    ) -> None:
        if not isinstance(stream_info, StreamInfo):
            raise TypeError(f"stream_info must be a StreamInfo instance, got {type(stream_info)}")
        if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
            raise ValueError(f"block_size must be an integer >= 1, got {block_size!r}")
        if isinstance(start_sample, bool) or not isinstance(start_sample, (int, np.integer)) or start_sample < 0:
            raise ValueError(f"start_sample must be an integer >= 0, got {start_sample!r}")
        if start_timestamp_ns is not None:
            if isinstance(start_timestamp_ns, bool) or not isinstance(start_timestamp_ns, (int, np.integer)) or start_timestamp_ns < 0:
                raise ValueError(f"start_timestamp_ns must be an integer >= 0, got {start_timestamp_ns!r}")

        self._stream_info = stream_info
        self._iq_data = _validate_and_freeze_iq_array(iq_data, "iq_data")
        self._block_size = int(block_size)
        self._start_sample = int(start_sample)
        self._start_timestamp_ns = int(start_timestamp_ns) if start_timestamp_ns is not None else None

        self._cursor: int = 0
        self._sequence_number: int = 0

    @property
    def stream_info(self) -> StreamInfo:
        """Stream metadata describing this source."""
        return self._stream_info

    @property
    def total_samples(self) -> int:
        """Total sample count of underlying IQ buffer."""
        return len(self._iq_data)

    def read_block(self) -> IQBlock | None:
        """Read and return the next sequential IQBlock, or None on EOF."""
        if self._cursor >= len(self._iq_data):
            return None

        end_idx = min(self._cursor + self._block_size, len(self._iq_data))
        block_iq = self._iq_data[self._cursor : end_idx]
        block_start_sample = self._start_sample + self._cursor

        block_ts_ns: int | None = None
        if self._start_timestamp_ns is not None:
            block_ts_ns = self._start_timestamp_ns + sample_offset_ns(
                self._cursor, self._stream_info.sample_rate_hz)

        block = IQBlock(
            stream=self._stream_info,
            sequence_number=self._sequence_number,
            start_sample=block_start_sample,
            timestamp_ns=block_ts_ns,
            iq=block_iq,
            dropped_samples_before=0,
            overflow=False,
        )

        self._cursor = end_idx
        self._sequence_number += 1
        return block

    def reset(self) -> None:
        """Reset stream position to the beginning."""
        self._cursor = 0
        self._sequence_number = 0
