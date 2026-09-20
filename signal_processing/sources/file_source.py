"""Streaming complex64 file IQ source using memory mapping."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np

from signal_processing.core.contracts import IQBlock, StreamInfo


class RecordedFileIQSource:
    """Streaming IQ source reading raw binary complex64 files via memory mapping.

    Avoids loading arbitrarily large files into RAM by memory-mapping the file
    and reading bounded sequential chunks. Supports explicit endianness, resource
    lifecycle via close() / context manager, and empty recording handling.
    """

    def __init__(
        self,
        file_path: str | Path,
        stream_info: StreamInfo,
        block_size: int = 1024,
        start_sample: int = 0,
        start_timestamp_ns: int | None = None,
        endianness: Literal["little", "big"] = "little",
    ) -> None:
        if not isinstance(stream_info, StreamInfo):
            raise TypeError(f"stream_info must be a StreamInfo instance, got {type(stream_info)}")
        if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
            raise ValueError(f"block_size must be an integer >= 1, got {block_size!r}")
        if isinstance(start_sample, bool) or not isinstance(start_sample, (int, np.integer)) or start_sample < 0:
            raise ValueError(f"start_sample must be an integer >= 0, got {start_sample!r}")
        if endianness not in ("little", "big"):
            raise ValueError(f"endianness must be 'little' or 'big', got {endianness!r}")

        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"IQ recording file not found: {path}")

        file_size = os.path.getsize(path)
        if file_size % 8 != 0:
            raise ValueError(
                f"File size ({file_size} bytes) is not a multiple of 8 (sizeof np.complex64)"
            )

        dtype = np.dtype("<c8") if endianness == "little" else np.dtype(">c8")

        self._path = path
        self._stream_info = stream_info
        self._block_size = int(block_size)
        self._start_sample = int(start_sample)
        self._start_timestamp_ns = int(start_timestamp_ns) if start_timestamp_ns is not None else None
        self._endianness = endianness
        self._closed: bool = False

        if file_size == 0:
            self._mmap: np.ndarray | None = np.array([], dtype=np.complex64)
            self._total_samples = 0
        else:
            self._mmap = np.memmap(self._path, dtype=dtype, mode="r")
            self._total_samples = len(self._mmap)

        self._cursor: int = 0
        self._sequence_number: int = 0

    @property
    def stream_info(self) -> StreamInfo:
        """Stream metadata describing this source."""
        return self._stream_info

    @property
    def total_samples(self) -> int:
        """Total sample count in recorded file."""
        return self._total_samples

    @property
    def closed(self) -> bool:
        """Whether the file source has been closed."""
        return self._closed

    def close(self) -> None:
        """Close memory mapping and release file resources."""
        if not self._closed:
            self._closed = True
            if hasattr(self._mmap, "_mmap") and self._mmap._mmap is not None:
                try:
                    self._mmap._mmap.close()
                except Exception:
                    pass
            self._mmap = None

    def __enter__(self) -> RecordedFileIQSource:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def read_block(self) -> IQBlock | None:
        """Read and return the next sequential IQBlock, or None on EOF."""
        if self._closed:
            raise ValueError("RecordedFileIQSource is closed")

        if self._mmap is None or self._cursor >= self._total_samples:
            return None

        end_idx = min(self._cursor + self._block_size, self._total_samples)
        chunk = np.array(self._mmap[self._cursor : end_idx], dtype=np.complex64)
        block_start_sample = self._start_sample + self._cursor

        block_ts_ns: int | None = None
        if self._start_timestamp_ns is not None:
            offset_s = self._cursor / self._stream_info.sample_rate_hz
            block_ts_ns = self._start_timestamp_ns + int(round(offset_s * 1e9))

        block = IQBlock(
            stream=self._stream_info,
            sequence_number=self._sequence_number,
            start_sample=block_start_sample,
            timestamp_ns=block_ts_ns,
            iq=chunk,
            dropped_samples_before=0,
            overflow=False,
        )

        self._cursor = end_idx
        self._sequence_number += 1
        return block

    def reset(self) -> None:
        """Reset read cursor back to initial sample."""
        if self._closed:
            raise ValueError("RecordedFileIQSource is closed")
        self._cursor = 0
        self._sequence_number = 0
