"""Streaming complex64 IQ with bounded mapping or sequential buffered reads."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np

from signal_processing.core.contracts import IQBlock, StreamInfo
from signal_processing.core.timebase import sample_offset_ns


class RecordedFileIQSource:
    """Read raw complex64 through bounded mapping or sequential buffered I/O.

    Maps at most one bounded sequential region, so closing a long recording
    does not unmap its entire received history on the EOF path. Supports endianness, resource
    lifecycle via close() / context manager, and empty recording handling.
    Buffered mode uses at most a 1-MiB read buffer or one larger transport block.
    """

    def __init__(
        self,
        file_path: str | Path,
        stream_info: StreamInfo,
        block_size: int = 1024,
        start_sample: int = 0,
        start_timestamp_ns: int | None = None,
        endianness: Literal["little", "big"] = "little",
        read_backend: Literal["mmap", "buffered"] = "mmap",
    ) -> None:
        if not isinstance(stream_info, StreamInfo):
            raise TypeError(f"stream_info must be a StreamInfo instance, got {type(stream_info)}")
        if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
            raise ValueError(f"block_size must be an integer >= 1, got {block_size!r}")
        if isinstance(start_sample, bool) or not isinstance(start_sample, (int, np.integer)) or start_sample < 0:
            raise ValueError(f"start_sample must be an integer >= 0, got {start_sample!r}")
        if endianness not in ("little", "big"):
            raise ValueError(f"endianness must be 'little' or 'big', got {endianness!r}")
        if read_backend not in ('mmap','buffered'):
            raise ValueError('read_backend must be mmap or buffered')

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
        self._dtype = dtype
        self._read_backend = read_backend
        self._buffer_bytes = max(1024*1024,self._block_size*8)
        if read_backend == 'buffered':
            self._file = open(path,'rb',buffering=self._buffer_bytes,
                opener=lambda name,flags: os.open(name,flags | getattr(os,'O_SEQUENTIAL',0)))
        else:
            self._file = path.open('rb')
        self._mmap: np.memmap | None = None
        self._mapped_start = 0
        self._mapping_samples = max(131072,self._block_size)
        self._total_samples = file_size//8

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
            self._unmap()
            self._file.close()
            self._closed = True

    def _unmap(self) -> None:
        if self._mmap is not None:
            self._mmap._mmap.close()
            self._mmap = None

    def __enter__(self) -> RecordedFileIQSource:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def read_block(self) -> IQBlock | None:
        """Read and return the next sequential IQBlock, or None on EOF."""
        if self._closed:
            raise ValueError("RecordedFileIQSource is closed")

        if self._cursor >= self._total_samples:
            return None

        end_idx = min(self._cursor + self._block_size, self._total_samples)
        if self._read_backend == 'buffered':
            size = (end_idx-self._cursor)*8
            payload = self._file.read(size)
            if len(payload) != size:
                raise ValueError('Recording changed or truncated during sequential reading')
            chunk = np.frombuffer(payload,dtype=self._dtype).astype(np.complex64,copy=False)
        else:
            if (self._mmap is None or self._cursor < self._mapped_start
                    or end_idx > self._mapped_start+len(self._mmap)):
                self._unmap()
                self._mapped_start = self._cursor
                self._mmap = np.memmap(self._file,dtype=self._dtype,mode='r',offset=self._cursor*8,
                                       shape=(min(self._mapping_samples,self._total_samples-self._cursor),))
            begin = self._cursor-self._mapped_start
            chunk = np.array(self._mmap[begin:begin+end_idx-self._cursor], dtype=np.complex64)
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
        if self._read_backend == 'buffered':
            self._file.seek(0)
