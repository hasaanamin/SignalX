"""Finite replay with an independent sample clock, bounded scratch and loss accounting."""
from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass

from signal_processing.core import IQBlock
from signal_processing.core.stream_compatibility import require_stream
from signal_processing.core.timebase import sample_offset_ns
from .streaming import (SourceBusy, SourceCancelled, SourceClosed, SourceEnvelope,
                        SourceMetadata, SourceReadTimeout, read_timeout)


class MonotonicReplayClock:
    """No global timer changes; wait/cancel responsiveness is at most 50 ms."""

    def now_ns(self):
        return time.perf_counter_ns()

    def wait_until_ns(self, deadline, cancellation):
        while not cancellation.is_set():
            remaining = deadline - self.now_ns()
            if remaining <= 0:
                return
            # Windows Event.wait can use a coarse timeout clock. Python's sleep
            # uses the platform high-resolution waitable timer. Poll cancellation
            # without changing the process/system timer resolution.
            time.sleep(min(remaining / 1e9, .05))


@dataclass(frozen=True)
class ReplaySnapshot:
    total_samples: int
    read_samples: int
    returned_samples: int
    expired_samples: int
    unread_samples: int
    pending_samples: int
    trailing_lost_samples: int
    trailing_unknown_loss: bool
    source_epoch: int
    finished: bool
    cancelled: bool
    max_lateness_ns: int


class ReplayIQSource:
    """Wrap a finite IQSource (array/raw-file) without generating new samples.

    Scheduled availability is the block END on an absolute monotonic schedule;
    expiration is availability + lateness_budget_s, checked after waiting. A slow
    consumer cannot move that schedule. Unpaced replay has no expiration.

    timeout preserves a prepared block for retry. cancel is sticky until reset;
    reset rewinds the recording and increments source_epoch. This class owns at
    most one pending block plus the underlying source's declared storage.
    """

    def __init__(self, source, *, scheduled=False, lateness_budget_s=.05,
                 metadata=None, clock=None, owns_source=False):
        if type(scheduled) is not bool or type(owns_source) is not bool:
            raise TypeError("scheduled and owns_source must be bool")
        read_timeout(lateness_budget_s)
        if lateness_budget_s is None:
            raise ValueError("A finite lateness budget is required")
        if not callable(getattr(source, "read_block", None)) or type(getattr(source, "total_samples", None)) is not int or source.total_samples < 0:
            raise TypeError("Replay requires a finite positioned IQSource with total_samples")
        self._source = source
        self._stream = source.stream_info
        self._metadata = metadata or SourceMetadata(timestamp_origin="recording")
        if not isinstance(self._metadata, SourceMetadata):
            raise TypeError("metadata must be SourceMetadata")
        self._clock = clock or MonotonicReplayClock()
        self._scheduled = scheduled
        self._budget_ns = round(lateness_budget_s * 1e9)
        self._owns_source = owns_source
        self._gate = threading.Lock()
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._closed = False
        self._clear()

    def _clear(self):
        self._started = self._origin = self._previous = self._pending = None
        self._read = self._returned = self._expired = self._lost = self._max_late = 0
        self._unknown = self._finished = False

    @property
    def stream_info(self):
        return self._stream

    @property
    def total_samples(self):
        return self._source.total_samples

    def cancel(self):
        self._cancel.set()

    def snapshot(self):
        with self._lock:
            return ReplaySnapshot(self.total_samples, self._read, self._returned,
                                  self._expired, self.total_samples-self._read,
                                  self._pending.num_samples if self._pending is not None else 0,
                                  self._lost, self._unknown, self._metadata.source_epoch,
                                  self._finished, self._cancel.is_set(), self._max_late)

    def read_block(self):
        result = self.read_envelope()
        return result.block if result is not None else None

    def read_envelope(self, timeout=None):
        read_timeout(timeout)
        if not self._gate.acquire(blocking=False):
            raise SourceBusy("Replay supports exactly one reader")
        try:
            if self._closed:
                raise SourceClosed("Replay is closed")
            deadline = None if timeout is None else self._clock.now_ns() + round(timeout*1e9)
            with self._lock:
                if self._started is None:
                    self._started = self._clock.now_ns()
            while True:
                if self._cancel.is_set():
                    raise SourceCancelled("Replay cancelled")
                if self._finished:
                    return None
                if self._pending is None:
                    block = self._source.read_block()
                    if block is None:
                        with self._lock:
                            if self._read != self.total_samples:
                                raise ValueError("Source EOF contradicts declared total_samples")
                            self._finished = True
                        return None
                    if not isinstance(block, IQBlock):
                        raise TypeError("Source returned a non-IQBlock")
                    require_stream(block.stream, self._stream)
                    prev = self._previous
                    if prev is not None and (block.sequence_number <= prev.sequence_number
                            or block.start_sample != prev.end_sample + block.dropped_samples_before):
                        raise ValueError("Replay source coordinates/sequence are inconsistent")
                    with self._lock:
                        if self._read + block.num_samples > self.total_samples:
                            raise ValueError("Source exceeds declared total_samples")
                        if self._origin is None:
                            self._origin = block.start_sample
                        self._previous = self._pending = block
                        self._read += block.num_samples
                block = self._pending
                release = self._started + sample_offset_ns(block.end_sample-self._origin, self._stream.sample_rate_hz)
                if self._scheduled:
                    target = release if deadline is None else min(release, deadline)
                    self._clock.wait_until_ns(target, self._cancel)
                    if self._cancel.is_set():
                        raise SourceCancelled("Replay cancelled")
                    now = self._clock.now_ns()
                    if now < release:
                        raise SourceReadTimeout("Replay data is not available yet")
                    with self._lock:
                        self._max_late = max(self._max_late, now-release)
                        if now > release + self._budget_ns:
                            self._expired += block.num_samples
                            self._lost += block.num_samples + block.dropped_samples_before
                            self._unknown |= block.overflow
                            self._pending = None
                            if deadline is not None and now >= deadline:
                                raise SourceReadTimeout("Replay read deadline reached while accounting expired input")
                            continue
                else:
                    now = self._clock.now_ns()
                with self._lock:
                    # replace preserves immutable bytes; IQBlock construction would
                    # defensively copy again, so only construct when adding loss.
                    if self._lost or self._unknown:
                        block = dataclasses.replace(block,
                            dropped_samples_before=block.dropped_samples_before+self._lost,
                            overflow=block.overflow or self._unknown)
                    envelope = SourceEnvelope(block, self._metadata, now,
                                              release if self._scheduled else None)
                    self._returned += block.num_samples
                    self._lost = 0
                    self._unknown = False
                    self._pending = None
                    return envelope
        finally:
            self._gate.release()

    def reset(self):
        if not self._gate.acquire(blocking=False):
            raise SourceBusy("Cancel and finish the active read before reset")
        try:
            if self._closed:
                raise SourceClosed("Replay is closed")
            self._source.reset()
            with self._lock:
                self._metadata = dataclasses.replace(self._metadata, source_epoch=self._metadata.source_epoch+1)
                self._clear()
                self._cancel.clear()
        finally:
            self._gate.release()

    def close(self):
        self.cancel()
        if not self._gate.acquire(blocking=False):
            raise SourceBusy("Read cancellation requested; finish the reader and retry close")
        try:
            if not self._closed:
                if self._owns_source and callable(getattr(self._source, "close", None)):
                    self._source.close()
                self._closed = True
        finally:
            self._gate.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
