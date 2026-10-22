"""One bounded source-reader thread; consumers explicitly acknowledge deliveries.

No processing or sink callback executes on this reader. Queue and checked-out
deliveries share one capacity reservation, preventing absent consumers from
creating unlimited in-flight IQ. Finished input and finished work are distinct.
"""
from __future__ import annotations

import dataclasses
import threading
import time
from collections import deque
from dataclasses import dataclass

from signal_processing.core.stream_compatibility import require_stream
from signal_processing.sources.streaming import (
    PositionedIQSource, SourceCancelled, SourceEnvelope, SourceError,
    SourceReadTimeout, read_timeout,
)


class SourceRunnerFailed(SourceError):
    """Reader failed. Earlier deliveries remain available and must be acknowledged."""


@dataclass(frozen=True)
class SourceDelivery:
    delivery_id: int
    envelope: SourceEnvelope


@dataclass(frozen=True)
class SourceRunnerSnapshot:
    state: str
    terminal_reason: str | None
    read_samples: int
    admitted_samples: int
    queue_dropped_samples: int
    unaccepted_cancelled_samples: int
    processed_samples: int
    failed_samples: int
    cancelled_samples: int
    queued_samples: int
    in_flight_samples: int
    outstanding_deliveries: int
    peak_outstanding: int
    read_timeouts: int
    trailing_lost_samples: int
    trailing_unknown_loss: bool
    reader_alive: bool
    error: str | None
    unaccepted_failed_samples: int = 0

    @property
    def accounting_balanced(self):
        return (self.read_samples == self.admitted_samples + self.queue_dropped_samples + self.unaccepted_cancelled_samples+self.unaccepted_failed_samples
                and self.admitted_samples == self.processed_samples + self.failed_samples
                + self.cancelled_samples + self.queued_samples + self.in_flight_samples)


class SourceRunner:
    """Explicitly started single-use reader for a PositionedIQSource.

    get() returns a lease or None when no more input can arrive. None does not
    mean all checked-out leases have completed. Each lease requires one ack()
    with processed/failed/cancelled. Identity is checked; copied/foreign/double
    acknowledgements reject without changing counters.

    Drop-newest overload never blocks acquisition. The next accepted block
    carries accumulated known loss and overflow uncertainty. Trailing loss is
    visible in snapshot even if no later block exists. No IQ is fabricated.

    shutdown stops future acquisition. drain=True waits for accepted leases;
    drain=False cancels queued leases but still waits for checked-out work.
    A timeout leaves CLOSING, with resources/work truthful for a later retry.
    """

    def __init__(self, source: PositionedIQSource, *, capacity=16,
                 max_block_samples=16384, owns_source=False):
        if not isinstance(source, PositionedIQSource):
            raise TypeError("SourceRunner requires the cancellable PositionedIQSource interface")
        for value, name in ((capacity, "capacity"), (max_block_samples, "max_block_samples")):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(owns_source) is not bool:
            raise TypeError("owns_source must be bool")
        self._source = source
        self._stream = source.stream_info
        self._capacity, self._max_block = capacity, max_block_samples
        self._owns_source = owns_source
        self._cleanup_pending = owns_source
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._queue = deque()
        self._leases = {}
        self._state = "NEW"
        self._thread = None
        self._done = False
        self._reason = self._error = None
        self._read = self._admitted = self._dropped = 0
        self._unaccepted_cancelled = 0
        self._processed = self._failed = self._cancelled = 0
        self._peak = self._timeouts = self._sequence = 0
        self._lost = 0
        self._unknown = False

    def start(self):
        with self._condition:
            if self._state != "NEW":
                raise SourceError("SourceRunner is single-use; start requires NEW")
            self._thread = threading.Thread(target=self._run, name="iq-source-reader", daemon=True)
            self._state = "RUNNING"
            try:
                self._thread.start()
            except BaseException:
                self._state = "NEW"
                self._thread = None
                raise
        return self

    def _run(self):
        previous = None
        reason = "cancelled"
        try:
            while not self._stop.is_set():
                try:
                    envelope = self._source.read_envelope(timeout=.05)
                except SourceReadTimeout:
                    with self._condition:
                        self._timeouts += 1
                    continue
                if envelope is None:
                    reason = "eof"
                    break
                if not isinstance(envelope, SourceEnvelope):
                    raise TypeError("Source returned an invalid envelope")
                block = envelope.block
                require_stream(block.stream, self._stream)
                if block.num_samples > self._max_block:
                    raise ValueError("Source block exceeds configured max_block_samples")
                if previous is not None:
                    p = previous.block
                    if envelope.metadata != previous.metadata:
                        raise ValueError("Source epoch/configuration changed; restart the runner explicitly")
                    if block.sequence_number <= p.sequence_number or block.start_sample != p.end_sample + block.dropped_samples_before:
                        raise ValueError("Source block position/sequence contradicts declared loss")
                previous = envelope
                with self._condition:
                    self._read += block.num_samples
                    stopping = self._stop.is_set()
                    if stopping or len(self._queue)+len(self._leases) >= self._capacity:
                        if stopping:
                            self._unaccepted_cancelled += block.num_samples
                        else:
                            self._dropped += block.num_samples
                        self._lost += block.num_samples + block.dropped_samples_before
                        self._unknown |= block.overflow
                    else:
                        if self._lost or self._unknown:
                            block = dataclasses.replace(block,
                                dropped_samples_before=block.dropped_samples_before+self._lost,
                                overflow=block.overflow or self._unknown)
                            envelope = dataclasses.replace(envelope, block=block)
                        self._lost = 0
                        self._unknown = False
                        self._sequence += 1
                        self._queue.append(SourceDelivery(self._sequence, envelope))
                        self._admitted += block.num_samples
                        self._peak = max(self._peak, len(self._queue)+len(self._leases))
                        self._condition.notify_all()
        except SourceCancelled:
            reason = "cancelled"
        except Exception as exc:
            reason = "failed"
            with self._condition:
                self._error = exc.with_traceback(None)
        finally:
            if self._owns_source:
                try:
                    self._source.close()
                    self._cleanup_pending = False
                except Exception as exc:
                    reason = "failed"
                    with self._condition:
                        if self._error is None:
                            self._error = exc.with_traceback(None)
            with self._condition:
                self._reason = "failed" if self._error is not None else reason
                self._done = True
                self._condition.notify_all()

    def get(self, timeout=None):
        read_timeout(timeout)
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            if self._state == "NEW":
                raise SourceError("Start the source runner before get")
            while not self._queue:
                if self._done:
                    if self._error is not None:
                        raise SourceRunnerFailed(f"Source reader failed: {type(self._error).__name__}: {self._error}") from self._error
                    return None
                remaining = None if deadline is None else deadline-time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise SourceReadTimeout("No source delivery before timeout")
                self._condition.wait(remaining)
            result = self._queue.popleft()
            self._leases[result.delivery_id] = result
            return result

    def ack(self, delivery: SourceDelivery, *, outcome="processed"):
        if outcome not in ("processed", "failed", "cancelled"):
            raise ValueError("outcome must be processed, failed or cancelled")
        with self._condition:
            if not isinstance(delivery, SourceDelivery) or self._leases.get(delivery.delivery_id) is not delivery:
                raise ValueError("Unknown, copied or already acknowledged source delivery")
            del self._leases[delivery.delivery_id]
            n = delivery.envelope.block.num_samples
            if outcome == "processed":
                self._processed += n
            elif outcome == "failed":
                self._failed += n
            else:
                self._cancelled += n
            self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            return SourceRunnerSnapshot(self._state, self._reason, self._read,
                self._admitted, self._dropped, self._unaccepted_cancelled, self._processed, self._failed,
                self._cancelled, sum(x.envelope.block.num_samples for x in self._queue),
                sum(x.envelope.block.num_samples for x in self._leases.values()),
                len(self._queue)+len(self._leases), self._peak, self._timeouts,
                self._lost, self._unknown, bool(self._thread and self._thread.is_alive()),
                None if self._error is None else f"{type(self._error).__name__}: {self._error}")

    def shutdown(self, *, drain=True, timeout=5.0):
        read_timeout(timeout)
        if type(drain) is not bool:
            raise TypeError("drain must be bool")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            if self._state == "CLOSED":
                return
            self._state = "CLOSING"
            self._stop.set()
            if not drain:
                while self._queue:
                    self._cancelled += self._queue.popleft().envelope.block.num_samples
            if self._thread is None:
                self._done = True
                self._reason = "cancelled"
        # Source code is external to lifecycle locks.
        try:
            self._source.cancel()
        except Exception as exc:
            with self._condition:
                if self._error is None:
                    self._error = exc.with_traceback(None)
                self._reason = "failed"
        if self._thread is threading.current_thread():
            raise SourceError("Source reader cannot synchronously drain itself")
        if self._thread is not None:
            self._thread.join(None if deadline is None else max(0, deadline-time.monotonic()))
        if (self._thread is None or not self._thread.is_alive()) and self._cleanup_pending:
            try:
                self._source.close()
                self._cleanup_pending = False
            except Exception as exc:
                raise SourceRunnerFailed("Source cleanup failed; runner remains CLOSING") from exc
        with self._condition:
            while (self._thread is not None and self._thread.is_alive()) or self._queue or self._leases:
                remaining = None if deadline is None else deadline-time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("Source runner remains CLOSING: reader or deliveries unfinished")
                self._condition.wait(remaining)
            self._state = "CLOSED"
            self._condition.notify_all()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.shutdown(drain=False)
