"""Spawn-isolated positioned acquisition with bounded, acknowledged byte slots.

The factory is an importable callable producing an owned PositionedIQSource.
Readiness precedes source-clock startup. Slots remain reserved until ack(), not
merely until get(). Decoding copies into immutable IQ before exposing a lease.
No user callback runs in the acquisition process or under a lifecycle lock.
"""
from __future__ import annotations

import dataclasses
import multiprocessing as mp
import sys
import threading
import time

from signal_processing.core.stream_compatibility import require_stream
from signal_processing.core.contracts import StreamInfo
from signal_processing.sources.streaming import (
    SourceCancelled, SourceEnvelope, SourceError, SourceReadTimeout, PositionedIQSource,
    encode_envelope, decode_envelope, read_timeout,
)
from .source_runner import SourceDelivery, SourceRunnerFailed, SourceRunnerSnapshot


def _acquire(factory, stream, capacity, maximum, slots, lengths, sample_counts, free_ids, ready_ids, counters,
             free, ready, stop, begin, status, performance_qos=False, acquisition_cpus=None):
    from contextlib import ExitStack
    resources = ExitStack()
    source_policy = None
    source = None
    previous = None
    lost, unknown, reason, error = 0, False, 'cancelled', None
    replay = None
    free_taken = 0
    try:
        if acquisition_cpus is not None:
            import psutil
            psutil.Process().cpu_affinity(list(acquisition_cpus))
        source = factory()
        if not isinstance(source,PositionedIQSource):
            raise TypeError('Factory must produce a cancellable PositionedIQSource')
        require_stream(source.stream_info, stream)
        # This fresh, owned acquisition interpreter has just imported NumPy and
        # initialized its source. Finish collecting temporary import graphs
        # before readiness, rather than charging that collection to a live RF
        # deadline. Automatic collection remains enabled throughout acquisition.
        import gc
        gc.collect()
        if performance_qos:
            from ._performance import performance_scope,realtime_scope
            source_policy = dict(
                performance=resources.enter_context(performance_scope(acquisition=True)),
                realtime=resources.enter_context(realtime_scope(priority=2)))
        status.send(('ready', source_policy))
        while not begin.wait(.05):
            if stop.is_set():
                return
        width = maximum*8+8200
        while not stop.is_set():
            try:
                envelope = source.read_envelope(timeout=.05)
            except SourceReadTimeout:
                counters[4] += 1
                continue
            if envelope is None:
                reason = 'eof'
                break
            if not isinstance(envelope, SourceEnvelope):
                raise TypeError('Invalid source envelope')
            b = envelope.block
            require_stream(b.stream, stream)
            if b.num_samples > maximum:
                raise ValueError('Source block exceeds max_block_samples')
            if previous is not None:
                p = previous.block
                if envelope.metadata != previous.metadata:
                    raise ValueError('Source epoch/configuration changed; restart explicitly')
                if b.sequence_number <= p.sequence_number or b.start_sample != p.end_sample+b.dropped_samples_before:
                    raise ValueError('Source position/sequence contradicts declared loss')
            previous = envelope
            counters[0] += b.num_samples
            stopping = stop.is_set()
            if stopping or not free.acquire(False):
                counters[3 if stopping else 2] += b.num_samples
                lost += b.num_samples+b.dropped_samples_before
                unknown |= b.overflow
                continue
            try:
                if lost or unknown:
                    b = dataclasses.replace(b, dropped_samples_before=b.dropped_samples_before+lost,
                                            overflow=b.overflow or unknown)
                    envelope = dataclasses.replace(envelope, block=b)
                payload = encode_envelope(envelope, max_samples=maximum)
                index = free_ids[free_taken % capacity]
                free_taken += 1
                offset = index*width
                memoryview(slots).cast('B')[offset:offset+len(payload)] = payload
                lengths[index] = len(payload)
                sample_counts[index] = b.num_samples
                ready_ids[counters[5] % capacity] = index
                counters[1] += b.num_samples
                lost, unknown = 0, False
                counters[5] += 1
                counters[6] = max(counters[6], counters[5]-counters[7])
                ready.release()
            except BaseException:
                free.release()
                raise
    except SourceCancelled:
        reason = 'cancelled'
    except BaseException as exc:
        reason, error = 'failed', f'{type(exc).__name__}: {exc}'
    finally:
        if source is not None:
            if hasattr(source, 'snapshot'):
                try:
                    replay = source.snapshot()
                except BaseException as exc:
                    reason,error = 'failed',f'source accounting: {type(exc).__name__}: {exc}'
            try:
                source.close()
            except BaseException as exc:
                reason, error = 'failed', f'cleanup: {type(exc).__name__}: {exc}'
        try:
            resources.close()
        except BaseException as exc:
            reason,error = 'failed',f'scheduling cleanup: {type(exc).__name__}: {exc}'
        if source_policy is not None:
            status.send(('policy',source_policy))
        status.send(('done', (reason, error, lost, unknown, replay)))
        status.close()


class ProcessSourceRunner:
    """SourceRunner lease API, with one persistent spawn acquisition process.

    Factory initialization must finish within startup_timeout. It must support
    bounded read timeouts; cancellation is observed between reads. Shutdown
    timeout leaves CLOSING and never forcibly recycles an outstanding lease.
    A dead process is reported, never interpreted as a successful EOF.
    """
    def __init__(self, factory, stream_info, *, capacity=16, max_block_samples=16384,
                 startup_timeout=15.,performance_qos=False,acquisition_cpus=None):
        if type(performance_qos) is not bool:
            raise TypeError('performance_qos must be bool')
        self._performance_qos = performance_qos
        if acquisition_cpus is not None and (not isinstance(acquisition_cpus,(tuple,list))
                or not acquisition_cpus or any(type(cpu) is not int or cpu < 0 for cpu in acquisition_cpus)):
            raise ValueError('Acquisition CPU placement must be a nonempty sequence of CPU indices')
        self._acquisition_cpus = None if acquisition_cpus is None else tuple(acquisition_cpus)
        self._source_policy = None
        if sys.maxsize<=2**32:
            raise ValueError('ProcessSourceRunner requires a 64-bit interpreter')
        if not isinstance(stream_info,StreamInfo):
            raise TypeError('stream_info must be an immutable StreamInfo')
        for value in (capacity, max_block_samples):
            if type(value) is not int or value < 1:
                raise ValueError('Capacity and max_block_samples must be positive integers')
        read_timeout(startup_timeout)
        if not callable(factory):
            raise TypeError('factory must be an importable spawn-compatible callable')
        self._ctx = mp.get_context('spawn')
        self._factory, self._stream = factory, stream_info
        self._capacity, self._maximum = capacity, max_block_samples
        self._startup_timeout = startup_timeout
        self._width = max_block_samples*8+8200
        self._slots = self._ctx.RawArray('B', capacity*self._width)
        self._lengths = self._ctx.RawArray('q', capacity)
        self._sample_counts = self._ctx.RawArray('q', capacity)
        self._free_ids = self._ctx.RawArray('q', list(range(capacity)))
        self._ready_ids = self._ctx.RawArray('q', capacity)
        # Single writer per counter (producer 0..6, parent 7).
        self._counts = self._ctx.RawArray('q', 8)
        self._free, self._ready = self._ctx.Semaphore(capacity), self._ctx.Semaphore(0)
        self._stop, self._begin = self._ctx.Event(), self._ctx.Event()
        self._status, self._send_status = self._ctx.Pipe(duplex=False)
        self._process = None
        self._process_closed = False
        self._pid = None
        self._handshake_pending = False
        self._state, self._reason, self._error = 'NEW', None, None
        self._consumer_error = None
        self._terminal = False
        self._replay = None
        self._lost, self._unknown = 0, False
        self._taken = self._processed = self._failed = self._cancelled = 0
        self._taken_samples = 0
        self._unaccepted_failed = 0
        self._leases = {}
        self._lease_slots = {}
        self._lock = threading.RLock()

    @property
    def pid(self):
        """Owned acquisition PID for explicit host placement/diagnostics."""
        return self._pid

    def start(self, *, start_paused=False):
        if type(start_paused) is not bool:
            raise TypeError('start_paused must be bool')
        with self._lock:
            if self._state != 'NEW':
                raise SourceError('ProcessSourceRunner is single-use')
            self._process = self._ctx.Process(target=_acquire, name='positioned-iq-source',
                args=(self._factory,self._stream,self._capacity,self._maximum,self._slots,
                      self._lengths,self._sample_counts,self._free_ids,self._ready_ids,self._counts,self._free,self._ready,self._stop,self._begin,self._send_status,self._performance_qos,self._acquisition_cpus))
            self._state,self._handshake_pending = 'STARTING',True
        # Spawn/pickling/readiness may involve external source code. A concurrent
        # shutdown must still honor its own deadline, not wait on this lock.
        deadline=None if self._startup_timeout is None else time.monotonic()+self._startup_timeout
        try:
            try:
                self._process.start()
                self._pid = self._process.pid
            except BaseException as exc:
                with self._lock:
                    self._process.close()
                    self._process = None
                    self._state,self._reason,self._terminal = 'CLOSING','failed',True
                    self._error=f'Spawn failed: {type(exc).__name__}: {exc}'
                raise
            self._send_status.close()
            try:
                while True:
                    if self._stop.is_set():
                        raise SourceCancelled('Acquisition cancelled during startup')
                    remaining=None if deadline is None else deadline-time.monotonic()
                    if remaining is not None and remaining<=0:
                        raise TimeoutError('Acquisition process did not become ready')
                    if self._status.poll(.05 if remaining is None else min(.05,remaining)):
                        break
                tag, value = self._status.recv()
            except (EOFError,BrokenPipeError) as exc:
                self._state,self._reason,self._terminal = 'CLOSING','failed',True
                self._error='Acquisition process disconnected before readiness'
                self._stop.set()
                raise SourceRunnerFailed(self._error) from exc
            with self._lock:
                if tag != 'ready':
                    self._accept_status(tag, value)
                    raise SourceRunnerFailed(self._error or 'Source initialization failed')
                if self._stop.is_set():
                    raise SourceCancelled('Acquisition cancelled during startup')
                self._source_policy = value
                self._state = 'READY'
                if not start_paused:
                    self.begin()
        except BaseException:
            with self._lock:
                if self._state!='NEW': self._state='CLOSING'
                self._stop.set()
            raise
        finally:
            with self._lock: self._handshake_pending=False
        return self

    def begin(self):
        """Start input exactly once after READY (e.g. after owned CPU placement)."""
        with self._lock:
            if self._state != 'READY':
                raise SourceError('begin requires a ready, not-yet-started acquisition')
            self._state = 'RUNNING'
            self._begin.set()

    def _accept_status(self, tag, value):
        if tag == 'policy':
            self._source_policy = value
        if tag == 'done':
            self._reason,self._error,self._lost,self._unknown,self._replay = value
            if self._consumer_error is not None:
                self._reason,self._error = 'failed',self._consumer_error
            self._terminal = True

    def source_execution(self):
        """Detached scheduling diagnostics from the owned acquisition process."""
        import copy
        with self._lock:
            self._poll()
            return copy.deepcopy(self._source_policy)

    def _poll(self):
        if self._handshake_pending:
            return # The startup caller is the sole status-pipe reader until ready.
        if not self._terminal:
            while True:
                try:
                    if not self._status.poll():
                        break
                    self._accept_status(*self._status.recv())
                    if self._terminal:
                        break
                except (EOFError, BrokenPipeError):
                    self._reason,self._error,self._terminal = 'failed','Acquisition process disconnected without terminal accounting',True
                    break
            if self._process is not None and not self._alive() and not self._terminal:
                self._reason, self._error, self._terminal = 'failed', 'Acquisition process died without terminal accounting', True

    def _alive(self):
        return bool(self._process is not None and not self._process_closed and self._process.is_alive())

    def _reconcile_dead_publication(self):
        """A confirmed dead writer cannot finish a partly published slot.

        Drain all real ready tokens first. Only then classify any committed
        admissions lacking a ready token as failed, not queued forever. Read IQ
        not yet admitted is a distinct software-failure category, NOT RF loss.
        """
        if not self._error or self._handshake_pending or self._alive(): return
        if self._ready.acquire(False):
            self._ready.release()
            return
        remaining=self._counts[1]-self._taken_samples
        if remaining>0:
            self._failed+=remaining
            self._taken_samples+=remaining
        missing=self._counts[5]-self._taken
        if missing>0:
            self._taken=self._counts[5]
            self._counts[7]+=missing
        self._unaccepted_failed=max(0,self._counts[0]-self._counts[1]-self._counts[2]-self._counts[3])

    def get(self, timeout=None, *, spin=False):
        """Receive one lease; spin=True keeps a reserved DSP core active.

        Blocking is the compatibility default. Spin mode still releases the
        lifecycle lock between polls, observes the same timeout/terminal state
        and transfers ownership through the same ready semaphore exactly once.
        """
        read_timeout(timeout)
        if type(spin) is not bool:
            raise ValueError('Spin wait selection must be bool')
        deadline = None if timeout is None else time.monotonic()+timeout
        with self._lock:
            if self._state == 'NEW':
                raise SourceError('Start the source runner before get')
            while True:
                if self._state == 'CLOSED':
                    if self._error: raise SourceRunnerFailed(self._error)
                    return None
                self._poll()
                if self._ready.acquire(False):
                    index = self._ready_ids[self._taken % self._capacity]
                    offset = index*self._width
                    payload = bytes(memoryview(self._slots).cast('B')[offset:offset+self._lengths[index]])
                    try:
                        if not 0 < self._lengths[index] <= self._width:
                            raise ValueError('Malformed source slot length')
                        envelope = decode_envelope(payload, max_samples=self._maximum)
                        require_stream(envelope.block.stream,self._stream)
                        if envelope.block.num_samples != self._sample_counts[index]:
                            raise ValueError('Source slot sample count contradicts envelope')
                    except Exception as exc:
                        # This accepted delivery failed before a public lease
                        # could be formed. Account/release it exactly once;
                        # remaining valid slots remain drainable during cleanup.
                        self._taken += 1
                        n = self._sample_counts[index]
                        self._taken_samples += n
                        self._failed += n
                        self._free_ids[self._counts[7] % self._capacity] = index
                        self._counts[7] += 1
                        self._free.release()
                        self._stop.set()
                        self._consumer_error = f'Invalid acquired envelope: {type(exc).__name__}: {exc}'
                        self._reason,self._error = 'failed',self._consumer_error
                        raise SourceRunnerFailed(f'Invalid acquired envelope: {exc}') from exc
                    self._taken += 1
                    self._taken_samples += envelope.block.num_samples
                    lease = SourceDelivery(self._taken, envelope)
                    self._leases[lease.delivery_id] = lease
                    self._lease_slots[lease.delivery_id] = index
                    return lease
                self._reconcile_dead_publication()
                if self._terminal and self._taken == self._counts[5]:
                    if self._error:
                        raise SourceRunnerFailed(self._error)
                    return None
                remaining = None if deadline is None else deadline-time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise SourceReadTimeout('No process source delivery before timeout')
                # Blocking uses a semaphore wakeup. Explicit spin keeps this
                # reserved caller active between samples, but releases the
                # lifecycle lock so cancellation/shutdown can still proceed.
                readiness = self._ready
                self._lock.release()
                try:
                    if not spin and readiness.acquire(timeout=.01 if remaining is None else min(.01,remaining)):
                        readiness.release()
                finally:
                    self._lock.acquire()

    def ack(self, delivery, *, outcome='processed'):
        if outcome not in ('processed','failed','cancelled'):
            raise ValueError('Invalid delivery outcome')
        with self._lock:
            if not isinstance(delivery,SourceDelivery) or self._leases.get(delivery.delivery_id) is not delivery:
                raise ValueError('Unknown, copied or acknowledged source delivery')
            del self._leases[delivery.delivery_id]
            index = self._lease_slots.pop(delivery.delivery_id)
            n = delivery.envelope.block.num_samples
            if outcome == 'processed': self._processed += n
            elif outcome == 'failed': self._failed += n
            else: self._cancelled += n
            self._free_ids[self._counts[7] % self._capacity] = index
            self._counts[7] += 1
            self._free.release()

    def source_snapshot(self):
        with self._lock:
            self._poll()
            if not self._terminal:
                raise SourceError('Source snapshot is available after acquisition terminates')
            return self._replay

    def snapshot(self):
        with self._lock:
            self._poll()
            c = self._counts
            return SourceRunnerSnapshot(self._state,self._reason,c[0],c[1],c[2],c[3],
                self._processed,self._failed,self._cancelled,c[1]-self._taken_samples,
                sum(x.envelope.block.num_samples for x in self._leases.values()),
                c[5]-c[7],c[6],c[4],self._lost,self._unknown,
                self._alive(),self._error,self._unaccepted_failed)

    def shutdown(self, *, drain=True, timeout=5.):
        read_timeout(timeout)
        if type(drain) is not bool:
            raise TypeError('drain must be bool')
        deadline = None if timeout is None else time.monotonic()+timeout
        with self._lock:
            if self._state == 'CLOSED': return
            self._state = 'CLOSING'
            self._stop.set()
            self._begin.set()
        while True:
            with self._lock:
                self._poll()
                if not drain:
                    while self._taken < self._counts[5]:
                        try: delivery = self.get(timeout=0)
                        except (SourceReadTimeout,SourceRunnerFailed): break
                        self.ack(delivery,outcome='cancelled')
                self._reconcile_dead_publication()
                done = not self._alive() and not self._handshake_pending
                if done and self._taken == self._counts[5] and not self._leases:
                    if self._process is not None and not self._process_closed:
                        self._process.join(timeout=0)
                        self._process.close()
                        self._process_closed = True
                    self._status.close()
                    self._send_status.close()
                    # No reader/lease can still touch the pool. Release the
                    # finite IQ storage and semaphore handles even if the
                    # application keeps this completed runner's diagnostics.
                    self._slots = self._lengths = self._sample_counts = None
                    self._free_ids = self._ready_ids = None
                    self._free = self._ready = None
                    if not self._terminal:
                        self._terminal,self._reason = True,'cancelled'
                    self._state = 'CLOSED'
                    return
            remaining = None if deadline is None else deadline-time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError('Process source remains CLOSING: acquisition or leases unfinished')
            if self._process is not None and not self._process_closed and self._pid is not None:
                self._process.join(timeout=.005 if remaining is None else min(.005,remaining))
            else:
                time.sleep(.005 if remaining is None else min(.005,remaining))


@dataclasses.dataclass(frozen=True)
class ReplayFileFactory:
    """Spawn-safe recorded-source configuration; clock starts on first read."""
    path: str
    stream: object
    block_size: int = 16384
    scheduled: bool = True
    lateness_budget_s: float = .01
    start_sample: int = 0
    read_backend: str = 'buffered'

    def __call__(self):
        from signal_processing.sources import RecordedFileIQSource, ReplayIQSource, SourceMetadata
        recording = RecordedFileIQSource(self.path,self.stream,self.block_size,
            start_sample=self.start_sample,read_backend=self.read_backend)
        try:
            if self.scheduled:
                # Warm finite file I/O, validators and bounded scratch before
                # readiness. Rewind without advancing the replay sample clock;
                # the first scheduled block is still returned in full.
                recording.read_block()
                recording.reset()
            return ReplayIQSource(recording,scheduled=self.scheduled,
                lateness_budget_s=self.lateness_budget_s,
                metadata=SourceMetadata(),owns_source=True)
        except BaseException:
            recording.close()
            raise
