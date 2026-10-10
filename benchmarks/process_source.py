"""Isolated version of the SAME scheduled synthetic source, not a hardware claim.

The RNG, bursts, absolute schedule, expiry checks and drop-newest policy match
the compatibility benchmark. A terminal queue record, not Queue.empty(), proves
all submitted records have crossed the feeder. Capacity remains 16 by default.
"""
import math
import multiprocessing
import queue
import time

import numpy as np

from signal_processing.core import IQBlock
from signal_processing.runtime.scheduling import wait_until


class _ScheduledIQ:
    """Finite producer scratch; same seed/draw order and complex64 workload.

    Returned view is borrowed until the next generate call. The source copies it
    into immutable bytes before publication; no borrowed view reaches a consumer.
    """
    def __init__(self, max_samples):
        self.rng=np.random.default_rng(730)
        self.noise=np.empty(2*max_samples,np.float64)
        self.iq=np.empty(max_samples,np.complex128)
        self.output=np.empty(max_samples,np.complex64)
        self.indices=np.arange(max_samples,dtype=np.int64)
        self.scale=math.sqrt(1e-4/2)

    def generate(self,start,n):
        values=self.noise[:2*n].reshape(2,n)
        self.rng.standard_normal(out=values)
        iq=self.iq[:n]
        np.multiply(values[0],self.scale,out=iq.real)
        np.multiply(values[1],self.scale,out=iq.imag)
        iq[((self.indices[:n]+start)%4096)<256]+=.1
        self.output[:n]=iq
        return self.output[:n]


def _run(connection, output, stop, total, block_size, rate):
    stats = dict(produced_samples=0, admitted_samples=0, producer_deadline_dropped_samples=0,
        queue_dropped_samples=0, trailing_dropped_samples=0, acquisition_queue_peak=0,
        max_producer_lateness_ms=0., producer_preparation_seconds=0.,
        producer_wait_seconds=0., producer_queue_seconds=0., errors=[])
    lost = 0
    try:
        generator = _ScheduledIQ(block_size)
        connection.send(('ready', None))
        started = connection.recv()
        cpu = time.thread_time()
        interval = block_size/rate
        for seq, start in enumerate(range(0, total, block_size)):
            if stop.is_set():
                break
            n = min(block_size, total-start)
            arrival = started+(start+n)/rate
            if time.perf_counter() > arrival+interval:
                stats['produced_samples'] += n
                stats['producer_deadline_dropped_samples'] += n
                lost += n
                continue
            before = time.perf_counter()
            iq = generator.generate(start,n)
            stats['producer_preparation_seconds'] += time.perf_counter()-before
            before = time.perf_counter()
            if not wait_until(arrival, stop):
                break
            now = time.perf_counter()
            stats['producer_wait_seconds'] += now-before
            stats['max_producer_lateness_ms'] = max(stats['max_producer_lateness_ms'],1000*max(0,now-arrival))
            stats['produced_samples'] += n
            if now > arrival+interval:
                stats['producer_deadline_dropped_samples'] += n
                lost += n
                continue
            before = time.perf_counter()
            try:
                output.put_nowait((arrival, seq, start, iq.tobytes(), lost))
                stats['admitted_samples'] += n
                stats['acquisition_queue_peak'] = max(stats['acquisition_queue_peak'], output.qsize())
                lost = 0
            except queue.Full:
                stats['queue_dropped_samples'] += n
                lost += n
            stats['producer_queue_seconds'] += time.perf_counter()-before
        stats['producer_cpu_seconds'] = time.thread_time()-cpu
        stats['producer_finished_seconds'] = time.perf_counter()-started
    except Exception as exc:
        stats['errors'].append(f'producer: {type(exc).__name__}: {exc}')
    finally:
        stats['trailing_dropped_samples'] = lost
        connection.send(('finished', stats))
        # FIFO terminal marker follows every accepted IQ record. During abort,
        # parent drains queued data before joining; no feeder join deadlock.
        while True:
            try:
                output.put(None, timeout=.05)
                break
            except queue.Full:
                if stop.is_set():
                    output.cancel_join_thread()
                    break
        connection.close()


class ProcessScheduledSource:
    def __init__(self, stream, total, block_size, capacity, transport='shared'):
        self.stream = stream
        ctx = multiprocessing.get_context('spawn')
        if transport == 'shared':
            from signal_processing.runtime.shared_transport import SharedIQTransport
            self._queue = SharedIQTransport(ctx, capacity, block_size)
        elif transport == 'pickle':
            self._queue = ctx.JoinableQueue(capacity)
        else:
            raise ValueError('Unknown transport')
        self.stop = ctx.Event()
        self._connection, child = ctx.Pipe()
        self._process = ctx.Process(target=_run,
            args=(child,self._queue,self.stop,total,block_size,stream.sample_rate_hz),
            name='RF-scheduled-source',daemon=True)
        self.done = False
        self._closed = False
        self._taken = self._acknowledged = 0
        try:
            self._process.start()
            child.close()
            if not self._connection.poll(15):
                raise TimeoutError('Scheduled source readiness timed out')
            if self._connection.recv() != ('ready', None):
                raise RuntimeError('Scheduled source failed startup')
        except BaseException:
            child.close()
            self.close()
            raise

    def start(self, started):
        self._connection.send(started)

    def get(self, block=True, timeout=None):
        try:
            record = self._queue.get(block, timeout)
        except queue.Empty:
            if not self._process.is_alive() and not self.done:
                raise RuntimeError('Scheduled source died before terminal record')
            raise
        if record is None:
            self._queue.task_done()
            self.done = True
            raise queue.Empty
        arrival, seq, start, data, lost = record
        self._taken += 1
        return arrival, IQBlock(self.stream,seq,start,None,np.frombuffer(data,np.complex64),
                               dropped_samples_before=lost)

    def get_nowait(self):
        return self.get(False)

    def task_done(self):
        self._queue.task_done()
        self._acknowledged += 1

    @property
    def unfinished_tasks(self):
        # For final diagnostics only; completion is the explicit terminal marker.
        return self._taken-self._acknowledged+self.qsize()

    def qsize(self):
        return self._queue.qsize()

    def empty(self):
        return self._queue.empty()

    def is_set(self):
        return self.done

    def is_alive(self):
        return self._process.is_alive()

    def join(self, timeout):
        self._process.join(timeout)

    def finish(self):
        if self._process.is_alive():
            self._process.terminate(); self._process.join(2)
            failed = True
        else:
            failed = False
        if not self._connection.poll(2):
            raise RuntimeError('Source final accounting unavailable')
        status, stats = self._connection.recv()
        self._connection.close()
        if status != 'finished':
            raise RuntimeError('Invalid source final accounting')
        if failed:
            stats['errors'].append('Producer required forced termination')
        return stats

    def close(self):
        """Idempotent teardown, also safe after partially completed startup.

        Never release slot mappings while a live producer may still access them.
        Unknown final counts remain unknown; close does not manufacture a report.
        """
        if self._closed:
            return
        self.stop.set()
        if self._process.pid is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(2)
            if self._process.is_alive():
                raise TimeoutError('Scheduled source cleanup incomplete')
        try:
            self._connection.close()
        finally:
            try:
                self._queue.close()
                self._queue.join_thread()
            finally:
                self._process.close()
        self._closed = True
