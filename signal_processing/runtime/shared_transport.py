"""Finite single-producer/single-consumer IQ transport for spawn processes.

Each slot holds <= max_samples complex64 samples plus position/loss/arrival
metadata. Semaphores transfer ownership: free -> producer -> ready -> consumer
-> free. Consumer copies bytes before releasing the slot. No reader retains a
view into recycled memory; no background pickle feeder or unbounded byte queue.

This is an internal transport, not an IQ validation boundary. The receiver
reconstructs a validated immutable IQBlock. Metadata uses signed int64 indexes.
"""
import queue
import sys
import time

import numpy as np


class SharedIQTransport:
    def __init__(self, context, capacity, max_samples, *, measure_timing=False):
        if type(measure_timing) is not bool: raise ValueError('measure_timing must be bool')
        if type(capacity) is not int or capacity < 1 or type(max_samples) is not int or max_samples < 1:
            raise ValueError('Transport capacity and max_samples must be positive integers')
        if sys.maxsize<=2**32:
            raise ValueError('SharedIQTransport requires a 64-bit process')
        self.capacity, self.max_samples = capacity, max_samples
        self._bytes = context.RawArray('B', capacity*max_samples*8)
        self._metadata = context.RawArray('q', capacity*4)
        self._arrivals = context.RawArray('d', capacity)
        self._publication_times=context.RawArray('d',capacity) if measure_timing else None
        self.last_publication_monotonic=None
        self._free = context.Semaphore(capacity)
        self._ready = context.Semaphore(0)
        # Each aligned int64 counter has ONE writer: publication by producer,
        # receipt and acknowledgement by consumer. No read-modify-write crosses
        # processes. Live snapshots are diagnostics only (may be stale); the
        # semaphores above, never counters/qsize(), arbitrate slot ownership.
        # After explicit EOF/join all counters are quiescent and exact.
        self._counts = context.RawArray('q', [0,0,0])
        self._read = self._write = 0
        self._acknowledged = 0

    def put_nowait(self, record):
        self.put(record, block=False)

    def put(self, record, block=True, timeout=None):
        if record is None:
            arrival, seq, start, data, lost = 0.,0,0,b'',0
            n = -1
        else:
            arrival, seq, start, data, lost = record
            if not isinstance(data, bytes) or len(data)%8 or len(data)>self.max_samples*8:
                raise ValueError('Invalid bounded complex64 byte payload')
            if any(type(v) is not int or not 0 <= v < 2**63 for v in (seq,start,lost)):
                raise ValueError('Transport coordinates must be nonnegative int64')
            n = len(data)//8
        if not self._free.acquire(block, timeout):
            raise queue.Full
        try:
            slot = self._write % self.capacity
            offset = slot*self.max_samples*8
            np.frombuffer(self._bytes,dtype=np.uint8)[offset:offset+len(data)] = np.frombuffer(data,dtype=np.uint8)
            self._metadata[slot*4:slot*4+4] = (n,seq,start,lost)
            self._arrivals[slot] = arrival
            if self._publication_times is not None:
                self._publication_times[slot]=time.perf_counter()
            self._write += 1
            self._counts[0] = self._write
        except BaseException:
            self._free.release()
            raise
        self._ready.release()

    def get(self, block=True, timeout=None):
        if not self._ready.acquire(block, timeout):
            raise queue.Empty
        slot = self._read % self.capacity
        n,seq,start,lost = self._metadata[slot*4:slot*4+4]
        offset = slot*self.max_samples*8
        data = np.frombuffer(self._bytes,dtype=np.uint8)[offset:offset+max(0,n)*8].tobytes()
        arrival = self._arrivals[slot]
        if self._publication_times is not None:
            self.last_publication_monotonic=self._publication_times[slot]
        self._read += 1
        self._counts[1] = self._read
        self._free.release()
        return None if n == -1 else (arrival,seq,start,data,lost)

    def task_done(self):
        if self._acknowledged >= self._read:
            raise ValueError('task_done called too many times')
        self._acknowledged += 1
        self._counts[2] = self._acknowledged

    @property
    def unfinished_tasks(self):
        return max(0,self._counts[0]-self._counts[2])

    def qsize(self):
        return min(self.capacity,max(0,self._counts[0]-self._counts[1]))

    def empty(self):
        return self.qsize() == 0

    def cancel_join_thread(self):
        pass  # There is deliberately no feeder thread.

    def join_thread(self):
        pass

    def close(self):
        # Call only after the peer has exited. Shared ctypes allocations release
        # their OS mappings when the final owning process drops its references.
        self._bytes = self._metadata = self._arrivals = None
        self._publication_times=None
