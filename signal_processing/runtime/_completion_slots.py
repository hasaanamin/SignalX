"""Bounded SPSC completion storage, never an unbounded queue.

The caller's existing completion-slot semaphore owns each slot through ack.
A separate ready semaphore publishes its type/header. Optional finite payload
slots carry full capture records. Records exceeding that declared size retain
the pipe fallback; ready is published BEFORE pipe writing so large writes can
be drained without a full-pipe deadlock. Reads copy bytes, never lend a mutable
slot to an event/worker. Exactly one producer and one reader.
"""


class CompletionSlots:
    def __init__(self, context, capacity, header_size, payload_size=0):
        self.capacity, self.header_size = capacity, header_size
        self._headers = context.RawArray('B', capacity*header_size)
        self._kinds = context.RawArray('b', capacity)
        self.payload_size=payload_size
        self._payloads=context.RawArray('B',capacity*payload_size) if payload_size else None
        self._lengths=context.RawArray('q',capacity) if payload_size else None
        self.ready = context.Semaphore(0)

    def publish(self, index, header=None, *, payload=None):
        slot = index % self.capacity
        if payload is not None:
            if header is not None or not isinstance(payload,bytes) or not 0<len(payload)<=self.payload_size:
                raise ValueError('Invalid bounded full completion payload')
            start=slot*self.payload_size
            memoryview(self._payloads).cast('B')[start:start+len(payload)]=payload
            self._lengths[slot]=len(payload)
            self._kinds[slot]=3
        elif header is not None:
            if len(header) != self.header_size:
                raise ValueError('Invalid bounded completion header')
            start = slot*self.header_size
            view = memoryview(self._headers).cast('B')
            view[start:start+self.header_size] = header
            self._kinds[slot] = 1
        else:
            self._kinds[slot] = 2
        self.ready.release()

    def read(self, index):
        slot = index % self.capacity
        kind = self._kinds[slot]
        if kind == 1:
            start = slot*self.header_size
            return bytes(memoryview(self._headers).cast('B')[start:start+self.header_size])
        if kind == 2:
            return None
        if kind == 3:
            start=slot*self.payload_size
            n=self._lengths[slot]
            if not 0<n<=self.payload_size: raise ValueError('Invalid shared completion length')
            return bytes(memoryview(self._payloads).cast('B')[start:start+n])
        raise ValueError('Invalid completion-slot publication')
