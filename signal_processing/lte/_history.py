"""Bounded single-owner mirrored IQ history; no stream-length allocations.

Views are borrowed until the next append/reset and never cross a public output
or asynchronous boundary. Mirroring writes each NEW sample twice, instead of
moving the entire retained history at every search horizon.
"""
import numpy as np


class IQHistory:
    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('History capacity must be a positive integer')
        self.capacity = capacity
        self._storage = np.empty(2*capacity, np.complex128)
        self.reset()

    def reset(self):
        self.start = self.end = 0

    def append(self, samples):
        n = len(samples)
        if n > self.capacity-(self.end-self.start):
            raise ValueError('LTE history capacity exceeded before discard')
        position = self.end % self.capacity
        first = min(n, self.capacity-position)
        for offset in (0, self.capacity):
            self._storage[position+offset:position+offset+first] = samples[:first]
            if n > first:
                self._storage[offset:offset+n-first] = samples[first:]
        self.end += n

    def discard_before(self, start):
        if not self.start <= start <= self.end:
            raise ValueError('History discard must remain within received extent')
        self.start = start

    def view(self):
        offset = self.start % self.capacity
        result = self._storage[offset:offset+self.end-self.start]
        result.flags.writeable = False
        return result
