"""Causal rational channel preparation with exact original-sample mapping."""
from fractions import Fraction
import math

import numpy as np
from scipy.signal import firwin, upfirdn


class LTEChannel:
    """Internal single-owner preparation; receiver validates positioned input.

    At 2 MS/s use 24/25 conversion. History starts on 25-input-sample boundaries,
    so each local upfirdn call has the same global polyphase alignment. Only causal
    outputs before the input horizon are released; no future zero tail is emitted.
    """

    def __init__(self, input_rate):
        if input_rate not in (1_920_000., 2_000_000.):
            raise ValueError("Unsupported input rate")
        self.up, self.down = (24, 25) if input_rate == 2_000_000 else (1, 1)
        self.taps = (firwin(1201, 700_000., fs=48_000_000., window=("kaiser", 8.6))*24
                     if self.up == 24 else np.ones(1))
        self.delay_ticks = (len(self.taps)-1)//2
        self.settling_samples = math.ceil((len(self.taps)-1)/self.down)
        self.reset()

    def reset(self):
        self._tail = np.empty(0, np.complex128)
        self._base = self.input_count = self.output_count = 0

    def input_position(self, output_index):
        return Fraction(int(output_index)*self.down-self.delay_ticks, self.up)

    @property
    def retained_samples(self):
        return len(self._tail)

    def process(self, iq):
        if self.up == 1:
            result = np.asarray(iq, dtype=np.complex128)
            start = self.output_count
            self.input_count += len(iq)
            self.output_count += len(iq)
            return start, result
        combined = np.concatenate((self._tail, np.asarray(iq, np.complex128)))
        end = self.input_count+len(iq)
        stop = (end*self.up+self.down-1)//self.down
        local_origin = self._base*self.up//self.down
        filtered = upfirdn(self.taps, combined, up=self.up, down=self.down)
        start = self.output_count
        result = filtered[start-local_origin:stop-local_origin].copy()
        keep_from = max(0, ((end-math.ceil(len(self.taps)/self.up)-25)//25)*25)
        self._tail = combined[keep_from-self._base:].copy()
        self._base = keep_from
        self.input_count = end
        self.output_count = stop
        return start, result

    def _process_received(self, iq):
        """Borrow admitted native-rate IQ until the receiver copies its history.

        Native preparation is the identity. History assignment already converts
        complex64 to complex128, so no separate transport-sized double array is
        needed. The public process() keeps its existing double-precision interface.
        """
        if self.up != 1:
            return self.process(iq)
        start = self.output_count
        self.input_count += len(iq)
        self.output_count += len(iq)
        return start, iq
