"""Optional double precision FFTW plans owned by one synchronizer.

Plans and scratch are created before acquisition. Received data and power
history are never shared between receivers. FFTW planning uses a process lock
because its planner is global; execution uses the receiver's lifecycle gate.
"""
import threading

import numpy as np

# A garbage-collected receiver can finalize its plans while this same thread
# creates ctypes/NumPy arguments for a new plan. Reentry precedes the C planner
# call; other threads must still serialize all planning/destruction.
_planner_lock = threading.RLock()


class FFTBank:
    def __init__(self, kernels, *, size=512):
        import pyfftw

        self.stride = size-127
        self.input = pyfftw.empty_aligned(size, "complex128")
        self.transformed = pyfftw.empty_aligned(self.input.shape, "complex128")
        self.product = pyfftw.empty_aligned((len(kernels),size), "complex128")
        self.correlations = pyfftw.empty_aligned(self.product.shape, "complex128")
        with _planner_lock:
            self.forward = pyfftw.FFTW(self.input, self.transformed, axes=(-1,),
                                      flags=("FFTW_MEASURE","FFTW_DESTROY_INPUT"), threads=1)
            self.inverse = pyfftw.FFTW(self.product, self.correlations, axes=(-1,),
                                      direction="FFTW_BACKWARD", flags=("FFTW_MEASURE","FFTW_DESTROY_INPUT"),
                                      threads=1)
        from scipy.fft import fft
        self.kernels = fft(kernels,n=size,axis=-1,workers=1)/size
        self.input.fill(0)
        self.product.fill(0)
        self.inverse.execute()

    def write_powers(self,samples,start,powers,capacity,mirror_limit):
        from ._native import _correlation_powers
        count = len(samples)-127
        for offset in range(0,count,self.stride):
            n = min(self.stride,count-offset)
            self.input.fill(0)
            region = samples[offset:offset+n+127]
            self.input[:len(region)] = region
            self.forward.execute()
            np.multiply(self.kernels,self.transformed,out=self.product)
            self.inverse.execute()
            position = (start+offset) % capacity
            first = min(n,capacity-position)
            _correlation_powers(self.correlations,powers,position,first,capacity,127,mirror_limit)
            if first < n:
                _correlation_powers(self.correlations,powers,0,n-first,capacity,127+first,mirror_limit)


def optional_bank(kernels):
    try:
        import pyfftw  # noqa: F401
    except ImportError:
        return None
    from ._native import numba
    if numba is not None:
        from ._fftw_native import optional_native_bank
        native = optional_native_bank(kernels)
        if native is not None:
            return native
    return FFTBank(kernels)
