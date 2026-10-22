"""Exact double-precision FFTW row execution on the Windows wheel's C ABI.

One receiver owns the plans and all aligned scratch. A complete overlap-save
loop runs without the GIL; only one filter's inverse scratch is active at once.
Other platforms or FFTW distributions retain the portable pyFFTW bank.
"""
from functools import lru_cache
from pathlib import Path
import ctypes
import sys
import weakref

import numba
import numpy as np
import pyfftw
from scipy.fft import fft


@numba.njit(nogil=True,inline='always')
def _write_row(values,powers,branch,position,count,capacity,source,mirror):
    mirrored = min(count,max(0,mirror-position))
    for j in range(mirrored):
        value = values[source+j]
        power = value.real*value.real+value.imag*value.imag
        powers[branch,position+j] = power
        powers[branch,position+j+capacity] = power
    for j in range(mirrored,count):
        value = values[source+j]
        powers[branch,position+j] = value.real*value.real+value.imag*value.imag


# Native function/plan addresses are runtime arguments, never cached machine
# addresses from another process. Compile/warm before the acquisition clock.
@numba.njit(nogil=True,cache=False)
def _execute(samples,start,powers,capacity,mirror,x,spectrum,product,correlations,
             kernels,forward,inverse,run):
    size = len(x)
    count = len(samples)-127
    for offset in range(0,count,size-127):
        n = min(size-127,count-offset)
        for j in range(size):
            x[j] = samples[offset+j] if j < n+127 else 0j
        run(forward)
        position = (start+offset) % capacity
        first = min(n,capacity-position)
        for branch in range(len(kernels)):
            for j in range(size):
                product[j] = kernels[branch,j]*spectrum[j]
            run(inverse)
            _write_row(correlations,powers,branch,position,first,capacity,127,mirror)
            if first < n:
                _write_row(correlations,powers,branch,0,n-first,capacity,127+first,mirror)


@lru_cache(maxsize=1)
def _library():
    if sys.platform != 'win32' or ctypes.sizeof(ctypes.c_void_p) != 8:
        return None
    path = Path(pyfftw.__file__).parent/'libfftw3-3.dll'
    if not path.is_file():
        return None
    try:
        library = ctypes.CDLL(str(path))
    except OSError:
        return None
    # uintptr_t and a plan pointer have the same Windows x64 ABI. The integer
    # declaration lets Numba pass opaque handles without dereferencing them.
    library.fftw_execute.argtypes = [ctypes.c_size_t]
    library.fftw_execute.restype = None
    library.fftw_destroy_plan.argtypes = [ctypes.c_size_t]
    library.fftw_destroy_plan.restype = None
    library.fftw_plan_dft_1d.argtypes = [ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,
                                       ctypes.c_int,ctypes.c_uint]
    library.fftw_plan_dft_1d.restype = ctypes.c_void_p
    if hasattr(library,'fftw_plan_with_nthreads'):
        library.fftw_plan_with_nthreads.argtypes = [ctypes.c_int]
        library.fftw_plan_with_nthreads.restype = None
    return library


def _destroy(library,lock,plan):
    # FFTW execution is independent per plan; its planner/destructor is global.
    with lock:
        library.fftw_destroy_plan(plan)


class NativeFFTBank:
    dispatch = 'numba-row-loop'

    def __init__(self,kernels,library):
        from ._fft_bank import _planner_lock
        self._library = library
        size = 1024
        self.input = pyfftw.empty_aligned(size,'complex128')
        self.transformed = pyfftw.empty_aligned(size,'complex128')
        self.product = pyfftw.empty_aligned(size,'complex128')
        self.correlations = pyfftw.empty_aligned(size,'complex128')
        input_pointer,transformed_pointer = self.input.ctypes.data,self.transformed.ctypes.data
        product_pointer,correlation_pointer = self.product.ctypes.data,self.correlations.ctypes.data
        with _planner_lock:
            if hasattr(library,'fftw_plan_with_nthreads'):
                library.fftw_plan_with_nthreads(1)
            forward = library.fftw_plan_dft_1d(size,input_pointer,transformed_pointer,-1,1)
            inverse = library.fftw_plan_dft_1d(size,product_pointer,correlation_pointer,1,1)
            if not forward or not inverse:
                for plan in (forward,inverse):
                    if plan:
                        library.fftw_destroy_plan(plan)
                raise RuntimeError('FFTW native plan creation failed')
        self._finalizers = [weakref.finalize(self,_destroy,library,_planner_lock,plan)
                            for plan in (forward,inverse)]
        self._forward,self._inverse = np.uintp(forward),np.uintp(inverse)
        self.kernels = fft(kernels,n=size,axis=-1,workers=1)/size
        self.write_powers(np.frombuffer(bytes((size+127)*16),np.complex128),0,
                          np.empty((len(kernels),2*size)),size,0)

    def write_powers(self,samples,start,powers,capacity,mirror_limit):
        _execute(samples,start,powers,capacity,mirror_limit,self.input,self.transformed,
                 self.product,self.correlations,self.kernels,self._forward,self._inverse,
                 self._library.fftw_execute)


def optional_native_bank(kernels):
    library = _library()
    return None if library is None else NativeFFTBank(kernels,library)
