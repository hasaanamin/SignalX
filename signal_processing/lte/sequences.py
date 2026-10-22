"""LTE synchronization equations, TS 36.211 v16.6.0 sections 6.11.1/6.11.2.

Standalone standards-based implementation. Outputs are frequency ordered from
subcarrier -31 through -1, then +1 through +31; DC is not a sequence element.
"""
import numpy as np
from functools import lru_cache


SYNC_BINS = np.r_[np.arange(-31, 0), np.arange(1, 32)] % 128
SYNC_BINS.setflags(write=False)


def pss_sequence(n_id_2):
    if type(n_id_2) is not int or not 0 <= n_id_2 <= 2:
        raise ValueError("PSS identity must be 0, 1 or 2")
    root = (25, 29, 34)[n_id_2]
    # Omit the middle element of the length-63 Zadoff-Chu sequence.
    k = np.r_[np.arange(31), np.arange(32, 63)]
    return np.exp(-1j*np.pi*root*k*(k+1)/63)


def _m_sequence(taps):
    bits = np.zeros(31, dtype=np.int8)
    bits[4] = 1
    for k in range(26):
        bits[k+5] = sum(int(bits[k+t]) for t in taps) % 2
    return 1-2*bits


_S, _C, _Z = (_m_sequence(taps) for taps in ((0, 2), (0, 3), (0, 1, 2, 4)))
for _sequence in (_S, _C, _Z):
    _sequence.setflags(write=False)


def sss_sequence(pci, half_frame):
    if type(pci) is not int or not 0 <= pci < 504:
        raise ValueError("PCI must be an integer in [0, 503]")
    if type(half_frame) is not int or half_frame not in (0, 1):
        raise ValueError("half_frame must be 0 (subframe 0) or 1 (subframe 5)")
    group, sector = divmod(pci, 3)
    q0 = group // 30
    q = (group+q0*(q0+1)//2)//30
    m = group+q*(q+1)//2
    a = m % 31
    b = (a+m//31+1) % 31
    s, c, z = _S, _C, _Z
    n = np.arange(31)
    first, second = (a, b) if half_frame == 0 else (b, a)
    result = np.empty(62, dtype=np.float64)
    result[0::2] = s[(n+first) % 31]*c[(n+sector) % 31]
    result[1::2] = s[(n+second) % 31]*c[(n+sector+3) % 31]*z[(n+first % 8) % 31]
    return result


@lru_cache(maxsize=3)
def sss_hypotheses(nid):
    """Fixed standards/configuration data only; never caches received samples."""
    if type(nid) is not int or not 0 <= nid < 3:
        raise ValueError("Invalid sector")
    bank = np.array([sss_sequence(3*g+nid, half) for g in range(168) for half in (0, 1)])
    return np.frombuffer(bank.tobytes(), dtype=bank.dtype).reshape(bank.shape)


def sync_symbol(sequence):
    sequence = np.asarray(sequence)
    if sequence.shape != (62,) or not np.all(np.isfinite(sequence)):
        raise ValueError("Expected 62 finite sync subcarriers")
    grid = np.zeros(128, np.complex128)
    grid[SYNC_BINS] = sequence
    return np.fft.ifft(grid)*np.sqrt(128)


def normal_cp_lengths():
    """One slot at 1.92 MS/s: seven symbols, 960 samples including CP."""
    return (10, 9, 9, 9, 9, 9, 9)
