"""Event/stream-independent physical power reductions (no running accumulator)."""
import numpy as np


def window_mean_power(iq128: np.ndarray) -> float:
    """Reduce exactly this window, never subtract powers of preceding samples.

    Callers promote stored complex64 IQ to complex128 before this operation.
    Cost is O(N) per window and O(1) additional storage; no stream-length drift
    or cancellation from a large sample outside the window is possible.
    """
    # Same component-square reduction order as the vectorized row primitive.
    # This also avoids a one-ulp threshold discrepancy solely from choosing
    # scalar versus block-oriented execution at a transport boundary.
    return float(np.mean(iq128.real*iq128.real + iq128.imag*iq128.imag))


def window_mean_powers(iq128: np.ndarray, size: int, hop: int) -> np.ndarray:
    """Direct per-window reductions in one NumPy call, never prefix differences.

    The strided window matrix is a view. Only O(input samples + output windows)
    storage is allocated; rows contain nonnegative float64 component squares.
    A preceding large transient cannot cancel a later quiet window's energy.
    """
    if len(iq128) < size:
        return np.empty(0, dtype=np.float64)
    powers = iq128.real*iq128.real + iq128.imag*iq128.imag
    # This private primitive receives a validated 1-D vector and positive N/H.
    # The newly allocated component-square vector is contiguous. Construct the
    # bounded row view directly, avoiding general sliding-window axis dispatch.
    # ndarray verifies the complete view fits the buffer; no as_strided overrun.
    count = 1+(len(powers)-size)//hop
    windows = np.ndarray((count,size), dtype=powers.dtype, buffer=powers,
                         strides=(hop*powers.itemsize,powers.itemsize))
    windows.flags.writeable = False
    return windows.mean(axis=1)
