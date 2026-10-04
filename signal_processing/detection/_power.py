"""Event/stream-independent physical power reductions (no running accumulator)."""
import numpy as np


def window_mean_power(iq128: np.ndarray) -> float:
    """Reduce exactly this window, never subtract powers of preceding samples.

    Callers promote stored complex64 IQ to complex128 before this operation.
    Cost is O(N) per window and O(1) additional storage; no stream-length drift
    or cancellation from a large sample outside the window is possible.
    """
    return float(np.vdot(iq128, iq128).real / len(iq128))
