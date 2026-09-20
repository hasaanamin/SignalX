"""Envelope calculation DSP primitive."""

import numpy as np

from signal_processing.transforms.validation import validate_iq


def compute_envelope(iq: np.ndarray) -> np.ndarray:
    """Compute the instantaneous amplitude envelope of complex IQ samples.

    Mathematical definition:
        envelope[n] = |iq[n]| = sqrt(I[n]^2 + Q[n]^2)

    Args:
        iq: One-dimensional complex IQ array.

    Returns:
        One-dimensional real-valued, non-negative, finite numpy array representing
        the signal envelope. The returned array is read-only and does not alias input.

    Raises:
        TransformError: If IQ input is invalid, non-complex, empty, or non-finite.
    """
    val_iq = validate_iq(iq, min_samples=1)
    env = np.abs(val_iq.astype(np.complex128, copy=False))
    c_env = np.ascontiguousarray(env)
    return np.frombuffer(c_env.tobytes(), dtype=c_env.dtype).reshape(c_env.shape)
