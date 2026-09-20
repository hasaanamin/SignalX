"""Phase calculation DSP primitive."""

import numpy as np

from signal_processing.core import PhaseResult
from signal_processing.transforms.validation import validate_iq


def compute_phase(
    iq: np.ndarray,
    *,
    amplitude_threshold: float = 0.0,
) -> PhaseResult:
    """Compute instantaneous wrapped and continuous unwrapped phase of complex IQ samples.

    Mathematical definitions:
        wrapped_rad[n] = angle(iq[n]) in [-pi, +pi]
        unwrapped_rad[n] = unwrap(wrapped_rad[n])

    For inputs containing zero or sub-threshold amplitude samples, phase is undefined.
    The returned PhaseResult includes a boolean `valid_mask` array. Contiguous valid runs
    are unwrapped independently without artificial jump bridging across invalid gaps.

    Args:
        iq: One-dimensional complex IQ array.
        amplitude_threshold: Non-negative amplitude threshold below which phase is marked
            invalid (default: 0.0).

    Returns:
        Immutable PhaseResult contract containing wrapped_rad, unwrapped_rad, and valid_mask.
        All arrays are 1-D, real-valued/boolean, finite, and read-only.

    Raises:
        TransformError: If IQ input is invalid, non-complex, empty, or non-finite.
    """
    val_iq = validate_iq(iq, min_samples=1)
    # Promote to complex128 before computing angle/unwrap to avoid long-capture precision drift
    z = val_iq.astype(np.complex128, copy=False)

    amp = np.abs(z)
    thresh = max(0.0, float(amplitude_threshold))
    valid_mask = amp > thresh

    wrapped = np.angle(z)

    n = len(z)
    if np.all(valid_mask):
        unwrapped = np.unwrap(wrapped)
    else:
        unwrapped = wrapped.copy()
        i = 0
        while i < n:
            if not valid_mask[i]:
                i += 1
                continue
            start = i
            while i < n and valid_mask[i]:
                i += 1
            end = i
            if end - start > 1:
                unwrapped[start:end] = np.unwrap(wrapped[start:end])

    return PhaseResult(wrapped_rad=wrapped, unwrapped_rad=unwrapped, valid_mask=valid_mask)
