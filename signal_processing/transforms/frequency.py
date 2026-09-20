"""Instantaneous frequency calculation DSP primitive."""

import math

import numpy as np

from signal_processing.core import PhaseResult
from signal_processing.transforms.exceptions import TransformError


def compute_instantaneous_frequency(
    phase: PhaseResult,
    sample_rate_hz: float,
    *,
    allow_gaps: bool = False,
) -> np.ndarray:
    """Compute instantaneous frequency offset from an unwrapped phase sequence.

    Mathematical definition:
        f_inst[n] = (Fs / (2 * pi)) * (phi[n+1] - phi[n])

    For an input phase sequence of length N, the resulting instantaneous
    frequency array has length N - 1 aligned with sample intervals [0..N-2].

    Args:
        phase: Input PhaseResult contract containing unwrapped phase in radians.
        sample_rate_hz: Sampling rate in Hertz (strictly positive finite float).
        allow_gaps: If True, returns an aligned array where invalid sample intervals
            are marked as NaN. If False (default), raises TransformError if phase
            contains undefined/zero-amplitude samples.

    Returns:
        One-dimensional real-valued numpy array representing the instantaneous
        baseband frequency offset in Hz. The returned array is immutable.

    Raises:
        TransformError: If phase is not a PhaseResult, has fewer than 2 samples,
            sample_rate_hz is non-positive or non-finite, or if gaps exist and
            allow_gaps=False.
    """
    if not isinstance(phase, PhaseResult):
        raise TransformError(
            f"Expected PhaseResult instance for phase, got {type(phase).__name__}"
        )

    if isinstance(sample_rate_hz, bool) or not isinstance(
        sample_rate_hz, (int, float, np.floating, np.integer)
    ):
        raise TransformError(
            f"sample_rate_hz must be a numeric float, got {sample_rate_hz!r}"
        )
    fs = float(sample_rate_hz)
    if not math.isfinite(fs) or fs <= 0.0:
        raise TransformError(
            f"sample_rate_hz must be finite and strictly positive, got {sample_rate_hz}"
        )

    unwrapped = phase.unwrapped_rad.astype(np.float64, copy=False)
    if len(unwrapped) < 2:
        raise TransformError(
            f"Instantaneous frequency requires at least 2 phase samples, got {len(unwrapped)}"
        )

    # Check validity mask for gaps
    if phase.valid_mask is not None and not np.all(phase.valid_mask):
        if not allow_gaps:
            raise TransformError(
                "PhaseResult contains invalid/zero-amplitude samples; "
                "instantaneous frequency is undefined across gaps (use allow_gaps=True for aligned NaN array)"
            )
        pair_valid = phase.valid_mask[:-1] & phase.valid_mask[1:]
        f_inst = np.full(len(unwrapped) - 1, np.nan, dtype=np.float64)
        if np.any(pair_valid):
            dphi = unwrapped[1:][pair_valid] - unwrapped[:-1][pair_valid]
            f_inst[pair_valid] = (fs / (2.0 * math.pi)) * dphi
        c_out = np.ascontiguousarray(f_inst)
        return np.frombuffer(c_out.tobytes(), dtype=c_out.dtype).reshape(c_out.shape)

    # Fully valid input path
    phase_diff = np.diff(unwrapped)
    f_inst = (fs / (2.0 * math.pi)) * phase_diff
    c_out = np.ascontiguousarray(f_inst)
    return np.frombuffer(c_out.tobytes(), dtype=c_out.dtype).reshape(c_out.shape)
