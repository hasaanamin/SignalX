"""Autocorrelation calculation DSP primitive."""

import numpy as np

from signal_processing.core import AutocorrelationResult
from signal_processing.transforms.exceptions import TransformError
from signal_processing.transforms.validation import validate_iq


def compute_autocorrelation(
    iq: np.ndarray,
    *,
    max_lag_samples: int | None = None,
    normalized: bool = True,
) -> AutocorrelationResult:
    """Compute the discrete complex autocorrelation for non-negative lags.

    Mathematical definition:
        For lag k >= 0:
            Rxx[k] = sum_{n=0}^{N - 1 - k} x[n+k] * conj(x[n])

        Convention:
            This implementation computes and returns NON-NEGATIVE lags only:
            k = 0, 1, 2, ..., max_lag_samples.

    Normalization:
        When normalized=True:
            Rxx_norm[k] = Rxx[k] / Rxx[0]
        such that Rxx_norm[0] == 1.0 + 0.0j for non-zero energy signals.
        If normalized=True and the signal has zero energy (Rxx[0] == 0),
        a TransformError is raised.

    Performance:
        For computational efficiency on long sequences, this function computes
        the linear cross-correlation via zero-padded Fast Fourier Transform (FFT)
        in O(N log N) time, avoiding O(N^2) sample-by-sample loops.

    Args:
        iq: One-dimensional complex IQ array.
        max_lag_samples: Maximum non-negative lag sample index to return (0 <= lag < N).
            If None, defaults to N - 1 (full sequence autocorrelation).
        normalized: Whether to normalize output such that lag 0 equals 1.0 + 0j
            (default: True).

    Returns:
        Immutable AutocorrelationResult contract containing integer lag_samples
        and corresponding complex values.

    Raises:
        TransformError: If IQ input is invalid, max_lag_samples is out of bounds or
            not an integer, or if normalization of a zero-energy signal is requested.
    """
    val_iq = validate_iq(iq, min_samples=1)
    n = len(val_iq)

    if not isinstance(normalized, bool):
        raise TransformError(f"normalized must be a boolean, got {type(normalized).__name__}")

    if max_lag_samples is None:
        max_lag = n - 1
    else:
        if isinstance(max_lag_samples, bool) or not isinstance(
            max_lag_samples, (int, np.integer)
        ):
            raise TransformError(
                f"max_lag_samples must be an integer, got {max_lag_samples!r}"
            )
        max_lag = int(max_lag_samples)
        if max_lag < 0 or max_lag >= n:
            raise TransformError(
                f"max_lag_samples must satisfy 0 <= max_lag < {n}, got {max_lag}"
            )

    # Compute total signal energy: Rxx[0] = sum(|x[n]|^2) in float64
    energy = float(
        np.sum(
            val_iq.real.astype(np.float64, copy=False) ** 2
            + val_iq.imag.astype(np.float64, copy=False) ** 2
        )
    )

    if normalized and energy <= 0.0:
        raise TransformError("Cannot normalize autocorrelation of a zero-energy signal")

    # FFT-based correlation requires zero-padding to at least 2*N - 1 to prevent circular wrap
    min_pad = 2 * n - 1
    fft_len = 1 << (min_pad.bit_length()) if min_pad > 1 else 2

    x_fft = np.fft.fft(val_iq.astype(np.complex128, copy=False), n=fft_len)
    power_spec = x_fft * np.conj(x_fft)
    r_full = np.fft.ifft(power_spec)

    # Extract non-negative lags 0 .. max_lag
    r = r_full[: max_lag + 1].copy()

    # Ensure lag 0 imaginary component is strictly zero before normalization
    r[0] = complex(energy, 0.0)

    if normalized:
        r = r / energy
        # Ensure exact 1.0 + 0.0j at lag 0
        r[0] = 1.0 + 0.0j

    lags = np.arange(max_lag + 1, dtype=np.int64)

    return AutocorrelationResult(lag_samples=lags, values=r)
