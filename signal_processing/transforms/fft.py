"""Fast Fourier Transform (FFT) DSP primitive."""

import math

import numpy as np

from signal_processing.transforms.exceptions import TransformError
from signal_processing.transforms.results import FFTResult
from signal_processing.transforms.validation import validate_iq
from signal_processing.transforms.windows import get_window


def compute_fft(
    iq: np.ndarray,
    sample_rate_hz: float,
    *,
    nfft: int | None = None,
    window: str | None = "rectangular",
    centered: bool = True,
) -> FFTResult:
    """Compute the discrete Fourier transform of complex IQ samples.

    Processing steps:
        1. Validates IQ input.
        2. Applies the specified window function of length len(iq).
        3. Computes the standard complex DFT of length nfft (with zero-padding if nfft > len(iq)).
        4. If centered=True, shifts zero-frequency DC component to the center of the spectrum.
        5. Computes the matching frequency axis in Hz relative to tuned center frequency.

    Args:
        iq: One-dimensional complex IQ array.
        sample_rate_hz: Sampling rate in Hertz (strictly positive finite float).
        nfft: FFT transform size. If None, defaults to len(iq). If supplied,
            must be >= len(iq) (truncation is rejected).
        window: Window function name ("rectangular", "hann", "hamming", "blackman")
            or None for rectangular (default: "rectangular").
        centered: If True, shifts zero frequency to center [-Fs/2, +Fs/2) (default: True).

    Returns:
        Immutable FFTResult contract containing frequency axis, complex spectrum,
        and transform parameters.

    Raises:
        TransformError: If input is invalid, nfft < len(iq), sample_rate_hz <= 0,
            or window name is unsupported.
    """
    val_iq = validate_iq(iq, min_samples=1)
    n = len(val_iq)

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

    if nfft is None:
        actual_nfft = n
    else:
        if isinstance(nfft, bool) or not isinstance(nfft, (int, np.integer)) or nfft <= 0:
            raise TransformError(f"nfft must be an integer > 0, got {nfft!r}")
        actual_nfft = int(nfft)
        if actual_nfft < n:
            raise TransformError(
                f"nfft ({actual_nfft}) must be >= input length ({n}); truncation is not permitted"
            )

    win, win_name, win_energy = get_window(window, n)

    # Window input signal with complex128 precision
    windowed = val_iq.astype(np.complex128, copy=False) * win

    # Standard NumPy DFT with zero-padding if actual_nfft > n
    spectrum = np.fft.fft(windowed, n=actual_nfft)
    freqs = np.fft.fftfreq(actual_nfft, d=1.0 / fs)

    if centered:
        spectrum = np.fft.fftshift(spectrum)
        freqs = np.fft.fftshift(freqs)

    return FFTResult(
        frequency_offset_hz=freqs,
        spectrum=spectrum,
        sample_rate_hz=fs,
        nfft=actual_nfft,
        input_samples=n,
        window_name=win_name,
        window_energy=win_energy,
        centered=centered,
    )
