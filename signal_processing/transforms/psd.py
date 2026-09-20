"""Power Spectral Density (PSD) calculation DSP primitive."""

import numpy as np

from signal_processing.core import PSDResult
from signal_processing.transforms.exceptions import TransformError
from signal_processing.transforms.fft import compute_fft
from signal_processing.transforms.results import FFTResult
from signal_processing.transforms.validation import validate_iq
from signal_processing.transforms.windows import get_window


def compute_psd_from_fft(
    fft_result: FFTResult,
    *,
    floor_dbfs_per_hz: float = -300.0,
) -> PSDResult:
    """Compute normalized Power Spectral Density (PSD) from an FFTResult.

    Mathematical definition (complex baseband periodogram):
        PSD_linear[k] = |FFT[k]|^2 / (Fs * window_energy)
        PSD_dBFS_per_Hz[k] = 10 * log10(max(PSD_linear[k], 10^(floor_dbfs_per_hz / 10)))

    Window energy normalization ensures that for a rectangular window:
        sum(PSD_linear) * (Fs / nfft) == mean(|iq|^2)

    Args:
        fft_result: Existing FFTResult container.
        floor_dbfs_per_hz: Floor applied to dBFS/Hz to prevent -inf on zero/negligible power
            (default: -300.0 dBFS/Hz).

    Returns:
        Immutable PSDResult contract containing frequency coordinates, linear PSD,
        and logarithmic PSD in dBFS/Hz.

    Raises:
        TransformError: If fft_result is not an FFTResult instance or floor_dbfs_per_hz is non-finite.
    """
    if not isinstance(fft_result, FFTResult):
        raise TransformError(
            f"Expected FFTResult instance, got {type(fft_result).__name__}"
        )

    if isinstance(floor_dbfs_per_hz, bool) or not isinstance(
        floor_dbfs_per_hz, (int, float, np.floating)
    ):
        raise TransformError(
            f"floor_dbfs_per_hz must be a numeric float, got {floor_dbfs_per_hz!r}"
        )
    floor_val = float(floor_dbfs_per_hz)
    if not np.isfinite(floor_val):
        raise TransformError(
            f"floor_dbfs_per_hz must be finite, got {floor_dbfs_per_hz}"
        )

    if fft_result.window_energy <= 0.0 or not np.isfinite(fft_result.window_energy):
        raise TransformError(
            f"fft_result.window_energy must be strictly positive, got {fft_result.window_energy}"
        )

    norm_factor = fft_result.sample_rate_hz * fft_result.window_energy
    if norm_factor <= 0.0 or not np.isfinite(norm_factor):
        raise TransformError(
            f"Invalid normalization factor ({norm_factor}) from sample_rate_hz={fft_result.sample_rate_hz} "
            f"and window_energy={fft_result.window_energy}"
        )

    spec = fft_result.spectrum
    mag_sq = spec.real ** 2 + spec.imag ** 2

    p_linear = mag_sq / norm_factor
    p_linear = np.maximum(p_linear, 0.0)

    # Safe log10 with floor to avoid -inf or underflow on extreme negative floor values
    with np.errstate(divide="ignore", invalid="ignore"):
        p_dbfs_per_hz = 10.0 * np.log10(p_linear)
        p_dbfs_per_hz = np.maximum(p_dbfs_per_hz, floor_val)
        p_dbfs_per_hz = np.where(np.isnan(p_dbfs_per_hz), floor_val, p_dbfs_per_hz)


    return PSDResult(
        frequency_offset_hz=fft_result.frequency_offset_hz,
        power_linear=p_linear,
        power_dbfs_per_hz=p_dbfs_per_hz,
    )


def compute_welch_psd(
    iq: np.ndarray,
    sample_rate_hz: float,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    window: str = "hann",
    centered: bool = True,
    floor_dbfs_per_hz: float = -300.0,
) -> PSDResult:
    """Compute averaged Welch Power Spectral Density (PSD) for complex baseband IQ data.

    Mathematical definition:
        PSD[k] = (1 / K) * sum(|FFT_i[k]|^2 / (Fs * S2))
        where S2 = sum(w^2) is the window energy normalization factor.

    Supports configurable segment size (nperseg), overlap (noverlap), window,
    and two-sided centered frequency coordinates [-Fs/2, +Fs/2).

    Args:
        iq: One-dimensional complex IQ array.
        sample_rate_hz: Sampling rate in Hertz (strictly positive finite float).
        nperseg: Length of each Welch segment (positive integer <= len(iq)).
            If None, defaults to min(len(iq), 1024).
        noverlap: Number of overlapping samples between segments (0 <= noverlap < nperseg).
            If None, defaults to nperseg // 2.
        window: Window name (e.g. "hann", "hamming", "rectangular", "blackman").
        centered: If True, centers zero frequency at DC (default: True).
        floor_dbfs_per_hz: Minimum power floor in dBFS/Hz (default: -300.0).

    Returns:
        Immutable PSDResult contract containing frequency_offset_hz, power_linear,
        and power_dbfs_per_hz.

    Raises:
        TransformError: If inputs or parameters are invalid or uncomputable.
    """
    val_iq = validate_iq(iq, min_samples=1)
    n = len(val_iq)

    if isinstance(sample_rate_hz, bool) or not isinstance(
        sample_rate_hz, (int, float, np.floating, np.integer)
    ):
        raise TransformError(f"sample_rate_hz must be a numeric float, got {sample_rate_hz!r}")
    fs = float(sample_rate_hz)
    if not np.isfinite(fs) or fs <= 0.0:
        raise TransformError(f"sample_rate_hz must be finite and strictly positive, got {sample_rate_hz}")

    if isinstance(floor_dbfs_per_hz, bool) or not isinstance(
        floor_dbfs_per_hz, (int, float, np.floating)
    ):
        raise TransformError(f"floor_dbfs_per_hz must be a numeric float, got {floor_dbfs_per_hz!r}")
    floor_val = float(floor_dbfs_per_hz)
    if not np.isfinite(floor_val):
        raise TransformError(f"floor_dbfs_per_hz must be finite, got {floor_dbfs_per_hz}")

    if nperseg is None:
        seg_len = min(n, 1024)
    else:
        if isinstance(nperseg, bool) or not isinstance(nperseg, (int, np.integer)) or nperseg <= 0:
            raise TransformError(f"nperseg must be a strictly positive integer, got {nperseg!r}")
        seg_len = int(nperseg)
        if seg_len > n:
            raise TransformError(f"nperseg ({seg_len}) cannot be greater than input length ({n})")

    if noverlap is None:
        ovlp = seg_len // 2
    else:
        if isinstance(noverlap, bool) or not isinstance(noverlap, (int, np.integer)) or noverlap < 0:
            raise TransformError(f"noverlap must be a non-negative integer, got {noverlap!r}")
        ovlp = int(noverlap)
        if ovlp >= seg_len:
            raise TransformError(f"noverlap ({ovlp}) must be strictly less than nperseg ({seg_len})")

    if nfft is not None:
        if isinstance(nfft, bool) or not isinstance(nfft, (int, np.integer)) or nfft <= 0:
            raise TransformError(f"nfft must be a strictly positive integer, got {nfft!r}")
        fft_len = int(nfft)
        if fft_len < seg_len:
            raise TransformError(f"nfft ({fft_len}) cannot be less than nperseg ({seg_len})")
    else:
        fft_len = seg_len

    win, _, win_energy = get_window(window, seg_len)
    norm_factor = fs * win_energy
    if norm_factor <= 0.0 or not np.isfinite(norm_factor):
        raise TransformError(f"Invalid normalization factor ({norm_factor}) in Welch PSD")

    step = max(1, seg_len - ovlp)
    segments_psd: list[np.ndarray] = []
    for start in range(0, n - seg_len + 1, step):
        seg = val_iq[start : start + seg_len].astype(np.complex128, copy=False) * win
        spec = np.fft.fft(seg, n=fft_len)
        p_seg = (spec.real ** 2 + spec.imag ** 2) / norm_factor
        segments_psd.append(p_seg)

    if not segments_psd:
        seg = val_iq[:seg_len].astype(np.complex128, copy=False) * win[: len(val_iq[:seg_len])]
        spec = np.fft.fft(seg, n=fft_len)
        p_seg = (spec.real ** 2 + spec.imag ** 2) / norm_factor
        segments_psd.append(p_seg)

    p_linear = np.mean(segments_psd, axis=0)
    freqs = np.fft.fftfreq(fft_len, d=1.0 / fs)

    if centered:
        p_linear = np.fft.fftshift(p_linear)
        freqs = np.fft.fftshift(freqs)

    p_linear = np.maximum(p_linear, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_dbfs = 10.0 * np.log10(p_linear)
        p_dbfs = np.maximum(p_dbfs, floor_val)
        p_dbfs = np.where(np.isnan(p_dbfs), floor_val, p_dbfs)

    return PSDResult(
        frequency_offset_hz=freqs,
        power_linear=p_linear,
        power_dbfs_per_hz=p_dbfs,
    )


def compute_psd(
    iq: np.ndarray,
    sample_rate_hz: float,
    *,
    mode: str = "periodogram",
    nfft: int | None = None,
    window: str | None = "hann",
    centered: bool = True,
    floor_dbfs_per_hz: float = -300.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> PSDResult:
    """Compute normalized Power Spectral Density (PSD) from complex IQ samples.

    Supports both single modified periodogram mode ("periodogram") and Welch's
    averaged periodogram mode ("welch").

    Args:
        iq: One-dimensional complex IQ array.
        sample_rate_hz: Sampling rate in Hertz (strictly positive finite float).
        mode: Estimator mode: "periodogram" or "welch" (default: "periodogram").
        nfft: Optional FFT size for periodogram mode (>= len(iq)).
        window: Window name (default: "hann").
        centered: If True, centers zero frequency at DC (default: True).
        floor_dbfs_per_hz: Logarithmic power floor in dBFS/Hz (default: -300.0).
        nperseg: Segment length for Welch mode (default: min(len(iq), 1024)).
        noverlap: Overlapping points for Welch mode (default: nperseg // 2).

    Returns:
        Immutable PSDResult contract containing frequency_offset_hz, power_linear,
        and power_dbfs_per_hz.

    Raises:
        TransformError: If inputs or configurations are invalid.
    """
    if mode == "welch":
        return compute_welch_psd(
            iq=iq,
            sample_rate_hz=sample_rate_hz,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            window=window or "hann",
            centered=centered,
            floor_dbfs_per_hz=floor_dbfs_per_hz,
        )
    elif mode == "periodogram":
        fft_res = compute_fft(
            iq=iq,
            sample_rate_hz=sample_rate_hz,
            nfft=nfft,
            window=window,
            centered=centered,
        )
        return compute_psd_from_fft(fft_res, floor_dbfs_per_hz=floor_dbfs_per_hz)
    else:
        raise TransformError(f"Unsupported PSD mode '{mode}'. Supported modes: 'periodogram', 'welch'")
