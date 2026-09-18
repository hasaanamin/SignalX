from __future__ import annotations
from typing import Any, Tuple
import numpy as np


def _as_iq_array(data: Any) -> np.ndarray:
    """
    Parses and canonicalizes arbitrary input formats into a contiguous 1D np.complex128 array.
    Supports:
      - 1D/2D complex numpy arrays
      - Real numpy arrays (promoted to complex with Q=0)
      - Dictionary format: {"I": ..., "Q": ...} or {"real": ..., "imag": ...} or {"iq": ...}
      - Tuple/List format: (I, Q) or [I, Q]
      - Planar 2xN or Nx2 arrays
    """
    if data is None:
        raise ValueError("Input IQ data cannot be None.")

    # 1. Dictionary format
    if isinstance(data, dict):
        if "iq" in data:
            return _as_iq_array(data["iq"])
        elif "I" in data and "Q" in data:
            i_arr = np.asarray(data["I"], dtype=np.float64).ravel()
            q_arr = np.asarray(data["Q"], dtype=np.float64).ravel()
            if i_arr.size != q_arr.size:
                raise ValueError(f"I and Q arrays must have identical length ({i_arr.size} vs {q_arr.size}).")
            return np.ascontiguousarray(i_arr + 1j * q_arr, dtype=np.complex128)
        elif "real" in data and "imag" in data:
            r_arr = np.asarray(data["real"], dtype=np.float64).ravel()
            im_arr = np.asarray(data["imag"], dtype=np.float64).ravel()
            return np.ascontiguousarray(r_arr + 1j * im_arr, dtype=np.complex128)
        else:
            raise ValueError(f"Dictionary must contain 'I' and 'Q' or 'real' and 'imag' or 'iq' keys. Got: {list(data.keys())}")

    # 2. Tuple/List of (I, Q)
    if isinstance(data, (tuple, list)):
        if len(data) == 2 and isinstance(data[0], (np.ndarray, list, tuple)) and isinstance(data[1], (np.ndarray, list, tuple)):
            i_arr = np.asarray(data[0], dtype=np.float64).ravel()
            q_arr = np.asarray(data[1], dtype=np.float64).ravel()
            if i_arr.size == q_arr.size and i_arr.size > 0:
                return np.ascontiguousarray(i_arr + 1j * q_arr, dtype=np.complex128)

    arr = np.asarray(data)

    # 3. 2D array (e.g. 2xN or Nx2)
    if arr.ndim == 2:
        if arr.shape[0] == 2:
            return np.ascontiguousarray(arr[0, :] + 1j * arr[1, :], dtype=np.complex128)
        elif arr.shape[1] == 2 and not np.iscomplexobj(arr):
            return np.ascontiguousarray(arr[:, 0] + 1j * arr[:, 1], dtype=np.complex128)
        else:
            arr = arr.ravel()

    # 4. 1D Array
    if np.iscomplexobj(arr):
        return np.ascontiguousarray(arr.ravel(), dtype=np.complex128)
    else:
        return np.ascontiguousarray(arr.ravel().astype(np.complex128), dtype=np.complex128)


def _extract_sample_rate(data: Any, sample_rate: float | None = None, default_fs: float | None = 1.0) -> float | None:
    """
    Extracts sample rate from metadata or kwargs.
    """
    if sample_rate is not None and np.isfinite(sample_rate) and sample_rate > 0:
        return float(sample_rate)

    if isinstance(data, dict):
        for key in ("sample_rate", "fs", "Fs", "sampling_rate", "rate"):
            if key in data and data[key] is not None:
                val = float(data[key])
                if np.isfinite(val) and val > 0:
                    return val

    return default_fs


def _compute_psd(
    z: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes averaged Welch Power Spectral Density (PSD) for complex baseband IQ data.
    Returns (freqs, psd).
    """
    n = len(z)
    if nperseg is None:
        nperseg = min(n, 1024)
        # Round down to power of 2
        nperseg = max(16, 1 << (nperseg.bit_length() - 1))

    if noverlap is None:
        noverlap = nperseg // 2

    # Window creation
    if window == "hann":
        win = np.hanning(nperseg)
    elif window == "hamming":
        win = np.hamming(nperseg)
    elif window == "blackman":
        win = np.blackman(nperseg)
    else:
        win = np.ones(nperseg)

    win_power = np.sum(win ** 2)
    step = nperseg - noverlap

    segments = []
    for start in range(0, n - nperseg + 1, step):
        seg = z[start : start + nperseg] * win
        fft_seg = np.fft.fftshift(np.fft.fft(seg))
        segments.append(np.abs(fft_seg) ** 2 / (fs * win_power))

    if not segments:
        seg = z[:nperseg] * win[:len(z[:nperseg])]
        fft_seg = np.fft.fftshift(np.fft.fft(seg, n=nperseg))
        segments.append(np.abs(fft_seg) ** 2 / (fs * win_power))

    psd = np.mean(segments, axis=0)
    freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1.0 / fs))
    return freqs, psd


def _estimate_fundamental_carrier(
    z: np.ndarray,
    fs: float = 1.0,
    guard_bins: int = 5
) -> Tuple[float, float, np.ndarray, float]:
    """
    Identifies fundamental signal frequency peak, integrates fundamental power and noise power,
    and returns (f_peak, p_signal, noise_residual, p_total).
    """
    n = len(z)
    total_power = float(np.mean(np.abs(z) ** 2))
    if n < 4:
        return 0.0, total_power, np.zeros_like(z), total_power

    freqs, psd = _compute_psd(z, fs=fs)
    df = fs / len(psd)

    peak_idx = int(np.argmax(psd))
    
    # Parabolic interpolation on log-PSD for exact sub-bin peak frequency
    if 0 < peak_idx < len(psd) - 1:
        y0 = float(np.log(max(1e-24, psd[peak_idx - 1])))
        y1 = float(np.log(max(1e-24, psd[peak_idx])))
        y2 = float(np.log(max(1e-24, psd[peak_idx + 1])))
        delta = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
        delta = np.clip(delta, -0.5, 0.5)
        f_peak = float(freqs[peak_idx] + delta * df)
    else:
        f_peak = float(freqs[peak_idx])

    # Integrate mainlobe fundamental peak (peak +- guard_bins)
    k_min = max(0, peak_idx - guard_bins)
    k_max = min(len(psd), peak_idx + guard_bins + 1)
    p_signal = float(np.sum(psd[k_min:k_max]) * df)
    p_signal = min(total_power, max(0.0, p_signal))

    # Reconstruct fundamental component with exact estimated carrier
    t = np.arange(n) / fs
    carrier = np.exp(1j * 2 * np.pi * f_peak * t)
    coeff = np.dot(z, np.conj(carrier)) / n
    signal_recon = coeff * carrier
    noise_residual = z - signal_recon

    return f_peak, p_signal, noise_residual, total_power
