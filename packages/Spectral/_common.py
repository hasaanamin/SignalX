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
        if "complex_samples" in data and data["complex_samples"] is not None:
            return _as_iq_array(data["complex_samples"])
        elif "iq" in data:
            return _as_iq_array(data["iq"])
        elif "i" in data and "q" in data:
            i_arr = np.asarray(data["i"], dtype=np.float64).ravel()
            q_arr = np.asarray(data["q"], dtype=np.float64).ravel()
            if i_arr.size != q_arr.size:
                raise ValueError(f"i and q arrays must have identical length ({i_arr.size} vs {q_arr.size}).")
            return np.ascontiguousarray(i_arr + 1j * q_arr, dtype=np.complex128)
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
            raise ValueError(f"Dictionary must contain 'complex_samples', 'i'/'q', 'I'/'Q', 'real'/'imag', or 'iq' keys. Got: {list(data.keys())}")

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


def _create_window(name: str, length: int) -> np.ndarray:
    """
    Generates window coefficients of specified length.
    Supports: 'hann'/'hanning', 'hamming', 'blackman', 'flattop', 'kaiser', 'rect'/'boxcar'.
    """
    w_name = name.lower().strip()
    if w_name in ("hann", "hanning"):
        return np.hanning(length)
    elif w_name == "hamming":
        return np.hamming(length)
    elif w_name == "blackman":
        return np.blackman(length)
    elif w_name in ("flattop", "flat_top"):
        # Standard symmetric flat-top window coefficients
        n = np.arange(length)
        a0, a1, a2, a3, a4 = 0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368
        return (
            a0
            - a1 * np.cos(2 * np.pi * n / (length - 1))
            + a2 * np.cos(4 * np.pi * n / (length - 1))
            - a3 * np.cos(6 * np.pi * n / (length - 1))
            + a4 * np.cos(8 * np.pi * n / (length - 1))
        )
    elif w_name == "kaiser":
        return np.kaiser(length, beta=14.0)
    elif w_name in ("rect", "rectangular", "boxcar", "none"):
        return np.ones(length)
    else:
        return np.hanning(length)


def _compute_welch_psd(
    z: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes averaged Welch Power Spectral Density (PSD) for complex baseband IQ data.
    Returns (freqs, psd) centered at DC [-fs/2, fs/2).
    """
    n = len(z)
    if nperseg is None:
        nperseg = min(n, 1024)
        nperseg = max(16, 1 << (nperseg.bit_length() - 1))

    if noverlap is None:
        noverlap = nperseg // 2

    win = _create_window(window, nperseg)
    win_power = np.sum(win ** 2)
    step = max(1, nperseg - noverlap)

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
