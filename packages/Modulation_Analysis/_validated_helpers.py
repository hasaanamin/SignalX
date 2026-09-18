from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import uniform_filter1d
from scipy.stats import trim_mean

from ._common import (
    _as_iq_array,
    _extract_sample_rate,
    _validate_sample_rate,
    _safe_divide,
    _safe_log,
    _safe_percentile,
    _robust_median_mad,
    _robust_scale,
    _complex_power,
    _amplitude_valid_mask,
    _circular_difference,
    _circular_mean,
    _safe_rms_normalize,
)


# ==============================================================================
# 1. FROZEN TIME-FREQUENCY HELPERS (PORTED FROM Time Frequency PACKAGE)
# ==============================================================================

def _get_tf_window(window_spec: str | np.ndarray, length: int) -> Tuple[np.ndarray, str]:
    """Generates window array of specified length matching Time Frequency package."""
    if length <= 0:
        raise ValueError(f"Window length must be > 0, got {length}")

    if isinstance(window_spec, np.ndarray):
        w = np.asarray(window_spec, dtype=np.float64).ravel()
        if len(w) != length:
            raise ValueError(f"Window array length ({len(w)}) != expected ({length}).")
        return w, "custom"

    name_clean = str(window_spec).lower().strip().replace("-", "_")
    if name_clean in ("hann", "hanning"):
        return np.hanning(length), "hann"
    elif name_clean in ("hamming", "hamm"):
        return np.hamming(length), "hamming"
    elif name_clean in ("blackman", "black"):
        return np.blackman(length), "blackman"
    elif name_clean in ("flattop", "flat_top", "flat"):
        return sp_signal.windows.flattop(length), "flattop"
    elif name_clean in ("rect", "rectangular", "boxcar", "none", "uniform"):
        return np.ones(length, dtype=np.float64), "rectangular"
    elif name_clean in ("bartlett", "triang", "triangular"):
        return np.bartlett(length), "bartlett"
    elif name_clean in ("kaiser",):
        return np.kaiser(length, 14.0), "kaiser"
    elif name_clean in ("blackmanharris", "blackman_harris"):
        return sp_signal.windows.blackmanharris(length), "blackmanharris"

    return np.hanning(length), "hann"


def _validated_stft(
    iq_data: Any,
    sample_rate: float = 1.0,
    window_length: int = 256,
    overlap: float | int = 0.75,
    nfft: int | None = None,
    window: str | np.ndarray = "hann",
    sample_offset: int = 0,
    eps: float = 1e-18,
) -> Dict[str, Any]:
    """
    Port of validated discrete STFT engine matching MATLAB `stft` and Time Frequency package.
    """
    z = _as_iq_array(iq_data)
    fs = _validate_sample_rate(sample_rate)
    n_samples = len(z)
    m_win = int(window_length)
    if m_win <= 0:
        raise ValueError(f"window_length must be > 0, got {window_length}")

    if isinstance(overlap, float):
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"Overlap fraction must be in [0, 1), got {overlap}")
        l_overlap = int(round(m_win * overlap))
    else:
        l_overlap = int(overlap)
        if not (0 <= l_overlap < m_win):
            raise ValueError(f"Overlap samples must be in [0, {m_win}), got {l_overlap}")

    hop_r = max(1, m_win - l_overlap)

    if nfft is None:
        n_fft = max(m_win, 256)
    else:
        n_fft = int(nfft)
        if n_fft < m_win:
            raise ValueError(f"nfft ({n_fft}) must be >= window_length ({m_win})")

    win_vec, win_name = _get_tf_window(window, m_win)

    if n_samples < m_win:
        freqs = (np.arange(n_fft) - (n_fft // 2)) * (fs / n_fft)
        return {
            "stft_complex": np.empty((n_fft, 0), dtype=np.complex128),
            "power_spectrogram": np.empty((n_fft, 0), dtype=np.float64),
            "frequencies_hz": freqs,
            "times_sec": np.empty(0, dtype=np.float64),
            "sample_rate": fs,
            "hop_samples": hop_r,
            "window_length": m_win,
            "nfft": n_fft,
        }

    num_frames = (n_samples - m_win) // hop_r + 1
    start_indices = np.arange(num_frames) * hop_r

    shape = (num_frames, m_win)
    strides = (z.strides[0] * hop_r, z.strides[0])
    frames = np.lib.stride_tricks.as_strided(z, shape=shape, strides=strides)

    windowed_frames = frames * win_vec
    stft_raw = np.fft.fft(windowed_frames, n=n_fft, axis=1)
    stft_centered = np.fft.fftshift(stft_raw, axes=1)
    stft_complex = stft_centered.T

    k_indices = np.arange(n_fft)
    freqs = (k_indices - (n_fft // 2)) * (fs / n_fft)
    times_sec = (sample_offset + start_indices + (m_win - 1) / 2.0) / fs
    power_spectrogram = np.abs(stft_complex) ** 2

    return {
        "stft_complex": stft_complex,
        "power_spectrogram": power_spectrogram,
        "frequencies_hz": freqs,
        "times_sec": times_sec,
        "sample_rate": fs,
        "hop_samples": hop_r,
        "window_length": m_win,
        "nfft": n_fft,
    }


def _frequency_trajectory(
    iq_or_tf: Any,
    sample_rate: float = 1.0,
    trajectory_mode: str = "tfmoment",
    window_length: int = 256,
    overlap: float = 0.75,
    nfft: int | None = None,
    eps: float = 1e-18,
) -> np.ndarray:
    """
    Port of validated Frequency Trajectory (first conditional spectral moment or ridge).
    """
    if isinstance(iq_or_tf, dict) and "power_spectrogram" in iq_or_tf and "frequencies_hz" in iq_or_tf:
        p_linear = np.asarray(iq_or_tf["power_spectrogram"], dtype=np.float64)
        freqs = np.asarray(iq_or_tf["frequencies_hz"], dtype=np.float64)
    else:
        stft_res = _validated_stft(iq_or_tf, sample_rate=sample_rate, window_length=window_length, overlap=overlap, nfft=nfft, eps=eps)
        p_linear = stft_res["power_spectrogram"]
        freqs = stft_res["frequencies_hz"]

    if p_linear.size == 0 or p_linear.shape[1] == 0:
        return np.empty(0, dtype=np.float64)

    mode_clean = str(trajectory_mode).lower().strip().replace("-", "_")
    if mode_clean in ("ridge", "peak", "argmax"):
        max_bin_indices = np.argmax(p_linear, axis=0)
        return freqs[max_bin_indices]

    den = np.sum(p_linear, axis=0) + eps
    num = np.sum(freqs[:, None] * p_linear, axis=0)
    return num / den


def _instantaneous_bandwidth(
    iq_or_tf: Any,
    sample_rate: float = 1.0,
    scale_factor: float | None = None,
    window_length: int = 256,
    overlap: float = 0.75,
    nfft: int | None = None,
    eps: float = 1e-18,
) -> np.ndarray:
    """
    Port of validated Instantaneous Bandwidth (IBW = sqrt(4*pi)*sigma_f).
    """
    if isinstance(iq_or_tf, dict) and "power_spectrogram" in iq_or_tf and "frequencies_hz" in iq_or_tf:
        p_linear = np.asarray(iq_or_tf["power_spectrogram"], dtype=np.float64)
        freqs = np.asarray(iq_or_tf["frequencies_hz"], dtype=np.float64)
    else:
        stft_res = _validated_stft(iq_or_tf, sample_rate=sample_rate, window_length=window_length, overlap=overlap, nfft=nfft, eps=eps)
        p_linear = stft_res["power_spectrogram"]
        freqs = stft_res["frequencies_hz"]

    if p_linear.size == 0 or p_linear.shape[1] == 0:
        return np.empty(0, dtype=np.float64)

    c_scale = float(scale_factor) if scale_factor is not None else math.sqrt(4.0 * math.pi)

    den = np.sum(p_linear, axis=0) + eps
    num = np.sum(freqs[:, None] * p_linear, axis=0)
    f_inst = num / den

    diff_f = freqs[:, None] - f_inst[None, :]
    var_f = np.sum((diff_f ** 2) * p_linear, axis=0) / den
    sigma_f = np.sqrt(np.maximum(0.0, var_f))

    return c_scale * sigma_f


def _time_frequency_entropy(
    iq_or_tf: Any,
    sample_rate: float = 1.0,
    normalized: bool = True,
    window_length: int = 256,
    overlap: float = 0.75,
    nfft: int | None = None,
    eps: float = 1e-18,
) -> np.ndarray:
    """
    Port of validated Time-Frequency Shannon Entropy per STFT frame.
    """
    if isinstance(iq_or_tf, dict) and "power_spectrogram" in iq_or_tf:
        p_linear = np.asarray(iq_or_tf["power_spectrogram"], dtype=np.float64)
    else:
        stft_res = _validated_stft(iq_or_tf, sample_rate=sample_rate, window_length=window_length, overlap=overlap, nfft=nfft, eps=eps)
        p_linear = stft_res["power_spectrogram"]

    if p_linear.size == 0 or p_linear.shape[1] == 0:
        return np.empty(0, dtype=np.float64)

    n_bins = p_linear.shape[0]
    max_entropy = math.log2(n_bins) if n_bins > 1 else 1.0

    sum_p = np.sum(p_linear, axis=0, keepdims=True) + eps
    prob = p_linear / sum_p

    h_raw = -np.sum(prob * np.log2(np.maximum(1e-300, prob)), axis=0)
    h_norm = np.clip(h_raw / max_entropy, 0.0, 1.0)

    return h_norm if normalized else h_raw


# ==============================================================================
# 2. FROZEN SPECTRAL HELPERS (PORTED FROM Spectral PACKAGE)
# ==============================================================================

def _create_spectral_window(name: str, length: int) -> np.ndarray:
    """Generates window matching Spectral package conventions."""
    w_name = name.lower().strip()
    if w_name in ("hann", "hanning"):
        return np.hanning(length)
    elif w_name == "hamming":
        return np.hamming(length)
    elif w_name == "blackman":
        return np.blackman(length)
    elif w_name in ("flattop", "flat_top"):
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
    return np.hanning(length)


def _validated_welch_psd(
    z: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """Port of validated Welch PSD for complex baseband IQ."""
    n = len(z)
    if nperseg is None:
        nperseg = min(n, 1024)
        nperseg = max(16, 1 << (nperseg.bit_length() - 1))

    if noverlap is None:
        noverlap = nperseg // 2

    win = _create_spectral_window(window, nperseg)
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


def _spectral_centroid(iq_data: Any, sample_rate: float = 1.0, nperseg: int | None = None, window: str = "hann") -> float:
    """Port of validated Spectral Centroid."""
    z = _as_iq_array(iq_data)
    fs = _validate_sample_rate(sample_rate)
    if z.size == 0:
        return 0.0
    freqs, psd = _validated_welch_psd(z, fs=fs, nperseg=nperseg, window=window)
    total_pwr = np.sum(psd)
    if total_pwr <= 0:
        return 0.0
    return float(np.sum(freqs * psd) / total_pwr)


def _spectral_variance(iq_data: Any, sample_rate: float = 1.0, nperseg: int | None = None, window: str = "hann") -> float:
    """Port of validated Spectral Spread (std in Hz)."""
    z = _as_iq_array(iq_data)
    fs = _validate_sample_rate(sample_rate)
    if z.size == 0:
        return 0.0
    freqs, psd = _validated_welch_psd(z, fs=fs, nperseg=nperseg, window=window)
    total_pwr = np.sum(psd)
    if total_pwr <= 0:
        return 0.0
    fc = np.sum(freqs * psd) / total_pwr
    var_hz2 = float(np.sum(((freqs - fc) ** 2) * psd) / total_pwr)
    return float(np.sqrt(max(0.0, var_hz2)))


def _spectral_skewness(iq_data: Any, sample_rate: float = 1.0, nperseg: int | None = None, window: str = "hann") -> float:
    """Port of validated Spectral Skewness."""
    z = _as_iq_array(iq_data)
    fs = _validate_sample_rate(sample_rate)
    if z.size == 0:
        return 0.0
    freqs, psd = _validated_welch_psd(z, fs=fs, nperseg=nperseg, window=window)
    total_pwr = np.sum(psd)
    if total_pwr <= 0:
        return 0.0
    fc = np.sum(freqs * psd) / total_pwr
    var_hz2 = np.sum(((freqs - fc) ** 2) * psd) / total_pwr
    sigma_f = np.sqrt(max(1e-24, var_hz2))
    m3 = np.sum(((freqs - fc) ** 3) * psd) / total_pwr
    return float(m3 / (sigma_f ** 3))


def _spectral_kurtosis(iq_data: Any, sample_rate: float = 1.0, nperseg: int | None = None, window: str = "hann") -> float:
    """Port of validated Fourth Spectral Moment Kurtosis."""
    z = _as_iq_array(iq_data)
    fs = _validate_sample_rate(sample_rate)
    if z.size == 0:
        return 0.0
    freqs, psd = _validated_welch_psd(z, fs=fs, nperseg=nperseg, window=window)
    total_pwr = np.sum(psd)
    if total_pwr <= 0:
        return 0.0
    fc = np.sum(freqs * psd) / total_pwr
    var_hz2 = np.sum(((freqs - fc) ** 2) * psd) / total_pwr
    sigma_f = np.sqrt(max(1e-24, var_hz2))
    m4 = np.sum(((freqs - fc) ** 4) * psd) / total_pwr
    return float(m4 / (sigma_f ** 4))


def _spectral_entropy(iq_data: Any, sample_rate: float = 1.0, nperseg: int | None = None, window: str = "hann", normalize: bool = True) -> float:
    """Port of validated Spectral Shannon Entropy."""
    z = _as_iq_array(iq_data)
    fs = _validate_sample_rate(sample_rate)
    if z.size == 0:
        return 0.0
    freqs, psd = _validated_welch_psd(z, fs=fs, nperseg=nperseg, window=window)
    total_pwr = np.sum(psd)
    if total_pwr <= 0 or psd.size <= 1:
        return 0.0
    p_k = psd / total_pwr
    valid = p_k > 1e-24
    p_valid = p_k[valid]
    h_bits = float(-np.sum(p_valid * np.log2(p_valid)))
    h_max = float(np.log2(len(psd)))
    h_norm = float(np.clip(h_bits / max(1e-12, h_max), 0.0, 1.0))
    return h_norm if normalize else h_bits


def _spectral_flatness(iq_data: Any, sample_rate: float = 1.0, nperseg: int | None = None, window: str = "hann") -> float:
    """Port of validated Spectral Flatness (Wiener entropy in [0, 1])."""
    z = _as_iq_array(iq_data)
    fs = _validate_sample_rate(sample_rate)
    if z.size == 0:
        return 0.0
    freqs, psd = _validated_welch_psd(z, fs=fs, nperseg=nperseg, window=window)
    if psd.size == 0:
        return 0.0
    arithmetic_mean = float(np.mean(psd))
    if arithmetic_mean <= 0:
        return 0.0
    log_psd = np.log(np.maximum(1e-24, psd))
    geometric_mean = float(np.exp(np.mean(log_psd)))
    return float(np.clip(geometric_mean / arithmetic_mean, 0.0, 1.0))


def _spectral_peak_frequency(iq_data: Any, sample_rate: float = 1.0, nperseg: int | None = None, window: str = "hann") -> float:
    """Port of validated Spectral Peak Frequency with parabolic sub-bin refinement."""
    z = _as_iq_array(iq_data)
    fs = _validate_sample_rate(sample_rate)
    if z.size == 0:
        return 0.0
    freqs, psd = _validated_welch_psd(z, fs=fs, nperseg=nperseg, window=window)
    if psd.size == 0:
        return 0.0
    df = fs / len(psd)
    psd_db = 10.0 * np.log10(np.maximum(1e-24, psd))
    global_idx = int(np.argmax(psd))
    f_global_bin = float(freqs[global_idx])

    if 0 < global_idx < len(psd) - 1:
        y0 = psd_db[global_idx - 1]
        y1 = psd_db[global_idx]
        y2 = psd_db[global_idx + 1]
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        return float(f_global_bin + delta * df)
    return f_global_bin


def _occupied_bandwidth(iq_data: Any, sample_rate: float = 1.0, percent_power: float = 99.0, nperseg: int | None = None, window: str = "hann") -> float:
    """Port of validated Occupied Bandwidth (OBW)."""
    z = _as_iq_array(iq_data)
    fs = _validate_sample_rate(sample_rate)
    if z.size == 0:
        return 0.0
    freqs, psd = _validated_welch_psd(z, fs=fs, nperseg=nperseg, window=window)
    df = fs / len(psd)
    cum_power = np.cumsum(psd * df)
    total_power = cum_power[-1]
    if total_power <= 0:
        return 0.0
    beta = np.clip(percent_power, 1.0, 99.99)
    tail_pct = (100.0 - beta) / 2.0
    p_low_target = total_power * (tail_pct / 100.0)
    p_high_target = total_power * ((100.0 - tail_pct) / 100.0)
    f_low = float(np.interp(p_low_target, cum_power, freqs))
    f_high = float(np.interp(p_high_target, cum_power, freqs))
    return float(max(0.0, f_high - f_low))


# ==============================================================================
# 3. FROZEN CYCLOSTATIONARY HELPERS (PORTED FROM Cyclostationary Analysis PACKAGE)
# ==============================================================================

def _next_power_of_two(n: int) -> int:
    """Returns smallest power of two >= n."""
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


def _validated_cyclic_frequency_candidates(
    iq_data: Any,
    sample_rate: float = 1.0,
    nfft_alpha: int | None = None,
    cyclic_lags: Tuple[int, ...] = (0, 1, 2, 4, 8),
    window: str = "hann",
    num_candidates: int = 5,
    min_alpha_hz: float | None = None,
    snr_threshold_db: float = 8.0,
) -> Dict[str, Any]:
    """
    Port of validated Fine Cyclic Frequency Profile & Candidate Search.
    """
    z = _as_iq_array(iq_data)
    fs = _validate_sample_rate(sample_rate)
    n_samples = z.size
    if n_samples < 8:
        empty = np.array([], dtype=np.float64)
        return {
            "alpha_axis_hz": empty,
            "alpha_profile": empty,
            "delta_alpha_hz": fs,
            "alpha_bin_spacing_hz": fs,
            "alpha_observation_resolution_hz": fs,
            "nfft_alpha": 0,
            "num_samples": n_samples,
            "candidate_alphas_hz": [],
            "candidate_scores": [],
            "candidates": [],
            "primary_signed_alpha_hz": None,
            "primary_absolute_rate_hz": None,
            "estimated_cyclic_rate_hz": None,
            "raw_primary_alpha_hz": 0.0,
            "raw_primary_rate_hz": 0.0,
            "raw_cyclic_snr_db": 0.0,
            "raw_peak_score": 0.0,
            "cyclostationary_detected": False,
            "noise_floor": 1e-18,
        }

    if nfft_alpha is None or nfft_alpha <= 0:
        nfft_alpha = max(256, _next_power_of_two(n_samples))

    delta_alpha_spacing = fs / nfft_alpha
    delta_alpha_obs = fs / max(1, n_samples)
    alpha_axis = np.fft.fftshift(np.fft.fftfreq(nfft_alpha, d=1.0 / fs))

    valid_lags = [int(ell) for ell in cyclic_lags if abs(int(ell)) < n_samples]
    if not valid_lags:
        valid_lags = [0]

    combined_profile = np.zeros(nfft_alpha, dtype=np.float64)

    for ell in valid_lags:
        if ell >= 0:
            y = z[ell:] * np.conj(z[: n_samples - ell])
        else:
            a = abs(ell)
            y = z[: n_samples - a] * np.conj(z[a:])

        m_len = len(y)
        if m_len == 0:
            continue

        w = _create_spectral_window(window, m_len)
        y_win = y * w
        w_norm = np.sum(w)
        if w_norm <= 0:
            w_norm = float(m_len)

        Y_alpha = np.fft.fftshift(np.fft.fft(y_win, n=nfft_alpha)) / w_norm
        combined_profile += (np.abs(Y_alpha) ** 2)

    combined_profile /= float(len(valid_lags))

    if min_alpha_hz is None or min_alpha_hz <= 0:
        min_alpha_hz = max(3.0 * delta_alpha_obs, 0.01 * fs)

    valid_mask = np.abs(alpha_axis) >= min_alpha_hz
    p_search = combined_profile.copy()
    p_search[~valid_mask] = 0.0

    if np.any(valid_mask):
        noise_floor = float(np.median(combined_profile[valid_mask]))
    else:
        noise_floor = float(np.median(combined_profile))
    noise_floor = max(1e-20, noise_floor)

    min_dist_bins = max(3, int(0.005 * nfft_alpha))
    peak_indices, _ = sp_signal.find_peaks(
        p_search,
        height=noise_floor * 1.5,
        distance=min_dist_bins,
    )

    if len(peak_indices) > 0:
        sorted_order = np.argsort(p_search[peak_indices])[::-1]
        top_indices = peak_indices[sorted_order[:num_candidates]]
    else:
        top_indices = np.array([], dtype=int)

    candidates: List[Dict[str, Any]] = []
    for rank, p_idx in enumerate(top_indices, start=1):
        signed_alpha = float(alpha_axis[p_idx])
        abs_rate = abs(signed_alpha)
        score_val = float(combined_profile[p_idx])
        cyclic_snr_db = float(10.0 * np.log10(max(1e-18, score_val / noise_floor)))

        candidates.append({
            "rank": rank,
            "signed_alpha_hz": signed_alpha,
            "absolute_cyclic_rate_hz": abs_rate,
            "alpha_hz": signed_alpha,
            "score": score_val,
            "magnitude": score_val,
            "cyclic_snr_db": cyclic_snr_db,
            "bin_index": int(p_idx),
            "estimated_cyclic_rate_hz": abs_rate,
            "estimated_symbol_rate_hz": abs_rate,
        })

    if candidates:
        raw_top_alpha = candidates[0]["signed_alpha_hz"]
        raw_top_rate = candidates[0]["absolute_cyclic_rate_hz"]
        raw_top_snr = candidates[0]["cyclic_snr_db"]
        raw_top_score = candidates[0]["score"]
        cyclostationary_detected = bool(raw_top_snr >= snr_threshold_db and raw_top_rate > 0)
    else:
        raw_top_alpha = 0.0
        raw_top_rate = 0.0
        raw_top_snr = 0.0
        raw_top_score = float(np.max(p_search)) if p_search.size > 0 else 0.0
        cyclostationary_detected = False

    return {
        "alpha_axis_hz": alpha_axis,
        "alpha_profile": combined_profile,
        "delta_alpha_hz": delta_alpha_spacing,
        "alpha_bin_spacing_hz": delta_alpha_spacing,
        "alpha_observation_resolution_hz": delta_alpha_obs,
        "nfft_alpha": nfft_alpha,
        "num_samples": n_samples,
        "candidate_alphas_hz": [c["signed_alpha_hz"] for c in candidates],
        "candidate_scores": [c["score"] for c in candidates],
        "candidates": candidates,
        "primary_signed_alpha_hz": raw_top_alpha if cyclostationary_detected else None,
        "primary_absolute_rate_hz": raw_top_rate if cyclostationary_detected else None,
        "estimated_cyclic_rate_hz": raw_top_rate if cyclostationary_detected else None,
        "raw_primary_alpha_hz": raw_top_alpha,
        "raw_primary_rate_hz": raw_top_rate,
        "raw_cyclic_snr_db": raw_top_snr,
        "raw_peak_score": raw_top_score,
        "cyclostationary_detected": cyclostationary_detected,
        "noise_floor": noise_floor,
    }


# ==============================================================================
# 4. SHARED AM / ENVELOPE PRIMITIVES
# ==============================================================================

def _analyze_envelope(
    iq_data: Any,
    sample_rate: float = 1.0,
    ref_method: str = "robust",
    smoothing: str = "none",
    smoothing_window: int = 5,
    threshold_db: float = -60.0,
) -> Dict[str, Any]:
    """
    Canonical envelope and AM modulation extractor.
    """
    z = _as_iq_array(iq_data)
    fs = _validate_sample_rate(sample_rate)
    n = len(z)

    if n == 0:
        empty = np.array([], dtype=np.float64)
        return {
            "envelope": empty,
            "normalized_envelope": empty,
            "recovered_modulation": empty,
            "times_sec": empty,
            "valid_mask": np.array([], dtype=bool),
            "reference_amplitude": 0.0,
            "envelope_mean": 0.0,
            "envelope_std": 0.0,
            "envelope_rms_variation": 0.0,
            "envelope_peak_to_peak": 0.0,
            "am_activity_score": 0.0,
            "valid_sample_fraction": 0.0,
            "sample_rate": fs,
        }

    env = np.abs(z)

    if smoothing == "moving_average" and smoothing_window > 1 and n >= smoothing_window:
        kernel = np.ones(smoothing_window, dtype=np.float64) / smoothing_window
        env_smooth = np.convolve(env, kernel, mode="same")
    else:
        env_smooth = env

    valid_mask = _amplitude_valid_mask(z, threshold_db=threshold_db)
    valid_fraction = float(np.mean(valid_mask)) if n > 0 else 0.0

    if not np.any(valid_mask):
        env_masked = np.full(n, np.nan, dtype=np.float64)
        times_sec = np.arange(n, dtype=np.float64) / fs
        return {
            "envelope": env_masked,
            "normalized_envelope": env_masked.copy(),
            "recovered_modulation": env_masked.copy(),
            "times_sec": times_sec,
            "valid_mask": valid_mask,
            "reference_amplitude": 0.0,
            "envelope_mean": 0.0,
            "envelope_std": 0.0,
            "envelope_rms_variation": 0.0,
            "envelope_peak_to_peak": 0.0,
            "am_activity_score": 0.0,
            "valid_sample_fraction": 0.0,
            "sample_rate": fs,
        }

    active_env = env_smooth[valid_mask]

    ref_clean = str(ref_method).lower().strip()
    if ref_clean in ("robust", "trimmed_mean", "trimmed"):
        if len(active_env) >= 10:
            ac = float(trim_mean(active_env, proportiontocut=0.1))
        else:
            ac = float(np.median(active_env))
    elif ref_clean == "mean":
        ac = float(np.mean(active_env))
    else:
        ac = float(np.median(active_env))

    ac = max(1e-18, ac)

    norm_env = env_smooth / ac
    rec_mod = (env_smooth - ac) / ac

    env_masked = env_smooth.copy()
    env_masked[~valid_mask] = np.nan
    norm_env_masked = norm_env.copy()
    norm_env_masked[~valid_mask] = np.nan
    rec_mod_masked = rec_mod.copy()
    rec_mod_masked[~valid_mask] = np.nan

    mean_val = float(np.mean(active_env))
    std_val = float(np.std(active_env))
    rms_var = float(np.sqrt(np.mean((active_env - mean_val) ** 2)))
    p2p_val = float(np.max(active_env) - np.min(active_env)) if len(active_env) > 0 else 0.0

    am_activity = float(std_val / max(1e-12, mean_val))
    times_sec = np.arange(n, dtype=np.float64) / fs

    return {
        "envelope": env_masked,
        "normalized_envelope": norm_env_masked,
        "recovered_modulation": rec_mod_masked,
        "times_sec": times_sec,
        "valid_mask": valid_mask,
        "reference_amplitude": ac,
        "envelope_mean": mean_val,
        "envelope_std": std_val,
        "envelope_rms_variation": rms_var,
        "envelope_peak_to_peak": p2p_val,
        "am_activity_score": am_activity,
        "valid_sample_fraction": valid_fraction,
        "sample_rate": fs,
    }


def _estimate_am_depth(
    envelope: np.ndarray,
    estimator: str = "robust",
    valid_mask: np.ndarray | None = None,
    eps: float = 1e-18,
) -> Dict[str, Any]:
    """
    Canonical AM modulation depth estimator.
    """
    if envelope.size == 0:
        return {
            "modulation_depth": None,
            "modulation_depth_percent": None,
            "envelope_max": 0.0,
            "envelope_min": 0.0,
            "estimator": estimator,
            "overmodulated": False,
            "valid": False,
            "confidence": 0.0,
            "reason": "Empty envelope input.",
        }

    if valid_mask is not None and np.any(valid_mask):
        act_env = envelope[valid_mask]
    else:
        act_env = envelope[np.isfinite(envelope)]

    if act_env.size == 0:
        return {
            "modulation_depth": None,
            "modulation_depth_percent": None,
            "envelope_max": 0.0,
            "envelope_min": 0.0,
            "estimator": estimator,
            "overmodulated": False,
            "valid": False,
            "confidence": 0.0,
            "reason": "No valid active envelope samples.",
        }

    est_clean = str(estimator).lower().strip()
    if est_clean in ("exact", "extrema", "minmax"):
        a_max = float(np.max(act_env))
        a_min = float(np.min(act_env))
    else:
        a_max = float(np.percentile(act_env, 99.5))
        a_min = float(np.percentile(act_env, 0.5))

    a_min = max(0.0, a_min)
    a_max = max(a_min, a_max)

    denom = a_max + a_min + eps
    m_depth = float((a_max - a_min) / denom)
    m_percent = float(m_depth * 100.0)

    med_val = float(np.median(act_env))
    
    # Overmodulation detection: folded cusps produce secondary local peaks between zero-crossings
    overmod = False
    if m_depth >= 0.99 and len(act_env) >= 100:
        env_norm = act_env / max(1e-12, a_max)
        sub_env = env_norm[:min(len(env_norm), 10000)]
        pks, _ = sp_signal.find_peaks(sub_env, height=0.01, distance=15)
        if len(pks) >= 6:
            pk_vals = sub_env[pks]
            low_pks = pk_vals[pk_vals < 0.75]
            if len(low_pks) >= 3:
                overmod = True

    if overmod:
        is_valid = False
        reason_str = "Conventional envelope-depth definition is not uniquely valid for overmodulated AM."
        conf = 0.95
    else:
        is_valid = True
        reason_str = "Normal AM modulation depth."
        conf = float(np.clip(1.0 - (np.std(act_env) / max(1e-12, med_val) * 0.1), 0.5, 1.0))

    return {
        "modulation_depth": m_depth,
        "modulation_depth_percent": m_percent,
        "envelope_max": a_max,
        "envelope_min": a_min,
        "estimator": est_clean,
        "overmodulated": overmod,
        "valid": is_valid,
        "confidence": conf,
        "reason": reason_str,
    }


# ==============================================================================
# 5. SHARED FM / INSTANTANEOUS FREQUENCY PRIMITIVES
# ==============================================================================

def _instantaneous_frequency(
    iq_data: Any,
    sample_rate: float = 1.0,
    smoothing: str = "none",
    smoothing_window: int = 5,
    threshold_db: float = -60.0,
) -> Dict[str, Any]:
    """
    Canonical instantaneous frequency primitive via conjugate product discrimination.
    """
    z = _as_iq_array(iq_data)
    fs = _validate_sample_rate(sample_rate)
    n = len(z)

    if n < 2:
        empty = np.array([], dtype=np.float64)
        return {
            "instantaneous_phase_rad": empty,
            "instantaneous_frequency_hz": empty,
            "times_sec": empty,
            "valid_mask": np.array([], dtype=bool),
            "valid_sample_fraction": 0.0,
            "sample_rate": fs,
        }

    conj_prod = z[1:] * np.conj(z[:-1])
    d_phi = np.angle(conj_prod)
    f_inst = (fs / (2.0 * np.pi)) * d_phi
    inst_phase = np.unwrap(np.angle(z))

    valid_base = _amplitude_valid_mask(z, threshold_db=threshold_db)
    valid_pairs = valid_base[1:] & valid_base[:-1] & np.isfinite(f_inst)

    if smoothing == "moving_average" and smoothing_window > 1 and len(f_inst) >= smoothing_window:
        kernel = np.ones(smoothing_window, dtype=np.float64) / smoothing_window
        f_inst_smooth = np.convolve(f_inst, kernel, mode="same")
    else:
        f_inst_smooth = f_inst

    f_inst_full = np.empty(n, dtype=np.float64)
    f_inst_full[0] = f_inst_smooth[0] if len(f_inst_smooth) > 0 else 0.0
    f_inst_full[1:] = f_inst_smooth

    valid_full = np.empty(n, dtype=bool)
    valid_full[0] = bool(valid_base[0] and (valid_pairs[0] if len(valid_pairs) > 0 else True))
    valid_full[1:] = valid_pairs

    f_inst_masked = f_inst_full.copy()
    f_inst_masked[~valid_full] = np.nan

    inst_phase_masked = inst_phase.copy()
    inst_phase_masked[~valid_full] = np.nan

    times_sec = np.arange(n, dtype=np.float64) / fs
    valid_fraction = float(np.mean(valid_full)) if n > 0 else 0.0

    return {
        "instantaneous_phase_rad": inst_phase_masked,
        "instantaneous_frequency_hz": f_inst_masked,
        "times_sec": times_sec,
        "valid_mask": valid_full,
        "valid_sample_fraction": valid_fraction,
        "sample_rate": fs,
    }


def _center_frequency_deviation(
    f_inst: np.ndarray,
    valid_mask: np.ndarray | None = None,
    method: str = "median",
) -> Tuple[float, np.ndarray]:
    """
    Estimates center frequency f_0 and computes frequency deviation f_d[n] = f_i[n] - f_0.
    """
    if f_inst.size == 0:
        return 0.0, np.array([], dtype=np.float64)

    if valid_mask is not None and np.any(valid_mask):
        act_f = f_inst[valid_mask]
    else:
        act_f = f_inst[np.isfinite(f_inst)]

    if act_f.size == 0:
        return 0.0, np.zeros_like(f_inst)

    m_clean = str(method).lower().strip()
    if m_clean in ("median", "exact_median"):
        f0 = float(np.median(act_f))
    elif m_clean in ("robust", "trimmed"):
        if len(act_f) >= 10:
            act_f_filtered = sp_signal.medfilt(act_f, 3)
            f0 = float(0.5 * (np.percentile(act_f_filtered, 99.5) + np.percentile(act_f_filtered, 0.5)))
        else:
            f0 = float(np.median(act_f))
    elif m_clean == "mean":
        f0 = float(np.mean(act_f))
    elif m_clean in ("midrange", "minmax", "extrema"):
        f0 = float(0.5 * (np.max(act_f) + np.min(act_f)))
    else:
        # Default center estimation: robust midrange (0.5*(P99.5 + P0.5))
        # which eliminates discrete sampling bias of sinusoidal FM while remaining robust against noise.
        if len(act_f) >= 10:
            act_f_filtered = sp_signal.medfilt(act_f, 3)
            f0 = float(0.5 * (np.percentile(act_f_filtered, 99.5) + np.percentile(act_f_filtered, 0.5)))
        elif len(act_f) >= 2:
            f0 = float(0.5 * (np.max(act_f) + np.min(act_f)))
        else:
            f0 = float(np.median(act_f))

    f_dev = f_inst - f0
    if valid_mask is not None:
        f_dev[~valid_mask] = np.nan
    return f0, f_dev


def _estimate_frequency_deviation(
    f_dev: np.ndarray,
    valid_mask: np.ndarray | None = None,
    use_robust_extrema: bool = False,
) -> Dict[str, float]:
    """
    Canonical frequency deviation statistical moments.
    """
    if f_dev.size == 0:
        return {
            "peak_deviation_hz": 0.0,
            "positive_deviation_hz": 0.0,
            "negative_deviation_hz": 0.0,
            "peak_to_peak_deviation_hz": 0.0,
            "rms_deviation_hz": 0.0,
            "mean_deviation_hz": 0.0,
            "std_deviation_hz": 0.0,
        }

    if valid_mask is not None and np.any(valid_mask):
        act_dev = f_dev[valid_mask]
    else:
        act_dev = f_dev[np.isfinite(f_dev)]

    if act_dev.size == 0:
        return {
            "peak_deviation_hz": 0.0,
            "positive_deviation_hz": 0.0,
            "negative_deviation_hz": 0.0,
            "peak_to_peak_deviation_hz": 0.0,
            "rms_deviation_hz": 0.0,
            "mean_deviation_hz": 0.0,
            "std_deviation_hz": 0.0,
        }

    if use_robust_extrema and len(act_dev) >= 50:
        pos_dev = float(np.percentile(act_dev, 99.5))
        neg_dev = float(abs(np.percentile(act_dev, 0.5)))
    else:
        pos_dev = float(np.max(act_dev))
        neg_dev = float(abs(np.min(act_dev)))

    peak_dev = max(pos_dev, neg_dev)
    p2p_dev = float(np.max(act_dev) - np.min(act_dev))
    rms_dev = float(np.sqrt(np.mean(act_dev ** 2)))
    mean_dev = float(np.mean(act_dev))
    std_dev = float(np.std(act_dev))

    return {
        "peak_deviation_hz": peak_dev,
        "positive_deviation_hz": pos_dev,
        "negative_deviation_hz": neg_dev,
        "peak_to_peak_deviation_hz": p2p_dev,
        "rms_deviation_hz": rms_dev,
        "mean_deviation_hz": mean_dev,
        "std_deviation_hz": std_dev,
    }


# ==============================================================================
# 6. SHARED PM / DETRENDED PHASE PRIMITIVES
# ==============================================================================

def _detrended_phase(
    iq_data: Any,
    sample_rate: float = 1.0,
    fit_method: str = "least_squares",
    threshold_db: float = -60.0,
    smoothing: str = "none",
    smoothing_window: int = 5,
) -> Dict[str, Any]:
    """
    Canonical PM unwrapping and linear carrier trend removal via centered least-squares.
    """
    z = _as_iq_array(iq_data)
    fs = _validate_sample_rate(sample_rate)
    n = len(z)

    if n == 0:
        empty = np.array([], dtype=np.float64)
        return {
            "unwrapped_phase_rad": empty,
            "phase_trend_rad": empty,
            "phase_deviation_rad": empty,
            "times_sec": empty,
            "valid_mask": np.array([], dtype=bool),
            "estimated_carrier_frequency_hz": 0.0,
            "constant_phase_rad": 0.0,
            "valid_sample_fraction": 0.0,
            "sample_rate": fs,
        }

    valid_mask = _amplitude_valid_mask(z, threshold_db=threshold_db) & np.isfinite(z)
    valid_fraction = float(np.mean(valid_mask)) if n > 0 else 0.0

    if not np.any(valid_mask):
        empty_nan = np.full(n, np.nan, dtype=np.float64)
        return {
            "unwrapped_phase_rad": empty_nan,
            "phase_trend_rad": empty_nan.copy(),
            "phase_deviation_rad": empty_nan.copy(),
            "times_sec": np.arange(n, dtype=np.float64) / fs,
            "valid_mask": valid_mask,
            "estimated_carrier_frequency_hz": 0.0,
            "constant_phase_rad": 0.0,
            "valid_sample_fraction": 0.0,
            "sample_rate": fs,
        }

    indices = np.arange(n, dtype=np.float64)
    valid_idx = indices[valid_mask]
    valid_pairs = valid_mask[1:] & valid_mask[:-1]

    if np.all(valid_mask):
        # 1. Full unwrapped phase directly from complex IQ with robust midrange carrier slope
        d_phi_inst = np.angle(z[1:] * np.conj(z[:-1]))
        fi = (fs / (2.0 * np.pi)) * d_phi_inst
        if len(fi) >= 50:
            win_s = min(31, max(3, len(fi) // 8))
            fi_s = uniform_filter1d(fi, size=win_s, mode="nearest")
            carrier_freq = float(0.5 * (np.percentile(fi_s, 99.5) + np.percentile(fi_s, 0.5)))
        elif len(fi) >= 2:
            carrier_freq = float(0.5 * (np.max(fi) + np.min(fi)))
        else:
            carrier_freq = 0.0
            
        slope_a = float(2.0 * np.pi * carrier_freq / fs)
        raw_phase = np.unwrap(np.angle(z))
        trend = slope_a * indices
        phase_dev = raw_phase - trend
        intercept_b = float(0.5 * (np.max(phase_dev) + np.min(phase_dev)))
        phase_dev = phase_dev - intercept_b
        trend = trend + intercept_b
    elif np.any(valid_pairs):
        # Gapped / Discontinuous signal: conjugate-product on valid pairs avoids phase wrapping artifacts across gaps
        conj_prod = z[1:][valid_pairs] * np.conj(z[:-1][valid_pairs])
        omega_c = float(np.angle(np.sum(conj_prod)))
        slope_a = omega_c
        carrier_freq = float((slope_a * fs) / (2.0 * np.pi))
        z_rot = z * np.exp(-1j * (slope_a * indices))
        phase_dev = np.angle(z_rot)
        trend = slope_a * indices
        raw_phase = phase_dev + trend
        intercept_b = float(0.5 * (np.max(phase_dev[valid_mask]) + np.min(phase_dev[valid_mask]))) if np.any(valid_mask) else 0.0
        phase_dev = phase_dev - intercept_b
        trend = trend + intercept_b
    else:
        slope_a = 0.0
        intercept_b = 0.0
        carrier_freq = 0.0
        trend = np.zeros(n, dtype=np.float64)
        raw_phase = np.zeros(n, dtype=np.float64)
        phase_dev = np.zeros(n, dtype=np.float64)

    if smoothing == "moving_average" and smoothing_window > 1 and n >= smoothing_window:
        kernel = np.ones(smoothing_window, dtype=np.float64) / smoothing_window
        phase_dev = np.convolve(phase_dev, kernel, mode="same")

    phase_dev_masked = phase_dev.copy()
    phase_dev_masked[~valid_mask] = np.nan
    raw_phase_masked = raw_phase.copy()
    raw_phase_masked[~valid_mask] = np.nan
    trend_masked = trend.copy()
    trend_masked[~valid_mask] = np.nan
    times_sec = indices / fs

    return {
        "unwrapped_phase_rad": raw_phase_masked,
        "phase_trend_rad": trend_masked,
        "phase_deviation_rad": phase_dev_masked,
        "times_sec": times_sec,
        "valid_mask": valid_mask,
        "estimated_carrier_frequency_hz": carrier_freq,
        "constant_phase_rad": intercept_b,
        "valid_sample_fraction": valid_fraction,
        "sample_rate": fs,
    }


def _estimate_phase_deviation(
    phase_dev: np.ndarray,
    valid_mask: np.ndarray | None = None,
    use_robust_extrema: bool = False,
) -> Dict[str, float]:
    """
    Canonical PM phase deviation statistical moments.
    """
    if phase_dev.size == 0:
        return {
            "peak_deviation_rad": 0.0,
            "positive_deviation_rad": 0.0,
            "negative_deviation_rad": 0.0,
            "peak_to_peak_deviation_rad": 0.0,
            "rms_deviation_rad": 0.0,
            "phase_deviation_std_rad": 0.0,
            "peak_deviation_deg": 0.0,
        }

    if valid_mask is not None and np.any(valid_mask):
        act_phi = phase_dev[valid_mask]
    else:
        act_phi = phase_dev[np.isfinite(phase_dev)]

    if act_phi.size == 0:
        return {
            "peak_deviation_rad": 0.0,
            "positive_deviation_rad": 0.0,
            "negative_deviation_rad": 0.0,
            "peak_to_peak_deviation_rad": 0.0,
            "rms_deviation_rad": 0.0,
            "phase_deviation_std_rad": 0.0,
            "peak_deviation_deg": 0.0,
        }

    if use_robust_extrema and len(act_phi) >= 50:
        pos_phi = float(np.percentile(act_phi, 99.5))
        neg_phi = float(abs(np.percentile(act_phi, 0.5)))
    else:
        pos_phi = float(np.max(act_phi))
        neg_phi = float(abs(np.min(act_phi)))

    peak_phi = max(pos_phi, neg_phi)
    p2p_phi = float(np.max(act_phi) - np.min(act_phi))
    rms_phi = float(np.sqrt(np.mean(act_phi ** 2)))
    std_phi = float(np.std(act_phi))
    peak_deg = float(np.degrees(peak_phi))

    return {
        "peak_deviation_rad": peak_phi,
        "positive_deviation_rad": pos_phi,
        "negative_deviation_rad": neg_phi,
        "peak_to_peak_deviation_rad": p2p_phi,
        "rms_deviation_rad": rms_phi,
        "phase_deviation_std_rad": std_phi,
        "peak_deviation_deg": peak_deg,
    }


# ==============================================================================
# 7. DOMINANT MODULATION FREQUENCY HELPER
# ==============================================================================

def _dominant_modulation_frequency(
    waveform: np.ndarray,
    sample_rate: float = 1.0,
    valid_mask: np.ndarray | None = None,
    min_freq_hz: float | None = None,
    max_freq_hz: float | None = None,
    snr_threshold_db: float = 6.0,
) -> Dict[str, Any]:
    """
    Canonical dominant modulating tone / message frequency estimator.
    """
    fs = _validate_sample_rate(sample_rate)
    n = len(waveform)

    if n < 16:
        return {
            "frequency_hz": None,
            "peak_strength": 0.0,
            "noise_floor": 0.0,
            "snr_db": 0.0,
            "confidence": 0.0,
            "valid": False,
        }

    w = np.asarray(waveform, dtype=np.float64).copy()
    if valid_mask is not None:
        mask = valid_mask & np.isfinite(w)
    else:
        mask = np.isfinite(w)

    if not np.any(mask) or np.count_nonzero(mask) < 16:
        return {
            "frequency_hz": None,
            "peak_strength": 0.0,
            "noise_floor": 0.0,
            "snr_db": 0.0,
            "confidence": 0.0,
            "valid": False,
        }

    # Interpolate masked or invalid samples on the original time grid preserving array length N and Fs
    if not np.all(mask):
        indices = np.arange(n)
        valid_indices = indices[mask]
        valid_values = w[mask]
        if len(valid_indices) >= 2:
            w = np.interp(indices, valid_indices, valid_values)
        elif len(valid_values) > 0:
            w[~mask] = valid_values[0]

    w = w - np.mean(w)
    win = np.hanning(n)
    w_win = w * win
    w_sum = np.sum(win)

    nfft = max(256, _next_power_of_two(n * 2))
    fft_mag = np.abs(np.fft.rfft(w_win, n=nfft)) / max(1e-12, w_sum)
    fft_freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    df = fs / nfft

    if min_freq_hz is None:
        min_freq_hz = max(0.5 * df, 10.0)
    if max_freq_hz is None:
        max_freq_hz = min(0.20 * fs, 500e3)

    search_mask = (fft_freqs >= min_freq_hz) & (fft_freqs <= max_freq_hz)
    if not np.any(search_mask):
        return {
            "frequency_hz": None,
            "peak_strength": 0.0,
            "noise_floor": 0.0,
            "snr_db": 0.0,
            "confidence": 0.0,
            "valid": False,
        }

    search_mag = fft_mag.copy()
    search_mag[~search_mask] = 0.0

    peak_idx = int(np.argmax(search_mag))
    peak_val = float(search_mag[peak_idx])

    noise_floor = float(np.median(fft_mag[search_mask]))
    noise_floor = max(1e-20, noise_floor)
    snr_db = float(10.0 * np.log10(max(1e-18, (peak_val ** 2) / (noise_floor ** 2))))

    if snr_db < snr_threshold_db or peak_val <= 0:
        return {
            "frequency_hz": None,
            "peak_strength": peak_val,
            "noise_floor": noise_floor,
            "snr_db": snr_db,
            "confidence": 0.0,
            "valid": False,
        }

    if 0 < peak_idx < len(fft_mag) - 1:
        y0 = float(fft_mag[peak_idx - 1])
        y1 = float(fft_mag[peak_idx])
        y2 = float(fft_mag[peak_idx + 1])
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        f_est = float(fft_freqs[peak_idx] + delta * df)
        peak_amp = float(y1 - 0.25 * (y0 - y2) * delta)
    else:
        f_est = float(fft_freqs[peak_idx])
        peak_amp = peak_val

    conf = float(np.clip((snr_db - snr_threshold_db) / 20.0 + 0.5, 0.5, 1.0))

    return {
        "frequency_hz": f_est,
        "peak_strength": peak_amp,
        "noise_floor": noise_floor,
        "snr_db": snr_db,
        "confidence": conf,
        "valid": True,
    }


# ==============================================================================
# 8. STATE ESTIMATION PRIMITIVES (PURE NUMPY/SCIPY GMM + BIC)
# ==============================================================================

def _fit_gmm_1d(
    x: np.ndarray,
    k: int,
    max_iter: int = 40,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Fits 1D Gaussian Mixture Model via Expectation-Maximization in pure NumPy.
    Returns (means, variances, weights, bic, labels).
    """
    n = len(x)
    if k == 1:
        mean_val = float(np.mean(x))
        var_val = float(max(1e-6, np.var(x)))
        log_lh = -0.5 * n * (math.log(2.0 * math.pi * var_val) + 1.0)
        bic = 2.0 * math.log(n) - 2.0 * log_lh
        return np.array([mean_val]), np.array([var_val]), np.array([1.0]), bic, np.zeros(n, dtype=int)

    percentiles = np.linspace(10.0, 90.0, k)
    means = np.percentile(x, percentiles)
    variances = np.full(k, max(1e-6, np.var(x) / (k ** 2)))
    weights = np.full(k, 1.0 / k)

    prev_ll = -float("inf")

    for _ in range(max_iter):
        diff = x[:, None] - means[None, :]
        safe_vars = np.maximum(1e-8, variances)
        exponent = -0.5 * (diff ** 2) / safe_vars[None, :]
        log_prob = -0.5 * np.log(2.0 * np.pi * safe_vars[None, :]) + exponent
        max_lp = np.max(log_prob, axis=1, keepdims=True)
        prob = weights[None, :] * np.exp(np.clip(log_prob - max_lp, -50.0, 50.0))
        sum_prob = np.sum(prob, axis=1, keepdims=True) + 1e-18
        gamma = prob / sum_prob

        n_j = np.sum(gamma, axis=0) + 1e-12
        weights = n_j / n
        means = np.sum(gamma * x[:, None], axis=0) / n_j
        variances = np.sum(gamma * ((x[:, None] - means[None, :]) ** 2), axis=0) / n_j
        variances = np.maximum(1e-8, variances)

        ll = float(np.sum(np.log(np.maximum(1e-300, sum_prob)) + max_lp))
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    labels = np.argmax(gamma, axis=1)
    num_params = 3 * k - 1
    bic = num_params * math.log(n) - 2.0 * prev_ll

    return means, variances, weights, bic, labels


def _estimate_frequency_states(
    f_inst: np.ndarray,
    valid_mask: np.ndarray | None = None,
    k_max: int = 8,
    sample_rate: float = 1.0,
) -> Dict[str, Any]:
    """
    Fits 1-D Gaussian Mixture Models across K = 1..k_max to identify discrete frequency states (FSK).
    """
    if f_inst.size == 0:
        return {
            "num_frequency_states": 1,
            "state_frequencies_hz": np.array([0.0]),
            "state_std_hz": np.array([0.0]),
            "occupancies": np.array([1.0]),
            "labels": np.array([], dtype=int),
            "transition_matrix": np.eye(1),
            "minimum_state_separation": 0.0,
            "confidence": 0.0,
            "valid": False,
            "reason": "Empty frequency data.",
        }

    if valid_mask is not None and np.any(valid_mask):
        f_data = f_inst[valid_mask]
    else:
        f_data = f_inst[np.isfinite(f_inst)]

    n_samples = len(f_data)
    if n_samples < 32:
        med_f = float(np.median(f_data)) if n_samples > 0 else 0.0
        std_f = float(np.std(f_data)) if n_samples > 0 else 0.0
        return {
            "num_frequency_states": 1,
            "state_frequencies_hz": np.array([med_f]),
            "state_std_hz": np.array([std_f]),
            "occupancies": np.array([1.0]),
            "labels": np.zeros(n_samples, dtype=int),
            "transition_matrix": np.eye(1),
            "minimum_state_separation": 0.0,
            "confidence": 0.3,
            "valid": False,
            "reason": "Insufficient samples for discrete frequency state estimation.",
        }

    # Detect frequency transitions and exclude transient guard samples
    if n_samples >= 64:
        df = np.abs(np.diff(f_data))
        med_df = float(np.median(df))
        mad_df = float(np.median(np.abs(df - med_df)))
        th = med_df + 2.8 * max(1e-3, mad_df)

        trans_indices = np.where(df > th)[0]
        stable_mask = np.ones(n_samples, dtype=bool)
        guard = 4
        for t_idx in trans_indices:
            s_start = max(0, t_idx - guard)
            s_end = min(n_samples, t_idx + guard + 2)
            stable_mask[s_start:s_end] = False

        stable_data = f_data[stable_mask]
        if len(stable_data) < 32:
            stable_data = f_data
    else:
        stable_data = f_data

    max_k = min(int(k_max), max(1, len(stable_data) // 16))
    best_bic = float("inf")
    best_means = np.array([np.mean(stable_data)])
    best_vars = np.array([np.var(stable_data)])
    best_weights = np.array([1.0])
    best_k = 1

    for k in range(1, max_k + 1):
        try:
            m, v, w, bic, _ = _fit_gmm_1d(stable_data, k=k)
            min_w = float(np.min(w))
            if min_w < (0.20 / k):
                continue
            if k > 1:
                sort_m = np.sort(m)
                sort_s = np.sqrt(v[np.argsort(m)])
                sep_k = min(
                    abs(sort_m[i + 1] - sort_m[i]) / math.sqrt(sort_s[i] ** 2 + sort_s[i + 1] ** 2 + 1e-12)
                    for i in range(k - 1)
                )
                if sep_k < 0.65:
                    continue
            if bic < best_bic - 20.0:
                best_bic = bic
                best_means = m
                best_vars = v
                best_weights = w
                best_k = k
        except Exception:
            continue

    # Assign all samples f_data to nearest state
    diffs_all = np.abs(f_data[:, None] - best_means[None, :])
    best_labels = np.argmin(diffs_all, axis=1)

    sort_idx = np.argsort(best_means)
    sorted_means = best_means[sort_idx]
    sorted_stds = np.sqrt(np.maximum(1e-12, best_vars[sort_idx]))
    sorted_weights = best_weights[sort_idx]

    label_remap = {orig: new for new, orig in enumerate(sort_idx)}
    remapped_labels = np.array([label_remap[lbl] for lbl in best_labels], dtype=int)

    # Transition matrix
    trans_mat = np.zeros((best_k, best_k), dtype=np.float64)
    for i in range(len(remapped_labels) - 1):
        trans_mat[remapped_labels[i], remapped_labels[i + 1]] += 1.0
    row_sums = trans_mat.sum(axis=1, keepdims=True)
    trans_mat = np.divide(trans_mat, np.maximum(1.0, row_sums), where=row_sums > 0)

    # State separation S_ij = |mu_i - mu_j| / sqrt(sigma_i^2 + sigma_j^2)
    if best_k > 1:
        min_sep = float("inf")
        for i in range(best_k):
            for j in range(i + 1, best_k):
                d_mu = abs(sorted_means[i] - sorted_means[j])
                d_sig = math.sqrt(sorted_stds[i] ** 2 + sorted_stds[j] ** 2 + 1e-12)
                sep = d_mu / d_sig
                if sep < min_sep:
                    min_sep = sep
    else:
        min_sep = 0.0

    valid = bool(best_k > 1 and min_sep >= 1.2 and np.all(sorted_weights >= (0.25 / best_k)))
    conf = float(np.clip(min_sep / 4.0, 0.4, 0.95)) if valid else 0.5

    return {
        "num_frequency_states": best_k,
        "state_frequencies_hz": sorted_means,
        "state_std_hz": sorted_stds,
        "occupancies": sorted_weights,
        "labels": remapped_labels,
        "transition_matrix": trans_mat,
        "minimum_state_separation": min_sep if math.isfinite(min_sep) else 0.0,
        "confidence": conf,
        "valid": valid or best_k == 1,
        "reason": f"Detected {best_k} discrete frequency states." if best_k > 1 else "Continuous or single-state frequency distribution.",
    }


def _estimate_phase_states(
    symbol_points: np.ndarray,
    k_candidates: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
) -> Dict[str, Any]:
    """
    Fits circular phase clustering to determine discrete phase states (BPSK/QPSK/8PSK/16PSK).
    """
    n_syms = len(symbol_points)
    if n_syms < 16:
        return {
            "num_phase_states": 1,
            "state_angles_rad": np.array([0.0]),
            "state_angles_deg": np.array([0.0]),
            "occupancies": np.array([1.0]),
            "concentrations": np.array([1.0]),
            "labels": np.zeros(n_syms, dtype=int),
            "symbol_points": symbol_points,
            "model_scores": {1: 0.0},
            "confidence": 0.0,
            "valid": False,
            "reason": "Insufficient symbol points for phase state estimation.",
        }

    angles = np.angle(symbol_points)
    model_scores: Dict[int, float] = {}
    k_evals = {}

    for k in k_candidates:
        if k > n_syms // 4:
            continue
        if k == 1:
            sin_s = np.sum(np.sin(angles))
            cos_s = np.sum(np.cos(angles))
            r = math.sqrt(sin_s ** 2 + cos_s ** 2) / n_syms
            model_scores[1] = float(r)
            k_evals[1] = {
                "conc": r,
                "min_occ": 1.0,
                "labels": np.zeros(n_syms, dtype=int),
                "counts": [n_syms],
                "valid": True,
            }
            continue

        n_grid = max(24, k * 3)
        offsets = np.linspace(0, 2 * np.pi / k, n_grid, endpoint=False)
        best_off_r = -1.0
        best_lbls = np.zeros(n_syms, dtype=int)
        best_counts = [0] * k

        for off in offsets:
            grid = (np.linspace(0, 2 * np.pi, k, endpoint=False) + off + np.pi) % (2.0 * np.pi) - np.pi
            diffs = (angles[:, None] - grid[None, :] + np.pi) % (2.0 * np.pi) - np.pi
            lbls = np.argmin(np.abs(diffs), axis=1)

            concs = []
            counts = []
            for j in range(k):
                mask = (lbls == j)
                cnt = int(np.sum(mask))
                counts.append(cnt)
                if cnt > 0:
                    s = np.sum(np.sin(angles[mask]))
                    c = np.sum(np.cos(angles[mask]))
                    concs.append(math.sqrt(s ** 2 + c ** 2) / cnt)
                else:
                    concs.append(0.0)

            min_occ = min(counts) / n_syms
            avg_conc = float(np.mean(concs))

            if min_occ < (0.35 / k):
                avg_conc *= 0.1

            if avg_conc > best_off_r:
                best_off_r = avg_conc
                best_lbls = lbls
                best_counts = counts

        min_occ_final = min(best_counts) / n_syms
        is_k_valid = bool(min_occ_final >= (0.35 / k) and best_off_r >= 0.70)
        model_scores[k] = float(best_off_r)
        k_evals[k] = {
            "conc": best_off_r,
            "min_occ": min_occ_final,
            "labels": best_lbls,
            "counts": best_counts,
            "valid": is_k_valid,
        }

    # Hierarchical Sub-Harmonic Model Selection for Discrete Phase States
    c2 = model_scores.get(2, 0.0)
    c4 = model_scores.get(4, 0.0)
    c8 = model_scores.get(8, 0.0)
    c16 = model_scores.get(16, 0.0)

    # 16PSK: requires high c16 (>=0.990) and (c16 - c8) >= 0.012 and c8 >= 0.94
    if 16 in k_candidates and c16 >= 0.990 and (c16 - c8) >= 0.012 and c8 >= 0.94:
        chosen_k = 16
    # 8PSK: requires c8 >= 0.965 and c8 > c4 + 0.035
    elif 8 in k_candidates and c8 >= 0.965 and (c8 - c4) >= 0.035:
        chosen_k = 8
    # QPSK: requires c4 >= 0.94 and c4 > c2 + 0.08
    elif 4 in k_candidates and c4 >= 0.94 and (c4 - c2) >= 0.08:
        chosen_k = 4
    # BPSK: requires c2 >= 0.80
    elif 2 in k_candidates and c2 >= 0.80:
        chosen_k = 2
    else:
        chosen_k = 1

    selected_info = k_evals[chosen_k]
    selected_lbls = selected_info["labels"]

    # Compute circular state statistics
    state_angles = []
    state_concs = []
    state_counts = []

    for k_idx in range(chosen_k):
        k_mask = (selected_lbls == k_idx)
        n_k = int(np.sum(k_mask))
        state_counts.append(n_k)
        if n_k > 0:
            k_ang = angles[k_mask]
            sin_s = np.sum(np.sin(k_ang))
            cos_s = np.sum(np.cos(k_ang))
            mean_ang = math.atan2(sin_s, cos_s)
            r_conc = math.sqrt(sin_s ** 2 + cos_s ** 2) / n_k
            state_angles.append(mean_ang)
            state_concs.append(r_conc)
        else:
            state_angles.append(0.0)
            state_concs.append(0.0)

    state_angles_arr = np.array(state_angles)
    sort_idx = np.argsort(state_angles_arr)
    sorted_angles = state_angles_arr[sort_idx]
    sorted_concs = np.array(state_concs)[sort_idx]
    sorted_counts = np.array(state_counts)[sort_idx]
    occupancies = sorted_counts / max(1, np.sum(sorted_counts))

    label_remap = {orig: new for new, orig in enumerate(sort_idx)}
    remapped_labels = np.array([label_remap[lbl] for lbl in selected_lbls], dtype=int)

    avg_conc = float(np.mean(sorted_concs)) if len(sorted_concs) > 0 else 0.0
    valid = bool(chosen_k in (2, 4, 8, 16) and avg_conc >= 0.70)
    conf = float(np.clip(avg_conc, 0.4, 0.98)) if valid else 0.5

    return {
        "num_phase_states": chosen_k,
        "state_angles_rad": sorted_angles,
        "state_angles_deg": np.degrees(sorted_angles),
        "occupancies": occupancies,
        "concentrations": sorted_concs,
        "labels": remapped_labels,
        "symbol_points": symbol_points,
        "model_scores": model_scores,
        "confidence": conf,
        "valid": valid or chosen_k == 1,
        "reason": f"Identified {chosen_k} discrete phase states with mean concentration {avg_conc:.2f}." if valid else "Continuous or diffuse phase distribution.",
    }


# ==============================================================================
# 9. CLASSIFICATION FEATURE EXTRACTOR
# ==============================================================================

def _extract_modulation_features(iq_data: Any, sample_rate: float = 1.0) -> Dict[str, float]:
    """
    Canonical feature extraction pipeline for Modulation Type Classification.
    """
    z = _as_iq_array(iq_data)
    fs = _validate_sample_rate(sample_rate)

    am_info = _analyze_envelope(z, sample_rate=fs)
    env = am_info["envelope"]
    env_mean = am_info["envelope_mean"]
    env_std = am_info["envelope_std"]
    env_cv = float(env_std / max(1e-12, env_mean))
    am_activity = am_info["am_activity_score"]
    am_depth_info = _estimate_am_depth(env, estimator="robust", valid_mask=am_info["valid_mask"])
    am_depth = float(am_depth_info["modulation_depth"] or 0.0)

    # Use mild smoothing on instantaneous frequency to prevent differentiated noise explosions
    fm_info = _instantaneous_frequency(z, sample_rate=fs, smoothing="moving_average", smoothing_window=15)
    f_inst = fm_info["instantaneous_frequency_hz"]
    f0, f_dev = _center_frequency_deviation(f_inst, valid_mask=fm_info["valid_mask"])
    fm_dev_info = _estimate_frequency_deviation(f_dev, valid_mask=fm_info["valid_mask"])
    fm_peak_dev = fm_dev_info["peak_deviation_hz"]
    fm_rms_dev = fm_dev_info["rms_deviation_hz"]
    fm_std_dev = fm_dev_info["std_deviation_hz"]
    fm_activity = float(fm_rms_dev / max(1.0, fs * 0.01))

    f_state_info = _estimate_frequency_states(f_inst, valid_mask=fm_info["valid_mask"], sample_rate=fs)
    num_f_states = float(f_state_info["num_frequency_states"])
    f_sep = float(f_state_info["minimum_state_separation"])

    pm_info = _detrended_phase(z, sample_rate=fs)
    phi_dev = pm_info["phase_deviation_rad"]
    pm_dev_info = _estimate_phase_deviation(phi_dev, valid_mask=pm_info["valid_mask"])
    pm_peak_dev = pm_dev_info["peak_deviation_rad"]
    pm_rms_dev = pm_dev_info["rms_deviation_rad"]
    pm_std_dev = pm_dev_info["phase_deviation_std_rad"]

    spec_cent = _spectral_centroid(z, sample_rate=fs)
    spec_var = _spectral_variance(z, sample_rate=fs)
    spec_skew = _spectral_skewness(z, sample_rate=fs)
    spec_kurt = _spectral_kurtosis(z, sample_rate=fs)
    spec_entropy = _spectral_entropy(z, sample_rate=fs)
    spec_flatness = _spectral_flatness(z, sample_rate=fs)
    obw = _occupied_bandwidth(z, sample_rate=fs)

    tf_entropy_vec = _time_frequency_entropy(z, sample_rate=fs)
    tf_entropy_mean = float(np.mean(tf_entropy_vec)) if len(tf_entropy_vec) > 0 else 0.0
    ibw_vec = _instantaneous_bandwidth(z, sample_rate=fs)
    ibw_mean = float(np.mean(ibw_vec)) if len(ibw_vec) > 0 else 0.0

    cyclo_info = _validated_cyclic_frequency_candidates(z, sample_rate=fs)
    cyclo_detected = 1.0 if cyclo_info["cyclostationary_detected"] else 0.0
    cyclo_snr = float(cyclo_info["raw_cyclic_snr_db"])
    cyclo_rate = float(cyclo_info["raw_primary_rate_hz"])

    pwr = float(np.mean(_complex_power(z)))
    fourth_moment = float(np.mean(np.abs(z) ** 4) / max(1e-12, pwr ** 2)) if pwr > 0 else 0.0
    circularity = float(np.abs(np.mean(z ** 2)) / max(1e-12, pwr)) if pwr > 0 else 0.0

    return {
        "envelope_mean": env_mean,
        "envelope_std": env_std,
        "envelope_cv": env_cv,
        "am_activity": am_activity,
        "am_depth": am_depth,
        "fm_peak_deviation_hz": fm_peak_dev,
        "fm_rms_deviation_hz": fm_rms_dev,
        "fm_std_deviation_hz": fm_std_dev,
        "fm_activity": fm_activity,
        "num_frequency_states": num_f_states,
        "frequency_state_separation": f_sep,
        "pm_peak_deviation_rad": pm_peak_dev,
        "pm_rms_deviation_rad": pm_rms_dev,
        "pm_std_deviation_rad": pm_std_dev,
        "spectral_centroid_hz": spec_cent,
        "spectral_spread_hz": spec_var,
        "spectral_skewness": spec_skew,
        "spectral_kurtosis": spec_kurt,
        "spectral_entropy": spec_entropy,
        "spectral_flatness": spec_flatness,
        "occupied_bandwidth_hz": obw,
        "tf_entropy_mean": tf_entropy_mean,
        "instantaneous_bandwidth_hz": ibw_mean,
        "cyclo_detected": cyclo_detected,
        "cyclo_snr_db": cyclo_snr,
        "cyclo_rate_hz": cyclo_rate,
        "normalized_fourth_moment": fourth_moment,
        "circularity_coefficient": circularity,
    }
