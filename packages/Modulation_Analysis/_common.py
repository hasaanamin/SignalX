from __future__ import annotations
from typing import Any, Tuple
import math
import numpy as np


def _as_iq_array(data: Any) -> np.ndarray:
    """
    Standardizes input into a 1D complex128 NumPy array.
    Supports complex ndarray, tuple/list of (I, Q), canonical dictionary, or structured arrays.
    Validates finite values where appropriate (handles NaNs gracefully).
    """
    if data is None:
        raise ValueError("Input IQ data cannot be None.")

    if hasattr(data, "iq_samples"):
        return _as_iq_array(data.iq_samples)

    arr = None
    if isinstance(data, dict):
        if "iq" in data and data["iq"] is not None:
            return _as_iq_array(data["iq"])
        if "I" in data and "Q" in data:
            i_arr = np.asarray(data["I"], dtype=np.float64).ravel()
            q_arr = np.asarray(data["Q"], dtype=np.float64).ravel()
            min_len = min(len(i_arr), len(q_arr))
            arr = (i_arr[:min_len] + 1j * q_arr[:min_len]).astype(np.complex128)
        elif "real" in data and "imag" in data:
            r_arr = np.asarray(data["real"], dtype=np.float64).ravel()
            im_arr = np.asarray(data["imag"], dtype=np.float64).ravel()
            min_len = min(len(r_arr), len(im_arr))
            arr = (r_arr[:min_len] + 1j * im_arr[:min_len]).astype(np.complex128)
        else:
            raise ValueError(f"Dictionary missing recognized IQ keys ('iq', 'I'/'Q', 'real'/'imag'). Found: {list(data.keys())}")

    elif isinstance(data, (tuple, list)):
        if len(data) == 2 and not np.iscomplexobj(data[0]):
            try:
                i_arr = np.asarray(data[0], dtype=np.float64).ravel()
                q_arr = np.asarray(data[1], dtype=np.float64).ravel()
                min_len = min(len(i_arr), len(q_arr))
                arr = (i_arr[:min_len] + 1j * q_arr[:min_len]).astype(np.complex128)
            except Exception:
                pass
        if arr is None:
            raw = np.asarray(data)
            if raw.dtype.kind == "V":
                names = raw.dtype.names or ()
                if "I" in names and "Q" in names:
                    arr = (raw["I"].astype(np.float64) + 1j * raw["Q"].astype(np.float64)).ravel().astype(np.complex128)
            elif np.iscomplexobj(raw):
                arr = raw.astype(np.complex128).ravel()
            elif raw.ndim == 2 and raw.shape[1] == 2:
                arr = (raw[:, 0].astype(np.float64) + 1j * raw[:, 1].astype(np.float64)).ravel().astype(np.complex128)
            elif raw.ndim == 2 and raw.shape[0] == 2:
                arr = (raw[0, :].astype(np.float64) + 1j * raw[1, :].astype(np.float64)).ravel().astype(np.complex128)
            else:
                arr = raw.astype(np.complex128).ravel()

    elif isinstance(data, np.ndarray):
        if data.dtype.kind == "V":
            names = data.dtype.names or ()
            if "I" in names and "Q" in names:
                arr = (data["I"].astype(np.float64) + 1j * data["Q"].astype(np.float64)).ravel().astype(np.complex128)
        elif np.iscomplexobj(data):
            arr = data.astype(np.complex128).ravel()
        elif data.ndim == 2 and data.shape[1] == 2:
            arr = (data[:, 0].astype(np.float64) + 1j * data[:, 1].astype(np.float64)).ravel().astype(np.complex128)
        elif data.ndim == 2 and data.shape[0] == 2:
            arr = (data[0, :].astype(np.float64) + 1j * data[1, :].astype(np.float64)).ravel().astype(np.complex128)
        else:
            arr = data.astype(np.complex128).ravel()

    if arr is None:
        raise TypeError(f"Unsupported IQ data type '{type(data).__name__}'.")

    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError("Input IQ data contains NaN or Inf values.")

    return arr


def _validate_sample_rate(sample_rate: float) -> float:
    """Validates that sample rate is a positive finite float."""
    fs = float(sample_rate)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Sample rate must be a positive finite float, got {sample_rate}")
    return fs


def _extract_sample_rate(
    data: Any = None,
    kwargs: dict[str, Any] | None = None,
    sample_rate: float | None = None,
    default_fs: float = 1.0,
) -> float:
    """Extracts a positive finite sample rate in Hz."""
    if sample_rate is not None:
        return _validate_sample_rate(sample_rate)

    if kwargs:
        for key in ["sample_rate", "fs", "Fs", "sampling_rate", "sample_rate_hz"]:
            if key in kwargs and kwargs[key] is not None:
                return _validate_sample_rate(kwargs[key])

    if isinstance(data, dict):
        for key in ["sample_rate", "fs", "Fs", "sampling_rate", "sample_rate_hz"]:
            if key in data and data[key] is not None:
                try:
                    return _validate_sample_rate(data[key])
                except (TypeError, ValueError):
                    pass

    return float(default_fs)


def _safe_divide(numerator: np.ndarray | float, denominator: np.ndarray | float, eps: float = 1e-18, default: float = 0.0) -> np.ndarray | float:
    """Safely divides numerator by denominator with epsilon floor or zero substitution."""
    if isinstance(numerator, np.ndarray) or isinstance(denominator, np.ndarray):
        num = np.asarray(numerator, dtype=np.float64)
        den = np.asarray(denominator, dtype=np.float64)
        out = np.full_like(num, default, dtype=np.float64)
        valid = np.abs(den) > eps
        out[valid] = num[valid] / den[valid]
        return out
    else:
        den = float(denominator)
        if abs(den) <= eps:
            return float(default)
        return float(numerator) / den


def _safe_log(x: np.ndarray | float, base: float = math.e, eps: float = 1e-300) -> np.ndarray | float:
    """Safely computes logarithm with positive clipping."""
    if isinstance(x, np.ndarray):
        clipped = np.maximum(eps, np.asarray(x, dtype=np.float64))
        if base == 2.0:
            return np.log2(clipped)
        elif base == 10.0:
            return np.log10(clipped)
        return np.log(clipped)
    else:
        val = max(eps, float(x))
        if base == 2.0:
            return math.log2(val)
        elif base == 10.0:
            return math.log10(val)
        return math.log(val)


def _safe_percentile(arr: np.ndarray, q: float, default: float = 0.0) -> float:
    """Computes percentile safely over valid finite array."""
    if arr.size == 0:
        return float(default)
    finite_arr = arr[np.isfinite(arr)]
    if finite_arr.size == 0:
        return float(default)
    return float(np.percentile(finite_arr, q))


def _robust_median_mad(arr: np.ndarray) -> Tuple[float, float]:
    """Computes median and Median Absolute Deviation (MAD * 1.4826) safely."""
    if arr.size == 0:
        return 0.0, 0.0
    finite_arr = arr[np.isfinite(arr)]
    if finite_arr.size == 0:
        return 0.0, 0.0
    med = float(np.median(finite_arr))
    mad = float(np.median(np.abs(finite_arr - med))) * 1.482602218505602
    return med, mad


def _robust_scale(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Centers by median and scales by MAD."""
    if arr.size == 0:
        return arr.copy()
    med, mad = _robust_median_mad(arr)
    scale = max(eps, mad)
    return (arr - med) / scale


def _complex_power(z: np.ndarray) -> np.ndarray:
    """Computes instantaneous linear power |z|^2."""
    if z.size == 0:
        return np.array([], dtype=np.float64)
    return np.real(z * np.conj(z))


def _amplitude_valid_mask(
    iq: np.ndarray,
    threshold_db: float = -60.0,
    min_active_samples: int = 1,
) -> np.ndarray:
    """
    Computes boolean mask of reliable IQ samples based on relative power threshold.
    Masks regions where |x[n]| falls below threshold relative to median/peak power.
    """
    n = len(iq)
    if n == 0:
        return np.array([], dtype=bool)

    power = _complex_power(iq)
    finite_mask = np.isfinite(power)
    if not np.any(finite_mask):
        return np.zeros(n, dtype=bool)

    valid_power = power[finite_mask]
    med_pwr = float(np.median(valid_power))
    max_pwr = float(np.max(valid_power))
    ref_pwr = max(med_pwr, max_pwr * 0.01)

    if ref_pwr <= 1e-24:
        # All zeros or deep silence
        return np.zeros(n, dtype=bool)

    thresh_linear = ref_pwr * (10.0 ** (threshold_db / 10.0))
    active_mask = (power >= thresh_linear) & finite_mask

    # Short notch bridging: bridge 1-2 sample isolated nulls
    if np.sum(active_mask) >= min_active_samples:
        for i in range(1, n - 1):
            if not active_mask[i] and active_mask[i - 1] and active_mask[i + 1]:
                active_mask[i] = True

    return active_mask


def _circular_difference(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """Computes principal circular angle difference in [-pi, pi)."""
    diff = a - b
    return (diff + np.pi) % (2.0 * np.pi) - np.pi


def _circular_mean(angles: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Computes mean angle on unit circle."""
    if len(angles) == 0:
        return 0.0
    if weights is not None:
        sin_sum = np.sum(weights * np.sin(angles))
        cos_sum = np.sum(weights * np.cos(angles))
    else:
        sin_sum = np.sum(np.sin(angles))
        cos_sum = np.sum(np.cos(angles))
    return float(np.arctan2(sin_sum, cos_sum))


def _safe_rms_normalize(z: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
    """RMS normalizes complex vector to unit average power."""
    if z.size == 0:
        return z.copy(), 1.0
    pwr = float(np.mean(_complex_power(z)))
    rms = math.sqrt(max(0.0, pwr))
    scale = max(eps, rms)
    return z / scale, scale
