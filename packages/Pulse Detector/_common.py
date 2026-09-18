from __future__ import annotations
from typing import Any
import numpy as np
from scipy.ndimage import median_filter


def _validate_sample_rate(fs: Any) -> float:
    """
    Validates that sample rate is finite, real, and strictly greater than zero.
    Raises ValueError otherwise.
    """
    try:
        fs_val = float(fs)
    except (TypeError, ValueError):
        raise ValueError(f"sample_rate must be finite and greater than zero, got {fs!r}")

    if not np.isfinite(fs_val) or fs_val <= 0:
        raise ValueError(f"sample_rate must be finite and greater than zero, got {fs_val}")
    return fs_val


def _as_iq_array(data: Any) -> np.ndarray:
    """
    Standardizes input into a 1D complex128 NumPy array.
    Supports complex ndarray, tuple/list of (I, Q), or canonical dictionary.
    Enforces that separate I and Q arrays have identical lengths without silent truncation.
    """
    if data is None:
        raise ValueError("Input IQ data cannot be None.")

    if isinstance(data, dict):
        if "iq" in data and data["iq"] is not None:
            return _as_iq_array(data["iq"])
        if "I" in data and "Q" in data:
            i_arr = np.asarray(data["I"], dtype=np.float64).ravel()
            q_arr = np.asarray(data["Q"], dtype=np.float64).ravel()
            if len(i_arr) != len(q_arr):
                raise ValueError(f"I/Q length mismatch: I={len(i_arr)}, Q={len(q_arr)}")
            return i_arr + 1j * q_arr
        if "real" in data and "imag" in data:
            r_arr = np.asarray(data["real"], dtype=np.float64).ravel()
            im_arr = np.asarray(data["imag"], dtype=np.float64).ravel()
            if len(r_arr) != len(im_arr):
                raise ValueError(f"I/Q length mismatch: real={len(r_arr)}, imag={len(im_arr)}")
            return r_arr + 1j * im_arr
        raise ValueError(f"Dictionary missing recognized IQ keys ('iq', 'I'/'Q', 'real'/'imag'). Found: {list(data.keys())}")

    if isinstance(data, (tuple, list)):
        if len(data) == 2 and not np.iscomplexobj(data[0]):
            try:
                i_arr = np.asarray(data[0], dtype=np.float64).ravel()
                q_arr = np.asarray(data[1], dtype=np.float64).ravel()
                if len(i_arr) != len(q_arr):
                    raise ValueError(f"I/Q length mismatch: I={len(i_arr)}, Q={len(q_arr)}")
                return i_arr + 1j * q_arr
            except ValueError:
                raise
            except Exception:
                pass
        arr = np.asarray(data)
        if arr.dtype.kind == "V":
            names = arr.dtype.names or ()
            if "I" in names and "Q" in names:
                i_arr = arr["I"].astype(np.float64).ravel()
                q_arr = arr["Q"].astype(np.float64).ravel()
                if len(i_arr) != len(q_arr):
                    raise ValueError(f"I/Q length mismatch: I={len(i_arr)}, Q={len(q_arr)}")
                return i_arr + 1j * q_arr
        if np.iscomplexobj(arr):
            return arr.astype(np.complex128).ravel()
        if arr.ndim == 2 and arr.shape[1] == 2:
            return (arr[:, 0].astype(np.float64) + 1j * arr[:, 1].astype(np.float64)).ravel()
        if arr.ndim == 2 and arr.shape[0] == 2:
            return (arr[0, :].astype(np.float64) + 1j * arr[1, :].astype(np.float64)).ravel()
        return arr.astype(np.complex128).ravel()

    if isinstance(data, np.ndarray):
        if data.dtype.kind == "V":
            names = data.dtype.names or ()
            if "I" in names and "Q" in names:
                i_arr = data["I"].astype(np.float64).ravel()
                q_arr = data["Q"].astype(np.float64).ravel()
                if len(i_arr) != len(q_arr):
                    raise ValueError(f"I/Q length mismatch: I={len(i_arr)}, Q={len(q_arr)}")
                return i_arr + 1j * q_arr
        if np.iscomplexobj(data):
            return data.astype(np.complex128).ravel()
        if data.ndim == 2 and data.shape[1] == 2:
            return (data[:, 0].astype(np.float64) + 1j * data[:, 1].astype(np.float64)).ravel()
        if data.ndim == 2 and data.shape[0] == 2:
            return (data[0, :].astype(np.float64) + 1j * data[1, :].astype(np.float64)).ravel()
        return data.astype(np.complex128).ravel()

    raise TypeError(f"Unsupported IQ data type '{type(data).__name__}'.")


def _extract_sample_rate(kwargs: dict[str, Any], default: float | None = None) -> float:
    """Extracts and validates sample rate parameter from kwargs."""
    for key in ["sample_rate", "fs", "Fs", "sampling_rate", "sample_rate_hz"]:
        if key in kwargs and kwargs[key] is not None:
            return _validate_sample_rate(kwargs[key])
    if default is not None:
        return _validate_sample_rate(default)
    raise ValueError("sample_rate must be specified and strictly positive.")


def _causal_moving_average(x: np.ndarray, L: int, history: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes a causal moving average filter of window length L:
      bar{P}[n] = (1 / L) * sum_{k=0}^{L-1} P[n-k]
    Maintains streaming filter history of (L - 1) power samples across successive processing chunks.

    Parameters:
        x: 1D real power array.
        L: Filter window length.
        history: Optional history array of up to (L - 1) previous power samples.

    Returns:
        (smoothed_x, new_history)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if L <= 1 or x.size == 0:
        new_hist = x[-max(0, L - 1):] if L > 1 and x.size > 0 else np.array([], dtype=np.float64)
        return x.copy(), new_hist

    L = int(L)
    n_hist_needed = L - 1

    if history is not None and len(history) > 0:
        hist_clean = np.asarray(history, dtype=np.float64).ravel()[-n_hist_needed:]
    else:
        # For initial startup, pad with initial sample to avoid cold-start attenuation
        hist_clean = np.full(n_hist_needed, x[0], dtype=np.float64)

    extended = np.concatenate([hist_clean, x])
    cumsum = np.cumsum(extended, dtype=np.float64)
    # Output at index i corresponds to sum over window of length L divided by L
    # extended has length len(hist_clean) + len(x)
    # The first output sample corresponding to x[0] is extended[n_hist_needed - (L-1) : n_hist_needed + 1] -> length L
    out = (cumsum[n_hist_needed:] - np.concatenate([[0.0], cumsum[:len(x) - 1]])) / float(L)

    new_history = extended[-n_hist_needed:].copy() if n_hist_needed > 0 else np.array([], dtype=np.float64)
    return out, new_history


def _moving_median(x: np.ndarray, W: int) -> np.ndarray:
    """
    Computes robust moving median over local window W:
      N[n] = median{P_env[n - W/2 : n + W/2]}
    Uses constant baseline padding to prevent boundary edge replication inflation.
    Equivalent to MATLAB `movmedian(Penv, W)`.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if W <= 1 or x.size <= 1:
        return x.copy()
    W = int(min(W, x.size))
    if W % 2 == 0:
        W -= 1
    W = max(1, W)
    if W <= 1:
        return x.copy()
    cval = float(np.median(x))
    return median_filter(x, size=W, mode="constant", cval=cval)


def _moving_mad(x: np.ndarray, W: int, noise_floor: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimates background variation robustly using Median Absolute Deviation (MAD):
      MAD[n] = median(|x[n] - N[n]|, W)
      sigma_N[n] = 1.4826 * MAD[n]
    Equivalent to MATLAB:
      madNoise = movmedian(abs(Penv - noiseFloor), W);
      sigmaN = 1.4826 * madNoise;
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if noise_floor is None:
        noise_floor = _moving_median(x, W)
    abs_dev = np.abs(x - noise_floor)
    mad = _moving_median(abs_dev, W)
    # 1.4826 factor scales MAD to normal standard deviation
    sigma_n = 1.4826 * np.maximum(1e-12, mad)
    return mad, sigma_n

