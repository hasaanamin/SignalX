from __future__ import annotations
from typing import Any
import math
import numpy as np

from .base import (
    CorrelationResult,
    TemplateMatchResult,
    SequenceSimilarityResult,
)


def _as_iq_array(data: Any) -> np.ndarray:
    """
    Standardizes input into a 1D complex128 NumPy array.
    Supports complex ndarray, tuple/list of (I, Q), canonical dictionary, or pulse candidates.
    """
    if data is None:
        raise ValueError("Input IQ data cannot be None.")

    if isinstance(data, CorrelationResult):
        raise TypeError("Expected raw IQ data, but received CorrelationResult.")

    if hasattr(data, "iq_samples"):
        # PulseCandidate object from Pulse Detector
        return _as_iq_array(data.iq_samples)

    if isinstance(data, dict):
        if "iq" in data and data["iq"] is not None:
            return _as_iq_array(data["iq"])
        if "I" in data and "Q" in data:
            i_arr = np.asarray(data["I"], dtype=np.float64).ravel()
            q_arr = np.asarray(data["Q"], dtype=np.float64).ravel()
            min_len = min(len(i_arr), len(q_arr))
            return i_arr[:min_len] + 1j * q_arr[:min_len]
        if "real" in data and "imag" in data:
            r_arr = np.asarray(data["real"], dtype=np.float64).ravel()
            im_arr = np.asarray(data["imag"], dtype=np.float64).ravel()
            min_len = min(len(r_arr), len(im_arr))
            return r_arr[:min_len] + 1j * im_arr[:min_len]
        raise ValueError(f"Dictionary missing recognized IQ keys ('iq', 'I'/'Q', 'real'/'imag'). Found: {list(data.keys())}")

    if isinstance(data, (tuple, list)):
        if len(data) == 2 and not np.iscomplexobj(data[0]):
            try:
                i_arr = np.asarray(data[0], dtype=np.float64).ravel()
                q_arr = np.asarray(data[1], dtype=np.float64).ravel()
                min_len = min(len(i_arr), len(q_arr))
                return i_arr[:min_len] + 1j * q_arr[:min_len]
            except Exception:
                pass
        arr = np.asarray(data)
        if arr.dtype.kind == "V":
            names = arr.dtype.names or ()
            if "I" in names and "Q" in names:
                return (arr["I"].astype(np.float64) + 1j * arr["Q"].astype(np.float64)).ravel()
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
                return (data["I"].astype(np.float64) + 1j * data["Q"].astype(np.float64)).ravel()
        if np.iscomplexobj(data):
            return data.astype(np.complex128).ravel()
        if data.ndim == 2 and data.shape[1] == 2:
            return (data[:, 0].astype(np.float64) + 1j * data[:, 1].astype(np.float64)).ravel()
        if data.ndim == 2 and data.shape[0] == 2:
            return (data[0, :].astype(np.float64) + 1j * data[1, :].astype(np.float64)).ravel()
        return data.astype(np.complex128).ravel()

    raise TypeError(f"Unsupported IQ data type '{type(data).__name__}'.")


def _extract_sample_rate(
    data: Any = None,
    kwargs: dict[str, Any] | None = None,
    sample_rate: float | None = None,
    default_fs: float = 1.0,
) -> float:
    """
    Extracts a positive, finite sampling rate (Fs) in Hz.
    """
    if sample_rate is not None:
        fs = float(sample_rate)
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError(f"Sample rate must be finite and > 0, got {sample_rate}")
        return fs

    if kwargs:
        for key in ["sample_rate", "fs", "Fs", "sampling_rate", "sample_rate_hz"]:
            if key in kwargs and kwargs[key] is not None:
                fs = float(kwargs[key])
                if not np.isfinite(fs) or fs <= 0:
                    raise ValueError(f"Sample rate must be finite and > 0, got {kwargs[key]}")
                return fs

    if isinstance(data, dict):
        for key in ["sample_rate", "fs", "Fs", "sampling_rate", "sample_rate_hz"]:
            if key in data and data[key] is not None:
                try:
                    fs = float(data[key])
                    if np.isfinite(fs) and fs > 0:
                        return fs
                except (TypeError, ValueError):
                    pass

    return float(default_fs)


def _compute_xcorr_core(
    x_in: Any,
    y_in: Any = None,
    *,
    maxlag: int | None = None,
    scaleopt: str = "coeff",
    method: str = "auto",
    sample_rate: float = 1.0,
    eps: float = 1e-18,
) -> CorrelationResult:
    """
    Unified Cross-Correlation / Autocorrelation Engine matching MATLAB `xcorr(x, y, maxlag, scaleopt)`.

    MATLAB Mathematical Definition:
      R_xy[m] = sum_n x[n + m] * y^*[n]
      Negative lag symmetry: R_xy[-m] = R_yx^*[m]
      Autocorrelation: r_xx[h] = sum_n x^*[n] * x[n + h], r_xx[-h] = r_xx^*[h]

    Normalization modes (scaleopt):
      - 'coeff' (default): rho_xy[m] = R_xy[m] / sqrt(R_xx[0] * R_yy[0])  (rho_xx[0] = 1.0)
      - 'none': Raw deterministic sum R_xy[m]
      - 'biased': (1 / N) * R_xy[m], where N = max(Nx, Ny)
      - 'unbiased': (1 / (N - |m|)) * R_xy[m]

    Delay Convention (MATLAB finddelay):
      Peak lag: m_p = argmax_m |rho_xy[m]|
      Estimated delay: D = -m_p,  Delay in sec: D_sec = -m_p / Fs
      Phase offset at peak: phi = angle(rho_xy[m_p])
    """
    x = _as_iq_array(x_in)
    is_autocorr = (y_in is None)
    y = x if is_autocorr else _as_iq_array(y_in)

    nx = len(x)
    ny = len(y)
    fs = float(sample_rate)

    if nx == 0 or ny == 0:
        empty_c = np.empty(0, dtype=np.complex128)
        empty_f = np.empty(0, dtype=np.float64)
        empty_i = np.empty(0, dtype=np.int64)
        return CorrelationResult(
            correlation_complex=empty_c,
            correlation_magnitude=empty_f,
            normalized_correlation=empty_c,
            lags_samples=empty_i,
            lags_sec=empty_f,
            peak_index=0,
            peak_value_complex=0.0 + 0.0j,
            peak_magnitude=0.0,
            peak_lag_samples=0,
            peak_lag_sec=0.0,
            estimated_delay_samples=0,
            estimated_delay_sec=0.0,
            phase_at_peak_rad=0.0,
            sample_rate=fs,
            normalization=scaleopt,
        )

    # If coeff normalization on unequal lengths, zero-pad shorter to match longer (MATLAB compatibility)
    scaleopt_clean = str(scaleopt).lower().strip()
    if scaleopt_clean == "coeff" and nx != ny:
        max_n = max(nx, ny)
        if nx < max_n:
            x_padded = np.pad(x, (0, max_n - nx), mode="constant")
            y_padded = y
        else:
            x_padded = x
            y_padded = np.pad(y, (0, max_n - ny), mode="constant")
        x_calc = x_padded
        y_calc = y_padded
    else:
        x_calc = x
        y_calc = y

    n_calc_x = len(x_calc)
    n_calc_y = len(y_calc)
    n_full_lags = n_calc_x + n_calc_y - 1
    zero_lag_idx = n_calc_y - 1  # Index in full correlation where lag = 0

    # Auto-choose calculation method
    method_clean = str(method).lower().strip()
    if method_clean == "auto":
        use_fft = (n_full_lags >= 64)
    elif method_clean in ("fft", "freq", "frequency"):
        use_fft = True
    else:
        use_fft = False

    if use_fft:
        # FFT implementation: R_xy = IFFT(X(f) * Y^*(f))
        # Zero-padded length to avoid circular wrap-around
        n_fft = int(2 ** math.ceil(math.log2(max(16, n_full_lags))))
        x_fft = np.fft.fft(x_calc, n=n_fft)
        y_fft = np.fft.fft(y_calc, n=n_fft)
        # R_xy[m] = sum_n x[n + m] * y^*[n]
        # In frequency domain: X(f) * conj(Y(f)) gives sum_n x[n] * y^*[n - m]
        # When inverse FFT'd, indices [0 .. n_calc_x - 1] correspond to non-negative lags m >= 0
        # and indices [n_fft - (n_calc_y - 1) .. n_fft - 1] correspond to negative lags m < 0.
        r_fft = np.fft.ifft(x_fft * np.conj(y_fft))
        # Reassemble into standard lag order: m = -(n_calc_y - 1) ... +(n_calc_x - 1)
        neg_part = r_fft[n_fft - (n_calc_y - 1) :]
        pos_part = r_fft[:n_calc_x]
        raw_corr = np.concatenate([neg_part, pos_part])
    else:
        # Direct discrete convolution: correlate(x, y, mode='full')
        # In scipy/numpy correlate: np.correlate(x, y, mode='full') computes sum_n x[n] y^*[n - m]
        # To match MATLAB xcorr(x, y) = sum_n x[n + m] y^*[n], we correlate x with y:
        raw_corr = np.correlate(x_calc, y_calc, mode="full")

    # Autocorrelation exact zero-lag power and conjugate symmetry enforcement
    e_x = float(np.sum(np.abs(x) ** 2))
    e_y = float(np.sum(np.abs(y) ** 2)) if not is_autocorr else e_x

    if is_autocorr:
        # Enforce exact real value at zero lag
        raw_corr[zero_lag_idx] = complex(e_x, 0.0)

    # Full lags array: -(n_calc_y - 1) to +(n_calc_x - 1)
    full_lags = np.arange(-(n_calc_y - 1), n_calc_x)

    # Crop to maxlag if specified
    if maxlag is not None:
        m_lag = int(abs(maxlag))
        mask = (full_lags >= -m_lag) & (full_lags <= m_lag)
        corr_slice = raw_corr[mask]
        lags_slice = full_lags[mask]
    else:
        corr_slice = raw_corr
        lags_slice = full_lags

    # Apply normalization (scaleopt)
    n_ref = max(n_calc_x, n_calc_y)
    norm_factor = math.sqrt(max(eps, e_x * e_y))
    coeff_corr = corr_slice / norm_factor

    if scaleopt_clean in ("coeff", "normalized", "correlation_coefficient"):
        norm_corr = coeff_corr
    elif scaleopt_clean in ("biased", "bias"):
        norm_corr = corr_slice / max(1, n_ref)
    elif scaleopt_clean in ("unbiased", "unbias"):
        unbiased_den = np.maximum(1, n_ref - np.abs(lags_slice))
        norm_corr = corr_slice / unbiased_den
    else:  # 'none' / raw
        norm_corr = corr_slice

    corr_magnitude = np.abs(norm_corr)
    coeff_mag = np.abs(coeff_corr)

    # Find peak correlation (MATLAB finddelay convention based on normalized correlation)
    if len(coeff_mag) > 0:
        peak_idx = int(np.argmax(coeff_mag))
        peak_val = coeff_corr[peak_idx]
        peak_mag = float(coeff_mag[peak_idx])
        peak_lag = int(lags_slice[peak_idx])
        phase_peak = float(np.angle(peak_val))
    else:
        peak_idx = 0
        peak_val = 0.0 + 0.0j
        peak_mag = 0.0
        peak_lag = 0
        phase_peak = 0.0

    peak_lag_sec = float(peak_lag / fs)
    est_delay_samples = -peak_lag  # MATLAB finddelay: D = -m_p
    est_delay_sec = float(est_delay_samples / fs)

    lags_sec = lags_slice / fs

    return CorrelationResult(
        correlation_complex=norm_corr,
        correlation_magnitude=corr_magnitude,
        normalized_correlation=coeff_corr,
        lags_samples=lags_slice,
        lags_sec=lags_sec,
        peak_index=peak_idx,
        peak_value_complex=peak_val,
        peak_magnitude=peak_mag,
        peak_lag_samples=peak_lag,
        peak_lag_sec=peak_lag_sec,
        estimated_delay_samples=est_delay_samples,
        estimated_delay_sec=est_delay_sec,
        phase_at_peak_rad=phase_peak,
        sample_rate=fs,
        normalization=scaleopt_clean,
    )



def _compute_findsignal_core(
    data_in: Any,
    template_in: Any,
    *,
    metric: str = "squared",
    normalization: str = "none",
    sample_rate: float = 1.0,
    eps: float = 1e-18,
) -> TemplateMatchResult:
    """
    Template Matching Engine matching MATLAB `findsignal(data, signal)`.

    MATLAB Mathematical Definition:
      For template s[n], n = 0 ... M - 1, and candidate location i:
      - 'squared' (default): D[i] = sum_{n=0}^{M-1} |x[i + n] - s[n]|^2
      - 'euclidean':         D[i] = sqrt(sum_{n=0}^{M-1} |x[i + n] - s[n]|^2)
      - 'absolute':          D[i] = sum_{n=0}^{M-1} |x[i + n] - s[n]|

      Best match index: i_best = argmin_i D[i]
      Stop index:       i_stop = i_best + M - 1

    Normalization Modes:
      - 'none' (default): Raw signal values
      - 'center': Mean centered: x_i - mu(x_i), s - mu(s)
      - 'power':  L2 normalized: x_i / ||x_i||_2, s / ||s||_2
      - 'zscore': Z-score normalized: (x_i - mu) / sigma
    """
    x = _as_iq_array(data_in)
    s = _as_iq_array(template_in)
    fs = float(sample_rate)

    nx = len(x)
    m = len(s)

    if m == 0 or nx < m:
        empty_f = np.empty(0, dtype=np.float64)
        empty_c = np.empty(0, dtype=np.complex128)
        return TemplateMatchResult(
            best_start_sample=0,
            best_stop_sample=0,
            best_start_sec=0.0,
            best_stop_sec=0.0,
            minimum_distance=float("inf"),
            distance_profile=empty_f,
            matched_segment=empty_c,
            template_method="matlab_findsignal",
            metric=metric,
            normalization=normalization,
            sample_rate=fs,
        )

    # Normalize template s according to normalization mode
    norm_clean = str(normalization).lower().strip().replace("-", "_")
    if norm_clean in ("center", "centered"):
        s_proc = s - np.mean(s)
    elif norm_clean in ("power", "norm", "l2"):
        s_norm = np.linalg.norm(s)
        s_proc = s / (s_norm if s_norm > eps else 1.0)
    elif norm_clean in ("zscore", "z_score", "standardized"):
        s_std = np.std(s)
        s_proc = (s - np.mean(s)) / (s_std if s_std > eps else 1.0)
    else:
        s_proc = s.copy()

    # Fast sliding window distance computation via strided array
    num_positions = nx - m + 1
    shape = (num_positions, m)
    strides = (x.strides[0], x.strides[0])
    windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # Apply normalization across sliding windows
    if norm_clean in ("center", "centered"):
        w_means = np.mean(windows, axis=1, keepdims=True)
        w_proc = windows - w_means
    elif norm_clean in ("power", "norm", "l2"):
        w_norms = np.linalg.norm(windows, axis=1, keepdims=True)
        w_proc = windows / np.maximum(eps, w_norms)
    elif norm_clean in ("zscore", "z_score", "standardized"):
        w_means = np.mean(windows, axis=1, keepdims=True)
        w_stds = np.std(windows, axis=1, keepdims=True)
        w_proc = (windows - w_means) / np.maximum(eps, w_stds)
    else:
        w_proc = windows

    # Compute distance profile D[i]
    metric_clean = str(metric).lower().strip().replace("-", "_")
    diff = w_proc - s_proc[None, :]

    if metric_clean in ("squared", "sqeuclidean", "mse"):
        # Squared Euclidean distance
        d_profile = np.sum(np.abs(diff) ** 2, axis=1)
    elif metric_clean in ("euclidean", "l2", "rmse"):
        d_profile = np.sqrt(np.sum(np.abs(diff) ** 2, axis=1))
    elif metric_clean in ("absolute", "l1", "manhattan", "mae"):
        d_profile = np.sum(np.abs(diff), axis=1)
    else:
        d_profile = np.sum(np.abs(diff) ** 2, axis=1)

    # Best match index (minimizes distance)
    i_best = int(np.argmin(d_profile))
    min_dist = float(d_profile[i_best])
    i_stop = i_best + m - 1

    matched_seg = x[i_best : i_stop + 1].copy()

    return TemplateMatchResult(
        best_start_sample=i_best,
        best_stop_sample=i_stop,
        best_start_sec=float(i_best / fs),
        best_stop_sec=float(i_stop / fs),
        minimum_distance=min_dist,
        distance_profile=d_profile,
        matched_segment=matched_seg,
        template_method="matlab_findsignal",
        metric=metric_clean,
        normalization=norm_clean,
        sample_rate=fs,
    )


def _compute_dtw_core(
    seq_a_in: Any,
    seq_b_in: Any,
    *,
    metric: str = "euclidean",
    normalize_features: bool = True,
    eps: float = 1e-18,
) -> SequenceSimilarityResult:
    """
    Dynamic Time Warping (DTW) Engine matching MATLAB `dtw(x, y)`.

    MATLAB Mathematical Definition:
      For sequences X of length M and Y of length N (with K features):
      Local distance:
        d_mn = sqrt(sum_k (x_{k, m} - y_{k, n})^* (x_{k, m} - y_{k, n}))
      Global warping distance:
        D_DTW = min_P sum_{(m, n) in P} d_mn
      via dynamic programming:
        D[i, j] = d_{ij} + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    Feature Normalization:
      Z-score standardizes individual pulse feature dimensions across the sequence:
      z_q = (q - mu_q) / (sigma_q + eps)
    """
    a_arr = np.asarray(seq_a_in, dtype=np.float64)
    b_arr = np.asarray(seq_b_in, dtype=np.float64)

    # Ensure 2D shape: [length, num_features]
    if a_arr.ndim == 1:
        a_arr = a_arr[:, None]
    if b_arr.ndim == 1:
        b_arr = b_arr[:, None]

    m_len, k_a = a_arr.shape
    n_len, k_b = b_arr.shape

    if k_a != k_b:
        raise ValueError(f"Feature dimension mismatch: sequence A has {k_a} features, sequence B has {k_b} features.")

    if m_len == 0 or n_len == 0:
        empty_mat = np.empty((0, 0), dtype=np.float64)
        return SequenceSimilarityResult(
            dtw_distance=float("inf"),
            normalized_distance=float("inf"),
            similarity_score=0.0,
            warping_path=[],
            accumulated_cost_matrix=empty_mat,
            metric=metric,
            sequence_a_length=m_len,
            sequence_b_length=n_len,
        )

    # Standardize features across time (pooled Z-score per feature dimension)
    if normalize_features:
        pooled = np.vstack([a_arr, b_arr])
        mu_pooled = np.mean(pooled, axis=0, keepdims=True)
        std_pooled = np.std(pooled, axis=0, keepdims=True)
        a_norm = (a_arr - mu_pooled) / np.maximum(eps, std_pooled)
        b_norm = (b_arr - mu_pooled) / np.maximum(eps, std_pooled)
    else:
        a_norm = a_arr
        b_norm = b_arr


    # Compute pairwise local distance matrix d[i, j]: shape [m_len, n_len]
    metric_clean = str(metric).lower().strip().replace("-", "_")
    # Broadcast difference: [m_len, 1, K] - [1, n_len, K] -> [m_len, n_len, K]
    diff = a_norm[:, None, :] - b_norm[None, :, :]

    if metric_clean in ("squared", "sqeuclidean"):
        local_cost = np.sum(diff ** 2, axis=2)
    elif metric_clean in ("absolute", "l1", "manhattan"):
        local_cost = np.sum(np.abs(diff), axis=2)
    else:  # 'euclidean' (MATLAB default)
        local_cost = np.sqrt(np.sum(diff ** 2, axis=2))

    # Dynamic Programming Accumulated Cost Matrix D
    d_accum = np.zeros((m_len, n_len), dtype=np.float64)
    d_accum[0, 0] = local_cost[0, 0]

    for i in range(1, m_len):
        d_accum[i, 0] = d_accum[i - 1, 0] + local_cost[i, 0]

    for j in range(1, n_len):
        d_accum[0, j] = d_accum[0, j - 1] + local_cost[0, j]

    for i in range(1, m_len):
        for j in range(1, n_len):
            d_accum[i, j] = local_cost[i, j] + min(
                d_accum[i - 1, j],      # insertion
                d_accum[i, j - 1],      # deletion
                d_accum[i - 1, j - 1],  # match
            )

    # Backtrack optimal warping path from (m_len - 1, n_len - 1) to (0, 0)
    i, j = m_len - 1, n_len - 1
    path = [(i, j)]

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            steps = [d_accum[i - 1, j - 1], d_accum[i - 1, j], d_accum[i, j - 1]]
            min_step = np.argmin(steps)
            if min_step == 0:
                i, j = i - 1, j - 1
            elif min_step == 1:
                i = i - 1
            else:
                j = j - 1
        path.append((i, j))

    path.reverse()
    path_len = max(1, len(path))

    dtw_dist = float(d_accum[m_len - 1, n_len - 1])
    norm_dist = float(dtw_dist / path_len)
    sim_score = float(1.0 / (1.0 + norm_dist))

    return SequenceSimilarityResult(
        dtw_distance=dtw_dist,
        normalized_distance=norm_dist,
        similarity_score=sim_score,
        warping_path=path,
        accumulated_cost_matrix=d_accum,
        metric=metric_clean,
        sequence_a_length=m_len,
        sequence_b_length=n_len,
    )
