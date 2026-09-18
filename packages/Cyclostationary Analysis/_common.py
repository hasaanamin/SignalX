from __future__ import annotations
from typing import Any
import numpy as np
from scipy import signal as sp_signal


def _as_iq_array(data: Any) -> np.ndarray:
    """
    Standardizes input into a 1D complex128 NumPy array.
    Supports complex ndarray, tuple/list of (I, Q), canonical dictionary, or interleaved bytes.
    """
    if data is None:
        raise ValueError("Input IQ data cannot be None.")

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


def _extract_sample_rate(kwargs: dict[str, Any], default: float = 1.0) -> float:
    """Extracts sample rate parameter from kwargs."""
    for key in ["sample_rate", "fs", "Fs", "sampling_rate", "sample_rate_hz"]:
        if key in kwargs and kwargs[key] is not None:
            val = float(kwargs[key])
            if val > 0:
                return val
    return float(default)


def _get_window(name: str, length: int) -> np.ndarray:
    """Generates standard window vector of given length."""
    name_clean = str(name).lower().strip()
    if length <= 0:
        return np.array([], dtype=np.float64)
    if name_clean in ("hamming", "hamm"):
        return np.hamming(length)
    elif name_clean in ("blackman", "black"):
        return np.blackman(length)
    elif name_clean in ("flattop", "flat_top", "flat"):
        return sp_signal.windows.flattop(length)
    elif name_clean in ("rect", "rectangular", "boxcar", "none", "uniform"):
        return np.ones(length, dtype=np.float64)
    return np.hanning(length)


def next_power_of_two(n: int) -> int:
    """Returns the smallest power of two greater than or equal to n."""
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


def _interpolate_psd(freq_axis: np.ndarray, psd: np.ndarray, eval_freqs: np.ndarray) -> np.ndarray:
    """
    Interpolates a zero-cycle PSD S_x^0(f) at requested evaluation frequencies.
    Requested frequencies outside [freq_axis[0], freq_axis[-1]] return NaN (no circular roll wrapping).
    """
    if freq_axis.size == 0 or psd.size == 0 or eval_freqs.size == 0:
        return np.full_like(eval_freqs, np.nan, dtype=np.float64)

    # Assumes monotonic freq_axis
    if freq_axis[0] > freq_axis[-1]:
        idx_sort = np.argsort(freq_axis)
        freq_axis = freq_axis[idx_sort]
        psd = psd[idx_sort]

    return np.interp(eval_freqs, freq_axis, psd, left=np.nan, right=np.nan)


def _compute_fine_alpha_profile(
    z: np.ndarray,
    fs: float = 1.0,
    nfft_alpha: int | None = None,
    cyclic_lags: list[int] | tuple[int, ...] = (0, 1, 2, 4, 8),
    window: str = "hann",
    num_candidates: int = 5,
    min_alpha_hz: float | None = None,
    snr_threshold_db: float = 8.0,
) -> dict[str, Any]:
    """
    Generic fine cyclic-frequency estimator operating directly on time-domain lag products:
      y_ell[n] = x[n + ell] x*[n]
      Y_ell(alpha) = sum_n y_ell[n] w[n] e^{-j 2pi alpha n / Fs}
      P_alpha = sum_{ell in L} w_ell |Y_ell(alpha)|^2

    Decouples cyclic-frequency resolution from spectral FFT length:
      Delta alpha_spacing = Fs / NFFT_alpha (grid spacing)
      Delta alpha_obs = Fs / N_samples = 1 / T_obs (physical observation resolution)

    Modulation-agnostic: Does not assume any known symbol rate or constellation type.
    """
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

    # Default nfft_alpha: next power of two of sample count (minimum 256)
    if nfft_alpha is None or nfft_alpha <= 0:
        nfft_alpha = max(256, next_power_of_two(n_samples))

    delta_alpha_spacing = fs / nfft_alpha
    delta_alpha_obs = fs / max(1, n_samples)
    alpha_axis = np.fft.fftshift(np.fft.fftfreq(nfft_alpha, d=1.0 / fs))

    # Clean lags list
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

        w = _get_window(window, m_len)
        y_win = y * w
        w_norm = np.sum(w)
        if w_norm <= 0:
            w_norm = float(m_len)

        Y_alpha = np.fft.fftshift(np.fft.fft(y_win, n=nfft_alpha)) / w_norm
        combined_profile += (np.abs(Y_alpha) ** 2)

    combined_profile /= float(len(valid_lags))

    # Mask DC region (|alpha| < min_alpha_hz) to avoid false DC triggers
    if min_alpha_hz is None or min_alpha_hz <= 0:
        min_alpha_hz = max(3.0 * delta_alpha_obs, 0.01 * fs)

    valid_mask = np.abs(alpha_axis) >= min_alpha_hz
    p_search = combined_profile.copy()
    p_search[~valid_mask] = 0.0

    # Background profile noise floor
    if np.any(valid_mask):
        noise_floor = float(np.median(combined_profile[valid_mask]))
    else:
        noise_floor = float(np.median(combined_profile))
    noise_floor = max(1e-20, noise_floor)

    # Local peak detection on fine alpha profile
    min_dist_bins = max(3, int(0.005 * nfft_alpha))
    peak_indices, _ = sp_signal.find_peaks(
        p_search,
        height=noise_floor * 1.5,
        distance=min_dist_bins
    )

    if len(peak_indices) > 0:
        # Rank peaks descending by score
        sorted_order = np.argsort(p_search[peak_indices])[::-1]
        top_indices = peak_indices[sorted_order[:num_candidates]]
    else:
        top_indices = np.array([], dtype=int)

    candidates: list[dict[str, Any]] = []
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
            "estimated_symbol_rate_hz": abs_rate,  # deprecated alias
        })

    # Decision logic
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


def _compute_fast_scf(
    z: np.ndarray,
    fs: float = 1.0,
    nfft_f: int = 128,
    nfft_alpha: int = 128,
    overlap: float = 0.5,
    window: str = "hann",
    max_alpha: float | None = None,
    **kwargs: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the 2D Spectral Correlation Function (SCF) S_x(alpha, f) surface
    for coarse exploration and visual heatmap rendering.

    Parameters:
        z: 1D complex128 signal array.
        fs: Sampling frequency in Hz.
        nfft_f: Spectral frequency FFT length (f-axis resolution).
        nfft_alpha: Cyclic frequency FFT length (alpha-axis resolution).
        overlap: Fractional block overlap (0.0 to 0.75).
        window: Window type ('hann', 'hamming', 'blackman', 'rect').
        max_alpha: Optional maximum cyclic frequency to compute (Hz).

    Returns:
        scf_matrix: 2D complex array of shape [num_alpha, nfft_f] representing S_x(alpha, f).
        freq_axis: 1D array of spectral frequencies f in Hz [-fs/2 to fs/2).
        alpha_axis: 1D array of cyclic frequencies alpha in Hz [-fs to fs].
    """
    if "nfft" in kwargs:
        nfft_f = int(kwargs["nfft"])
        if "nfft_alpha" not in kwargs:
            nfft_alpha = nfft_f

    n_samples = z.size
    p_min = -nfft_alpha // 2
    p_max = nfft_alpha // 2
    p_bins = np.arange(p_min, p_max + 1, dtype=np.int32)
    df_alpha = fs / nfft_alpha
    alpha_axis = p_bins * df_alpha
    freq_axis = np.fft.fftshift(np.fft.fftfreq(nfft_f, d=1.0 / fs))

    # Lag domain window
    max_lag = min(max(16, nfft_f // 4), n_samples - 1)
    lags = np.arange(-max_lag, max_lag + 1, dtype=np.int32)
    n_lags = len(lags)
    win = _get_window(window, n_lags)

    num_alphas = len(p_bins)
    R_matrix = np.zeros((num_alphas, n_lags), dtype=np.complex128)

    # FFT-folding over time
    for l_idx, ell in enumerate(lags):
        if ell >= 0:
            prod = z[ell:] * np.conj(z[: n_samples - ell])
            shift_phase = ell / 2.0
        else:
            abs_ell = abs(ell)
            prod = z[: n_samples - abs_ell] * np.conj(z[abs_ell:])
            shift_phase = abs_ell / 2.0  # Midpoint time n + a/2

        n_prod = len(prod)
        if n_prod == 0:
            continue

        pad_len = int(np.ceil(n_prod / nfft_alpha)) * nfft_alpha
        prod_padded = np.pad(prod, (0, pad_len - n_prod))
        prod_folded = prod_padded.reshape(-1, nfft_alpha).sum(axis=0)
        fft_folded = np.fft.fftshift(np.fft.fft(prod_folded)) / max(1, n_prod)

        # Apply phase correction for midpoint shift_phase
        phase_adj = np.exp(-1j * 2.0 * np.pi * p_bins[:nfft_alpha] * shift_phase / nfft_alpha)
        R_matrix[:nfft_alpha, l_idx] = fft_folded * phase_adj
        R_matrix[nfft_alpha, l_idx] = R_matrix[0, l_idx]

    # Compute S_x^alpha(f) = FFT over lag tau of R_x^alpha[ell] * win[ell]
    R_windowed = R_matrix * win[None, :]
    scf_matrix = np.fft.fftshift(np.fft.fft(R_windowed, n=nfft_f, axis=1), axes=1) / fs

    return scf_matrix, freq_axis, alpha_axis


class StreamingCyclostationaryEstimator:
    """
    Stateful streaming cyclostationary estimator.
    Maintains a rolling buffer of IQ samples across arbitrary transport chunk sizes,
    evaluating fine cyclic-frequency estimation and cyclostationary metrics over a configured
    observation window with configurable hop size.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        analysis_window_samples: int = 16384,
        analysis_hop_samples: int = 8192,
        nfft_f: int = 128,
        nfft_alpha: int | None = None,
        cyclic_lags: list[int] | tuple[int, ...] = (0, 1, 2, 4, 8),
        num_candidates: int = 5,
        snr_threshold_db: float = 8.0,
        window: str = "hann",
        **kwargs: Any
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.analysis_window_samples = int(analysis_window_samples)
        self.analysis_hop_samples = int(analysis_hop_samples)
        self.nfft_f = int(nfft_f)
        self.nfft_alpha = nfft_alpha
        self.cyclic_lags = tuple(cyclic_lags)
        self.num_candidates = int(num_candidates)
        self.snr_threshold_db = float(snr_threshold_db)
        self.window = str(window)
        self.extra_kwargs = kwargs

        self._buffer: np.ndarray = np.array([], dtype=np.complex128)
        self.absolute_sample_index: int = 0
        self._results_history: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Resets estimator internal buffer and state."""
        self._buffer = np.array([], dtype=np.complex128)
        self.absolute_sample_index = 0
        self._results_history.clear()

    def process_chunk(self, chunk: Any) -> list[dict[str, Any]]:
        """
        Appends incoming complex IQ chunk to the rolling persistent buffer.
        Evaluates fine cyclostationary analysis over complete analysis windows of size
        analysis_window_samples every analysis_hop_samples.

        Returns:
            List of result dictionaries computed during this chunk ingestion.
        """
        z = _as_iq_array(chunk)
        if z.size == 0:
            return []

        if self._buffer.size == 0:
            self._buffer = z
        else:
            self._buffer = np.concatenate([self._buffer, z])

        results: list[dict[str, Any]] = []

        while len(self._buffer) >= self.analysis_window_samples:
            window_data = self._buffer[: self.analysis_window_samples]
            start_sample_idx = self.absolute_sample_index

            # Run fine alpha profile on window
            fine_res = _compute_fine_alpha_profile(
                window_data,
                fs=self.sample_rate,
                nfft_alpha=self.nfft_alpha,
                cyclic_lags=self.cyclic_lags,
                window=self.window,
                num_candidates=self.num_candidates,
                snr_threshold_db=self.snr_threshold_db,
            )

            res_entry = {
                "absolute_sample_index": start_sample_idx,
                "window_samples": self.analysis_window_samples,
                "sample_rate_hz": self.sample_rate,
                "cyclostationary_detected": fine_res["cyclostationary_detected"],
                "estimated_cyclic_rate_hz": fine_res["estimated_cyclic_rate_hz"],
                "primary_signed_alpha_hz": fine_res["primary_signed_alpha_hz"],
                "raw_primary_alpha_hz": fine_res["raw_primary_alpha_hz"],
                "raw_cyclic_snr_db": fine_res["raw_cyclic_snr_db"],
                "candidates": fine_res["candidates"],
                "fine_alpha_axis_hz": fine_res["alpha_axis_hz"],
                "fine_alpha_profile": fine_res["alpha_profile"],
                "delta_alpha_hz": fine_res["delta_alpha_hz"],
                "alpha_bin_spacing_hz": fine_res["alpha_bin_spacing_hz"],
                "alpha_observation_resolution_hz": fine_res["alpha_observation_resolution_hz"],
            }
            results.append(res_entry)
            self._results_history.append(res_entry)

            # Advance buffer by hop
            self._buffer = self._buffer[self.analysis_hop_samples :]
            self.absolute_sample_index += self.analysis_hop_samples

        return results
