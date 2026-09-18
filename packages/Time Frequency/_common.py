from __future__ import annotations
from typing import Any
import math
import numpy as np
from scipy import signal as sp_signal

from .base import TimeFrequencyResult, WaveletResult


def _as_iq_array(data: Any) -> np.ndarray:
    """
    Standardizes input into a 1D complex128 NumPy array.
    Supports complex ndarray, tuple/list of (I, Q), canonical dictionary, or structured arrays.
    """
    if data is None:
        raise ValueError("Input IQ data cannot be None.")

    if isinstance(data, TimeFrequencyResult):
        raise TypeError("Expected raw IQ data, but received TimeFrequencyResult.")

    if isinstance(data, WaveletResult):
        raise TypeError("Expected raw IQ data, but received WaveletResult.")

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
    Checks explicit parameter first, then kwargs dictionary, then data metadata.
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


def _get_window(window_spec: str | np.ndarray, length: int) -> tuple[np.ndarray, str]:
    """
    Generates or validates a window array of the requested length.
    Returns (window_vector, window_name).
    """
    if length <= 0:
        raise ValueError(f"Window length must be > 0, got {length}")

    if isinstance(window_spec, np.ndarray):
        w = np.asarray(window_spec, dtype=np.float64).ravel()
        if len(w) != length:
            raise ValueError(f"Provided window array length ({len(w)}) does not match expected length ({length}).")
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

    # Default fallback
    return np.hanning(length), "hann"


def _compute_stft_core(
    iq_data: Any,
    *,
    sample_rate: float = 1.0,
    window_length: int = 256,
    overlap: float | int = 0.75,
    nfft: int | None = None,
    window: str | np.ndarray = "hann",
    sample_offset: int = 0,
    eps: float = 1e-18,
) -> TimeFrequencyResult:
    """
    Core discrete STFT engine matching MATLAB `stft` and `spectrogram` definitions for complex IQ.
    Produces two-sided centered spectrum [-Fs/2, Fs/2) via fftshift and computes window-center timestamps.
    """
    z = _as_iq_array(iq_data)
    fs = float(sample_rate)
    if fs <= 0 or not np.isfinite(fs):
        raise ValueError(f"Sample rate must be > 0, got {sample_rate}")

    n_samples = len(z)
    m_win = int(window_length)
    if m_win <= 0:
        raise ValueError(f"window_length must be > 0, got {window_length}")

    # Determine overlap samples L and hop size R = M - L
    if isinstance(overlap, float):
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"Float overlap fraction must be in [0, 1), got {overlap}")
        l_overlap = int(round(m_win * overlap))
    else:
        l_overlap = int(overlap)
        if not (0 <= l_overlap < m_win):
            raise ValueError(f"Integer overlap samples must satisfy 0 <= overlap < window_length ({m_win}), got {l_overlap}")

    hop_r = max(1, m_win - l_overlap)

    # Determine N_fft (>= window_length)
    if nfft is None:
        n_fft = max(m_win, 256)
    else:
        n_fft = int(nfft)
        if n_fft < m_win:
            raise ValueError(f"nfft ({n_fft}) must be >= window_length ({m_win})")

    win_vec, win_name = _get_window(window, m_win)

    # Empty signal check
    if n_samples < m_win:
        # Zero frames
        freqs = (np.arange(n_fft) - (n_fft // 2)) * (fs / n_fft)
        return TimeFrequencyResult(
            stft_complex=np.empty((n_fft, 0), dtype=np.complex128),
            power_spectrogram=np.empty((n_fft, 0), dtype=np.float64),
            spectrogram_db=np.empty((n_fft, 0), dtype=np.float64),
            frequencies_hz=freqs,
            times_sec=np.empty(0, dtype=np.float64),
            sample_rate=fs,
            hop_samples=hop_r,
            window_length=m_win,
            nfft=n_fft,
            window_name=win_name,
            sample_offset=sample_offset,
        )

    # Number of frames
    num_frames = (n_samples - m_win) // hop_r + 1
    start_indices = np.arange(num_frames) * hop_r

    # Extract framed segments via strided view or index gathering
    # Shape: [num_frames, m_win]
    shape = (num_frames, m_win)
    strides = (z.strides[0] * hop_r, z.strides[0])
    frames = np.lib.stride_tricks.as_strided(z, shape=shape, strides=strides)

    # Apply window: [num_frames, m_win] * [m_win]
    windowed_frames = frames * win_vec

    # Compute FFT across each frame (axis 1) and shift frequencies to center [-Fs/2, Fs/2)
    # X_fft: [num_frames, n_fft]
    stft_raw = np.fft.fft(windowed_frames, n=n_fft, axis=1)
    stft_centered = np.fft.fftshift(stft_raw, axes=1)

    # Transpose to canonical MATLAB [K_freqs, M_frames]
    stft_complex = stft_centered.T  # Shape: [n_fft, num_frames]

    # Frequency vector: [-Fs/2, Fs/2)
    k_indices = np.arange(n_fft)
    freqs = (k_indices - (n_fft // 2)) * (fs / n_fft)

    # Window center timestamps: t_m = (n_start,m + (M - 1) / 2) / F_s
    times_sec = (sample_offset + start_indices + (m_win - 1) / 2.0) / fs

    # Linear power spectrogram: P[k, m] = |X[k, m]|^2
    power_spectrogram = np.abs(stft_complex) ** 2

    # dB spectrogram for visualization: 10 * log10(P + eps)
    spectrogram_db = 10.0 * np.log10(power_spectrogram + eps)

    return TimeFrequencyResult(
        stft_complex=stft_complex,
        power_spectrogram=power_spectrogram,
        spectrogram_db=spectrogram_db,
        frequencies_hz=freqs,
        times_sec=times_sec,
        sample_rate=fs,
        hop_samples=hop_r,
        window_length=m_win,
        nfft=n_fft,
        window_name=win_name,
        sample_offset=sample_offset,
    )


def calculate_morse_support_samples(
    sample_rate: float,
    min_frequency_hz: float,
    gamma: float = 3.0,
    time_bandwidth: float = 60.0,
    support_tolerance: float = 1e-6,
) -> int:
    """
    Calculates the required temporal support half-width (in samples) for a Generalized Morse Wavelet
    at the lowest requested frequency (largest scale).
    Ensures wavelet envelope decays to below support_tolerance at boundary context limits.
    """
    fs = float(sample_rate)
    f_min = float(min_frequency_hz)
    if fs <= 0 or f_min <= 0:
        return 256
    # Empirical and analytical support factor for generalized Morse wavelets
    tb = float(time_bandwidth)
    scale_factor = math.sqrt(max(1.0, tb) / 60.0)
    samples = int(math.ceil(6.32 * (fs / f_min) * scale_factor))
    return max(64, samples)


class CWTFilterBank:
    """
    Precomputed Continuous Wavelet Transform (CWT) Filter Bank matching MATLAB `cwtfilterbank`.
    Uses Generalized Morse Wavelets with L1 or L2 normalization.
    Supports analytic positive-frequency ('analytic_positive') and two-sided ('two_sided') frequency modes.
    Caches frequency-domain wavelets across logarithmically spaced scales for high-throughput batch and streaming CWT.
    """

    def __init__(
        self,
        signal_length: int,
        sample_rate: float = 1.0,
        wavelet_name: str = "morse",
        voices_per_octave: int = 10,
        gamma: float = 3.0,
        time_bandwidth: float = 60.0,
        frequency_limits: tuple[float, float] | None = None,
        min_frequency_hz: float | None = None,
        max_frequency_hz: float | None = None,
        scales: np.ndarray | None = None,
        normalization: str = "L1",
        frequency_mode: str = "analytic_positive",
    ) -> None:
        self.signal_length = int(signal_length)
        if self.signal_length <= 0:
            raise ValueError(f"signal_length must be > 0, got {signal_length}")

        self.sample_rate = float(sample_rate)
        if self.sample_rate <= 0 or not np.isfinite(self.sample_rate):
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")

        self.wavelet_name = str(wavelet_name).lower().strip()
        if self.wavelet_name not in ("morse",):
            raise ValueError(f"Unsupported wavelet_name '{wavelet_name}'. Expected one of: 'morse'.")

        self.voices_per_octave = int(voices_per_octave)
        if self.voices_per_octave < 1:
            raise ValueError(f"voices_per_octave must be >= 1, got {voices_per_octave}")

        self.gamma = float(gamma)
        if self.gamma <= 0 or not np.isfinite(self.gamma):
            raise ValueError(f"gamma must be > 0, got {gamma}")

        self.time_bandwidth = float(time_bandwidth)
        if self.time_bandwidth <= 0 or not np.isfinite(self.time_bandwidth):
            raise ValueError(f"time_bandwidth must be > 0, got {time_bandwidth}")

        self.normalization = str(normalization).upper().strip()
        if self.normalization not in ("L1", "L2"):
            raise ValueError(f"Unsupported normalization '{normalization}'. Expected one of: 'L1', 'L2'.")

        self.frequency_mode = str(frequency_mode).lower().strip()
        if self.frequency_mode not in ("analytic_positive", "two_sided"):
            raise ValueError(f"Unsupported frequency_mode '{frequency_mode}'. Expected one of: 'analytic_positive', 'two_sided'.")

        # Compute angular frequencies for FFT length N
        n = self.signal_length
        # Standard FFT discrete angular frequencies omega: [0, 2*pi*(1 - 1/N)]
        self.omega = (2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / self.sample_rate)) / self.sample_rate

        # Wavelet peak frequency omega_0 and constant a_{beta, gamma}
        # Time-bandwidth P^2 = beta * gamma  =>  beta = P^2 / gamma
        self.beta = self.time_bandwidth / self.gamma
        self.omega_0 = (self.beta / self.gamma) ** (1.0 / self.gamma)
        self.f_0 = self.omega_0 / (2.0 * np.pi)  # Peak normalized frequency

        # Determine frequency limits [f_min, f_max]
        nyquist = self.sample_rate / 2.0
        if scales is not None:
            base_scales = np.asarray(scales, dtype=np.float64).ravel()
            if len(base_scales) == 0 or np.any(base_scales <= 0):
                raise ValueError("Provided scales array must be non-empty and contain positive values.")
            num_scales = len(base_scales)
            s_min = float(base_scales[0])
            s_max = float(base_scales[-1])
            base_freqs = (self.f_0 * self.sample_rate) / base_scales
            self.min_frequency_hz = float(np.min(base_freqs))
            self.max_frequency_hz = float(np.max(base_freqs))
        else:
            if min_frequency_hz is not None or max_frequency_hz is not None:
                f_min = float(min_frequency_hz) if min_frequency_hz is not None else max(self.sample_rate / n * 2.0, self.sample_rate / (2 ** (np.log2(n) - 1)))
                f_max = float(max_frequency_hz) if max_frequency_hz is not None else nyquist * 0.95
            elif frequency_limits is not None:
                f_min = float(frequency_limits[0])
                f_max = float(frequency_limits[1])
            else:
                # Default MATLAB frequency span: from ~2 cycles over the signal length up to near Nyquist
                f_min = max(self.sample_rate / n * 2.0, self.sample_rate / (2 ** (np.log2(n) - 1)))
                f_max = nyquist * 0.95

            if f_min <= 0 or not np.isfinite(f_min):
                raise ValueError(f"min_frequency_hz / f_min must be > 0, got {f_min}")
            if f_max <= f_min:
                raise ValueError(f"max_frequency_hz ({f_max}) must be > min_frequency_hz ({f_min})")
            if f_max >= nyquist:
                raise ValueError(f"max_frequency_hz ({f_max}) must be < Nyquist ({nyquist})")

            self.min_frequency_hz = f_min
            self.max_frequency_hz = f_max

            # Calculate scales: s_j = s_0 * 2^(j / V)
            # f_j = (f_0 / s_j) * F_s  =>  s_j = (f_0 * F_s) / f_j
            s_min = (self.f_0 * self.sample_rate) / f_max
            s_max = (self.f_0 * self.sample_rate) / f_min

            num_octaves = math.log2(s_max / s_min)
            num_scales = max(1, int(round(num_octaves * self.voices_per_octave)) + 1)
            scale_indices = np.arange(num_scales, dtype=np.float64)
            base_scales = s_min * (2.0 ** (scale_indices / self.voices_per_octave))
            base_freqs = (self.f_0 * self.sample_rate) / base_scales

        # Peak normalization constant a_{beta, gamma} such that peak amplitude = 2 (analytic Fourier domain)
        # a = 2 * (exp(1) * gamma / beta)^(beta / gamma)
        self.a_norm = 2.0 * ((math.e * self.gamma / self.beta) ** (self.beta / self.gamma))

        pos_omega_mask = self.omega > 0.0
        pos_omega = self.omega[pos_omega_mask]

        psi_pos = np.zeros((num_scales, n), dtype=np.float64)
        for idx, s in enumerate(base_scales):
            s_omega = s * pos_omega
            # Morse formula: a * (s*omega)^beta * exp(-(s*omega)^gamma)
            # Use log-domain computation for extreme numerical stability
            log_psi = (
                math.log(self.a_norm)
                + self.beta * np.log(np.maximum(1e-300, s_omega))
                - (s_omega ** self.gamma)
            )
            psi_vals = np.exp(np.clip(log_psi, -700.0, 700.0))
            if self.normalization == "L2":
                psi_vals = psi_vals * math.sqrt(s / s_min)
            psi_pos[idx, pos_omega_mask] = psi_vals

        if self.frequency_mode == "two_sided":
            neg_omega_mask = self.omega < 0.0
            neg_omega = -self.omega[neg_omega_mask]

            psi_neg = np.zeros((num_scales, n), dtype=np.float64)
            for idx, s in enumerate(base_scales):
                s_omega = s * neg_omega
                log_psi = (
                    math.log(self.a_norm)
                    + self.beta * np.log(np.maximum(1e-300, s_omega))
                    - (s_omega ** self.gamma)
                )
                psi_vals = np.exp(np.clip(log_psi, -700.0, 700.0))
                if self.normalization == "L2":
                    psi_vals = psi_vals * math.sqrt(s / s_min)
                psi_neg[idx, neg_omega_mask] = psi_vals

            self.psi_ft = np.vstack([psi_pos, psi_neg])
            self.scales = np.concatenate([base_scales, base_scales])
            self.frequencies_hz = np.concatenate([base_freqs, -base_freqs])
        else:
            self.psi_ft = psi_pos
            self.scales = base_scales
            self.frequencies_hz = base_freqs

    @property
    def filters(self) -> np.ndarray:
        """Returns precomputed frequency-domain wavelet filters [num_scales, signal_length]."""
        return self.psi_ft

    def compute_cwt(self, z: np.ndarray, sample_offset: int = 0) -> WaveletResult:
        """
        Executes Fast CWT on complex IQ input vector matching MATLAB `cwt`.
        Uses precomputed frequency-domain filters.
        """
        n = len(z)
        if n != self.signal_length:
            raise ValueError(f"Input signal length ({n}) does not match filter bank length ({self.signal_length}).")

        # FFT of complex signal
        z_fft = np.fft.fft(z)

        # CWT: W[s, n] = IFFT(Z(omega) * Psi^*(s*omega))
        # Since Psi is real-valued in frequency domain, Psi^* = Psi
        # Matrix multiply in frequency domain: [num_scales, N] * [1, N]
        w_fft = self.psi_ft * z_fft[None, :]

        # IFFT back to time domain
        cwt_coeffs = np.fft.ifft(w_fft, axis=1)  # Shape: [num_scales, N]

        # Wavelet coefficient energy map
        coeff_energy = np.abs(cwt_coeffs) ** 2

        times_sec = (sample_offset + np.arange(n)) / self.sample_rate

        return WaveletResult(
            coefficients=cwt_coeffs,
            frequencies_hz=self.frequencies_hz,
            scales=self.scales,
            coefficient_energy=coeff_energy,
            times_sec=times_sec,
            sample_rate=self.sample_rate,
            wavelet_name=self.wavelet_name,
            voices_per_octave=self.voices_per_octave,
            gamma=self.gamma,
            time_bandwidth=self.time_bandwidth,
            normalization=self.normalization,
            frequency_mode=self.frequency_mode,
            sample_offset=sample_offset,
        )


def _compute_cwt_core(
    iq_data: Any,
    *,
    sample_rate: float = 1.0,
    wavelet_name: str = "morse",
    voices_per_octave: int = 10,
    gamma: float = 3.0,
    time_bandwidth: float = 60.0,
    frequency_limits: tuple[float, float] | None = None,
    min_frequency_hz: float | None = None,
    max_frequency_hz: float | None = None,
    scales: np.ndarray | None = None,
    normalization: str = "L1",
    frequency_mode: str = "analytic_positive",
    sample_offset: int = 0,
) -> WaveletResult:
    """
    Computes Continuous Wavelet Transform on complex IQ array matching MATLAB `cwt` defaults.
    """
    z = _as_iq_array(iq_data)
    fs = float(sample_rate)
    n = len(z)

    if n == 0:
        return WaveletResult(
            coefficients=np.empty((0, 0), dtype=np.complex128),
            frequencies_hz=np.empty(0, dtype=np.float64),
            scales=np.empty(0, dtype=np.float64),
            coefficient_energy=np.empty((0, 0), dtype=np.float64),
            times_sec=np.empty(0, dtype=np.float64),
            sample_rate=fs,
            wavelet_name=wavelet_name,
            voices_per_octave=voices_per_octave,
            gamma=gamma,
            time_bandwidth=time_bandwidth,
            normalization=normalization,
            frequency_mode=frequency_mode,
            sample_offset=sample_offset,
        )

    fb = CWTFilterBank(
        signal_length=n,
        sample_rate=fs,
        wavelet_name=wavelet_name,
        voices_per_octave=voices_per_octave,
        gamma=gamma,
        time_bandwidth=time_bandwidth,
        frequency_limits=frequency_limits,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        scales=scales,
        normalization=normalization,
        frequency_mode=frequency_mode,
    )
    return fb.compute_cwt(z, sample_offset=sample_offset)

