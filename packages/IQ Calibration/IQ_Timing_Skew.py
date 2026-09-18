from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Tuple
import numpy as np


def _as_complex_array(x: Any, name: str = "iq") -> np.ndarray:
    """Helper to convert various input formats into a 1D complex numpy array."""
    if x is None:
        raise ValueError(f"{name} cannot be None")
    if isinstance(x, tuple) and len(x) == 2:
        i_arr = np.asarray(x[0], dtype=np.float64).ravel()
        q_arr = np.asarray(x[1], dtype=np.float64).ravel()
        if len(i_arr) != len(q_arr):
            raise ValueError(f"{name} I and Q tuple components must have equal length ({len(i_arr)} != {len(q_arr)})")
        return i_arr + 1j * q_arr
    arr = np.asarray(x)
    if arr.size == 0:
        return np.array([], dtype=np.complex128)
    if arr.ndim == 2:
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            return arr[0, :].astype(np.float64) + 1j * arr[1, :].astype(np.float64)
        elif arr.shape[1] == 2:
            return arr[:, 0].astype(np.float64) + 1j * arr[:, 1].astype(np.float64)
    if np.iscomplexobj(arr):
        return arr.ravel()
    return arr.ravel().astype(np.complex128)


def _validate_finite_iq(arr: np.ndarray, name: str = "iq") -> None:
    """Validates that all samples in the complex array are finite."""
    if not np.all(np.isfinite(arr)):
        nan_count = int(np.sum(np.isnan(arr)))
        inf_count = int(np.sum(np.isinf(arr)))
        raise ValueError(f"{name} contains non-finite values: {nan_count} NaNs, {inf_count} Infs out of {len(arr)} samples")


def _fractional_delay_fft(signal_1d: np.ndarray, delay_samples: float) -> np.ndarray:
    """Applies precise sub-sample fractional delay tau using frequency-domain phase shift."""
    n = len(signal_1d)
    if abs(delay_samples) < 1e-12:
        return signal_1d.copy()
    freqs = np.fft.fftfreq(n)
    phase_shift = np.exp(-1j * 2.0 * np.pi * freqs * float(delay_samples))
    is_real = not np.iscomplexobj(signal_1d)
    spec = np.fft.fft(signal_1d) * phase_shift
    out = np.fft.ifft(spec)
    return out.real if is_real else out


def _compute_cross_spectrum_and_coherence(
    i_signal: np.ndarray,
    q_signal: np.ndarray,
    fs: float,
    nfft: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes Welch Cross-Spectral Density S_iq(f) and Magnitude-Squared Coherence."""
    n = min(len(i_signal), len(q_signal))
    nperseg = min(nfft, n)
    win = np.hanning(nperseg)
    win_power = np.sum(win ** 2)
    step = max(1, nperseg // 2)

    s_ii = np.zeros(nperseg, dtype=np.float64)
    s_qq = np.zeros(nperseg, dtype=np.float64)
    s_iq = np.zeros(nperseg, dtype=np.complex128)
    count = 0

    for start in range(0, n - nperseg + 1, step):
        i_seg = i_signal[start : start + nperseg] * win
        q_seg = q_signal[start : start + nperseg] * win
        f_i = np.fft.fftshift(np.fft.fft(i_seg))
        f_q = np.fft.fftshift(np.fft.fft(q_seg))

        s_ii += np.abs(f_i) ** 2 / (fs * win_power)
        s_qq += np.abs(f_q) ** 2 / (fs * win_power)
        s_iq += (f_i * np.conj(f_q)) / (fs * win_power)
        count += 1

    if count == 0:
        count = 1

    s_ii /= count
    s_qq /= count
    s_iq /= count

    denom = s_ii * s_qq
    coherence = np.where(denom > 1e-20, (np.abs(s_iq) ** 2) / denom, 0.0)
    freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1.0 / fs))
    return freqs, s_iq, coherence


@dataclass
class IQTimingSkewResult:
    """
    Result container for I/Q Timing Skew (relative ADC / group delay mismatch).

    Sign Convention
    ---------------
    Positive timing_skew_seconds > 0 means the Q (quadrature) branch is delayed
    relative to the I (in-phase) branch (Q arrives later).
    """
    timing_skew_samples: float = 0.0
    timing_skew_seconds: float = 0.0
    fractional_skew_samples: float = 0.0
    confidence: float = 1.0
    valid: bool = True
    method: str = "cross_spectrum_slope"
    diagnostic: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def skew_samples(self) -> float:
        return self.timing_skew_samples

    @property
    def skew_seconds(self) -> float:
        return self.timing_skew_seconds


class IQTimingSkewAnalyzer:
    """
    I/Q Timing Skew (Inter-Channel ADC Delay) Analyzer.

    Estimates relative group delay and sub-sample clock skew between In-Phase (I)
    and Quadrature (Q) analog/digital conversion paths.
    Does NOT import or depend on any sibling analysis modules.
    """

    def __init__(self) -> None:
        self.result: IQTimingSkewResult | None = None

    def compute(
        self,
        iq: Any,
        sample_rate_hz: float = 1.0,
        dc_result: Any | None = None,
        method: str = "cross_spectrum_slope",
        reference_iq: Any | None = None,
        coherence_threshold: float = 0.6,
        **kwargs: Any,
    ) -> IQTimingSkewResult:
        """
        Estimates relative timing skew between I and Q channels.

        Parameters
        ----------
        iq : Any
            Input complex IQ signal.
        sample_rate_hz : float
            Sampling rate in Hz (> 0).
        dc_result : Any | None
            Optional precomputed DC offset result to reuse.
        method : str
            Estimation method: 'cross_spectrum_slope', 'cross_correlation', or 'reference_aided'.
        reference_iq : Any | None
            Clean reference signal for reference-aided estimation.
        coherence_threshold : float
            Minimum spectral coherence required to include a frequency bin in the delay slope fit.

        Returns
        -------
        IQTimingSkewResult
            Delay in seconds, total samples, and fractional sample offset.
        """
        x = _as_complex_array(iq, name="iq")
        _validate_finite_iq(x, name="iq")
        n = len(x)

        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be strictly positive, got {sample_rate_hz}")
        fs = float(sample_rate_hz)

        if n < 16:
            self.result = IQTimingSkewResult(
                timing_skew_samples=0.0,
                timing_skew_seconds=0.0,
                fractional_skew_samples=0.0,
                confidence=0.0,
                valid=False,
                method=method,
                diagnostic="Sample count too small (< 16 samples) for timing-skew fitting",
            )
            return self.result

        # Remove DC offset
        if dc_result is not None:
            dc_c = getattr(dc_result, "dc_complex", complex(dc_result))
        else:
            dc_c = complex(float(np.mean(x.real)), float(np.mean(x.imag)))

        x_ac = x - dc_c
        i_ac = x_ac.real
        q_ac = x_ac.imag

        # Bandwidth check to detect single CW tones (which lack excitation bandwidth for delay slope fitting)
        fft_mag = np.abs(np.fft.fft(x_ac))
        total_pwr = float(np.sum(fft_mag ** 2))

        if total_pwr < 1e-20:
            self.result = IQTimingSkewResult(
                timing_skew_samples=0.0,
                timing_skew_seconds=0.0,
                fractional_skew_samples=0.0,
                confidence=0.0,
                valid=False,
                method=method,
                diagnostic="Zero AC signal power; unable to estimate timing skew",
            )
            return self.result

        prob = (fft_mag ** 2) / (total_pwr + 1e-30)
        spectral_entropy = float(-np.sum(prob * np.log(prob + 1e-30)) / np.log(n))
        pwr_sorted = np.sort(prob)[::-1]
        cum_pwr = np.cumsum(pwr_sorted)
        top_bins_99 = int(np.searchsorted(cum_pwr, 0.99)) + 1
        is_narrowband = bool((spectral_entropy < 0.25 or top_bins_99 < 8) and reference_iq is None)

        method_clean = str(method).lower().strip()
        meta: dict[str, Any] = {
            "sample_count": n,
            "sample_rate_hz": fs,
            "spectral_entropy": spectral_entropy,
            "top_bins_99": top_bins_99,
        }

        if method_clean in ("cross_spectrum_slope", "cross_spectrum"):
            if is_narrowband and reference_iq is None:
                self.result = IQTimingSkewResult(
                    timing_skew_samples=0.0,
                    timing_skew_seconds=0.0,
                    fractional_skew_samples=0.0,
                    confidence=0.05,
                    valid=False,
                    method="cross_spectrum_slope",
                    diagnostic="Insufficient signal bandwidth for blind timing-skew estimation (single tone / CW detected)",
                    metadata=meta,
                )
                return self.result

            nfft = min(512, max(32, n // 2))
            freqs, s_iq, coherence = _compute_cross_spectrum_and_coherence(i_ac, q_ac, fs, nfft=nfft)

            # Fit on valid positive frequency bins
            valid_mask = (coherence >= coherence_threshold) & (freqs > 0.05 * fs) & (freqs < 0.45 * fs)
            if np.sum(valid_mask) < 4:
                valid_mask = (coherence >= 0.15) & (freqs > 0.02 * fs) & (freqs < 0.48 * fs)

            if np.sum(valid_mask) < 4:
                self.result = IQTimingSkewResult(
                    timing_skew_samples=0.0,
                    timing_skew_seconds=0.0,
                    fractional_skew_samples=0.0,
                    confidence=0.1,
                    valid=False,
                    method="cross_spectrum_slope",
                    diagnostic="Insufficient coherent frequency bins across I/Q cross-spectrum",
                    metadata=meta,
                )
                return self.result

            f_fit = freqs[valid_mask]
            phase_fit = np.unwrap(np.angle(s_iq[valid_mask]))
            weights = coherence[valid_mask]

            # Linear Least Squares: S_IQ(f) = |I(f)|^2 * exp(j * (phi_0 + 2*pi*f*tau))
            poly = np.polyfit(f_fit, phase_fit, deg=1, w=weights)
            slope = float(poly[0])
            tau_sec = float(slope / (2.0 * np.pi))
            tau_samples = float(tau_sec * fs)

            frac_samples = float(tau_samples - np.round(tau_samples))
            if abs(frac_samples) > 0.5:
                frac_samples = float(np.sign(frac_samples) * (abs(frac_samples) - 1.0))

            mean_coherence = float(np.mean(weights))
            confidence = float(np.clip(mean_coherence * (1.0 - 5.0 / np.sqrt(n)), 0.1, 1.0))

            meta["fit_slope"] = slope
            meta["mean_coherence"] = mean_coherence
            meta["coherent_bins_count"] = int(np.sum(valid_mask))

            self.result = IQTimingSkewResult(
                timing_skew_samples=tau_samples,
                timing_skew_seconds=tau_sec,
                fractional_skew_samples=frac_samples,
                confidence=confidence,
                valid=True,
                method="cross_spectrum_slope",
                diagnostic=None,
                metadata=meta,
            )
            return self.result

        elif method_clean in ("cross_correlation", "correlation"):
            corr = np.correlate(i_ac - np.mean(i_ac), q_ac - np.mean(q_ac), mode="full")
            lags = np.arange(-n + 1, n)
            center = n - 1
            search_rad = min(64, n // 4)
            window_slice = slice(center - search_rad, center + search_rad + 1)
            corr_win = np.abs(corr[window_slice])
            lags_win = lags[window_slice]

            peak_local = int(np.argmax(corr_win))
            peak_lag = float(lags_win[peak_local])

            if 0 < peak_local < len(corr_win) - 1:
                alpha = corr_win[peak_local - 1]
                beta = corr_win[peak_local]
                gamma = corr_win[peak_local + 1]
                denom = alpha - 2.0 * beta + gamma
                delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-20 else 0.0
                peak_lag += delta

            tau_samples = float(peak_lag)
            tau_sec = float(tau_samples / fs)
            frac_samples = float(tau_samples - np.round(tau_samples))

            self.result = IQTimingSkewResult(
                timing_skew_samples=tau_samples,
                timing_skew_seconds=tau_sec,
                fractional_skew_samples=frac_samples,
                confidence=0.85,
                valid=True,
                method="cross_correlation",
                metadata=meta,
            )
            return self.result

        elif method_clean == "reference_aided":
            if reference_iq is None:
                raise ValueError("reference_iq must be provided when method='reference_aided'")
            ref = _as_complex_array(reference_iq, name="reference_iq")
            _validate_finite_iq(ref, name="reference_iq")

            freqs_i, s_ref_i, _ = _compute_cross_spectrum_and_coherence(ref.real, i_ac, fs)
            freqs_q, s_ref_q, _ = _compute_cross_spectrum_and_coherence(ref.imag, q_ac, fs)

            mask = (freqs_i > 0.05 * fs) & (freqs_i < 0.45 * fs)
            slope_i = float(np.polyfit(freqs_i[mask], np.unwrap(np.angle(s_ref_i[mask])), 1)[0])
            slope_q = float(np.polyfit(freqs_q[mask], np.unwrap(np.angle(s_ref_q[mask])), 1)[0])

            tau_i_sec = float(slope_i / (2.0 * np.pi))
            tau_q_sec = float(slope_q / (2.0 * np.pi))
            tau_sec = float(tau_q_sec - tau_i_sec)
            tau_samples = float(tau_sec * fs)
            frac_samples = float(tau_samples - np.round(tau_samples))

            self.result = IQTimingSkewResult(
                timing_skew_samples=tau_samples,
                timing_skew_seconds=tau_sec,
                fractional_skew_samples=frac_samples,
                confidence=1.0,
                valid=True,
                method="reference_aided",
                metadata=meta,
            )
            return self.result

        else:
            raise ValueError(f"Unknown timing skew method '{method}'. Must be 'cross_spectrum_slope', 'cross_correlation', or 'reference_aided'")

    def analyze(self, *args: Any, **kwargs: Any) -> IQTimingSkewResult:
        """Alias for compute()."""
        return self.compute(*args, **kwargs)


IQTimingSkewEstimator = IQTimingSkewAnalyzer


def compute_timing_skew(
    iq: Any = None,
    sample_rate_hz: float = 1.0,
    sample_rate: float | None = None,
    dc_result: Any | None = None,
    dc_offset_result: Any | None = None,
    dc_offset: Any | None = None,
    method: str = "cross_spectrum_slope",
    reference_iq: Any | None = None,
    coherence_threshold: float = 0.6,
    iq_data: Any = None,
    **kwargs: Any,
) -> IQTimingSkewResult:
    """Convenience functional wrapper for IQTimingSkewAnalyzer."""
    data = iq if iq is not None else iq_data
    fs = sample_rate if sample_rate is not None else sample_rate_hz
    dc_res = dc_result if dc_result is not None else (dc_offset_result if dc_offset_result is not None else dc_offset)

    analyzer = IQTimingSkewAnalyzer()
    return analyzer.compute(
        iq=data,
        sample_rate_hz=fs,
        dc_result=dc_res,
        method=method,
        reference_iq=reference_iq,
        coherence_threshold=coherence_threshold,
        **kwargs,
    )


estimate_timing_skew = compute_timing_skew

__all__ = [
    "IQTimingSkewAnalyzer",
    "IQTimingSkewEstimator",
    "IQTimingSkewResult",
    "compute_timing_skew",
    "estimate_timing_skew",
]

