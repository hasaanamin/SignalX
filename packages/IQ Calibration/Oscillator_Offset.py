from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
import numpy as np


def _as_complex_array(x: Any, name: str = "iq") -> np.ndarray:
    """Converts input IQ data to a 1D complex128 numpy array."""
    if isinstance(x, np.ndarray):
        if np.iscomplexobj(x):
            arr = x.astype(np.complex128, copy=False)
        else:
            if x.ndim == 2 and x.shape[0] == 2:
                arr = (x[0, :] + 1j * x[1, :]).astype(np.complex128)
            elif x.ndim == 2 and x.shape[1] == 2:
                arr = (x[:, 0] + 1j * x[:, 1]).astype(np.complex128)
            else:
                arr = x.astype(np.complex128)
    elif isinstance(x, (list, tuple)):
        if len(x) == 2 and isinstance(x[0], (list, tuple, np.ndarray)) and isinstance(x[1], (list, tuple, np.ndarray)):
            arr = (np.asarray(x[0], dtype=np.float64) + 1j * np.asarray(x[1], dtype=np.float64)).astype(np.complex128)
        else:
            arr = np.asarray(x, dtype=np.complex128)
    else:
        arr = np.asarray(x, dtype=np.complex128)

    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


def _validate_finite_iq(arr: np.ndarray, name: str = "iq") -> None:
    """Ensures input complex array contains no NaNs or Infs."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Input '{name}' contains non-finite values (NaN or Inf)")


def _validate_sample_rate(fs: float) -> float:
    """Validates that sample rate is positive and finite."""
    fs_f = float(fs)
    if not np.isfinite(fs_f) or fs_f <= 0.0:
        raise ValueError(f"Sample rate must be positive and finite, got {fs}")
    return fs_f


def _sub_bin_peak_interpolation(
    mag_spec: np.ndarray,
    peak_idx: int,
    freqs: np.ndarray,
) -> tuple[float, float]:
    """Performs quadratic (parabolic) interpolation around peak spectral bin."""
    n = len(mag_spec)
    if peak_idx <= 0 or peak_idx >= n - 1:
        return float(freqs[peak_idx]), float(mag_spec[peak_idx])

    alpha = float(mag_spec[peak_idx - 1])
    beta = float(mag_spec[peak_idx])
    gamma = float(mag_spec[peak_idx + 1])

    denom = alpha - 2.0 * beta + gamma
    if abs(denom) < 1e-12:
        return float(freqs[peak_idx]), beta

    delta = 0.5 * (alpha - gamma) / denom
    peak_val = beta - 0.25 * (alpha - gamma) * delta
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    refined_freq = float(freqs[peak_idx] + delta * df)
    return refined_freq, peak_val


@dataclass
class OscillatorOffsetResult:
    """
    Result container for Carrier Frequency Offset (CFO) / Oscillator Offset.

    Attributes
    ----------
    oscillator_offset_hz : float
        Carrier Frequency Offset in Hz (measured_freq - expected_freq).
    normalized_offset : float
        CFO normalized by sample rate: f_offset / fs (cycles per sample).
    measured_frequency_hz : float | None
        Measured peak/center frequency in Hz.
    expected_frequency_hz : float | None
        Nominal or reference frequency in Hz.
    valid : bool
        True if the estimate is valid and non-corrupted.
    confidence : float
        Confidence score in [0.0, 1.0].
    method : str
        Algorithm utilized ('tone_peak', 'mth_power', 'autocorrelation', 'reference_aided').
    metadata : dict[str, Any]
        Diagnostics and computational metadata.
    """
    oscillator_offset_hz: float = 0.0
    normalized_offset: float = 0.0
    measured_frequency_hz: float | None = None
    expected_frequency_hz: float | None = None
    valid: bool = True
    confidence: float = 1.0
    method: str = "tone_peak"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frequency_offset_hz(self) -> float:
        return self.oscillator_offset_hz

    @property
    def cfo_hz(self) -> float:
        return self.oscillator_offset_hz

    @property
    def diagnostic(self) -> str | None:
        return self.metadata.get("diagnostic") or self.metadata.get("reason")


class OscillatorOffsetAnalyzer:
    """
    Oscillator Offset / Carrier Frequency Offset (CFO) analysis calculator.

    Estimates carrier frequency displacement from nominal / expected values
    due to local oscillator drift, Doppler shift, or tuning errors.
    """

    def __init__(self) -> None:
        self.result: OscillatorOffsetResult | None = None

    def compute(
        self,
        iq: Any,
        sample_rate: float = 1.0,
        expected_frequency_hz: float | None = None,
        method: str = "tone_peak",
        modulation_order_m: int = 4,
        reference_iq: Any | None = None,
        expected_frequency: float | None = None,
        modulation_order: int | None = None,
        iq_data: Any = None,
        **kwargs: Any,
    ) -> OscillatorOffsetResult:
        """
        Executes Carrier Frequency Offset (CFO) estimation.
        """
        data = iq if iq is not None else iq_data

        if expected_frequency_hz is None:
            if expected_frequency is not None:
                expected_frequency_hz = expected_frequency
            elif "expected_frequency_hz" in kwargs:
                expected_frequency_hz = kwargs["expected_frequency_hz"]

        if modulation_order is not None:
            modulation_order_m = modulation_order
        elif "modulation_order_m" in kwargs:
            modulation_order_m = kwargs["modulation_order_m"]

        fs = _validate_sample_rate(sample_rate)
        x = _as_complex_array(data, name="iq")
        _validate_finite_iq(x, name="iq")
        n = len(x)

        if n < 8:
            res = OscillatorOffsetResult(
                oscillator_offset_hz=0.0,
                normalized_offset=0.0,
                measured_frequency_hz=None,
                expected_frequency_hz=expected_frequency_hz,
                valid=False,
                confidence=0.0,
                method=method,
                metadata={"reason": "Sample count too small (< 8 samples)"},
            )
            self.result = res
            return res

        method_clean = str(method).lower().strip()
        meta: dict[str, Any] = {"sample_count": n}

        if method_clean in ("tone_peak", "calibration_tone", "tone"):
            if expected_frequency_hz is None:
                res = OscillatorOffsetResult(
                    oscillator_offset_hz=0.0,
                    normalized_offset=0.0,
                    measured_frequency_hz=None,
                    expected_frequency_hz=None,
                    valid=False,
                    confidence=0.0,
                    method="tone_peak",
                    metadata={"reason": "expected_frequency_hz is required for tone_peak CFO estimation to prevent arbitrary ambiguity"},
                )
                self.result = res
                return res

            # High-resolution windowed FFT
            nfft = max(2048, int(2 ** np.ceil(np.log2(n))))
            win = np.blackman(n)
            padded = np.pad(x * win, (0, nfft - n))
            spectrum = np.fft.fftshift(np.fft.fft(padded))
            mag_spec = np.abs(spectrum)
            freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))

            # Search in neighborhood around expected frequency (+/- 0.25 * fs)
            f_exp = float(expected_frequency_hz)
            search_mask = np.abs(freqs - f_exp) <= (0.25 * fs)
            if not np.any(search_mask):
                search_mask = np.ones(nfft, dtype=bool)

            masked_mag = np.where(search_mask, mag_spec, 0.0)
            peak_idx = int(np.argmax(masked_mag))

            f_measured, _ = _sub_bin_peak_interpolation(mag_spec, peak_idx, freqs)
            cfo_hz = float(f_measured - f_exp)
            norm_offset = float(cfo_hz / fs)

            confidence = float(np.clip(1.0 - 5.0 / np.sqrt(n), 0.2, 1.0))
            meta["f_measured_hz"] = f_measured
            meta["f_expected_hz"] = f_exp
            meta["nfft"] = nfft

            res = OscillatorOffsetResult(
                oscillator_offset_hz=cfo_hz,
                normalized_offset=norm_offset,
                measured_frequency_hz=f_measured,
                expected_frequency_hz=f_exp,
                valid=True,
                confidence=confidence,
                method="tone_peak",
                metadata=meta,
            )
            self.result = res
            return res

        elif method_clean in ("mth_power", "psk"):
            m = int(modulation_order_m)
            if m not in (2, 4, 8, 16):
                raise ValueError(f"modulation_order_m must be 2, 4, 8, or 16, got {m}")

            # Raise signal to m-th power: (x[n])^M strips M-PSK data modulation
            x_m = (x / (np.abs(x) + 1e-12)) ** m
            nfft = max(2048, int(2 ** np.ceil(np.log2(n))))
            win = np.hanning(n)
            padded = np.pad(x_m * win, (0, nfft - n))
            spec = np.fft.fftshift(np.fft.fft(padded))
            mag_spec = np.abs(spec)
            freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))

            peak_idx = int(np.argmax(mag_spec))
            f_m_peak, _ = _sub_bin_peak_interpolation(mag_spec, peak_idx, freqs)

            # CFO is peak frequency divided by M
            cfo_hz = float(f_m_peak / m)
            norm_offset = float(cfo_hz / fs)
            f_exp = float(expected_frequency_hz) if expected_frequency_hz is not None else 0.0
            f_measured = f_exp + cfo_hz

            meta["mth_peak_hz"] = f_m_peak
            meta["modulation_order"] = m

            res = OscillatorOffsetResult(
                oscillator_offset_hz=cfo_hz,
                normalized_offset=norm_offset,
                measured_frequency_hz=f_measured,
                expected_frequency_hz=expected_frequency_hz,
                valid=True,
                confidence=0.90,
                method=f"mth_power_M{m}",
                metadata=meta,
            )
            self.result = res
            return res

        elif method_clean in ("autocorrelation", "kay", "fitz"):
            # Lag-1 phase autocorrelation
            r1 = np.sum(x[1:] * np.conj(x[:-1]))
            angle_r1 = float(np.angle(r1))
            cfo_hz = float(angle_r1 * fs / (2.0 * np.pi))
            norm_offset = float(cfo_hz / fs)

            f_exp = float(expected_frequency_hz) if expected_frequency_hz is not None else 0.0
            f_measured = f_exp + cfo_hz

            meta["autocorr_lag1_angle_rad"] = angle_r1

            res = OscillatorOffsetResult(
                oscillator_offset_hz=cfo_hz,
                normalized_offset=norm_offset,
                measured_frequency_hz=f_measured,
                expected_frequency_hz=expected_frequency_hz,
                valid=True,
                confidence=0.85,
                method="autocorrelation",
                metadata=meta,
            )
            self.result = res
            return res

        elif method_clean == "reference_aided":
            if reference_iq is None:
                raise ValueError("reference_iq must be provided when method='reference_aided'")
            ref = _as_complex_array(reference_iq, name="reference_iq")
            _validate_finite_iq(ref, name="reference_iq")
            if len(ref) != n:
                raise ValueError(f"reference_iq length ({len(ref)}) != iq length ({n})")

            # Complex conjugate multiplication x[n] * conj(ref[n])
            mult = x * np.conj(ref)
            phase_diff = np.unwrap(np.angle(mult))
            t = np.arange(n) / fs

            # Linear regression on phase slope: phase(t) = 2*pi * cfo * t + theta_0
            poly = np.polyfit(t, phase_diff, deg=1)
            slope = float(poly[0])
            cfo_hz = float(slope / (2.0 * np.pi))
            norm_offset = float(cfo_hz / fs)

            f_exp = float(expected_frequency_hz) if expected_frequency_hz is not None else 0.0
            f_measured = f_exp + cfo_hz

            meta["phase_slope_rad_per_sec"] = slope

            res = OscillatorOffsetResult(
                oscillator_offset_hz=cfo_hz,
                normalized_offset=norm_offset,
                measured_frequency_hz=f_measured,
                expected_frequency_hz=expected_frequency_hz,
                valid=True,
                confidence=1.0,
                method="reference_aided",
                metadata=meta,
            )
            self.result = res
            return res

        else:
            raise ValueError(f"Unknown oscillator offset method '{method}'. Must be 'tone_peak', 'mth_power', 'autocorrelation', or 'reference_aided'")

    def analyze(self, *args: Any, **kwargs: Any) -> OscillatorOffsetResult:
        """Alias for compute()."""
        return self.compute(*args, **kwargs)


OscillatorOffsetEstimator = OscillatorOffsetAnalyzer


def compute_oscillator_offset(
    iq: Any = None,
    sample_rate: float = 1.0,
    expected_frequency_hz: float | None = None,
    method: str = "tone_peak",
    modulation_order_m: int = 4,
    reference_iq: Any | None = None,
    expected_frequency: float | None = None,
    modulation_order: int | None = None,
    iq_data: Any = None,
    **kwargs: Any,
) -> OscillatorOffsetResult:
    """
    Functional API to estimate Carrier Frequency Offset (CFO).
    """
    data = iq if iq is not None else iq_data
    analyzer = OscillatorOffsetAnalyzer()
    return analyzer.compute(
        iq=data,
        sample_rate=sample_rate,
        expected_frequency_hz=expected_frequency_hz,
        method=method,
        modulation_order_m=modulation_order_m,
        reference_iq=reference_iq,
        expected_frequency=expected_frequency,
        modulation_order=modulation_order,
        **kwargs,
    )


estimate_oscillator_offset = compute_oscillator_offset

__all__ = [
    "OscillatorOffsetResult",
    "OscillatorOffsetAnalyzer",
    "OscillatorOffsetEstimator",
    "compute_oscillator_offset",
    "estimate_oscillator_offset",
]


def main() -> None:
    print("=" * 60)
    print("  Oscillator Offset Module Demonstration")
    print("=" * 60)
    fs = 1e6
    n = 4096
    t = np.arange(n) / fs
    clean_tone = np.exp(1j * 2 * np.pi * 50e3 * t)  # 50 kHz tone

    # Injected +350 Hz CFO
    impaired = clean_tone * np.exp(1j * 2 * np.pi * 350.0 * t)

    analyzer = OscillatorOffsetAnalyzer()
    res = analyzer.compute(impaired, fs, expected_frequency_hz=50e3, method="tone_peak")
    assert res is analyzer.result

    print(f"Expected Frequency: {res.expected_frequency_hz:.1f} Hz")
    print(f"Measured Frequency: {res.measured_frequency_hz:.1f} Hz")
    print(f"Oscillator Offset (CFO): {res.oscillator_offset_hz:.2f} Hz (Injected: +350.0 Hz)")
    print(f"Normalized Offset: {res.normalized_offset:.8f} cycles/sample")
    print(f"Validity: {res.valid}, Confidence: {res.confidence:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
