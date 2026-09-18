from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Tuple
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


def _compute_psd_welch(
    x: np.ndarray,
    fs: float,
    nfft: int = 2048,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes two-sided Welch Power Spectral Density for complex signals."""
    n = len(x)
    nperseg = min(nfft, n)
    noverlap = int(nperseg * overlap)
    step = nperseg - noverlap

    if step <= 0 or n < nperseg:
        win = np.hanning(n)
        scale = 1.0 / (fs * np.sum(win ** 2))
        spec = np.fft.fftshift(np.fft.fft(x * win, n=nfft))
        psd = (np.abs(spec) ** 2) * scale
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
        return freqs, psd

    win = np.hanning(nperseg)
    win_scale = 1.0 / (fs * np.sum(win ** 2))
    num_segments = (n - noverlap) // step
    psd_accum = np.zeros(nfft, dtype=np.float64)

    for i in range(num_segments):
        start = i * step
        segment = x[start : start + nperseg] * win
        spec = np.fft.fftshift(np.fft.fft(segment, n=nfft))
        psd_accum += (np.abs(spec) ** 2) * win_scale

    psd = psd_accum / float(num_segments)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    return freqs, psd


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
class ImageRejectionResult:
    """
    Result container for Image Rejection Ratio (IRR) analysis.

    Attributes
    ----------
    desired_power : float
        Integrated power of the desired signal in Watts.
    image_power : float
        Integrated power in the mirror image band in Watts.
    image_rejection_ratio_linear : float
        Linear ratio of desired power to mirror image power (P_desired / P_image).
    image_rejection_db : float
        Image Rejection Ratio in decibels: 10 * log10(P_desired / P_image).
    desired_frequency_hz : float | None
        Center frequency of the desired signal in Hz.
    image_frequency_hz : float | None
        Center frequency of the mirror image (-desired_frequency) in Hz.
    confidence : float
        Confidence score in [0.0, 1.0].
    valid : bool
        True if the estimate is valid and non-corrupted.
    metadata : dict[str, Any]
        Diagnostics and computational metadata.
    """
    desired_power: float = 1.0
    image_power: float = 1e-10
    image_rejection_ratio_linear: float = 1e10
    image_rejection_db: float = 100.0
    desired_frequency_hz: float | None = None
    image_frequency_hz: float | None = None
    confidence: float = 1.0
    valid: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def theoretical_irr_db(self) -> float | None:
        return self.metadata.get("theoretical_irr_db")

    @property
    def diagnostic(self) -> str | None:
        return self.metadata.get("diagnostic") or self.metadata.get("reason")


class ImageRejectionAnalyzer:
    """
    Image Rejection Ratio (IRR) analysis calculator.

    Measures desired signal power, mirror image power, and Image Rejection Ratio
    (IRR in linear and decibels) across the complex baseband spectrum.

    Definition
    ----------
    For a desired signal centered at frequency +f_d, the mirror image created by
    I/Q gain and phase imbalance appears at -f_d.
    IRR_dB = 10 * log10(P_desired / P_image).
    """

    def __init__(self) -> None:
        self.result: ImageRejectionResult | None = None

    def compute(
        self,
        iq: Any,
        sample_rate: float,
        expected_signal_frequency_hz: float | None = None,
        signal_bandwidth_hz: float | None = None,
        oscillator_offset: Any | None = None,
        oscillator_offset_result: Any | None = None,
        gain_imbalance: Any | None = None,
        gain_imbalance_result: Any | None = None,
        phase_imbalance: Any | None = None,
        phase_imbalance_result: Any | None = None,
        psd_result: Tuple[np.ndarray, np.ndarray] | None = None,
        tone_frequency: float | None = None,
        tone_frequency_hz: float | None = None,
        expected_frequency: float | None = None,
        **kwargs: Any,
    ) -> ImageRejectionResult:
        """
        Executes Image Rejection Ratio (IRR) estimation.
        """
        if expected_signal_frequency_hz is None:
            if tone_frequency is not None:
                expected_signal_frequency_hz = tone_frequency
            elif tone_frequency_hz is not None:
                expected_signal_frequency_hz = tone_frequency_hz
            elif expected_frequency is not None:
                expected_signal_frequency_hz = expected_frequency
            elif "expected_frequency_hz" in kwargs:
                expected_signal_frequency_hz = kwargs["expected_frequency_hz"]

        cfo_res = oscillator_offset if oscillator_offset is not None else oscillator_offset_result
        gain_res = gain_imbalance if gain_imbalance is not None else gain_imbalance_result
        phase_res = phase_imbalance if phase_imbalance is not None else phase_imbalance_result

        fs = _validate_sample_rate(sample_rate)
        x = _as_complex_array(iq, name="iq")
        _validate_finite_iq(x, name="iq")
        n = len(x)

        if n < 8:
            res = ImageRejectionResult(
                desired_power=0.0,
                image_power=0.0,
                image_rejection_ratio_linear=1.0,
                image_rejection_db=0.0,
                desired_frequency_hz=None,
                image_frequency_hz=None,
                confidence=0.0,
                valid=False,
                metadata={"reason": "Sample count too small (< 8 samples)"},
            )
            self.result = res
            return res

        # 1. Compute or reuse Welch PSD
        if psd_result is not None:
            freqs, psd = psd_result
        else:
            freqs, psd = _compute_psd_welch(x, fs, nfft=min(2048, n))

        df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

        # 2. Determine desired signal center frequency
        if expected_signal_frequency_hz is not None:
            f_desired = float(expected_signal_frequency_hz)
            if cfo_res is not None and getattr(cfo_res, "valid", False):
                f_desired += getattr(cfo_res, "oscillator_offset_hz", 0.0)
        else:
            # Find dominant non-DC spectral peak
            dc_mask = np.abs(freqs) < (3.0 * df)
            masked_psd = psd.copy()
            masked_psd[dc_mask] = 0.0

            peak_idx = int(np.argmax(masked_psd))
            f_desired = float(freqs[peak_idx])

        f_image = -f_desired

        # 3. Determine integration bandwidth
        if signal_bandwidth_hz is not None and signal_bandwidth_hz > 0:
            bw = float(signal_bandwidth_hz)
        else:
            bw = float(max(5.0 * df, 0.05 * abs(f_desired) if abs(f_desired) > 1e-3 else 0.05 * fs))

        # 4. Integrate power in desired band: [f_desired - bw/2, f_desired + bw/2]
        desired_mask = (freqs >= (f_desired - bw / 2.0)) & (freqs <= (f_desired + bw / 2.0))
        p_desired = float(np.sum(psd[desired_mask]) * df) if np.any(desired_mask) else 0.0

        # 5. Integrate power in mirror image band: [f_image - bw/2, f_image + bw/2]
        image_mask = (freqs >= (f_image - bw / 2.0)) & (freqs <= (f_image + bw / 2.0))
        p_image = float(np.sum(psd[image_mask]) * df) if np.any(image_mask) else 0.0

        # 6. Calculate IRR
        eps = 1e-20
        p_desired_safe = max(p_desired, eps)
        p_image_safe = max(p_image, eps)
        irr_linear = float(p_desired / p_image_safe)
        irr_db = float(10.0 * np.log10(irr_linear))

        meta: dict[str, Any] = {
            "integration_bandwidth_hz": bw,
            "df_hz": df,
            "f_desired_hz": f_desired,
            "f_image_hz": f_image,
        }

        # Theoretical IRR comparison
        if gain_res is not None and hasattr(gain_res, "gain_ratio"):
            g = float(gain_res.gain_ratio)
        else:
            p_i = float(np.mean((x.real - np.mean(x.real)) ** 2))
            p_q = float(np.mean((x.imag - np.mean(x.imag)) ** 2))
            g = float(np.sqrt(p_i / (p_q + 1e-20)) if p_q > 1e-20 else 1.0)

        if phase_res is not None and hasattr(phase_res, "phase_imbalance_rad"):
            phi = float(phase_res.phase_imbalance_rad)
        else:
            i_ac = x.real - np.mean(x.real)
            q_ac = x.imag - np.mean(x.imag)
            denom = np.mean(i_ac ** 2) + np.mean((q_ac * g) ** 2)
            sin_phi = float(np.clip(2.0 * np.mean(i_ac * q_ac * g) / (denom + 1e-20), -1.0, 1.0))
            phi = float(np.arcsin(sin_phi))

        # Theoretical formula: IRR = (1 + 2*g*cos(phi) + g^2) / (1 - 2*g*cos(phi) + g^2)
        denom_th = 1.0 - 2.0 * g * np.cos(phi) + g ** 2
        num_th = 1.0 + 2.0 * g * np.cos(phi) + g ** 2
        if denom_th > 1e-20:
            irr_th_linear = float(num_th / denom_th)
            meta["theoretical_irr_db"] = float(10.0 * np.log10(irr_th_linear))
        else:
            meta["theoretical_irr_db"] = 120.0

        confidence = float(np.clip(1.0 - (5.0 / np.sqrt(n)), 0.2, 1.0))

        res = ImageRejectionResult(
            desired_power=p_desired,
            image_power=p_image,
            image_rejection_ratio_linear=irr_linear,
            image_rejection_db=irr_db,
            desired_frequency_hz=f_desired,
            image_frequency_hz=f_image,
            confidence=confidence,
            valid=True,
            metadata=meta,
        )
        self.result = res
        return res

    def analyze(self, *args: Any, **kwargs: Any) -> ImageRejectionResult:
        """Alias for compute()."""
        return self.compute(*args, **kwargs)


ImageRejectionEstimator = ImageRejectionAnalyzer


def compute_image_rejection(
    iq: Any = None,
    sample_rate: float = 1.0,
    expected_signal_frequency_hz: float | None = None,
    signal_bandwidth_hz: float | None = None,
    oscillator_offset: Any | None = None,
    oscillator_offset_result: Any | None = None,
    gain_imbalance: Any | None = None,
    gain_imbalance_result: Any | None = None,
    phase_imbalance: Any | None = None,
    phase_imbalance_result: Any | None = None,
    psd_result: Tuple[np.ndarray, np.ndarray] | None = None,
    tone_frequency: float | None = None,
    tone_frequency_hz: float | None = None,
    expected_frequency: float | None = None,
    iq_data: Any = None,
    **kwargs: Any,
) -> ImageRejectionResult:
    """
    Functional API to estimate Image Rejection Ratio (IRR).
    """
    data = iq if iq is not None else iq_data
    analyzer = ImageRejectionAnalyzer()
    return analyzer.compute(
        iq=data,
        sample_rate=sample_rate,
        expected_signal_frequency_hz=expected_signal_frequency_hz,
        signal_bandwidth_hz=signal_bandwidth_hz,
        oscillator_offset=oscillator_offset,
        oscillator_offset_result=oscillator_offset_result,
        gain_imbalance=gain_imbalance,
        gain_imbalance_result=gain_imbalance_result,
        phase_imbalance=phase_imbalance,
        phase_imbalance_result=phase_imbalance_result,
        psd_result=psd_result,
        tone_frequency=tone_frequency,
        tone_frequency_hz=tone_frequency_hz,
        expected_frequency=expected_frequency,
        **kwargs,
    )


estimate_image_rejection = compute_image_rejection

__all__ = [
    "ImageRejectionResult",
    "ImageRejectionAnalyzer",
    "ImageRejectionEstimator",
    "compute_image_rejection",
    "estimate_image_rejection",
]


def main() -> None:
    print("=" * 60)
    print("  Image Rejection Module Demonstration")
    print("=" * 60)
    fs = 1e6
    n = 4096
    t = np.arange(n) / fs
    clean_tone = np.exp(1j * 2 * np.pi * 100e3 * t)  # desired tone at +100 kHz

    # Inject gain and phase imbalance manually
    i_sig = clean_tone.real * 10.0 ** (0.5 / 20.0)
    phi = np.radians(4.0)
    q_sig = (clean_tone.real * np.sin(phi) + clean_tone.imag * np.cos(phi)) * 10.0 ** (-0.5 / 20.0)
    impaired = i_sig + 1j * q_sig

    analyzer = ImageRejectionAnalyzer()
    res = analyzer.compute(impaired, fs, expected_signal_frequency_hz=100e3)
    assert res is analyzer.result

    print(f"Desired Frequency: {res.desired_frequency_hz / 1e3:.1f} kHz")
    print(f"Image Frequency: {res.image_frequency_hz / 1e3:.1f} kHz")
    print(f"Desired Power: {res.desired_power:.6f} W")
    print(f"Image Power: {res.image_power:.8f} W")
    print(f"Image Rejection Ratio: {res.image_rejection_db:.2f} dB")
    print(f"Validity: {res.valid}, Confidence: {res.confidence:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
