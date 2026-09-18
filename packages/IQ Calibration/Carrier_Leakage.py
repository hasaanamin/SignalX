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


@dataclass
class CarrierLeakageResult:
    """
    Result container for Residual Carrier / LO Leakage estimation.

    Attributes
    ----------
    leakage_power : float
        Spectral power integrated around zero-IF / DC in Watts.
    reference_signal_power : float
        Spectral power of the remainder of the baseband signal in Watts.
    leakage_ratio : float
        Linear ratio of carrier leakage power to signal power.
    leakage_db : float
        Carrier leakage relative to carrier / signal power in dBc.
    leakage_frequency_hz : float
        Measured frequency of the leakage peak in Hz (typically ~0 Hz).
    valid : bool
        True if the estimation is valid and uncorrupted.
    confidence : float
        Confidence score in [0.0, 1.0].
    metadata : dict[str, Any]
        Diagnostics and computational metadata.
    """
    leakage_power: float = 0.0
    reference_signal_power: float = 1.0
    leakage_ratio: float = 0.0
    leakage_db: float = -100.0
    leakage_frequency_hz: float = 0.0
    valid: bool = True
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def carrier_leakage_dbc(self) -> float:
        return self.leakage_db

    @property
    def diagnostic(self) -> str | None:
        return self.metadata.get("diagnostic") or self.metadata.get("reason")


class CarrierLeakageAnalyzer:
    """
    Residual Carrier / Local Oscillator (LO) Leakage analysis calculator.

    Quantifies the spectral power concentrated at or near zero-IF baseband (DC)
    relative to the total or reference signal power.

    Distinction
    -----------
    DC Offset is the time-domain mean/bias vector, whereas Carrier Leakage is the
    spectral power ratio concentrated at the carrier frequency relative to the signal.
    """

    def __init__(self) -> None:
        self.result: CarrierLeakageResult | None = None

    def compute(
        self,
        iq: Any,
        sample_rate: float = 1.0,
        leakage_bandwidth_hz: float | None = None,
        dc_offset: Any | None = None,
        dc_offset_result: Any | None = None,
        psd_result: Tuple[np.ndarray, np.ndarray] | None = None,
        iq_data: Any = None,
        **kwargs: Any,
    ) -> CarrierLeakageResult:
        """
        Estimates spectral carrier / LO leakage at zero-IF.
        """
        data = iq if iq is not None else iq_data
        dc_res = dc_offset if dc_offset is not None else dc_offset_result

        fs = _validate_sample_rate(sample_rate)
        x = _as_complex_array(data, name="iq")
        _validate_finite_iq(x, name="iq")
        n = len(x)

        if n < 8:
            res = CarrierLeakageResult(
                leakage_power=0.0,
                reference_signal_power=0.0,
                leakage_ratio=0.0,
                leakage_db=-100.0,
                leakage_frequency_hz=0.0,
                valid=False,
                confidence=0.0,
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

        # 2. Determine leakage integration bandwidth
        if leakage_bandwidth_hz is not None and leakage_bandwidth_hz > 0:
            bw_leakage = float(leakage_bandwidth_hz)
        else:
            bw_leakage = max(3.0 * df, 0.01 * fs)

        # Carrier leakage notch mask around 0 Hz
        dc_mask = np.abs(freqs) <= (bw_leakage / 2.0)
        sig_mask = ~dc_mask

        leakage_pwr = float(np.sum(psd[dc_mask]) * df) if np.any(dc_mask) else 0.0
        ref_sig_pwr = float(np.sum(psd[sig_mask]) * df) if np.any(sig_mask) else 0.0

        eps = 1e-25
        ref_sig_pwr_safe = max(ref_sig_pwr, eps)
        leakage_ratio = float(leakage_pwr / ref_sig_pwr_safe)
        leakage_db = float(10.0 * np.log10(leakage_ratio + eps))

        # Peak frequency within the DC notch
        if np.any(dc_mask):
            notch_psd = psd[dc_mask]
            notch_freqs = freqs[dc_mask]
            peak_idx = int(np.argmax(notch_psd))
            leakage_freq = float(notch_freqs[peak_idx])
        else:
            leakage_freq = 0.0

        meta = {
            "leakage_bandwidth_hz": bw_leakage,
            "df_hz": df,
            "total_spectrum_power": leakage_pwr + ref_sig_pwr,
            "notch_bins_count": int(np.sum(dc_mask)),
        }

        if dc_res is not None and hasattr(dc_res, "dc_power"):
            meta["time_domain_dc_power"] = getattr(dc_res, "dc_power")

        confidence = float(np.clip(1.0 - 5.0 / np.sqrt(n), 0.2, 1.0))

        res = CarrierLeakageResult(
            leakage_power=leakage_pwr,
            reference_signal_power=ref_sig_pwr,
            leakage_ratio=leakage_ratio,
            leakage_db=leakage_db,
            leakage_frequency_hz=leakage_freq,
            valid=True,
            confidence=confidence,
            metadata=meta,
        )
        self.result = res
        return res

    def analyze(self, *args: Any, **kwargs: Any) -> CarrierLeakageResult:
        """Alias for compute()."""
        return self.compute(*args, **kwargs)


CarrierLeakageEstimator = CarrierLeakageAnalyzer


def compute_carrier_leakage(
    iq: Any = None,
    sample_rate: float = 1.0,
    leakage_bandwidth_hz: float | None = None,
    dc_offset: Any | None = None,
    dc_offset_result: Any | None = None,
    psd_result: Tuple[np.ndarray, np.ndarray] | None = None,
    iq_data: Any = None,
    **kwargs: Any,
) -> CarrierLeakageResult:
    """
    Functional API to estimate residual Carrier / LO Leakage.
    """
    data = iq if iq is not None else iq_data
    analyzer = CarrierLeakageAnalyzer()
    return analyzer.compute(
        iq=data,
        sample_rate=sample_rate,
        leakage_bandwidth_hz=leakage_bandwidth_hz,
        dc_offset=dc_offset,
        dc_offset_result=dc_offset_result,
        psd_result=psd_result,
        **kwargs,
    )


estimate_carrier_leakage = compute_carrier_leakage

__all__ = [
    "CarrierLeakageResult",
    "CarrierLeakageAnalyzer",
    "CarrierLeakageEstimator",
    "compute_carrier_leakage",
    "estimate_carrier_leakage",
]


def main() -> None:
    print("=" * 60)
    print("  Carrier Leakage Module Demonstration")
    print("=" * 60)
    fs = 1e6
    n = 4096
    t = np.arange(n) / fs
    clean_sig = np.exp(1j * 2 * np.pi * 150e3 * t)  # signal at 150 kHz

    # Injected 0.1 V LO leakage at DC (power = 0.01 W)
    impaired = clean_sig + (0.1 + 0.0j)

    analyzer = CarrierLeakageAnalyzer()
    res = analyzer.compute(impaired, fs)
    assert res is analyzer.result

    print(f"Leakage Power: {res.leakage_power:.6f} W")
    print(f"Signal Power: {res.reference_signal_power:.6f} W")
    print(f"Leakage Ratio: {res.leakage_ratio:.6f}")
    print(f"Leakage in dB: {res.leakage_db:.2f} dBc")
    print(f"Validity: {res.valid}, Confidence: {res.confidence:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
