from __future__ import annotations
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    _pkg_dir = Path(__file__).resolve().parent
    _src_dir = _pkg_dir.parent.parent
    for _p in [str(_src_dir), str(_pkg_dir.parent)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import importlib
    importlib.import_module(f"packages.{_pkg_dir.name}")
    __package__ = f"packages.{_pkg_dir.name}"

from typing import Any
import numpy as np
from .base import BaseSpectralMetric, SpectralMetricFactory
from ._common import _as_iq_array, _extract_sample_rate, _compute_welch_psd


@SpectralMetricFactory.register(["spectral_peak", "spectral_peaks", "peak_frequency", "carrier_peak", "find_spectral_peaks"])
class SpectralPeakMetric(BaseSpectralMetric):
    """
    Detects and characterizes Spectral Peaks:
      - Global maximum carrier frequency f_peak and power level (linear, dBFS, dBm).
      - Sub-bin peak frequency refinement via parabolic interpolation on log-PSD.
      - Enumeration of local spectral peaks satisfying prominence and threshold criteria.
      - Peak-to-Average Spectral Ratio (PASR).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "spectral_peak"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        prominence_db: float = 6.0,
        max_peaks: int = 10,
        nperseg: int | None = None,
        window: str = "hann",
        as_dict: bool = False,
        **kwargs: Any,
    ) -> float | dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {"peak_frequency_hz": 0.0, "peak_power_dbfs": -np.inf, "peaks": []} if as_dict else 0.0

        freqs, psd = _compute_welch_psd(z, fs=fs, nperseg=nperseg, window=window)
        df = fs / len(psd)
        psd_db = 10.0 * np.log10(np.maximum(1e-24, psd))

        # 1. Global Peak
        global_idx = int(np.argmax(psd))
        f_global_bin = float(freqs[global_idx])

        # Parabolic interpolation for sub-bin peak accuracy
        if 0 < global_idx < len(psd) - 1:
            y0 = psd_db[global_idx - 1]
            y1 = psd_db[global_idx]
            y2 = psd_db[global_idx + 1]
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-12:
                delta = 0.5 * (y0 - y2) / denom
                delta = np.clip(delta, -0.5, 0.5)
            else:
                delta = 0.0
            f_global = float(f_global_bin + delta * df)
            p_global_db = float(y1 - 0.25 * (y0 - y2) * delta)
        else:
            f_global = f_global_bin
            p_global_db = float(psd_db[global_idx])

        # 2. Local Peak Finding with Prominence
        peaks = []
        noise_floor_db = float(np.median(psd_db))
        min_peak_db = noise_floor_db + prominence_db

        for i in range(1, len(psd) - 1):
            if psd_db[i] > psd_db[i - 1] and psd_db[i] > psd_db[i + 1]:
                if psd_db[i] >= min_peak_db:
                    # Compute local prominence
                    left_base = np.min(psd_db[max(0, i - 10) : i])
                    right_base = np.min(psd_db[i + 1 : min(len(psd_db), i + 11)])
                    prom = psd_db[i] - max(left_base, right_base)
                    if prom >= prominence_db:
                        peaks.append({
                            "frequency_hz": float(freqs[i]),
                            "power_dbfs": float(psd_db[i]),
                            "power_linear": float(psd[i]),
                            "prominence_db": float(prom),
                        })

        # Sort peaks descending by power
        peaks.sort(key=lambda p: p["power_dbfs"], reverse=True)
        peaks = peaks[:max_peaks]

        # Peak-to-Average Spectral Ratio
        mean_psd_linear = float(np.mean(psd))
        pasr_lin = psd[global_idx] / max(1e-24, mean_psd_linear)
        pasr_db = float(10.0 * np.log10(pasr_lin))

        if as_dict:
            return {
                "peak_frequency_hz": f_global,
                "peak_power_dbfs": p_global_db,
                "peak_power_dbm": float(p_global_db + 30.0),
                "peak_power_linear": float(psd[global_idx]),
                "num_peaks_detected": len(peaks),
                "peaks": peaks,
                "pasr_db": pasr_db,
                "sample_rate": fs,
            }
        return f_global


def spectral_peak(
    iq_data: Any,
    sample_rate: float | None = None,
    prominence_db: float = 6.0,
    max_peaks: int = 10,
    nperseg: int | None = None,
    window: str = "hann",
    as_dict: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """
    Functional DSP wrapper for Spectral Peak detection.
    """
    metric = SpectralPeakMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, prominence_db=prominence_db, max_peaks=max_peaks, nperseg=nperseg, window=window, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  SpectralPeakMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = SpectralPeakMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
