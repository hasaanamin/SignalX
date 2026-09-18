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

import numpy as np
from .base import BaseFrequencyMetric, FrequencyMetricFactory
from ._common import _as_iq_array, _extract_sample_rate


@FrequencyMetricFactory.register(["center_frequency", "spectral_centroid", "rf_frequency", "carrier_frequency", "center_freq"])
class CenterFrequencyMetric(BaseFrequencyMetric):
    """
    Computes Spectral Centroid and Center/RF Frequency for arbitrary unknown modulation:
      - Spectral Centroid: sum(f * PSD) / sum(PSD)
      - FFT Peak Frequency: argmax(PSD)
      - Absolute RF Carrier Frequency: reported only when RF tuning/reference frequency (rf_lo_hz) is available.
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes CenterFrequencyMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up CenterFrequencyMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "center_frequency"

    def compute(
        self,
        iq_data,
        *,
        sample_rate: float | None = None,
        method: str = "spectral_centroid",
        rf_lo_hz: float | None = None,
        rf_center_hz: float | None = None,
        mode: str = "centroid",
        **kwargs,
    ) -> float | dict:
        # Step 1: Format IQ array and extract required sample rate
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=None)
        if fs is None or not np.isfinite(fs) or fs <= 0:
            raise ValueError("A valid positive sample_rate (Fs) is required for center frequency calculation.")

        n = z.size
        if n < 1:
            return 0.0

        # Step 2: Compute power spectrum (PSD)
        fft_vals = np.fft.fftshift(np.fft.fft(z))
        psd = np.abs(fft_vals) ** 2
        freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))

        # Step 3: Spectral Centroid = sum(f * PSD) / sum(PSD)
        total_power = np.sum(psd)
        if total_power <= 0:
            spectral_centroid_hz = 0.0
        else:
            spectral_centroid_hz = float(np.sum(freqs * psd) / total_power)

        # Step 4: FFT Peak
        max_idx = int(np.argmax(psd))
        fft_peak_hz = float(freqs[max_idx])

        # Step 5: RF frequency if tuning reference is provided
        rf_ref = rf_lo_hz if rf_lo_hz is not None else rf_center_hz
        rf_carrier_hz = (float(rf_ref) + spectral_centroid_hz) if rf_ref is not None else None

        m = str(method).lower()
        if m in ("fft_peak", "peak", "max"):
            baseband_result = fft_peak_hz
        else:
            baseband_result = spectral_centroid_hz

        final_val = (baseband_result + float(rf_ref)) if rf_ref is not None else baseband_result

        out_mode = str(mode).lower()
        if out_mode in ("all", "dict", "full"):
            return {
                "spectral_centroid_hz": spectral_centroid_hz,
                "fft_peak_hz": fft_peak_hz,
                "estimated_rf_carrier_hz": rf_carrier_hz,
                "rf_reference_hz": float(rf_ref) if rf_ref is not None else None,
                "unit": "Hz",
            }

        return float(final_val)


_instance = CenterFrequencyMetric()
center_frequency = _instance.compute
spectral_centroid = _instance.compute
rf_frequency = _instance.compute

__all__ = ["CenterFrequencyMetric", "center_frequency", "spectral_centroid", "rf_frequency"]


def main() -> None:
    print("=" * 60)
    print("  CenterFrequencyMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = CenterFrequencyMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
