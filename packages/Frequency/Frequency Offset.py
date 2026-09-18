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


@FrequencyMetricFactory.register(["frequency_offset", "cfo", "carrier_offset", "freq_offset"])
class FrequencyOffsetMetric(BaseFrequencyMetric):
    """
    Estimates Baseband Carrier Frequency Offset (CFO) for unknown modulation:
      - Modulation-independent spectral peak estimator with sub-bin parabolic interpolation.
      - If rf_center_hz is provided: estimated_rf_carrier = rf_center_hz + estimated_cfo.
      - Does not assume any modulation type, order, or constellation structure.
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes FrequencyOffsetMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up FrequencyOffsetMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "frequency_offset"

    def compute(
        self,
        iq_data,
        *,
        sample_rate: float | None = None,
        nominal_freq_hz: float = 0.0,
        rf_center_hz: float | None = None,
        return_info: bool = False,
        mode: str = "offset",
        **kwargs,
    ) -> float | dict:
        # Step 1: Format IQ array and extract required sample rate
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=None)
        if fs is None or not np.isfinite(fs) or fs <= 0:
            raise ValueError("A valid positive sample_rate (Fs) is required for frequency offset estimation.")

        n = z.size
        if n < 2:
            return 0.0 if not return_info else {"offset_hz": 0.0, "estimated_cfo_hz": 0.0}

        # Step 2: High-resolution FFT grid with zero-padding
        n_fft = max(2048, 1 << int(np.ceil(np.log2(n)) + 2))
        fft_vals = np.fft.fftshift(np.fft.fft(z, n=n_fft))
        psd = np.abs(fft_vals) ** 2
        freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / fs))

        # Step 3: Spectral peak location
        k_max = int(np.argmax(psd))

        # Step 4: 3-point parabolic sub-bin peak interpolation on log-PSD
        if 0 < k_max < n_fft - 1:
            y1 = float(np.log(max(1e-20, psd[k_max - 1])))
            y2 = float(np.log(max(1e-20, psd[k_max])))
            y3 = float(np.log(max(1e-20, psd[k_max + 1])))
            denom = 2.0 * (2.0 * y2 - y1 - y3)
            delta = (y3 - y1) / denom if abs(denom) > 1e-14 else 0.0
            delta = float(max(-0.5, min(0.5, delta)))
            df = float(freqs[1] - freqs[0])
            estimated_cfo_hz = float(freqs[k_max] + delta * df)
        else:
            estimated_cfo_hz = float(freqs[k_max])

        # Step 5: Offset relative to nominal / expected baseband carrier
        offset_hz = estimated_cfo_hz - float(nominal_freq_hz)

        # Step 6: Absolute RF carrier frequency if RF center is supplied
        estimated_rf_carrier = (float(rf_center_hz) + estimated_cfo_hz) if rf_center_hz is not None else None

        m = str(mode).lower()
        if return_info or m in ("all", "full", "dict"):
            return {
                "offset_hz": offset_hz,
                "estimated_cfo_hz": estimated_cfo_hz,
                "estimated_rf_carrier_hz": estimated_rf_carrier,
                "nominal_freq_hz": float(nominal_freq_hz),
                "rf_center_hz": float(rf_center_hz) if rf_center_hz is not None else None,
                "method": "modulation_independent_spectral_peak_interpolation",
            }

        return offset_hz


_instance = FrequencyOffsetMetric()
frequency_offset = _instance.compute
cfo = _instance.compute

__all__ = ["FrequencyOffsetMetric", "frequency_offset", "cfo"]


def main() -> None:
    print("=" * 60)
    print("  FrequencyOffsetMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = FrequencyOffsetMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
