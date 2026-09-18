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
from .base import BaseNoiseMetric, NoiseMetricFactory
from ._common import _as_iq_array, _extract_sample_rate


@NoiseMetricFactory.register(["amplitude_noise", "am_noise", "residual_am", "envelope_noise", "amplitude_jitter"])
class AmplitudeNoiseMetric(BaseNoiseMetric):
    """
    Computes Amplitude Noise / Residual AM:
      - Residual AM (% RMS): (std(A) / mean(A)) * 100%
      - Peak-to-Peak Amplitude Ripple: (max(A) - min(A)) / mean(A)
      - Amplitude SNR (dB): 20 * log10(mean(A) / std(A))
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "amplitude_noise"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        as_dict: bool = False,
        **kwargs: Any,
    ) -> float | dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {"residual_am_pct": 0.0, "amplitude_snr_db": np.inf} if as_dict else 0.0

        amp = np.abs(z)
        mean_amp = float(np.mean(amp))
        std_amp = float(np.std(amp))

        if mean_amp <= 0:
            residual_am_pct = 0.0
            am_snr_db = 0.0
            p2p_ripple = 0.0
        else:
            residual_am_pct = float((std_amp / mean_amp) * 100.0)
            am_snr_lin = mean_amp / max(1e-12, std_amp)
            am_snr_db = float(20.0 * np.log10(am_snr_lin))
            p2p_ripple = float((np.max(amp) - np.min(amp)) / mean_amp)

        if as_dict:
            return {
                "residual_am_pct": residual_am_pct,
                "amplitude_snr_db": am_snr_db,
                "peak_to_peak_ripple": p2p_ripple,
                "mean_amplitude": mean_amp,
                "std_amplitude": std_amp,
                "max_amplitude": float(np.max(amp)),
                "min_amplitude": float(np.min(amp)),
                "sample_rate": fs,
            }
        return residual_am_pct


def amplitude_noise(
    iq_data: Any,
    sample_rate: float | None = None,
    as_dict: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """
    Functional DSP wrapper for amplitude noise / residual AM calculation.
    """
    metric = AmplitudeNoiseMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  AmplitudeNoiseMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = AmplitudeNoiseMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
