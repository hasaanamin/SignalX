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


@SpectralMetricFactory.register(["spectral_rolloff", "spectral_roll_off", "rolloff_frequency", "spectral_cutoff"])
class SpectralRolloffMetric(BaseSpectralMetric):
    """
    Computes Spectral Roll-off Frequency:
      Frequency f_roll below which alpha (default 85% or 95%) of the total spectral power is contained:
        int_{-fs/2}^{f_roll} S(f) df = (alpha / 100) * int_{-fs/2}^{fs/2} S(f) df
    MATLAB Reference: `spectralRolloff(x, fs)`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "spectral_rolloff"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        percentage: float = 85.0,
        nperseg: int | None = None,
        window: str = "hann",
        as_dict: bool = False,
        **kwargs: Any,
    ) -> float | dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {"rolloff_frequency_hz": 0.0, "percentage": percentage} if as_dict else 0.0

        freqs, psd = _compute_welch_psd(z, fs=fs, nperseg=nperseg, window=window)
        df = fs / len(psd)

        cum_power = np.cumsum(psd * df)
        total_power = cum_power[-1]

        if total_power <= 0:
            return {"rolloff_frequency_hz": 0.0, "percentage": percentage} if as_dict else 0.0

        alpha = np.clip(percentage, 1.0, 99.99) / 100.0
        target_pwr = total_power * alpha

        # Linear interpolation of exact roll-off frequency
        f_rolloff = float(np.interp(target_pwr, cum_power, freqs))

        if as_dict:
            return {
                "rolloff_frequency_hz": f_rolloff,
                "percentage": float(percentage),
                "total_power_linear": float(total_power),
                "target_power_linear": float(target_pwr),
                "normalized_rolloff": float(f_rolloff / fs),
                "sample_rate": fs,
            }
        return f_rolloff


def spectral_rolloff(
    iq_data: Any,
    sample_rate: float | None = None,
    percentage: float = 85.0,
    nperseg: int | None = None,
    window: str = "hann",
    as_dict: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """
    Functional DSP wrapper for Spectral Roll-off computation.
    """
    metric = SpectralRolloffMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, percentage=percentage, nperseg=nperseg, window=window, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  SpectralRolloffMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = SpectralRolloffMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
