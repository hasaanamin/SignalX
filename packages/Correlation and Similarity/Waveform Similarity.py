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

from .base import BaseCorrelationMetric, CorrelationMetricFactory, CorrelationResult
from ._common import _extract_sample_rate, _compute_xcorr_core


@CorrelationMetricFactory.register(["waveform_similarity", "rf_waveform_similarity", "pulse_similarity", "signal_similarity"])
class WaveformSimilarityMetric(BaseCorrelationMetric):
    """
    Waveform Similarity Stage based on MATLAB normalized cross-correlation.

    Mathematical Formulation:
      From normalized complex cross-correlation rho_xy[m] in [-1, 1]:
        S_waveform = max_m |rho_xy[m]|,   0 <= S_waveform <= 1

      Phase offset at optimal alignment:
        phi_offset = angle(rho_xy[m_peak])

      Lag & Delay:
        tau = m_peak / Fs
        delay = -m_peak / Fs (MATLAB finddelay convention)

    Note:
      Reuses existing CorrelationResult with zero duplicate transform steps.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "waveform_similarity"

    def compute(
        self,
        data: Any,
        y: Any = None,
        *,
        sample_rate: float | None = None,
        maxlag: int | None = None,
        method: str = "auto",
        as_dict: bool = False,
        **kwargs: Any,
    ) -> float | dict[str, Any]:
        """
        Computes the peak normalized waveform similarity between two complex signals.

        Parameters:
            data: First signal x[n] or existing CorrelationResult.
            y: Second signal y[n] (ignored if data is CorrelationResult).
            sample_rate (Fs): Sampling rate in Hz.
            maxlag: Maximum lag search range.
            as_dict: If True, returns full dictionary of similarity and alignment parameters.
                     If False, returns scalar similarity score S_waveform in [0, 1].

        Returns:
            Float scalar similarity score in [0, 1] or dictionary.
        """
        if isinstance(data, CorrelationResult):
            corr_res = data
        elif isinstance(data, dict) and "peak_magnitude" in data and "phase_at_peak_rad" in data:
            if as_dict:
                return {
                    "similarity": float(data["peak_magnitude"]),
                    "peak_lag_samples": data.get("peak_lag_samples", 0),
                    "peak_lag_sec": data.get("peak_lag_sec", 0.0),
                    "estimated_delay_samples": data.get("estimated_delay_samples", 0),
                    "estimated_delay_sec": data.get("estimated_delay_sec", 0.0),
                    "phase_offset_rad": data["phase_at_peak_rad"],
                    "phase_offset_deg": float(np.degrees(data["phase_at_peak_rad"])),
                    "sample_rate": data.get("sample_rate", 1.0),
                }
            return float(data["peak_magnitude"])
        else:
            fs = _extract_sample_rate(data=data, kwargs=kwargs, sample_rate=sample_rate, default_fs=1.0)
            corr_res = _compute_xcorr_core(
                data,
                y,
                maxlag=maxlag,
                scaleopt="coeff",
                method=method,
                sample_rate=fs,
            )

        if as_dict:
            return {
                "similarity": corr_res.peak_magnitude,
                "peak_lag_samples": corr_res.peak_lag_samples,
                "peak_lag_sec": corr_res.peak_lag_sec,
                "estimated_delay_samples": corr_res.estimated_delay_samples,
                "estimated_delay_sec": corr_res.estimated_delay_sec,
                "phase_offset_rad": corr_res.phase_at_peak_rad,
                "phase_offset_deg": float(np.degrees(corr_res.phase_at_peak_rad)),
                "sample_rate": corr_res.sample_rate,
            }

        return corr_res.peak_magnitude


def compute_waveform_similarity(
    x: Any,
    y: Any = None,
    *,
    sample_rate: float | None = None,
    maxlag: int | None = None,
    method: str = "auto",
    as_dict: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """
    Convenience functional API for waveform similarity computation.
    """
    metric = WaveformSimilarityMetric()
    return metric.compute(
        x,
        y,
        sample_rate=sample_rate,
        maxlag=maxlag,
        method=method,
        as_dict=as_dict,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  WaveformSimilarityMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    s1 = np.exp(1j * 2 * np.pi * 50e3 * t)
    s2 = np.roll(s1, 10) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    metric = WaveformSimilarityMetric()
    result = metric.compute(s1, s2, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
