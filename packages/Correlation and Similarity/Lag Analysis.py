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


@CorrelationMetricFactory.register(["lag_analysis", "finddelay", "delay_estimation", "lag_estimation", "time_delay"])
class LagAnalysisMetric(BaseCorrelationMetric):
    """
    Lag Analysis and Delay Estimation Stage matching MATLAB `finddelay(x, y)`.

    MATLAB Mathematical Convention:
      Cross-correlation peak lag:
        m_p = argmax_m |rho_xy[m]|
      Estimated delay (MATLAB finddelay):
        D_samples = -m_p
        D_sec = -m_p / Fs

      Phase at peak:
        phi_peak = angle(rho_xy[m_p])

    Note:
      A positive delay D means y is delayed relative to x (y lags x).
      This metric directly reuses existing CorrelationResult without recomputing xcorr.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "lag_analysis"

    def compute(
        self,
        data: Any,
        y: Any = None,
        *,
        sample_rate: float | None = None,
        maxlag: int | None = None,
        scaleopt: str = "coeff",
        method: str = "auto",
        as_dict: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any] | tuple[int, float]:
        """
        Estimates the time lag and delay between two signals matching MATLAB finddelay.

        Parameters:
            data: First signal x[n] or existing CorrelationResult.
            y: Second signal y[n] (ignored if data is CorrelationResult).
            sample_rate (Fs): Sampling rate in Hz.
            maxlag: Maximum lag search range.
            as_dict: If True (default), returns dictionary of lag/delay parameters.
                     If False, returns tuple of (estimated_delay_samples, estimated_delay_sec).

        Returns:
            Dictionary or tuple of delay estimates.
        """
        if isinstance(data, CorrelationResult):
            corr_res = data
        elif isinstance(data, dict) and "estimated_delay_samples" in data and "peak_lag_samples" in data:
            if as_dict:
                return data
            return (data["estimated_delay_samples"], data.get("estimated_delay_sec", float(data["estimated_delay_samples"])))
        else:
            fs = _extract_sample_rate(data=data, kwargs=kwargs, sample_rate=sample_rate, default_fs=1.0)
            corr_res = _compute_xcorr_core(
                data,
                y,
                maxlag=maxlag,
                scaleopt=scaleopt,
                method=method,
                sample_rate=fs,
            )

        if as_dict:
            return {
                "estimated_delay_samples": corr_res.estimated_delay_samples,
                "estimated_delay_sec": corr_res.estimated_delay_sec,
                "peak_lag_samples": corr_res.peak_lag_samples,
                "peak_lag_sec": corr_res.peak_lag_sec,
                "peak_magnitude": corr_res.peak_magnitude,
                "phase_at_peak_rad": corr_res.phase_at_peak_rad,
                "phase_at_peak_deg": float(np.degrees(corr_res.phase_at_peak_rad)),
                "sample_rate": corr_res.sample_rate,
            }

        return (corr_res.estimated_delay_samples, corr_res.estimated_delay_sec)


def compute_lag_analysis(
    x: Any,
    y: Any = None,
    *,
    sample_rate: float | None = None,
    maxlag: int | None = None,
    scaleopt: str = "coeff",
    method: str = "auto",
    as_dict: bool = True,
    **kwargs: Any,
) -> dict[str, Any] | tuple[int, float]:
    """
    Convenience functional API for lag analysis and delay estimation matching MATLAB `finddelay`.
    """
    metric = LagAnalysisMetric()
    return metric.compute(
        x,
        y,
        sample_rate=sample_rate,
        maxlag=maxlag,
        scaleopt=scaleopt,
        method=method,
        as_dict=as_dict,
        **kwargs,
    )


def finddelay(
    x: Any,
    y: Any,
    *,
    sample_rate: float | None = None,
    maxlag: int | None = None,
    **kwargs: Any,
) -> int:
    """
    Exact MATLAB `finddelay(x, y)` wrapper returning integer delay samples.
    """
    res = compute_lag_analysis(x, y, sample_rate=sample_rate, maxlag=maxlag, as_dict=True, **kwargs)
    return int(res["estimated_delay_samples"])


def main() -> None:
    print("=" * 60)
    print("  LagAnalysisMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    s1 = np.exp(1j * 2 * np.pi * 50e3 * t)
    s2 = np.roll(s1, 10) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    metric = LagAnalysisMetric()
    result = metric.compute(s1, s2, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
