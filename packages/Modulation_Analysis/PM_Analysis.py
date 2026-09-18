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

from .base import BaseModulationMetric, ModulationMetricFactory
from ._results import PMAnalysisResult
from ._validated_helpers import (
    _detrended_phase,
    _estimate_phase_deviation,
    _dominant_modulation_frequency,
)


@ModulationMetricFactory.register(["pm_analysis", "phase_modulation", "pm_analyzer", "pm"])
class PMAnalysis(BaseModulationMetric):
    """
    PM Analysis Module.
    Independently analyzes phase modulation directly from raw complex IQ.
    Recovers unwrapped phase, linear carrier phase progression trend,
    demodulated PM waveform, RMS phase deviation, and dominant modulation frequency.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        fit_method: str = "robust",
        smoothing: str = "none",
        smoothing_window: int = 5,
        threshold_db: float = -60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        fit_val = kwargs.get("fit_method", fit_method)
        fit_clean = str(fit_val).lower().strip()
        if fit_clean not in ("robust", "linear", "ols", "mean", "median", "amplitude_weighted", "weighted"):
            raise ValueError(f"Unknown fit method '{fit_val}'.")
        sm_clean = str(smoothing).lower().strip()
        if sm_clean not in ("none", "moving_average", "gaussian", "savgol"):
            raise ValueError(f"Unknown smoothing mode '{smoothing}'.")

        self.fit_method = fit_clean
        self.smoothing = sm_clean
        self.smoothing_window = int(kwargs.get("smoothing_length", smoothing_window))
        self.threshold_db = float(threshold_db)

    @property
    def name(self) -> str:
        return "pm_analysis"

    def compute(
        self,
        iq_data: Any,
        sample_rate: float | None = None,
        **kwargs: Any,
    ) -> PMAnalysisResult:
        """Executes PM analysis directly on raw complex IQ."""
        fs = float(sample_rate) if sample_rate is not None else self.sample_rate

        pm_info = _detrended_phase(
            iq_data,
            sample_rate=fs,
            fit_method=self.fit_method,
            threshold_db=self.threshold_db,
            smoothing=self.smoothing,
            smoothing_window=self.smoothing_window,
        )

        unwrapped = pm_info["unwrapped_phase_rad"]
        trend = pm_info["phase_trend_rad"]
        phase_dev = pm_info["phase_deviation_rad"]
        valid_mask = pm_info["valid_mask"]
        times_sec = pm_info["times_sec"]

        dev_stats = _estimate_phase_deviation(phase_dev, valid_mask=valid_mask)

        # Dominant modulating frequency
        if len(phase_dev) >= 16 and dev_stats["rms_deviation_rad"] > 0.05 and np.any(valid_mask):
            dom_info = _dominant_modulation_frequency(
                phase_dev,
                sample_rate=fs,
                valid_mask=valid_mask,
            )
            dom_freq = dom_info["frequency_hz"] if dom_info["valid"] else None
            conf = float(dom_info["confidence"]) if dom_info["valid"] else (0.95 if pm_info["valid_sample_fraction"] > 0.5 else 0.0)
        else:
            dom_freq = None
            conf = 0.95 if pm_info["valid_sample_fraction"] > 0.5 else 0.0

        if not np.any(valid_mask) or pm_info["valid_sample_fraction"] == 0.0:
            pm_activity = 0.0
            conf = 0.0
        elif dev_stats["rms_deviation_rad"] < 1e-3:
            pm_activity = 0.0
        else:
            pm_activity = float(np.clip(dev_stats["rms_deviation_rad"] / 1.0, 0.0, 1.0))

        return PMAnalysisResult(
            unwrapped_phase_rad=unwrapped,
            phase_trend_rad=trend,
            phase_deviation_rad=phase_dev,
            times_sec=times_sec,
            valid_mask=valid_mask,
            estimated_carrier_frequency_hz=pm_info["estimated_carrier_frequency_hz"],
            constant_phase_rad=pm_info["constant_phase_rad"],
            rms_phase_deviation_rad=dev_stats["rms_deviation_rad"],
            phase_deviation_std_rad=dev_stats["phase_deviation_std_rad"],
            dominant_modulation_frequency_hz=dom_freq,
            pm_activity_score=pm_activity,
            valid_sample_fraction=pm_info["valid_sample_fraction"],
            confidence=conf,
            sample_rate=fs,
            smoothing=self.smoothing,
        )


def compute_pm_analysis(
    iq_data: Any,
    sample_rate: float = 1.0,
    **kwargs: Any,
) -> PMAnalysisResult:
    """Functional convenience wrapper for PMAnalysis."""
    return PMAnalysis(sample_rate=sample_rate, **kwargs).compute(iq_data)


def pm_analysis_summary(data: Any, sample_rate: float = 1.0, **kwargs: Any) -> dict[str, Any]:
    """Generates human-readable summary dictionary of PM analysis results from IQ data or PMAnalysisResult."""
    if isinstance(data, PMAnalysisResult):
        res = data
    else:
        res = compute_pm_analysis(data, sample_rate=sample_rate, **kwargs)
    return {
        "unwrapped_phase_rad": res.unwrapped_phase_rad,
        "phase_trend_rad": res.phase_trend_rad,
        "phase_deviation_rad": res.phase_deviation_rad,
        "estimated_carrier_frequency_hz": res.estimated_carrier_frequency_hz,
        "constant_phase_rad": res.constant_phase_rad,
        "phase_deviation_std_rad": res.phase_deviation_std_rad,
        "rms_phase_deviation_rad": res.rms_phase_deviation_rad,
        "dominant_modulation_frequency_hz": res.dominant_modulation_frequency_hz,
        "pm_activity_score": res.pm_activity_score,
        "valid_sample_fraction": res.valid_sample_fraction,
        "confidence": res.confidence,
    }


def main() -> None:
    print("=" * 60)
    print("  PMAnalysis: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = PMAnalysis()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
