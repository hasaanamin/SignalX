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
from ._results import FMAnalysisResult
from ._validated_helpers import (
    _instantaneous_frequency,
    _center_frequency_deviation,
    _estimate_frequency_deviation,
    _dominant_modulation_frequency,
)


@ModulationMetricFactory.register(["fm_analysis", "frequency_modulation", "fm_analyzer", "fm"])
class FMAnalysis(BaseModulationMetric):
    """
    FM Analysis Module.
    Independently analyzes frequency modulation directly from raw complex IQ.
    Extracts instantaneous phase, instantaneous frequency, frequency deviation,
    center frequency, deviation moments, and dominant modulation frequency.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        center_method: str = "robust",
        smoothing: str = "none",
        smoothing_window: int = 5,
        threshold_db: float = -60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        c_val = kwargs.get("center_frequency_method", center_method)
        c_clean = str(c_val).lower().strip()
        if c_clean not in ("median", "mean", "robust", "trimmed", "trimmed_mean"):
            raise ValueError(f"Unknown center frequency method '{c_val}'.")
        sm_clean = str(smoothing).lower().strip()
        if sm_clean not in ("none", "moving_average", "gaussian", "savgol"):
            raise ValueError(f"Unknown smoothing mode '{smoothing}'.")

        self.center_method = c_clean
        self.smoothing = sm_clean
        self.smoothing_window = int(smoothing_window)
        self.threshold_db = float(threshold_db)

    @property
    def name(self) -> str:
        return "fm_analysis"

    def compute(
        self,
        iq_data: Any,
        sample_rate: float | None = None,
        **kwargs: Any,
    ) -> FMAnalysisResult:
        """Executes FM analysis directly on raw complex IQ."""
        fs = float(sample_rate) if sample_rate is not None else self.sample_rate

        fm_info = _instantaneous_frequency(
            iq_data,
            sample_rate=fs,
            smoothing=self.smoothing,
            smoothing_window=self.smoothing_window,
            threshold_db=self.threshold_db,
        )

        f_inst = fm_info["instantaneous_frequency_hz"]
        valid_mask = fm_info["valid_mask"]
        inst_phase = fm_info["instantaneous_phase_rad"]
        times_sec = fm_info["times_sec"]

        f0, f_dev = _center_frequency_deviation(
            f_inst,
            valid_mask=valid_mask,
            method=self.center_method,
        )

        dev_stats = _estimate_frequency_deviation(f_dev, valid_mask=valid_mask)

        # Dominant modulating frequency
        if len(f_dev) >= 16 and dev_stats["rms_deviation_hz"] > 10.0 and np.any(valid_mask):
            dom_info = _dominant_modulation_frequency(
                f_dev,
                sample_rate=fs,
                valid_mask=valid_mask,
            )
            dom_freq = dom_info["frequency_hz"] if dom_info["valid"] else None
            conf = float(dom_info["confidence"]) if dom_info["valid"] else (0.95 if fm_info["valid_sample_fraction"] > 0.5 else 0.0)
        else:
            dom_freq = None
            conf = 0.95 if fm_info["valid_sample_fraction"] > 0.5 else 0.0

        if not np.any(valid_mask) or fm_info["valid_sample_fraction"] == 0.0:
            fm_activity = 0.0
            conf = 0.0
        elif dev_stats["rms_deviation_hz"] < 1e-3:
            fm_activity = 0.0
        else:
            act_dev = f_dev[valid_mask] if np.any(valid_mask) else np.array([])
            if act_dev.size > 0:
                mad = float(np.median(np.abs(act_dev - np.median(act_dev))))
                continuity = float(np.clip(mad / max(1e-12, 0.25 * dev_stats["rms_deviation_hz"]), 0.0, 1.0))
                if dom_freq is not None and dom_info["valid"]:
                    raw_act = float(np.clip(conf * (dev_stats["rms_deviation_hz"] / max(50.0, dev_stats["rms_deviation_hz"] + 500.0)) * 1.5, 0.0, 1.0))
                    fm_activity = raw_act * continuity
                else:
                    fm_activity = float(np.clip(mad / max(1.0, fs * 0.05), 0.0, 0.35)) * continuity
            else:
                fm_activity = 0.0

        return FMAnalysisResult(
            instantaneous_phase_rad=inst_phase,
            instantaneous_frequency_hz=f_inst,
            frequency_deviation_hz=f_dev,
            times_sec=times_sec,
            valid_mask=valid_mask,
            center_frequency_hz=f0,
            mean_deviation_hz=dev_stats["mean_deviation_hz"],
            std_deviation_hz=dev_stats["std_deviation_hz"],
            rms_deviation_hz=dev_stats["rms_deviation_hz"],
            dominant_modulation_frequency_hz=dom_freq,
            fm_activity_score=fm_activity,
            valid_sample_fraction=fm_info["valid_sample_fraction"],
            confidence=conf,
            sample_rate=fs,
            smoothing=self.smoothing,
        )


def compute_fm_analysis(
    iq_data: Any,
    sample_rate: float = 1.0,
    **kwargs: Any,
) -> FMAnalysisResult:
    """Functional convenience wrapper for FMAnalysis."""
    return FMAnalysis(sample_rate=sample_rate, **kwargs).compute(iq_data)


def fm_analysis_summary(data: Any, sample_rate: float = 1.0, **kwargs: Any) -> dict[str, Any]:
    """Generates human-readable summary dictionary of FM analysis results from IQ data or FMAnalysisResult."""
    if isinstance(data, FMAnalysisResult):
        res = data
    else:
        res = compute_fm_analysis(data, sample_rate=sample_rate, **kwargs)
    return {
        "instantaneous_frequency_hz": res.instantaneous_frequency_hz,
        "frequency_deviation_hz": res.frequency_deviation_hz,
        "instantaneous_phase_rad": res.instantaneous_phase_rad,
        "center_frequency_hz": res.center_frequency_hz,
        "mean_deviation_hz": res.mean_deviation_hz,
        "std_deviation_hz": res.std_deviation_hz,
        "rms_deviation_hz": res.rms_deviation_hz,
        "dominant_modulation_frequency_hz": res.dominant_modulation_frequency_hz,
        "fm_activity_score": res.fm_activity_score,
        "valid_sample_fraction": res.valid_sample_fraction,
        "confidence": res.confidence,
    }


def main() -> None:
    print("=" * 60)
    print("  FMAnalysis: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = FMAnalysis()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
