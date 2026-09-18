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
from ._results import AMAnalysisResult
from ._validated_helpers import (
    _analyze_envelope,
    _dominant_modulation_frequency,
)


@ModulationMetricFactory.register(["am_analysis", "amplitude_modulation", "am_analyzer", "am"])
class AMAnalysis(BaseModulationMetric):
    """
    AM Analysis Module.
    Independently analyzes amplitude modulation directly from raw complex IQ.
    Extracts envelope behavior, reference amplitude, RMS envelope variation,
    AM activity score, and dominant modulation frequency.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        ref_method: str = "robust",
        smoothing: str = "none",
        smoothing_window: int = 5,
        threshold_db: float = -60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        ref_val = kwargs.get("reference_method", ref_method)
        ref_clean = str(ref_val).lower().strip()
        if ref_clean not in ("robust", "median", "mean", "trimmed", "trimmed_mean"):
            raise ValueError(f"Unknown reference method '{ref_val}'.")
        sm_clean = str(smoothing).lower().strip()
        if sm_clean not in ("none", "moving_average", "gaussian", "savgol"):
            raise ValueError(f"Unknown smoothing mode '{smoothing}'.")

        self.ref_method = ref_clean
        self.smoothing = sm_clean
        self.smoothing_window = int(smoothing_window)
        self.threshold_db = float(threshold_db)

    @property
    def name(self) -> str:
        return "am_analysis"

    def compute(
        self,
        iq_data: Any,
        sample_rate: float | None = None,
        **kwargs: Any,
    ) -> AMAnalysisResult:
        """Executes AM analysis directly on raw complex IQ."""
        fs = float(sample_rate) if sample_rate is not None else self.sample_rate

        env_info = _analyze_envelope(
            iq_data,
            sample_rate=fs,
            ref_method=self.ref_method,
            smoothing=self.smoothing,
            smoothing_window=self.smoothing_window,
            threshold_db=self.threshold_db,
        )

        env = env_info["envelope"]
        valid_mask = env_info["valid_mask"]
        rec_mod = env_info["recovered_modulation"]

        # Dominant modulating frequency
        if len(rec_mod) >= 16 and env_info["am_activity_score"] > 0.01:
            dom_info = _dominant_modulation_frequency(
                rec_mod,
                sample_rate=fs,
                valid_mask=valid_mask,
            )
            dom_freq = dom_info["frequency_hz"] if dom_info["valid"] else None
            conf = float(dom_info["confidence"]) if dom_info["valid"] else (0.95 if env_info["valid_sample_fraction"] > 0.5 else 0.0)
        else:
            dom_freq = None
            conf = 0.95 if env_info["valid_sample_fraction"] > 0.5 else 0.0

        return AMAnalysisResult(
            envelope=env,
            normalized_envelope=env_info["normalized_envelope"],
            recovered_modulation=rec_mod,
            times_sec=env_info["times_sec"],
            valid_mask=valid_mask,
            reference_amplitude=env_info["reference_amplitude"],
            envelope_mean=env_info["envelope_mean"],
            envelope_std=env_info["envelope_std"],
            envelope_rms_variation=env_info["envelope_rms_variation"],
            envelope_peak_to_peak=env_info["envelope_peak_to_peak"],
            dominant_modulation_frequency_hz=dom_freq,
            am_activity_score=env_info["am_activity_score"],
            valid_sample_fraction=env_info["valid_sample_fraction"],
            confidence=conf,
            sample_rate=fs,
            smoothing=self.smoothing,
        )


def compute_am_analysis(
    iq_data: Any,
    sample_rate: float = 1.0,
    **kwargs: Any,
) -> AMAnalysisResult:
    """Functional convenience wrapper for AMAnalysis."""
    return AMAnalysis(sample_rate=sample_rate, **kwargs).compute(iq_data)


def am_analysis_summary(data: Any, sample_rate: float = 1.0, **kwargs: Any) -> dict[str, Any]:
    """Generates human-readable summary dictionary of AM analysis results from IQ data or AMAnalysisResult."""
    if isinstance(data, AMAnalysisResult):
        res = data
    else:
        res = compute_am_analysis(data, sample_rate=sample_rate, **kwargs)
    return {
        "envelope": res.envelope,
        "normalized_envelope": res.normalized_envelope,
        "recovered_modulation": res.recovered_modulation,
        "reference_amplitude": res.reference_amplitude,
        "envelope_mean": res.envelope_mean,
        "envelope_std": res.envelope_std,
        "envelope_rms_variation": res.envelope_rms_variation,
        "envelope_peak_to_peak": res.envelope_peak_to_peak,
        "dominant_modulation_frequency_hz": res.dominant_modulation_frequency_hz,
        "am_activity_score": res.am_activity_score,
        "valid_sample_fraction": res.valid_sample_fraction,
        "confidence": res.confidence,
    }


def main() -> None:
    print("=" * 60)
    print("  AMAnalysis: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = AMAnalysis()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
