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
from ._results import PMDeviationResult
from ._validated_helpers import (
    _detrended_phase,
    _estimate_phase_deviation,
    _dominant_modulation_frequency,
)
import math


@ModulationMetricFactory.register(["pm_deviation", "phase_deviation", "pm_dev", "pm_deviation_estimator"])
class PMDeviation(BaseModulationMetric):
    """
    PM Deviation Module.
    Independently estimates peak, positive, negative, peak-to-peak, and RMS phase deviation
    in both radians and degrees directly from raw complex IQ.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        fit_method: str = "robust",
        use_robust_extrema: bool = False,
        threshold_db: float = -60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.fit_method = str(fit_method)
        self.use_robust_extrema = bool(use_robust_extrema)
        self.threshold_db = float(threshold_db)

    @property
    def name(self) -> str:
        return "pm_deviation"

    def compute(
        self,
        iq_data: Any,
        sample_rate: float | None = None,
        **kwargs: Any,
    ) -> PMDeviationResult:
        """Executes PM Deviation estimation directly on raw complex IQ."""
        fs = float(sample_rate) if sample_rate is not None else self.sample_rate

        pm_info = _detrended_phase(
            iq_data,
            sample_rate=fs,
            fit_method=self.fit_method,
            threshold_db=self.threshold_db,
        )

        phase_dev = pm_info["phase_deviation_rad"]
        valid_mask = pm_info["valid_mask"]

        dev_stats = _estimate_phase_deviation(
            phase_dev,
            valid_mask=valid_mask,
            use_robust_extrema=self.use_robust_extrema,
        )

        # Check for dominant sinusoidal modulating tone for canonical agreement with ModulationIndex
        dom_info = _dominant_modulation_frequency(
            phase_dev,
            sample_rate=fs,
            valid_mask=valid_mask,
        )

        if dom_info["valid"] and dom_info["frequency_hz"] is not None and dom_info["frequency_hz"] > 0:
            tone_peak = float(2.0 * dom_info["peak_strength"])
            peak_dev = tone_peak
            pos_dev = tone_peak
            neg_dev = tone_peak
            p2p_dev = 2.0 * tone_peak
            rms_dev = float(tone_peak / math.sqrt(2.0))
            peak_deg = float(np.degrees(tone_peak))
            conf = float(dom_info["confidence"])
        else:
            peak_dev = dev_stats["peak_deviation_rad"]
            pos_dev = dev_stats["positive_deviation_rad"]
            neg_dev = dev_stats["negative_deviation_rad"]
            p2p_dev = dev_stats["peak_to_peak_deviation_rad"]
            rms_dev = dev_stats["rms_deviation_rad"]
            peak_deg = dev_stats["peak_deviation_deg"]
            conf = float(np.clip(pm_info["valid_sample_fraction"], 0.0, 1.0))

        return PMDeviationResult(
            peak_deviation_rad=peak_dev,
            positive_deviation_rad=pos_dev,
            negative_deviation_rad=neg_dev,
            peak_to_peak_deviation_rad=p2p_dev,
            rms_deviation_rad=rms_dev,
            peak_deviation_deg=peak_deg,
            valid_sample_fraction=pm_info["valid_sample_fraction"],
            confidence=conf,
        )


def compute_pm_deviation(
    iq_data: Any,
    sample_rate: float = 1.0,
    **kwargs: Any,
) -> PMDeviationResult:
    """Functional convenience wrapper for PMDeviation."""
    return PMDeviation(sample_rate=sample_rate, **kwargs).compute(iq_data)


def main() -> None:
    print("=" * 60)
    print("  PMDeviation: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = PMDeviation()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
