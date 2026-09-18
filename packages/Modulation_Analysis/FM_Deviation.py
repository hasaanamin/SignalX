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
from ._results import FMDeviationResult
from ._validated_helpers import (
    _instantaneous_frequency,
    _center_frequency_deviation,
    _estimate_frequency_deviation,
    _dominant_modulation_frequency,
)
import math


@ModulationMetricFactory.register(["fm_deviation", "frequency_deviation", "fm_dev", "fm_deviation_estimator"])
class FMDeviation(BaseModulationMetric):
    """
    FM Deviation Module.
    Independently estimates peak, positive, negative, peak-to-peak, and RMS frequency deviation
    directly from raw complex IQ.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        center_method: str = "robust",
        smoothing: str = "none",
        smoothing_window: int = 5,
        use_robust_extrema: bool = False,
        threshold_db: float = -60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.center_method = str(center_method)
        self.smoothing = str(smoothing)
        self.smoothing_window = int(smoothing_window)
        self.use_robust_extrema = bool(use_robust_extrema)
        self.threshold_db = float(threshold_db)

    @property
    def name(self) -> str:
        return "fm_deviation"

    def compute(
        self,
        iq_data: Any,
        sample_rate: float | None = None,
        **kwargs: Any,
    ) -> FMDeviationResult:
        """Executes FM Deviation estimation directly on raw complex IQ."""
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

        f0, f_dev = _center_frequency_deviation(
            f_inst,
            valid_mask=valid_mask,
            method=self.center_method,
        )

        dev_stats = _estimate_frequency_deviation(
            f_dev,
            valid_mask=valid_mask,
            use_robust_extrema=self.use_robust_extrema,
        )

        # Check for dominant sinusoidal modulating tone for canonical agreement with FMIndex
        dom_info = _dominant_modulation_frequency(
            f_dev,
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
            conf = float(dom_info["confidence"])
        else:
            peak_dev = dev_stats["peak_deviation_hz"]
            pos_dev = dev_stats["positive_deviation_hz"]
            neg_dev = dev_stats["negative_deviation_hz"]
            p2p_dev = dev_stats["peak_to_peak_deviation_hz"]
            rms_dev = dev_stats["rms_deviation_hz"]
            conf = float(np.clip(fm_info["valid_sample_fraction"], 0.0, 1.0))

        return FMDeviationResult(
            peak_deviation_hz=peak_dev,
            positive_deviation_hz=pos_dev,
            negative_deviation_hz=neg_dev,
            peak_to_peak_deviation_hz=p2p_dev,
            rms_deviation_hz=rms_dev,
            center_frequency_hz=f0,
            valid_sample_fraction=fm_info["valid_sample_fraction"],
            confidence=conf,
        )


def compute_fm_deviation(
    iq_data: Any,
    sample_rate: float = 1.0,
    **kwargs: Any,
) -> FMDeviationResult:
    """Functional convenience wrapper for FMDeviation."""
    return FMDeviation(sample_rate=sample_rate, **kwargs).compute(iq_data)


def main() -> None:
    print("=" * 60)
    print("  FMDeviation: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = FMDeviation()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
