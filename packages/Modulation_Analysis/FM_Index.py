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
import math
import numpy as np

from .base import BaseModulationMetric, ModulationMetricFactory
from ._results import FMIndexResult
from ._validated_helpers import (
    _instantaneous_frequency,
    _center_frequency_deviation,
    _estimate_frequency_deviation,
    _dominant_modulation_frequency,
)


@ModulationMetricFactory.register(["fm_index", "fm_modulation_index", "modulation_index_fm", "beta_fm"])
class FMIndex(BaseModulationMetric):
    """
    FM Index Module.
    Independently computes the FM modulation index (beta = Delta_f / f_m) directly from raw complex IQ.
    Distinguishes single-tone sinusoidal FM from arbitrary/wideband FM or noise.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        center_method: str = "median",
        smoothing: str = "none",
        smoothing_window: int = 5,
        threshold_db: float = -60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.center_method = str(center_method)
        self.smoothing = str(smoothing)
        self.smoothing_window = int(smoothing_window)
        self.threshold_db = float(threshold_db)

    @property
    def name(self) -> str:
        return "fm_index"

    def compute(
        self,
        iq_data: Any,
        sample_rate: float | None = None,
        **kwargs: Any,
    ) -> FMIndexResult:
        """Executes FM Modulation Index estimation directly on raw complex IQ."""
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

        if len(f_inst) < 16 or fm_info["valid_sample_fraction"] < 0.2:
            return FMIndexResult(
                modulation_index=None,
                peak_deviation_hz=0.0,
                modulation_frequency_hz=None,
                valid=False,
                interpretation="Insufficient active signal samples.",
                confidence=0.0,
            )

        f0, f_dev = _center_frequency_deviation(
            f_inst,
            valid_mask=valid_mask,
            method=self.center_method,
        )

        dev_stats = _estimate_frequency_deviation(f_dev, valid_mask=valid_mask)
        peak_dev = dev_stats["peak_deviation_hz"]

        if peak_dev < 1.0:
            return FMIndexResult(
                modulation_index=0.0,
                peak_deviation_hz=peak_dev,
                modulation_frequency_hz=None,
                valid=True,
                interpretation="Unmodulated carrier tone (zero FM index).",
                confidence=0.9,
            )

        # Estimate dominant modulation tone fm
        dom_info = _dominant_modulation_frequency(
            f_dev,
            sample_rate=fs,
            valid_mask=valid_mask,
        )

        if dom_info["valid"] and dom_info["frequency_hz"] is not None and dom_info["frequency_hz"] > 0:
            fm = float(dom_info["frequency_hz"])
            # For sinusoidal tone FM, Fourier domain single-sided peak strength * 2.0 gives exact peak tone deviation without out-of-band noise bias
            tone_peak = float(2.0 * dom_info["peak_strength"])
            beta = float(tone_peak / fm)
            conf = float(dom_info["confidence"])
            interp = "single_tone_sinusoidal_fm"
            valid = True
            peak_dev = tone_peak
        else:
            fm = None
            beta = None
            conf = 0.3
            interp = "wideband_or_diffuse_fm_no_single_tone"
            valid = False

        return FMIndexResult(
            modulation_index=beta,
            peak_deviation_hz=peak_dev,
            modulation_frequency_hz=fm,
            valid=valid,
            interpretation=interp,
            confidence=conf,
        )


def compute_fm_index(
    iq_data: Any,
    sample_rate: float = 1.0,
    **kwargs: Any,
) -> FMIndexResult:
    """Functional convenience wrapper for FMIndex."""
    return FMIndex(sample_rate=sample_rate, **kwargs).compute(iq_data)


def main() -> None:
    print("=" * 60)
    print("  FMIndex: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = FMIndex()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
