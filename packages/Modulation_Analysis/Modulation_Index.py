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
from ._results import ModulationIndexResult
from ._validated_helpers import (
    _analyze_envelope,
    _estimate_am_depth,
    _instantaneous_frequency,
    _center_frequency_deviation,
    _estimate_frequency_deviation,
    _detrended_phase,
    _estimate_phase_deviation,
    _dominant_modulation_frequency,
)


@ModulationMetricFactory.register(["modulation_index", "mod_index", "modulation_depth_or_index", "unified_modulation_index"])
class ModulationIndex(BaseModulationMetric):
    """
    Modulation Index Module.
    Independently computes the modulation index for AM (m in [0, 1+]), FM (beta = Delta_f / f_m),
    or PM (Delta_phi in radians) directly from raw complex IQ.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        mode: str = "auto",
        threshold_db: float = -60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.mode = str(mode).lower().strip()
        self.threshold_db = float(threshold_db)

    @property
    def name(self) -> str:
        return "modulation_index"

    def compute(
        self,
        iq_data: Any,
        sample_rate: float | None = None,
        **kwargs: Any,
    ) -> ModulationIndexResult:
        """Executes Modulation Index computation directly on raw complex IQ."""
        fs = float(sample_rate) if sample_rate is not None else self.sample_rate

        target_mode = self.mode

        if target_mode == "auto":
            # Determine dominant modulation mechanism via internal private helpers
            env_info = _analyze_envelope(iq_data, sample_rate=fs, threshold_db=self.threshold_db)
            fm_info = _instantaneous_frequency(iq_data, sample_rate=fs, threshold_db=self.threshold_db)
            _, f_dev = _center_frequency_deviation(fm_info["instantaneous_frequency_hz"], valid_mask=fm_info["valid_mask"])
            fm_stats = _estimate_frequency_deviation(f_dev, valid_mask=fm_info["valid_mask"])
            pm_info = _detrended_phase(iq_data, sample_rate=fs, threshold_db=self.threshold_db)
            pm_stats = _estimate_phase_deviation(pm_info["phase_deviation_rad"], valid_mask=pm_info["valid_mask"])

            am_act = env_info["am_activity_score"]
            fm_dev_hz = fm_stats["peak_deviation_hz"]
            pm_dev_rad = pm_stats["peak_deviation_rad"]

            if fm_dev_hz > 500.0 and fm_dev_hz > (fs * 0.005):
                target_mode = "fm"
            elif pm_dev_rad > 0.4 and am_act < 0.2:
                target_mode = "pm"
            elif am_act > 0.15:
                target_mode = "am"
            else:
                target_mode = "am"  # default fallback

        if target_mode == "am":
            env_info = _analyze_envelope(iq_data, sample_rate=fs, threshold_db=self.threshold_db)
            depth_info = _estimate_am_depth(env_info["envelope"], estimator="robust", valid_mask=env_info["valid_mask"])
            return ModulationIndexResult(
                modulation_type="AM",
                modulation_index=depth_info["modulation_depth"],
                definition="m = (A_max - A_min) / (A_max + A_min)",
                units="dimensionless",
                valid=depth_info["valid"],
                confidence=depth_info["confidence"],
                reason=depth_info["reason"],
            )

        elif target_mode == "fm":
            fm_info = _instantaneous_frequency(
                iq_data,
                sample_rate=fs,
                smoothing="none",
                smoothing_window=5,
                threshold_db=self.threshold_db,
            )
            f_inst = fm_info["instantaneous_frequency_hz"]
            valid_mask = fm_info["valid_mask"]
            _, f_dev = _center_frequency_deviation(f_inst, valid_mask=valid_mask)
            dev_stats = _estimate_frequency_deviation(f_dev, valid_mask=valid_mask)
            peak_dev = dev_stats["peak_deviation_hz"]

            if peak_dev < 1.0:
                return ModulationIndexResult(
                    modulation_type="FM",
                    modulation_index=0.0,
                    definition="beta = Delta_f / f_m",
                    units="dimensionless",
                    valid=True,
                    confidence=0.9,
                    reason="Unmodulated carrier (zero FM index).",
                )

            dom_info = _dominant_modulation_frequency(f_dev, sample_rate=fs, valid_mask=valid_mask)
            if dom_info["valid"] and dom_info["frequency_hz"] is not None and dom_info["frequency_hz"] > 0:
                fm = float(dom_info["frequency_hz"])
                tone_peak = float(2.0 * dom_info["peak_strength"])
                beta = float(tone_peak / fm)
                return ModulationIndexResult(
                    modulation_type="FM",
                    modulation_index=beta,
                    definition="beta = Delta_f / f_m",
                    units="dimensionless",
                    valid=True,
                    confidence=float(dom_info["confidence"]),
                    reason=f"Estimated FM index beta = {beta:.2f} (Delta_f = {tone_peak:.1f} Hz, f_m = {fm:.1f} Hz).",
                )

            return ModulationIndexResult(
                modulation_type="FM",
                modulation_index=None,
                definition="beta = Delta_f / f_m",
                units="dimensionless",
                valid=False,
                confidence=0.3,
                reason="No distinct modulating tone detected in FM deviation waveform.",
            )

        elif target_mode == "pm":
            pm_info = _detrended_phase(iq_data, sample_rate=fs, threshold_db=self.threshold_db)
            phase_dev = pm_info["phase_deviation_rad"]
            valid_mask = pm_info["valid_mask"]
            dev_stats = _estimate_phase_deviation(phase_dev, valid_mask=valid_mask)
            peak_phi = dev_stats["peak_deviation_rad"]

            return ModulationIndexResult(
                modulation_type="PM",
                modulation_index=peak_phi,
                definition="beta_p = Delta_phi",
                units="radians",
                valid=pm_info["valid_sample_fraction"] > 0.3,
                confidence=float(np.clip(pm_info["valid_sample_fraction"], 0.0, 0.95)),
                reason=f"Phase deviation index Delta_phi = {peak_phi:.3f} rad ({dev_stats['peak_deviation_deg']:.1f} deg).",
            )

        else:
            return ModulationIndexResult(
                modulation_type=target_mode.upper(),
                modulation_index=None,
                definition="N/A",
                units="N/A",
                valid=False,
                confidence=0.0,
                reason=f"Digital modulation type '{target_mode}' does not have a single standard analog modulation index.",
            )


def compute_modulation_index(
    iq_data: Any,
    sample_rate: float = 1.0,
    **kwargs: Any,
) -> ModulationIndexResult:
    """Functional convenience wrapper for ModulationIndex."""
    return ModulationIndex(sample_rate=sample_rate, **kwargs).compute(iq_data)


def main() -> None:
    print("=" * 60)
    print("  ModulationIndex: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = ModulationIndex()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
