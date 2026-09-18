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

import numpy as np
from .base import BasePhaseMetric, PhaseMetricFactory
from ._common import _as_iq_array, _extract_sample_rate


@PhaseMetricFactory.register(["phase_discontinuity", "phase_jump", "phase_transitions", "phase_slip"])
class PhaseDiscontinuityMetric(BasePhaseMetric):
    """
    Detects Phase Discontinuities matching MATLAB reference:
      1. phase_step = angle(z[1:] * conj(z[:-1]))
      2. median_step = median(phase_step)
      3. mad = median(|phase_step - median_step|)
      4. robust_sigma = 1.4826 * mad
      5. threshold = max(pi/2, 6 * robust_sigma)
      6. Count samples where |phase_step - median_step| > threshold
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes PhaseDiscontinuityMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up PhaseDiscontinuityMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "phase_discontinuity"

    def compute(
        self,
        iq_data,
        *,
        sample_rate: float | None = None,
        return_count_only: bool = False,
        deg: bool = False,
        **kwargs,
    ) -> int | dict:
        # Step 1: Format IQ array and sample rate
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n < 2:
            return 0 if return_count_only else {"discontinuity_count": 0, "sample_indices": [], "timestamps_seconds": []}

        # Step 2: Phase step calculation
        phase_step = np.angle(z[1:] * np.conj(z[:-1]))

        # Step 3: Median and robust MAD estimation
        median_step = float(np.median(phase_step))
        mad = float(np.median(np.abs(phase_step - median_step)))
        robust_sigma = float(1.4826 * mad)

        # Step 4: Threshold = max(pi/2, 6 * robust_sigma)
        threshold = float(max(np.pi / 2.0, 6.0 * robust_sigma))

        # Step 5: Detect jump events
        jumps_mask = np.abs(phase_step - median_step) > threshold
        jump_indices = (np.where(jumps_mask)[0] + 1).tolist()
        discontinuity_count = int(np.sum(jumps_mask))
        jump_timestamps = [(idx / fs) for idx in jump_indices]

        if return_count_only:
            return discontinuity_count

        return {
            "discontinuity_count": discontinuity_count,
            "sample_indices": jump_indices,
            "timestamps_seconds": jump_timestamps,
            "threshold": float(np.degrees(threshold)) if deg else threshold,
            "median_step": float(np.degrees(median_step)) if deg else median_step,
            "robust_sigma": float(np.degrees(robust_sigma)) if deg else robust_sigma,
            "unit": "degrees" if deg else "radians",
        }


_instance = PhaseDiscontinuityMetric()
phase_discontinuity = _instance.compute

__all__ = ["PhaseDiscontinuityMetric", "phase_discontinuity"]


def main() -> None:
    print("=" * 60)
    print("  PhaseDiscontinuityMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = PhaseDiscontinuityMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
