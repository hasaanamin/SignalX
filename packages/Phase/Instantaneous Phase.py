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
from ._common import _as_iq_array, _compute_phase_arrays


@PhaseMetricFactory.register(["instantaneous_phase", "phase", "phase_vs_time"])
class InstantaneousPhaseMetric(BasePhaseMetric):
    """
    Computes Instantaneous Phase theta(t) = atan2(Q, I):
      - Wrapped: [-pi, pi) rad or [-180, 180) deg
      - Unwrapped: Continuous phase trajectory across sample transitions
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes InstantaneousPhaseMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up InstantaneousPhaseMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "instantaneous_phase"

    def compute(self, iq_data, *, unwrap: bool = False, deg: bool = False, **kwargs) -> np.ndarray:
        # Step 1: Extract complex array
        z = _as_iq_array(iq_data)
        
        # Step 2: Calculate phase array (wrapped or unwrapped, rad or deg)
        phase_array = _compute_phase_arrays(z, unwrap=unwrap, deg=deg)
        
        return phase_array


_instance = InstantaneousPhaseMetric()
instantaneous_phase = _instance.compute

__all__ = ["InstantaneousPhaseMetric", "instantaneous_phase"]


def main() -> None:
    print("=" * 60)
    print("  InstantaneousPhaseMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = InstantaneousPhaseMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
