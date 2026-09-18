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
from ._common import _as_iq_array


@PhaseMetricFactory.register(["phase_unwrapping", "unwrap_phase", "unwrapped_phase"])
class PhaseUnwrappingMetric(BasePhaseMetric):
    """
    Performs Phase Unwrapping on instantaneous phase to eliminate 2*pi modulo discontinuities:
      phi_unwrapped = unwrap(angle(z), discont=pi)
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes PhaseUnwrappingMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up PhaseUnwrappingMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "phase_unwrapping"

    def compute(
        self,
        iq_data,
        *,
        discont: float = np.pi,
        deg: bool = False,
        **kwargs,
    ) -> np.ndarray:
        # Step 1: Format IQ array or raw phase input
        if isinstance(iq_data, dict) or (isinstance(iq_data, tuple) and len(iq_data) == 2 and not np.isscalar(iq_data[0])) or np.iscomplexobj(iq_data):
            z = _as_iq_array(iq_data)
            raw_phase_rad = np.angle(z)
        else:
            # Input is already a 1D real phase array
            raw_phase = np.asarray(iq_data, dtype=np.float64).reshape(-1)
            raw_phase_rad = np.radians(raw_phase) if deg else raw_phase

        # Step 2: Unwrap phase discontinuities
        unwrapped_rad = np.unwrap(raw_phase_rad, discont=discont)

        # Step 3: Convert to degrees if requested
        if deg:
            return np.degrees(unwrapped_rad)

        return unwrapped_rad


_instance = PhaseUnwrappingMetric()
phase_unwrapping = _instance.compute
unwrap_phase = _instance.compute

__all__ = ["PhaseUnwrappingMetric", "phase_unwrapping", "unwrap_phase"]


def main() -> None:
    print("=" * 60)
    print("  PhaseUnwrappingMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = PhaseUnwrappingMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
