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


@PhaseMetricFactory.register(["phase_difference", "pdoa", "delta_phase", "differential_phase"])
class PhaseDifferenceMetric(BasePhaseMetric):
    """
    Computes Phase Difference matching MATLAB reference:
      - Single signal inter-sample phase difference:
          first value = NaN
          remaining values = angle(z[n] * conj(z[n-1]))
      - Two-channel cross phase difference (PDOA / Interferometry):
          diff = angle(z1 * conj(z2))
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes PhaseDifferenceMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up PhaseDifferenceMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "phase_difference"

    def compute(
        self,
        iq_data,
        z2=None,
        *,
        deg: bool = False,
        return_scalar_mean: bool = False,
        **kwargs,
    ) -> float | np.ndarray:
        # Step 1: Format first signal
        z1 = _as_iq_array(iq_data)
        n = z1.size

        if z2 is not None:
            # Step 2a: Two-channel cross-phase difference
            z2_arr = _as_iq_array(z2)
            if z1.size != z2_arr.size:
                raise ValueError(f"Signal size mismatch: z1 has {z1.size}, z2 has {z2_arr.size}")
            
            cross_product = z1 * np.conj(z2_arr)
            diff_array = np.angle(cross_product)
        else:
            # Step 2b: Single signal inter-sample difference matching MATLAB exactly
            if n < 1:
                return np.array([], dtype=np.float64)
            if n == 1:
                diff_array = np.array([np.nan], dtype=np.float64)
            else:
                diff_array = np.empty(n, dtype=np.float64)
                diff_array[0] = np.nan
                diff_array[1:] = np.angle(z1[1:] * np.conj(z1[:-1]))

        # Step 3: Convert to degrees if requested
        if deg:
            diff_array = np.degrees(diff_array)

        # Step 4: Return mean scalar or full array
        if return_scalar_mean:
            if z2 is not None:
                coherent_sum = np.sum(z1 * np.conj(_as_iq_array(z2)))
                mean_val = float(np.angle(coherent_sum))
                return float(np.degrees(mean_val)) if deg else mean_val
            return float(np.nanmean(diff_array))

        return diff_array


_instance = PhaseDifferenceMetric()
phase_difference = _instance.compute
pdoa = _instance.compute

__all__ = ["PhaseDifferenceMetric", "phase_difference", "pdoa"]


def main() -> None:
    print("=" * 60)
    print("  PhaseDifferenceMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = PhaseDifferenceMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
