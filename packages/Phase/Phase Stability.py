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


@PhaseMetricFactory.register(["phase_stability", "phase_noise_stability", "tdev"])
class PhaseStabilityMetric(BasePhaseMetric):
    """Computes Phase Stability matching MATLAB moving-average reference."""

    def __init__(self, **kwargs) -> None:
        """Constructor."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor."""
        super().__del__()

    @property
    def name(self) -> str:
        return "phase_stability"

    def compute(
        self,
        iq_data,
        *,
        sample_rate: float | None = None,
        deg: bool = False,
        method: str = "std",
        **kwargs,
    ) -> float | dict:
        # Step 1: Format IQ array and sample rate
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n < 2:
            return 0.0

        # Step 2: Unwrapped phase and time grid
        phi = np.unwrap(np.angle(z))
        t = np.arange(n) / fs

        # Step 3: Fit linear trend and compute detrended phase deviation
        slope, intercept = np.polyfit(t, phi, 1)
        trend = slope * t + intercept
        phase_deviation = phi - trend

        # Step 4: Window calculation (W = round(0.05*N) via MATLAB rounding, min 11, max 201, clamp to N, make odd)
        W = int(np.floor(0.05 * n + 0.5))
        W = max(11, min(201, W))
        W = min(W, n)
        if W % 2 == 0 and W > 3:
            W = W - 1
        if W > n:
            W = max(1, W - 2 if (W - 2) > 0 else 1)

        # Step 5: MATLAB movmean with Endpoints 'shrink'
        k = W // 2
        cumsum = np.concatenate(([0.0], np.cumsum(phase_deviation)))
        slow_phase = np.empty(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - k)
            end = min(n, i + k + 1)
            slow_phase[i] = (cumsum[end] - cumsum[start]) / (end - start)

        # Step 6: Stability = standard deviation of the slow component
        slow_std = float(np.std(slow_phase))

        if deg:
            slow_std = float(np.degrees(slow_std))

        m = str(method).lower()
        if m in ("std", "default", "phase_std"):
            return slow_std
        elif m in ("all", "dict", "full"):
            return {
                "phase_stability_std": slow_std,
                "phase_std": slow_std,
                "window_size_W": W,
                "unit": "degrees" if deg else "radians",
            }
        else:
            return slow_std


_instance = PhaseStabilityMetric()
phase_stability = _instance.compute

__all__ = ["PhaseStabilityMetric", "phase_stability"]


def main() -> None:
    print("=" * 60)
    print("  PhaseStabilityMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = PhaseStabilityMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
