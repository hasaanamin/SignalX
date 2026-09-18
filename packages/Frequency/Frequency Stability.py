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
from .base import BaseFrequencyMetric, FrequencyMetricFactory
from ._common import _as_iq_array, _extract_sample_rate


@FrequencyMetricFactory.register(["frequency_stability", "stability", "freq_stability"])
class FrequencyStabilityMetric(BaseFrequencyMetric):
    """Computes Frequency Stability matching MATLAB moving-average reference."""

    def __init__(self, **kwargs) -> None:
        """Constructor."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor."""
        super().__del__()

    @property
    def name(self) -> str:
        return "frequency_stability"

    def compute(
        self,
        iq_data,
        *,
        sample_rate: float | None = None,
        method: str = "std",
        mode: str = "stability",
        **kwargs,
    ) -> float | dict:
        # Step 1: Format IQ array and extract required sample rate
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=None)
        if fs is None or not np.isfinite(fs) or fs <= 0:
            raise ValueError("A valid positive sample_rate (Fs) is required for frequency stability calculation.")

        n = z.size
        if n < 2:
            return 0.0

        # Step 2: Full N-sample instantaneous frequency array
        phase_step = np.angle(z[1:] * np.conj(z[:-1]))
        f_inst = np.empty(n, dtype=np.float64)
        f_inst[0] = np.nan
        f_inst[1:] = (fs / (2.0 * np.pi)) * phase_step

        # Step 3: Identify valid finite samples and compute median center
        valid = np.isfinite(f_inst)
        if not np.any(valid):
            return 0.0

        f_center = float(np.median(f_inst[valid]))

        # Step 4: Impute non-finite entries with f_center before moving average
        temp_f = f_inst.copy()
        temp_f[~valid] = f_center

        # Step 5: Moving-average window W based on full record length N (MATLAB rounding)
        w = int(np.floor(0.05 * n + 0.5))
        w = max(11, min(201, w))
        w = min(w, n)
        if w % 2 == 0 and w > 3:
            w -= 1
        if w > n:
            w = max(1, w - 2 if (w - 2) > 0 else 1)

        # Step 6: Full N-sample MATLAB movmean with Endpoints 'shrink'
        k = w // 2
        cumsum = np.concatenate(([0.0], np.cumsum(temp_f)))
        slow_frequency = np.empty(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - k)
            end = min(n, i + k + 1)
            slow_frequency[i] = (cumsum[end] - cumsum[start]) / (end - start)

        # Step 7: Restore invalid positions to NaN
        slow_frequency[~valid] = np.nan

        # Step 8: Frequency stability = population standard deviation of slow frequency
        frequency_stability_hz = float(np.std(slow_frequency[valid], ddof=0))

        # Step 9: Separate Allan Deviation metric on valid samples
        f_valid = f_inst[valid]
        if f_valid.size > 1:
            dy = np.diff(f_valid)
            allan_deviation_hz = float(np.sqrt(0.5 * np.mean(dy ** 2)))
        else:
            allan_deviation_hz = 0.0

        meth = str(method).lower()
        if meth in ("allan", "allan_deviation", "adev"):
            return allan_deviation_hz

        out_mode = str(mode).lower()
        if out_mode in ("all", "dict", "full"):
            return {
                "frequency_stability_hz": frequency_stability_hz,
                "allan_deviation_hz": allan_deviation_hz,
                "window_size_W": w,
                "f_center_hz": f_center,
                "unit": "Hz",
            }

        return frequency_stability_hz


_instance = FrequencyStabilityMetric()
frequency_stability = _instance.compute
allan_deviation = lambda iq_data, **kw: _instance.compute(iq_data, method="allan", **kw)

__all__ = ["FrequencyStabilityMetric", "frequency_stability", "allan_deviation"]


def main() -> None:
    print("=" * 60)
    print("  FrequencyStabilityMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = FrequencyStabilityMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
