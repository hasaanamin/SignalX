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


@FrequencyMetricFactory.register(["frequency_deviation", "peak_deviation", "freq_deviation", "fm_deviation"])
class FrequencyDeviationMetric(BaseFrequencyMetric):
    """Computes Frequency Deviation relative to median instantaneous frequency."""

    def __init__(self, **kwargs) -> None:
        """Constructor."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor."""
        super().__del__()

    @property
    def name(self) -> str:
        return "frequency_deviation"

    def compute(
        self,
        iq_data,
        *,
        sample_rate: float | None = None,
        mode: str = "all",
        center_freq_hz: float | None = None,
        **kwargs,
    ) -> float | dict:
        # Step 1: Format IQ array and extract required sample rate
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=None)
        if fs is None or not np.isfinite(fs) or fs <= 0:
            raise ValueError("A valid positive sample_rate (Fs) is required for frequency deviation calculation.")

        n = z.size
        if n < 2:
            return 0.0 if mode != "all" else {
                "peak_deviation_hz": 0.0,
                "rms_deviation_hz": 0.0,
                "std_deviation_hz": 0.0,
                "peak_to_peak_deviation_hz": 0.0,
                "center_freq_hz": 0.0,
                "unit": "Hz",
            }

        # Step 2: Instantaneous frequency with NaN first sample
        phase_step = np.angle(z[1:] * np.conj(z[:-1]))
        f_inst = np.empty(n, dtype=np.float64)
        f_inst[0] = np.nan
        f_inst[1:] = (fs / (2.0 * np.pi)) * phase_step

        # Step 3: Extract finite samples
        valid_mask = np.isfinite(f_inst)
        f_valid = f_inst[valid_mask]
        if f_valid.size == 0:
            return 0.0

        # Step 4: Reference center frequency (MATLAB median reference)
        f_center = float(center_freq_hz) if center_freq_hz is not None else float(np.median(f_valid))

        # Step 5: Frequency deviation
        freq_dev = f_valid - f_center

        peak_dev = float(np.max(np.abs(freq_dev)))
        rms_dev = float(np.sqrt(np.mean(freq_dev ** 2)))
        std_dev = float(np.std(freq_dev, ddof=0))
        pp_dev = float(np.max(f_valid) - np.min(f_valid))

        m = str(mode).lower()
        if m in ("peak", "max"):
            return peak_dev
        elif m in ("rms", "root_mean_square"):
            return rms_dev
        elif m in ("std", "standard_deviation"):
            return std_dev
        elif m in ("peak_to_peak", "pp", "p-p", "ptp"):
            return pp_dev
        elif m in ("all", "dict", "full"):
            return {
                "peak_deviation_hz": peak_dev,
                "rms_deviation_hz": rms_dev,
                "std_deviation_hz": std_dev,
                "peak_to_peak_deviation_hz": pp_dev,
                "center_freq_hz": f_center,
                "unit": "Hz",
            }
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'peak', 'rms', 'std', 'peak_to_peak', or 'all'.")


_instance = FrequencyDeviationMetric()
frequency_deviation = _instance.compute
peak_deviation = _instance.compute

__all__ = ["FrequencyDeviationMetric", "frequency_deviation", "peak_deviation"]


def main() -> None:
    print("=" * 60)
    print("  FrequencyDeviationMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = FrequencyDeviationMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
