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


@FrequencyMetricFactory.register(["instantaneous_frequency", "inst_freq", "frequency_vs_time"])
class InstantaneousFrequencyMetric(BaseFrequencyMetric):
    """
    Computes Instantaneous Frequency vs Time matching MATLAB reference:
      1. phase_step = angle(z[1:] * conj(z[:-1]))
      2. f_inst[0] = NaN
      3. f_inst[1:] = (Fs / (2*pi)) * phase_step
      
    The first sample is NaN (no differential history) and is not duplicated.
    Subsequent calculations must ignore non-finite samples.
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes InstantaneousFrequencyMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up InstantaneousFrequencyMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "instantaneous_frequency"

    def compute(
        self,
        iq_data,
        *,
        sample_rate: float | None = None,
        rf_center_hz: float = 0.0,
        **kwargs,
    ) -> np.ndarray:
        # Step 1: Format IQ array and extract required sample rate
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=None)
        if fs is None or not np.isfinite(fs) or fs <= 0:
            raise ValueError("A valid positive sample_rate (Fs) is required for instantaneous frequency calculation.")

        n = z.size
        if n == 0:
            return np.array([], dtype=np.float64)
        if n == 1:
            return np.array([np.nan], dtype=np.float64)

        # Step 2: Compute differential phase step
        phase_step = np.angle(z[1:] * np.conj(z[:-1]))

        # Step 3: f_inst with first sample as NaN
        f_inst = np.empty(n, dtype=np.float64)
        f_inst[0] = np.nan
        f_inst[1:] = (fs / (2.0 * np.pi)) * phase_step

        # Step 4: Shift by RF center if specified
        if rf_center_hz != 0.0:
            f_inst[1:] += float(rf_center_hz)

        return f_inst


_instance = InstantaneousFrequencyMetric()
instantaneous_frequency = _instance.compute

__all__ = ["InstantaneousFrequencyMetric", "instantaneous_frequency"]


def main() -> None:
    print("=" * 60)
    print("  InstantaneousFrequencyMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = InstantaneousFrequencyMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
