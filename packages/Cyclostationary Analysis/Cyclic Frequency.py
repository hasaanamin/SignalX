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
import numpy as np
from .base import BaseCyclostationaryMetric, CyclostationaryMetricFactory
from ._common import _extract_sample_rate


@CyclostationaryMetricFactory.register([
    "cyclic_frequency",
    "alpha_grid",
    "cyclic_frequency_grid",
    "cyclic_bins",
    "alpha_frequencies"
])
class CyclicFrequencyMetric(BaseCyclostationaryMetric):
    """
    Cyclic Frequency Grid & Parameter Resolution.
    Evaluates cyclic frequency domain coordinates:
      alpha_r = r / T_0 = r * R_cyclic  (Harmonic cycle frequencies)
      alpha_p = p * (F_s / NFFT_alpha)  (Discrete FFT-bin cycle frequencies)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "cyclic_frequency"

    def compute(self, data: Any = None, **kwargs: Any) -> np.ndarray | dict[str, Any]:
        """
        Generates cyclic frequency axis and resolves harmonic cyclic frequencies.

        Parameters:
            data: Optional IQ array or None.
            sample_rate (Fs): Sampling rate in Hz (default: 1.0).
            nfft / nfft_alpha: Cyclic frequency FFT length / bin count (default: 256).
            cyclic_rate / symbol_rate (Rs): Optional fundamental cyclic rate in Hz (or 1/T_0).
            num_harmonics (R): Number of cyclic harmonics to compute (default: 5).
            as_dict: If True, returns comprehensive dictionary.

        Returns:
            1D array of cyclic frequencies alpha_p (or dictionary if as_dict=True).
        """
        params = {**self._config, **kwargs}
        fs = _extract_sample_rate(params, default=1.0)
        nfft = int(params.get("nfft_alpha", params.get("nfft", params.get("N_FFT", 256))))
        if nfft < 4:
            nfft = 256

        # Discrete FFT-bin cyclic grid: alpha_p = p * (Fs / NFFT_alpha)
        p_min = -nfft // 2
        p_max = nfft // 2
        p_bins = np.arange(p_min, p_max + 1, dtype=np.int32)
        delta_alpha = fs / nfft
        alpha_grid = p_bins * delta_alpha

        # Theoretical harmonic frequencies: alpha_r = r / T_0 = r * R_cyclic
        rc = params.get("cyclic_rate", params.get("symbol_rate", params.get("baud_rate", params.get("Rs"))))
        t0 = params.get("T_0", params.get("period"))
        if t0 is not None and float(t0) > 0:
            rc = 1.0 / float(t0)

        num_harmonics = int(params.get("num_harmonics", 5))
        harmonic_alphas = []
        harmonic_bins = []
        if rc is not None and float(rc) > 0:
            rc_val = float(rc)
            for r in range(-num_harmonics, num_harmonics + 1):
                a_r = r * rc_val
                p_r = int(np.round(a_r / delta_alpha))
                harmonic_alphas.append(a_r)
                harmonic_bins.append(p_r)

        if params.get("as_dict", False):
            return {
                "alpha_grid_hz": alpha_grid,
                "bin_indices": p_bins,
                "delta_alpha_hz": delta_alpha,
                "alpha_bin_spacing_hz": delta_alpha,
                "sample_rate_hz": fs,
                "nfft_alpha": nfft,
                "nfft": nfft,
                "num_alpha_bins": len(alpha_grid),
                "cyclic_rate_hz": float(rc) if rc is not None else None,
                "symbol_rate_hz": float(rc) if rc is not None else None,  # deprecated alias
                "harmonic_alphas_hz": np.array(harmonic_alphas, dtype=np.float64),
                "harmonic_bins": np.array(harmonic_bins, dtype=np.int32),
            }

        return alpha_grid


def cyclic_frequency(
    data: Any = None,
    sample_rate: float = 1.0,
    nfft: int = 256,
    symbol_rate: float | None = None,
    cyclic_rate: float | None = None,
    as_dict: bool = False,
    **kwargs: Any
) -> np.ndarray | dict[str, Any]:
    """Functional wrapper for CyclicFrequencyMetric."""
    rate = cyclic_rate if cyclic_rate is not None else symbol_rate
    metric = CyclicFrequencyMetric(
        sample_rate=sample_rate, nfft=nfft, cyclic_rate=rate, as_dict=as_dict, **kwargs
    )
    return metric.compute(data)


def main() -> None:
    print("=" * 60)
    print("  CyclicFrequencyMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = CyclicFrequencyMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
