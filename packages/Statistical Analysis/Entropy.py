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

import math
import numpy as np
from .base import BaseStatisticalAnalyzer, StatisticalAnalysisFactory
from ._common import _as_real_array, _as_array


@StatisticalAnalysisFactory.register(["entropy", "shannon_entropy"])
class EntropyAnalyzer(BaseStatisticalAnalyzer):
    """
    Calculates Shannon Entropy:
      H = - sum(p_i * log_base(p_i))
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes EntropyAnalyzer instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up EntropyAnalyzer resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "entropy"

    def compute(self, data, *, bins: int = 64, base: float = 2.0, mode: str = "auto") -> float:
        x = _as_real_array(data, mode=mode)
        if x.size == 0:
            return 0.0

        counts, _ = np.histogram(x, bins=bins)
        probs = counts / np.sum(counts)
        probs = probs[probs > 0]

        log_func = math.log2 if base == 2.0 else (math.log if base == math.e else lambda p: math.log(p, base))
        ent = -sum(p * log_func(p) for p in probs)
        return float(ent)


@StatisticalAnalysisFactory.register(["spectral_entropy", "spectral_flatness"])
class SpectralEntropyAnalyzer(BaseStatisticalAnalyzer):
    """
    Calculates Spectral Entropy / Spectral Flatness from power spectral density.
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes SpectralEntropyAnalyzer instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up SpectralEntropyAnalyzer resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "spectral_entropy"

    def compute(self, data, *, nperseg: int = 256, base: float = 2.0) -> float:
        x = _as_array(data)
        n = x.size
        if n == 0:
            return 0.0

        fft_len = min(n, nperseg)
        fft_vals = np.fft.fft(x[:fft_len])
        psd = np.abs(fft_vals) ** 2
        psd_sum = np.sum(psd)

        if psd_sum == 0:
            return 0.0

        p_norm = psd / psd_sum
        p_norm = p_norm[p_norm > 0]

        log_func = np.log2 if base == 2.0 else (np.log if base == math.e else lambda p: np.log(p) / np.log(base))
        h = -np.sum(p_norm * log_func(p_norm))
        max_h = log_func(len(p_norm)) if len(p_norm) > 1 else 1.0

        return float(h / max_h) if max_h > 0 else 0.0


_ent_inst = EntropyAnalyzer()
_spec_inst = SpectralEntropyAnalyzer()

shannon_entropy = _ent_inst.compute
entropy = _ent_inst.compute
spectral_entropy = _spec_inst.compute

__all__ = ["EntropyAnalyzer", "SpectralEntropyAnalyzer", "shannon_entropy", "spectral_entropy", "entropy"]


def main() -> None:
    print("=" * 60)
    print("  EntropyAnalyzer: Demonstration")
    print("=" * 60)
    data = np.random.randn(1024)
    analyzer = EntropyAnalyzer()
    result = analyzer.compute(data)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
