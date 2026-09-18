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

from .base import BaseModulationMetric, ModulationMetricFactory
from ._results import SymbolRateResult
from ._constellation_helpers import _estimate_symbol_rate


@ModulationMetricFactory.register(["symbol_rate", "baud_rate", "symbol_rate_estimator", "baud"])
class SymbolRate(BaseModulationMetric):
    """
    Symbol Rate Module.
    Independently estimates digital symbol rate (Baud) directly from raw complex IQ
    using validated cyclostationary fine alpha lag-product processing.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        min_rate_baud: float | None = None,
        max_rate_baud: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.min_rate_baud = min_rate_baud
        self.max_rate_baud = max_rate_baud

    @property
    def name(self) -> str:
        return "symbol_rate"

    def compute(
        self,
        iq_data: Any,
        sample_rate: float | None = None,
        **kwargs: Any,
    ) -> SymbolRateResult:
        """Executes Symbol Rate estimation directly on raw complex IQ."""
        fs = float(sample_rate) if sample_rate is not None else self.sample_rate

        res = _estimate_symbol_rate(
            iq_data,
            sample_rate=fs,
            min_rate_baud=self.min_rate_baud,
            max_rate_baud=self.max_rate_baud,
        )

        return SymbolRateResult(
            symbol_rate_baud=res["symbol_rate_baud"],
            samples_per_symbol=res["samples_per_symbol"],
            candidates=res["candidates"],
            confidence=res["confidence"],
            valid=res["valid"],
            reason=res["reason"],
        )


def compute_symbol_rate(
    iq_data: Any,
    sample_rate: float = 1.0,
    **kwargs: Any,
) -> SymbolRateResult:
    """Functional convenience wrapper for SymbolRate."""
    return SymbolRate(sample_rate=sample_rate, **kwargs).compute(iq_data)


def main() -> None:
    print("=" * 60)
    print("  SymbolRate: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = SymbolRate()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
