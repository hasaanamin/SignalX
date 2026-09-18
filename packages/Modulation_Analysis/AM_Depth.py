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
from ._results import AMDepthResult
from ._validated_helpers import (
    _analyze_envelope,
    _estimate_am_depth,
)


@ModulationMetricFactory.register(["am_depth", "modulation_depth", "am_modulation_depth", "am_depth_estimator"])
class AMDepth(BaseModulationMetric):
    """
    AM Depth Module.
    Independently estimates amplitude modulation depth (m = (A_max - A_min) / (A_max + A_min))
    directly from raw complex IQ.
    Supports 'robust' (quantile-based) and 'exact' estimators.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        estimator: str = "robust",
        threshold_db: float = -60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.estimator = str(estimator)
        self.threshold_db = float(threshold_db)

    @property
    def name(self) -> str:
        return "am_depth"

    def compute(
        self,
        iq_data: Any,
        sample_rate: float | None = None,
        **kwargs: Any,
    ) -> AMDepthResult:
        """Executes AM Depth estimation directly on raw complex IQ."""
        fs = float(sample_rate) if sample_rate is not None else self.sample_rate

        env_info = _analyze_envelope(
            iq_data,
            sample_rate=fs,
            ref_method="robust",
            threshold_db=self.threshold_db,
        )

        depth_info = _estimate_am_depth(
            env_info["envelope"],
            estimator=self.estimator,
            valid_mask=env_info["valid_mask"],
        )

        return AMDepthResult(
            modulation_depth=depth_info["modulation_depth"],
            modulation_depth_percent=depth_info["modulation_depth_percent"],
            envelope_max=depth_info["envelope_max"],
            envelope_min=depth_info["envelope_min"],
            estimator=depth_info["estimator"],
            overmodulated=depth_info["overmodulated"],
            valid=depth_info["valid"],
            confidence=depth_info["confidence"],
            reason=depth_info["reason"],
        )


def compute_am_depth(
    iq_data: Any,
    sample_rate: float = 1.0,
    **kwargs: Any,
) -> AMDepthResult:
    """Functional convenience wrapper for AMDepth."""
    return AMDepth(sample_rate=sample_rate, **kwargs).compute(iq_data)


def main() -> None:
    print("=" * 60)
    print("  AMDepth: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = AMDepth()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
