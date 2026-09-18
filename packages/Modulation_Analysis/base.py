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

from abc import ABC, abstractmethod
from typing import Any, Callable, Type
import numpy as np

from ._results import (
    AMAnalysisResult,
    FMAnalysisResult,
    PMAnalysisResult,
    AMDepthResult,
    FMDeviationResult,
    FMIndexResult,
    PMDeviationResult,
    SymbolRateCandidate,
    SymbolRateResult,
    ConstellationResult,
    PhaseStatesResult,
    FrequencyStatesResult,
    ConstellationFeaturesResult,
    ModulationClassificationResult,
    ModulationIndexResult,
)


class BaseModulationMetric(ABC):
    """
    Abstract base class for all modulation analysis modules.
    Ensures consistent constructor and raw IQ compute() signature.
    """

    def __init__(self, sample_rate: float = 1.0, **kwargs: Any) -> None:
        self.sample_rate = float(sample_rate)
        if self.sample_rate <= 0 or not np.isfinite(self.sample_rate):
            raise ValueError(f"sample_rate must be a positive finite float, got {sample_rate}")
        self.config = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the modulation analysis stage."""
        pass

    @abstractmethod
    def compute(self, data: Any, **kwargs: Any) -> Any:
        """Executes modulation analysis directly on raw complex IQ."""
        pass

    def process(self, data: Any, **kwargs: Any) -> Any:
        """Pipeline compatibility alias for compute."""
        return self.compute(data, **kwargs)

    def __call__(self, data: Any, **kwargs: Any) -> Any:
        """Callable processor interface."""
        return self.compute(data, **kwargs)


class ModulationMetricFactory:
    """
    Factory for registering and instantiating modulation analysis metrics.
    """
    _registry: dict[str, Type[BaseModulationMetric]] = {}

    @classmethod
    def register(cls, names: str | list[str]) -> Callable[[Type[BaseModulationMetric]], Type[BaseModulationMetric]]:
        """Decorator to register a modulation metric under one or more alias names."""
        if isinstance(names, str):
            names = [names]

        def decorator(subclass: Type[BaseModulationMetric]) -> Type[BaseModulationMetric]:
            for name in names:
                normalized_name = name.lower().strip().replace(" ", "_").replace("-", "_")
                cls._registry[normalized_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, name: str, sample_rate: float = 1.0, **kwargs: Any) -> BaseModulationMetric:
        """Creates an instance of the requested modulation metric."""
        normalized_name = name.lower().strip().replace(" ", "_").replace("-", "_")
        if normalized_name not in cls._registry:
            valid = sorted(cls._registry.keys())
            raise ValueError(f"Unknown modulation metric '{name}'. Available: {valid}")
        metric_class = cls._registry[normalized_name]
        return metric_class(sample_rate=sample_rate, **kwargs)

    @classmethod
    def list_metrics(cls) -> list[str]:
        """Returns sorted list of registered metric names."""
        return sorted(cls._registry.keys())


def main() -> None:
    print("=" * 60)
    print("  ModulationMetricFactory: Demonstration")
    print("=" * 60)
    metrics = ModulationMetricFactory.list_metrics()
    print(f"Registered Modulation Metrics: {metrics}")
    print("=" * 60)


if __name__ == "__main__":
    main()
