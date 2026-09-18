from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Type
import numpy as np


class BaseStatisticalAnalyzer(ABC):
    """
    Abstract Base Class for all Statistical Analyzers.
    Defines the contract that every statistical metric algorithm must implement.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Constructor: Initializes analyzer configuration, options, and lifecycle state.
        """
        self._config: dict[str, Any] = dict(kwargs)
        self._is_active: bool = True

    def __del__(self) -> None:
        """
        Destructor: Performs cleanup and releases resources when instance is garbage collected.
        """
        self._is_active = False
        if hasattr(self, "_config") and isinstance(self._config, dict):
            self._config.clear()

    @property
    def is_active(self) -> bool:
        """Indicates if the analyzer instance is active and ready for computation."""
        return getattr(self, "_is_active", True)

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name for this metric (e.g. 'mean', 'kurtosis')."""
        pass

    @property
    def description(self) -> str:
        """Human-readable description of what this statistical metric measures."""
        return self.__doc__ or ""

    @abstractmethod
    def compute(self, data: Any, **kwargs: Any) -> Any:
        """
        Executes the statistical calculation on the input signal/data.
        
        Parameters:
            data: Complex IQ array, canonical dictionary, tuple of arrays, or 1D real numbers.
            **kwargs: Additional metric-specific options (e.g. mode, bias, sample_variance, bins).
        
        Returns:
            Computed numerical scalar, array, or dictionary of metrics.
        """
        pass

    def __call__(self, data: Any, **kwargs: Any) -> Any:
        """Allows analyzer instances to be called directly as functions: analyzer(data)."""
        return self.compute(data, **kwargs)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', active={self.is_active})>"


class StatisticalAnalysisFactory:
    """
    Central Factory Registry for Statistical Analysis.
    Manages discovery, instantiation, and execution of all registered statistical analyzers.
    """

    _registry: dict[str, Type[BaseStatisticalAnalyzer]] = {}

    def __init__(self) -> None:
        """Constructor: Initializes a local factory instance."""
        self._instance_created = True

    def __del__(self) -> None:
        """Destructor: Cleans up factory instance state."""
        self._instance_created = False

    @classmethod
    def register(cls, alias: str | list[str]) -> Callable[[Type[BaseStatisticalAnalyzer]], Type[BaseStatisticalAnalyzer]]:
        """
        Decorator to register an analyzer class with one or multiple aliases.
        
        Example:
            @StatisticalAnalysisFactory.register(["kurtosis", "excess_kurtosis"])
            class KurtosisAnalyzer(BaseStatisticalAnalyzer):
                ...
        """
        def decorator(analyzer_cls: Type[BaseStatisticalAnalyzer]) -> Type[BaseStatisticalAnalyzer]:
            aliases = [alias] if isinstance(alias, str) else alias
            for a in aliases:
                normalized = a.strip().lower().replace(" ", "_").replace("-", "_")
                cls._registry[normalized] = analyzer_cls
            return analyzer_cls
        return decorator

    @classmethod
    def create(cls, metric_name: str, **kwargs: Any) -> BaseStatisticalAnalyzer:
        """
        Instantiates and returns an analyzer instance for the requested metric.
        
        Example:
            analyzer = StatisticalAnalysisFactory.create("kurtosis")
            result = analyzer.compute(iq_data)
        """
        key = metric_name.strip().lower().replace(" ", "_").replace("-", "_")
        if key not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown statistical metric '{metric_name}'. "
                f"Registered metrics: {available}"
            )
        return cls._registry[key](**kwargs)

    @classmethod
    def compute(cls, metric_name: str, data: Any, **kwargs: Any) -> Any:
        """
        Convenience one-liner: creates the analyzer and immediately computes the metric.
        
        Example:
            kurt = StatisticalAnalysisFactory.compute("kurtosis", iq_data, excess=True)
        """
        analyzer = cls.create(metric_name)
        return analyzer.compute(data, **kwargs)

    @classmethod
    def compute_all(cls, data: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Runs all unique registered analyzers across the input signal and returns
        a consolidated statistical dictionary.
        """
        results: dict[str, Any] = {}
        executed_classes: set[Type[BaseStatisticalAnalyzer]] = set()

        for key, analyzer_cls in cls._registry.items():
            if analyzer_cls in executed_classes:
                continue
            executed_classes.add(analyzer_cls)

            instance = analyzer_cls()
            try:
                results[instance.name] = instance.compute(data, **kwargs)
            except Exception as exc:
                results[instance.name] = f"Error: {exc}"

        return results

    @classmethod
    def list_metrics(cls) -> list[str]:
        """Returns a sorted list of all registered metric names and aliases."""
        return sorted(cls._registry.keys())

    @classmethod
    def get_catalog(cls) -> list[dict[str, str]]:
        """Returns catalog metadata for all registered analyzers."""
        seen: set[Type[BaseStatisticalAnalyzer]] = set()
        catalog = []
        for analyzer_cls in cls._registry.values():
            if analyzer_cls in seen:
                continue
            seen.add(analyzer_cls)
            instance = analyzer_cls()
            catalog.append({
                "name": instance.name,
                "class": analyzer_cls.__name__,
                "description": instance.description.strip(),
            })
        return catalog


__all__ = ["BaseStatisticalAnalyzer", "StatisticalAnalysisFactory"]
