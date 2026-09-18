from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Type


class BaseCyclostationaryMetric(ABC):
    """
    Abstract Base Class for all Cyclostationary Analysis Metrics and Algorithms.
    Defines the standard enterprise interface for cyclic frequency, cyclic autocorrelation,
    spectral correlation, spectral correlation function (SCF / FAM), cyclic peaks, and cyclic coherence.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Constructor: Initializes metric configuration, parameters, and lifecycle state."""
        self._config: dict[str, Any] = dict(kwargs)
        self._is_active: bool = True

    def __del__(self) -> None:
        """Destructor: Performs cleanup and releases resources when instance is destroyed."""
        self._is_active = False
        if hasattr(self, "_config") and isinstance(self._config, dict):
            self._config.clear()

    @property
    def is_active(self) -> bool:
        """Indicates if the metric instance is active and ready for computation."""
        return getattr(self, "_is_active", True)

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name for this cyclostationary metric."""
        pass

    @property
    def description(self) -> str:
        """Human-readable description of what this metric measures."""
        return self.__doc__ or ""

    @abstractmethod
    def compute(self, data: Any, **kwargs: Any) -> Any:
        """
        Executes the cyclostationary measurement or transform on input data.

        Parameters:
            data: Complex IQ array, canonical dictionary, tuple of (I, Q), or precomputed SCF matrix.
            **kwargs: Additional parameters (e.g. sample_rate, nfft, alpha, max_lag, window).

        Returns:
            Computed numerical scalar, 1D spectral profile, 2D bifrequency matrix, or diagnostic dictionary.
        """
        pass

    def __call__(self, data: Any, **kwargs: Any) -> Any:
        """Allows metric instances to be called directly as functions: metric(iq_data)."""
        return self.compute(data, **kwargs)

    def reset(self) -> None:
        """Resets any internal accumulator or streaming state."""
        pass


class CyclostationaryMetricFactory:
    """
    Enterprise Factory Pattern for Cyclostationary Analysis Metrics.
    Maintains a decoupled registry of metric classes and handles lifecycle creation.
    """

    _registry: dict[str, Type[BaseCyclostationaryMetric]] = {}

    @classmethod
    def register(cls, aliases: str | list[str]) -> Callable[[Type[BaseCyclostationaryMetric]], Type[BaseCyclostationaryMetric]]:
        """Decorator to register a concrete BaseCyclostationaryMetric subclass under one or more alias names."""
        if isinstance(aliases, str):
            aliases = [aliases]

        def decorator(subclass: Type[BaseCyclostationaryMetric]) -> Type[BaseCyclostationaryMetric]:
            for alias in aliases:
                normalized = alias.strip().lower().replace(" ", "_").replace("-", "_")
                cls._registry[normalized] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, metric_name: str, **kwargs: Any) -> BaseCyclostationaryMetric:
        """Factory creation method: Instantiates and returns the requested metric implementation."""
        normalized = metric_name.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized not in cls._registry:
            valid_keys = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown cyclostationary metric '{metric_name}'. Registered metrics: [{valid_keys}]")
        return cls._registry[normalized](**kwargs)

    @classmethod
    def list_metrics(cls) -> list[str]:
        """Returns a sorted list of all unique registered cyclostationary metric names."""
        return sorted(list(set(cls._registry.keys())))
