from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Type


class BaseFrequencyMetric(ABC):
    """
    Abstract Base Class for all Frequency Metrics.
    Defines the standard interface for frequency estimation, stability, offset, and chirp parameters.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Constructor: Initializes metric configuration, parameters, and lifecycle state.
        """
        self._config: dict[str, Any] = dict(kwargs)
        self._is_active: bool = True

    def __del__(self) -> None:
        """
        Destructor: Performs cleanup and releases resources when instance is destroyed.
        """
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
        """Unique identifier name for this metric (e.g. 'instantaneous_frequency', 'chirp_rate')."""
        pass

    @property
    def description(self) -> str:
        """Human-readable description of what this frequency metric measures."""
        return self.__doc__ or ""

    @abstractmethod
    def compute(self, iq_data: Any, **kwargs: Any) -> Any:
        """
        Executes the frequency measurement on the input IQ data.
        
        Parameters:
            iq_data: Complex IQ array, canonical dictionary, tuple of (I, Q), or list of pairs.
            **kwargs: Additional parameters (e.g. sample_rate, rf_center_freq, nominal_freq).
            
        Returns:
            Computed numerical scalar (Hz), frequency time-series array, or parameter dictionary.
        """
        pass

    def __call__(self, iq_data: Any, **kwargs: Any) -> Any:
        """Allows metric instances to be called directly as functions: metric(iq_data)."""
        return self.compute(iq_data, **kwargs)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', active={self.is_active})>"


class FrequencyMetricFactory:
    """
    Central Factory Registry for Frequency Analysis Metrics.
    Manages discovery, instantiation, and execution of all registered frequency metrics.
    """

    _registry: dict[str, Type[BaseFrequencyMetric]] = {}

    def __init__(self) -> None:
        """Constructor: Initializes a local factory instance."""
        self._instance_created = True

    def __del__(self) -> None:
        """Destructor: Cleans up factory instance state."""
        self._instance_created = False

    @classmethod
    def register(cls, alias: str | list[str]) -> Callable[[Type[BaseFrequencyMetric]], Type[BaseFrequencyMetric]]:
        """
        Decorator to register a frequency metric class into the factory.
        
        Example:
            @FrequencyMetricFactory.register(["chirp_rate", "lfm_slope"])
            class ChirpRateMetric(BaseFrequencyMetric):
                ...
        """
        def decorator(metric_cls: Type[BaseFrequencyMetric]) -> Type[BaseFrequencyMetric]:
            aliases = [alias] if isinstance(alias, str) else alias
            for a in aliases:
                normalized = a.strip().lower().replace(" ", "_").replace("-", "_")
                cls._registry[normalized] = metric_cls
            return metric_cls
        return decorator

    @classmethod
    def create(cls, metric_name: str, **kwargs: Any) -> BaseFrequencyMetric:
        """
        Instantiates and returns the requested frequency metric analyzer.
        
        Example:
            metric = FrequencyMetricFactory.create("instantaneous_frequency")
            f_inst = metric.compute(iq_data, sample_rate=1e6)
        """
        key = metric_name.strip().lower().replace(" ", "_").replace("-", "_")
        if key not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown frequency metric '{metric_name}'. "
                f"Registered metrics: {available}"
            )
        return cls._registry[key](**kwargs)

    @classmethod
    def compute(cls, metric_name: str, iq_data: Any, **kwargs: Any) -> Any:
        """
        Convenience method: creates and executes the metric in one call.
        
        Example:
            cfo = FrequencyMetricFactory.compute("frequency_offset", iq_data, sample_rate=1e6)
        """
        metric = cls.create(metric_name)
        return metric.compute(iq_data, **kwargs)

    @classmethod
    def compute_all(cls, iq_data: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Runs all registered frequency metrics across the input IQ data
        and returns a consolidated summary dictionary.
        """
        results: dict[str, Any] = {}
        executed_classes: set[Type[BaseFrequencyMetric]] = set()

        for key, metric_cls in cls._registry.items():
            if metric_cls in executed_classes:
                continue
            executed_classes.add(metric_cls)

            instance = metric_cls()
            try:
                results[instance.name] = instance.compute(iq_data, **kwargs)
            except Exception as exc:
                results[instance.name] = f"Error: {exc}"

        return results

    @classmethod
    def list_metrics(cls) -> list[str]:
        """Returns a sorted list of all registered metric names and aliases."""
        return sorted(cls._registry.keys())

    @classmethod
    def get_catalog(cls) -> list[dict[str, str]]:
        """Returns catalog metadata for all registered metrics."""
        seen: set[Type[BaseFrequencyMetric]] = set()
        catalog = []
        for metric_cls in cls._registry.values():
            if metric_cls in seen:
                continue
            seen.add(metric_cls)
            instance = metric_cls()
            catalog.append({
                "name": instance.name,
                "class": metric_cls.__name__,
                "description": instance.description.strip(),
            })
        return catalog


__all__ = ["BaseFrequencyMetric", "FrequencyMetricFactory"]
