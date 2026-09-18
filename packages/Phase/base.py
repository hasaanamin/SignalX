from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Type


class BasePhaseMetric(ABC):
    """
    Abstract Base Class for all Phase Metrics.
    Defines the standard interface for phase estimation, offset, deviation, jitter, and continuity analysis.
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
        """Unique identifier name for this metric (e.g. 'instantaneous_phase', 'phase_jitter')."""
        pass

    @property
    def description(self) -> str:
        """Human-readable description of what this phase metric measures."""
        return self.__doc__ or ""

    @abstractmethod
    def compute(self, iq_data: Any, **kwargs: Any) -> Any:
        """
        Executes the phase measurement on the input IQ data.
        
        Parameters:
            iq_data: Complex IQ array, canonical dictionary, tuple of (I, Q), or list of pairs.
            **kwargs: Metric-specific parameters (e.g. unwrap, deg, sample_rate, threshold).
            
        Returns:
            Computed numerical scalar (rad/deg), phase time-series array, or parameter dictionary.
        """
        pass

    def __call__(self, iq_data: Any, **kwargs: Any) -> Any:
        """Allows metric instances to be called directly as functions: metric(iq_data)."""
        return self.compute(iq_data, **kwargs)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', active={self.is_active})>"


class PhaseMetricFactory:
    """
    Central Factory Registry for Phase Analysis Metrics.
    Manages discovery, instantiation, and execution of all registered phase metrics.
    """

    _registry: dict[str, Type[BasePhaseMetric]] = {}

    def __init__(self) -> None:
        """Constructor: Initializes a local factory instance."""
        self._instance_created = True

    def __del__(self) -> None:
        """Destructor: Cleans up factory instance state."""
        self._instance_created = False

    @classmethod
    def register(cls, alias: str | list[str]) -> Callable[[Type[BasePhaseMetric]], Type[BasePhaseMetric]]:
        """
        Decorator to register a phase metric class into the factory.
        
        Example:
            @PhaseMetricFactory.register(["phase_jitter", "rms_phase_jitter"])
            class PhaseJitterMetric(BasePhaseMetric):
                ...
        """
        def decorator(metric_cls: Type[BasePhaseMetric]) -> Type[BasePhaseMetric]:
            aliases = [alias] if isinstance(alias, str) else alias
            for a in aliases:
                normalized = a.strip().lower().replace(" ", "_").replace("-", "_")
                cls._registry[normalized] = metric_cls
            return metric_cls
        return decorator

    @classmethod
    def create(cls, metric_name: str, **kwargs: Any) -> BasePhaseMetric:
        """
        Instantiates and returns the requested phase metric analyzer.
        
        Example:
            metric = PhaseMetricFactory.create("instantaneous_phase")
            phase_rad = metric.compute(iq_data, unwrap=True)
        """
        key = metric_name.strip().lower().replace(" ", "_").replace("-", "_")
        if key not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown phase metric '{metric_name}'. "
                f"Registered metrics: {available}"
            )
        return cls._registry[key](**kwargs)

    @classmethod
    def compute(cls, metric_name: str, iq_data: Any, **kwargs: Any) -> Any:
        """
        Convenience method: creates and executes the metric in one call.
        
        Example:
            offset = PhaseMetricFactory.compute("phase_offset", iq_data, deg=True)
        """
        metric = cls.create(metric_name)
        return metric.compute(iq_data, **kwargs)

    @classmethod
    def compute_all(cls, iq_data: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Runs all registered phase metrics across the input IQ data
        and returns a consolidated summary dictionary.
        """
        results: dict[str, Any] = {}
        executed_classes: set[Type[BasePhaseMetric]] = set()

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
        seen: set[Type[BasePhaseMetric]] = set()
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


__all__ = ["BasePhaseMetric", "PhaseMetricFactory"]
