from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Type


class BasePulseDetectionMetric(ABC):
    """
    Abstract Base Class for all Pulse Detection & Estimation metrics.
    Defines lifecycle and execution contract for pulse analyzers.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._config: dict[str, Any] = dict(kwargs)
        self._is_active: bool = True

    def __del__(self) -> None:
        self._is_active = False
        if hasattr(self, "_config") and isinstance(self._config, dict):
            self._config.clear()

    @property
    def is_active(self) -> bool:
        return getattr(self, "_is_active", True)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def description(self) -> str:
        return self.__doc__ or ""

    @abstractmethod
    def compute(self, signal: Any, **kwargs: Any) -> Any:
        pass

    def __call__(self, signal: Any, **kwargs: Any) -> Any:
        return self.compute(signal, **kwargs)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', active={self.is_active})>"


class PulseDetectionFactory:
    """
    Central Factory Registry for Pulse Detection & Estimation Metrics.
    Manages discovery, instantiation, and pipeline execution.
    """

    _registry: dict[str, Type[BasePulseDetectionMetric]] = {}

    def __init__(self) -> None:
        self._instance_created = True

    def __del__(self) -> None:
        self._instance_created = False

    @classmethod
    def register(cls, alias: str | list[str]) -> Callable[[Type[BasePulseDetectionMetric]], Type[BasePulseDetectionMetric]]:
        """Decorator to register a pulse detection metric class into the factory."""
        def decorator(metric_cls: Type[BasePulseDetectionMetric]) -> Type[BasePulseDetectionMetric]:
            aliases = [alias] if isinstance(alias, str) else alias
            for a in aliases:
                normalized = a.strip().lower().replace(" ", "_").replace("-", "_")
                cls._registry[normalized] = metric_cls
            return metric_cls
        return decorator

    @classmethod
    def create(cls, metric_name: str, **kwargs: Any) -> BasePulseDetectionMetric:
        """Instantiates and returns requested pulse detection analyzer."""
        key = metric_name.strip().lower().replace(" ", "_").replace("-", "_")
        if key not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown pulse detection metric '{metric_name}'. "
                f"Registered metrics: {available}"
            )
        return cls._registry[key](**kwargs)

    @classmethod
    def compute(cls, metric_name: str, signal: Any, **kwargs: Any) -> Any:
        """Executes the specified metric in one call."""
        metric = cls.create(metric_name)
        return metric.compute(signal, **kwargs)

    @classmethod
    def compute_all(cls, signal: Any, sample_rate: float = 1.0, **kwargs: Any) -> dict[str, Any]:
        """Runs all registered pulse metrics across the input signal."""
        results: dict[str, Any] = {}
        executed_classes: set[Type[BasePulseDetectionMetric]] = set()

        for key, metric_cls in cls._registry.items():
            if metric_cls in executed_classes:
                continue
            executed_classes.add(metric_cls)

            instance = metric_cls()
            try:
                results[instance.name] = instance.compute(signal, sample_rate=sample_rate, **kwargs)
            except Exception as exc:
                results[instance.name] = f"Error: {exc}"

        return results

    @classmethod
    def detect_pulses(cls, signal: Any, sample_rate: float = 1.0, **kwargs: Any) -> list[dict[str, Any]]:
        """Executes full end-to-end pulse detection & segmentation pipeline."""
        segmenter = cls.create("pulse_segmentation")
        return segmenter.compute(signal, sample_rate=sample_rate, **kwargs)

    @classmethod
    def list_metrics(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def get_catalog(cls) -> list[dict[str, str]]:
        seen: set[Type[BasePulseDetectionMetric]] = set()
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


__all__ = ["BasePulseDetectionMetric", "PulseDetectionFactory"]
