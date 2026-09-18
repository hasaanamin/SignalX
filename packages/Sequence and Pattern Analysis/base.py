from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, Union
import numpy as np


class BaseSequencePatternMetric(ABC):
    """
    Abstract base class for all sequence and pattern analysis processors.
    Enforces unified interfaces, parameter passing, and validation contracts.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.params = kwargs

    @property
    def name(self) -> str:
        """Returns the canonical processor name."""
        return getattr(self, "metric_name", self.__class__.__name__.lower())

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the sequence or pattern analysis."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"


class SequencePatternMetricFactory:
    """
    Registry and factory for all Sequence and Pattern Analysis metrics and processors.
    """
    _registry: dict[str, type[BaseSequencePatternMetric]] = {}

    @classmethod
    def register(
        cls,
        names: list[str] | str,
        subclass: Optional[type[BaseSequencePatternMetric]] = None,
    ) -> Any:
        """
        Registers a processor class under one or multiple aliases.
        Supports both decorator `@SequencePatternMetricFactory.register(...)`
        and functional `SequencePatternMetricFactory.register("name", Class)`.
        """
        alias_list = [names] if isinstance(names, str) else list(names)

        def decorator(cls_to_reg: type[BaseSequencePatternMetric]) -> type[BaseSequencePatternMetric]:
            for name in alias_list:
                key = cls._normalize_key(name)
                cls._registry[key] = cls_to_reg
            return cls_to_reg

        if subclass is not None:
            return decorator(subclass)
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseSequencePatternMetric:
        """Instantiates a registered processor by name."""
        key = cls._normalize_key(name)
        if key not in cls._registry:
            available = sorted(list(cls._registry.keys()))
            raise KeyError(f"Unknown sequence pattern metric '{name}'. Available: {available}")
        return cls._registry[key](**kwargs)

    @classmethod
    def available_metrics(cls) -> list[str]:
        """Returns list of all registered metric keys."""
        return sorted(list(cls._registry.keys()))

    @classmethod
    def list_registered(cls) -> list[str]:
        """Alias for available_metrics."""
        return cls.available_metrics()

    @staticmethod
    def _normalize_key(key: str) -> str:
        return key.lower().replace(" ", "_").replace("-", "_").strip()
