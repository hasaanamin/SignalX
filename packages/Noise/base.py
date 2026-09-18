from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Type


class BaseNoiseMetric(ABC):
    """
    Abstract Base Class for all Noise and Signal Fidelity Metrics.
    Defines the standard enterprise interface for noise power, SNR, SINR, phase noise,
    interference, clipping, saturation, and signal quality measurements.
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
        """Unique identifier name for this metric (e.g. 'snr', 'phase_noise', 'clipping_assessment')."""
        pass

    @property
    def description(self) -> str:
        """Human-readable description of what this metric measures."""
        return self.__doc__ or ""

    @abstractmethod
    def compute(self, iq_data: Any, **kwargs: Any) -> Any:
        """
        Executes the noise / signal-fidelity measurement on the input IQ data.

        Parameters:
            iq_data: Complex IQ array, canonical dictionary, tuple of (I, Q), or list of pairs.
            **kwargs: Additional parameters (e.g. sample_rate, threshold, offsets_hz, nfft).

        Returns:
            Computed numerical scalar (linear/dB), time-series array, or diagnostic dictionary.
        """
        pass

    def __call__(self, iq_data: Any, **kwargs: Any) -> Any:
        """Allows metric instances to be called directly as functions: metric(iq_data)."""
        return self.compute(iq_data, **kwargs)

    def reset(self) -> None:
        """Resets any internal accumulator or streaming state."""
        pass


class NoiseMetricFactory:
    """
    Enterprise Factory Pattern for Noise and Signal-Fidelity Metrics.
    Maintains a decoupled registry of metric classes and handles lifecycle creation.
    """

    _registry: dict[str, Type[BaseNoiseMetric]] = {}

    @classmethod
    def register(cls, aliases: str | list[str]) -> Callable[[Type[BaseNoiseMetric]], Type[BaseNoiseMetric]]:
        """
        Decorator to register a concrete BaseNoiseMetric subclass under one or more alias names.
        """
        if isinstance(aliases, str):
            aliases = [aliases]

        def decorator(subclass: Type[BaseNoiseMetric]) -> Type[BaseNoiseMetric]:
            for alias in aliases:
                normalized = alias.strip().lower().replace(" ", "_").replace("-", "_")
                cls._registry[normalized] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, metric_name: str, **kwargs: Any) -> BaseNoiseMetric:
        """
        Factory creation method: Instantiates and returns the requested metric implementation.
        """
        normalized = metric_name.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized not in cls._registry:
            valid_keys = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown noise metric '{metric_name}'. Registered metrics: [{valid_keys}]")
        return cls._registry[normalized](**kwargs)

    @classmethod
    def list_metrics(cls) -> list[str]:
        """Returns sorted list of all registered metric aliases."""
        return sorted(list(cls._registry.keys()))

    @classmethod
    def get_registered_classes(cls) -> dict[str, Type[BaseNoiseMetric]]:
        """Returns shallow copy of registry mapping aliases to classes."""
        return dict(cls._registry)
