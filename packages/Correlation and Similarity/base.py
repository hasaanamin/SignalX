from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Type
import numpy as np


@dataclass
class CorrelationResult:
    """
    Standard shared container for Autocorrelation and Cross-Correlation results.
    Reused across Waveform Similarity, Lag Analysis, and Inter-Pulse Correlation
    to eliminate duplicate cross-correlation calculations.
    """
    correlation_complex: np.ndarray  # 1D array of complex cross-correlation R_xy[m]
    correlation_magnitude: np.ndarray  # 1D array |R_xy[m]|
    normalized_correlation: np.ndarray  # 1D array rho_xy[m] scaled in [-1, 1]
    lags_samples: np.ndarray  # 1D array of integer lags m in [-maxlag, maxlag]
    lags_sec: np.ndarray  # 1D array of lags in seconds (m / Fs)
    peak_index: int  # Index in array where |rho_xy| is maximized
    peak_value_complex: complex  # Complex value at peak lag rho_xy[m_p]
    peak_magnitude: float  # Maximum correlation magnitude |rho_xy[m_p]| in [0, 1]
    peak_lag_samples: int  # Lag m_p where correlation is maximized
    peak_lag_sec: float  # Lag in seconds m_p / Fs
    estimated_delay_samples: int  # MATLAB finddelay convention: D = -m_p
    estimated_delay_sec: float  # Delay in seconds D / Fs = -m_p / Fs
    phase_at_peak_rad: float  # Phase angle at peak lag: angle(rho_xy[m_p])
    sample_rate: float = 1.0  # Fs in Hz
    normalization: str = "coeff"  # 'coeff', 'none', 'biased', 'unbiased'
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Converts result container to a canonical dictionary representation."""
        return {
            "correlation_complex": self.correlation_complex,
            "correlation_magnitude": self.correlation_magnitude,
            "normalized_correlation": self.normalized_correlation,
            "lags_samples": self.lags_samples,
            "lags_sec": self.lags_sec,
            "peak_index": self.peak_index,
            "peak_value_complex": self.peak_value_complex,
            "peak_magnitude": self.peak_magnitude,
            "peak_lag_samples": self.peak_lag_samples,
            "peak_lag_sec": self.peak_lag_sec,
            "estimated_delay_samples": self.estimated_delay_samples,
            "estimated_delay_sec": self.estimated_delay_sec,
            "phase_at_peak_rad": self.phase_at_peak_rad,
            "sample_rate": self.sample_rate,
            "normalization": self.normalization,
            **self.extra
        }


@dataclass
class InterPulseCorrelationResult:
    """
    Container for pairwise or matrix inter-pulse correlation results.
    """
    pulse_id_a: int | str
    pulse_id_b: int | str
    correlation_result: CorrelationResult
    peak_similarity: float  # C_ij = max_h |rho_ij[h]|
    peak_lag_samples: int  # h_ij = argmax_h |rho_ij[h]|
    peak_lag_sec: float
    phase_offset_rad: float  # phi_ij = angle(rho_ij[h_ij])
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pulse_id_a": self.pulse_id_a,
            "pulse_id_b": self.pulse_id_b,
            "peak_similarity": self.peak_similarity,
            "peak_lag_samples": self.peak_lag_samples,
            "peak_lag_sec": self.peak_lag_sec,
            "phase_offset_rad": self.phase_offset_rad,
            "correlation": self.correlation_result.to_dict(),
            **self.extra
        }


@dataclass
class TemplateMatchResult:
    """
    Container for template matching results matching MATLAB findsignal.
    """
    best_start_sample: int  # i_best (0-indexed)
    best_stop_sample: int  # i_stop = i_best + M - 1
    best_start_sec: float
    best_stop_sec: float
    minimum_distance: float  # Minimum distance or score at best match
    distance_profile: np.ndarray  # 1D array of distance D[i] across candidate locations
    matched_segment: np.ndarray  # Complex IQ slice at best match location
    template_method: str = "matlab_findsignal"
    metric: str = "squared"
    normalization: str = "none"
    sample_rate: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_start_sample": self.best_start_sample,
            "best_stop_sample": self.best_stop_sample,
            "best_start_sec": self.best_start_sec,
            "best_stop_sec": self.best_stop_sec,
            "minimum_distance": self.minimum_distance,
            "distance_profile": self.distance_profile,
            "matched_segment": self.matched_segment,
            "template_method": self.template_method,
            "metric": self.metric,
            "normalization": self.normalization,
            "sample_rate": self.sample_rate,
            **self.extra
        }


@dataclass
class SequenceSimilarityResult:
    """
    Container for sequence similarity and Dynamic Time Warping (DTW) results.
    """
    dtw_distance: float  # D_DTW: total minimum warping path distance
    normalized_distance: float  # D_DTW / len(warping_path)
    similarity_score: float  # 1 / (1 + normalized_distance) in [0, 1]
    warping_path: list[tuple[int, int]]  # Optimal alignment path [(m, n)]
    accumulated_cost_matrix: np.ndarray  # 2D DP distance matrix
    metric: str = "euclidean"
    sequence_a_length: int = 0
    sequence_b_length: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtw_distance": self.dtw_distance,
            "normalized_distance": self.normalized_distance,
            "similarity_score": self.similarity_score,
            "warping_path": self.warping_path,
            "accumulated_cost_matrix": self.accumulated_cost_matrix,
            "metric": self.metric,
            "sequence_a_length": self.sequence_a_length,
            "sequence_b_length": self.sequence_b_length,
            **self.extra
        }


class BaseCorrelationMetric(ABC):
    """
    Abstract Base Class for all Correlation and Similarity Analysis Metrics and Stages.
    Defines the standard lifecycle interface for autocorrelation, cross-correlation,
    inter-pulse correlation, waveform similarity, template matching, sequence similarity, and lag analysis.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Constructor: Initializes metric configuration and lifecycle state."""
        self._config: dict[str, Any] = dict(kwargs)
        self._is_active: bool = True

    def __del__(self) -> None:
        """Destructor: Releases resources and clears internal configuration."""
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
        """Unique identifier name for this correlation/similarity metric or stage."""
        pass

    @property
    def description(self) -> str:
        """Human-readable description of what this metric measures."""
        return self.__doc__ or ""

    @abstractmethod
    def compute(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Executes the correlation, matching, or similarity measurement on input data.

        Parameters:
            data: Complex IQ array, canonical dictionary, pulse object, or feature sequence.
            *args: Optional second signal/template/sequence.
            **kwargs: Additional computation parameters (e.g. sample_rate, maxlag, normalization).

        Returns:
            Computed numerical scalar, CorrelationResult, TemplateMatchResult, SequenceSimilarityResult, or dictionary.
        """
        pass

    def __call__(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        """Allows metric instances to be called directly as functions: metric(x, y)."""
        return self.compute(data, *args, **kwargs)

    def reset(self) -> None:
        """Resets any internal accumulator or streaming state."""
        pass


class CorrelationMetricFactory:
    """
    Enterprise Factory Pattern for Correlation and Similarity Analysis Metrics and Stages.
    Maintains a decoupled registry of metric classes and handles lifecycle creation.
    """

    _registry: dict[str, Type[BaseCorrelationMetric]] = {}

    @classmethod
    def register(cls, aliases: str | list[str]) -> Callable[[Type[BaseCorrelationMetric]], Type[BaseCorrelationMetric]]:
        """Decorator to register a concrete BaseCorrelationMetric subclass under one or more alias names."""
        if isinstance(aliases, str):
            aliases = [aliases]

        def decorator(subclass: Type[BaseCorrelationMetric]) -> Type[BaseCorrelationMetric]:
            for alias in aliases:
                normalized = alias.strip().lower().replace(" ", "_").replace("-", "_")
                cls._registry[normalized] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, metric_name: str, **kwargs: Any) -> BaseCorrelationMetric:
        """Factory creation method: Instantiates and returns the requested metric implementation."""
        normalized = metric_name.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized not in cls._registry:
            valid_keys = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown correlation/similarity metric '{metric_name}'. Registered metrics: [{valid_keys}]")
        return cls._registry[normalized](**kwargs)

    @classmethod
    def list_metrics(cls) -> list[str]:
        """Returns a sorted list of all unique registered correlation/similarity metric names."""
        return sorted(list(set(cls._registry.keys())))
