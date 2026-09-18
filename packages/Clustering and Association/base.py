from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Type
import numpy as np


@dataclass
class IQEvent:
    """
    Represents a detected pulse/activity region directly extracted from complex IQ or PulseCandidate.
    Preserves sample-level and temporal traceability into the original IQ capture.
    """
    start_sample: int
    end_sample: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    mean_power_db: float
    spectral_centroid_hz: float
    bandwidth_hz: float
    spectral_entropy: float
    spectral_flatness: float
    chirp_slope_hz_per_sec: float = 0.0
    chirp_slope_valid: bool = False
    carrier_freq_hz: float | None = None
    peak_power_db: float | None = None
    snr_db: float | None = None
    is_synthetic_time: bool = False
    raw_iq: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Converts IQEvent to a serializable dictionary."""
        d = {
            "start_sample": int(self.start_sample),
            "end_sample": int(self.end_sample),
            "start_time_sec": float(self.start_time_sec),
            "end_time_sec": float(self.end_time_sec),
            "duration_sec": float(self.duration_sec),
            "mean_power_db": float(self.mean_power_db),
            "spectral_centroid_hz": float(self.spectral_centroid_hz),
            "bandwidth_hz": float(self.bandwidth_hz),
            "spectral_entropy": float(self.spectral_entropy),
            "spectral_flatness": float(self.spectral_flatness),
            "chirp_slope_hz_per_sec": float(self.chirp_slope_hz_per_sec),
            "chirp_slope_valid": bool(self.chirp_slope_valid),
            "is_synthetic_time": bool(self.is_synthetic_time),
        }
        if self.carrier_freq_hz is not None:
            d["carrier_freq_hz"] = float(self.carrier_freq_hz)
        if self.peak_power_db is not None:
            d["peak_power_db"] = float(self.peak_power_db)
        if self.snr_db is not None:
            d["snr_db"] = float(self.snr_db)
        if self.extra:
            d.update(self.extra)
        return d

    def as_dict(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass
class PulseBurst:
    """
    Represents a grouped burst / dwell of multiple consecutive pulses from an emitter.
    """
    burst_id: int
    pulse_indices: list[int]
    pulse_count: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    bri_sec: float
    intra_burst_pri_mean_sec: float
    intra_burst_pri_std_sec: float
    duty_cycle: float
    mean_frequency_hz: float
    mean_power_db: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "burst_id": int(self.burst_id),
            "pulse_indices": list(self.pulse_indices),
            "pulse_count": int(self.pulse_count),
            "start_time_sec": float(self.start_time_sec),
            "end_time_sec": float(self.end_time_sec),
            "duration_sec": float(self.duration_sec),
            "bri_sec": float(self.bri_sec),
            "intra_burst_pri_mean_sec": float(self.intra_burst_pri_mean_sec),
            "intra_burst_pri_std_sec": float(self.intra_burst_pri_std_sec),
            "duty_cycle": float(self.duty_cycle),
            "mean_frequency_hz": float(self.mean_frequency_hz),
            "mean_power_db": float(self.mean_power_db),
            **self.extra,
        }

    def as_dict(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass
class EmitterTrack:
    """
    Represents a sequential track of pulses maintained by kinematic/PRI prediction filtering.
    """
    track_id: int
    state: str  # 'tentative', 'confirmed', 'coasting', 'terminated'
    pulse_indices: list[int]
    toas: list[float]
    estimated_pri_sec: float
    estimated_pri_jitter: float
    estimated_fc_hz: float
    estimated_pw_sec: float
    last_toa_sec: float
    predicted_next_toa_sec: float
    hits: int
    misses: int
    total_updates: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": int(self.track_id),
            "state": str(self.state),
            "pulse_indices": list(self.pulse_indices),
            "toas": [float(t) for t in self.toas],
            "estimated_pri_sec": float(self.estimated_pri_sec),
            "estimated_pri_jitter": float(self.estimated_pri_jitter),
            "estimated_fc_hz": float(self.estimated_fc_hz),
            "estimated_pw_sec": float(self.estimated_pw_sec),
            "last_toa_sec": float(self.last_toa_sec),
            "predicted_next_toa_sec": float(self.predicted_next_toa_sec),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "total_updates": int(self.total_updates),
            **self.extra,
        }

    def as_dict(self) -> dict[str, Any]:
        return self.to_dict()


class _DictCompatibleResult:
    """Mixin for dict indexing and compatibility on all result dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def as_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}.")

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()

    def keys(self) -> Any:
        return self.to_dict().keys()

    def values(self) -> Any:
        return self.to_dict().values()

    def items(self) -> Any:
        return self.to_dict().items()


@dataclass
class PulseClusterResult(_DictCompatibleResult):
    """
    Result container for RF pulse clustering / deinterleaving.
    """
    events: list[IQEvent]
    cluster_labels: np.ndarray
    num_clusters: int
    feature_names: tuple[str, ...]
    raw_features: np.ndarray
    normalized_features: np.ndarray
    cluster_centroids: np.ndarray
    cluster_spreads: np.ndarray
    cluster_sizes: np.ndarray
    confidence: np.ndarray
    sample_rate: float = 1.0
    distance_threshold: float = 1.5
    linkage: str = "average"
    feature_weights: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [ev.to_dict() for ev in self.events],
            "cluster_labels": self.cluster_labels,
            "num_clusters": int(self.num_clusters),
            "feature_names": self.feature_names,
            "raw_features": self.raw_features,
            "normalized_features": self.normalized_features,
            "cluster_centroids": self.cluster_centroids,
            "cluster_spreads": self.cluster_spreads,
            "cluster_sizes": self.cluster_sizes,
            "confidence": self.confidence,
            "sample_rate": float(self.sample_rate),
            "distance_threshold": float(self.distance_threshold),
            "linkage": self.linkage,
            "feature_weights": self.feature_weights,
            **self.extra,
        }


@dataclass
class EmitterGroupResult(_DictCompatibleResult):
    """
    Result container for radar/comms emitter grouping and PRI sequence profiling.
    """
    events: list[IQEvent]
    emitter_ids: np.ndarray
    num_emitters: int
    emitter_profiles: list[dict[str, Any]]
    raw_features: np.ndarray
    normalized_features: np.ndarray
    confidence: np.ndarray
    sample_rate: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [ev.to_dict() for ev in self.events],
            "emitter_ids": self.emitter_ids,
            "num_emitters": int(self.num_emitters),
            "emitter_profiles": self.emitter_profiles,
            "raw_features": self.raw_features,
            "normalized_features": self.normalized_features,
            "confidence": self.confidence,
            "sample_rate": float(self.sample_rate),
            **self.extra,
        }


@dataclass
class BurstGroupResult(_DictCompatibleResult):
    """
    Result container for pulse burst / dwell temporal segmentation.
    """
    events: list[IQEvent]
    burst_labels: np.ndarray
    num_bursts: int
    bursts: list[PulseBurst]
    inter_burst_gap_threshold_sec: float
    sample_rate: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [ev.to_dict() for ev in self.events],
            "burst_labels": self.burst_labels,
            "num_bursts": int(self.num_bursts),
            "bursts": [b.to_dict() for b in self.bursts],
            "inter_burst_gap_threshold_sec": float(self.inter_burst_gap_threshold_sec),
            "sample_rate": float(self.sample_rate),
            **self.extra,
        }


@dataclass
class ParameterSpaceClusterResult(_DictCompatibleResult):
    """
    Result container for multi-dimensional partition clustering (K-Means, GMM, Medoids).
    """
    events: list[IQEvent]
    cluster_labels: np.ndarray
    num_clusters: int
    method: str
    feature_names: tuple[str, ...]
    raw_features: np.ndarray
    normalized_features: np.ndarray
    cluster_centroids: np.ndarray
    cluster_spreads: np.ndarray
    cluster_sizes: np.ndarray
    inertia: float
    silhouette_score: float
    confidence: np.ndarray
    sample_rate: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [ev.to_dict() for ev in self.events],
            "cluster_labels": self.cluster_labels,
            "num_clusters": int(self.num_clusters),
            "method": self.method,
            "feature_names": self.feature_names,
            "raw_features": self.raw_features,
            "normalized_features": self.normalized_features,
            "cluster_centroids": self.cluster_centroids,
            "cluster_spreads": self.cluster_spreads,
            "cluster_sizes": self.cluster_sizes,
            "inertia": float(self.inertia),
            "silhouette_score": float(self.silhouette_score),
            "confidence": self.confidence,
            "sample_rate": float(self.sample_rate),
            **self.extra,
        }


@dataclass
class DensityClusterResult(_DictCompatibleResult):
    """
    Result container for non-parametric density-based clustering (DBSCAN / noise isolation).
    """
    events: list[IQEvent]
    cluster_labels: np.ndarray  # -1 represents noise / unassociated pulses
    num_clusters: int           # number of valid clusters excluding noise
    num_noise: int
    noise_indices: np.ndarray
    core_indices: np.ndarray
    eps: float
    min_samples: int
    feature_names: tuple[str, ...]
    raw_features: np.ndarray
    normalized_features: np.ndarray
    cluster_centroids: np.ndarray
    cluster_sizes: np.ndarray
    auto_eps_method: str | None = None
    k_distance_k: int | None = None
    candidate_eps: list[float] | None = None
    sample_rate: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "events": [ev.to_dict() for ev in self.events],
            "cluster_labels": self.cluster_labels,
            "num_clusters": int(self.num_clusters),
            "num_noise": int(self.num_noise),
            "noise_indices": self.noise_indices,
            "core_indices": self.core_indices,
            "eps": float(self.eps),
            "eps_used": float(self.eps),
            "min_samples": int(self.min_samples),
            "feature_names": self.feature_names,
            "raw_features": self.raw_features,
            "normalized_features": self.normalized_features,
            "cluster_centroids": self.cluster_centroids,
            "cluster_sizes": self.cluster_sizes,
            "sample_rate": float(self.sample_rate),
        }
        if self.auto_eps_method is not None:
            d["auto_eps_method"] = str(self.auto_eps_method)
        if self.k_distance_k is not None:
            d["k_distance_k"] = int(self.k_distance_k)
        if self.candidate_eps is not None:
            d["candidate_eps"] = [float(e) for e in self.candidate_eps]
        d.update(self.extra)
        return d


@dataclass
class AssociationResult(_DictCompatibleResult):
    """
    Result container for nearest neighbor template / library matching.
    """
    events: list[IQEvent]
    matches: list[dict[str, Any]]
    unassociated_queries: list[int]
    unassociated_templates: list[int | str]
    distance_matrix: np.ndarray
    gated_distance_matrix: np.ndarray
    assignment_method: str
    gate_threshold: float
    metric: str
    requested_gate_threshold: float | None = None
    effective_gate_threshold: float | None = None
    sample_rate: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "events": [ev.to_dict() for ev in self.events],
            "matches": self.matches,
            "unassociated_queries": self.unassociated_queries,
            "unassociated_templates": self.unassociated_templates,
            "distance_matrix": self.distance_matrix,
            "gated_distance_matrix": self.gated_distance_matrix,
            "assignment_method": self.assignment_method,
            "gate_threshold": float(self.gate_threshold),
            "metric": self.metric,
            "sample_rate": float(self.sample_rate),
        }
        if self.requested_gate_threshold is not None:
            d["requested_gate_threshold"] = float(self.requested_gate_threshold)
        if self.effective_gate_threshold is not None:
            d["effective_gate_threshold"] = float(self.effective_gate_threshold)
        d.update(self.extra)
        return d


@dataclass
class TrackSequenceResult(_DictCompatibleResult):
    """
    Result container for sequential pulse stream tracking.
    """
    events: list[IQEvent]
    tracks: list[EmitterTrack]
    pulse_track_map: np.ndarray  # per-pulse track ID (-1 if unassigned)
    num_active_tracks: int
    num_confirmed_tracks: int
    sample_rate: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [ev.to_dict() for ev in self.events],
            "tracks": [t.to_dict() for t in self.tracks],
            "pulse_track_map": self.pulse_track_map,
            "num_active_tracks": int(self.num_active_tracks),
            "num_confirmed_tracks": int(self.num_confirmed_tracks),
            "sample_rate": float(self.sample_rate),
            **self.extra,
        }


class BaseClusteringMetric(ABC):
    """
    Abstract base class for all clustering and association processors.
    """

    def __init__(self, sample_rate: float = 1.0, **kwargs: Any) -> None:
        self.sample_rate = float(sample_rate)
        if self.sample_rate <= 0 or not np.isfinite(self.sample_rate):
            raise ValueError(f"sample_rate must be a positive finite float, got {sample_rate}")
        self.config = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the clustering stage."""
        pass

    @abstractmethod
    def compute(self, data: Any, **kwargs: Any) -> Any:
        """Executes the clustering algorithm on input data."""
        pass

    def fit_predict(self, data: Any, **kwargs: Any) -> np.ndarray:
        """Scikit-Learn compatible fit_predict method returning cluster_labels/labels."""
        res = self.compute(data, **kwargs)
        if hasattr(res, "cluster_labels"):
            return res.cluster_labels
        if hasattr(res, "emitter_ids"):
            return res.emitter_ids
        if hasattr(res, "burst_labels"):
            return res.burst_labels
        if hasattr(res, "pulse_track_map"):
            return res.pulse_track_map
        if isinstance(res, dict) and "cluster_labels" in res:
            return res["cluster_labels"]
        return np.empty(0, dtype=int)

    def process(self, data: Any, **kwargs: Any) -> Any:
        """Pipeline compatibility alias for compute."""
        return self.compute(data, **kwargs)

    def __call__(self, data: Any, **kwargs: Any) -> Any:
        """Callable processor interface."""
        return self.compute(data, **kwargs)


class ClusteringMetricFactory:
    """
    Factory for creating and discovering clustering / association metrics.
    """
    _registry: dict[str, Type[BaseClusteringMetric]] = {}

    @classmethod
    def register(cls, names: str | list[str]) -> Callable[[Type[BaseClusteringMetric]], Type[BaseClusteringMetric]]:
        """Decorator to register a clustering metric under one or more alias names."""
        if isinstance(names, str):
            names = [names]

        def decorator(subclass: Type[BaseClusteringMetric]) -> Type[BaseClusteringMetric]:
            for name in names:
                normalized_name = name.lower().strip().replace(" ", "_").replace("-", "_")
                cls._registry[normalized_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, name: str, sample_rate: float = 1.0, **kwargs: Any) -> BaseClusteringMetric:
        """Creates an instance of the requested clustering metric."""
        normalized_name = name.lower().strip().replace(" ", "_").replace("-", "_")
        if normalized_name not in cls._registry:
            valid = sorted(cls._registry.keys())
            raise ValueError(f"Unknown clustering metric '{name}'. Available: {valid}")
        metric_class = cls._registry[normalized_name]
        return metric_class(sample_rate=sample_rate, **kwargs)

    @classmethod
    def list_metrics(cls) -> list[str]:
        """Returns sorted list of registered metric names."""
        return sorted(cls._registry.keys())
