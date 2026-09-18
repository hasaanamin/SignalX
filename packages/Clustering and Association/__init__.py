from __future__ import annotations
from typing import Any

# 1. Base interface, Factory & Data Structures
try:
    from .base import (
        BaseClusteringMetric,
        ClusteringMetricFactory,
        IQEvent,
        PulseBurst,
        EmitterTrack,
        PulseClusterResult,
        EmitterGroupResult,
        BurstGroupResult,
        ParameterSpaceClusterResult,
        DensityClusterResult,
        AssociationResult,
        TrackSequenceResult,
    )
    from ._common import (
        RawIQAdapter,
        FEATURE_NAMES,
        DEFAULT_FEATURE_WEIGHTS,
        detect_iq_events,
        extract_event_features,
        extract_events_from_input,
        normalize_features,
        resolve_feature_weights,
        compute_weighted_distance_matrix,
        perform_pulse_clustering,
        analyze_pri_sequence,
        hungarian_matching,
        kmeans_clustering,
        gmm_clustering,
        kmedoids_clustering,
        dbscan_clustering,
        compute_silhouette_score,
    )
    from .Pulse_Clustering import PulseClustering, compute_pulse_clustering
    from .Emitter_Grouping import EmitterGrouping, compute_emitter_grouping
    from .Burst_Grouping import BurstGrouping, compute_burst_grouping
    from .Parameter_Space_Clustering import ParameterSpaceClustering, compute_parameter_space_clustering
    from .Density_Based_Clustering import DensityBasedClustering, compute_density_based_clustering
    from .Nearest_Neighbor_Association import NearestNeighborAssociation, compute_nearest_neighbor_association
    from .Track_Sequence_Association import TrackSequenceAssociation, compute_track_sequence_association
except (ImportError, ValueError):
    from base import (
        BaseClusteringMetric,
        ClusteringMetricFactory,
        IQEvent,
        PulseBurst,
        EmitterTrack,
        PulseClusterResult,
        EmitterGroupResult,
        BurstGroupResult,
        ParameterSpaceClusterResult,
        DensityClusterResult,
        AssociationResult,
        TrackSequenceResult,
    )
    from _common import (
        RawIQAdapter,
        FEATURE_NAMES,
        DEFAULT_FEATURE_WEIGHTS,
        detect_iq_events,
        extract_event_features,
        extract_events_from_input,
        normalize_features,
        resolve_feature_weights,
        compute_weighted_distance_matrix,
        perform_pulse_clustering,
        analyze_pri_sequence,
        hungarian_matching,
        kmeans_clustering,
        gmm_clustering,
        kmedoids_clustering,
        dbscan_clustering,
        compute_silhouette_score,
    )
    from Pulse_Clustering import PulseClustering, compute_pulse_clustering
    from Emitter_Grouping import EmitterGrouping, compute_emitter_grouping
    from Burst_Grouping import BurstGrouping, compute_burst_grouping
    from Parameter_Space_Clustering import ParameterSpaceClustering, compute_parameter_space_clustering
    from Density_Based_Clustering import DensityBasedClustering, compute_density_based_clustering
    from Nearest_Neighbor_Association import NearestNeighborAssociation, compute_nearest_neighbor_association
    from Track_Sequence_Association import TrackSequenceAssociation, compute_track_sequence_association


def clustering_summary(
    data: Any,
    sample_rate: float = 1.0,
    threshold_k: float = 3.5,
    min_event_samples: int = 16,
    merge_gap_samples: int = 8,
    distance_threshold: float = 1.5,
    n_clusters: int | None = None,
    linkage: str = "average",
    feature_weights: dict[str, float] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Unified summary pipeline for RF pulse clustering and association.
    Directly extracts pulse events from complex IQ and clusters them based on RF waveform signatures.
    """
    processor = PulseClustering(
        sample_rate=sample_rate,
        threshold_k=threshold_k,
        min_event_samples=min_event_samples,
        merge_gap_samples=merge_gap_samples,
        distance_threshold=distance_threshold,
        n_clusters=n_clusters,
        linkage=linkage,
        feature_weights=feature_weights,
        **kwargs,
    )
    res = processor.compute(data)
    return res.to_dict()


__all__ = [
    # Base & Factory
    "BaseClusteringMetric",
    "ClusteringMetricFactory",
    # Dataclasses
    "IQEvent",
    "PulseBurst",
    "EmitterTrack",
    "PulseClusterResult",
    "EmitterGroupResult",
    "BurstGroupResult",
    "ParameterSpaceClusterResult",
    "DensityClusterResult",
    "AssociationResult",
    "TrackSequenceResult",
    # 7 Canonical Module Classes
    "PulseClustering",
    "EmitterGrouping",
    "BurstGrouping",
    "ParameterSpaceClustering",
    "DensityBasedClustering",
    "NearestNeighborAssociation",
    "TrackSequenceAssociation",
    # 7 Functional APIs
    "compute_pulse_clustering",
    "compute_emitter_grouping",
    "compute_burst_grouping",
    "compute_parameter_space_clustering",
    "compute_density_based_clustering",
    "compute_nearest_neighbor_association",
    "compute_track_sequence_association",
    # Summary Pipeline
    "clustering_summary",
    # Constants & Helpers
    "RawIQAdapter",
    "FEATURE_NAMES",
    "DEFAULT_FEATURE_WEIGHTS",
    "detect_iq_events",
    "extract_event_features",
    "extract_events_from_input",
    "normalize_features",
    "resolve_feature_weights",
    "compute_weighted_distance_matrix",
    "perform_pulse_clustering",
    "analyze_pri_sequence",
    "hungarian_matching",
    "kmeans_clustering",
    "gmm_clustering",
    "kmedoids_clustering",
    "dbscan_clustering",
    "compute_silhouette_score",
]
