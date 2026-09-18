from __future__ import annotations
import importlib

_pkg = importlib.import_module("packages.Clustering and Association")

# Base & Factory
BaseClusteringMetric = _pkg.BaseClusteringMetric
ClusteringMetricFactory = _pkg.ClusteringMetricFactory

# Dataclasses
IQEvent = _pkg.IQEvent
PulseBurst = _pkg.PulseBurst
EmitterTrack = _pkg.EmitterTrack
PulseClusterResult = _pkg.PulseClusterResult
EmitterGroupResult = _pkg.EmitterGroupResult
BurstGroupResult = _pkg.BurstGroupResult
ParameterSpaceClusterResult = _pkg.ParameterSpaceClusterResult
DensityClusterResult = _pkg.DensityClusterResult
AssociationResult = _pkg.AssociationResult
TrackSequenceResult = _pkg.TrackSequenceResult

# 7 Canonical Module Classes
PulseClustering = _pkg.PulseClustering
EmitterGrouping = _pkg.EmitterGrouping
BurstGrouping = _pkg.BurstGrouping
ParameterSpaceClustering = _pkg.ParameterSpaceClustering
DensityBasedClustering = _pkg.DensityBasedClustering
NearestNeighborAssociation = _pkg.NearestNeighborAssociation
TrackSequenceAssociation = _pkg.TrackSequenceAssociation

# 7 Functional APIs
compute_pulse_clustering = _pkg.compute_pulse_clustering
compute_emitter_grouping = _pkg.compute_emitter_grouping
compute_burst_grouping = _pkg.compute_burst_grouping
compute_parameter_space_clustering = _pkg.compute_parameter_space_clustering
compute_density_based_clustering = _pkg.compute_density_based_clustering
compute_nearest_neighbor_association = _pkg.compute_nearest_neighbor_association
compute_track_sequence_association = _pkg.compute_track_sequence_association

# Summary Pipeline
clustering_summary = _pkg.clustering_summary

# Constants & Helpers
RawIQAdapter = _pkg.RawIQAdapter
FEATURE_NAMES = _pkg.FEATURE_NAMES
DEFAULT_FEATURE_WEIGHTS = _pkg.DEFAULT_FEATURE_WEIGHTS
detect_iq_events = _pkg.detect_iq_events
extract_event_features = _pkg.extract_event_features
extract_events_from_input = _pkg.extract_events_from_input
normalize_features = _pkg.normalize_features
resolve_feature_weights = _pkg.resolve_feature_weights
compute_weighted_distance_matrix = _pkg.compute_weighted_distance_matrix
perform_pulse_clustering = _pkg.perform_pulse_clustering
analyze_pri_sequence = _pkg.analyze_pri_sequence
hungarian_matching = _pkg.hungarian_matching
kmeans_clustering = _pkg.kmeans_clustering
gmm_clustering = _pkg.gmm_clustering
kmedoids_clustering = _pkg.kmedoids_clustering
dbscan_clustering = _pkg.dbscan_clustering
compute_silhouette_score = _pkg.compute_silhouette_score

__all__ = list(_pkg.__all__)
