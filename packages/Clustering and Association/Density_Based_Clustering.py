from __future__ import annotations
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    _pkg_dir = Path(__file__).resolve().parent
    _src_dir = _pkg_dir.parent.parent
    for _p in [str(_src_dir), str(_pkg_dir.parent)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import importlib
    importlib.import_module(f"packages.{_pkg_dir.name}")
    __package__ = f"packages.{_pkg_dir.name}"

from typing import Any
import numpy as np

from .base import BaseClusteringMetric, ClusteringMetricFactory, IQEvent, DensityClusterResult
from ._common import (
    FEATURE_NAMES,
    extract_events_from_input,
    normalize_features,
    resolve_feature_weights,
    dbscan_clustering,
)


@ClusteringMetricFactory.register([
    "density_based_clustering",
    "density_clustering",
    "dbscan_clustering",
    "densitybasedclustering",
    "dbscan",
    "density_cluster",
])
class DensityBasedClustering(BaseClusteringMetric):
    """
    Non-Parametric Density-Based RF Pulse Clustering Processor (DBSCAN).

    Identifies arbitrary-geometry pulse clusters and explicitly isolates sparse
    unsynchronized noise pulses or rare outliers by labeling them with cluster_label = -1.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        eps: float | None = None,
        min_samples: int = 3,
        feature_weights: dict[str, float] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)

        self.eps = float(eps) if eps is not None else None
        if self.eps is not None and self.eps <= 0:
            raise ValueError(f"eps must be > 0 when specified, got {eps}")

        self.min_samples = int(min_samples)
        if self.min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}")

        self.feature_weights = feature_weights

    @property
    def name(self) -> str:
        return "density_based_clustering"

    def compute(self, data: Any, **kwargs: Any) -> DensityClusterResult:
        """
        Executes DBSCAN density-based pulse clustering.

        Parameters
        ----------
        data : complex ndarray, list of PulseCandidate, list of IQEvent, dict, or 2D feature matrix.

        Returns
        -------
        DensityClusterResult
        """
        sample_rate = float(kwargs.get("sample_rate", self.sample_rate))
        eps_param = kwargs.get("eps", self.eps)
        min_samples = int(kwargs.get("min_samples", self.min_samples))
        custom_weights = kwargs.get("feature_weights", self.feature_weights)

        events, raw_features, feat_names = extract_events_from_input(
            data=data,
            sample_rate=sample_rate,
            feature_names=FEATURE_NAMES,
            **kwargs,
        )

        n_events = len(events)
        n_feat = len(feat_names)

        if n_events == 0:
            return DensityClusterResult(
                events=[],
                cluster_labels=np.empty(0, dtype=int),
                num_clusters=0,
                num_noise=0,
                noise_indices=np.empty(0, dtype=int),
                core_indices=np.empty(0, dtype=int),
                eps=0.0,
                min_samples=min_samples,
                feature_names=feat_names,
                raw_features=np.empty((0, n_feat), dtype=np.float64),
                normalized_features=np.empty((0, n_feat), dtype=np.float64),
                cluster_centroids=np.empty((0, n_feat), dtype=np.float64),
                cluster_sizes=np.empty(0, dtype=int),
                sample_rate=sample_rate,
            )

        norm_features = normalize_features(raw_features, sample_rate=sample_rate, feature_names=feat_names)
        weights_arr = resolve_feature_weights(feat_names, custom_weights)
        weighted_features = norm_features * np.sqrt(np.maximum(weights_arr, 0.0))[None, :]

        # Perform DBSCAN
        labels, core_indices, eps_used, diag = dbscan_clustering(
            features=weighted_features,
            eps=eps_param,
            min_samples=min_samples,
            return_diagnostics=True,
        )

        # Separate valid clusters from noise points (label == -1)
        valid_cluster_labels = [l for l in sorted(list(set(labels))) if l >= 0]
        num_clusters = len(valid_cluster_labels)

        # Remap valid clusters to 0..K-1
        lbl_map = {old: new for new, old in enumerate(valid_cluster_labels)}
        remapped_labels = np.array([lbl_map[l] if l >= 0 else -1 for l in labels], dtype=int)

        noise_mask = remapped_labels == -1
        noise_indices = np.where(noise_mask)[0]
        num_noise = int(len(noise_indices))

        cluster_centroids = np.zeros((num_clusters, n_feat), dtype=np.float64)
        cluster_sizes = np.zeros(num_clusters, dtype=int)

        for c in range(num_clusters):
            mask = remapped_labels == c
            c_raw = raw_features[mask]
            cluster_sizes[c] = len(c_raw)
            cluster_centroids[c] = np.mean(c_raw, axis=0)

        return DensityClusterResult(
            events=events,
            cluster_labels=remapped_labels,
            num_clusters=num_clusters,
            num_noise=num_noise,
            noise_indices=noise_indices,
            core_indices=core_indices,
            eps=eps_used,
            min_samples=min_samples,
            feature_names=feat_names,
            raw_features=raw_features,
            normalized_features=norm_features,
            cluster_centroids=cluster_centroids,
            cluster_sizes=cluster_sizes,
            auto_eps_method=diag.get("auto_eps_method"),
            k_distance_k=diag.get("k_distance_k"),
            candidate_eps=diag.get("candidate_eps"),
            sample_rate=sample_rate,
        )


def compute_density_based_clustering(
    data: Any,
    sample_rate: float = 1.0,
    eps: float | None = None,
    min_samples: int = 3,
    feature_weights: dict[str, float] | np.ndarray | None = None,
    **kwargs: Any,
) -> DensityClusterResult:
    """Convenience functional API for density-based pulse clustering."""
    processor = DensityBasedClustering(
        sample_rate=sample_rate,
        eps=eps,
        min_samples=min_samples,
        feature_weights=feature_weights,
        **kwargs,
    )
    return processor.compute(data)


def main() -> None:
    print("=" * 60)
    print("  DensityBasedClustering: Demonstration")
    print("=" * 60)
    pulses = [
        {"toa": 0.001, "freq": 1e9, "pw": 1e-6, "pri": 100e-6, "amp": 1.0},
        {"toa": 0.0011, "freq": 1e9, "pw": 1e-6, "pri": 100e-6, "amp": 1.0},
        {"toa": 0.002, "freq": 2e9, "pw": 2e-6, "pri": 200e-6, "amp": 0.8},
        {"toa": 0.0022, "freq": 2e9, "pw": 2e-6, "pri": 200e-6, "amp": 0.8},
    ]
    metric = DensityBasedClustering()
    result = metric.compute(pulses)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
