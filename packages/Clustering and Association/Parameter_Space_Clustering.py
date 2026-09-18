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

from .base import BaseClusteringMetric, ClusteringMetricFactory, IQEvent, ParameterSpaceClusterResult
from ._common import (
    FEATURE_NAMES,
    extract_events_from_input,
    normalize_features,
    resolve_feature_weights,
    kmeans_clustering,
    gmm_clustering,
    kmedoids_clustering,
    compute_silhouette_score,
)


@ClusteringMetricFactory.register([
    "parameter_space_clustering",
    "parameter_clustering",
    "kmeans_clustering",
    "gmm_clustering",
    "parameterspaceclustering",
    "partition_clustering",
])
class ParameterSpaceClustering(BaseClusteringMetric):
    """
    Multi-Dimensional Parameter Space Clustering Processor.

    Partitions pulses in arbitrary feature subspaces (e.g. carrier frequency, pulse width,
    power, bandwidth) using K-Means, Gaussian Mixture Models (GMM), or K-Medoids.
    Supports automated cluster number (K) selection via Silhouette maximization.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        method: str = "kmeans",
        n_clusters: int | None = None,
        max_auto_k: int = 15,
        min_auto_cluster_size: int = 2,
        selected_features: list[str] | tuple[str, ...] | None = None,
        feature_weights: dict[str, float] | np.ndarray | None = None,
        max_iter: int = 100,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)

        self.method = str(method).lower().strip()
        if self.method not in ("kmeans", "k-means", "gmm", "gaussian_mixture", "kmedoids", "k-medoids", "medoids"):
            raise ValueError(f"Unsupported method '{method}'. Expected 'kmeans', 'gmm', or 'kmedoids'.")

        self.n_clusters = int(n_clusters) if n_clusters is not None else None
        if self.n_clusters is not None and self.n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1 when specified, got {n_clusters}")

        self.max_auto_k = int(max_auto_k)
        self.min_auto_cluster_size = int(min_auto_cluster_size)
        self.selected_features = tuple(selected_features) if selected_features is not None else None
        self.feature_weights = feature_weights
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)

    @property
    def name(self) -> str:
        return "parameter_space_clustering"

    def compute(self, data: Any, **kwargs: Any) -> ParameterSpaceClusterResult:
        """
        Executes parameter space partition clustering.

        Parameters
        ----------
        data : complex ndarray, list of PulseCandidate, list of IQEvent, dict, or 2D feature matrix.

        Returns
        -------
        ParameterSpaceClusterResult
        """
        sample_rate = float(kwargs.get("sample_rate", self.sample_rate))
        method = str(kwargs.get("method", self.method)).lower().strip()
        n_clusters = kwargs.get("n_clusters", self.n_clusters)
        max_auto_k = int(kwargs.get("max_auto_k", self.max_auto_k))
        min_auto_size = int(kwargs.get("min_auto_cluster_size", self.min_auto_cluster_size))
        sel_features = kwargs.get("selected_features", self.selected_features)
        custom_weights = kwargs.get("feature_weights", self.feature_weights)
        max_iter = int(kwargs.get("max_iter", self.max_iter))
        rng_seed = int(kwargs.get("random_state", self.random_state))

        events, raw_features, feat_names = extract_events_from_input(
            data=data,
            sample_rate=sample_rate,
            feature_names=FEATURE_NAMES,
            **kwargs,
        )

        n_events = len(events)
        if n_events == 0:
            return ParameterSpaceClusterResult(
                events=[],
                cluster_labels=np.empty(0, dtype=int),
                num_clusters=0,
                method=method,
                feature_names=feat_names,
                raw_features=np.empty((0, len(feat_names)), dtype=np.float64),
                normalized_features=np.empty((0, len(feat_names)), dtype=np.float64),
                cluster_centroids=np.empty((0, len(feat_names)), dtype=np.float64),
                cluster_spreads=np.empty((0, len(feat_names)), dtype=np.float64),
                cluster_sizes=np.empty(0, dtype=int),
                inertia=0.0,
                silhouette_score=0.0,
                confidence=np.empty(0, dtype=np.float64),
                sample_rate=sample_rate,
            )

        # Apply feature selection subspace if requested
        if sel_features is not None and len(sel_features) > 0:
            col_indices = [feat_names.index(f) for f in sel_features if f in feat_names]
            if len(col_indices) > 0:
                feat_names = tuple(feat_names[i] for i in col_indices)
                raw_features = raw_features[:, col_indices]

        n_feat = len(feat_names)
        norm_features = normalize_features(raw_features, sample_rate=sample_rate, feature_names=feat_names)
        weights_arr = resolve_feature_weights(feat_names, custom_weights)
        weighted_features = norm_features * np.sqrt(np.maximum(weights_arr, 0.0))[None, :]

        if n_events == 1:
            return ParameterSpaceClusterResult(
                events=events,
                cluster_labels=np.array([0], dtype=int),
                num_clusters=1,
                method=method,
                feature_names=feat_names,
                raw_features=raw_features,
                normalized_features=norm_features,
                cluster_centroids=raw_features.copy(),
                cluster_spreads=np.zeros((1, n_feat), dtype=np.float64),
                cluster_sizes=np.array([1], dtype=int),
                inertia=0.0,
                silhouette_score=1.0,
                confidence=np.array([1.0], dtype=np.float64),
                sample_rate=sample_rate,
            )

        # Automatic K selection if not provided
        if n_clusters is None:
            upper_bound = min(int(max_auto_k), n_events - 1)
            if upper_bound < 2:
                k_opt = 1
            else:
                best_sil = -2.0
                best_k = 2
                found_valid = False
                for test_k in range(2, upper_bound + 1):
                    if method in ("gmm", "gaussian_mixture"):
                        lbls, _, _ = gmm_clustering(weighted_features, test_k, max_iter=30, random_state=rng_seed)
                    elif method in ("kmedoids", "k-medoids", "medoids"):
                        lbls, _, _ = kmedoids_clustering(weighted_features, test_k, max_iter=30, random_state=rng_seed)
                    else:
                        lbls, _, _ = kmeans_clustering(weighted_features, test_k, max_iter=30, random_state=rng_seed)

                    unique_lbls, counts = np.unique(lbls, return_counts=True)
                    if len(unique_lbls) < test_k:
                        continue
                    if min_auto_size > 1 and np.min(counts) < min_auto_size:
                        continue

                    sil = compute_silhouette_score(weighted_features, lbls)
                    if sil > best_sil:
                        best_sil = sil
                        best_k = test_k
                        found_valid = True

                if not found_valid:
                    for test_k in range(2, min(8, upper_bound) + 1):
                        lbls, _, _ = kmeans_clustering(weighted_features, test_k, max_iter=30, random_state=rng_seed)
                        sil = compute_silhouette_score(weighted_features, lbls)
                        if sil > best_sil:
                            best_sil = sil
                            best_k = test_k
                k_opt = best_k
        else:
            k_opt = min(int(n_clusters), n_events)

        if k_opt <= 1:
            labels = np.zeros(n_events, dtype=int)
            centroids = np.mean(raw_features, axis=0, keepdims=True)
            inertia = float(np.sum((weighted_features - np.mean(weighted_features, axis=0)) ** 2))
        else:
            if method in ("gmm", "gaussian_mixture"):
                labels, _, inertia = gmm_clustering(weighted_features, k=k_opt, max_iter=max_iter, random_state=rng_seed)
            elif method in ("kmedoids", "k-medoids", "medoids"):
                labels, _, inertia = kmedoids_clustering(weighted_features, k=k_opt, max_iter=max_iter, random_state=rng_seed)
            else:  # kmeans / default
                labels, _, inertia = kmeans_clustering(weighted_features, k=k_opt, max_iter=max_iter, random_state=rng_seed)

        # Re-index cluster labels to 0..K-1
        unique_labels = sorted(list(set(labels)))
        lbl_map = {old: new for new, old in enumerate(unique_labels)}
        cluster_labels = np.array([lbl_map[l] for l in labels], dtype=int)
        num_clusters = len(unique_labels)

        cluster_centroids = np.zeros((num_clusters, n_feat), dtype=np.float64)
        cluster_spreads = np.zeros((num_clusters, n_feat), dtype=np.float64)
        cluster_sizes = np.zeros(num_clusters, dtype=int)
        norm_centroids = np.zeros((num_clusters, n_feat), dtype=np.float64)

        for c in range(num_clusters):
            mask = cluster_labels == c
            c_raw = raw_features[mask]
            c_norm = weighted_features[mask]
            cluster_sizes[c] = len(c_raw)
            cluster_centroids[c] = np.mean(c_raw, axis=0)
            norm_centroids[c] = np.mean(c_norm, axis=0)
            if len(c_raw) > 1:
                cluster_spreads[c] = np.std(c_raw, axis=0)
            else:
                cluster_spreads[c] = 0.0

        silhouette = compute_silhouette_score(weighted_features, cluster_labels) if num_clusters > 1 else 1.0

        confidence = np.ones(n_events, dtype=np.float64)
        if num_clusters > 1:
            for i in range(n_events):
                c_own = cluster_labels[i]
                z_i = weighted_features[i]
                mu_own = norm_centroids[c_own]
                d_own = float(np.linalg.norm(z_i - mu_own))

                d_other = float("inf")
                for c_other in range(num_clusters):
                    if c_other != c_own:
                        mu_other = norm_centroids[c_other]
                        d_c = float(np.linalg.norm(z_i - mu_other))
                        if d_c < d_other:
                            d_other = d_c

                max_d = max(d_other, d_own, 1e-6)
                conf = (d_other - d_own) / max_d
                confidence[i] = float(np.clip(0.5 * (conf + 1.0), 0.0, 1.0))

        return ParameterSpaceClusterResult(
            events=events,
            cluster_labels=cluster_labels,
            num_clusters=num_clusters,
            method=method,
            feature_names=feat_names,
            raw_features=raw_features,
            normalized_features=norm_features,
            cluster_centroids=cluster_centroids,
            cluster_spreads=cluster_spreads,
            cluster_sizes=cluster_sizes,
            inertia=inertia,
            silhouette_score=silhouette,
            confidence=confidence,
            sample_rate=sample_rate,
        )


def compute_parameter_space_clustering(
    data: Any,
    sample_rate: float = 1.0,
    method: str = "kmeans",
    n_clusters: int | None = None,
    selected_features: list[str] | tuple[str, ...] | None = None,
    feature_weights: dict[str, float] | np.ndarray | None = None,
    max_iter: int = 100,
    random_state: int = 42,
    **kwargs: Any,
) -> ParameterSpaceClusterResult:
    """Convenience functional API for parameter space partition clustering."""
    processor = ParameterSpaceClustering(
        sample_rate=sample_rate,
        method=method,
        n_clusters=n_clusters,
        selected_features=selected_features,
        feature_weights=feature_weights,
        max_iter=max_iter,
        random_state=random_state,
        **kwargs,
    )
    return processor.compute(data)


def main() -> None:
    print("=" * 60)
    print("  ParameterSpaceClustering: Demonstration")
    print("=" * 60)
    pulses = [
        {"toa": 0.001, "freq": 1e9, "pw": 1e-6, "pri": 100e-6, "amp": 1.0},
        {"toa": 0.0011, "freq": 1e9, "pw": 1e-6, "pri": 100e-6, "amp": 1.0},
        {"toa": 0.002, "freq": 2e9, "pw": 2e-6, "pri": 200e-6, "amp": 0.8},
        {"toa": 0.0022, "freq": 2e9, "pw": 2e-6, "pri": 200e-6, "amp": 0.8},
    ]
    metric = ParameterSpaceClustering()
    result = metric.compute(pulses)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
