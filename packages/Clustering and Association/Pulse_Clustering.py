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

from .base import BaseClusteringMetric, ClusteringMetricFactory, IQEvent, PulseClusterResult
from ._common import (
    FEATURE_NAMES,
    DEFAULT_FEATURE_WEIGHTS,
    extract_events_from_input,
    normalize_features,
    resolve_feature_weights,
    perform_pulse_clustering,
)


@ClusteringMetricFactory.register([
    "pulse_clustering",
    "pulse_cluster",
    "pulse_association",
    "pulseclustering",
    "pulse_classifier",
    "rf_pulse_clustering",
])
class PulseClustering(BaseClusteringMetric):
    """
    Production-Quality RF Pulse Clustering and Association Processor.

    Accepts raw complex IQ, list of PulseCandidate objects, list of IQEvent objects,
    list of dicts, or precomputed 2D feature matrices.
    Performs instantaneous power calculation / pulse segmentation,
    spectral & temporal RF feature extraction, robust MAD normalization,
    weighted distance computation, and agglomerative clustering.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        threshold_k: float = 3.5,
        min_event_samples: int = 16,
        merge_gap_samples: int = 8,
        distance_threshold: float = 1.5,
        n_clusters: int | None = None,
        linkage: str = "average",
        feature_weights: dict[str, float] | np.ndarray | None = None,
        return_iq_segments: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)

        self.threshold_k = float(threshold_k)
        if self.threshold_k <= 0 or not np.isfinite(self.threshold_k):
            raise ValueError(f"threshold_k must be a positive finite float, got {threshold_k}")

        self.min_event_samples = int(min_event_samples)
        if self.min_event_samples < 1:
            raise ValueError(f"min_event_samples must be >= 1, got {min_event_samples}")

        self.merge_gap_samples = int(merge_gap_samples)
        if self.merge_gap_samples < 0:
            raise ValueError(f"merge_gap_samples must be >= 0, got {merge_gap_samples}")

        self.distance_threshold = float(distance_threshold)
        if self.distance_threshold <= 0 or not np.isfinite(self.distance_threshold):
            raise ValueError(f"distance_threshold must be a positive finite float, got {distance_threshold}")

        self.n_clusters = int(n_clusters) if n_clusters is not None else None
        if self.n_clusters is not None and self.n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1 when specified, got {n_clusters}")

        self.linkage = str(linkage).lower().strip()
        if self.linkage not in ("average", "complete", "single", "ward"):
            raise ValueError(f"Unsupported linkage '{linkage}'. Expected one of: 'average', 'complete', 'single', 'ward'.")

        self.feature_weights = feature_weights
        self.return_iq_segments = bool(return_iq_segments)

    @property
    def name(self) -> str:
        return "pulse_clustering"

    def compute(self, data: Any, **kwargs: Any) -> PulseClusterResult:
        """
        Executes end-to-end pulse activity detection and clustering.

        Parameters
        ----------
        data : complex ndarray, list of PulseCandidate, list of IQEvent, dict, or 2D feature matrix.

        Returns
        -------
        PulseClusterResult
            Structured result containing detected events, cluster labels, centroids,
            spreads, cluster sizes, and per-event confidence scores.
        """
        sample_rate = float(kwargs.get("sample_rate", self.sample_rate))
        threshold_k = float(kwargs.get("threshold_k", self.threshold_k))
        min_event_samples = int(kwargs.get("min_event_samples", self.min_event_samples))
        merge_gap_samples = int(kwargs.get("merge_gap_samples", self.merge_gap_samples))
        distance_threshold = float(kwargs.get("distance_threshold", self.distance_threshold))
        n_clusters = kwargs.get("n_clusters", self.n_clusters)
        linkage_method = str(kwargs.get("linkage", self.linkage)).lower().strip()
        custom_weights = kwargs.get("feature_weights", self.feature_weights)
        return_iq = bool(kwargs.get("return_iq_segments", self.return_iq_segments))

        events, raw_features, feat_names = extract_events_from_input(
            data=data,
            sample_rate=sample_rate,
            threshold_k=threshold_k,
            min_event_samples=min_event_samples,
            merge_gap_samples=merge_gap_samples,
            return_iq_segments=return_iq,
            feature_names=FEATURE_NAMES,
        )

        n_events = len(events)
        if n_events == 0:
            w_arr = resolve_feature_weights(feat_names, custom_weights)
            return perform_pulse_clustering(
                events=[],
                raw_features=np.empty((0, len(feat_names)), dtype=np.float64),
                norm_features=np.empty((0, len(feat_names)), dtype=np.float64),
                feature_names=feat_names,
                feature_weights=w_arr,
                distance_threshold=distance_threshold,
                n_clusters=n_clusters,
                linkage_method=linkage_method,
                sample_rate=sample_rate,
            )

        norm_features = normalize_features(raw_features, sample_rate=sample_rate, feature_names=feat_names)
        weights_arr = resolve_feature_weights(feat_names, custom_weights)

        result = perform_pulse_clustering(
            events=events,
            raw_features=raw_features,
            norm_features=norm_features,
            feature_names=feat_names,
            feature_weights=weights_arr,
            distance_threshold=distance_threshold,
            n_clusters=n_clusters,
            linkage_method=linkage_method,
            sample_rate=sample_rate,
        )

        return result


def compute_pulse_clustering(
    data: Any,
    sample_rate: float = 1.0,
    threshold_k: float = 3.5,
    min_event_samples: int = 16,
    merge_gap_samples: int = 8,
    distance_threshold: float = 1.5,
    n_clusters: int | None = None,
    linkage: str = "average",
    feature_weights: dict[str, float] | np.ndarray | None = None,
    return_iq_segments: bool = False,
    **kwargs: Any,
) -> PulseClusterResult:
    """Convenience functional API for pulse clustering."""
    processor = PulseClustering(
        sample_rate=sample_rate,
        threshold_k=threshold_k,
        min_event_samples=min_event_samples,
        merge_gap_samples=merge_gap_samples,
        distance_threshold=distance_threshold,
        n_clusters=n_clusters,
        linkage=linkage,
        feature_weights=feature_weights,
        return_iq_segments=return_iq_segments,
        **kwargs,
    )
    return processor.compute(data)


def main() -> None:
    print("=" * 60)
    print("  PulseClustering: Demonstration")
    print("=" * 60)
    pulses = [
        {"toa": 0.001, "freq": 1e9, "pw": 1e-6, "pri": 100e-6, "amp": 1.0},
        {"toa": 0.0011, "freq": 1e9, "pw": 1e-6, "pri": 100e-6, "amp": 1.0},
        {"toa": 0.002, "freq": 2e9, "pw": 2e-6, "pri": 200e-6, "amp": 0.8},
        {"toa": 0.0022, "freq": 2e9, "pw": 2e-6, "pri": 200e-6, "amp": 0.8},
    ]
    metric = PulseClustering()
    result = metric.compute(pulses)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
