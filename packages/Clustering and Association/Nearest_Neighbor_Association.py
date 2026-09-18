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
import math
import numpy as np

from .base import BaseClusteringMetric, ClusteringMetricFactory, IQEvent, AssociationResult
from ._common import (
    FEATURE_NAMES,
    extract_events_from_input,
    normalize_features,
    resolve_feature_weights,
    compute_weighted_distance_matrix,
    hungarian_matching,
)


@ClusteringMetricFactory.register([
    "nearest_neighbor_association",
    "template_matching",
    "nn_association",
    "emitter_library_association",
    "nearestneighborassociation",
    "template_association",
])
class NearestNeighborAssociation(BaseClusteringMetric):
    """
    Template & Emitter Library Nearest-Neighbor Association Processor.

    Associates observed pulse feature vectors to known emitter template libraries or reference
    waveforms using Euclidean, Mahalanobis, Manhattan, or Cosine distance metrics.
    Employs distance gating to reject unknown or unassociated emitters and Hungarian
    optimal assignment to prevent duplicate conflicts.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        templates: list[dict[str, Any]] | np.ndarray | None = None,
        gate_threshold: float | None = None,
        metric: str = "euclidean",
        assignment_method: str = "hungarian",
        feature_weights: dict[str, float] | np.ndarray | None = None,
        covariance: np.ndarray | None = None,
        inv_covariance: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)

        self.templates = templates
        self.gate_threshold = float(gate_threshold) if gate_threshold is not None else None
        if self.gate_threshold is not None and (self.gate_threshold <= 0 or not np.isfinite(self.gate_threshold)):
            raise ValueError(f"gate_threshold must be a positive finite float, got {gate_threshold}")

        self.metric = str(metric).lower().strip()
        if self.metric not in ("euclidean", "manhattan", "cosine", "mahalanobis"):
            raise ValueError(f"Unsupported metric '{metric}'. Expected 'euclidean', 'manhattan', 'cosine', or 'mahalanobis'.")

        self.assignment_method = str(assignment_method).lower().strip()
        if self.assignment_method not in ("hungarian", "greedy", "nearest"):
            raise ValueError(f"Unsupported assignment_method '{assignment_method}'. Expected 'hungarian', 'greedy', or 'nearest'.")

        self.feature_weights = feature_weights
        self.covariance = covariance
        self.inv_covariance = inv_covariance

    @property
    def name(self) -> str:
        return "nearest_neighbor_association"

    def compute(self, data: Any, **kwargs: Any) -> AssociationResult:
        """
        Matches query pulses against template library.

        Parameters
        ----------
        data : complex ndarray, list of PulseCandidate, list of IQEvent, dict, or 2D feature matrix.

        Returns
        -------
        AssociationResult
        """
        sample_rate = float(kwargs.get("sample_rate", self.sample_rate))
        templates_input = kwargs.get("templates", self.templates)
        gate_th = kwargs.get("gate_threshold", self.gate_threshold)
        metric = str(kwargs.get("metric", self.metric)).lower().strip()
        method = str(kwargs.get("assignment_method", self.assignment_method)).lower().strip()
        custom_weights = kwargs.get("feature_weights", self.feature_weights)

        # 1. Parse query pulse features
        query_events, query_raw, feat_names = extract_events_from_input(
            data=data,
            sample_rate=sample_rate,
            feature_names=FEATURE_NAMES,
            **kwargs,
        )

        n_queries = len(query_events)

        # 2. Parse template library
        if templates_input is None:
            template_events, template_raw = [], np.empty((0, len(feat_names)), dtype=np.float64)
            template_labels = []
        elif isinstance(templates_input, np.ndarray):
            template_events, template_raw, _ = extract_events_from_input(
                data=templates_input,
                sample_rate=sample_rate,
                feature_names=feat_names,
            )
            template_labels = [f"template_{i}" for i in range(len(template_events))]
        elif isinstance(templates_input, list):
            if len(templates_input) > 0 and isinstance(templates_input[0], dict):
                template_labels = [d.get("label", d.get("name", d.get("id", f"template_{i}"))) for i, d in enumerate(templates_input)]
            else:
                template_labels = [f"template_{i}" for i in range(len(templates_input))]
            template_events, template_raw, _ = extract_events_from_input(
                data=templates_input,
                sample_rate=sample_rate,
                feature_names=feat_names,
            )
        else:
            template_events, template_raw, _ = extract_events_from_input(
                data=templates_input,
                sample_rate=sample_rate,
                feature_names=feat_names,
            )
            template_labels = [f"template_{i}" for i in range(len(template_events))]

        n_templates = len(template_events)

        # Resolve metric-specific gate thresholds
        req_gate = float(gate_th) if gate_th is not None else None
        if metric == "cosine":
            if req_gate is None:
                eff_gate = 0.25
            elif req_gate >= 1.0:
                eff_gate = 0.25  # Incompatible generic gate converted to standard cosine distance space
            else:
                eff_gate = req_gate
        elif metric == "manhattan":
            eff_gate = req_gate if req_gate is not None else 5.0
        else:  # euclidean / mahalanobis
            eff_gate = req_gate if req_gate is not None else 3.0

        if n_queries == 0 or n_templates == 0:
            return AssociationResult(
                events=query_events,
                matches=[],
                unassociated_queries=list(range(n_queries)),
                unassociated_templates=list(range(n_templates)),
                distance_matrix=np.empty((n_queries, n_templates), dtype=np.float64),
                gated_distance_matrix=np.empty((n_queries, n_templates), dtype=np.float64),
                assignment_method=method,
                gate_threshold=eff_gate,
                requested_gate_threshold=req_gate,
                effective_gate_threshold=eff_gate,
                metric=metric,
                sample_rate=sample_rate,
            )

        # 3. Template-Independent Normalization
        template_norm = normalize_features(template_raw, sample_rate=sample_rate, feature_names=feat_names)

        n_feat = len(feat_names)
        feature_eps = {
            "spectral_centroid_hz": max(2000.0, 0.01 * sample_rate),
            "carrier_freq_hz": max(2000.0, 0.01 * sample_rate),
            "bandwidth_hz": max(2000.0, 0.01 * sample_rate),
            "duration_sec": max(1e-6, 20.0 / sample_rate),
            "pulse_width_sec": max(1e-6, 20.0 / sample_rate),
            "mean_power_db": 5.0,
            "power_db": 5.0,
            "spectral_entropy": 0.15,
            "spectral_flatness": 0.15,
            "chirp_slope_hz_per_sec": max(1e7, 0.01 * (sample_rate ** 2) / 1000.0),
        }

        query_norm = np.zeros_like(query_raw, dtype=np.float64)
        for j in range(n_feat):
            t_col = template_raw[:, j]
            valid_t = t_col[np.isfinite(t_col)]
            name = feat_names[j] if j < len(feat_names) else f"feat_{j}"
            eps_val = feature_eps.get(name, 1e-6)
            if len(valid_t) > 0:
                med = float(np.median(valid_t))
                mad = float(np.median(np.abs(valid_t - med)))
                scale = 1.4826 * mad + eps_val
            else:
                med = 0.0
                scale = eps_val
            query_norm[:, j] = (query_raw[:, j] - med) / scale

        weights_arr = resolve_feature_weights(feat_names, custom_weights)
        sqrt_w = np.sqrt(np.maximum(weights_arr, 0.0))

        q_weighted = query_norm * sqrt_w[None, :]
        t_weighted = template_norm * sqrt_w[None, :]

        # 4. Pairwise Distance Computation
        if metric == "manhattan":
            diff = np.abs(q_weighted[:, None, :] - t_weighted[None, :, :])
            dist_mat = np.sum(diff, axis=-1)
        elif metric == "cosine":
            feature_scales = {
                "spectral_centroid_hz": max(2000.0, 0.01 * sample_rate),
                "carrier_freq_hz": max(2000.0, 0.01 * sample_rate),
                "bandwidth_hz": max(2000.0, 0.01 * sample_rate),
                "duration_sec": max(1e-6, 20.0 / sample_rate),
                "pulse_width_sec": max(1e-6, 20.0 / sample_rate),
                "mean_power_db": 10.0,
                "power_db": 10.0,
                "spectral_entropy": 0.5,
                "spectral_flatness": 0.5,
                "chirp_slope_hz_per_sec": max(1e7, 0.01 * (sample_rate ** 2) / 1000.0),
            }
            scales = np.array([feature_scales.get(name, 1.0) for name in feat_names], dtype=np.float64)
            q_sc = (query_raw / scales[None, :]) * sqrt_w[None, :]
            t_sc = (template_raw / scales[None, :]) * sqrt_w[None, :]
            q_norms = np.maximum(np.linalg.norm(q_sc, axis=1, keepdims=True), 1e-15)
            t_norms = np.maximum(np.linalg.norm(t_sc, axis=1, keepdims=True), 1e-15)
            q_normed = q_sc / q_norms
            t_normed = t_sc / t_norms
            cos_sim = np.clip(q_normed @ t_normed.T, -1.0, 1.0)
            dist_mat = 1.0 - cos_sim
        elif metric == "mahalanobis":
            # True regularized Mahalanobis distance with covariance inversion
            supplied_cov = kwargs.get("covariance", getattr(self, "covariance", None))
            supplied_inv_cov = kwargs.get("inv_covariance", getattr(self, "inv_covariance", None))

            if supplied_inv_cov is not None:
                inv_cov = np.asarray(supplied_inv_cov, dtype=np.float64)
            elif supplied_cov is not None:
                cov_mat = np.asarray(supplied_cov, dtype=np.float64)
                inv_cov = np.linalg.pinv(cov_mat + 1e-4 * np.eye(cov_mat.shape[0]))
            else:
                if n_templates > 1:
                    cov_mat = np.cov(t_weighted, rowvar=False)
                    if cov_mat.ndim == 0:
                        cov_mat = np.array([[cov_mat]])
                    reg_cov = cov_mat + 1e-3 * np.eye(n_feat)
                    inv_cov = np.linalg.pinv(reg_cov)
                else:
                    inv_cov = np.diag(np.maximum(weights_arr, 1e-3))

            dist_mat = np.zeros((n_queries, n_templates), dtype=np.float64)
            for i in range(n_queries):
                q_vec = q_weighted[i]
                for j in range(n_templates):
                    t_vec = t_weighted[j]
                    d_vec = q_vec - t_vec
                    quad = float(d_vec @ inv_cov @ d_vec)
                    dist_mat[i, j] = math.sqrt(max(quad, 0.0))
        else:  # euclidean
            diff = q_weighted[:, None, :] - t_weighted[None, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            dist_mat = np.sqrt(np.maximum(dist_sq, 0.0))

        # 5. Gating
        gated_dist = dist_mat.copy()
        gated_dist[dist_mat > eff_gate] = 1e6

        # 6. Matching
        assigned_pairs: list[tuple[int, int]] = []

        if method == "hungarian":
            row_ind, col_ind = hungarian_matching(gated_dist)
            for r, c in zip(row_ind, col_ind):
                if dist_mat[r, c] <= eff_gate:
                    assigned_pairs.append((int(r), int(c)))
        elif method == "greedy":
            flat_indices = np.argsort(dist_mat, axis=None)
            used_q = set()
            used_t = set()
            for idx in flat_indices:
                r = int(idx // n_templates)
                c = int(idx % n_templates)
                if dist_mat[r, c] > eff_gate:
                    break
                if r not in used_q and c not in used_t:
                    used_q.add(r)
                    used_t.add(c)
                    assigned_pairs.append((r, c))
        else:  # nearest
            for r in range(n_queries):
                best_c = int(np.argmin(dist_mat[r]))
                if dist_mat[r, best_c] <= eff_gate:
                    assigned_pairs.append((r, best_c))

        # 7. Construct Result Records
        matched_queries = {r: c for r, c in assigned_pairs}
        matches: list[dict[str, Any]] = []

        for q_i in range(n_queries):
            if q_i in matched_queries:
                t_i = matched_queries[q_i]
                d_val = float(dist_mat[q_i, t_i])
                if metric == "cosine":
                    sim = float(max(0.0, 1.0 - d_val))
                else:
                    sim = float(np.exp(-d_val / max(eff_gate, 1e-6)))
                t_lbl = template_labels[t_i]
                residuals = {name: float(query_raw[q_i, j] - template_raw[t_i, j]) for j, name in enumerate(feat_names)}
                matches.append({
                    "query_idx": q_i,
                    "template_idx": t_i,
                    "template_label": t_lbl,
                    "distance": d_val,
                    "similarity_score": sim,
                    "is_associated": True,
                    "feature_residuals": residuals,
                })
            else:
                matches.append({
                    "query_idx": q_i,
                    "template_idx": -1,
                    "template_label": "UNASSOCIATED",
                    "distance": float("inf"),
                    "similarity_score": 0.0,
                    "is_associated": False,
                    "feature_residuals": {},
                })

        unassociated_queries = [q_i for q_i in range(n_queries) if q_i not in matched_queries]
        matched_templates = {c for _, c in assigned_pairs}
        unassociated_templates = [t_i for t_i in range(n_templates) if t_i not in matched_templates]

        return AssociationResult(
            events=query_events,
            matches=matches,
            unassociated_queries=unassociated_queries,
            unassociated_templates=unassociated_templates,
            distance_matrix=dist_mat,
            gated_distance_matrix=gated_dist,
            assignment_method=method,
            gate_threshold=eff_gate,
            requested_gate_threshold=req_gate,
            effective_gate_threshold=eff_gate,
            metric=metric,
            sample_rate=sample_rate,
        )


def compute_nearest_neighbor_association(
    data: Any,
    templates: list[dict[str, Any]] | np.ndarray | None = None,
    sample_rate: float = 1.0,
    gate_threshold: float | None = None,
    metric: str = "euclidean",
    assignment_method: str = "hungarian",
    feature_weights: dict[str, float] | np.ndarray | None = None,
    **kwargs: Any,
) -> AssociationResult:
    """Convenience functional API for nearest-neighbor template matching."""
    processor = NearestNeighborAssociation(
        sample_rate=sample_rate,
        templates=templates,
        gate_threshold=gate_threshold,
        metric=metric,
        assignment_method=assignment_method,
        feature_weights=feature_weights,
        **kwargs,
    )
    return processor.compute(data, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  NearestNeighborAssociation: Demonstration")
    print("=" * 60)
    pulses = np.array([[100.0, 1.0], [100.2, 1.05], [200.0, 2.0], [200.1, 1.95]])
    metric = NearestNeighborAssociation()
    result = metric.compute(pulses)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
