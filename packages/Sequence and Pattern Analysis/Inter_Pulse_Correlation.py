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

import math
from typing import Any, Optional, Union
import numpy as np

from .models import InterPulseCorrelationPatternResult
from .base import BaseSequencePatternMetric, SequencePatternMetricFactory
from .validation import validate_matrix_input


@SequencePatternMetricFactory.register([
    "inter_pulse_correlation",
    "inter_pulse_correlation_pattern",
    "interpulsecorrelation",
])
class InterPulseCorrelation(BaseSequencePatternMetric):
    """
    Analyzes temporal patterns, coherence decay, lag decorrelation, and pulse family clustering
    from an already-computed N x N inter-pulse correlation or similarity matrix.
    """

    @property
    def name(self) -> str:
        return "inter_pulse_correlation"

    def compute(
        self,
        correlation_matrix: np.ndarray,
        decorrelation_threshold: float = 0.368,
        family_threshold: float = 0.8,
        **kwargs: Any,
    ) -> InterPulseCorrelationPatternResult:
        """
        Analyzes structure and temporal evolution of an inter-pulse correlation matrix.

        Parameters
        ----------
        correlation_matrix : np.ndarray
            N x N precomputed pairwise inter-pulse correlation matrix.
        decorrelation_threshold : float, default=0.368
            Correlation value threshold (1/e) for determining decorrelation length.
        family_threshold : float, default=0.8
            Pairwise correlation threshold for clustering pulses into the same family.

        Returns
        -------
        InterPulseCorrelationPatternResult
        """
        if decorrelation_threshold is None or not (0.0 <= float(decorrelation_threshold) <= 1.0) or not math.isfinite(float(decorrelation_threshold)):
            raise ValueError(f"decorrelation_threshold must be a finite float in [0, 1], got {decorrelation_threshold}")
        if family_threshold is None or not (0.0 <= float(family_threshold) <= 1.0) or not math.isfinite(float(family_threshold)):
            raise ValueError(f"family_threshold must be a finite float in [0, 1], got {family_threshold}")

        if correlation_matrix is None:
            mat = np.empty((0, 0), dtype=np.float64)
            is_finite_mask = np.empty((0, 0), dtype=bool)
        else:
            raw_mat = np.asarray(correlation_matrix)
            if raw_mat.ndim != 2 or (raw_mat.ndim == 2 and raw_mat.shape[0] != raw_mat.shape[1]):
                return InterPulseCorrelationPatternResult(
                    pattern_type="unresolved",
                    num_pulses=raw_mat.shape[0] if raw_mat.ndim >= 1 else 0,
                    mean_inter_pulse_correlation=0.0,
                    min_inter_pulse_correlation=0.0,
                    decorrelation_length=None,
                    correlation_trend=0.0,
                    family_clusters=np.empty(0, dtype=int),
                    num_families=0,
                    transition_count=0,
                    stability_score=0.0,
                    confidence=0.0,
                    valid=False,
                    diagnostics={"reason": "matrix_must_be_square", "shape": list(raw_mat.shape)},
                )
            if not np.all(np.isfinite(raw_mat)):
                return InterPulseCorrelationPatternResult(
                    pattern_type="unresolved",
                    num_pulses=raw_mat.shape[0],
                    mean_inter_pulse_correlation=0.0,
                    min_inter_pulse_correlation=0.0,
                    decorrelation_length=None,
                    correlation_trend=0.0,
                    family_clusters=np.empty(0, dtype=int),
                    num_families=0,
                    transition_count=0,
                    stability_score=0.0,
                    confidence=0.0,
                    valid=False,
                    diagnostics={"reason": "non_finite_elements"},
                )
            mat, is_finite_mask = validate_matrix_input(correlation_matrix, min_size=2, allow_empty=True)

        n = mat.shape[0]

        if n < 2:
            return InterPulseCorrelationPatternResult(
                pattern_type="unresolved",
                num_pulses=n,
                mean_inter_pulse_correlation=0.0,
                min_inter_pulse_correlation=0.0,
                decorrelation_length=None,
                correlation_trend=0.0,
                family_clusters=np.empty(0, dtype=int),
                num_families=0,
                transition_count=0,
                stability_score=0.0,
                confidence=0.0,
                valid=False,
                diagnostics={"reason": "invalid_or_insufficient_matrix", "dim": n},
            )

        # Off-diagonal elements
        mask_off_diag = ~np.eye(n, dtype=bool)
        off_diag_vals = mat[mask_off_diag]
        mean_corr = float(np.mean(off_diag_vals))
        min_corr = float(np.min(off_diag_vals))

        # Lag analysis: average correlation at diagonal offset k = 1 .. n-1
        lag_corrs = []
        for k in range(1, n):
            diag_k = np.diag(mat, k=k)
            lag_corrs.append(float(np.mean(diag_k)))

        # Decorrelation length (first lag where lag_corr drops below threshold)
        decorr_len = None
        for k, c_val in enumerate(lag_corrs, start=1):
            if c_val < decorrelation_threshold:
                decorr_len = k
                break

        # Correlation trend with lag
        if len(lag_corrs) >= 2:
            lags = np.arange(1, len(lag_corrs) + 1, dtype=np.float64)
            lag_mean = float(np.mean(lags))
            c_mean = float(np.mean(lag_corrs))
            denom = float(np.sum((lags - lag_mean) ** 2))
            trend = float(np.sum((lags - lag_mean) * (np.array(lag_corrs) - c_mean)) / denom) if denom > 1e-12 else 0.0
        else:
            trend = 0.0

        # Family clustering via threshold graph connected components
        family_labels = np.full(n, -1, dtype=int)
        curr_family = 0
        visited = np.zeros(n, dtype=bool)

        for i in range(n):
            if not visited[i]:
                queue = [i]
                visited[i] = True
                family_labels[i] = curr_family
                while queue:
                    curr_node = queue.pop(0)
                    neighbors = np.where((mat[curr_node] >= family_threshold) & (~visited))[0]
                    for neighbor in neighbors:
                        visited[neighbor] = True
                        family_labels[neighbor] = curr_family
                        queue.append(neighbor)
                curr_family += 1

        num_families = curr_family

        # Count family transitions along pulse train
        transition_count = int(np.sum(family_labels[:-1] != family_labels[1:]))

        # Classify pattern type
        if mean_corr < 0.15 and abs(trend) < 0.01:
            pattern_type = "decorrelated_noise"
        elif num_families == 2 and transition_count >= n // 2:
            pattern_type = "alternating_bimodal"
        elif num_families > 1 and num_families <= 8:
            if transition_count <= num_families + 1:
                pattern_type = "abrupt_step"
            else:
                pattern_type = "family_grouped"
        elif mean_corr >= 0.85 and min_corr >= 0.7:
            pattern_type = "stable_high"
        elif trend < -0.01 or (decorr_len is not None and decorr_len < n):
            pattern_type = "gradual_decorrelation"
        elif mean_corr < 0.25:
            pattern_type = "decorrelated_noise"
        else:
            pattern_type = "unresolved"

        # Stability score
        stability_score = max(0.0, min(1.0, mean_corr * (1.0 - min(1.0, transition_count / max(1, n - 1)))))

        confidence = min(1.0, max(0.2, float(n) / 10.0))

        diagnostics = {
            "lag_correlations": lag_corrs,
            "decorrelation_length": decorr_len,
            "trend_per_lag": trend,
            "num_families": num_families,
            "transition_count": transition_count,
        }

        return InterPulseCorrelationPatternResult(
            pattern_type=pattern_type,
            num_pulses=n,
            mean_inter_pulse_correlation=mean_corr,
            min_inter_pulse_correlation=min_corr,
            decorrelation_length=decorr_len,
            correlation_trend=trend,
            family_clusters=family_labels,
            num_families=num_families,
            transition_count=transition_count,
            stability_score=stability_score,
            confidence=confidence,
            valid=True,
            diagnostics=diagnostics,
        )


def analyze_inter_pulse_correlation_pattern(
    correlation_matrix: np.ndarray,
    decorrelation_threshold: float = 0.368,
    family_threshold: float = 0.8,
    **kwargs: Any,
) -> InterPulseCorrelationPatternResult:
    """
    Functional wrapper to analyze inter-pulse correlation matrix pattern.
    """
    metric = InterPulseCorrelation()
    return metric.compute(
        correlation_matrix=correlation_matrix,
        decorrelation_threshold=decorrelation_threshold,
        family_threshold=family_threshold,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  InterPulseCorrelation: Demonstration")
    print("=" * 60)
    sim_mat = np.eye(5) + 0.05 * np.ones((5, 5))
    metric = InterPulseCorrelation()
    result = metric.compute(sim_mat)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
