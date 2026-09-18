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

from .base import BaseCorrelationMetric, CorrelationMetricFactory, SequenceSimilarityResult
from ._common import _compute_dtw_core


@CorrelationMetricFactory.register(["sequence_similarity", "dtw", "pulse_sequence_similarity", "dynamic_time_warping"])
class SequenceSimilarityMetric(BaseCorrelationMetric):
    """
    Pulse Sequence Similarity and Dynamic Time Warping (DTW) Stage matching MATLAB `dtw`.

    Mathematical Formulation:
      Operates on multidimensional pulse-level feature sequences:
        p_i = [PW_i, PRI_i, A_i, f_{c, i}, BW_i, ...]

      Per-dimension feature standardization:
        z_q = (q - mu_q) / (sigma_q + eps)

      Local Euclidean Distance:
        d_mn = sqrt(sum_k (x_{k, m} - y_{k, n})^* (x_{k, m} - y_{k, n}))

      Global Warping Distance:
        D_DTW = min_P sum_{(m, n) in P} d_mn
        via DP: D[i, j] = d_ij + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

      Normalized Distance & Similarity Score:
        D_norm = D_DTW / len(P)
        S_DTW = 1 / (1 + D_norm)  in (0, 1]
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "sequence_similarity"

    def compute(
        self,
        seq_a: Any,
        seq_b: Any,
        *,
        metric: str = "euclidean",
        normalize_features: bool = True,
        as_dict: bool = False,
        **kwargs: Any,
    ) -> SequenceSimilarityResult | dict[str, Any] | float:
        """
        Computes Dynamic Time Warping (DTW) distance and alignment between two pulse sequences.

        Parameters:
            seq_a: First feature sequence array [M, K] or list of feature vectors.
            seq_b: Second feature sequence array [N, K] or list of feature vectors.
            metric: Distance metric ('euclidean', 'squared', 'absolute').
            normalize_features: If True (default), applies Z-score standardization across feature dimensions.
            as_dict: If True, returns canonical dictionary. If False, returns SequenceSimilarityResult.

        Returns:
            SequenceSimilarityResult (or dict if as_dict=True).
        """
        res = _compute_dtw_core(
            seq_a,
            seq_b,
            metric=metric,
            normalize_features=normalize_features,
        )
        return res.to_dict() if as_dict else res


def compute_sequence_similarity(
    seq_a: Any,
    seq_b: Any,
    *,
    metric: str = "euclidean",
    normalize_features: bool = True,
    as_dict: bool = False,
    **kwargs: Any,
) -> SequenceSimilarityResult | dict[str, Any]:
    """
    Convenience functional API for pulse sequence similarity and DTW alignment.
    """
    stage = SequenceSimilarityMetric()
    return stage.compute(
        seq_a,
        seq_b,
        metric=metric,
        normalize_features=normalize_features,
        as_dict=as_dict,
        **kwargs,
    )


def dtw(
    x: Any,
    y: Any,
    *,
    metric: str = "euclidean",
    normalize_features: bool = True,
    **kwargs: Any,
) -> float:
    """
    Exact MATLAB `dtw(x, y)` wrapper returning the scalar DTW distance.
    """
    res = compute_sequence_similarity(
        x,
        y,
        metric=metric,
        normalize_features=normalize_features,
        as_dict=False,
        **kwargs,
    )
    return float(res.dtw_distance)


def main() -> None:
    print("=" * 60)
    print("  SequenceSimilarityMetric: Demonstration")
    print("=" * 60)
    s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s2 = np.array([1.05, 1.95, 3.02, 4.01, 4.98])
    metric = SequenceSimilarityMetric()
    result = metric.compute(s1, s2)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
