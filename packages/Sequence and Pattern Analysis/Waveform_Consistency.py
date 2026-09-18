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
from typing import Any, Optional
import numpy as np

from .base import BaseSequencePatternMetric, SequencePatternMetricFactory
from .models import WaveformConsistencyResult
from .validation import validate_sequence_inputs, validate_matrix_input


@SequencePatternMetricFactory.register([
    "waveform_consistency",
    "waveform_similarity_consistency",
    "similarity_consistency",
    "waveformconsistency",
])
class WaveformConsistency(BaseSequencePatternMetric):
    """
    Temporal Waveform Consistency Analyzer.

    Analyzes long-term stability, epoch segmentation, and degradation trends across precomputed
    waveform similarity metrics.

    This processor does NOT compute cross-correlation or waveform alignment from raw IQ.
    The caller must provide precomputed similarity scores or similarity matrices from the
    Correlation and Similarity package.
    """

    @property
    def name(self) -> str:
        return "waveform_consistency"

    def compute(
        self,
        similarity_scores: Any | None = None,
        similarity_matrix: Any | None = None,
        correlation_matrix: Any | None = None,
        timestamps_sec: Any | None = None,
        change_threshold: float = 0.20,
        **kwargs: Any,
    ) -> WaveformConsistencyResult:
        """
        Analyzes waveform consistency across precomputed similarity scores or matrices.

        Parameters
        ----------
        similarity_scores : array-like, optional
            1D array of consecutive pulse waveform similarity scores in [0, 1].
        similarity_matrix : array-like, optional
            N x N pairwise pulse similarity matrix.
        correlation_matrix : array-like, optional
            Alias for similarity_matrix.
        timestamps_sec : array-like, optional
            Corresponding pulse time-of-arrival timestamps.
        change_threshold : float, default=0.20
            Drop in similarity considered a significant change point.

        Returns
        -------
        WaveformConsistencyResult
        """
        if change_threshold is None or not (0.0 <= float(change_threshold) <= 1.0) or not math.isfinite(float(change_threshold)):
            raise ValueError(f"change_threshold must be a finite float in [0, 1], got {change_threshold}")

        matrix_input = similarity_matrix if similarity_matrix is not None else correlation_matrix

        if similarity_scores is None and matrix_input is None:
            raise ValueError("Either similarity_scores or similarity_matrix must be provided.")

        if matrix_input is not None:
            mat, valid_mask_mat = validate_matrix_input(matrix_input, min_size=1, allow_empty=True)
            n = mat.shape[0]
            is_valid = bool(n > 0 and np.all(valid_mask_mat))
            if not is_valid or n == 0:
                return WaveformConsistencyResult(
                    num_pulses=0,
                    mean_similarity=0.0,
                    std_similarity=0.0,
                    min_similarity=0.0,
                    consistency_score=0.0,
                    temporal_stability=0.0,
                    change_point_indices=[],
                    epoch_labels=np.empty(0, dtype=int),
                    num_epochs=0,
                    confidence=0.0,
                    valid=False,
                    num_similarity_pairs=0,
                    diagnostics={"reason": "Input similarity matrix is empty or invalid.", "missing_pairs_count": 0},
                )
            if n == 1:
                return WaveformConsistencyResult(
                    num_pulses=1,
                    mean_similarity=1.0,
                    std_similarity=0.0,
                    min_similarity=1.0,
                    consistency_score=1.0,
                    temporal_stability=1.0,
                    change_point_indices=[],
                    epoch_labels=np.zeros(1, dtype=int),
                    num_epochs=1,
                    confidence=1.0,
                    valid=True,
                    num_similarity_pairs=0,
                    diagnostics={"missing_pairs_count": 0},
                )
            # Extract adjacent similarities from subdiagonal mat[i, i+1]
            scores_full = np.array([mat[i, i + 1] for i in range(n - 1)], dtype=np.float64)
            valid_mask = np.isfinite(scores_full)
            num_pulses = n
            num_pairs = len(scores_full)
        else:
            val_arr, ts_arr, valid_mask, orig_indices = validate_sequence_inputs(
                values=similarity_scores,
                timestamps_sec=timestamps_sec,
                allow_empty=True,
            )
            scores_full = val_arr
            num_pairs = len(scores_full)
            num_pulses = len(scores_full) + 1 if len(scores_full) > 0 else 0

        finite_scores = scores_full[valid_mask] if len(scores_full) > 0 else np.empty(0, dtype=np.float64)
        missing_pairs_count = int(np.sum(~valid_mask)) if len(scores_full) > 0 else 0

        if len(finite_scores) == 0:
            return WaveformConsistencyResult(
                num_pulses=num_pulses,
                mean_similarity=0.0,
                std_similarity=0.0,
                min_similarity=0.0,
                consistency_score=0.0,
                temporal_stability=0.0,
                change_point_indices=[],
                epoch_labels=np.full(num_pairs, -1, dtype=int) if num_pairs > 0 else np.empty(0, dtype=int),
                num_epochs=0,
                confidence=0.0,
                valid=False,
                num_similarity_pairs=num_pairs,
                diagnostics={"reason": "Insufficient valid similarity scores.", "missing_pairs_count": missing_pairs_count},
            )

        # Validate similarity score values in [0, 1]
        for s in finite_scores:
            s_val = float(s)
            if s_val < -1e-6 or s_val > (1.0 + 1e-6):
                raise ValueError(f"similarity scores must be in range [0, 1], got {s_val}")

        mean_sim = float(np.mean(finite_scores))
        std_sim = float(np.std(finite_scores))
        min_sim = float(np.min(finite_scores))

        # Detect change points where adjacent similarity transitions significantly (|score[i] - score[i-1]| >= threshold)
        change_points = []
        for i in range(1, num_pairs):
            if valid_mask[i - 1] and valid_mask[i]:
                if abs(scores_full[i] - scores_full[i - 1]) >= change_threshold:
                    change_points.append(i)

        # Segment into consistent epochs preserving index positions
        epoch_labels = np.full(num_pairs, -1, dtype=int)
        curr_epoch = 0
        in_epoch = False

        for i in range(num_pairs):
            if not valid_mask[i]:
                in_epoch = False
            else:
                if not in_epoch:
                    if i > 0 and not in_epoch and np.any(valid_mask[:i]):
                        curr_epoch += 1
                    in_epoch = True
                elif i in change_points:
                    curr_epoch += 1
                epoch_labels[i] = curr_epoch

        num_epochs = len(np.unique(epoch_labels[epoch_labels >= 0])) if np.any(valid_mask) else 0

        # Temporal stability score
        # 1.0 = consistently identical, < 0.5 = frequent degradations/changes
        stability = float(np.clip(mean_sim * (1.0 - min(1.0, std_sim * 2.0)), 0.0, 1.0))
        consistency_score = float(np.clip(mean_sim, 0.0, 1.0))
        confidence = float(np.clip(min(1.0, len(finite_scores) / 5.0) * (1.0 - 0.5 * std_sim), 0.0, 1.0))

        return WaveformConsistencyResult(
            num_pulses=num_pulses,
            mean_similarity=mean_sim,
            std_similarity=std_sim,
            min_similarity=min_sim,
            consistency_score=consistency_score,
            temporal_stability=stability,
            change_point_indices=change_points,
            epoch_labels=epoch_labels,
            num_epochs=num_epochs,
            confidence=confidence,
            valid=True,
            num_similarity_pairs=num_pairs,
            diagnostics={
                "num_pairs_evaluated": len(finite_scores),
                "num_change_points": len(change_points),
                "missing_pairs_count": missing_pairs_count,
            },
        )


def analyze_waveform_consistency(
    similarity_scores: Any | None = None,
    similarity_matrix: Any | None = None,
    correlation_matrix: Any | None = None,
    timestamps_sec: Any | None = None,
    change_threshold: float = 0.20,
    **kwargs: Any,
) -> WaveformConsistencyResult:
    """
    Convenience function to analyze waveform consistency over time.

    This function does not compute waveform cross-correlation.
    The caller must provide precomputed similarity scores from the Correlation and Similarity package.
    """
    processor = WaveformConsistency(**kwargs)
    return processor.compute(
        similarity_scores=similarity_scores,
        similarity_matrix=similarity_matrix,
        correlation_matrix=correlation_matrix,
        timestamps_sec=timestamps_sec,
        change_threshold=change_threshold,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  WaveformConsistency: Demonstration")
    print("=" * 60)
    sim_mat = np.eye(5) + 0.05 * np.ones((5, 5))
    metric = WaveformConsistency()
    result = metric.compute(sim_mat)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
