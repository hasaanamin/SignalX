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
from .models import PRIPatternResult
from .validation import validate_sequence_inputs
from ._common import (
    estimate_discrete_states,
    build_transition_matrix,
    compute_dwell_lengths,
    detect_symbolic_periodicity,
    estimate_linear_trend,
    adjacent_difference_with_gaps,
)


@SequencePatternMetricFactory.register([
    "pri_patterns",
    "pri_pattern",
    "pri_pattern_analysis",
    "pripatterns",
])
class PRIPatterns(BaseSequencePatternMetric):
    """
    Pulse Repetition Interval (PRI) Pattern Analyzer.

    Analyzes sequencing, staggering, jitter, dwell-and-switch behavior, and periodicity across
    precomputed PRI sequences.

    This processor does NOT estimate PRI from raw IQ. The caller must provide precomputed PRI
    values (e.g., from Clustering and Association or Pulse Detector TOA differencing).
    """

    @property
    def name(self) -> str:
        return "pri_patterns"

    def compute(
        self,
        pri_sec: Any | None = None,
        pri_values_sec: Any | None = None,
        timestamps_sec: Any | None = None,
        sort_by_time: bool = False,
        tolerance: float = 0.03,
        **kwargs: Any,
    ) -> PRIPatternResult:
        """
        Analyzes pattern structure across an already-available PRI sequence or timestamps.

        Parameters
        ----------
        pri_sec : array-like, optional
            1D array of pre-extracted PRI values in seconds.
        pri_values_sec : array-like, optional
            Alias for pri_sec.
        timestamps_sec : array-like, optional
            Corresponding pulse time-of-arrival timestamps.
        sort_by_time : bool, default=False
            Whether to sort PRIs chronologically by timestamps before evaluation.
        tolerance : float, default=0.03
            Relative tolerance (e.g. 3%) for discrete PRI state clustering.

        Returns
        -------
        PRIPatternResult
        """
        if tolerance is not None:
            tol_val = float(tolerance)
            if tol_val < 0.0 or not math.isfinite(tol_val):
                raise ValueError(f"tolerance must be non-negative and finite, got {tolerance}")

        input_pris = pri_sec if pri_sec is not None else pri_values_sec
        derived_num_pulses = None

        if input_pris is None:
            if timestamps_sec is not None:
                ts = np.asarray(timestamps_sec, dtype=np.float64)
                derived_num_pulses = len(ts)
                if len(ts) >= 2:
                    input_pris = np.full(len(ts) - 1, np.nan, dtype=np.float64)
                    for i in range(len(ts) - 1):
                        if np.isfinite(ts[i]) and np.isfinite(ts[i + 1]):
                            diff_t = float(ts[i + 1] - ts[i])
                            if diff_t > 0:
                                input_pris[i] = diff_t
                else:
                    input_pris = np.empty(0, dtype=np.float64)
            else:
                input_pris = np.empty(0, dtype=np.float64)
            ts_to_validate = None
        else:
            k = len(input_pris)
            if timestamps_sec is not None:
                m = len(timestamps_sec)
                if m == k + 1:
                    derived_num_pulses = m
                    ts_to_validate = np.asarray(timestamps_sec)[:k]
                elif m == k:
                    derived_num_pulses = k + 1 if k > 0 else 0
                    ts_to_validate = np.asarray(timestamps_sec)
                else:
                    raise ValueError(
                        f"timestamps_sec length ({m}) must either match PRI count ({k}) or pulse count ({k + 1})."
                    )
            else:
                ts_to_validate = None

        val_arr, ts_arr, valid_mask, orig_indices = validate_sequence_inputs(
            values=input_pris,
            timestamps_sec=ts_to_validate,
            sort_by_time=sort_by_time,
            allow_empty=True,
        )

        n_pri = len(val_arr)
        num_pulses = derived_num_pulses if derived_num_pulses is not None else (n_pri + 1 if n_pri > 0 else 0)
        pos_valid_mask = valid_mask & (val_arr > 0)
        valid_pris = val_arr[pos_valid_mask]  # PRIs must be positive
        num_valid_pri = len(valid_pris)

        if len(valid_pris) < 2:
            return PRIPatternResult(
                pattern_type="unresolved",
                num_pulses=num_pulses,
                pri_states_sec=[],
                state_sequence=np.full(n_pri, -1, dtype=int),
                transition_matrix=np.empty((0, 0), dtype=np.float64),
                dwell_lengths=[],
                trend_slope_sec_per_pulse=0.0,
                mean_pri_sec=float(np.mean(valid_pris)) if len(valid_pris) > 0 else 0.0,
                std_pri_sec=0.0,
                jitter_percent=0.0,
                repetition_length=None,
                stagger_levels=0,
                stability_score=0.0,
                confidence=0.0,
                valid=False,
                num_pri_samples=n_pri,
                num_valid_pri_samples=num_valid_pri,
                diagnostics={
                    "reason": "Insufficient valid positive PRI samples (< 2).",
                    "num_valid_pri_samples": num_valid_pri,
                    "num_pri_samples": n_pri,
                    "num_pulses": num_pulses,
                },
            )

        mean_pri = float(np.mean(valid_pris))
        std_pri = float(np.std(valid_pris))
        jitter_pct = float((std_pri / max(mean_pri, 1e-15)) * 100.0)

        # Trend estimation
        trend_slope = estimate_linear_trend(
            valid_pris,
            ts_arr[pos_valid_mask] if (ts_arr is not None and len(ts_arr) == n_pri) else None
        )

        # Estimate discrete PRI states on positive valid values
        pri_states, valid_state_seq = estimate_discrete_states(valid_pris, tolerance=tolerance, relative_tol=True)
        num_states = len(pri_states)

        # Full sequence state mapping with -1 sentinels for missing/non-positive elements
        full_state_seq = np.full(n_pri, -1, dtype=int)
        if len(valid_state_seq) > 0 and np.any(pos_valid_mask):
            full_state_seq[pos_valid_mask] = valid_state_seq

        # Build transition matrix & dwell run lengths with -1 breaking adjacency
        trans_mat = build_transition_matrix(full_state_seq, num_states, missing_value=-1)
        dwells = compute_dwell_lengths(full_state_seq, missing_value=-1)

        # Periodicity / repetition length detection on state sequence with gaps preserved
        is_periodic, rep_len, motif, fidelity, harmonics = detect_symbolic_periodicity(full_state_seq, missing_value=-1)

        # Monotonic linear change across adjacent pulses without bridging gaps
        adjacent_diffs, _, _ = adjacent_difference_with_gaps(val_arr, pos_valid_mask)
        is_monotonic_slope = (
            len(adjacent_diffs) > 5
            and (np.all(adjacent_diffs >= -1e-12) or np.all(adjacent_diffs <= 1e-12))
            and abs(trend_slope) > (0.001 * mean_pri)
        )

        # Classify PRI Pattern Type
        pattern_type = "unresolved"
        confidence = 0.5
        mean_dwell = float(np.mean(dwells)) if dwells else 1.0

        if jitter_pct < 2.0 and num_states <= 1:
            pattern_type = "constant"
            stability_score = 1.0 - (jitter_pct / 2.0) * 0.2
            confidence = float(np.clip(0.95 - (jitter_pct / 50.0), 0.7, 1.0))
        elif is_monotonic_slope or (abs(trend_slope) > (0.01 * mean_pri) and jitter_pct > 5.0 and num_states >= 3 and mean_dwell < 2.0):
            pattern_type = "sliding"
            stability_score = 0.65
            confidence = 0.75
        elif is_periodic and rep_len is not None and rep_len >= 2 and num_states >= 2:
            if mean_dwell < 1.8:
                pattern_type = "staggered"
                confidence = float(np.clip(fidelity * 0.95, 0.6, 1.0))
            else:
                pattern_type = "dwell_and_switch"
                confidence = float(np.clip(fidelity * 0.90, 0.6, 1.0))
            stability_score = float(fidelity)
        elif num_states >= 2 and len(dwells) >= 2 and mean_dwell >= 2.0:
            pattern_type = "dwell_and_switch"
            stability_score = 0.75
            confidence = 0.80
        elif jitter_pct <= 30.0:
            pattern_type = "jittered"
            stability_score = float(np.clip(1.0 - (jitter_pct / 50.0), 0.3, 0.8))
            confidence = float(np.clip(0.85 - (jitter_pct / 100.0), 0.5, 0.9))
        else:
            pattern_type = "agile"
            stability_score = 0.2
            confidence = 0.70

        return PRIPatternResult(
            pattern_type=pattern_type,
            num_pulses=num_pulses,
            pri_states_sec=pri_states,
            state_sequence=full_state_seq,
            transition_matrix=trans_mat,
            dwell_lengths=dwells,
            trend_slope_sec_per_pulse=trend_slope,
            mean_pri_sec=mean_pri,
            std_pri_sec=std_pri,
            jitter_percent=jitter_pct,
            repetition_length=rep_len if is_periodic else None,
            stagger_levels=num_states if pattern_type == "staggered" else (1 if pattern_type == "constant" else num_states),
            stability_score=stability_score,
            confidence=confidence,
            valid=True,
            num_pri_samples=n_pri,
            num_valid_pri_samples=num_valid_pri,
            diagnostics={
                "is_periodic": is_periodic,
                "periodic_fidelity": fidelity,
                "repeating_motif": motif,
                "harmonic_periods": harmonics,
                "num_valid_pri_samples": num_valid_pri,
                "num_pri_samples": n_pri,
                "num_pulses": num_pulses,
            },
        )


def analyze_pri_pattern(
    pri_sec: Any | None = None,
    pri_values_sec: Any | None = None,
    timestamps_sec: Any | None = None,
    sort_by_time: bool = False,
    tolerance: float = 0.03,
    **kwargs: Any,
) -> PRIPatternResult:
    """
    Convenience function to analyze PRI patterns.

    This function does not estimate PRI from raw IQ.
    The caller must provide precomputed PRI values from upstream packages.
    """
    processor = PRIPatterns(**kwargs)
    return processor.compute(
        pri_sec=pri_sec,
        pri_values_sec=pri_values_sec,
        timestamps_sec=timestamps_sec,
        sort_by_time=sort_by_time,
        tolerance=tolerance,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  PRIPatterns: Demonstration")
    print("=" * 60)
    pri_sequence = np.array([100e-6, 100e-6, 100e-6, 100e-6, 100e-6])
    metric = PRIPatterns()
    result = metric.compute(pri_sequence)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
