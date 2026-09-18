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

from .models import BurstStructureResult
from .base import BaseSequencePatternMetric, SequencePatternMetricFactory
from .validation import validate_sequence_inputs
from ._common import estimate_discrete_states, detect_symbolic_periodicity


@SequencePatternMetricFactory.register([
    "burst_structure",
    "burst_patterns",
    "burststructure",
])
class BurstStructure(BaseSequencePatternMetric):
    """
    Analyzes higher-level temporal patterns across burst sequences, including pulses-per-burst
    stability, burst duration uniformity, and Inter-Burst Interval (IBI) regularity/jitter/stagger.
    """

    @property
    def name(self) -> str:
        return "burst_structure"

    def compute(
        self,
        burst_labels: Optional[np.ndarray] = None,
        timestamps_sec: Optional[np.ndarray] = None,
        burst_start_times_sec: Optional[np.ndarray] = None,
        burst_durations_sec: Optional[np.ndarray] = None,
        burst_pulse_counts: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> BurstStructureResult:
        """
        Analyzes burst structure from precomputed burst labels or extracted burst parameters.

        Parameters
        ----------
        burst_labels : np.ndarray, optional
            Array of burst IDs for each pulse (e.g., from upstream Clustering_and_Association.Burst_Grouping).
        timestamps_sec : np.ndarray, optional
            Array of pulse Time of Arrival (TOA) timestamps in seconds.
        burst_start_times_sec : np.ndarray, optional
            Pre-aggregated burst start times in seconds.
        burst_durations_sec : np.ndarray, optional
            Pre-aggregated burst durations in seconds.
        burst_pulse_counts : np.ndarray, optional
            Pre-aggregated pulse counts per burst.

        Returns
        -------
        BurstStructureResult
        """
        starts = []
        durations = []
        counts = []

        if burst_start_times_sec is not None and len(burst_start_times_sec) > 0:
            n_starts = len(burst_start_times_sec)
            if burst_durations_sec is not None and len(burst_durations_sec) != n_starts:
                raise ValueError(f"burst_durations_sec length ({len(burst_durations_sec)}) must match burst_start_times_sec length ({n_starts}).")
            if burst_pulse_counts is not None and len(burst_pulse_counts) != n_starts:
                raise ValueError(f"burst_pulse_counts length ({len(burst_pulse_counts)}) must match burst_start_times_sec length ({n_starts}).")

            # Collect finite elements preserving synchronized indices
            valid_indices = []
            for i, t in enumerate(burst_start_times_sec):
                if t is not None and np.isfinite(t):
                    valid_indices.append(i)

            if valid_indices:
                sorted_valid = sorted(valid_indices, key=lambda idx: burst_start_times_sec[idx])
                starts = [float(burst_start_times_sec[i]) for i in sorted_valid]
                if burst_durations_sec is not None:
                    durations = [float(burst_durations_sec[i]) if (burst_durations_sec[i] is not None and np.isfinite(burst_durations_sec[i])) else None for i in sorted_valid]
                else:
                    durations = []
                if burst_pulse_counts is not None:
                    counts = [int(burst_pulse_counts[i]) if (burst_pulse_counts[i] is not None and np.isfinite(burst_pulse_counts[i])) else None for i in sorted_valid]
                else:
                    counts = []
        elif burst_labels is not None:
            b_arr, t_arr, valid_mask, _ = validate_sequence_inputs(burst_labels, timestamps_sec, allow_empty=True)
            valid_b = b_arr[valid_mask]
            valid_t = t_arr[valid_mask] if len(t_arr) == len(b_arr) else np.arange(len(valid_b), dtype=np.float64)

            # Filter out unassigned pulses (e.g. -1 noise labels)
            assigned_mask = valid_b >= 0
            if np.any(assigned_mask):
                ab_labels = valid_b[assigned_mask]
                ab_times = valid_t[assigned_mask]
                
                unique_bursts = np.unique(ab_labels)
                for b_id in unique_bursts:
                    b_t = ab_times[ab_labels == b_id]
                    if len(b_t) > 0:
                        starts.append(float(np.min(b_t)))
                        durations.append(float(np.max(b_t) - np.min(b_t)))
                        counts.append(int(len(b_t)))

                # Sort by start time
                sort_idx = np.argsort(starts)
                starts = [starts[i] for i in sort_idx]
                durations = [durations[i] for i in sort_idx]
                counts = [counts[i] for i in sort_idx]

        num_bursts = len(starts)
        finite_counts = [c for c in counts if c is not None and np.isfinite(c)]
        finite_durations = [d for d in durations if d is not None and np.isfinite(d)]
        missing_duration_count = sum(1 for d in durations if d is None) if (burst_durations_sec is not None and len(durations) > 0) else 0
        missing_pulse_count_count = sum(1 for c in counts if c is None) if (burst_pulse_counts is not None and len(counts) > 0) else 0

        total_pulses = int(sum(finite_counts)) if finite_counts else 0

        if num_bursts < 2:
            return BurstStructureResult(
                pattern_type="unresolved",
                num_bursts=num_bursts,
                total_pulses=total_pulses,
                mean_pulses_per_burst=float(finite_counts[0]) if finite_counts else 0.0,
                std_pulses_per_burst=0.0,
                min_pulses_per_burst=int(finite_counts[0]) if finite_counts else 0,
                max_pulses_per_burst=int(finite_counts[0]) if finite_counts else 0,
                mean_burst_duration_sec=float(finite_durations[0]) if finite_durations else 0.0,
                std_burst_duration_sec=0.0,
                mean_ibi_sec=0.0,
                std_ibi_sec=0.0,
                ibi_jitter_percent=0.0,
                burst_repetition_rate_hz=0.0,
                burst_stability_score=0.0,
                confidence=0.0,
                valid=False,
                diagnostics={
                    "reason": "insufficient_bursts",
                    "num_bursts": num_bursts,
                    "missing_duration_count": missing_duration_count,
                    "missing_pulse_count_count": missing_pulse_count_count,
                },
            )

        # Compute IBIs (Inter-Burst Intervals) between start times of consecutive bursts
        ibis = np.diff(np.array(starts, dtype=np.float64))
        valid_ibis = ibis[np.isfinite(ibis) & (ibis > 0)]

        if len(valid_ibis) == 0:
            mean_ibi = 0.0
            std_ibi = 0.0
            jitter_pct = 0.0
            brr_hz = 0.0
        else:
            mean_ibi = float(np.mean(valid_ibis))
            std_ibi = float(np.std(valid_ibis))
            jitter_pct = float((std_ibi / mean_ibi) * 100.0) if mean_ibi > 0 else 0.0
            brr_hz = float(1.0 / mean_ibi) if mean_ibi > 0 else 0.0

        mean_counts = float(np.mean(finite_counts)) if finite_counts else 0.0
        std_counts = float(np.std(finite_counts)) if finite_counts else 0.0
        min_counts = int(np.min(finite_counts)) if finite_counts else 0
        max_counts = int(np.max(finite_counts)) if finite_counts else 0

        mean_dur = float(np.mean(finite_durations)) if finite_durations else 0.0
        std_dur = float(np.std(finite_durations)) if finite_durations else 0.0

        # Pattern classification
        if len(valid_ibis) >= 2:
            is_p, p_len, motif, fidelity, harmonics = detect_symbolic_periodicity(valid_ibis, tolerance=mean_ibi * 0.02)
            ibi_states, _ = estimate_discrete_states(valid_ibis, relative_tol=True)

            if jitter_pct < 1.0:
                pattern_type = "regular"
            elif is_p and len(ibi_states) >= 2 and p_len is not None and p_len > 1:
                pattern_type = "staggered"
            elif jitter_pct < 20.0 and len(ibi_states) > 3:
                pattern_type = "jittered"
            else:
                pattern_type = "irregular"
        else:
            pattern_type = "regular" if jitter_pct < 5.0 else "irregular"

        # Stability score based on count uniformity and IBI regularity
        count_stability = max(0.0, 1.0 - (std_counts / (mean_counts + 1e-9))) if finite_counts else 0.0
        ibi_stability = max(0.0, 1.0 - (jitter_pct / 100.0))
        burst_stability = (0.5 * count_stability + 0.5 * ibi_stability) if finite_counts else ibi_stability

        confidence = min(1.0, max(0.2, float(num_bursts) / 5.0))

        diagnostics = {
            "ibis_sec": [float(x) for x in valid_ibis],
            "burst_durations_sec": durations,
            "burst_pulse_counts": counts,
            "count_stability": count_stability,
            "ibi_stability": ibi_stability,
            "missing_duration_count": missing_duration_count,
            "missing_pulse_count_count": missing_pulse_count_count,
        }

        return BurstStructureResult(
            pattern_type=pattern_type,
            num_bursts=num_bursts,
            total_pulses=total_pulses,
            mean_pulses_per_burst=mean_counts,
            std_pulses_per_burst=std_counts,
            min_pulses_per_burst=min_counts,
            max_pulses_per_burst=max_counts,
            mean_burst_duration_sec=mean_dur,
            std_burst_duration_sec=std_dur,
            mean_ibi_sec=mean_ibi,
            std_ibi_sec=std_ibi,
            ibi_jitter_percent=jitter_pct,
            burst_repetition_rate_hz=brr_hz,
            burst_stability_score=burst_stability,
            confidence=confidence,
            valid=True,
            diagnostics=diagnostics,
        )


def analyze_burst_structure(
    burst_labels: Optional[np.ndarray] = None,
    timestamps_sec: Optional[np.ndarray] = None,
    burst_start_times_sec: Optional[np.ndarray] = None,
    burst_durations_sec: Optional[np.ndarray] = None,
    burst_pulse_counts: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> BurstStructureResult:
    """
    Functional wrapper to analyze burst sequence structure.
    """
    metric = BurstStructure()
    return metric.compute(
        burst_labels=burst_labels,
        timestamps_sec=timestamps_sec,
        burst_start_times_sec=burst_start_times_sec,
        burst_durations_sec=burst_durations_sec,
        burst_pulse_counts=burst_pulse_counts,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  BurstStructure: Demonstration")
    print("=" * 60)
    pri_sequence = np.array([100e-6, 100e-6, 100e-6, 100e-6, 100e-6])
    metric = BurstStructure()
    result = metric.compute(pri_sequence)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
