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
from .models import PhasePatternResult
from .validation import validate_sequence_inputs
from ._common import (
    wrap_circular_difference,
    estimate_discrete_states,
    build_transition_matrix,
    detect_symbolic_periodicity,
    adjacent_difference_with_gaps,
)


@SequencePatternMetricFactory.register([
    "phase_patterns",
    "phase_pattern",
    "phase_sequence_patterns",
    "phasepatterns",
])
class PhasePatterns(BaseSequencePatternMetric):
    """
    Inter-Pulse Phase Pattern Analyzer.

    Analyzes pulse-to-pulse phase sequences, circular phase progression, discrete phase codes,
    and inter-pulse phase coherence.

    This processor does NOT compute raw instantaneous phase or atan2(I,Q) from IQ waveforms.
    The caller must supply pre-extracted phase measurements from the Phase or Modulation_Analysis package.
    """

    @property
    def name(self) -> str:
        return "phase_patterns"

    def compute(
        self,
        phase_rad: Any,
        timestamps_sec: Any | None = None,
        sort_by_time: bool = False,
        tolerance_rad: float = 0.15,
        **kwargs: Any,
    ) -> PhasePatternResult:
        """
        Analyzes pattern structure across an already-available inter-pulse phase sequence.

        Parameters
        ----------
        phase_rad : array-like
            1D array of pre-extracted pulse phase measurements in radians.
        timestamps_sec : array-like, optional
            Corresponding pulse time-of-arrival timestamps.
        sort_by_time : bool, default=False
            Whether to sort phases chronologically by timestamps before evaluation.
        tolerance_rad : float, default=0.15
            Clustering tolerance in radians for discrete phase state estimation.

        Returns
        -------
        PhasePatternResult
        """
        if tolerance_rad is not None:
            tol_val = float(tolerance_rad)
            if tol_val < 0.0 or not math.isfinite(tol_val):
                raise ValueError(f"tolerance_rad must be non-negative and finite, got {tolerance_rad}")

        val_arr, ts_arr, valid_mask, orig_indices = validate_sequence_inputs(
            values=phase_rad,
            timestamps_sec=timestamps_sec,
            sort_by_time=sort_by_time,
            allow_empty=True,
        )

        n = len(val_arr)
        valid_phases = val_arr[valid_mask]

        if len(valid_phases) < 2:
            return PhasePatternResult(
                pattern_type="unresolved",
                num_pulses=n,
                circular_deltas_rad=np.empty(0, dtype=np.float64),
                mean_circular_delta_rad=0.0,
                circular_variance=1.0,
                phase_states_rad=[],
                state_sequence=np.full(n, -1, dtype=int),
                transition_matrix=np.empty((0, 0), dtype=np.float64),
                repetition_length=None,
                phase_stability_score=0.0,
                confidence=0.0,
                valid=False,
                diagnostics={"reason": "Insufficient valid phase samples (< 2)."},
            )

        # 1. Circular phase differencing with missing-data gap handling
        circ_deltas, _, _ = adjacent_difference_with_gaps(val_arr, valid_mask, is_circular=True)

        # 2. Circular mean and circular variance of deltas: R = |mean(exp(j*delta))|
        if len(circ_deltas) > 0:
            complex_phasors = np.exp(1j * circ_deltas)
            mean_phasor = np.mean(complex_phasors)
            r_mag = float(np.abs(mean_phasor))
            mean_circ_delta = float(np.angle(mean_phasor))
            circ_var = float(1.0 - r_mag)  # 0 = completely coherent/unvarying, 1 = completely uniform random
        else:
            r_mag = 1.0
            mean_circ_delta = 0.0
            circ_var = 0.0

        # 3. Discrete phase state estimation (wrapped into [-pi, pi) with circular distance)
        wrapped_phases = np.asarray(wrap_circular_difference(valid_phases), dtype=np.float64)
        tol_rad = float(tolerance_rad) if tolerance_rad is not None else 0.15

        clusters: list[list[float]] = [[float(wrapped_phases[0])]]
        for p in wrapped_phases[1:]:
            assigned = False
            for c in clusters:
                c_mean = float(np.angle(np.mean(np.exp(1j * np.array(c)))))
                dist = abs(float(wrap_circular_difference(p - c_mean)))
                if dist <= tol_rad:
                    c.append(float(p))
                    assigned = True
                    break
            if not assigned:
                clusters.append([float(p)])

        merged = True
        while merged and len(clusters) > 1:
            merged = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    m_i = float(np.angle(np.mean(np.exp(1j * np.array(clusters[i])))))
                    m_j = float(np.angle(np.mean(np.exp(1j * np.array(clusters[j])))))
                    dist = abs(float(wrap_circular_difference(m_i - m_j)))
                    if dist <= tol_rad:
                        clusters[i].extend(clusters[j])
                        clusters.pop(j)
                        merged = True
                        break
                if merged:
                    break

        phase_states = [float(np.angle(np.mean(np.exp(1j * np.array(c))))) for c in clusters]
        phase_states.sort()
        num_states = len(phase_states)

        valid_state_seq = np.zeros(len(wrapped_phases), dtype=int)
        for i, p in enumerate(wrapped_phases):
            dists = [abs(float(wrap_circular_difference(p - lvl))) for lvl in phase_states]
            valid_state_seq[i] = int(np.argmin(dists))

        full_state_seq = np.full(n, -1, dtype=int)
        if len(valid_state_seq) > 0 and np.any(valid_mask):
            full_state_seq[valid_mask] = valid_state_seq

        # 4. Transition matrix and periodicity with missing data breaks
        trans_mat = build_transition_matrix(full_state_seq, num_states, missing_value=-1)
        is_periodic, rep_len, motif, fidelity, harmonics = detect_symbolic_periodicity(full_state_seq, missing_value=-1)

        # 5. Pattern Classification
        pattern_type = "unresolved"
        confidence = 0.5

        if circ_var < 0.05 and abs(mean_circ_delta) < 0.05:
            pattern_type = "coherent_stable"
            stability_score = 1.0 - circ_var
            confidence = 0.95
        elif num_states == 2 and is_periodic and rep_len == 2:
            pattern_type = "discrete_biphase"
            stability_score = float(fidelity)
            confidence = 0.90
        elif num_states in (2, 4, 8) and is_periodic and rep_len is not None and rep_len > 2:
            pattern_type = "polyphase_code" if num_states > 2 else "discrete_biphase"
            stability_score = float(fidelity)
            confidence = 0.90
        elif circ_var < 0.10 and abs(mean_circ_delta) >= 0.05:
            pattern_type = "linear_ramp"
            stability_score = 1.0 - circ_var
            confidence = 0.90
        elif is_periodic and rep_len is not None and rep_len >= 2:
            pattern_type = "repeating_code"
            stability_score = float(fidelity)
            confidence = float(np.clip(fidelity * 0.95, 0.7, 1.0))
        elif num_states in (2, 4, 8) and circ_var < 0.40:
            pattern_type = "discrete_biphase" if num_states == 2 else "polyphase_code"
            stability_score = 0.70
            confidence = 0.85
        elif num_states >= 2 and circ_var < 0.50:
            pattern_type = "switching"
            stability_score = 0.60
            confidence = 0.75
        else:
            pattern_type = "random_incoherent"
            stability_score = 0.10
            confidence = 0.80

        return PhasePatternResult(
            pattern_type=pattern_type,
            num_pulses=n,
            circular_deltas_rad=circ_deltas,
            mean_circular_delta_rad=mean_circ_delta,
            circular_variance=circ_var,
            phase_states_rad=phase_states,
            state_sequence=full_state_seq,
            transition_matrix=trans_mat,
            repetition_length=rep_len if is_periodic else None,
            phase_stability_score=stability_score,
            confidence=confidence,
            valid=True,
            diagnostics={
                "r_magnitude": r_mag,
                "is_periodic": is_periodic,
                "periodic_fidelity": fidelity,
                "repeating_motif": motif,
            },
        )


def analyze_phase_pattern(
    phase_rad: Any,
    timestamps_sec: Any | None = None,
    sort_by_time: bool = False,
    tolerance_rad: float = 0.15,
    **kwargs: Any,
) -> PhasePatternResult:
    """
    Convenience function to analyze phase patterns.

    This function does not estimate phase from raw IQ.
    The caller must provide precomputed phase values from the Phase package.
    """
    processor = PhasePatterns(**kwargs)
    return processor.compute(
        phase_rad=phase_rad,
        timestamps_sec=timestamps_sec,
        sort_by_time=sort_by_time,
        tolerance_rad=tolerance_rad,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  PhasePatterns: Demonstration")
    print("=" * 60)
    seq = np.array([10.0, 10.2, 9.9, 10.1, 10.0, 10.05, 9.95])
    metric = PhasePatterns()
    result = metric.compute(seq)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
