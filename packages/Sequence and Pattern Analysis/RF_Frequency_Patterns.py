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
from .models import RFFrequencyPatternResult
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
    "rf_frequency_patterns",
    "rf_patterns",
    "frequency_patterns",
    "rf_frequency_pattern_analysis",
    "rffrequencypatterns",
])
class RFFrequencyPatterns(BaseSequencePatternMetric):
    """
    RF / Carrier Frequency Pattern Analyzer.

    Analyzes pulse-to-pulse carrier frequency sequences, frequency hopping, dwell-and-switch,
    repeated hop patterns, and frequency agility.

    This processor does NOT estimate RF from IQ waveforms or compute FFTs. The caller must
    supply pre-extracted carrier_freq_hz values from upstream packages (e.g., Frequency or Modulation_Analysis).
    """

    @property
    def name(self) -> str:
        return "rf_frequency_patterns"

    def compute(
        self,
        carrier_freq_hz: Any,
        timestamps_sec: Any | None = None,
        frequency_states_hz: list[float] | None = None,
        frequency_state_labels: list[int] | np.ndarray | None = None,
        sort_by_time: bool = False,
        tolerance_hz: float | None = None,
        freq_tolerance_hz: float | None = None,
        **kwargs: Any,
    ) -> RFFrequencyPatternResult:
        """
        Analyzes pattern structure across an already-available RF / carrier frequency sequence.

        Parameters
        ----------
        carrier_freq_hz : array-like
            1D array of precomputed carrier/RF frequency values in Hz.
        timestamps_sec : array-like, optional
            Corresponding pulse time-of-arrival timestamps.
        frequency_states_hz : list of float, optional
            Precomputed discrete frequency state centers from upstream Modulation_Analysis.
        frequency_state_labels : array-like, optional
            Precomputed discrete state assignments for each pulse.
        sort_by_time : bool, default=False
            Whether to sort frequencies chronologically by timestamps before evaluation.
        tolerance_hz : float, optional
            Frequency clustering tolerance in Hz (if states not supplied).
        freq_tolerance_hz : float, optional
            Alias for tolerance_hz.

        Returns
        -------
        RFFrequencyPatternResult
        """
        tol = freq_tolerance_hz if freq_tolerance_hz is not None else tolerance_hz
        if tol is not None:
            tol_val = float(tol)
            if tol_val < 0.0 or not math.isfinite(tol_val):
                raise ValueError(f"tolerance_hz must be non-negative and finite, got {tol}")

        if carrier_freq_hz is not None and frequency_state_labels is not None:
            try:
                c_len = len(carrier_freq_hz)
                f_len = len(frequency_state_labels)
                if c_len > 0 and f_len != c_len:
                    raise ValueError(f"frequency_state_labels length ({f_len}) must match carrier_freq_hz length ({c_len}).")
            except TypeError:
                pass

        val_arr, ts_arr, valid_mask, orig_indices = validate_sequence_inputs(
            values=carrier_freq_hz,
            timestamps_sec=timestamps_sec,
            sort_by_time=sort_by_time,
            allow_empty=True,
        )

        n = len(val_arr)
        valid_freqs = val_arr[valid_mask]

        if len(valid_freqs) < 2:
            return RFFrequencyPatternResult(
                pattern_type="unresolved",
                num_pulses=n,
                frequency_states_hz=[],
                state_sequence=np.full(n, -1, dtype=int),
                transition_matrix=np.empty((0, 0), dtype=np.float64),
                dwell_lengths=[],
                hop_sizes_hz=[],
                repetition_length=None,
                trend_slope_hz_per_pulse=0.0,
                mean_frequency_hz=float(np.mean(valid_freqs)) if len(valid_freqs) > 0 else 0.0,
                std_frequency_hz=0.0,
                stability_score=0.0,
                confidence=0.0,
                valid=False,
                diagnostics={"reason": "Insufficient valid carrier frequency samples (< 2)."},
            )

        mean_freq = float(np.mean(valid_freqs))
        std_freq = float(np.std(valid_freqs))
        effective_tol = tol if tol is not None else max(1e3, mean_freq * 1e-4)

        # 1. State assignment: utilize upstream states if supplied, otherwise estimate
        if frequency_states_hz is not None and frequency_state_labels is not None:
            freq_states = [float(f) for f in frequency_states_hz]
            state_labels_arr = np.asarray(frequency_state_labels, dtype=int)
            if len(state_labels_arr) == len(orig_indices):
                state_labels_arr = state_labels_arr[orig_indices]
            valid_state_seq = state_labels_arr[valid_mask]
            num_states = len(freq_states)
        elif frequency_states_hz is not None:
            freq_states = [float(f) for f in frequency_states_hz]
            freq_states_arr = np.array(freq_states)
            valid_state_seq = np.array([int(np.argmin(np.abs(f - freq_states_arr))) for f in valid_freqs], dtype=int)
            num_states = len(freq_states)
        else:
            freq_states, valid_state_seq = estimate_discrete_states(valid_freqs, tolerance=effective_tol)
            num_states = len(freq_states)

        full_state_seq = np.full(n, -1, dtype=int)
        if len(valid_state_seq) > 0 and np.any(valid_mask):
            full_state_seq[valid_mask] = valid_state_seq

        # 2. Build Markov transition matrix and dwell run-lengths with missing data breaks
        trans_mat = build_transition_matrix(full_state_seq, num_states, missing_value=-1)
        dwells = compute_dwell_lengths(full_state_seq, missing_value=-1)

        # 3. Hop size sequence across adjacent valid transitions
        hop_sizes = []
        for i in range(n - 1):
            if valid_mask[i] and valid_mask[i + 1]:
                hop_mag = abs(float(val_arr[i + 1] - val_arr[i]))
                if hop_mag > effective_tol:
                    hop_sizes.append(hop_mag)

        # Fraction of pulses that stay at the same frequency
        diffs, _, _ = adjacent_difference_with_gaps(val_arr, valid_mask)
        same_freq_fraction = float(np.mean(np.abs(diffs) <= effective_tol)) if len(diffs) > 0 else 1.0

        # 4. Trend and periodicity estimation
        trend_slope = estimate_linear_trend(valid_freqs, ts_arr[valid_mask] if (ts_arr is not None and len(ts_arr) == n) else None)
        is_periodic, rep_len, motif, fidelity, harmonics = detect_symbolic_periodicity(full_state_seq, missing_value=-1)

        # Monotonic check across individual steps (linear agile / chirping pulse train)
        is_monotonic_trend = len(diffs) > 3 and (np.all(diffs >= -1e-9) or np.all(diffs <= 1e-9)) and (same_freq_fraction < 0.4) and abs(trend_slope) > max(1e3, mean_freq * 1e-6)

        # 5. Pattern Type Classification
        pattern_type = "unresolved"
        confidence = 0.5
        mean_dwell = float(np.mean(dwells)) if dwells else 1.0

        if std_freq < max(1e3, mean_freq * 1e-4) or num_states <= 1:
            pattern_type = "fixed"
            stability_score = float(np.clip(1.0 - (std_freq / max(mean_freq * 1e-3, 1e3)), 0.7, 1.0))
            confidence = 0.95
        elif is_monotonic_trend:
            pattern_type = "linear_agile"
            stability_score = 0.70
            confidence = 0.85
        elif is_periodic and rep_len is not None and rep_len >= 2:
            if mean_dwell < 1.8:
                pattern_type = "frequency_hopping"
                confidence = float(np.clip(fidelity * 0.95, 0.7, 1.0))
            else:
                pattern_type = "dwell_and_switch"
                confidence = float(np.clip(fidelity * 0.90, 0.65, 1.0))
            stability_score = float(fidelity)
        elif num_states >= 2 and mean_dwell >= 2.0 and same_freq_fraction >= 0.4:
            pattern_type = "dwell_and_switch"
            stability_score = 0.75
            confidence = 0.80
        elif num_states >= 2 and len(hop_sizes) >= (0.25 * len(valid_freqs)):
            pattern_type = "frequency_hopping"
            stability_score = 0.60
            confidence = 0.85
        elif abs(trend_slope) > max(1e3, mean_freq * 1e-4):
            pattern_type = "linear_agile"
            stability_score = 0.55
            confidence = 0.75
        elif num_states > 6:
            pattern_type = "agile"
            stability_score = 0.30
            confidence = 0.70
        else:
            pattern_type = "discrete_states"
            stability_score = 0.65
            confidence = 0.80

        return RFFrequencyPatternResult(
            pattern_type=pattern_type,
            num_pulses=n,
            frequency_states_hz=freq_states,
            state_sequence=full_state_seq,
            transition_matrix=trans_mat,
            dwell_lengths=dwells,
            hop_sizes_hz=hop_sizes,
            repetition_length=rep_len if is_periodic else None,
            trend_slope_hz_per_pulse=trend_slope,
            mean_frequency_hz=mean_freq,
            std_frequency_hz=std_freq,
            stability_score=stability_score,
            confidence=confidence,
            valid=True,
            diagnostics={
                "is_periodic": is_periodic,
                "periodic_fidelity": fidelity,
                "repeating_motif": motif,
                "num_hops": len(hop_sizes),
                "same_freq_fraction": same_freq_fraction,
            },
        )


def analyze_rf_pattern(
    carrier_freq_hz: Any,
    timestamps_sec: Any | None = None,
    frequency_states_hz: list[float] | None = None,
    frequency_state_labels: list[int] | np.ndarray | None = None,
    sort_by_time: bool = False,
    tolerance_hz: float | None = None,
    freq_tolerance_hz: float | None = None,
    **kwargs: Any,
) -> RFFrequencyPatternResult:
    """
    Convenience function to analyze RF / carrier frequency patterns.

    This function does not estimate RF from IQ waveforms.
    The caller must provide precomputed carrier frequency values from upstream packages.
    """
    processor = RFFrequencyPatterns(**kwargs)
    return processor.compute(
        carrier_freq_hz=carrier_freq_hz,
        timestamps_sec=timestamps_sec,
        frequency_states_hz=frequency_states_hz,
        frequency_state_labels=frequency_state_labels,
        sort_by_time=sort_by_time,
        tolerance_hz=tolerance_hz,
        freq_tolerance_hz=freq_tolerance_hz,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  RFFrequencyPatterns: Demonstration")
    print("=" * 60)
    seq = np.array([10.0, 10.2, 9.9, 10.1, 10.0, 10.05, 9.95])
    metric = RFFrequencyPatterns()
    result = metric.compute(seq)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
