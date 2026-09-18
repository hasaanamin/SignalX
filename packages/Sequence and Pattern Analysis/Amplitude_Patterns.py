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
from .models import AmplitudePatternResult
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
    "amplitude_patterns",
    "amplitude_pattern",
    "power_patterns",
    "amplitudepatterns",
])
class AmplitudePatterns(BaseSequencePatternMetric):
    """
    Amplitude / Power Sequence Pattern Analyzer.

    Analyzes pulse-to-pulse amplitude level sequences, power stepping, antenna scan modulations,
    and amplitude stability.

    This processor does NOT estimate amplitude or power from raw IQ waveforms. The caller must
    supply pre-extracted power or amplitude measurements from the Amplitude package.
    """

    @property
    def name(self) -> str:
        return "amplitude_patterns"

    def compute(
        self,
        amplitude_values: Any,
        timestamps_sec: Any | None = None,
        sort_by_time: bool = False,
        tolerance: float | None = None,
        unit: str = "dB",
        **kwargs: Any,
    ) -> AmplitudePatternResult:
        """
        Analyzes pattern structure across an already-available amplitude/power sequence.

        Parameters
        ----------
        amplitude_values : array-like
            1D array of pre-extracted pulse amplitude or power measurements (e.g. power_db, peak_amplitude).
        timestamps_sec : array-like, optional
            Corresponding pulse time-of-arrival timestamps.
        sort_by_time : bool, default=False
            Whether to sort values chronologically by timestamps before evaluation.
        tolerance : float, optional
            Clustering tolerance for discrete amplitude levels (defaults to 1.5 dB or 5% linear).
        unit : str, default="dB"
            Unit descriptor ("dB", "V", "W").

        Returns
        -------
        AmplitudePatternResult
        """
        val_arr, ts_arr, valid_mask, orig_indices = validate_sequence_inputs(
            values=amplitude_values,
            timestamps_sec=timestamps_sec,
            sort_by_time=sort_by_time,
            allow_empty=True,
        )

        n = len(val_arr)
        valid_amps = val_arr[valid_mask]

        if len(valid_amps) < 2:
            return AmplitudePatternResult(
                pattern_type="unresolved",
                num_pulses=n,
                amplitude_levels=[],
                state_sequence=np.full(n, -1, dtype=int),
                transition_matrix=np.empty((0, 0), dtype=np.float64),
                dwell_lengths=[],
                trend_slope_per_pulse=0.0,
                mean_amplitude=float(np.mean(valid_amps)) if len(valid_amps) > 0 else 0.0,
                std_amplitude=0.0,
                dynamic_range=0.0,
                stability_score=0.0,
                confidence=0.0,
                valid=False,
                unit=unit,
                diagnostics={"reason": "Insufficient valid amplitude samples (< 2)."},
            )

        mean_amp = float(np.mean(valid_amps))
        std_amp = float(np.std(valid_amps))
        dyn_range = float(np.max(valid_amps) - np.min(valid_amps))

        # 1. State estimation
        if tolerance is not None:
            tol_val = float(tolerance)
            if tol_val < 0.0 or not math.isfinite(tol_val):
                raise ValueError(f"tolerance must be non-negative and finite, got {tolerance}")

        tol = tolerance if tolerance is not None else (1.5 if unit.lower() == "db" else max(mean_amp * 0.05, 1e-3))
        amp_levels, valid_state_seq = estimate_discrete_states(valid_amps, tolerance=tol)
        num_states = len(amp_levels)

        full_state_seq = np.full(n, -1, dtype=int)
        if len(valid_state_seq) > 0 and np.any(valid_mask):
            full_state_seq[valid_mask] = valid_state_seq

        # 2. Markov matrix and dwells with missing data breaks
        trans_mat = build_transition_matrix(full_state_seq, num_states, missing_value=-1)
        dwells = compute_dwell_lengths(full_state_seq, missing_value=-1)

        # 3. Trend and periodicity
        trend_slope = estimate_linear_trend(valid_amps, ts_arr[valid_mask] if (ts_arr is not None and len(ts_arr) == n) else None)
        is_periodic, rep_len, motif, fidelity, harmonics = detect_symbolic_periodicity(full_state_seq, missing_value=-1)

        # Check for smooth periodic scan (e.g. sinusoidal/conical antenna scan)
        diffs, _, _ = adjacent_difference_with_gaps(val_arr, valid_mask)
        sign_changes = int(np.sum(diffs[:-1] * diffs[1:] < 0)) if len(diffs) >= 2 else 0

        centered_amps = valid_amps - mean_amp
        var_amp = float(np.var(centered_amps))
        has_autocorr_peak = False
        if var_amp > 1e-12 and len(centered_amps) >= 6:
            max_lag = min(len(centered_amps) // 2, 30)
            for lag in range(3, max_lag + 1):
                if len(centered_amps) - lag >= 3:
                    c = float(np.corrcoef(centered_amps[:-lag], centered_amps[lag:])[0, 1])
                    if math.isfinite(c) and c >= 0.5:
                        has_autocorr_peak = True
                        break

        # Monotonic trends
        is_monotonic_decay = len(diffs) > 3 and np.all(diffs <= 1e-6) and trend_slope < -0.05
        is_monotonic_rising = len(diffs) > 3 and np.all(diffs >= -1e-6) and trend_slope > 0.05

        # R^2 linear fit
        t_vec = np.arange(len(valid_amps), dtype=np.float64)
        t_mean = float(np.mean(t_vec))
        y_fit = mean_amp + trend_slope * (t_vec - t_mean)
        res_var = float(np.var(valid_amps - y_fit))
        tot_var = float(np.var(valid_amps))
        r2_trend = max(0.0, min(1.0, 1.0 - (res_var / max(tot_var, 1e-12)))) if tot_var > 1e-12 else 1.0
        is_true_trend = (r2_trend >= 0.70 and abs(trend_slope) > (0.05 if unit.lower() == "db" else abs(mean_amp) * 0.005))

        is_scan = (sign_changes >= 2) and (dyn_range > (3.0 if unit.lower() == "db" else 0.1 * abs(mean_amp))) and (
            (is_periodic and rep_len is not None and rep_len >= 2) or has_autocorr_peak
        )

        # 4. Pattern classification
        pattern_type = "unresolved"
        confidence = 0.5
        mean_dwell = float(np.mean(dwells)) if dwells else 1.0

        if dyn_range < (1.0 if unit.lower() == "db" else mean_amp * 0.05) or num_states <= 1:
            pattern_type = "stable_flat"
            stability_score = float(np.clip(1.0 - (std_amp / max(dyn_range, 1.0)), 0.8, 1.0))
            confidence = 0.95
        elif is_monotonic_decay:
            pattern_type = "decaying"
            stability_score = 0.70
            confidence = 0.85
        elif is_monotonic_rising:
            pattern_type = "trending"
            stability_score = 0.70
            confidence = 0.85
        elif is_scan or (is_periodic and rep_len is not None and rep_len >= 2):
            pattern_type = "periodic_scan"
            stability_score = float(fidelity if is_periodic else 0.8)
            confidence = float(np.clip((fidelity if is_periodic else 0.85) * 0.95, 0.7, 1.0))
        elif num_states >= 2 and mean_dwell >= 2.5:
            pattern_type = "switching"
            stability_score = 0.75
            confidence = 0.80
        elif is_true_trend:
            pattern_type = "decaying" if trend_slope < 0 else "trending"
            stability_score = 0.60
            confidence = 0.75
        elif num_states >= 2 and num_states <= 6:
            pattern_type = "discrete_levels"
            stability_score = 0.65
            confidence = 0.80
        else:
            pattern_type = "irregular"
            stability_score = 0.30
            confidence = 0.60

        return AmplitudePatternResult(
            pattern_type=pattern_type,
            num_pulses=n,
            amplitude_levels=amp_levels,
            state_sequence=full_state_seq,
            transition_matrix=trans_mat,
            dwell_lengths=dwells,
            trend_slope_per_pulse=trend_slope,
            mean_amplitude=mean_amp,
            std_amplitude=std_amp,
            dynamic_range=dyn_range,
            stability_score=stability_score,
            confidence=confidence,
            valid=True,
            unit=unit,
            diagnostics={
                "is_periodic": is_periodic,
                "periodic_fidelity": fidelity,
                "repeating_motif": motif,
                "sign_changes": int(sign_changes),
            },
        )


def analyze_amplitude_pattern(
    amplitude_values: Any,
    timestamps_sec: Any | None = None,
    sort_by_time: bool = False,
    tolerance: float | None = None,
    unit: str = "dB",
    **kwargs: Any,
) -> AmplitudePatternResult:
    """
    Convenience function to analyze amplitude/power patterns.

    This function does not estimate amplitude from raw IQ.
    The caller must provide precomputed amplitude values from the Amplitude package.
    """
    processor = AmplitudePatterns(**kwargs)
    return processor.compute(
        amplitude_values=amplitude_values,
        timestamps_sec=timestamps_sec,
        sort_by_time=sort_by_time,
        tolerance=tolerance,
        unit=unit,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  AmplitudePatterns: Demonstration")
    print("=" * 60)
    amplitudes = np.array([1.0, 1.05, 0.98, 1.02, 1.0, 0.99, 1.01])
    metric = AmplitudePatterns()
    result = metric.compute(amplitudes)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
