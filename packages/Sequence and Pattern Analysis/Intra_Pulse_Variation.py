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

from .models import IntraPulseVariationResult
from .base import BaseSequencePatternMetric, SequencePatternMetricFactory
from .validation import validate_sequence_inputs
from ._common import estimate_linear_trend


@SequencePatternMetricFactory.register([
    "intra_pulse_variation",
    "intra_pulse",
    "intrapulsevariation",
])
class IntraPulseVariation(BaseSequencePatternMetric):
    """
    Analyzes variation, linearity, and transition structure within an individual pulse's
    pre-extracted instantaneous parameter trace (e.g., instantaneous frequency, instantaneous phase,
    or instantaneous amplitude envelope).
    """

    @property
    def name(self) -> str:
        return "intra_pulse_variation"

    def compute(
        self,
        intra_pulse_trace: np.ndarray,
        sample_rate_hz: Optional[float] = None,
        trace_type: str = "frequency",
        step_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> IntraPulseVariationResult:
        """
        Computes intra-pulse variation statistics across pre-extracted 1D instantaneous trace.

        Parameters
        ----------
        intra_pulse_trace : np.ndarray
            1D array of instantaneous measurements inside a single pulse (e.g., Hz, rad, or linear amplitude).
        sample_rate_hz : float, optional
            Sampling rate in Hz of the intra-pulse trace.
        trace_type : str, default="frequency"
            Type of trace: 'frequency', 'phase', 'amplitude'.
        step_threshold : float, optional
            Threshold on first difference to identify internal change-points/sub-pulse transitions.

        Returns
        -------
        IntraPulseVariationResult
        """
        if sample_rate_hz is not None:
            sr = float(sample_rate_hz)
            if sr <= 0.0 or not math.isfinite(sr):
                raise ValueError(f"sample_rate_hz must be a positive finite float, got {sample_rate_hz}")
        if step_threshold is not None:
            st = float(step_threshold)
            if st < 0.0 or not math.isfinite(st):
                raise ValueError(f"step_threshold must be a non-negative finite float, got {step_threshold}")

        arr, _, valid_mask, _ = validate_sequence_inputs(intra_pulse_trace, allow_empty=True)
        valid_vals = arr[valid_mask]
        n_valid = len(valid_vals)

        if n_valid < 2:
            diag = {"reason": "insufficient_valid_samples", "num_valid": n_valid}
            if trace_type.lower() == "phase":
                diag["phase_input_requirement"] = "unwrapped"
                diag["possible_wrapped_phase_input"] = False
            return IntraPulseVariationResult(
                num_samples=n_valid,
                variation_magnitude=0.0,
                trend_slope=0.0,
                smoothness_score=0.0,
                is_piecewise_linear=False,
                intra_pulse_stability=0.0,
                change_point_indices=[],
                max_gradient=0.0,
                mean_gradient=0.0,
                confidence=0.0,
                valid=False,
                diagnostics=diag,
            )

        # Scale time if sample rate provided
        dt = 1.0 / float(sample_rate_hz) if (sample_rate_hz is not None and sample_rate_hz > 0) else 1.0
        time_vec = np.arange(n_valid, dtype=np.float64) * dt

        # Basic statistics
        val_min = float(np.min(valid_vals))
        val_max = float(np.max(valid_vals))
        val_range = val_max - val_min
        val_std = float(np.std(valid_vals))

        # Linear trend / chirp rate
        slope = estimate_linear_trend(valid_vals, time_vec)

        # Check for wrapped phase input jumps (~ +/- 2*pi)
        possible_wrapped_phase = False
        if trace_type.lower() == "phase":
            raw_diffs = np.diff(valid_vals)
            for d in raw_diffs:
                abs_d = abs(float(d))
                if abs(abs_d - 2.0 * math.pi) < 0.35 or abs_d > 5.8:
                    possible_wrapped_phase = True
                    break

        # Gradients (first difference across adjacent valid pairs)
        n_total = len(arr)
        adjacent_diffs_list = []
        adjacent_jump_indices = []
        for i in range(n_total - 1):
            if valid_mask[i] and valid_mask[i + 1]:
                g = abs(float(arr[i + 1] - arr[i])) / dt
                adjacent_diffs_list.append(g)
                adjacent_jump_indices.append((i + 1, g))

        abs_diffs = np.array(adjacent_diffs_list, dtype=np.float64)
        max_grad = float(np.max(abs_diffs)) if len(abs_diffs) > 0 else 0.0
        mean_grad = float(np.mean(abs_diffs)) if len(abs_diffs) > 0 else 0.0

        # Change points (sharp jumps / sub-pulse steps)
        if step_threshold is None:
            med_diff = float(np.median(abs_diffs)) if len(abs_diffs) > 0 else 0.0
            std_diff = float(np.std(abs_diffs)) if len(abs_diffs) > 0 else 0.0
            thresh = med_diff + 3.0 * std_diff
        else:
            thresh = float(step_threshold)

        # Find change point indices (relative to original array indices)
        change_points = []
        if len(abs_diffs) > 0 and thresh > 1e-9:
            change_points = [idx for idx, g in adjacent_jump_indices if g > thresh]

        # Linearity assessment (residuals from linear fit)
        mean_t = float(np.mean(time_vec))
        mean_y = float(np.mean(valid_vals))
        y_fit = slope * (time_vec - mean_t) + mean_y
        residuals = valid_vals - y_fit
        res_var = float(np.var(residuals))
        total_var = float(np.var(valid_vals))

        if total_var < 1e-12:
            linearity_r2 = 1.0  # Constant line
            is_piecewise_linear = True
            smoothness_score = 1.0
        else:
            linearity_r2 = max(0.0, min(1.0, 1.0 - (res_var / total_var)))
            is_piecewise_linear = (linearity_r2 >= 0.85) or (len(change_points) > 0 and len(change_points) <= 16)
            # Smoothness: ratio of second differences to first differences
            if len(valid_vals) > 2:
                second_diffs = np.diff(valid_vals, n=2)
                second_energy = float(np.mean(second_diffs ** 2))
                smoothness_score = float(1.0 / (1.0 + second_energy / (val_range ** 2 + 1e-12)))
            else:
                smoothness_score = 1.0

        # Stability: high if variation is predictable or very low
        if val_range < 1e-9:
            stability = 1.0
        elif linearity_r2 > 0.9:
            stability = float(linearity_r2)
        else:
            stability = float(max(0.0, 1.0 - (val_std / (val_range + 1e-9))))

        confidence = min(1.0, max(0.2, float(n_valid) / 20.0))

        diagnostics = {
            "trace_type": trace_type,
            "val_min": val_min,
            "val_max": val_max,
            "val_std": val_std,
            "linearity_r2": linearity_r2,
            "num_change_points": len(change_points),
        }
        if trace_type.lower() == "phase":
            diagnostics["phase_input_requirement"] = "unwrapped"
            diagnostics["possible_wrapped_phase_input"] = possible_wrapped_phase

        return IntraPulseVariationResult(
            num_samples=n_valid,
            variation_magnitude=val_range,
            trend_slope=slope,
            smoothness_score=smoothness_score,
            is_piecewise_linear=is_piecewise_linear,
            intra_pulse_stability=stability,
            change_point_indices=change_points,
            max_gradient=max_grad,
            mean_gradient=mean_grad,
            confidence=confidence,
            valid=True,
            diagnostics=diagnostics,
        )


def analyze_intra_pulse_variation(
    intra_pulse_trace: np.ndarray,
    sample_rate_hz: Optional[float] = None,
    trace_type: str = "frequency",
    step_threshold: Optional[float] = None,
    **kwargs: Any,
) -> IntraPulseVariationResult:
    """
    Functional wrapper to analyze intra-pulse variation across pre-extracted instantaneous trace.
    """
    metric = IntraPulseVariation()
    return metric.compute(
        intra_pulse_trace=intra_pulse_trace,
        sample_rate_hz=sample_rate_hz,
        trace_type=trace_type,
        step_threshold=step_threshold,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  IntraPulseVariation: Demonstration")
    print("=" * 60)
    t = np.linspace(0, 1e-6, 128)
    pulses = np.array([np.exp(1j * 2 * np.pi * 10e6 * t) for _ in range(5)])
    metric = IntraPulseVariation()
    result = metric.compute(pulses)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
