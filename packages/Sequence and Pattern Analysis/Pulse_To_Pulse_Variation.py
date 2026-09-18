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
from .models import PulseToPulseVariationResult
from .validation import validate_sequence_inputs
from ._common import estimate_linear_trend, wrap_circular_difference, adjacent_difference_with_gaps


@SequencePatternMetricFactory.register([
    "pulse_to_pulse_variation",
    "pulse_variation",
    "p2p_variation",
    "pulsetopulsevariation",
])
class PulseToPulseVariation(BaseSequencePatternMetric):
    """
    Pulse-to-Pulse Variation Processor.

    Analyzes sequential variation and stability across precomputed pulse-level measurements
    (e.g., carrier frequency, pulse width, amplitude, power, phase, or bandwidth).

    This processor does NOT estimate raw pulse features from IQ waveforms. The caller must
    supply pre-extracted pulse sequences produced by upstream measurement packages.
    """

    @property
    def name(self) -> str:
        return "pulse_to_pulse_variation"

    def compute(
        self,
        values: Any,
        timestamps_sec: Any | None = None,
        sort_by_time: bool = False,
        normalize: bool = False,
        is_circular: bool = False,
        step_threshold: float | None = None,
        metric_name: str = "generic",
        strict_adjacency: bool = True,
        **kwargs: Any,
    ) -> PulseToPulseVariationResult:
        """
        Computes consecutive differences and stability statistics across a pulse feature sequence.

        Parameters
        ----------
        values : array-like
            1D array of pre-extracted pulse measurements (e.g., carrier_freq_hz, power_db, pw_sec).
        timestamps_sec : array-like, optional
            Corresponding pulse time-of-arrival timestamps.
        sort_by_time : bool, default=False
            Whether to sort values chronologically by timestamps_sec before differencing.
        normalize : bool, default=False
            Whether to normalize deltas by the mean value magnitude.
        is_circular : bool, default=False
            Whether values represent circular angular measurements (e.g. radians in [-pi, pi)).
        step_threshold : float, optional
            Threshold for detecting abrupt step change points.
        metric_name : str, default="generic"
            Descriptor name for diagnostic reporting.
        strict_adjacency : bool, default=True
            If True (default), only strictly adjacent sample pairs (i, i+1) are differenced (missing samples break deltas).
            If False (legacy opt-in), deltas are computed across the valid subsequence of measurements.

        Returns
        -------
        PulseToPulseVariationResult
        """
        if step_threshold is not None:
            st = float(step_threshold)
            if st < 0.0 or not math.isfinite(st):
                raise ValueError(f"step_threshold must be a non-negative finite float, got {step_threshold}")

        val_arr, ts_arr, valid_mask, orig_indices = validate_sequence_inputs(
            values=values,
            timestamps_sec=timestamps_sec,
            sort_by_time=sort_by_time,
            allow_empty=True,
        )

        n = len(val_arr)
        valid_indices = np.where(valid_mask)[0]
        valid_vals = val_arr[valid_indices]

        if n < 2 or len(valid_vals) < 2:
            return PulseToPulseVariationResult(
                metric_name=metric_name,
                num_pulses=n,
                num_valid_pairs=0,
                delta=np.empty(0, dtype=np.float64),
                absolute_delta=np.empty(0, dtype=np.float64),
                relative_delta=np.empty(0, dtype=np.float64),
                signed_delta=np.empty(0, dtype=np.float64),
                valid_mask=valid_mask,
                mean_delta=0.0,
                std_delta=0.0,
                max_delta=0.0,
                stability_score=0.0,
                trend_slope=0.0,
                confidence=0.0,
                valid=False,
                normalized_delta=np.empty(0, dtype=np.float64) if normalize else None,
                delta_regularity_score=0.0,
                feature_stability_score=0.0,
                step_change_indices=[],
                diagnostics={"reason": "Insufficient valid pulses (< 2)."},
            )

        if strict_adjacency:
            delta, target_indices, signed_delta = adjacent_difference_with_gaps(
                values=val_arr,
                valid_mask=valid_mask,
                is_circular=is_circular,
            )
            num_pairs = len(delta)
            abs_delta = np.abs(delta)
            if num_pairs > 0:
                from_indices = target_indices - 1
                denom = np.maximum(np.abs(val_arr[from_indices]), 1e-12)
                rel_delta = abs_delta / denom
                mean_d = float(np.mean(abs_delta))
                std_d = float(np.std(abs_delta))
                max_d = float(np.max(abs_delta))
            else:
                rel_delta = np.empty(0, dtype=np.float64)
                mean_d = 0.0
                std_d = 0.0
                max_d = 0.0
        else:
            if is_circular:
                diff_raw = np.diff(valid_vals)
                delta = np.asarray(wrap_circular_difference(diff_raw), dtype=np.float64)
            else:
                delta = np.diff(valid_vals)
            signed_delta = delta
            abs_delta = np.abs(delta)
            num_pairs = len(delta)
            target_indices = valid_indices[1:]
            from_vals = valid_vals[:-1]
            denom = np.maximum(np.abs(from_vals), 1e-12)
            rel_delta = abs_delta / denom
            mean_d = float(np.mean(abs_delta)) if num_pairs > 0 else 0.0
            std_d = float(np.std(abs_delta)) if num_pairs > 0 else 0.0
            max_d = float(np.max(abs_delta)) if num_pairs > 0 else 0.0

        valid_indices = np.where(valid_mask)[0]
        valid_vals = val_arr[valid_indices]

        # Linear trend across valid measurements
        trend_slope = estimate_linear_trend(valid_vals, ts_arr[valid_indices] if len(ts_arr) == n else None)

        # Step change detection
        if step_threshold is None:
            thresh = mean_d + 3.0 * std_d if std_d > 1e-9 else (mean_d * 2.0 if mean_d > 0 else 1.0)
        else:
            thresh = float(step_threshold)

        step_indices: list[int] = []
        if len(abs_delta) > 0 and thresh > 1e-9:
            jump_mask = np.where(abs_delta >= thresh)[0]
            step_indices = [int(target_indices[j]) for j in jump_mask]

        # Reference scale and feature stability
        mean_mag = float(np.mean(np.abs(valid_vals))) if len(valid_vals) > 0 else 0.0
        val_std = float(np.std(valid_vals)) if len(valid_vals) > 0 else 0.0
        val_range = float(np.max(valid_vals) - np.min(valid_vals)) if len(valid_vals) > 0 else 0.0
        ref_scale = max(mean_mag, val_std, 1e-6)
        norm_delta = delta / ref_scale

        # Feature stability score: measures how little the feature itself varies (1.0 = constant, << 0.5 for large ramp)
        spread_metric = (val_range + 2.0 * val_std) / ref_scale
        feature_stability_score = float(np.clip(1.0 / (1.0 + spread_metric), 0.0, 1.0))

        # Delta regularity score: measures how consistent the deltas/steps are (1.0 for constant ramp or constant CW)
        delta_std = float(np.std(delta)) if len(delta) > 1 else 0.0
        delta_mean_mag = float(np.mean(np.abs(delta))) if len(delta) > 0 else 0.0
        delta_regularity_score = float(np.clip(1.0 / (1.0 + (delta_std / max(delta_mean_mag, 1e-6))), 0.0, 1.0)) if len(delta) > 1 else 1.0

        stability_score = feature_stability_score

        # Confidence based on valid pair sample size and finite coverage
        coverage = num_pairs / max(n - 1, 1)
        sample_conf = min(1.0, num_pairs / 10.0)
        confidence = float(np.clip(0.5 * coverage + 0.5 * sample_conf, 0.0, 1.0))

        return PulseToPulseVariationResult(
            metric_name=metric_name,
            num_pulses=n,
            num_valid_pairs=num_pairs,
            delta=delta,
            absolute_delta=abs_delta,
            relative_delta=rel_delta,
            signed_delta=signed_delta,
            valid_mask=valid_mask,
            mean_delta=mean_d,
            std_delta=std_d,
            max_delta=max_d,
            stability_score=stability_score,
            trend_slope=trend_slope,
            confidence=confidence,
            valid=True,
            normalized_delta=norm_delta if normalize else None,
            delta_regularity_score=delta_regularity_score,
            feature_stability_score=feature_stability_score,
            step_change_indices=step_indices,
            diagnostics={
                "original_indices": orig_indices.tolist(),
                "normalized_variance": std_d / ref_scale,
                "is_circular": is_circular,
                "normalized_delta_applied": normalize,
                "missing_pairs_count": max(0, (n - 1) - num_pairs),
            },
        )


def analyze_pulse_to_pulse_variation(
    values: Any,
    timestamps_sec: Any | None = None,
    sort_by_time: bool = False,
    normalize: bool = False,
    is_circular: bool = False,
    step_threshold: float | None = None,
    metric_name: str = "generic",
    strict_adjacency: bool = True,
    **kwargs: Any,
) -> PulseToPulseVariationResult:
    """
    Convenience function to analyze pulse-to-pulse variation.

    This function does not estimate raw pulse features from IQ.
    The caller must provide precomputed values from upstream packages.
    """
    processor = PulseToPulseVariation(**kwargs)
    return processor.compute(
        values=values,
        timestamps_sec=timestamps_sec,
        sort_by_time=sort_by_time,
        normalize=normalize,
        is_circular=is_circular,
        step_threshold=step_threshold,
        metric_name=metric_name,
        strict_adjacency=strict_adjacency,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  PulseToPulseVariation: Demonstration")
    print("=" * 60)
    seq = np.array([10.0, 10.2, 9.9, 10.1, 10.0, 10.05, 9.95])
    metric = PulseToPulseVariation()
    result = metric.compute(seq)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
