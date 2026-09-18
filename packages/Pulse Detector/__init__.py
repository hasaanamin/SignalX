from __future__ import annotations
from typing import Any
import math
import warnings
import numpy as np
import importlib
from pathlib import Path

# 1. Base interface & Factory & Data Structures
from .base import BasePulseDetectorStage, PulseDetectorFactory, PulseCandidate
from . import _common
from ._common import _validate_sample_rate

# 2. Individual Module Imports
EnvelopeExtractionStage = importlib.import_module(".Envelope Extraction", package=__name__).EnvelopeExtractionStage
extract_envelope = importlib.import_module(".Envelope Extraction", package=__name__).envelope_extraction

ThresholdDetectionStage = importlib.import_module(".Threshold Detection", package=__name__).ThresholdDetectionStage
detect_threshold = importlib.import_module(".Threshold Detection", package=__name__).threshold_detection

PulseDetectionStage = importlib.import_module(".Pulse Detection", package=__name__).PulseDetectionStage
detect_pulse_state = importlib.import_module(".Pulse Detection", package=__name__).detect_pulse_state

PulseBoundariesStage = importlib.import_module(".Pulse Boundaries", package=__name__).PulseBoundariesStage
find_pulse_boundaries = importlib.import_module(".Pulse Boundaries", package=__name__).find_pulse_boundaries

LeadingTrailingEdgeDetectionStage = importlib.import_module(".Leading Trailing Edge Detection", package=__name__).LeadingTrailingEdgeDetectionStage
refine_edges = importlib.import_module(".Leading Trailing Edge Detection", package=__name__).refine_edges

PulseWidthStage = importlib.import_module(".Pulse Width", package=__name__).PulseWidthStage
evaluate_pulse_width = importlib.import_module(".Pulse Width", package=__name__).evaluate_pulse_width

PulseSegmentationStage = importlib.import_module(".Pulse Segmentation", package=__name__).PulseSegmentationStage
segment_pulses = importlib.import_module(".Pulse Segmentation", package=__name__).segment_pulses

StreamingPulseDetector = importlib.import_module(".streaming", package=__name__).StreamingPulseDetector

_KNOWN_CANONICAL_KWARGS = {
    "sample_rate", "envelope_mode", "smooth_len",
    "threshold_window", "k_high", "k_low",
    "edge_search_window", "pw_min", "pw_max",
    "pw_min_samples", "pw_max_samples", "min_rise_slope",
    "background_window_sec", "include_rejected",
    "as_dict", "sample_offset"
}


def detect_pulse_candidates(
    iq_data: Any,
    sample_rate: float | None = None,
    envelope_mode: str = "power",
    smooth_len: int = 1,
    threshold_window: int = 101,
    k_high: float = 5.0,
    k_low: float = 3.0,
    edge_search_window: int = 8,
    pw_min: float | None = None,
    pw_max: float | None = None,
    pw_min_samples: int = 3,
    pw_max_samples: int | None = None,
    start_confirm_samples: int = 2,
    end_confirm_samples: int = 5,
    min_rise_slope: float = 0.0,
    background_window_sec: float = 10e-6,
    include_rejected: bool = False,
    as_dict: bool = False,
    sample_offset: int = 0,
    **kwargs: Any
) -> list[PulseCandidate] | dict[str, Any]:
    """
    End-to-End Pulse Detector Pipeline.
    Executes the exact 7-stage chain:
      Conditioned IQ -> Envelope -> Threshold -> Pulse State -> Boundaries -> Edges -> Width -> Segmentation
    Yields validated Pulse Candidates and segregates transient/pulsed activity from continuous RF signals.

    Parameters:
        iq_data: Complex IQ array or dictionary/tuple.
        sample_rate (Fs): Sampling rate in Hz (must be explicitly provided, finite, and > 0).
        envelope_mode: Must be 'power' (|x|^2).
        smooth_len (L): Causal moving average envelope smoothing window.
        threshold_window (W): Moving median/MAD window length for local noise floor.
        k_high (KH): High threshold multiplier (starts pulse).
        k_low (KL): Low threshold multiplier (ends pulse).
        edge_search_window (W_edge): Neighborhood search window for dP/dt edge refinement.
        pw_min: Minimum pulse width in seconds (or None).
        pw_max: Maximum pulse width in seconds (or None).
        pw_min_samples: Minimum pulse width in samples (default: 3).
        pw_max_samples: Maximum pulse width in samples (or None).
        start_confirm_samples: Number of consecutive samples > TH required to enter pulse (default: 2).
        end_confirm_samples: Number of consecutive samples < TL required to exit pulse (default: 5).
        min_rise_slope: Minimum leading edge slope S_min in W/s.
        background_window_sec: Pre/post isolation duration in seconds.
        include_rejected: If True, returns rejected events flagged as is_valid_pulse=False.
        as_dict: If True, returns comprehensive dictionary containing all intermediate stage arrays.
        sample_offset: Global sample index offset for streaming TOA computation.

    Returns:
        List of segmented PulseCandidate objects (or dictionary if as_dict=True).
    """
    # BUG 7 Fix: Reject unknown kwargs / broken aliases
    if kwargs:
        unknown = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unknown pulse detector parameters: {unknown}")

    # BUG 6 Fix: Validate sample rate strictly (no silent 1 Hz default)
    if sample_rate is None:
        raise ValueError("sample_rate must be explicitly provided.")
    fs = _validate_sample_rate(sample_rate)

    # BUG 8 Fix: Validate all configuration settings
    mode = str(envelope_mode).strip().lower()
    if mode not in {"power"}:
        raise ValueError(f"Unsupported envelope mode: '{mode}'. Pulse detector requires envelope_mode='power'.")

    smooth_len = int(smooth_len)
    if smooth_len < 1:
        raise ValueError(f"smooth_len must be >= 1, got {smooth_len}")

    threshold_window = int(threshold_window)
    if threshold_window < 1:
        raise ValueError(f"threshold_window must be >= 1, got {threshold_window}")

    edge_search_window = int(edge_search_window)
    if edge_search_window < 1:
        raise ValueError(f"edge_search_window must be >= 1, got {edge_search_window}")

    pw_min_samples = int(pw_min_samples)
    if pw_min_samples < 1:
        raise ValueError(f"pw_min_samples must be >= 1, got {pw_min_samples}")

    start_confirm_samples = int(start_confirm_samples)
    if start_confirm_samples < 1:
        raise ValueError(f"start_confirm_samples must be >= 1, got {start_confirm_samples}")

    end_confirm_samples = int(end_confirm_samples)
    if end_confirm_samples < 1:
        raise ValueError(f"end_confirm_samples must be >= 1, got {end_confirm_samples}")

    k_high = float(k_high)
    k_low = float(k_low)
    if not np.isfinite(k_high) or not np.isfinite(k_low):
        raise ValueError("Threshold multipliers must be finite")
    if k_low < 0:
        raise ValueError(f"k_low must be >= 0, got {k_low}")
    if k_high <= k_low:
        raise ValueError(f"k_high ({k_high}) must be greater than k_low ({k_low})")

    if pw_min is not None:
        pw_min = float(pw_min)
        if pw_min <= 0:
            raise ValueError(f"pw_min must be > 0, got {pw_min}")

    if pw_max is not None:
        pw_max = float(pw_max)
        if pw_max <= 0:
            raise ValueError(f"pw_max must be > 0, got {pw_max}")

    if pw_min is not None and pw_max is not None and pw_max <= pw_min:
        raise ValueError(f"pw_max ({pw_max}) must be greater than pw_min ({pw_min})")

    if pw_max_samples is not None:
        pw_max_samples = int(pw_max_samples)
        if pw_max_samples < pw_min_samples:
            raise ValueError(f"pw_max_samples ({pw_max_samples}) must be >= pw_min_samples ({pw_min_samples})")

    min_rise_slope = float(min_rise_slope)
    if min_rise_slope < 0:
        raise ValueError(f"min_rise_slope must be >= 0, got {min_rise_slope}")

    background_window_sec = float(background_window_sec)
    if background_window_sec <= 0:
        raise ValueError(f"background_window_sec must be > 0, got {background_window_sec}")

    # BUG 10 Fix: Guard warning if threshold window is too short relative to max expected pulse width
    effective_max_samples = pw_max_samples
    if effective_max_samples is None and pw_max is not None:
        effective_max_samples = int(math.floor(pw_max * fs))
    if effective_max_samples is not None and threshold_window <= 2 * effective_max_samples:
        warnings.warn(
            f"threshold_window ({threshold_window}) is too short relative to maximum expected pulse width ({effective_max_samples} samples); "
            f"pulses may contaminate the background estimate.",
            UserWarning,
            stacklevel=2,
        )

    # 1. Envelope Extraction
    stage_env = EnvelopeExtractionStage(mode=mode, smooth_len=smooth_len)
    env_res = stage_env.compute(iq_data, as_dict=True)
    p_env = env_res["envelope"]
    raw_power = env_res["raw_power"]

    # 2. Threshold Detection
    stage_th = ThresholdDetectionStage(window_len=threshold_window, k_high=k_high, k_low=k_low)
    th_dict = stage_th.compute(p_env)
    th_high = th_dict["th_high"]
    th_low = th_dict["th_low"]
    noise_floor = th_dict["noise_floor"]
    sigma_n = th_dict["sigma_n"]

    # 3. Pulse Detection (Hysteresis State Machine with Debouncing)
    stage_state = PulseDetectionStage(
        start_confirm_samples=start_confirm_samples,
        end_confirm_samples=end_confirm_samples,
    )
    pulse_mask = stage_state.compute(
        p_env,
        th_high=th_high,
        th_low=th_low,
        start_confirm_samples=start_confirm_samples,
        end_confirm_samples=end_confirm_samples,
    )


    # 4. Pulse Boundaries
    stage_bound = PulseBoundariesStage()
    boundaries = stage_bound.compute(pulse_mask, final_chunk=True)

    # 5. Leading / Trailing Edge Detection (using raw_power for derivative refinement)
    stage_edges = LeadingTrailingEdgeDetectionStage(search_window=edge_search_window, sample_rate=fs)
    refined_edges = stage_edges.compute(p_env, boundaries=boundaries, raw_power=raw_power)

    # 6. Pulse Width & Shape Gating
    stage_width = PulseWidthStage(
        sample_rate=fs,
        pw_min=pw_min,
        pw_max=pw_max,
        pw_min_samples=pw_min_samples,
        pw_max_samples=pw_max_samples,
        min_rise_slope=min_rise_slope,
        background_window_sec=background_window_sec,
    )
    validated_events = stage_width.compute(
        refined_edges,
        sample_rate=fs,
        envelope=p_env,
        raw_power=raw_power,
        noise_floor=noise_floor,
        sigma_n=sigma_n,
        smooth_len=smooth_len,
    )

    # 7. Pulse Segmentation
    stage_seg = PulseSegmentationStage(sample_rate=fs, include_rejected=include_rejected)
    candidates = stage_seg.compute(
        iq_data,
        validated_events=validated_events,
        sample_rate=fs,
        envelope=p_env,
        noise_floor=noise_floor,
        sample_offset=sample_offset,
    )

    if as_dict:
        return {
            "candidates": candidates,
            "candidate_dicts": [c.as_dict() for c in candidates],
            "num_candidates": len(candidates),
            "num_valid_candidates": len([c for c in candidates if c.is_valid_pulse]),
            "envelope": p_env,
            "raw_power": raw_power,
            "th_high": th_high,
            "th_low": th_low,
            "noise_floor": noise_floor,
            "sigma_n": sigma_n,
            "pulse_mask": pulse_mask,
            "preliminary_boundaries": boundaries,
            "refined_edges": refined_edges,
            "validated_events": validated_events,
            "sample_rate": fs,
        }

    return candidates


def pulse_detector_summary(iq_data: Any, sample_rate: float, **kwargs: Any) -> dict[str, Any]:
    """Provides a high-level summary of pulse detector candidate extraction."""
    return detect_pulse_candidates(iq_data, sample_rate=sample_rate, as_dict=True, **kwargs)


__all__ = [
    # Base & Factory
    "BasePulseDetectorStage",
    "PulseDetectorFactory",
    "PulseCandidate",
    # Stage 1
    "EnvelopeExtractionStage",
    "extract_envelope",
    # Stage 2
    "ThresholdDetectionStage",
    "detect_threshold",
    # Stage 3
    "PulseDetectionStage",
    "detect_pulse_state",
    # Stage 4
    "PulseBoundariesStage",
    "find_pulse_boundaries",
    # Stage 5
    "LeadingTrailingEdgeDetectionStage",
    "refine_edges",
    # Stage 6
    "PulseWidthStage",
    "evaluate_pulse_width",
    # Stage 7
    "PulseSegmentationStage",
    "segment_pulses",
    # Streaming
    "StreamingPulseDetector",
    # End-to-End Pipeline
    "detect_pulse_candidates",
    "pulse_detector_summary",
]
