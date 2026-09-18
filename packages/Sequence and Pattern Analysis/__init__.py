from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import numpy as np

# Models / Dataclasses
from .models import (
    PulseToPulseVariationResult,
    PRIPatternResult,
    RFFrequencyPatternResult,
    AmplitudePatternResult,
    PhasePatternResult,
    ModulationPatternResult,
    WaveformConsistencyResult,
    IntraPulseVariationResult,
    InterPulseCorrelationPatternResult,
    BurstStructureResult,
    SequencePeriodicityResult,
    PatternResult,
    SequenceSummaryResult,
)

# Base & Factory
from .base import (
    BaseSequencePatternMetric,
    SequencePatternMetricFactory,
)

# Validation & Utilities
from .validation import (
    validate_sequence_inputs,
    validate_matrix_input,
)
from ._common import (
    wrap_circular_difference,
    estimate_discrete_states,
    build_transition_matrix,
    compute_dwell_lengths,
    detect_symbolic_periodicity,
    estimate_linear_trend,
)

# Module implementations and functional APIs
from .Pulse_To_Pulse_Variation import PulseToPulseVariation, analyze_pulse_to_pulse_variation
from .PRI_Patterns import PRIPatterns, analyze_pri_pattern
from .RF_Frequency_Patterns import RFFrequencyPatterns, analyze_rf_pattern
from .Amplitude_Patterns import AmplitudePatterns, analyze_amplitude_pattern
from .Phase_Patterns import PhasePatterns, analyze_phase_pattern
from .Modulation_Patterns import ModulationPatterns, analyze_modulation_pattern
from .Waveform_Consistency import WaveformConsistency, analyze_waveform_consistency
from .Intra_Pulse_Variation import IntraPulseVariation, analyze_intra_pulse_variation
from .Inter_Pulse_Correlation import InterPulseCorrelation, analyze_inter_pulse_correlation_pattern
from .Burst_Structure import BurstStructure, analyze_burst_structure
from .Periodicity import SequencePeriodicity, analyze_sequence_periodicity


def sequence_and_pattern_summary(
    carrier_freq_hz: Optional[np.ndarray] = None,
    pri_values_sec: Optional[np.ndarray] = None,
    timestamps_sec: Optional[np.ndarray] = None,
    amplitude_values: Optional[np.ndarray] = None,
    phase_rad: Optional[np.ndarray] = None,
    modulation_labels: Optional[List[str] | np.ndarray] = None,
    frequency_states_hz: Optional[List[List[float]]] = None,
    similarity_scores: Optional[np.ndarray] = None,
    correlation_matrix: Optional[np.ndarray] = None,
    burst_labels: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> SequenceSummaryResult:
    """
    Comprehensive multi-domain sequence and pattern analysis orchestrator.
    Executes sequence pattern extraction across all provided precomputed parameter streams
    without re-evaluating raw IQ signal transformations.

    Parameters
    ----------
    carrier_freq_hz : np.ndarray, optional
        Precomputed pulse carrier frequencies in Hz (e.g. from Frequency package).
    pri_values_sec : np.ndarray, optional
        Precomputed pulse repetition intervals in seconds.
    timestamps_sec : np.ndarray, optional
        Pulse Time of Arrival (TOA) timestamps in seconds.
    amplitude_values : np.ndarray, optional
        Precomputed pulse amplitude levels in dB or V (e.g. from Amplitude package).
    phase_rad : np.ndarray, optional
        Precomputed pulse phase offsets in radians (e.g. from Phase package).
    modulation_labels : list[str] or np.ndarray, optional
        Pre-classified modulation type labels (e.g. from Modulation_Analysis package).
    frequency_states_hz : list[list[float]], optional
        Pre-extracted discrete frequency states per pulse.
    similarity_scores : np.ndarray, optional
        Consecutive pairwise waveform similarity scores (e.g. from Correlation_and_Similarity).
    correlation_matrix : np.ndarray, optional
        N x N pairwise inter-pulse correlation matrix.
    burst_labels : np.ndarray, optional
        Burst grouping cluster assignments (e.g. from Clustering_and_Association).

    Returns
    -------
    SequenceSummaryResult
    """
    pri_res = None
    if pri_values_sec is not None or timestamps_sec is not None:
        pri_res = analyze_pri_pattern(pri_values_sec=pri_values_sec, timestamps_sec=timestamps_sec)

    rf_res = None
    if carrier_freq_hz is not None:
        rf_res = analyze_rf_pattern(carrier_freq_hz=carrier_freq_hz, timestamps_sec=timestamps_sec)

    amp_res = None
    if amplitude_values is not None:
        amp_res = analyze_amplitude_pattern(amplitude_values=amplitude_values, timestamps_sec=timestamps_sec)

    phase_res = None
    if phase_rad is not None:
        phase_res = analyze_phase_pattern(phase_rad=phase_rad, timestamps_sec=timestamps_sec)

    mod_res = None
    if modulation_labels is not None:
        mod_res = analyze_modulation_pattern(
            modulation_labels=modulation_labels,
            frequency_states_hz=frequency_states_hz,
            timestamps_sec=timestamps_sec,
        )

    wf_res = None
    if similarity_scores is not None or correlation_matrix is not None:
        wf_ts = None
        if timestamps_sec is not None:
            if similarity_scores is not None and len(timestamps_sec) == len(similarity_scores):
                wf_ts = timestamps_sec
            elif correlation_matrix is not None and hasattr(correlation_matrix, "shape") and len(timestamps_sec) == correlation_matrix.shape[0]:
                wf_ts = timestamps_sec
        wf_res = analyze_waveform_consistency(
            similarity_scores=similarity_scores,
            similarity_matrix=kwargs.get("similarity_matrix", correlation_matrix),
            correlation_matrix=correlation_matrix,
            timestamps_sec=wf_ts,
        )

    burst_res = None
    burst_starts = kwargs.get("burst_start_times_sec", None)
    if burst_labels is not None or burst_starts is not None:
        burst_res = analyze_burst_structure(
            burst_labels=burst_labels,
            timestamps_sec=timestamps_sec,
            burst_start_times_sec=burst_starts,
            burst_durations_sec=kwargs.get("burst_durations_sec", None),
            burst_pulse_counts=kwargs.get("burst_pulse_counts", None),
        )

    # Determine pulse counts per active domain, respecting interval/pair semantics (K -> K+1)
    domain_pulse_counts: dict[str, int] = {}
    if pri_res is not None:
        domain_pulse_counts["pri"] = pri_res.num_pulses
    elif pri_values_sec is not None:
        domain_pulse_counts["pri"] = len(pri_values_sec) + 1 if len(pri_values_sec) > 0 else 0
    elif timestamps_sec is not None:
        domain_pulse_counts["timestamps"] = len(timestamps_sec)

    if rf_res is not None:
        domain_pulse_counts["rf"] = rf_res.num_pulses
    elif carrier_freq_hz is not None:
        domain_pulse_counts["rf"] = len(carrier_freq_hz)

    if amp_res is not None:
        domain_pulse_counts["amplitude"] = amp_res.num_pulses
    elif amplitude_values is not None:
        domain_pulse_counts["amplitude"] = len(amplitude_values)

    if phase_res is not None:
        domain_pulse_counts["phase"] = phase_res.num_pulses
    elif phase_rad is not None:
        domain_pulse_counts["phase"] = len(phase_rad)

    if mod_res is not None:
        domain_pulse_counts["modulation"] = mod_res.num_pulses
    elif modulation_labels is not None:
        domain_pulse_counts["modulation"] = len(modulation_labels)

    if wf_res is not None:
        domain_pulse_counts["waveform_consistency"] = wf_res.num_pulses
    elif similarity_scores is not None:
        domain_pulse_counts["waveform_consistency"] = len(similarity_scores) + 1 if len(similarity_scores) > 0 else 0

    if burst_res is not None and burst_res.total_pulses > 0:
        domain_pulse_counts["burst"] = burst_res.total_pulses
    elif burst_labels is not None:
        domain_pulse_counts["burst"] = len(burst_labels)

    counts_list = list(domain_pulse_counts.values())
    if not counts_list:
        num_pulses = 0
        pulse_count_mismatch = False
    else:
        unique_counts = set(counts_list)
        pulse_count_mismatch = len(unique_counts) > 1
        if not pulse_count_mismatch:
            num_pulses = counts_list[0]
        else:
            # Multi-domain pulse count consensus: use mode/majority; if tied, prefer direct pulse domain or max
            from collections import Counter
            counter = Counter(counts_list)
            most_common = counter.most_common()
            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                direct_domains = ["rf", "amplitude", "phase", "modulation", "timestamps", "burst"]
                direct_counts = [domain_pulse_counts[d] for d in direct_domains if d in domain_pulse_counts]
                num_pulses = direct_counts[0] if direct_counts else max(unique_counts)
            else:
                num_pulses = most_common[0][0]

    all_analyses = [pri_res, rf_res, amp_res, phase_res, mod_res, wf_res, burst_res]
    attempted_count = sum(1 for r in all_analyses if r is not None)

    # Compute overall summary confidence and stability
    confidences = []
    stabilities = []
    for r in all_analyses:
        if r is not None and getattr(r, "valid", False):
            confidences.append(r.confidence)
            if hasattr(r, "stability_score") and r.stability_score is not None:
                stabilities.append(r.stability_score)
            elif hasattr(r, "phase_stability"):
                stabilities.append(r.phase_stability)
            elif hasattr(r, "burst_stability_score"):
                stabilities.append(r.burst_stability_score)
            elif hasattr(r, "temporal_stability"):
                stabilities.append(r.temporal_stability)

    overall_conf = float(np.mean(confidences)) if confidences else 0.0
    overall_stab = float(np.mean(stabilities)) if stabilities else 0.0

    return SequenceSummaryResult(
        num_pulses=num_pulses,
        pri_pattern=pri_res,
        rf_pattern=rf_res,
        amplitude_pattern=amp_res,
        phase_pattern=phase_res,
        modulation_pattern=mod_res,
        waveform_consistency=wf_res,
        burst_structure=burst_res,
        overall_confidence=overall_conf,
        overall_stability=overall_stab,
        attempted_analyses=attempted_count,
        diagnostics={
            "active_analyses": len(confidences),
            "attempted_analyses": attempted_count,
            "pulse_count_mismatch": pulse_count_mismatch,
            "domain_pulse_counts": domain_pulse_counts,
        },
    )


__all__ = [
    # Models
    "PulseToPulseVariationResult",
    "PRIPatternResult",
    "RFFrequencyPatternResult",
    "AmplitudePatternResult",
    "PhasePatternResult",
    "ModulationPatternResult",
    "WaveformConsistencyResult",
    "IntraPulseVariationResult",
    "InterPulseCorrelationPatternResult",
    "BurstStructureResult",
    "SequencePeriodicityResult",
    "PatternResult",
    "SequenceSummaryResult",
    # Base / Factory
    "BaseSequencePatternMetric",
    "SequencePatternMetricFactory",
    # Validation / Common
    "validate_sequence_inputs",
    "validate_matrix_input",
    "wrap_circular_difference",
    "estimate_discrete_states",
    "build_transition_matrix",
    "compute_dwell_lengths",
    "detect_symbolic_periodicity",
    "estimate_linear_trend",
    # Metric classes
    "PulseToPulseVariation",
    "PRIPatterns",
    "RFFrequencyPatterns",
    "AmplitudePatterns",
    "PhasePatterns",
    "ModulationPatterns",
    "WaveformConsistency",
    "IntraPulseVariation",
    "InterPulseCorrelation",
    "BurstStructure",
    "SequencePeriodicity",
    # Functional APIs
    "analyze_pulse_to_pulse_variation",
    "analyze_pri_pattern",
    "analyze_rf_pattern",
    "analyze_amplitude_pattern",
    "analyze_phase_pattern",
    "analyze_modulation_pattern",
    "analyze_waveform_consistency",
    "analyze_intra_pulse_variation",
    "analyze_inter_pulse_correlation_pattern",
    "analyze_burst_structure",
    "analyze_sequence_periodicity",
    "sequence_and_pattern_summary",
]
