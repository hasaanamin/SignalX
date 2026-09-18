from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class PulseToPulseVariationResult:
    """
    Result of pulse-to-pulse variation analysis across sequential pulse measurements.
    """
    metric_name: str
    num_pulses: int
    num_valid_pairs: int
    delta: np.ndarray
    absolute_delta: np.ndarray
    relative_delta: np.ndarray
    signed_delta: np.ndarray
    valid_mask: np.ndarray
    mean_delta: float
    std_delta: float
    max_delta: float
    stability_score: float
    trend_slope: float
    confidence: float
    valid: bool
    normalized_delta: Optional[np.ndarray] = None
    delta_regularity_score: float = 0.0
    feature_stability_score: float = 0.0
    step_change_indices: list[int] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        return self.num_pulses

    @property
    def var_delta(self) -> float:
        return float(self.std_delta ** 2)

    @property
    def variation_metric(self) -> float:
        return self.mean_delta



@dataclass
class PRIPatternResult:
    """
    Result of Pulse Repetition Interval (PRI) sequence pattern analysis.
    """
    pattern_type: str  # constant, jittered, staggered, dwell_and_switch, sliding, agile, unresolved
    num_pulses: int
    pri_states_sec: list[float]
    state_sequence: np.ndarray
    transition_matrix: np.ndarray
    dwell_lengths: list[int]
    trend_slope_sec_per_pulse: float
    mean_pri_sec: float
    std_pri_sec: float
    jitter_percent: float
    repetition_length: Optional[int]
    stagger_levels: int
    stability_score: float
    confidence: float
    valid: bool
    num_pri_samples: int = 0
    num_valid_pri_samples: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def num_states(self) -> int:
        return len(self.pri_states_sec)

    @property
    def stagger_levels_sec(self) -> list[float]:
        return self.pri_states_sec

    @property
    def stagger_frame_length(self) -> Optional[int]:
        return self.repetition_length

    @property
    def pri_jitter_percent(self) -> float:
        return self.jitter_percent

    @property
    def pri_trend(self) -> float:
        return self.trend_slope_sec_per_pulse


@dataclass
class RFFrequencyPatternResult:
    """
    Result of RF / carrier frequency sequence pattern analysis.
    """
    pattern_type: str  # fixed, stable, frequency_hopping, dwell_and_switch, repeated_hop, linear_agile, sliding_trend, agile, discrete_states, unresolved
    num_pulses: int
    frequency_states_hz: list[float]
    state_sequence: np.ndarray
    transition_matrix: np.ndarray
    dwell_lengths: list[int]
    hop_sizes_hz: list[float]
    repetition_length: Optional[int]
    trend_slope_hz_per_pulse: float
    mean_frequency_hz: float
    std_frequency_hz: float
    stability_score: float
    confidence: float
    valid: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def num_states(self) -> int:
        return len(self.frequency_states_hz)

    @property
    def frequency_channels_hz(self) -> list[float]:
        return self.frequency_states_hz

    @property
    def agility_span_hz(self) -> float:
        if self.frequency_states_hz and len(self.frequency_states_hz) > 1:
            return float(max(self.frequency_states_hz) - min(self.frequency_states_hz))
        return 0.0

    @property
    def frequency_trend_hz_per_sec(self) -> float:
        return self.trend_slope_hz_per_pulse


@dataclass
class AmplitudePatternResult:
    """
    Result of amplitude/power sequence pattern analysis.
    """
    pattern_type: str  # stable_flat, stable, discrete_levels, switching, decaying, trending, periodic_scan, conical_scan, stepped_levels, fluctuating, irregular, unresolved
    num_pulses: int
    amplitude_levels: list[float]
    state_sequence: np.ndarray
    transition_matrix: np.ndarray
    dwell_lengths: list[int]
    trend_slope_per_pulse: float
    mean_amplitude: float
    std_amplitude: float
    dynamic_range: float
    stability_score: float
    confidence: float
    valid: bool
    unit: str = "dB"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def num_states(self) -> int:
        return len(self.amplitude_levels)

    @property
    def modulation_depth_db(self) -> float:
        return self.dynamic_range

    @property
    def amplitude_stability_score(self) -> float:
        return self.stability_score

    @property
    def amplitude_trend(self) -> float:
        return self.trend_slope_per_pulse


@dataclass
class PhasePatternResult:
    """
    Result of inter-pulse phase sequence pattern analysis with circular phase differences.
    """
    pattern_type: str  # coherent_stable, linear_ramp, linear_progression, discrete_biphase, polyphase_code, discrete_phases, switching, repeating_code, random_incoherent, random_phase, unresolved
    num_pulses: int
    circular_deltas_rad: np.ndarray
    mean_circular_delta_rad: float
    circular_variance: float
    phase_states_rad: list[float]
    state_sequence: np.ndarray
    transition_matrix: np.ndarray
    repetition_length: Optional[int]
    phase_stability_score: float
    confidence: float
    valid: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def num_states(self) -> int:
        return len(self.phase_states_rad)

    @property
    def phase_stability(self) -> float:
        return self.phase_stability_score


@dataclass
class ModulationPatternResult:
    """
    Result of modulation classification sequence pattern analysis.
    """
    pattern_type: str  # constant_mode, stable_single_mode, switching, mode_switching, dwell_and_switch, repeating_code, periodic_schedule, adaptive_rate, irregular, unresolved
    num_pulses: int
    unique_modulations: list[str]
    modulation_sequence: list[str]
    dwell_lengths: list[int]
    transition_matrix: np.ndarray
    repetition_length: Optional[int]
    consistency_score: float
    confidence: float
    valid: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def num_unique_modulations(self) -> int:
        return len(self.unique_modulations)

    @property
    def code_period(self) -> Optional[int]:
        return self.repetition_length

    @property
    def mode_transition_count(self) -> int:
        if not self.modulation_sequence or len(self.modulation_sequence) < 2:
            return 0
        return sum(1 for i in range(len(self.modulation_sequence) - 1) if self.modulation_sequence[i] != self.modulation_sequence[i + 1])

    @property
    def upstream_mean_confidence(self) -> Optional[float]:
        return self.diagnostics.get("upstream_mean_confidence")

    @property
    def pattern_confidence(self) -> Optional[float]:
        return self.diagnostics.get("pattern_confidence")


@dataclass
class WaveformConsistencyResult:
    """
    Result of temporal waveform similarity and consistency analysis.
    """
    num_pulses: int
    mean_similarity: float
    std_similarity: float
    min_similarity: float
    consistency_score: float
    temporal_stability: float
    change_point_indices: list[int]
    epoch_labels: np.ndarray
    num_epochs: int
    confidence: float
    valid: bool
    num_similarity_pairs: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def change_point_count(self) -> int:
        return len(self.change_point_indices)


@dataclass
class IntraPulseVariationResult:
    """
    Result of variation analysis across instantaneous frequency, phase, or amplitude within a pulse.
    """
    num_samples: int
    variation_magnitude: float
    trend_slope: float
    smoothness_score: float
    is_piecewise_linear: bool
    intra_pulse_stability: float
    change_point_indices: list[int]
    max_gradient: float
    mean_gradient: float
    confidence: float
    valid: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterPulseCorrelationPatternResult:
    """
    Result of inter-pulse correlation matrix pattern evolution analysis.
    """
    pattern_type: str  # stable_high, gradual_decorrelation, alternating_bimodal, family_grouped, abrupt_step, decorrelated_noise, unresolved
    num_pulses: int
    mean_inter_pulse_correlation: float
    min_inter_pulse_correlation: float
    decorrelation_length: Optional[int]
    correlation_trend: float
    family_clusters: np.ndarray
    num_families: int
    transition_count: int
    stability_score: float
    confidence: float
    valid: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class BurstStructureResult:
    """
    Result of higher-level burst structure and interval pattern analysis.
    """
    pattern_type: str  # regular, jittered, staggered, irregular, unresolved
    num_bursts: int
    total_pulses: int
    mean_pulses_per_burst: float
    std_pulses_per_burst: float
    min_pulses_per_burst: int
    max_pulses_per_burst: int
    mean_burst_duration_sec: float
    std_burst_duration_sec: float
    mean_ibi_sec: float
    std_ibi_sec: float
    ibi_jitter_percent: float
    burst_repetition_rate_hz: float
    burst_stability_score: float
    confidence: float
    valid: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class SequencePeriodicityResult:
    """
    Result of symbolic / parameter sequence periodicity and motif analysis.
    """
    is_periodic: bool
    period_length: Optional[int]
    repeating_motif: list[Any]
    num_cycles: float
    periodic_fidelity: float
    harmonic_periods: list[int]
    confidence: float
    valid: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternResult:
    """
    Generic polymorphic container for sequence pattern evaluation.
    """
    valid: bool
    confidence: float
    pattern_type: Optional[str]
    num_samples: int
    state_values: list[Any]
    state_labels: np.ndarray
    transition_matrix: np.ndarray
    dwell_lengths: list[int]
    repetition_length: Optional[int]
    trend: float
    stability_score: Optional[float]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class SequenceSummaryResult:
    """
    Comprehensive multi-domain summary combining sequence analyses.
    """
    num_pulses: int
    pri_pattern: Optional[PRIPatternResult] = None
    rf_pattern: Optional[RFFrequencyPatternResult] = None
    amplitude_pattern: Optional[AmplitudePatternResult] = None
    phase_pattern: Optional[PhasePatternResult] = None
    modulation_pattern: Optional[ModulationPatternResult] = None
    waveform_consistency: Optional[WaveformConsistencyResult] = None
    burst_structure: Optional[BurstStructureResult] = None
    overall_confidence: float = 0.0
    overall_stability: float = 0.0
    attempted_analyses: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def active_analyses(self) -> int:
        return sum(
            1 for p in [
                self.pri_pattern,
                self.rf_pattern,
                self.amplitude_pattern,
                self.phase_pattern,
                self.modulation_pattern,
                self.waveform_consistency,
                self.burst_structure,
            ] if (p is not None and getattr(p, "valid", False))
        )

    @property
    def pulse_count_mismatch(self) -> bool:
        return bool(self.diagnostics.get("pulse_count_mismatch", False))

    @property
    def domain_pulse_counts(self) -> dict[str, int]:
        return self.diagnostics.get("domain_pulse_counts", {})

