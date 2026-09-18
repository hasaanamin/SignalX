from __future__ import annotations
import importlib

_pkg = importlib.import_module("packages.Sequence and Pattern Analysis")

# Dataclasses / Models
PulseToPulseVariationResult = _pkg.PulseToPulseVariationResult
PRIPatternResult = _pkg.PRIPatternResult
RFFrequencyPatternResult = _pkg.RFFrequencyPatternResult
AmplitudePatternResult = _pkg.AmplitudePatternResult
PhasePatternResult = _pkg.PhasePatternResult
ModulationPatternResult = _pkg.ModulationPatternResult
WaveformConsistencyResult = _pkg.WaveformConsistencyResult
IntraPulseVariationResult = _pkg.IntraPulseVariationResult
InterPulseCorrelationPatternResult = _pkg.InterPulseCorrelationPatternResult
BurstStructureResult = _pkg.BurstStructureResult
SequencePeriodicityResult = _pkg.SequencePeriodicityResult
PatternResult = _pkg.PatternResult
SequenceSummaryResult = _pkg.SequenceSummaryResult

# Base & Factory
BaseSequencePatternMetric = _pkg.BaseSequencePatternMetric
SequencePatternMetricFactory = _pkg.SequencePatternMetricFactory

# Metric Classes
PulseToPulseVariation = _pkg.PulseToPulseVariation
PRIPatterns = _pkg.PRIPatterns
RFFrequencyPatterns = _pkg.RFFrequencyPatterns
AmplitudePatterns = _pkg.AmplitudePatterns
PhasePatterns = _pkg.PhasePatterns
ModulationPatterns = _pkg.ModulationPatterns
WaveformConsistency = _pkg.WaveformConsistency
IntraPulseVariation = _pkg.IntraPulseVariation
InterPulseCorrelation = _pkg.InterPulseCorrelation
BurstStructure = _pkg.BurstStructure
SequencePeriodicity = _pkg.SequencePeriodicity

# Functional APIs
analyze_pulse_to_pulse_variation = _pkg.analyze_pulse_to_pulse_variation
analyze_pri_pattern = _pkg.analyze_pri_pattern
analyze_rf_pattern = _pkg.analyze_rf_pattern
analyze_amplitude_pattern = _pkg.analyze_amplitude_pattern
analyze_phase_pattern = _pkg.analyze_phase_pattern
analyze_modulation_pattern = _pkg.analyze_modulation_pattern
analyze_waveform_consistency = _pkg.analyze_waveform_consistency
analyze_intra_pulse_variation = _pkg.analyze_intra_pulse_variation
analyze_inter_pulse_correlation_pattern = _pkg.analyze_inter_pulse_correlation_pattern
analyze_burst_structure = _pkg.analyze_burst_structure
analyze_sequence_periodicity = _pkg.analyze_sequence_periodicity
sequence_and_pattern_summary = _pkg.sequence_and_pattern_summary

# Common / Validation Helpers
validate_sequence_inputs = _pkg.validate_sequence_inputs
validate_matrix_input = _pkg.validate_matrix_input
wrap_circular_difference = _pkg.wrap_circular_difference
estimate_discrete_states = _pkg.estimate_discrete_states
build_transition_matrix = _pkg.build_transition_matrix
compute_dwell_lengths = _pkg.compute_dwell_lengths
detect_symbolic_periodicity = _pkg.detect_symbolic_periodicity
estimate_linear_trend = _pkg.estimate_linear_trend

__all__ = list(_pkg.__all__)
