from __future__ import annotations
import importlib

_pkg = importlib.import_module("packages.Time and Timing Analysis")

# Constants
UINT64_MIN = _pkg.UINT64_MIN
UINT64_MAX = _pkg.UINT64_MAX

# Models & Adapter
TimingInput = _pkg.TimingInput
build_timing_input = _pkg.build_timing_input
TOAResult = _pkg.TOAResult
SignalDurationResult = _pkg.SignalDurationResult
TimeOffsetResult = _pkg.TimeOffsetResult
InterPulseIntervalResult = _pkg.InterPulseIntervalResult
PRIResult = _pkg.PRIResult
PRFResult = _pkg.PRFResult
JitterResult = _pkg.JitterResult
BurstTimingResult = _pkg.BurstTimingResult
TimingPeriodicityResult = _pkg.TimingPeriodicityResult
PulseDensityResult = _pkg.PulseDensityResult
TimeAnalysisResult = _pkg.TimeAnalysisResult

# Classes
TimeOfArrival = _pkg.TimeOfArrival
SignalDuration = _pkg.SignalDuration
TimeOffset = _pkg.TimeOffset
InterPulseInterval = _pkg.InterPulseInterval
PulseRepetitionInterval = _pkg.PulseRepetitionInterval
PulseRepetitionFrequency = _pkg.PulseRepetitionFrequency
TimingJitter = _pkg.TimingJitter
BurstTiming = _pkg.BurstTiming
TimingPeriodicity = _pkg.TimingPeriodicity
PulseDensity = _pkg.PulseDensity
TimeAnalysisProcessor = _pkg.TimeAnalysisProcessor

# Functional APIs
compute_toa = _pkg.compute_toa
compute_signal_duration = _pkg.compute_signal_duration
compute_time_offset = _pkg.compute_time_offset
compute_inter_pulse_interval = _pkg.compute_inter_pulse_interval
compute_pri = _pkg.compute_pri
compute_prf = _pkg.compute_prf
compute_jitter = _pkg.compute_jitter
compute_burst_timing = _pkg.compute_burst_timing
compute_periodicity = _pkg.compute_periodicity
compute_pulse_density = _pkg.compute_pulse_density
analyze_time_and_timing = _pkg.analyze_time_and_timing

# Common utilities
validate_sample_rate = _pkg.validate_sample_rate
validate_uint64_timestamp = _pkg.validate_uint64_timestamp
calculate_hardware_toas_ns = _pkg.calculate_hardware_toas_ns
calculate_relative_toas_sec = _pkg.calculate_relative_toas_sec
compute_gap_aware_diff = _pkg.compute_gap_aware_diff
robust_nan_statistics = _pkg.robust_nan_statistics

__all__ = list(_pkg.__all__)

