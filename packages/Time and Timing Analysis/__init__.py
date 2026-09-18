from __future__ import annotations
"""
Time and Timing Analysis Package.

Provides high-precision timing, TOA, pulse duration, time offset, IPI, PRI, PRF,
jitter, burst timing, periodicity, and pulse density analysis with strict
modular decoupling, genuine hardware timestamp nanosecond support, and zero
cross-module import dependencies.
"""

from .models import (
    UINT64_MIN,
    UINT64_MAX,
    TimingInput,
    TOAResult,
    SignalDurationResult,
    TimeOffsetResult,
    InterPulseIntervalResult,
    PRIResult,
    PRFResult,
    JitterResult,
    BurstTimingResult,
    TimingPeriodicityResult,
    PulseDensityResult,
    TimeAnalysisResult,
    build_timing_input,
)

from ._common import (
    validate_sample_rate,
    validate_uint64_timestamp,
    calculate_hardware_toas_ns,
    calculate_relative_toas_sec,
    compute_gap_aware_diff,
    robust_nan_statistics,
)

from .TOA import TimeOfArrival, compute_toa
from .Signal_Duration import SignalDuration, compute_signal_duration
from .Time_Offset import TimeOffset, compute_time_offset
from .Inter_Pulse_Interval import InterPulseInterval, compute_inter_pulse_interval
from .PRI import PulseRepetitionInterval, compute_pri
from .PRF import PulseRepetitionFrequency, compute_prf
from .Jitter import TimingJitter, compute_jitter
from .Burst_Timing import BurstTiming, compute_burst_timing
from .Periodicity import TimingPeriodicity, compute_periodicity
from .Pulse_Density import PulseDensity, compute_pulse_density

from .processor import TimeAnalysisProcessor
from .main import analyze_time_and_timing

__all__ = [
    # Constants
    "UINT64_MIN",
    "UINT64_MAX",
    # Models & Adapter
    "TimingInput",
    "build_timing_input",
    "TOAResult",
    "SignalDurationResult",
    "TimeOffsetResult",
    "InterPulseIntervalResult",
    "PRIResult",
    "PRFResult",
    "JitterResult",
    "BurstTimingResult",
    "TimingPeriodicityResult",
    "PulseDensityResult",
    "TimeAnalysisResult",
    # Classes
    "TimeOfArrival",
    "SignalDuration",
    "TimeOffset",
    "InterPulseInterval",
    "PulseRepetitionInterval",
    "PulseRepetitionFrequency",
    "TimingJitter",
    "BurstTiming",
    "TimingPeriodicity",
    "PulseDensity",
    "TimeAnalysisProcessor",
    # Functions
    "compute_toa",
    "compute_signal_duration",
    "compute_time_offset",
    "compute_inter_pulse_interval",
    "compute_pri",
    "compute_prf",
    "compute_jitter",
    "compute_burst_timing",
    "compute_periodicity",
    "compute_pulse_density",
    "analyze_time_and_timing",
    # Common Utilities
    "validate_sample_rate",
    "validate_uint64_timestamp",
    "calculate_hardware_toas_ns",
    "calculate_relative_toas_sec",
    "compute_gap_aware_diff",
    "robust_nan_statistics",
]

