"""Stateless DSP primitives for RF signal processing.

This package provides pure, stateless mathematical transformations operating on
complex baseband IQ sequences. It contains canonical implementations for envelope,
instantaneous power, phase, instantaneous frequency, Fourier transform, Power Spectral
Density (PSD), and autocorrelation.
"""

from signal_processing.transforms.autocorrelation import compute_autocorrelation
from signal_processing.transforms.envelope import compute_envelope
from signal_processing.transforms.exceptions import TransformError
from signal_processing.transforms.fft import compute_fft
from signal_processing.transforms.frequency import compute_instantaneous_frequency
from signal_processing.transforms.phase import compute_phase
from signal_processing.transforms.power import (
    compute_mean_power_dbfs,
    compute_mean_power_linear,
    compute_power,
    linear_power_to_dbfs,
)
from signal_processing.transforms.psd import (
    compute_psd,
    compute_psd_from_fft,
    compute_welch_psd,
)
from signal_processing.transforms.results import FFTResult

__all__ = [
    "FFTResult",
    "TransformError",
    "compute_autocorrelation",
    "compute_envelope",
    "compute_fft",
    "compute_instantaneous_frequency",
    "compute_mean_power_dbfs",
    "compute_mean_power_linear",
    "compute_phase",
    "compute_power",
    "compute_psd",
    "compute_psd_from_fft",
    "compute_welch_psd",
    "linear_power_to_dbfs",
]
