from __future__ import annotations
from typing import Any
import numpy as np
import importlib
from pathlib import Path

# 1. Base Class & Factory
from .base import BaseCyclostationaryMetric, CyclostationaryMetricFactory
from ._common import (
    _as_iq_array,
    _extract_sample_rate,
    _compute_fine_alpha_profile,
    _interpolate_psd,
    next_power_of_two,
    StreamingCyclostationaryEstimator,
)

# 2. Dynamic module imports to register all metrics in Factory
CyclicFrequencyMetric = importlib.import_module(".Cyclic Frequency", package=__name__).CyclicFrequencyMetric
cyclic_frequency = importlib.import_module(".Cyclic Frequency", package=__name__).cyclic_frequency

CyclicAutocorrelationMetric = importlib.import_module(".Cyclic Autocorrelation", package=__name__).CyclicAutocorrelationMetric
cyclic_autocorrelation = importlib.import_module(".Cyclic Autocorrelation", package=__name__).cyclic_autocorrelation

SpectralCorrelationMetric = importlib.import_module(".Spectral Correlation", package=__name__).SpectralCorrelationMetric
spectral_correlation = importlib.import_module(".Spectral Correlation", package=__name__).spectral_correlation

SpectralCorrelationFunctionMetric = importlib.import_module(".Spectral Correlation Function", package=__name__).SpectralCorrelationFunctionMetric
spectral_correlation_function = importlib.import_module(".Spectral Correlation Function", package=__name__).spectral_correlation_function

CyclicSpectralPeaksMetric = importlib.import_module(".Cyclic Spectral Peaks", package=__name__).CyclicSpectralPeaksMetric
cyclic_spectral_peaks = importlib.import_module(".Cyclic Spectral Peaks", package=__name__).cyclic_spectral_peaks

CyclicCoherenceMetric = importlib.import_module(".Cyclic Coherence", package=__name__).CyclicCoherenceMetric
cyclic_coherence = importlib.import_module(".Cyclic Coherence", package=__name__).cyclic_coherence


def cyclostationary_summary(
    iq_data: Any,
    sample_rate: float = 1.0,
    nfft_f: int = 128,
    nfft_alpha: int | None = None,
    num_peaks: int = 5,
    snr_threshold_db: float = 8.0,
    coherence_type: str = "magnitude_squared",
    cyclic_lags: list[int] | tuple[int, ...] = (0, 1, 2, 4, 8),
    window: str = "hann",
    **kwargs: Any
) -> dict[str, Any]:
    """
    Executes a comprehensive, modulation-agnostic cyclostationary analysis:
      1. Generic fine cyclic-frequency estimation over lag products (Y_ell(alpha) -> P_alpha).
      2. Multi-candidate peak extraction with independent NFFT_alpha resolution.
      3. Exact floating-point CAF R_x^{hat{alpha}}[ell] at primary candidate.
      4. Exact 1D Spectral Correlation slice S_x^{hat{alpha}}(f) over frequency.
      5. Exact Cyclic Coherence with interpolated zero-cycle PSD (no circular roll).
      6. Coarse 2D Spectral Correlation Function (SCF) surface for visualization.

    Parameters:
        iq_data: Complex IQ array or dictionary/tuple.
        sample_rate: Sampling frequency in Hz.
        nfft_f: FFT length for ordinary spectral-frequency resolution (default: 128).
        nfft_alpha: FFT length for cyclic-frequency resolution (default: next power of two).
        num_peaks: Number of dominant cyclic candidate peaks to extract (default: 5).
        snr_threshold_db: Detection SNR threshold in dB (default: 3.0).
        coherence_type: 'magnitude_squared' (default) or 'magnitude'.
        cyclic_lags: Lag list for fine cyclic profile (default: [0, 1, 2, 4, 8]).
        window: Window type (default: 'hann').

    Returns:
        Unified dictionary with authoritative cyclic rates, candidate peaks, CAF, SCF, coherence,
        and coarse 2D surface for visualization.
    """
    if "nfft" in kwargs:
        nfft_f = int(kwargs["nfft"])

    z = _as_iq_array(iq_data)
    fs = _extract_sample_rate(kwargs, default=sample_rate)

    # 1. Authoritative Fine Cyclic-Frequency Estimation
    fine_res = _compute_fine_alpha_profile(
        z,
        fs=fs,
        nfft_alpha=nfft_alpha,
        cyclic_lags=cyclic_lags,
        window=window,
        num_candidates=num_peaks,
        snr_threshold_db=snr_threshold_db,
        min_alpha_hz=kwargs.get("min_alpha_hz", None),
    )

    candidates = fine_res["candidates"]
    cyclostationary_detected = fine_res["cyclostationary_detected"]
    raw_top_alpha = fine_res["raw_primary_alpha_hz"]
    raw_top_rate = fine_res["raw_primary_rate_hz"]
    raw_top_snr = fine_res["raw_cyclic_snr_db"]
    raw_top_score = fine_res["raw_peak_score"]

    # 2. Coarse 2D Spectral Correlation Function (SCF) Surface (for visual heatmaps)
    coarse_nfft_alpha = min(256, fine_res["nfft_alpha"])
    scf_res = spectral_correlation_function(
        z, sample_rate=fs, nfft_f=nfft_f, nfft_alpha=coarse_nfft_alpha, as_dict=True, window=window, **kwargs
    )

    # 3. Exact floating-point CAF R_x^{hat{alpha}}[ell] at primary candidate
    caf_res = cyclic_autocorrelation(
        z, alpha=raw_top_alpha, sample_rate=fs, as_dict=True
    )

    # 4. Exact 1D Spectral Correlation slice S_x^{hat{alpha}}(f) at primary candidate
    sc_slice_res = spectral_correlation(
        z, alpha=raw_top_alpha, sample_rate=fs, nfft=nfft_f, as_dict=True, window=window
    )
    peak_freq_hz = float(sc_slice_res.get("peak_frequency_hz", 0.0))

    # 5. Exact Cyclic Coherence using interpolated PSD
    coh_res = cyclic_coherence(
        scf_res,
        alpha=raw_top_alpha,
        s_alpha=sc_slice_res["s_alpha"],
        sample_rate=fs,
        nfft=nfft_f,
        coherence_type=coherence_type,
        as_dict=True
    )
    peak_cyclic_coh = float(coh_res.get("peak_coherence", coh_res.get("peak_cyclic_coherence", 0.0)))

    # 6. Cyclic frequency grid
    freq_grid_res = cyclic_frequency(
        sample_rate=fs, nfft=coarse_nfft_alpha, cyclic_rate=raw_top_rate, as_dict=True
    )

    # Format peaks dictionary
    peaks_dict = {
        "peaks": candidates,
        "num_detected_peaks": len(candidates),
        "cyclostationary_detected": cyclostationary_detected,
        "primary_signed_alpha_hz": raw_top_alpha if cyclostationary_detected else None,
        "primary_absolute_rate_hz": raw_top_rate if cyclostationary_detected else None,
        "estimated_cyclic_rate_hz": raw_top_rate if cyclostationary_detected else None,
        "primary_alpha_hz": raw_top_alpha,
        "primary_frequency_hz": peak_freq_hz if cyclostationary_detected else None,
        "primary_cyclic_snr_db": raw_top_snr,
        "raw_primary_alpha_hz": raw_top_alpha,
        "raw_primary_rate_hz": raw_top_rate,
        "raw_cyclic_snr_db": raw_top_snr,
        "max_spectral_profile": fine_res["alpha_profile"],
        "alpha_grid": fine_res["alpha_axis_hz"],
        "fine_alpha_profile": fine_res["alpha_profile"],
        "noise_floor_alpha": fine_res["noise_floor"],
        "alpha_bin_spacing_hz": fine_res["alpha_bin_spacing_hz"],
        "alpha_observation_resolution_hz": fine_res["alpha_observation_resolution_hz"],
        "sample_rate_hz": fs,
    }

    # Detection-aware user facing fields
    primary_signed_alpha_hz = raw_top_alpha if cyclostationary_detected else None
    estimated_cyclic_rate_hz = raw_top_rate if cyclostationary_detected else None
    primary_freq_out = peak_freq_hz if cyclostationary_detected else None

    return {
        # Detection & Primary Metrics
        "cyclostationary_detected": cyclostationary_detected,
        "estimated_cyclic_rate_hz": estimated_cyclic_rate_hz,
        "primary_signed_alpha_hz": primary_signed_alpha_hz,
        "primary_frequency_hz": primary_freq_out,
        "primary_alpha_hz": raw_top_alpha,  # legacy compatibility
        "estimated_symbol_rate_hz": estimated_cyclic_rate_hz,  # deprecated alias

        # Raw estimator metrics (for debugging / AWGN evaluation)
        "raw_primary_alpha_hz": raw_top_alpha,
        "raw_primary_rate_hz": raw_top_rate,
        "raw_cyclic_snr_db": raw_top_snr,
        "raw_peak_score": raw_top_score,
        "raw_peak_coherence": peak_cyclic_coh,

        # Candidate cyclic frequencies
        "candidate_alphas_hz": fine_res["candidate_alphas_hz"],
        "candidate_scores": fine_res["candidate_scores"],
        "candidates": candidates,

        # Resolutions
        "alpha_bin_spacing_hz": fine_res["alpha_bin_spacing_hz"],
        "alpha_observation_resolution_hz": fine_res["alpha_observation_resolution_hz"],
        "delta_alpha_hz": fine_res["delta_alpha_hz"],
        "delta_f_hz": fs / nfft_f,

        # Profiles and Surfaces
        "fine_alpha_axis_hz": fine_res["alpha_axis_hz"],
        "fine_alpha_profile": fine_res["alpha_profile"],
        "coarse_scf": scf_res["scf_matrix"],
        "coarse_alpha_axis_hz": scf_res["alpha_grid"],
        "frequencies": scf_res["frequencies"],

        # Slices at primary candidate alpha
        "primary_caf": caf_res,
        "primary_spectral_correlation": sc_slice_res,
        "scf_slice_at_primary_alpha": sc_slice_res["s_alpha"],
        "cyclic_coherence_at_primary_alpha": coh_res.get("coherence_slice", coh_res.get("coherence_matrix")),
        "peak_cyclic_coherence": peak_cyclic_coh,
        "primary_cyclic_snr_db": raw_top_snr,
        "coherence_type": coherence_type,

        # Sub-modules
        "scf": scf_res,
        "peaks": peaks_dict,
        "coherence": coh_res,
        "cyclic_frequencies": freq_grid_res,

        "sample_rate_hz": fs,
        "nfft_f": nfft_f,
        "nfft_alpha": fine_res["nfft_alpha"],
        "nfft": nfft_f,  # legacy
    }


__all__ = [
    # Base & Factory
    "BaseCyclostationaryMetric",
    "CyclostationaryMetricFactory",
    # 1. Cyclic Frequency
    "CyclicFrequencyMetric",
    "cyclic_frequency",
    # 2. Cyclic Autocorrelation
    "CyclicAutocorrelationMetric",
    "cyclic_autocorrelation",
    # 3. Spectral Correlation
    "SpectralCorrelationMetric",
    "spectral_correlation",
    # 4. Spectral Correlation Function
    "SpectralCorrelationFunctionMetric",
    "spectral_correlation_function",
    # 5. Cyclic Spectral Peaks
    "CyclicSpectralPeaksMetric",
    "cyclic_spectral_peaks",
    # 6. Cyclic Coherence
    "CyclicCoherenceMetric",
    "cyclic_coherence",
    # Comprehensive Summary & Streaming
    "cyclostationary_summary",
    "StreamingCyclostationaryEstimator",
    "next_power_of_two",
]
