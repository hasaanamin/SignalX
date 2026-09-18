from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class AMAnalysisResult:
    """
    Result container for AM Analysis module.
    """
    envelope: np.ndarray
    normalized_envelope: np.ndarray
    recovered_modulation: np.ndarray
    times_sec: np.ndarray
    valid_mask: np.ndarray

    reference_amplitude: float
    envelope_mean: float
    envelope_std: float
    envelope_rms_variation: float
    envelope_peak_to_peak: float

    dominant_modulation_frequency_hz: Optional[float]

    am_activity_score: float
    valid_sample_fraction: float
    confidence: float

    sample_rate: float = 1.0
    smoothing: str = "none"


@dataclass
class FMAnalysisResult:
    """
    Result container for FM Analysis module.
    """
    instantaneous_phase_rad: np.ndarray
    instantaneous_frequency_hz: np.ndarray
    frequency_deviation_hz: np.ndarray

    times_sec: np.ndarray
    valid_mask: np.ndarray

    center_frequency_hz: float

    mean_deviation_hz: float
    std_deviation_hz: float
    rms_deviation_hz: float

    dominant_modulation_frequency_hz: Optional[float]

    fm_activity_score: float
    valid_sample_fraction: float
    confidence: float

    sample_rate: float = 1.0
    smoothing: str = "none"


@dataclass
class PMAnalysisResult:
    """
    Result container for PM Analysis module.
    """
    unwrapped_phase_rad: np.ndarray
    phase_trend_rad: np.ndarray
    phase_deviation_rad: np.ndarray

    times_sec: np.ndarray
    valid_mask: np.ndarray

    estimated_carrier_frequency_hz: float
    constant_phase_rad: float

    rms_phase_deviation_rad: float
    phase_deviation_std_rad: float

    dominant_modulation_frequency_hz: Optional[float]

    pm_activity_score: float
    valid_sample_fraction: float
    confidence: float

    sample_rate: float = 1.0
    smoothing: str = "none"


@dataclass
class AMDepthResult:
    """
    Result container for AM Depth module.
    """
    modulation_depth: Optional[float]
    modulation_depth_percent: Optional[float]

    envelope_max: float
    envelope_min: float

    estimator: str

    overmodulated: bool

    valid: bool
    confidence: float
    reason: str


@dataclass
class FMDeviationResult:
    """
    Result container for FM Deviation module.
    """
    peak_deviation_hz: float
    positive_deviation_hz: float
    negative_deviation_hz: float
    peak_to_peak_deviation_hz: float
    rms_deviation_hz: float

    center_frequency_hz: float

    valid_sample_fraction: float
    confidence: float


@dataclass
class FMIndexResult:
    """
    Result container for FM Index module.
    """
    modulation_index: Optional[float]
    peak_deviation_hz: float
    modulation_frequency_hz: Optional[float]

    valid: bool
    interpretation: str
    confidence: float


@dataclass
class PMDeviationResult:
    """
    Result container for PM Deviation module.
    """
    peak_deviation_rad: float
    positive_deviation_rad: float
    negative_deviation_rad: float

    peak_to_peak_deviation_rad: float
    rms_deviation_rad: float

    peak_deviation_deg: float

    valid_sample_fraction: float
    confidence: float


@dataclass
class SymbolRateCandidate:
    """
    Candidate cyclic frequency / symbol rate entry.
    """
    rate_baud: float
    score: float
    source_cyclic_frequency_hz: float
    source: str


@dataclass
class SymbolRateResult:
    """
    Result container for Symbol Rate module.
    """
    symbol_rate_baud: Optional[float]
    samples_per_symbol: Optional[float]

    candidates: List[SymbolRateCandidate]

    confidence: float
    valid: bool
    reason: str


@dataclass
class ConstellationResult:
    """
    Result container for Constellation Generation module.
    """
    raw_points: np.ndarray
    normalized_points: np.ndarray
    symbol_points: np.ndarray
    symbol_times_sec: np.ndarray

    estimated_symbol_rate_baud: Optional[float]
    samples_per_symbol: Optional[float]
    timing_offset_samples: Optional[float]

    estimated_frequency_offset_hz: float
    estimated_phase_offset_rad: float

    dc_removed: complex
    normalization_scale: float

    frequency_correction_method: str
    timing_method: str

    confidence: float
    valid: bool


@dataclass
class PhaseStatesResult:
    """
    Result container for Phase States module.
    """
    num_phase_states: int

    state_angles_rad: np.ndarray
    state_angles_deg: np.ndarray

    occupancies: np.ndarray
    concentrations: np.ndarray

    labels: np.ndarray
    symbol_points: np.ndarray

    model_scores: Dict[int, float]

    confidence: float
    valid: bool
    reason: str


@dataclass
class FrequencyStatesResult:
    """
    Result container for Frequency States module.
    """
    num_frequency_states: int

    state_frequencies_hz: np.ndarray
    state_std_hz: np.ndarray
    occupancies: np.ndarray

    labels: np.ndarray
    transition_matrix: np.ndarray

    minimum_state_separation: float

    confidence: float
    valid: bool
    reason: str


@dataclass
class ConstellationFeaturesResult:
    """
    Result container for Constellation Features module.
    """
    num_symbol_points: int

    mean_complex: complex
    mean_power: float

    normalized_fourth_moment: float
    circularity_coefficient: float

    mean_radius: float
    radius_std: float
    radius_cv: float
    radius_skewness: float
    radius_kurtosis: float

    radial_level_count: int

    apparent_phase_state_count: int
    phase_concentration: float
    phase_occupancy_uniformity: float

    iq_covariance: np.ndarray
    covariance_eigenvalues: np.ndarray
    covariance_axis_ratio: float

    constellation_cluster_count: int
    cluster_compactness: float
    minimum_cluster_separation: float

    evm_rms: Optional[float]
    evm_percent: Optional[float]
    evm_valid: bool

    confidence: float
    valid: bool


@dataclass
class ModulationClassificationResult:
    """
    Result container for Modulation Type Classification module.
    """
    predicted_class: str

    confidence: float
    class_scores: Dict[str, float]

    extracted_features: Dict[str, float]

    valid_classes: Tuple[str, ...]

    unknown: bool
    reason: str

    classifier_mode: str


@dataclass
class ModulationIndexResult:
    """
    Result container for Modulation Index module.
    """
    modulation_type: str
    modulation_index: Optional[float]

    definition: str
    units: str

    valid: bool
    confidence: float

    reason: str
