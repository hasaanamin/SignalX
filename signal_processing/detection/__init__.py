"""Signal detection package."""

from signal_processing.detection.energy_detector import EnergyDetector
from signal_processing.detection.ar_noise import AR1NoiseModel, fit_ar1_noise
from signal_processing.detection.online_noise import OnlineNoiseConfig, OnlineNoiseTracker
from signal_processing.detection.exceptions import (
    DetectionError,
    FilterConfigurationError,
    NoiseEstimationError,
    PolicyCompatibilityError,
    UncalibratedNoiseError,
)
from signal_processing.detection.noise_estimation import (
    AdaptiveRobustNoiseEstimator,
    EmpiricalNoiseCalibrator,
    GaussianNoiseEstimator,
    NoiseLifecycleState,
    ThresholdPolicy,
)
from signal_processing.detection.profiles import (
    DetectionProfile, LONG_BURST_IID, LONG_BURST_AR1, AdaptiveDetectionProfile, POSITIVE_SNR_IID,
    POSITIVE_SNR_IID_1920,
)

__all__ = [
    "EnergyDetector",
    "AR1NoiseModel",
    "fit_ar1_noise",
    "OnlineNoiseConfig",
    "OnlineNoiseTracker",
    "DetectionProfile",
    "AdaptiveDetectionProfile",
    "POSITIVE_SNR_IID",
    "POSITIVE_SNR_IID_1920",
    "LONG_BURST_IID",
    "LONG_BURST_AR1",
    "GaussianNoiseEstimator",
    "AdaptiveRobustNoiseEstimator",
    "EmpiricalNoiseCalibrator",
    "NoiseLifecycleState",
    "ThresholdPolicy",
    "DetectionError",
    "NoiseEstimationError",
    "FilterConfigurationError",
    "PolicyCompatibilityError",
    "UncalibratedNoiseError",
]
