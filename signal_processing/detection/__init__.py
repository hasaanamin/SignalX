"""Signal detection package."""

from signal_processing.detection.energy_detector import EnergyDetector
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

__all__ = [
    "EnergyDetector",
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
