"""Amplitude and envelope feature extraction package."""

from signal_processing.features.amplitude.config import AmplitudeFeatureConfig
from signal_processing.features.amplitude.exceptions import AmplitudeFeatureError
from signal_processing.features.amplitude.extractor import AmplitudeFeatureExtractor

__all__ = [
    "AmplitudeFeatureConfig",
    "AmplitudeFeatureError",
    "AmplitudeFeatureExtractor",
]
