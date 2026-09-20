"""Frequency-domain feature extraction package."""

from signal_processing.features.frequency.exceptions import FrequencyFeatureError
from signal_processing.features.frequency.extractor import FrequencyFeatureExtractor

__all__ = [
    "FrequencyFeatureError",
    "FrequencyFeatureExtractor",
]
