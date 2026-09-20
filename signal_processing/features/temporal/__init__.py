"""Temporal-domain feature extraction package."""

from signal_processing.features.temporal.exceptions import TemporalFeatureError
from signal_processing.features.temporal.extractor import TemporalFeatureExtractor

__all__ = [
    "TemporalFeatureError",
    "TemporalFeatureExtractor",
]
