"""Correlation feature extraction package."""

from signal_processing.features.correlation.exceptions import CorrelationFeatureError
from signal_processing.features.correlation.extractor import (
    CorrelationFeatureExtractor,
)

__all__ = [
    "CorrelationFeatureError",
    "CorrelationFeatureExtractor",
]
