"""Statistics feature extraction package."""

from signal_processing.features.statistics.exceptions import StatisticsFeatureError
from signal_processing.features.statistics.extractor import StatisticsFeatureExtractor

__all__ = [
    "StatisticsFeatureError",
    "StatisticsFeatureExtractor",
]
