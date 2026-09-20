"""Higher-order statistics feature extraction package."""

from signal_processing.features.higher_order.exceptions import HigherOrderFeatureError
from signal_processing.features.higher_order.extractor import (
    HigherOrderFeatureExtractor,
)

__all__ = [
    "HigherOrderFeatureError",
    "HigherOrderFeatureExtractor",
]
