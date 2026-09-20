"""Spectral feature extraction package."""

from signal_processing.features.spectral.exceptions import SpectralFeatureError
from signal_processing.features.spectral.extractor import SpectralFeatureExtractor

__all__ = [
    "SpectralFeatureError",
    "SpectralFeatureExtractor",
]
