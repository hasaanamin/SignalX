"""Tests for zero-amplitude signal handling and invalid crest-factor semantics."""

import math

import numpy as np

from signal_processing.features.amplitude import AmplitudeFeatureExtractor


class DummyContext:
    def __init__(self, envelope: np.ndarray) -> None:
        self._envelope = envelope

    def envelope(self) -> np.ndarray:
        return self._envelope


class TestZeroAmplitude:
    """Test extraction on zero-amplitude signals without throwing exceptions."""

    def test_all_zero_envelope(self) -> None:
        """All-zero signal must produce 0.0 for standard metrics and invalid FeatureValue for crest factor."""
        env = np.zeros(50, dtype=np.float64)
        ctx = DummyContext(env)
        extractor = AmplitudeFeatureExtractor()

        fset = extractor.extract(ctx)

        assert fset.features["amplitude.mean"].value == 0.0
        assert fset.features["amplitude.mean"].valid is True

        assert fset.features["amplitude.rms"].value == 0.0
        assert fset.features["amplitude.rms"].valid is True

        assert fset.features["amplitude.peak"].value == 0.0
        assert fset.features["amplitude.peak"].valid is True

        assert fset.features["amplitude.min"].value == 0.0
        assert fset.features["amplitude.min"].valid is True

        assert fset.features["amplitude.std"].value == 0.0
        assert fset.features["amplitude.std"].valid is True

        assert fset.features["amplitude.variance"].value == 0.0
        assert fset.features["amplitude.variance"].valid is True

        assert fset.features["amplitude.median"].value == 0.0
        assert fset.features["amplitude.median"].valid is True

        assert fset.features["amplitude.peak_to_peak"].value == 0.0
        assert fset.features["amplitude.peak_to_peak"].valid is True

        # Crest factor must be marked invalid with explanatory reason and NaN value
        cf = fset.features["amplitude.crest_factor"]
        assert cf.valid is False
        assert math.isnan(cf.value)
        assert cf.unit is None
        assert cf.reason is not None
        assert "RMS amplitude is zero" in cf.reason
