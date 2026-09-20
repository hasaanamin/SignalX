"""Tests for PhaseFeatureExtractor class contract and edge cases."""

import math

import numpy as np

from signal_processing.context import SignalContext
from signal_processing.core import (
    FeatureSet,
    FeatureValue,
    PhaseResult,
    SignalEvent,
    StreamInfo,
)
from signal_processing.features.phase import PhaseFeatureExtractor


class DummyPhaseContext:
    def __init__(self, wrapped_rad: np.ndarray, unwrapped_rad: np.ndarray, mean_power: float = 1.0) -> None:
        self._phase = PhaseResult(wrapped_rad=wrapped_rad, unwrapped_rad=unwrapped_rad)
        self._mean_power = mean_power

    def phase(self) -> PhaseResult:
        return self._phase

    def mean_power_linear(self) -> float:
        return self._mean_power


class TestPhaseExtractorContract:
    """Validate FeatureSet container invariants, typing, and single-sample edge cases."""

    def test_case_i_one_nonzero_sample(self) -> None:
        """Test Case I: Single-sample signal has valid circular features (R=1, Var=0) and invalid increments."""
        stream = StreamInfo(
            stream_id="stream-single-sample",
            source_id="synthetic-src",
            sample_rate_hz=1_000_000.0,
            center_frequency_hz=915_000_000.0,
        )
        iq = np.array([1.0 + 0.0j], dtype=np.complex64)
        event = SignalEvent(
            event_id="ev-single-sample",
            stream=stream,
            iq=iq,
            event_start_sample=0,
            event_end_sample=1,
            detected_start_sample=0,
            detected_end_sample=1,
            trigger_sample=0,
        )
        ctx = SignalContext(event)

        extractor = PhaseFeatureExtractor()
        fset = extractor.extract(ctx)

        assert isinstance(fset, FeatureSet)
        assert len(fset.features) == 6

        # Circular features are valid
        r = fset.features["phase.circular_resultant_length"]
        v = fset.features["phase.circular_variance"]
        assert r.valid is True
        assert r.value == 1.0
        assert r.unit is None

        assert v.valid is True
        assert v.value == 0.0
        assert v.unit is None

        # Increment features are invalid
        increment_keys = [
            "phase.increment_mean_rad",
            "phase.increment_std_rad",
            "phase.increment_rms_rad",
            "phase.increment_abs_mean_rad",
        ]
        for key in increment_keys:
            fv = fset.features[key]
            assert fv.valid is False
            assert math.isnan(fv.value)
            assert fv.unit == "rad"
            assert fv.reason is not None
            assert "at least two phase samples" in fv.reason.lower()

    def test_extractor_metadata_and_python_float_typing(self) -> None:
        """Verify extractor name, version, 6 exact keys, and pure Python float values."""
        wrapped = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        unwrapped = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        ctx = DummyPhaseContext(wrapped, unwrapped)

        extractor = PhaseFeatureExtractor()
        assert extractor.name == "phase"
        assert extractor.version == "1.0"

        fset = extractor.extract(ctx)
        assert fset.extractor == "phase"
        assert fset.extractor_version == "1.0"
        assert len(fset.features) == 6

        expected_units = {
            "phase.circular_resultant_length": None,
            "phase.circular_variance": None,
            "phase.increment_mean_rad": "rad",
            "phase.increment_std_rad": "rad",
            "phase.increment_rms_rad": "rad",
            "phase.increment_abs_mean_rad": "rad",
        }
        assert set(fset.features.keys()) == set(expected_units.keys())

        for key, expected_unit in expected_units.items():
            fv = fset.features[key]
            assert isinstance(fv, FeatureValue)
            assert fv.valid is True
            assert fv.unit == expected_unit
            assert type(fv.value) is float
            assert not isinstance(fv.value, (np.floating, np.integer))

    def test_repeated_extraction_is_deterministic(self) -> None:
        """Verify repeated extraction on same context is identical."""
        wrapped = np.array([0.1, -0.2, 0.5, -0.1], dtype=np.float64)
        unwrapped = np.array([0.1, -0.2, 0.5, -0.1], dtype=np.float64)
        ctx = DummyPhaseContext(wrapped, unwrapped)

        extractor = PhaseFeatureExtractor()
        fset1 = extractor.extract(ctx)
        fset2 = extractor.extract(ctx)

        for key in fset1.features:
            assert fset1.features[key].value == fset2.features[key].value
            assert fset1.features[key].valid == fset2.features[key].valid
