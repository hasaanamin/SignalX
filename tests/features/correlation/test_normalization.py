"""Tests for autocorrelation normalization and invariance properties."""

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import AutocorrelationResult, SignalEvent, StreamInfo
from signal_processing.features.correlation import (
    CorrelationFeatureError,
    CorrelationFeatureExtractor,
)


class DummyCorrelationContext:
    def __init__(
        self,
        lag_samples: np.ndarray,
        values: np.ndarray,
        sample_rate_hz: float = 1000.0,
        mean_power: float = 1.0,
    ) -> None:
        self._ac = AutocorrelationResult(lag_samples=lag_samples, values=values)
        self.sample_rate_hz = sample_rate_hz
        self._mean_power = mean_power

    def autocorrelation(self) -> AutocorrelationResult:
        return self._ac

    def mean_power_linear(self) -> float:
        return self._mean_power


def make_pattern_event(scale: float = 1.0, phase_rad: float = 0.0) -> SignalEvent:
    """Helper creating a deterministic repeating complex IQ event with optional scaling and phase rotation."""
    stream = StreamInfo(
        stream_id="stream-norm-corr",
        source_id="synthetic-src",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    # Repeating block of length 16
    block = np.array([1.0, 0.0, -1.0, 0.5, 2.0, -0.5, 1.0, 0.0] * 2, dtype=np.complex64)
    iq = (np.tile(block, 10) * scale * np.exp(1j * phase_rad)).astype(np.complex64)
    total = len(iq)

    return SignalEvent(
        event_id=f"ev-pattern-{scale}-{phase_rad}",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=total,
        detected_start_sample=0,
        detected_end_sample=total,
        trigger_sample=0,
    )


class TestNormalizationAndInvariance:
    """Test raw vs normalized input equivalence, amplitude scale invariance, and phase rotation invariance."""

    def test_case_b_raw_autocorrelation_normalized_locally(self) -> None:
        """Test Case B: Raw unnormalized autocorrelation is normalized by zero-lag value to match normalized input."""
        lags = np.array([0, 1, 2], dtype=np.int64)

        # Raw values: [10, 5, 2]
        raw_values = np.array([10.0 + 0.0j, 5.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
        ctx_raw = DummyCorrelationContext(lags, raw_values, mean_power=10.0)

        # Pre-normalized values: [1.0, 0.5, 0.2]
        norm_values = np.array([1.0 + 0.0j, 0.5 + 0.0j, 0.2 + 0.0j], dtype=np.complex128)
        ctx_norm = DummyCorrelationContext(lags, norm_values, mean_power=1.0)

        extractor = CorrelationFeatureExtractor()
        fset_raw = extractor.extract(ctx_raw)
        fset_norm = extractor.extract(ctx_norm)

        for key in fset_raw.features:
            assert pytest.approx(fset_raw.features[key].value, rel=1e-12) == fset_norm.features[key].value

    def test_case_g_amplitude_scale_invariance(self) -> None:
        """Test Case G: Scaling signal IQ by 0.2 and 5.0 preserves all 6 correlation features."""
        event_1 = make_pattern_event(scale=1.0)
        event_small = make_pattern_event(scale=0.2)
        event_large = make_pattern_event(scale=5.0)

        extractor = CorrelationFeatureExtractor()
        fset_1 = extractor.extract(SignalContext(event_1))
        fset_small = extractor.extract(SignalContext(event_small))
        fset_large = extractor.extract(SignalContext(event_large))

        for key in fset_1.features:
            v1 = fset_1.features[key].value
            v_small = fset_small.features[key].value
            v_large = fset_large.features[key].value

            assert pytest.approx(v_small, rel=1e-5, abs=1e-5) == v1
            assert pytest.approx(v_large, rel=1e-5, abs=1e-5) == v1

    def test_case_h_constant_phase_rotation_invariance(self) -> None:
        """Test Case H: Constant carrier phase rotation preserves autocorrelation and all 6 features."""
        event_orig = make_pattern_event(scale=1.0, phase_rad=0.0)
        event_rot = make_pattern_event(scale=1.0, phase_rad=1.2345)

        extractor = CorrelationFeatureExtractor()
        fset_orig = extractor.extract(SignalContext(event_orig))
        fset_rot = extractor.extract(SignalContext(event_rot))

        for key in fset_orig.features:
            v_orig = fset_orig.features[key].value
            v_rot = fset_rot.features[key].value

            assert pytest.approx(v_rot, rel=1e-5, abs=1e-5) == v_orig

    def test_material_violation_of_normalization_raises_error(self) -> None:
        """Verify that a normalized magnitude materially exceeding 1.0 raises CorrelationFeatureError."""
        lags = np.array([0, 1, 2], dtype=np.int64)
        # Lag 1 magnitude is 1.5, which materially exceeds 1.0
        values = np.array([1.0 + 0.0j, 1.5 + 0.0j, 0.2 + 0.0j], dtype=np.complex128)
        ctx = DummyCorrelationContext(lags, values)
        extractor = CorrelationFeatureExtractor()

        with pytest.raises(CorrelationFeatureError, match="exceeds 1.0"):
            extractor.extract(ctx)
