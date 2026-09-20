"""Tests for zero-power signal handling and invalid correlation feature semantics."""

import math

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import AutocorrelationResult, SignalEvent, StreamInfo
from signal_processing.features.correlation import CorrelationFeatureExtractor


class ContextTrackAutocorrelationCall:
    """Mock context verifying that autocorrelation() is NOT called when mean_power_linear is zero."""

    def __init__(self, mean_power: float = 0.0) -> None:
        self._mean_power = mean_power
        self.autocorrelation_called = False
        self.sample_rate_hz = 1_000_000.0

    def mean_power_linear(self) -> float:
        return self._mean_power

    def autocorrelation(self) -> AutocorrelationResult:
        self.autocorrelation_called = True
        raise AssertionError("autocorrelation() must not be called when signal power is zero!")


@pytest.fixture
def zero_event() -> SignalEvent:
    """Fixture producing an all-zero SignalEvent."""
    stream = StreamInfo(
        stream_id="stream-corr-zero",
        source_id="synthetic-zero",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    total = 500
    iq = np.zeros(total, dtype=np.complex64)

    return SignalEvent(
        event_id="ev-corr-zero-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=total,
        detected_start_sample=50,
        detected_end_sample=450,
        trigger_sample=100,
    )


class TestZeroPowerCorrelation:
    """Test extraction on zero-power signals without throwing exceptions."""

    def test_case_i_real_zero_signal_context(self, zero_event: SignalEvent) -> None:
        """Test Case I: Real SignalContext with all-zero IQ yields invalid features with clear reason."""
        ctx = SignalContext(zero_event)
        extractor = CorrelationFeatureExtractor()

        fset = extractor.extract(ctx)

        for key in [
            "correlation.nonzero_peak_magnitude",
            "correlation.nonzero_peak_lag_samples",
            "correlation.nonzero_peak_lag_s",
            "correlation.nonzero_peak_phase_rad",
            "correlation.mean_nonzero_magnitude",
            "correlation.rms_nonzero_magnitude",
        ]:
            fv = fset.features[key]
            assert fv.valid is False
            assert math.isnan(fv.value)
            assert "Signal has zero power" in fv.reason

    def test_zero_power_precheck_skips_autocorrelation_call(self) -> None:
        """Verify that when mean_power_linear is <= 0, autocorrelation() is not called at all."""
        mock_ctx = ContextTrackAutocorrelationCall(mean_power=0.0)
        extractor = CorrelationFeatureExtractor()

        fset = extractor.extract(mock_ctx)

        assert mock_ctx.autocorrelation_called is False
        for key in fset.features:
            assert fset.features[key].valid is False
            assert math.isnan(fset.features[key].value)
