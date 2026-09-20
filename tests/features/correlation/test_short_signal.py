"""Tests for short-signal and max-lag-zero edge cases where no positive lags exist."""

import math

import numpy as np

from signal_processing.context import AnalysisConfig, SignalContext
from signal_processing.core import AutocorrelationResult, SignalEvent, StreamInfo
from signal_processing.features.correlation import CorrelationFeatureExtractor


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


class TestShortSignalAndZeroMaxLag:
    """Test handling when no positive autocorrelation lags exist."""

    def test_case_j_one_sample_only_lag_zero(self) -> None:
        """Test Case J: When only lag 0 exists (e.g. single sample), all six features are marked invalid."""
        lags = np.array([0], dtype=np.int64)
        values = np.array([1.0 + 0.0j], dtype=np.complex128)
        ctx = DummyCorrelationContext(lags, values, mean_power=1.0)
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
            assert "At least one positive autocorrelation lag is required" in fv.reason

    def test_case_k_max_lag_zero_config(self) -> None:
        """Test Case K: AnalysisConfig with autocorrelation_max_lag_samples=0 returns only lag 0 -> invalid features."""
        stream = StreamInfo(
            stream_id="stream-maxlag0",
            source_id="synthetic-src",
            sample_rate_hz=1000.0,
            center_frequency_hz=915_000_000.0,
        )
        iq = np.array([1.0 + 0.0j, 2.0 + 1.0j, -1.0 + 0.5j], dtype=np.complex64)
        event = SignalEvent(
            event_id="ev-maxlag0",
            stream=stream,
            iq=iq,
            event_start_sample=0,
            event_end_sample=3,
            detected_start_sample=0,
            detected_end_sample=3,
            trigger_sample=0,
        )
        config = AnalysisConfig(autocorrelation_max_lag_samples=0)
        ctx = SignalContext(event, config=config)
        extractor = CorrelationFeatureExtractor()

        fset = extractor.extract(ctx)

        for key in fset.features:
            fv = fset.features[key]
            assert fv.valid is False
            assert math.isnan(fv.value)
            assert "At least one positive autocorrelation lag is required" in fv.reason
