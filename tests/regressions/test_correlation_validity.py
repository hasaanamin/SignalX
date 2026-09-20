"""Regression tests for correlation peak validity and statistics.

Requirement 5:
- For IQ [1, 0], normalized autocorrelation is [1+0j, 0+0j].
- Peak phase, peak lag in samples, and peak lag in seconds are marked invalid with reason.
- Magnitude statistics (nonzero_peak_magnitude, mean_nonzero_magnitude, rms_nonzero_magnitude) remain valid 0.0.
- Ties between multiple genuine positive-lag peaks resolve deterministically to the smallest lag.
- Complex positive-lag phase convention preserved.
"""

import math

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import AutocorrelationResult, SignalEvent, StreamInfo
from signal_processing.features.correlation import CorrelationFeatureExtractor


@pytest.fixture
def stream_info() -> StreamInfo:
    return StreamInfo(
        stream_id="stream-corr-validity",
        source_id="synthetic",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )


class TestZeroCorrelationPeakValidity:
    """Validate correlation behavior on zero positive-lag correlation signals."""

    def test_reproduction_two_sample_one_zero(self, stream_info: StreamInfo) -> None:
        """Reproduction: IQ [1, 0] has zero positive-lag autocorrelation magnitude.

        Magnitude statistics must be valid 0.0.
        Peak lag and phase must be invalid because there is no observed positive-lag peak.
        """
        iq = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex64)
        event = SignalEvent(
            event_id="ev-corr-10",
            stream=stream_info,
            iq=iq,
            event_start_sample=0,
            event_end_sample=2,
            detected_start_sample=0,
            detected_end_sample=2,
            trigger_sample=0,
        )

        ctx = SignalContext(event)
        extractor = CorrelationFeatureExtractor()
        fset = extractor.extract(ctx)

        # Magnitude statistics are valid 0.0
        assert fset.features["correlation.nonzero_peak_magnitude"].valid is True
        assert fset.features["correlation.nonzero_peak_magnitude"].value == 0.0

        assert fset.features["correlation.mean_nonzero_magnitude"].valid is True
        assert fset.features["correlation.mean_nonzero_magnitude"].value == 0.0

        assert fset.features["correlation.rms_nonzero_magnitude"].valid is True
        assert fset.features["correlation.rms_nonzero_magnitude"].value == 0.0

        # Peak location and phase are INVALID
        peak_lag_samples = fset.features["correlation.nonzero_peak_lag_samples"]
        assert peak_lag_samples.valid is False
        assert math.isnan(peak_lag_samples.value)
        assert "no positive-lag correlation peak observed" in peak_lag_samples.reason.lower()

        peak_lag_s = fset.features["correlation.nonzero_peak_lag_s"]
        assert peak_lag_s.valid is False
        assert math.isnan(peak_lag_s.value)
        assert "no positive-lag correlation peak observed" in peak_lag_s.reason.lower()

        peak_phase = fset.features["correlation.nonzero_peak_phase_rad"]
        assert peak_phase.valid is False
        assert math.isnan(peak_phase.value)
        assert "no positive-lag correlation peak observed" in peak_phase.reason.lower()

    def test_deterministic_smallest_lag_selection_on_ties(self) -> None:
        """When multiple positive lags have identical maximum magnitude, select smallest lag."""
        # Mock AutocorrelationResult with lags 0, 1, 2, 3 where lags 1 and 2 tie at mag 0.8
        class MockTieContext:
            sample_rate_hz = 1_000_000.0

            def mean_power_linear(self) -> float:
                return 1.0

            def autocorrelation(self) -> AutocorrelationResult:
                # Lag 0: 1.0; Lag 1: 0.8j; Lag 2: 0.8; Lag 3: 0.2
                lags = np.array([0, 1, 2, 3], dtype=np.int64)
                vals = np.array([1.0 + 0j, 0.0 + 0.8j, 0.8 + 0j, 0.2 + 0j], dtype=np.complex128)
                return AutocorrelationResult(lag_samples=lags, values=vals)

        ctx = MockTieContext()
        extractor = CorrelationFeatureExtractor()
        fset = extractor.extract(ctx)

        # Smallest lag (1) must be selected
        assert fset.features["correlation.nonzero_peak_lag_samples"].valid is True
        assert fset.features["correlation.nonzero_peak_lag_samples"].value == 1.0
        assert fset.features["correlation.nonzero_peak_lag_s"].value == 1.0 / 1_000_000.0

        # Phase at lag 1 is angle(0.0 + 0.8j) = pi/2
        assert fset.features["correlation.nonzero_peak_phase_rad"].valid is True
        assert pytest.approx(fset.features["correlation.nonzero_peak_phase_rad"].value, abs=1e-6) == np.pi / 2.0
