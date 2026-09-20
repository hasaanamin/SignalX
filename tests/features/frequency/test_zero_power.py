"""Tests for zero-power signal handling and invalid frequency feature semantics."""

import math

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import PSDResult, SignalEvent, StreamInfo
from signal_processing.features.frequency import FrequencyFeatureExtractor


class DummyZeroPSDContext:
    def __init__(self, size: int = 64) -> None:
        freqs = np.linspace(-500_000.0, 500_000.0, size, dtype=np.float64)
        powers = np.zeros(size, dtype=np.float64)
        self._psd = PSDResult(
            frequency_offset_hz=freqs,
            power_linear=powers,
            power_dbfs_per_hz=np.full(size, -300.0, dtype=np.float64),
        )

    def psd(self) -> PSDResult:
        return self._psd


@pytest.fixture
def zero_event() -> SignalEvent:
    """Fixture producing an all-zero SignalEvent."""
    stream = StreamInfo(
        stream_id="stream-freq-zero",
        source_id="synthetic-zero",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    total = 500
    iq = np.zeros(total, dtype=np.complex64)

    return SignalEvent(
        event_id="ev-freq-zero-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=total,
        detected_start_sample=50,
        detected_end_sample=450,
        trigger_sample=100,
    )


class TestZeroPowerFrequency:
    """Test extraction on zero-power signals without throwing exceptions."""

    def test_synthetic_all_zero_psd(self) -> None:
        """All-zero synthetic PSD must produce invalid FeatureValues with NaN and descriptive reason."""
        ctx = DummyZeroPSDContext(size=128)
        extractor = FrequencyFeatureExtractor()

        fset = extractor.extract(ctx)

        # 1. Peak
        peak = fset.features["frequency.peak_offset_hz"]
        assert peak.valid is False
        assert math.isnan(peak.value)
        assert peak.unit == "Hz"
        assert peak.reason == "PSD contains no positive signal power"

        # 2. Center
        center = fset.features["frequency.center_offset_hz"]
        assert center.valid is False
        assert math.isnan(center.value)
        assert center.unit == "Hz"
        assert center.reason == "PSD contains no positive signal power"

    def test_real_zero_signal_context_integration(self, zero_event: SignalEvent) -> None:
        """Test Case E: Real SignalContext on all-zero IQ produces invalid frequency features gracefully."""
        ctx = SignalContext(zero_event)
        extractor = FrequencyFeatureExtractor()

        fset = extractor.extract(ctx)

        peak = fset.features["frequency.peak_offset_hz"]
        assert peak.valid is False
        assert math.isnan(peak.value)
        assert peak.unit == "Hz"
        assert peak.reason == "PSD contains no positive signal power"

        center = fset.features["frequency.center_offset_hz"]
        assert center.valid is False
        assert math.isnan(center.value)
        assert center.unit == "Hz"
        assert center.reason == "PSD contains no positive signal power"
