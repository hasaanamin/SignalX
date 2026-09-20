"""Tests for zero-power signal handling and invalid bandwidth feature semantics."""

import math

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import PSDResult, SignalEvent, StreamInfo
from signal_processing.features.bandwidth import BandwidthFeatureExtractor


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
        stream_id="stream-bw-zero",
        source_id="synthetic-zero",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    total = 500
    iq = np.zeros(total, dtype=np.complex64)

    return SignalEvent(
        event_id="ev-bw-zero-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=total,
        detected_start_sample=50,
        detected_end_sample=450,
        trigger_sample=100,
    )


class TestZeroPowerBandwidth:
    """Test extraction on zero-power signals without throwing exceptions."""

    def test_case_f_synthetic_all_zero_psd(self) -> None:
        """Test Case F: All-zero synthetic PSD produces invalid FeatureValues with NaN and descriptive reason."""
        ctx = DummyZeroPSDContext(size=128)
        extractor = BandwidthFeatureExtractor()

        fset = extractor.extract(ctx)

        # 1. 99% OBW
        obw = fset.features["bandwidth.occupied_99_hz"]
        assert obw.valid is False
        assert math.isnan(obw.value)
        assert obw.unit == "Hz"
        assert obw.reason == "PSD contains no positive signal power"

        # 2. 3 dB
        three_db = fset.features["bandwidth.three_db_hz"]
        assert three_db.valid is False
        assert math.isnan(three_db.value)
        assert three_db.unit == "Hz"
        assert three_db.reason == "PSD contains no positive signal power"

    def test_real_zero_signal_context_integration(self, zero_event: SignalEvent) -> None:
        """Real SignalContext on all-zero IQ produces invalid bandwidth features gracefully."""
        ctx = SignalContext(zero_event)
        extractor = BandwidthFeatureExtractor()

        fset = extractor.extract(ctx)

        obw = fset.features["bandwidth.occupied_99_hz"]
        assert obw.valid is False
        assert math.isnan(obw.value)
        assert obw.unit == "Hz"
        assert obw.reason == "PSD contains no positive signal power"

        three_db = fset.features["bandwidth.three_db_hz"]
        assert three_db.valid is False
        assert math.isnan(three_db.value)
        assert three_db.unit == "Hz"
        assert three_db.reason == "PSD contains no positive signal power"
