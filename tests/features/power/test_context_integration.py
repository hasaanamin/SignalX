"""Integration tests connecting SignalEvent -> SignalContext -> PowerFeatureExtractor."""

import math

import numpy as np
import pytest

from signal_processing.context import AnalysisConfig, SignalContext
from signal_processing.core import FeatureSet, SignalEvent, StreamInfo
from signal_processing.features.power import PowerFeatureExtractor


@pytest.fixture
def tone_event() -> SignalEvent:
    """Fixture producing a 0.5-amplitude 50 kHz complex tone within a SignalEvent."""
    stream = StreamInfo(
        stream_id="stream-power-int",
        source_id="synthetic-tone",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=900_000_000.0,
    )
    total = 800
    n = np.arange(total)
    fs = stream.sample_rate_hz
    tone = (0.5 * np.exp(1j * 2.0 * np.pi * 50_000.0 * n / fs)).astype(np.complex64)

    iq = np.zeros(total, dtype=np.complex64)
    # Active signal region: 100 to 700 (length = 600)
    iq[100:700] = tone[100:700]

    return SignalEvent(
        event_id="ev-power-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=total,
        detected_start_sample=100,
        detected_end_sample=700,
        trigger_sample=200,
    )


@pytest.fixture
def zero_event() -> SignalEvent:
    """Fixture producing an all-zero SignalEvent."""
    stream = StreamInfo(
        stream_id="stream-power-zero",
        source_id="synthetic-zero",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=900_000_000.0,
    )
    total = 500
    iq = np.zeros(total, dtype=np.complex64)

    return SignalEvent(
        event_id="ev-power-zero-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=total,
        detected_start_sample=50,
        detected_end_sample=450,
        trigger_sample=100,
    )


class TestContextIntegration:
    """Test power feature extraction on real SignalContext instances."""

    def test_power_features_on_complex_tone(self, tone_event: SignalEvent) -> None:
        """Verify power features for a 0.5 amplitude (0.25 power) complex tone."""
        ctx = SignalContext(tone_event)
        extractor = PowerFeatureExtractor()

        fset = extractor.extract(ctx)

        assert isinstance(fset, FeatureSet)
        assert fset.extractor == "power"
        assert fset.extractor_version == "1.0"
        assert len(fset.features) == 10

        expected_dbfs = 10.0 * math.log10(0.25)

        assert pytest.approx(fset.features["power.mean_linear"].value, abs=1e-5) == 0.25
        assert pytest.approx(fset.features["power.mean_dbfs"].value, abs=1e-5) == expected_dbfs
        assert pytest.approx(fset.features["power.peak_linear"].value, abs=1e-5) == 0.25
        assert pytest.approx(fset.features["power.peak_dbfs"].value, abs=1e-5) == expected_dbfs
        assert pytest.approx(fset.features["power.min_linear"].value, abs=1e-5) == 0.25
        assert pytest.approx(fset.features["power.std_linear"].value, abs=1e-5) == 0.0
        assert pytest.approx(fset.features["power.variance_linear"].value, abs=1e-5) == 0.0
        assert pytest.approx(fset.features["power.median_linear"].value, abs=1e-5) == 0.25
        assert pytest.approx(fset.features["power.papr_linear"].value, abs=1e-5) == 1.0
        assert pytest.approx(fset.features["power.papr_db"].value, abs=1e-5) == 0.0

    def test_repeated_extraction_statelessness(self, tone_event: SignalEvent) -> None:
        """Calling extract multiple times must produce identical, consistent FeatureSets."""
        ctx = SignalContext(tone_event)
        extractor = PowerFeatureExtractor()

        fset1 = extractor.extract(ctx)
        fset2 = extractor.extract(ctx)

        for k in fset1.features:
            assert fset1.features[k].value == fset2.features[k].value
            assert fset1.features[k].valid == fset2.features[k].valid
            assert fset1.features[k].unit == fset2.features[k].unit
            assert fset1.features[k].reason == fset2.features[k].reason

    def test_custom_power_floor_integration(self, zero_event: SignalEvent) -> None:
        """Verify extractor respects the AnalysisConfig power_floor_dbfs on SignalContext."""
        config = AnalysisConfig(power_floor_dbfs=-180.0)
        ctx = SignalContext(zero_event, config=config)
        extractor = PowerFeatureExtractor()

        fset = extractor.extract(ctx)

        assert fset.features["power.mean_linear"].value == 0.0
        assert fset.features["power.mean_dbfs"].value == -180.0
        assert fset.features["power.peak_linear"].value == 0.0
        assert fset.features["power.peak_dbfs"].value == -180.0
        assert fset.features["power.papr_linear"].valid is False
        assert fset.features["power.papr_db"].valid is False
