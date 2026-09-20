"""Integration tests connecting SignalEvent -> SignalContext -> AmplitudeFeatureExtractor."""

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, SignalEvent, StreamInfo
from signal_processing.features.amplitude import AmplitudeFeatureExtractor


@pytest.fixture
def tone_event() -> SignalEvent:
    """Fixture producing a 0.5-amplitude 50 kHz complex tone within a SignalEvent."""
    stream = StreamInfo(
        stream_id="stream-feature-int",
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
        event_id="ev-tone-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=total,
        detected_start_sample=100,
        detected_end_sample=700,
        trigger_sample=200,
    )


class TestContextIntegration:
    """Test amplitude feature extraction on real SignalContext instances."""

    def test_amplitude_features_on_complex_tone(self, tone_event: SignalEvent) -> None:
        """Verify amplitude features for a 0.5 amplitude complex tone."""
        ctx = SignalContext(tone_event)
        extractor = AmplitudeFeatureExtractor()

        fset = extractor.extract(ctx)

        assert isinstance(fset, FeatureSet)
        assert fset.extractor == "amplitude"
        assert fset.extractor_version == "1.0"

        assert pytest.approx(fset.features["amplitude.mean"].value, abs=1e-5) == 0.5
        assert pytest.approx(fset.features["amplitude.rms"].value, abs=1e-5) == 0.5
        assert pytest.approx(fset.features["amplitude.peak"].value, abs=1e-5) == 0.5
        assert pytest.approx(fset.features["amplitude.min"].value, abs=1e-5) == 0.5
        assert pytest.approx(fset.features["amplitude.std"].value, abs=1e-5) == 0.0
        assert pytest.approx(fset.features["amplitude.variance"].value, abs=1e-5) == 0.0
        assert pytest.approx(fset.features["amplitude.median"].value, abs=1e-5) == 0.5
        assert pytest.approx(fset.features["amplitude.peak_to_peak"].value, abs=1e-5) == 0.0
        assert pytest.approx(fset.features["amplitude.crest_factor"].value, abs=1e-5) == 1.0

    def test_repeated_extraction_statelessness(self, tone_event: SignalEvent) -> None:
        """Calling extract multiple times must produce identical, consistent FeatureSets."""
        ctx = SignalContext(tone_event)
        extractor = AmplitudeFeatureExtractor()

        fset1 = extractor.extract(ctx)
        fset2 = extractor.extract(ctx)

        for k in fset1.features:
            assert fset1.features[k].value == fset2.features[k].value
            assert fset1.features[k].valid == fset2.features[k].valid
            assert fset1.features[k].unit == fset2.features[k].unit
            assert fset1.features[k].reason == fset2.features[k].reason
