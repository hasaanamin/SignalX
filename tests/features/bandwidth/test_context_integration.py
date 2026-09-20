"""Integration tests connecting SignalEvent -> SignalContext -> BandwidthFeatureExtractor."""

import numpy as np
import pytest

from signal_processing.context import AnalysisConfig, SignalContext
from signal_processing.core import FeatureSet, SignalEvent, StreamInfo
from signal_processing.features.bandwidth import BandwidthFeatureExtractor


@pytest.fixture
def stream_1mhz() -> StreamInfo:
    """Fixture providing standard StreamInfo at 1 MHz sample rate."""
    return StreamInfo(
        stream_id="stream-bw-int",
        source_id="synthetic-source",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )


def make_tone_event(
    stream: StreamInfo,
    f0_hz: float,
    amplitude: float = 1.0,
    nfft: int = 1000,
) -> SignalEvent:
    """Helper creating a SignalEvent with a complex tone."""
    total = nfft
    n = np.arange(total)
    fs = stream.sample_rate_hz
    iq = (amplitude * np.exp(1j * 2.0 * np.pi * f0_hz * n / fs)).astype(np.complex64)

    return SignalEvent(
        event_id=f"ev-tone-bw-{int(f0_hz)}",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=total,
        detected_start_sample=0,
        detected_end_sample=total,
        trigger_sample=total // 4,
    )


class TestContextIntegration:
    """Test bandwidth feature extraction on real SignalEvent and SignalContext instances."""

    def test_case_i_real_signal_context_integration(self, stream_1mhz: StreamInfo) -> None:
        """Test Case I: Real SignalContext with a deterministic tone extracts valid non-negative bandwidths."""
        config = AnalysisConfig(nfft=1000, spectral_window="rectangular")
        event = make_tone_event(stream_1mhz, f0_hz=100_000.0, nfft=1000)
        ctx = SignalContext(event, config=config)
        extractor = BandwidthFeatureExtractor()

        fset = extractor.extract(ctx)

        assert isinstance(fset, FeatureSet)
        assert fset.extractor == "bandwidth"
        assert fset.extractor_version == "1.0"
        assert len(fset.features) == 2

        obw = fset.features["bandwidth.occupied_99_hz"]
        three_db = fset.features["bandwidth.three_db_hz"]

        assert obw.valid is True
        assert obw.value > 0.0
        assert obw.unit == "Hz"

        assert three_db.valid is True
        assert three_db.value > 0.0
        assert three_db.unit == "Hz"

        # On a 1000-point rectangular windowed exact-bin tone, df = 1000 Hz
        # Both OBW and 3 dB bandwidth should be 1000 Hz
        assert pytest.approx(obw.value) == 1000.0
        assert pytest.approx(three_db.value) == 1000.0

    def test_no_bandwidth_edge_features_in_featureset(self, stream_1mhz: StreamInfo) -> None:
        """Verify edge frequencies are not output in version 1.0."""
        event = make_tone_event(stream_1mhz, f0_hz=50_000.0, nfft=500)
        ctx = SignalContext(event)
        extractor = BandwidthFeatureExtractor()

        fset = extractor.extract(ctx)

        forbidden_keys = [
            "bandwidth.occupied_low_hz",
            "bandwidth.occupied_high_hz",
            "bandwidth.three_db_low_hz",
            "bandwidth.three_db_high_hz",
            "bandwidth.error_hz",
            "bandwidth.percent_error",
        ]
        for key in forbidden_keys:
            assert key not in fset.features
