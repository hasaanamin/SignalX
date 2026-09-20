"""Integration tests connecting SignalEvent -> SignalContext -> FrequencyFeatureExtractor."""

import numpy as np
import pytest

from signal_processing.context import AnalysisConfig, SignalContext
from signal_processing.core import FeatureSet, SignalEvent, StreamInfo
from signal_processing.features.frequency import FrequencyFeatureExtractor


@pytest.fixture
def stream_1mhz() -> StreamInfo:
    """Fixture providing standard StreamInfo at 1 MHz sample rate."""
    return StreamInfo(
        stream_id="stream-freq-int",
        source_id="synthetic-source",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )


def make_tone_event(
    stream: StreamInfo,
    f0_hz: float,
    amplitude: float = 1.0,
    nfft: int = 1024,
) -> SignalEvent:
    """Helper creating a SignalEvent with a complex sinusoid."""
    total = nfft
    n = np.arange(total)
    fs = stream.sample_rate_hz
    iq = (amplitude * np.exp(1j * 2.0 * np.pi * f0_hz * n / fs)).astype(np.complex64)

    return SignalEvent(
        event_id=f"ev-tone-{int(f0_hz)}",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=total,
        detected_start_sample=0,
        detected_end_sample=total,
        trigger_sample=total // 4,
    )


class TestContextIntegration:
    """Test frequency feature extraction on real SignalEvent and SignalContext instances."""

    def test_positive_tone_context_integration(self, stream_1mhz: StreamInfo) -> None:
        """Verify extraction on a +100 kHz tone wrapped in SignalContext."""
        # Using 1000 samples and nfft=1000 so bin spacing is exactly 1 kHz and 100 kHz is exact on-bin
        config = AnalysisConfig(nfft=1000, spectral_window="rectangular")
        event = make_tone_event(stream_1mhz, f0_hz=100_000.0, nfft=1000)
        ctx = SignalContext(event, config=config)
        extractor = FrequencyFeatureExtractor()

        fset = extractor.extract(ctx)

        assert isinstance(fset, FeatureSet)
        assert fset.extractor == "frequency"
        assert fset.extractor_version == "1.0"
        assert len(fset.features) == 2

        assert pytest.approx(fset.features["frequency.peak_offset_hz"].value, abs=1.0) == 100_000.0
        assert pytest.approx(fset.features["frequency.center_offset_hz"].value, abs=5.0) == 100_000.0

    def test_negative_tone_context_integration(self, stream_1mhz: StreamInfo) -> None:
        """Verify extraction on a -100 kHz tone wrapped in SignalContext."""
        config = AnalysisConfig(nfft=1000, spectral_window="rectangular")
        event = make_tone_event(stream_1mhz, f0_hz=-100_000.0, nfft=1000)
        ctx = SignalContext(event, config=config)
        extractor = FrequencyFeatureExtractor()

        fset = extractor.extract(ctx)

        assert pytest.approx(fset.features["frequency.peak_offset_hz"].value, abs=1.0) == -100_000.0
        assert pytest.approx(fset.features["frequency.center_offset_hz"].value, abs=5.0) == -100_000.0

    def test_amplitude_independence_on_real_context(self, stream_1mhz: StreamInfo) -> None:
        """Verify frequency features are invariant to signal amplitude on real SignalContext."""
        config = AnalysisConfig(nfft=1000, spectral_window="rectangular")
        ev_1 = make_tone_event(stream_1mhz, f0_hz=50_000.0, amplitude=1.0, nfft=1000)
        ev_2 = make_tone_event(stream_1mhz, f0_hz=50_000.0, amplitude=0.2, nfft=1000)

        ctx_1 = SignalContext(ev_1, config=config)
        ctx_2 = SignalContext(ev_2, config=config)
        extractor = FrequencyFeatureExtractor()

        fset_1 = extractor.extract(ctx_1)
        fset_2 = extractor.extract(ctx_2)

        assert fset_1.features["frequency.peak_offset_hz"].value == fset_2.features["frequency.peak_offset_hz"].value
        assert pytest.approx(fset_1.features["frequency.center_offset_hz"].value, abs=1e-3) == fset_2.features["frequency.center_offset_hz"].value

    def test_no_absolute_rf_frequency_feature(self, stream_1mhz: StreamInfo) -> None:
        """Ensure absolute RF carrier/tuned frequency is not added to the baseband FeatureSet."""
        event = make_tone_event(stream_1mhz, f0_hz=100_000.0, nfft=500)
        ctx = SignalContext(event)
        extractor = FrequencyFeatureExtractor()

        fset = extractor.extract(ctx)

        assert "frequency.peak_rf_hz" not in fset.features
        assert "frequency.center_rf_hz" not in fset.features
        assert "frequency.error_hz" not in fset.features
