"""Integration tests connecting SignalEvent -> SignalContext -> SpectralFeatureExtractor."""

import numpy as np
import pytest

from signal_processing.context import AnalysisConfig, SignalContext
from signal_processing.core import FeatureSet, SignalEvent, StreamInfo
from signal_processing.features.spectral import SpectralFeatureExtractor


@pytest.fixture
def stream_1mhz() -> StreamInfo:
    """Fixture providing standard StreamInfo at 1 MHz sample rate."""
    return StreamInfo(
        stream_id="stream-spec-int",
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
        event_id=f"ev-tone-spec-{int(f0_hz)}",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=total,
        detected_start_sample=0,
        detected_end_sample=total,
        trigger_sample=total // 4,
    )


class TestContextIntegration:
    """Test spectral feature extraction on real SignalEvent and SignalContext instances."""

    def test_case_j_real_signal_context_integration(self, stream_1mhz: StreamInfo) -> None:
        """Test Case J: Real SignalContext with deterministic tone extracts valid finite spectral features."""
        config = AnalysisConfig(nfft=1000, spectral_window="rectangular")
        event = make_tone_event(stream_1mhz, f0_hz=100_000.0, nfft=1000)
        ctx = SignalContext(event, config=config)
        extractor = SpectralFeatureExtractor()

        fset = extractor.extract(ctx)

        assert isinstance(fset, FeatureSet)
        assert fset.extractor == "spectral"
        assert fset.extractor_version == "1.0"
        assert len(fset.features) == 4

        entropy = fset.features["spectral.entropy"]
        flatness = fset.features["spectral.flatness"]
        r85 = fset.features["spectral.rolloff_85_offset_hz"]
        r95 = fset.features["spectral.rolloff_95_offset_hz"]

        # 1. Entropy
        assert entropy.valid is True
        assert entropy.unit is None
        assert 0.0 <= entropy.value <= 1.0
        assert isinstance(entropy.value, float)

        # 2. Flatness
        assert flatness.valid is True
        assert flatness.unit is None
        assert 0.0 <= flatness.value <= 1.0
        assert isinstance(flatness.value, float)

        # 3. Rolloff 85%
        assert r85.valid is True
        assert r85.unit == "Hz"
        assert np.isfinite(r85.value)
        assert isinstance(r85.value, float)

        # 4. Rolloff 95%
        assert r95.valid is True
        assert r95.unit == "Hz"
        assert np.isfinite(r95.value)
        assert isinstance(r95.value, float)

        # Monotonicity of rolloffs
        assert r95.value >= r85.value

        # PSD cache was populated and is reused on repeated call
        assert "psd" in ctx.cached_keys
        fset2 = extractor.extract(ctx)
        for k in fset.features:
            assert fset.features[k].value == fset2.features[k].value
