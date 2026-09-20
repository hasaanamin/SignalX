"""Integration tests connecting SignalEvent -> SignalContext -> StatisticsFeatureExtractor."""

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, SignalEvent, StreamInfo
from signal_processing.features.statistics import StatisticsFeatureExtractor


@pytest.fixture
def stream_1mhz() -> StreamInfo:
    """Fixture providing standard StreamInfo at 1 MHz sample rate."""
    return StreamInfo(
        stream_id="stream-stat-int",
        source_id="synthetic-source",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )


def make_am_tone_event(
    stream: StreamInfo,
    carrier_hz: float = 100_000.0,
    mod_hz: float = 10_000.0,
    n_samples: int = 500,
) -> SignalEvent:
    """Helper creating a non-constant amplitude modulated SignalEvent."""
    n = np.arange(n_samples)
    fs = stream.sample_rate_hz
    # Envelope varies between 0.5 and 1.5
    env = 1.0 + 0.5 * np.cos(2.0 * np.pi * mod_hz * n / fs)
    carrier = np.exp(1j * 2.0 * np.pi * carrier_hz * n / fs)
    iq = (env * carrier).astype(np.complex64)

    return SignalEvent(
        event_id="ev-am-tone-stat",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=n_samples,
        detected_start_sample=50,
        detected_end_sample=450,
        trigger_sample=100,
    )


class TestContextIntegration:
    """Test statistics feature extraction on real SignalEvent and SignalContext instances."""

    def test_real_signal_context_integration(self, stream_1mhz: StreamInfo) -> None:
        """Real SignalContext with AM tone extracts all 9 valid statistical features."""
        event = make_am_tone_event(stream_1mhz)
        ctx = SignalContext(event)
        extractor = StatisticsFeatureExtractor()

        fset = extractor.extract(ctx)

        assert isinstance(fset, FeatureSet)
        assert fset.extractor == "statistics"
        assert fset.extractor_version == "1.0"
        assert len(fset.features) == 9

        # Skewness & Excess Kurtosis
        skew = fset.features["stats.amplitude.skewness"]
        assert skew.valid is True
        assert skew.unit is None
        assert np.isfinite(skew.value)
        assert isinstance(skew.value, float)

        kurt = fset.features["stats.amplitude.excess_kurtosis"]
        assert kurt.valid is True
        assert kurt.unit is None
        assert np.isfinite(kurt.value)
        assert isinstance(kurt.value, float)

        # Quantiles & IQR
        for key in [
            "stats.amplitude.q01",
            "stats.amplitude.q05",
            "stats.amplitude.q25",
            "stats.amplitude.q75",
            "stats.amplitude.q95",
            "stats.amplitude.q99",
            "stats.amplitude.iqr",
        ]:
            fv = fset.features[key]
            assert fv.valid is True
            assert fv.unit == "linear"
            assert np.isfinite(fv.value)
            assert isinstance(fv.value, float)

        # Envelope was cached and is reused on repeated call
        assert "envelope" in ctx.cached_keys
        fset2 = extractor.extract(ctx)
        for k in fset.features:
            assert fset.features[k].value == fset2.features[k].value
