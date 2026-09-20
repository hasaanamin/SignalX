"""Integration tests connecting SignalEvent -> SignalContext -> CorrelationFeatureExtractor."""

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, FeatureValue, SignalEvent, StreamInfo
from signal_processing.features.correlation import CorrelationFeatureExtractor


@pytest.fixture
def stream_1mhz() -> StreamInfo:
    """Standard StreamInfo fixture with 1 MHz sampling rate."""
    return StreamInfo(
        stream_id="stream-corr-int",
        source_id="synthetic-src",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )


def make_repeating_block_event(stream: StreamInfo, block_length: int = 16, num_blocks: int = 8) -> SignalEvent:
    """Create a deterministic repeating binary pattern event with a dominant positive lag peak at block_length."""
    rng = np.random.RandomState(42)
    block = rng.choice([-1.0, 1.0], size=block_length).astype(np.float32)
    iq = np.tile(block, num_blocks).astype(np.complex64)
    total = len(iq)

    return SignalEvent(
        event_id="ev-repeating-block-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=total,
        detected_start_sample=0,
        detected_end_sample=total,
        trigger_sample=0,
    )


class TestContextIntegration:
    """Test correlation feature extraction with real SignalEvent and SignalContext instances."""

    def test_case_e_real_repeating_pattern(self, stream_1mhz: StreamInfo) -> None:
        """Test Case E: Repeating deterministic pattern with period 16 produces a peak at lag 16."""
        event = make_repeating_block_event(stream_1mhz, block_length=16, num_blocks=8)
        ctx = SignalContext(event)
        extractor = CorrelationFeatureExtractor()

        fset = extractor.extract(ctx)

        peak_mag = fset.features["correlation.nonzero_peak_magnitude"]
        peak_lag_samples = fset.features["correlation.nonzero_peak_lag_samples"]
        peak_lag_s = fset.features["correlation.nonzero_peak_lag_s"]
        peak_phase = fset.features["correlation.nonzero_peak_phase_rad"]
        mean_mag = fset.features["correlation.mean_nonzero_magnitude"]
        rms_mag = fset.features["correlation.rms_nonzero_magnitude"]

        assert peak_mag.valid is True
        assert peak_mag.value > 0.0
        # For 8 blocks of 16 samples, overlap at lag 16 has 7 full matching blocks out of 8 = 7/8 = 0.875
        assert pytest.approx(peak_mag.value, rel=1e-5) == 0.875

        assert peak_lag_samples.valid is True
        assert peak_lag_samples.value == 16.0

        assert peak_lag_s.valid is True
        assert pytest.approx(peak_lag_s.value, rel=1e-9) == 16.0 / 1_000_000.0

        assert peak_phase.valid is True
        assert pytest.approx(peak_phase.value, abs=1e-5) == 0.0

        assert mean_mag.valid is True
        assert 0.0 < mean_mag.value <= 1.0

        assert rms_mag.valid is True
        assert 0.0 < rms_mag.value <= 1.0

    def test_feature_contract_types_and_units(self, stream_1mhz: StreamInfo) -> None:
        """Verify exact FeatureSet container, 6 canonical keys, Python float types, and units."""
        event = make_repeating_block_event(stream_1mhz, block_length=16, num_blocks=8)
        ctx = SignalContext(event)
        extractor = CorrelationFeatureExtractor()

        fset = extractor.extract(ctx)

        assert isinstance(fset, FeatureSet)
        assert fset.extractor == "correlation"
        assert fset.extractor_version == "1.0"
        assert len(fset.features) == 6

        expected_units = {
            "correlation.nonzero_peak_magnitude": None,
            "correlation.nonzero_peak_lag_samples": "samples",
            "correlation.nonzero_peak_lag_s": "s",
            "correlation.nonzero_peak_phase_rad": "rad",
            "correlation.mean_nonzero_magnitude": None,
            "correlation.rms_nonzero_magnitude": None,
        }

        assert set(fset.features.keys()) == set(expected_units.keys())

        for key, expected_unit in expected_units.items():
            fv = fset.features[key]
            assert isinstance(fv, FeatureValue)
            assert fv.valid is True
            assert fv.unit == expected_unit
            assert type(fv.value) is float  # Strict Python float, not np.float32/np.float64
            assert not isinstance(fv.value, (np.floating, np.integer))

    def test_cache_reuse_and_idempotence(self, stream_1mhz: StreamInfo) -> None:
        """Verify that autocorrelation result is retrieved from context cache and repeated extraction is idempotent."""
        event = make_repeating_block_event(stream_1mhz)
        ctx = SignalContext(event)
        extractor = CorrelationFeatureExtractor()

        # Before extraction, neither autocorrelation nor power is in cache
        assert "autocorrelation" not in ctx.cached_keys

        fset_1 = extractor.extract(ctx)

        # Autocorrelation should now be cached
        assert "autocorrelation" in ctx.cached_keys
        cached_ac = ctx.autocorrelation()

        # Second extraction on same context
        fset_2 = extractor.extract(ctx)

        # The cached object must be unchanged and reused
        assert ctx.autocorrelation() is cached_ac

        for key in fset_1.features:
            assert fset_1.features[key].value == fset_2.features[key].value
            assert fset_1.features[key].valid == fset_2.features[key].valid
            assert fset_1.features[key].unit == fset_2.features[key].unit

    def test_no_input_mutation(self, stream_1mhz: StreamInfo) -> None:
        """Verify that SignalEvent and AutocorrelationResult arrays are not mutated."""
        event = make_repeating_block_event(stream_1mhz)
        orig_sig_iq = event.signal_iq.copy()
        orig_event_iq = event.iq.copy()

        ctx = SignalContext(event)
        extractor = CorrelationFeatureExtractor()
        _ = extractor.extract(ctx)

        np.testing.assert_array_equal(event.signal_iq, orig_sig_iq)
        np.testing.assert_array_equal(event.iq, orig_event_iq)
        assert event.signal_iq.flags.writeable is False
        assert event.iq.flags.writeable is False

        ac = ctx.autocorrelation()
        assert ac.lag_samples.flags.writeable is False
        assert ac.values.flags.writeable is False
