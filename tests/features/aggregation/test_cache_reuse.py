"""Tests verifying shared transform cache reuse across feature extractors."""

import numpy as np
import pytest

import signal_processing.transforms.autocorrelation as ac_module
import signal_processing.transforms.envelope as env_module
import signal_processing.transforms.fft as fft_module
import signal_processing.transforms.frequency as freq_module
import signal_processing.transforms.phase as phase_module
import signal_processing.transforms.psd as psd_module
from signal_processing.context import SignalContext
from signal_processing.core import SignalEvent, StreamInfo
from signal_processing.features.aggregation import (
    FeatureAggregator,
    build_default_feature_registry,
)


@pytest.fixture
def complex_test_event() -> SignalEvent:
    """Fixture providing a deterministic complex IQ signal event for transform testing."""
    stream = StreamInfo(
        stream_id="stream-cache-reuse",
        source_id="synthetic-src",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    n = np.arange(512)
    iq = (
        0.7 * np.exp(1j * 2.0 * np.pi * 50_000.0 * n / 1_000_000.0)
        + 0.3 * np.exp(1j * 2.0 * np.pi * 120_000.0 * n / 1_000_000.0)
    ).astype(np.complex64)
    total = len(iq)

    return SignalEvent(
        event_id="ev-cache-reuse-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=total,
        detected_start_sample=16,
        detected_end_sample=total - 16,
        trigger_sample=64,
    )


class TestTransformCacheReuse:
    """Verify that multiple extractors sharing one SignalContext reuse cached transforms without redundant recalculation."""

    def test_shared_cache_invocation_counts(
        self, complex_test_event: SignalEvent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Shared transforms (FFT, PSD, envelope, phase, autocorrelation) must be computed at most once across all 10 extractors."""
        ctx = SignalContext(complex_test_event)

        fft_calls = 0
        orig_compute_fft = fft_module.compute_fft

        def spy_fft(*args: object, **kwargs: object) -> object:
            nonlocal fft_calls
            fft_calls += 1
            return orig_compute_fft(*args, **kwargs)

        psd_calls = 0
        orig_compute_psd = psd_module.compute_psd_from_fft

        def spy_psd(*args: object, **kwargs: object) -> object:
            nonlocal psd_calls
            psd_calls += 1
            return orig_compute_psd(*args, **kwargs)

        env_calls = 0
        orig_compute_env = env_module.compute_envelope

        def spy_env(*args: object, **kwargs: object) -> object:
            nonlocal env_calls
            env_calls += 1
            return orig_compute_env(*args, **kwargs)

        phase_calls = 0
        orig_compute_phase = phase_module.compute_phase

        def spy_phase(*args: object, **kwargs: object) -> object:
            nonlocal phase_calls
            phase_calls += 1
            return orig_compute_phase(*args, **kwargs)

        ac_calls = 0
        orig_compute_ac = ac_module.compute_autocorrelation

        def spy_ac(*args: object, **kwargs: object) -> object:
            nonlocal ac_calls
            ac_calls += 1
            return orig_compute_ac(*args, **kwargs)

        inst_freq_calls = 0
        orig_compute_inst_freq = freq_module.compute_instantaneous_frequency

        def spy_inst_freq(*args: object, **kwargs: object) -> object:
            nonlocal inst_freq_calls
            inst_freq_calls += 1
            return orig_compute_inst_freq(*args, **kwargs)

        monkeypatch.setattr(fft_module, "compute_fft", spy_fft)
        # Also patch context module where these functions are imported
        monkeypatch.setattr("signal_processing.context.signal_context.compute_fft", spy_fft)
        monkeypatch.setattr("signal_processing.context.signal_context.compute_psd_from_fft", spy_psd)
        monkeypatch.setattr("signal_processing.context.signal_context.compute_envelope", spy_env)
        monkeypatch.setattr("signal_processing.context.signal_context.compute_phase", spy_phase)
        monkeypatch.setattr("signal_processing.context.signal_context.compute_autocorrelation", spy_ac)
        monkeypatch.setattr("signal_processing.context.signal_context.compute_instantaneous_frequency", spy_inst_freq)

        # Before extraction: cache is empty
        assert ctx.cached_keys == frozenset()

        aggregator = FeatureAggregator(build_default_feature_registry())
        vector = aggregator.extract(ctx)

        # Verify all 56 features were extracted
        assert len(vector.features) == 56

        # Shared transforms: FFT and PSD shared by frequency, bandwidth, spectral -> computed exactly once!
        assert fft_calls == 1
        assert psd_calls == 1

        # Envelope shared by amplitude and statistics -> computed exactly once!
        assert env_calls == 1

        # Phase used by phase extractor -> computed exactly once!
        assert phase_calls == 1

        # Autocorrelation used by correlation extractor -> computed exactly once!
        assert ac_calls == 1

        # Instantaneous frequency is never needed by version-1.0 extractors
        assert inst_freq_calls == 0
        assert "instantaneous_frequency" not in ctx.cached_keys
