"""Tests for SignalContext lazy computation and memoization behavior."""

from unittest.mock import MagicMock

import numpy as np
import pytest

import signal_processing.context.signal_context as sc_module
from signal_processing.context import AnalysisConfig, SignalContext
from signal_processing.core import SignalEvent, StreamInfo


@pytest.fixture
def sample_event() -> SignalEvent:
    """Fixture providing a valid 100-sample SignalEvent."""
    stream = StreamInfo(
        stream_id="stream-caching-test",
        source_id="test-src",
        sample_rate_hz=100_000.0,
        center_frequency_hz=10_000_000.0,
    )
    n = np.arange(100)
    iq = np.exp(1j * 2.0 * np.pi * 10_000.0 * n / 100_000.0).astype(np.complex64)

    return SignalEvent(
        event_id="ev-caching-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=100,
        detected_start_sample=10,
        detected_end_sample=90,
        trigger_sample=20,
    )


class TestLazyComputation:
    """Verify that SignalContext does NOT calculate any transforms on construction."""

    def test_constructor_is_lazy(self, sample_event: SignalEvent, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constructor must not trigger any DSP transform functions."""
        mock_fft = MagicMock(side_effect=sc_module.compute_fft)
        mock_psd = MagicMock(side_effect=sc_module.compute_psd_from_fft)
        mock_phase = MagicMock(side_effect=sc_module.compute_phase)
        mock_env = MagicMock(side_effect=sc_module.compute_envelope)
        mock_power = MagicMock(side_effect=sc_module.compute_power)
        mock_autocorr = MagicMock(side_effect=sc_module.compute_autocorrelation)

        monkeypatch.setattr(sc_module, "compute_fft", mock_fft)
        monkeypatch.setattr(sc_module, "compute_psd_from_fft", mock_psd)
        monkeypatch.setattr(sc_module, "compute_phase", mock_phase)
        monkeypatch.setattr(sc_module, "compute_envelope", mock_env)
        monkeypatch.setattr(sc_module, "compute_power", mock_power)
        monkeypatch.setattr(sc_module, "compute_autocorrelation", mock_autocorr)

        ctx = SignalContext(sample_event)

        # No transforms called during instantiation
        mock_fft.assert_not_called()
        mock_psd.assert_not_called()
        mock_phase.assert_not_called()
        mock_env.assert_not_called()
        mock_power.assert_not_called()
        mock_autocorr.assert_not_called()

        assert ctx.cached_keys == frozenset()


class TestTransformCaching:
    """Verify that repeated transform requests return cached results without recomputation."""

    def test_envelope_caching(self, sample_event: SignalEvent, monkeypatch: pytest.MonkeyPatch) -> None:
        mock = MagicMock(side_effect=sc_module.compute_envelope)
        monkeypatch.setattr(sc_module, "compute_envelope", mock)

        ctx = SignalContext(sample_event)
        res1 = ctx.envelope()
        assert "envelope" in ctx.cached_keys
        assert mock.call_count == 1

        res2 = ctx.envelope()
        assert mock.call_count == 1
        assert res1 is res2

    def test_power_caching(self, sample_event: SignalEvent, monkeypatch: pytest.MonkeyPatch) -> None:
        mock = MagicMock(side_effect=sc_module.compute_power)
        monkeypatch.setattr(sc_module, "compute_power", mock)

        ctx = SignalContext(sample_event)
        res1 = ctx.power()
        assert "power" in ctx.cached_keys
        assert mock.call_count == 1

        res2 = ctx.power()
        assert mock.call_count == 1
        assert res1 is res2

    def test_mean_power_linear_caching(
        self, sample_event: SignalEvent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = MagicMock(side_effect=sc_module.compute_mean_power_linear)
        monkeypatch.setattr(sc_module, "compute_mean_power_linear", mock)

        ctx = SignalContext(sample_event)
        res1 = ctx.mean_power_linear()
        assert "mean_power_linear" in ctx.cached_keys
        assert mock.call_count == 1

        res2 = ctx.mean_power_linear()
        assert mock.call_count == 1
        assert res1 == res2

    def test_mean_power_dbfs_caching(
        self, sample_event: SignalEvent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = MagicMock(side_effect=sc_module.compute_mean_power_dbfs)
        monkeypatch.setattr(sc_module, "compute_mean_power_dbfs", mock)

        ctx = SignalContext(sample_event)
        res1 = ctx.mean_power_dbfs()
        assert "mean_power_dbfs" in ctx.cached_keys
        assert mock.call_count == 1

        res2 = ctx.mean_power_dbfs()
        assert mock.call_count == 1
        assert res1 == res2

    def test_phase_caching(self, sample_event: SignalEvent, monkeypatch: pytest.MonkeyPatch) -> None:
        mock = MagicMock(side_effect=sc_module.compute_phase)
        monkeypatch.setattr(sc_module, "compute_phase", mock)

        ctx = SignalContext(sample_event)
        res1 = ctx.phase()
        assert "phase" in ctx.cached_keys
        assert mock.call_count == 1

        res2 = ctx.phase()
        assert mock.call_count == 1
        assert res1 is res2

    def test_instantaneous_frequency_caching(
        self, sample_event: SignalEvent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = MagicMock(side_effect=sc_module.compute_instantaneous_frequency)
        monkeypatch.setattr(sc_module, "compute_instantaneous_frequency", mock)

        ctx = SignalContext(sample_event)
        res1 = ctx.instantaneous_frequency()
        assert "instantaneous_frequency" in ctx.cached_keys
        assert mock.call_count == 1

        res2 = ctx.instantaneous_frequency()
        assert mock.call_count == 1
        assert res1 is res2

    def test_fft_caching(self, sample_event: SignalEvent, monkeypatch: pytest.MonkeyPatch) -> None:
        mock = MagicMock(side_effect=sc_module.compute_fft)
        monkeypatch.setattr(sc_module, "compute_fft", mock)

        ctx = SignalContext(sample_event)
        res1 = ctx.fft()
        assert "fft" in ctx.cached_keys
        assert mock.call_count == 1

        res2 = ctx.fft()
        assert mock.call_count == 1
        assert res1 is res2

    def test_psd_caching(self, sample_event: SignalEvent, monkeypatch: pytest.MonkeyPatch) -> None:
        mock = MagicMock(side_effect=sc_module.compute_psd_from_fft)
        monkeypatch.setattr(sc_module, "compute_psd_from_fft", mock)

        ctx = SignalContext(sample_event)
        res1 = ctx.psd()
        assert "psd" in ctx.cached_keys
        assert mock.call_count == 1

        res2 = ctx.psd()
        assert mock.call_count == 1
        assert res1 is res2

    def test_autocorrelation_caching(
        self, sample_event: SignalEvent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = MagicMock(side_effect=sc_module.compute_autocorrelation)
        monkeypatch.setattr(sc_module, "compute_autocorrelation", mock)

        ctx = SignalContext(sample_event)
        res1 = ctx.autocorrelation()
        assert "autocorrelation" in ctx.cached_keys
        assert mock.call_count == 1

        res2 = ctx.autocorrelation()
        assert mock.call_count == 1
        assert res1 is res2


class TestContextIsolation:
    """Verify that caches are isolated per SignalContext instance."""

    def test_multiple_events_isolated(self, sample_event: SignalEvent) -> None:
        """Two different SignalContext instances must not share cache entries."""
        stream = sample_event.stream
        iq2 = np.ones(50, dtype=np.complex64)
        event2 = SignalEvent(
            event_id="ev-caching-002",
            stream=stream,
            iq=iq2,
            event_start_sample=0,
            event_end_sample=50,
            detected_start_sample=5,
            detected_end_sample=45,
            trigger_sample=10,
        )

        ctx_a = SignalContext(sample_event)
        ctx_b = SignalContext(event2)

        fft_a = ctx_a.fft()
        assert "fft" in ctx_a.cached_keys
        assert "fft" not in ctx_b.cached_keys

        fft_b = ctx_b.fft()
        assert "fft" in ctx_b.cached_keys

        assert fft_a is not fft_b
        assert len(fft_a.spectrum) != len(fft_b.spectrum)

    def test_same_event_different_configs_isolated(self, sample_event: SignalEvent) -> None:
        """Two contexts for the same event with different AnalysisConfigs must remain isolated."""
        ctx_hann = SignalContext(sample_event, AnalysisConfig(spectral_window="hann"))
        ctx_rect = SignalContext(sample_event, AnalysisConfig(spectral_window="rectangular"))

        fft_hann = ctx_hann.fft()
        fft_rect = ctx_rect.fft()

        assert fft_hann.window_name == "hann"
        assert fft_rect.window_name == "rectangular"
        assert fft_hann is not fft_rect
        assert not np.array_equal(fft_hann.spectrum, fft_rect.spectrum)
