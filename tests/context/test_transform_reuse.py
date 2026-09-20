"""Tests verifying upstream DSP dependency reuse and call-order independence."""

from unittest.mock import MagicMock

import numpy as np
import pytest

import signal_processing.context.signal_context as sc_module
from signal_processing.context import SignalContext
from signal_processing.core import SignalEvent, StreamInfo


@pytest.fixture
def sample_event() -> SignalEvent:
    """Fixture providing a deterministic 200-sample SignalEvent."""
    stream = StreamInfo(
        stream_id="stream-reuse-test",
        source_id="test-src",
        sample_rate_hz=500_000.0,
        center_frequency_hz=100_000_000.0,
    )
    n = np.arange(200)
    iq = np.exp(1j * 2.0 * np.pi * 25_000.0 * n / 500_000.0).astype(np.complex64)

    return SignalEvent(
        event_id="ev-reuse-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=200,
        detected_start_sample=20,
        detected_end_sample=180,
        trigger_sample=50,
    )


class TestDependencyReuse:
    """Verify that downstream transforms directly consume upstream cached contracts."""

    def test_instantaneous_frequency_reuses_phase(
        self, sample_event: SignalEvent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """instantaneous_frequency must trigger compute_phase once and populate phase cache."""
        mock_phase = MagicMock(side_effect=sc_module.compute_phase)
        mock_inst_freq = MagicMock(side_effect=sc_module.compute_instantaneous_frequency)
        monkeypatch.setattr(sc_module, "compute_phase", mock_phase)
        monkeypatch.setattr(sc_module, "compute_instantaneous_frequency", mock_inst_freq)

        ctx = SignalContext(sample_event)

        # Step 1: Call instantaneous_frequency()
        f_inst = ctx.instantaneous_frequency()
        assert len(f_inst) == 159
        mock_phase.assert_called_once()
        mock_inst_freq.assert_called_once()
        assert "phase" in ctx.cached_keys
        assert "instantaneous_frequency" in ctx.cached_keys

        # Step 2: Call phase() afterwards -> must NOT recompute phase
        phase_res = ctx.phase()
        assert mock_phase.call_count == 1
        assert len(phase_res.wrapped_rad) == 160

        # Step 3: Call instantaneous_frequency() again -> no additional calls
        _ = ctx.instantaneous_frequency()
        assert mock_phase.call_count == 1
        assert mock_inst_freq.call_count == 1

    def test_psd_reuses_fft(
        self, sample_event: SignalEvent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """psd must trigger compute_fft once and compute_psd_from_fft once, populating both caches."""
        mock_fft = MagicMock(side_effect=sc_module.compute_fft)
        mock_psd = MagicMock(side_effect=sc_module.compute_psd_from_fft)
        monkeypatch.setattr(sc_module, "compute_fft", mock_fft)
        monkeypatch.setattr(sc_module, "compute_psd_from_fft", mock_psd)

        ctx = SignalContext(sample_event)

        # Step 1: Call psd()
        psd_res = ctx.psd()
        assert len(psd_res.power_linear) == 160
        mock_fft.assert_called_once()
        mock_psd.assert_called_once()
        assert "fft" in ctx.cached_keys
        assert "psd" in ctx.cached_keys

        # Step 2: Call fft() afterwards -> must NOT recompute FFT
        fft_res = ctx.fft()
        assert mock_fft.call_count == 1
        assert len(fft_res.spectrum) == 160

        # Step 3: Call psd() again -> no additional calls
        _ = ctx.psd()
        assert mock_fft.call_count == 1
        assert mock_psd.call_count == 1


class TestOrderIndependence:
    """Verify that caching and reuse produce identical results regardless of invocation order."""

    def test_fft_then_psd_order(
        self, sample_event: SignalEvent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Case B: Calling fft() first then psd() must execute compute_fft once and compute_psd_from_fft once."""
        mock_fft = MagicMock(side_effect=sc_module.compute_fft)
        mock_psd = MagicMock(side_effect=sc_module.compute_psd_from_fft)
        monkeypatch.setattr(sc_module, "compute_fft", mock_fft)
        monkeypatch.setattr(sc_module, "compute_psd_from_fft", mock_psd)

        ctx = SignalContext(sample_event)

        # 1. FFT first
        fft_res = ctx.fft()
        assert mock_fft.call_count == 1
        assert mock_psd.call_count == 0

        # 2. PSD second
        psd_res = ctx.psd()
        assert mock_fft.call_count == 1
        assert mock_psd.call_count == 1

        assert "fft" in ctx.cached_keys
        assert "psd" in ctx.cached_keys
        assert len(fft_res.spectrum) == len(psd_res.power_linear)

    def test_phase_then_instantaneous_frequency_order(
        self, sample_event: SignalEvent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Case B: Calling phase() first then instantaneous_frequency() must execute compute_phase once."""
        mock_phase = MagicMock(side_effect=sc_module.compute_phase)
        mock_inst_freq = MagicMock(side_effect=sc_module.compute_instantaneous_frequency)
        monkeypatch.setattr(sc_module, "compute_phase", mock_phase)
        monkeypatch.setattr(sc_module, "compute_instantaneous_frequency", mock_inst_freq)

        ctx = SignalContext(sample_event)

        # 1. Phase first
        phase_res = ctx.phase()
        assert mock_phase.call_count == 1
        assert mock_inst_freq.call_count == 0

        # 2. Instantaneous frequency second
        f_inst = ctx.instantaneous_frequency()
        assert mock_phase.call_count == 1
        assert mock_inst_freq.call_count == 1

        assert "phase" in ctx.cached_keys
        assert "instantaneous_frequency" in ctx.cached_keys
        assert len(f_inst) == len(phase_res.wrapped_rad) - 1
