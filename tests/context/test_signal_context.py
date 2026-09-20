"""Tests for SignalContext initialization, property delegation, and region isolation."""

import numpy as np
import pytest

from signal_processing.context import AnalysisConfig, ContextError, SignalContext
from signal_processing.core import SignalEvent, StreamInfo


@pytest.fixture
def sample_stream() -> StreamInfo:
    """Fixture providing a standard 1 MHz StreamInfo."""
    return StreamInfo(
        stream_id="stream-test-01",
        source_id="synthetic-rf-gen",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )


@pytest.fixture
def sample_event(sample_stream: StreamInfo) -> SignalEvent:
    """Fixture providing a deterministic SignalEvent with pre, signal, and post regions.

    Layout:
        total samples: 1000
        event_start: 0, event_end: 1000
        detected_start: 200, detected_end: 800 (signal length = 600)
        trigger: 300
    """
    fs = sample_stream.sample_rate_hz
    n = np.arange(1000)
    # Signal region (200..800) is a +100 kHz complex tone
    tone = np.exp(1j * 2.0 * np.pi * 100_000.0 * n / fs).astype(np.complex64)
    # Margins have low amplitude noise / zeros
    iq_data = np.zeros(1000, dtype=np.complex64)
    iq_data[200:800] = tone[200:800]

    return SignalEvent(
        event_id="ev-test-12345",
        stream=sample_stream,
        iq=iq_data,
        event_start_sample=0,
        event_end_sample=1000,
        detected_start_sample=200,
        detected_end_sample=800,
        trigger_sample=300,
        timestamp_ns=1_000_000_000,
        dropped_samples=0,
        complete=True,
    )


class TestSignalContextBasics:
    """Test basic initialization, property access, and contract enforcement."""

    def test_default_initialization(self, sample_event: SignalEvent) -> None:
        """Context initialized with only SignalEvent must default to default AnalysisConfig."""
        ctx = SignalContext(sample_event)

        assert ctx.event is sample_event
        assert isinstance(ctx.config, AnalysisConfig)
        assert ctx.config.spectral_window == "hann"
        assert ctx.stream is sample_event.stream
        assert ctx.event_id == sample_event.event_id
        assert ctx.sample_rate_hz == sample_event.stream.sample_rate_hz
        assert ctx.center_frequency_hz == sample_event.stream.center_frequency_hz

    def test_custom_config_initialization(self, sample_event: SignalEvent) -> None:
        """Context must retain explicitly provided AnalysisConfig."""
        custom_config = AnalysisConfig(spectral_window="blackman", nfft=1024)
        ctx = SignalContext(sample_event, config=custom_config)

        assert ctx.config is custom_config
        assert ctx.config.spectral_window == "blackman"
        assert ctx.config.nfft == 1024

    @pytest.mark.parametrize("bad_event", [None, "not_an_event", 1234, [1, 2, 3], np.zeros(10)])
    def test_reject_invalid_event(self, bad_event: object) -> None:
        """SignalContext must reject any event input that is not a SignalEvent."""
        with pytest.raises(ContextError, match="SignalEvent"):
            SignalContext(bad_event)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_config", ["not_a_config", 1234, [1, 2, 3], True])
    def test_reject_invalid_config(
        self, sample_event: SignalEvent, bad_config: object
    ) -> None:
        """SignalContext must reject config input that is not AnalysisConfig or None."""
        with pytest.raises(ContextError, match="AnalysisConfig"):
            SignalContext(sample_event, config=bad_config)  # type: ignore[arg-type]

    def test_signal_and_event_iq_properties(self, sample_event: SignalEvent) -> None:
        """Verify signal_iq and event_iq property delegations and dimensions."""
        ctx = SignalContext(sample_event)

        np.testing.assert_array_equal(ctx.signal_iq, sample_event.signal_iq)
        np.testing.assert_array_equal(ctx.event_iq, sample_event.iq)

        assert ctx.signal_num_samples == sample_event.signal_samples
        assert ctx.signal_num_samples == 600
        assert pytest.approx(ctx.signal_duration_s) == sample_event.signal_duration_s
        assert pytest.approx(ctx.signal_duration_s) == 600 / 1_000_000.0


class TestEventRegionIsolation:
    """Verify SignalContext transforms operate strictly on signal_iq, NOT entire event.iq."""

    def test_transforms_operate_only_on_signal_region(self, sample_stream: StreamInfo) -> None:
        """Construct event with distinct margin frequencies and verify active signal isolation.

        Layout:
            pre-margin (0..200): +400 kHz tone
            signal region (200..800): +100 kHz tone
            post-margin (800..1000): -300 kHz tone
        """
        fs = sample_stream.sample_rate_hz
        n = np.arange(1000)

        pre_tone = np.exp(1j * 2.0 * np.pi * 400_000.0 * n / fs).astype(np.complex64)
        sig_tone = np.exp(1j * 2.0 * np.pi * 100_000.0 * n / fs).astype(np.complex64)
        post_tone = np.exp(1j * 2.0 * np.pi * -300_000.0 * n / fs).astype(np.complex64)

        iq_data = np.zeros(1000, dtype=np.complex64)
        iq_data[0:200] = pre_tone[0:200]
        iq_data[200:800] = sig_tone[200:800]
        iq_data[800:1000] = post_tone[800:1000]

        event = SignalEvent(
            event_id="ev-region-test",
            stream=sample_stream,
            iq=iq_data,
            event_start_sample=0,
            event_end_sample=1000,
            detected_start_sample=200,
            detected_end_sample=800,
            trigger_sample=250,
        )

        ctx = SignalContext(event, AnalysisConfig(spectral_window="rectangular"))

        # Output sample lengths must match detected signal length (600), not full event length (1000)
        assert len(ctx.envelope()) == 600
        assert len(ctx.power()) == 600
        assert len(ctx.phase().wrapped_rad) == 600
        assert len(ctx.instantaneous_frequency()) == 599  # length N - 1

        # Instantaneous frequency must be purely +100 kHz (the signal tone), not +400 kHz or -300 kHz
        np.testing.assert_allclose(ctx.instantaneous_frequency(), 100_000.0, atol=5.0)


        # Spectral peak must correspond strictly to +100 kHz
        fft_res = ctx.fft()
        peak_idx = int(np.argmax(np.abs(fft_res.spectrum)))
        peak_f = fft_res.frequency_offset_hz[peak_idx]
        assert pytest.approx(peak_f, abs=2000.0) == 100_000.0

        psd_res = ctx.psd()
        psd_peak_idx = int(np.argmax(psd_res.power_linear))
        psd_peak_f = psd_res.frequency_offset_hz[psd_peak_idx]
        assert pytest.approx(psd_peak_f, abs=2000.0) == 100_000.0


class TestNoInputMutation:
    """Verify SignalContext execution never alters the underlying SignalEvent arrays."""

    def test_immutable_arrays_untouched(self, sample_event: SignalEvent) -> None:
        """Capture snapshots of event arrays and verify they are strictly unchanged after transforms."""
        sig_copy = sample_event.signal_iq.copy()
        event_copy = sample_event.iq.copy()

        ctx = SignalContext(sample_event)

        # Invoke all transforms
        _ = ctx.envelope()
        _ = ctx.power()
        _ = ctx.mean_power_linear()
        _ = ctx.mean_power_dbfs()
        _ = ctx.phase()
        _ = ctx.instantaneous_frequency()
        _ = ctx.fft()
        _ = ctx.psd()
        _ = ctx.autocorrelation()

        # Check byte-for-byte equality
        np.testing.assert_array_equal(sample_event.signal_iq, sig_copy)
        np.testing.assert_array_equal(sample_event.iq, event_copy)
        np.testing.assert_array_equal(ctx.signal_iq, sig_copy)
        np.testing.assert_array_equal(ctx.event_iq, event_copy)

        # Verify read-only flags
        assert sample_event.iq.flags.writeable is False
        assert sample_event.signal_iq.flags.writeable is False
        assert ctx.signal_iq.flags.writeable is False
        assert ctx.event_iq.flags.writeable is False
