"""End-to-end integration tests verifying numerical correctness and error propagation."""

import numpy as np
import pytest

from signal_processing.context import AnalysisConfig, SignalContext
from signal_processing.core import SignalEvent, StreamInfo
from signal_processing.transforms import TransformError


@pytest.fixture
def stream_1mhz() -> StreamInfo:
    """Fixture providing a 1 MHz StreamInfo."""
    return StreamInfo(
        stream_id="stream-integration-test",
        source_id="synthetic-tone-gen",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )


def _make_tone_event(
    stream: StreamInfo,
    tone_freq_hz: float,
    amplitude: float = 1.0,
    n_samples: int = 1000,
) -> SignalEvent:
    """Helper creating a pure tone SignalEvent with 100 pre and 100 post margin samples."""
    total = n_samples + 200
    n = np.arange(total)
    fs = stream.sample_rate_hz
    tone = (amplitude * np.exp(1j * 2.0 * np.pi * tone_freq_hz * n / fs)).astype(np.complex64)

    iq_data = np.zeros(total, dtype=np.complex64)
    iq_data[100 : 100 + n_samples] = tone[100 : 100 + n_samples]

    det_start = 100
    det_end = 100 + n_samples
    trigger = det_start + (n_samples // 2 if n_samples > 1 else 0)

    return SignalEvent(
        event_id=f"ev-tone-{int(tone_freq_hz)}",
        stream=stream,
        iq=iq_data,
        event_start_sample=0,
        event_end_sample=total,
        detected_start_sample=det_start,
        detected_end_sample=det_end,
        trigger_sample=trigger,
    )


class TestNumericalIntegration:
    """Test full DSP transform pipeline on known deterministic RF signals."""

    def test_positive_tone_numerical_accuracy(self, stream_1mhz: StreamInfo) -> None:
        """Verify +100 kHz tone DSP metrics across all primitives."""
        f0 = 100_000.0
        event = _make_tone_event(stream_1mhz, tone_freq_hz=f0, amplitude=1.0, n_samples=1000)
        ctx = SignalContext(event, AnalysisConfig(spectral_window="hann", centered_spectrum=True))

        # 1. Envelope
        env = ctx.envelope()
        assert len(env) == 1000
        np.testing.assert_allclose(env, 1.0, atol=1e-5)

        # 2. Power
        p = ctx.power()
        assert len(p) == 1000
        np.testing.assert_allclose(p, 1.0, atol=1e-5)

        # 3. Mean power
        assert pytest.approx(ctx.mean_power_linear(), rel=1e-5) == 1.0
        assert pytest.approx(ctx.mean_power_dbfs(), abs=1e-3) == 0.0

        # 4. Instantaneous frequency
        f_inst = ctx.instantaneous_frequency()
        assert len(f_inst) == 999
        np.testing.assert_allclose(f_inst, f0, atol=10.0)

        # 5. FFT
        fft_res = ctx.fft()
        assert len(fft_res.spectrum) == 1000
        peak_idx = int(np.argmax(np.abs(fft_res.spectrum)))
        peak_freq = fft_res.frequency_offset_hz[peak_idx]
        assert pytest.approx(peak_freq, abs=1000.0) == f0

        # 6. PSD
        psd_res = ctx.psd()
        assert len(psd_res.power_linear) == 1000
        psd_peak_idx = int(np.argmax(psd_res.power_linear))
        psd_peak_freq = psd_res.frequency_offset_hz[psd_peak_idx]
        assert pytest.approx(psd_peak_freq, abs=1000.0) == f0

        # 7. Autocorrelation
        autocorr = ctx.autocorrelation()
        assert autocorr.values[0] == 1.0 + 0.0j

    def test_negative_tone_numerical_accuracy(self, stream_1mhz: StreamInfo) -> None:
        """Verify -100 kHz tone DSP metrics across frequency and spectral primitives."""
        f0 = -100_000.0
        event = _make_tone_event(stream_1mhz, tone_freq_hz=f0, amplitude=0.5, n_samples=1000)
        ctx = SignalContext(event, AnalysisConfig(spectral_window="hann", centered_spectrum=True))

        # Power
        assert pytest.approx(ctx.mean_power_linear(), rel=1e-5) == 0.25
        assert pytest.approx(ctx.mean_power_dbfs(), abs=1e-3) == 10.0 * np.log10(0.25)

        # Instantaneous frequency
        f_inst = ctx.instantaneous_frequency()
        np.testing.assert_allclose(f_inst, f0, atol=10.0)


        # FFT & PSD peaks
        fft_res = ctx.fft()
        peak_idx = int(np.argmax(np.abs(fft_res.spectrum)))
        assert pytest.approx(fft_res.frequency_offset_hz[peak_idx], abs=1000.0) == f0

        psd_res = ctx.psd()
        psd_peak_idx = int(np.argmax(psd_res.power_linear))
        assert pytest.approx(psd_res.frequency_offset_hz[psd_peak_idx], abs=1000.0) == f0



class TestTransformErrorPropagation:
    """Verify that transform configuration errors propagate deterministically with full detail."""

    def test_reject_nfft_smaller_than_signal_length(self, stream_1mhz: StreamInfo) -> None:
        """When config.nfft < signal length, compute_fft must raise TransformError on request."""
        event = _make_tone_event(stream_1mhz, 50_000.0, n_samples=500)
        ctx = SignalContext(event, AnalysisConfig(nfft=256))

        with pytest.raises(TransformError, match="truncation is not permitted"):
            ctx.fft()

        with pytest.raises(TransformError, match="truncation is not permitted"):
            ctx.psd()

    def test_reject_insufficient_samples_for_instantaneous_frequency(
        self, stream_1mhz: StreamInfo
    ) -> None:
        """When signal region has only 1 sample, instantaneous_frequency must raise TransformError."""
        event = SignalEvent(
            event_id="ev-short",
            stream=stream_1mhz,
            iq=np.array([1.0 + 0j, 1.0 + 1j, 0.0 + 0j], dtype=np.complex64),
            event_start_sample=0,
            event_end_sample=3,
            detected_start_sample=1,
            detected_end_sample=2,  # 1 sample
            trigger_sample=1,
        )
        ctx = SignalContext(event)

        with pytest.raises(TransformError, match="at least 2 phase samples"):
            ctx.instantaneous_frequency()

    def test_reject_autocorrelation_max_lag_exceeding_signal(
        self, stream_1mhz: StreamInfo
    ) -> None:
        """When autocorrelation max_lag >= signal length, compute_autocorrelation must raise TransformError."""
        event = _make_tone_event(stream_1mhz, 50_000.0, n_samples=50)
        ctx = SignalContext(event, AnalysisConfig(autocorrelation_max_lag_samples=100))

        with pytest.raises(TransformError, match="max_lag_samples must satisfy"):
            ctx.autocorrelation()

    def test_reject_zero_energy_normalized_autocorrelation(
        self, stream_1mhz: StreamInfo
    ) -> None:
        """When signal has zero energy and normalization is requested, TransformError must be raised."""
        event = SignalEvent(
            event_id="ev-zero",
            stream=stream_1mhz,
            iq=np.zeros(20, dtype=np.complex64),
            event_start_sample=0,
            event_end_sample=20,
            detected_start_sample=5,
            detected_end_sample=15,
            trigger_sample=10,
        )
        ctx = SignalContext(event, AnalysisConfig(autocorrelation_normalized=True))

        with pytest.raises(TransformError, match="zero-energy"):
            ctx.autocorrelation()
