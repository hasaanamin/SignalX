"""Integration tests demonstrating reuse and chaining of DSP primitives."""

import numpy as np
import pytest

from signal_processing.core import AutocorrelationResult, PhaseResult, PSDResult
from signal_processing.transforms import (
    FFTResult,
    compute_autocorrelation,
    compute_envelope,
    compute_fft,
    compute_instantaneous_frequency,
    compute_mean_power_dbfs,
    compute_mean_power_linear,
    compute_phase,
    compute_power,
    compute_psd,
    compute_psd_from_fft,
    linear_power_to_dbfs,
)


class TestTransformIntegration:
    """Demonstrate intended compositional reuse across DSP primitives."""

    def test_phase_to_instantaneous_frequency_pipeline(self) -> None:
        """Verify: IQ -> compute_phase -> compute_instantaneous_frequency.

        Demonstrates that instantaneous frequency directly consumes the PhaseResult
        contract produced by compute_phase without recomputing phase internally.
        """
        fs = 2_000_000.0
        f_tone = 250_000.0
        n = np.arange(1000)
        iq = np.exp(1j * 2.0 * np.pi * f_tone * n / fs).astype(np.complex128)

        # Step 1: Upstream Phase Transform
        phase_res = compute_phase(iq)
        assert isinstance(phase_res, PhaseResult)

        # Step 2: Downstream Frequency Transform reusing PhaseResult
        f_inst = compute_instantaneous_frequency(phase_res, sample_rate_hz=fs)

        assert len(f_inst) == len(iq) - 1
        np.testing.assert_allclose(f_inst, f_tone, atol=1e-5)

    def test_fft_to_psd_pipeline(self) -> None:
        """Verify: IQ -> compute_fft -> compute_psd_from_fft.

        Demonstrates that Power Spectral Density directly consumes the FFTResult
        contract produced by compute_fft, avoiding redundant Fourier transforms.
        """
        fs = 1_000_000.0
        f_carrier = -150_000.0
        nfft = 1024
        n = np.arange(nfft)
        iq = (0.8 * np.exp(1j * 2.0 * np.pi * f_carrier * n / fs)).astype(np.complex128)

        # Step 1: Upstream FFT Transform
        fft_res = compute_fft(iq, sample_rate_hz=fs, nfft=nfft, window="hann", centered=True)
        assert isinstance(fft_res, FFTResult)

        # Step 2: Downstream PSD Transform consuming FFTResult
        psd_res = compute_psd_from_fft(fft_res)
        assert isinstance(psd_res, PSDResult)

        # Direct convenience function produces matching PSD
        psd_convenience = compute_psd(iq, sample_rate_hz=fs, nfft=nfft, window="hann", centered=True)
        np.testing.assert_array_equal(psd_res.frequency_offset_hz, psd_convenience.frequency_offset_hz)
        np.testing.assert_allclose(psd_res.power_linear, psd_convenience.power_linear, rtol=1e-12)

        # Verify peak frequency
        peak_idx = int(np.argmax(psd_res.power_linear))
        peak_f = psd_res.frequency_offset_hz[peak_idx]
        assert pytest.approx(peak_f, abs=1000.0) == f_carrier

    def test_all_primitives_coexist_statelessly(self) -> None:
        """Verify that all 7 DSP primitives can execute on the same IQ sequence independently and statelessly."""
        fs = 1_000_000.0
        f0 = 100_000.0
        n = np.arange(512)
        amp = 0.5
        iq = (amp * np.exp(1j * 2.0 * np.pi * f0 * n / fs)).astype(np.complex64)
        iq_snapshot = iq.copy()

        # 1. Envelope
        env = compute_envelope(iq)
        assert len(env) == 512
        np.testing.assert_allclose(env, amp, atol=1e-6)

        # 2. Power
        p_lin = compute_power(iq)
        p_mean = compute_mean_power_linear(iq)
        p_mean_dbfs = compute_mean_power_dbfs(iq)
        p_dbfs_arr = linear_power_to_dbfs(p_lin)
        assert pytest.approx(p_mean, rel=1e-5) == amp ** 2
        assert pytest.approx(p_mean_dbfs, abs=1e-3) == 10.0 * np.log10(amp ** 2)
        assert len(p_dbfs_arr) == 512

        # 3. Phase
        phase = compute_phase(iq)
        assert isinstance(phase, PhaseResult)

        # 4. Instantaneous Frequency
        f_inst = compute_instantaneous_frequency(phase, fs)
        assert len(f_inst) == 511
        np.testing.assert_allclose(f_inst, f0, atol=5.0)

        # 5. FFT
        fft_res = compute_fft(iq, fs, window="rectangular", centered=True)
        assert isinstance(fft_res, FFTResult)

        # 6. PSD
        psd_res = compute_psd_from_fft(fft_res)
        assert isinstance(psd_res, PSDResult)

        # 7. Autocorrelation
        autocorr = compute_autocorrelation(iq, max_lag_samples=64, normalized=True)
        assert isinstance(autocorr, AutocorrelationResult)
        assert autocorr.values[0] == 1.0 + 0.0j

        # Input data was completely untouched throughout all transformations
        np.testing.assert_array_equal(iq, iq_snapshot)
