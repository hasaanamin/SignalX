"""Tests for Fast Fourier Transform (FFT) primitive."""

import numpy as np
import pytest

from signal_processing.transforms.exceptions import TransformError
from signal_processing.transforms.fft import compute_fft
from signal_processing.transforms.results import FFTResult


class TestFFTComputation:
    """Test mathematical accuracy and spectrum properties of compute_fft."""

    def test_positive_frequency_tone_peak(self) -> None:
        """Complex tone at +100 kHz must produce maximum spectral magnitude at +100 kHz."""
        fs = 1_000_000.0
        f0 = 100_000.0
        nfft = 1000
        n = np.arange(nfft)
        iq = np.exp(1j * 2.0 * np.pi * f0 * n / fs).astype(np.complex128)

        res = compute_fft(iq, fs, nfft=nfft, window="rectangular", centered=True)

        assert isinstance(res, FFTResult)
        assert len(res.frequency_offset_hz) == nfft
        assert len(res.spectrum) == nfft
        assert res.centered is True

        peak_idx = int(np.argmax(np.abs(res.spectrum)))
        peak_freq = res.frequency_offset_hz[peak_idx]
        assert pytest.approx(peak_freq, abs=1.0) == f0

    def test_negative_frequency_tone_peak(self) -> None:
        """Complex tone at -100 kHz must produce peak at -100 kHz in centered FFT."""
        fs = 1_000_000.0
        f0 = -100_000.0
        nfft = 1000
        n = np.arange(nfft)
        iq = np.exp(1j * 2.0 * np.pi * f0 * n / fs).astype(np.complex128)

        res = compute_fft(iq, fs, nfft=nfft, window="rectangular", centered=True)

        peak_idx = int(np.argmax(np.abs(res.spectrum)))
        peak_freq = res.frequency_offset_hz[peak_idx]
        assert pytest.approx(peak_freq, abs=1.0) == f0

    def test_impulse_response_flat_spectrum(self) -> None:
        """Impulse delta[n] at n=0 with rectangular window must yield flat unit spectrum."""
        fs = 1000.0
        nfft = 64
        iq = np.zeros(nfft, dtype=np.complex128)
        iq[0] = 1.0 + 0.0j

        # Uncentered: DFT of delta[0] is exactly 1 at all bins
        res = compute_fft(iq, fs, nfft=nfft, window="rectangular", centered=False)
        np.testing.assert_allclose(np.abs(res.spectrum), 1.0, atol=1e-12)

    def test_centered_vs_uncentered(self) -> None:
        """Test frequency axis and spectrum arrangement between centered and uncentered."""
        fs = 1000.0
        nfft = 128
        iq = np.ones(nfft, dtype=np.complex64)

        res_centered = compute_fft(iq, fs, nfft=nfft, centered=True)
        res_uncentered = compute_fft(iq, fs, nfft=nfft, centered=False)

        assert res_centered.centered is True
        assert res_uncentered.centered is False

        # In uncentered FFT, DC (0 Hz) is at index 0
        assert res_uncentered.frequency_offset_hz[0] == 0.0
        # In centered FFT, DC is in the middle (index nfft//2)
        assert res_centered.frequency_offset_hz[nfft // 2] == 0.0

    def test_zero_padding(self) -> None:
        """FFT length greater than input length must zero-pad appropriately."""
        fs = 1000.0
        iq = np.ones(50, dtype=np.complex64)
        nfft = 200

        res = compute_fft(iq, fs, nfft=nfft)
        assert res.input_samples == 50
        assert res.nfft == 200
        assert len(res.spectrum) == 200
        assert len(res.frequency_offset_hz) == 200

    def test_windows_applied(self) -> None:
        """Test that different supported windows are correctly recognized and computed."""
        fs = 1000.0
        iq = np.ones(100, dtype=np.complex64)

        for win in ["rectangular", "hann", "hamming", "blackman", None]:
            res = compute_fft(iq, fs, window=win)
            assert res.window_energy > 0.0
            expected_name = "rectangular" if win is None else win.lower()
            assert res.window_name == expected_name

    def test_output_and_input_immutability(self) -> None:
        """FFTResult arrays must be read-only and input must not be mutated."""
        fs = 1000.0
        iq = np.array([1.0 + 1.0j, 2.0 - 2.0j], dtype=np.complex64)
        iq_copy = iq.copy()

        res = compute_fft(iq, fs)

        np.testing.assert_array_equal(iq, iq_copy)
        assert res.spectrum.flags.writeable is False
        assert res.frequency_offset_hz.flags.writeable is False

        with pytest.raises((ValueError, TypeError)):
            res.spectrum[0] = 0.0  # type: ignore[misc]


class TestFFTValidation:
    """Test validation and error handling for compute_fft."""

    def test_reject_nfft_smaller_than_input(self) -> None:
        iq = np.ones(100, dtype=np.complex64)
        with pytest.raises(TransformError, match="truncation is not permitted"):
            compute_fft(iq, 1000.0, nfft=50)

    @pytest.mark.parametrize("bad_nfft", [0, -10, 1.5, "128", True])
    def test_reject_invalid_nfft_types_and_values(self, bad_nfft: object) -> None:
        iq = np.ones(10, dtype=np.complex64)
        with pytest.raises(TransformError):
            compute_fft(iq, 1000.0, nfft=bad_nfft)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_fs", [0.0, -100.0, float("nan"), float("inf"), True])
    def test_reject_invalid_sample_rate(self, bad_fs: object) -> None:
        iq = np.ones(10, dtype=np.complex64)
        with pytest.raises(TransformError):
            compute_fft(iq, bad_fs)  # type: ignore[arg-type]

    def test_reject_unsupported_window(self) -> None:
        iq = np.ones(10, dtype=np.complex64)
        with pytest.raises(TransformError, match="Unsupported window"):
            compute_fft(iq, 1000.0, window="chebyshev")

    @pytest.mark.parametrize("bad_win", ["hann", "blackman"])
    def test_reject_pathological_short_zero_energy_windows(self, bad_win: str) -> None:
        """Windows of length 2 that yield zero or negligible energy must be rejected."""
        iq = np.ones(2, dtype=np.complex64)
        with pytest.raises(TransformError, match="zero or negligible energy"):
            compute_fft(iq, 1000.0, window=bad_win)

    def test_valid_short_windows_succeed(self) -> None:
        """Rectangular, hamming(2), and hann(3) with positive energy must succeed."""
        iq1 = np.ones(1, dtype=np.complex64)
        res1 = compute_fft(iq1, 1000.0, window="rectangular")
        assert res1.window_energy == 1.0

        iq2 = np.ones(2, dtype=np.complex64)
        res2 = compute_fft(iq2, 1000.0, window="hamming")
        assert res2.window_energy > 0.0

        iq3 = np.ones(3, dtype=np.complex64)
        res3 = compute_fft(iq3, 1000.0, window="hann")
        assert res3.window_energy > 0.0

    def test_get_window_validation(self) -> None:
        from signal_processing.transforms.windows import get_window
        with pytest.raises(TransformError, match="length must be >= 1"):
            get_window("rectangular", 0)
        with pytest.raises(TransformError, match="Window name must be a string or None"):
            get_window(123, 10)  # type: ignore[arg-type]
        with pytest.raises(TransformError, match="Window name must be a string or None"):
            get_window("", 10)
        with pytest.raises(TransformError, match="Unsupported window"):
            get_window("chebyshev", 10)



class TestFFTResultContract:
    """Test direct contract instantiation and validation for FFTResult."""

    def test_reject_invalid_sample_rate(self) -> None:
        freqs = np.zeros(4)
        spec = np.zeros(4, dtype=np.complex64)
        with pytest.raises(TransformError, match="sample_rate_hz"):
            FFTResult(frequency_offset_hz=freqs, spectrum=spec, sample_rate_hz=True, nfft=4, input_samples=4, window_name="hann", window_energy=1.0)
        with pytest.raises(TransformError, match="strictly positive"):
            FFTResult(frequency_offset_hz=freqs, spectrum=spec, sample_rate_hz=-100.0, nfft=4, input_samples=4, window_name="hann", window_energy=1.0)

    def test_reject_invalid_nfft_and_samples(self) -> None:
        freqs = np.zeros(4)
        spec = np.zeros(4, dtype=np.complex64)
        with pytest.raises(TransformError, match="nfft"):
            FFTResult(frequency_offset_hz=freqs, spectrum=spec, sample_rate_hz=1000.0, nfft=0, input_samples=4, window_name="hann", window_energy=1.0)
        with pytest.raises(TransformError, match="input_samples"):
            FFTResult(frequency_offset_hz=freqs, spectrum=spec, sample_rate_hz=1000.0, nfft=4, input_samples=-1, window_name="hann", window_energy=1.0)
        with pytest.raises(TransformError, match="nfft .* must be >= input_samples"):
            FFTResult(frequency_offset_hz=freqs, spectrum=spec, sample_rate_hz=1000.0, nfft=2, input_samples=4, window_name="hann", window_energy=1.0)

    def test_reject_invalid_window_metadata(self) -> None:
        freqs = np.zeros(4)
        spec = np.zeros(4, dtype=np.complex64)
        with pytest.raises(TransformError, match="window_name"):
            FFTResult(frequency_offset_hz=freqs, spectrum=spec, sample_rate_hz=1000.0, nfft=4, input_samples=4, window_name="   ", window_energy=1.0)
        with pytest.raises(TransformError, match="window_energy"):
            FFTResult(frequency_offset_hz=freqs, spectrum=spec, sample_rate_hz=1000.0, nfft=4, input_samples=4, window_name="hann", window_energy=True)
        with pytest.raises(TransformError, match="strictly positive"):
            FFTResult(frequency_offset_hz=freqs, spectrum=spec, sample_rate_hz=1000.0, nfft=4, input_samples=4, window_name="hann", window_energy=0.0)

    def test_reject_invalid_centered(self) -> None:
        freqs = np.zeros(4)
        spec = np.zeros(4, dtype=np.complex64)
        with pytest.raises(TransformError, match="centered"):
            FFTResult(frequency_offset_hz=freqs, spectrum=spec, sample_rate_hz=1000.0, nfft=4, input_samples=4, window_name="hann", window_energy=1.0, centered="yes")  # type: ignore[arg-type]

    def test_reject_invalid_arrays(self) -> None:
        freqs = np.zeros((4, 2))
        spec = np.zeros(4, dtype=np.complex64)
        with pytest.raises(TransformError, match="1-dimensional"):
            FFTResult(frequency_offset_hz=freqs, spectrum=spec, sample_rate_hz=1000.0, nfft=4, input_samples=4, window_name="hann", window_energy=1.0)

        freqs_complex = np.zeros(4, dtype=np.complex64)
        with pytest.raises(TransformError, match="real-valued"):
            FFTResult(frequency_offset_hz=freqs_complex, spectrum=spec, sample_rate_hz=1000.0, nfft=4, input_samples=4, window_name="hann", window_energy=1.0)

        freqs_nan = np.array([0.0, np.nan, 2.0, 3.0])
        with pytest.raises(TransformError, match="non-finite"):
            FFTResult(frequency_offset_hz=freqs_nan, spectrum=spec, sample_rate_hz=1000.0, nfft=4, input_samples=4, window_name="hann", window_energy=1.0)

        freqs_ok = np.zeros(4)
        spec_2d = np.zeros((4, 2), dtype=np.complex64)
        with pytest.raises(TransformError, match="1-dimensional"):
            FFTResult(frequency_offset_hz=freqs_ok, spectrum=spec_2d, sample_rate_hz=1000.0, nfft=4, input_samples=4, window_name="hann", window_energy=1.0)

        spec_nan = np.array([0.0, np.nan, 2.0, 3.0], dtype=np.complex64)
        with pytest.raises(TransformError, match="non-finite"):
            FFTResult(frequency_offset_hz=freqs_ok, spectrum=spec_nan, sample_rate_hz=1000.0, nfft=4, input_samples=4, window_name="hann", window_energy=1.0)

        spec_diff_len = np.zeros(5, dtype=np.complex64)
        with pytest.raises(TransformError, match="does not match"):
            FFTResult(frequency_offset_hz=freqs_ok, spectrum=spec_diff_len, sample_rate_hz=1000.0, nfft=4, input_samples=4, window_name="hann", window_energy=1.0)

        with pytest.raises(TransformError, match="does not match nfft"):
            FFTResult(frequency_offset_hz=freqs_ok, spectrum=spec, sample_rate_hz=1000.0, nfft=8, input_samples=4, window_name="hann", window_energy=1.0)


