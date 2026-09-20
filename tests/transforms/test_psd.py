"""Tests for Power Spectral Density (PSD) calculation primitives."""

import numpy as np
import pytest

from signal_processing.core import PSDResult
from signal_processing.transforms.exceptions import TransformError
from signal_processing.transforms.fft import compute_fft
from signal_processing.transforms.power import compute_mean_power_linear
from signal_processing.transforms.psd import compute_psd, compute_psd_from_fft


class TestPSDComputation:
    """Test Power Spectral Density mathematical correctness and contracts."""

    def test_tone_peak_location(self) -> None:
        """Sinusoid at +100 kHz must exhibit peak power density at +100 kHz."""
        fs = 1_000_000.0
        f0 = 100_000.0
        nfft = 1024
        n = np.arange(nfft)
        iq = np.exp(1j * 2.0 * np.pi * f0 * n / fs).astype(np.complex128)

        psd = compute_psd(iq, fs, nfft=nfft, window="hann", centered=True)

        assert isinstance(psd, PSDResult)
        assert len(psd.frequency_offset_hz) == nfft
        assert len(psd.power_linear) == nfft
        assert len(psd.power_dbfs_per_hz) == nfft

        peak_idx = int(np.argmax(psd.power_linear))
        peak_freq = psd.frequency_offset_hz[peak_idx]
        assert pytest.approx(peak_freq, abs=1000.0) == f0

    def test_power_conservation_rectangular_window(self) -> None:
        """Parseval power conservation: sum(PSD_linear) * df == mean(|iq|^2)."""
        rng = np.random.default_rng(42)
        fs = 500_000.0
        nfft = 1024

        # Deterministic random complex IQ
        real_part = rng.standard_normal(nfft)
        imag_part = rng.standard_normal(nfft)
        iq = (real_part + 1j * imag_part).astype(np.complex128)

        mean_power = compute_mean_power_linear(iq)
        psd = compute_psd(iq, fs, nfft=nfft, window="rectangular", centered=True)

        df = fs / nfft
        integrated_psd_power = float(np.sum(psd.power_linear) * df)

        # Theoretical identity holds to numerical precision for rectangular window
        assert pytest.approx(integrated_psd_power, rel=1e-5) == mean_power

    def test_zero_signal_flooring(self) -> None:
        """All-zero signal must produce non-negative linear PSD and cleanly floored dBFS/Hz."""
        fs = 100_000.0
        iq = np.zeros(256, dtype=np.complex64)
        floor_val = -250.0

        psd = compute_psd(iq, fs, floor_dbfs_per_hz=floor_val)

        assert np.all(psd.power_linear == 0.0)
        assert np.all(psd.power_dbfs_per_hz == floor_val)
        assert np.all(np.isfinite(psd.power_dbfs_per_hz))

    def test_compute_psd_and_compute_psd_from_fft_consistency(self) -> None:
        """compute_psd must produce identical results to calling compute_fft then compute_psd_from_fft."""
        fs = 200_000.0
        n = np.arange(512)
        iq = (0.7 * np.exp(1j * 2.0 * np.pi * 30_000.0 * n / fs)).astype(np.complex64)

        psd_direct = compute_psd(iq, fs, nfft=512, window="hamming", centered=True)

        fft_res = compute_fft(iq, fs, nfft=512, window="hamming", centered=True)
        psd_from_fft = compute_psd_from_fft(fft_res)

        np.testing.assert_array_equal(psd_direct.frequency_offset_hz, psd_from_fft.frequency_offset_hz)
        np.testing.assert_allclose(psd_direct.power_linear, psd_from_fft.power_linear, rtol=1e-12)
        np.testing.assert_allclose(
            psd_direct.power_dbfs_per_hz, psd_from_fft.power_dbfs_per_hz, rtol=1e-12
        )

    def test_extreme_floor_underflow(self) -> None:
        """Extreme negative floor (e.g. -4000 dBFS/Hz) must not underflow to -inf on zero/negligible power."""
        fs = 100_000.0
        iq = np.zeros(128, dtype=np.complex64)
        floor_val = -4000.0

        psd = compute_psd(iq, fs, floor_dbfs_per_hz=floor_val)

        assert np.all(psd.power_linear == 0.0)
        assert np.all(psd.power_dbfs_per_hz == floor_val)
        assert np.all(np.isfinite(psd.power_dbfs_per_hz))
        assert not np.any(np.isneginf(psd.power_dbfs_per_hz))

    def test_extreme_floor_with_signal(self) -> None:
        """Non-zero signal with extreme negative floor must preserve valid spectral peaks and floor silent bins."""
        fs = 1_000_000.0
        f0 = 100_000.0
        nfft = 1000
        n = np.arange(nfft)
        iq = np.exp(1j * 2.0 * np.pi * f0 * n / fs).astype(np.complex128)
        floor_val = -3500.0

        psd = compute_psd(iq, fs, nfft=nfft, window="rectangular", floor_dbfs_per_hz=floor_val)

        assert np.all(np.isfinite(psd.power_dbfs_per_hz))
        assert np.all(psd.power_dbfs_per_hz >= floor_val)
        peak_idx = int(np.argmax(psd.power_dbfs_per_hz))
        assert pytest.approx(psd.frequency_offset_hz[peak_idx], abs=1000.0) == f0


    def test_immutability(self) -> None:
        """PSDResult arrays must be read-only."""
        fs = 1000.0
        iq = np.ones(32, dtype=np.complex64)
        psd = compute_psd(iq, fs)

        assert psd.frequency_offset_hz.flags.writeable is False
        assert psd.power_linear.flags.writeable is False
        assert psd.power_dbfs_per_hz.flags.writeable is False

        with pytest.raises((ValueError, TypeError)):
            psd.power_linear[0] = 0.0  # type: ignore[misc]


class TestPSDValidation:
    """Test validation and error handling for PSD functions."""

    def test_reject_non_fftresult_in_psd_from_fft(self) -> None:
        with pytest.raises(TransformError, match="FFTResult"):
            compute_psd_from_fft("not_an_fft_result")  # type: ignore[arg-type]

    def test_reject_non_positive_window_energy_in_psd_from_fft(self) -> None:
        fft_res = compute_fft(np.ones(64, dtype=np.complex64), 1000.0)
        # Using object.__setattr__ to bypass frozen dataclass to simulate corrupted / zero energy
        object.__setattr__(fft_res, "window_energy", 0.0)
        with pytest.raises(TransformError, match="window_energy"):
            compute_psd_from_fft(fft_res)

    @pytest.mark.parametrize("bad_floor", [float("nan"), float("inf"), float("-inf"), "floor", True])
    def test_reject_invalid_floor(self, bad_floor: object) -> None:
        iq = np.ones(32, dtype=np.complex64)
        with pytest.raises(TransformError):
            compute_psd(iq, 1000.0, floor_dbfs_per_hz=bad_floor)  # type: ignore[arg-type]

