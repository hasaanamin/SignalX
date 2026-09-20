"""Tests for spectral rolloff (85% and 95%) frequency offset calculation."""

import numpy as np
import pytest

from signal_processing.core import PSDResult
from signal_processing.features.spectral import (
    SpectralFeatureError,
    SpectralFeatureExtractor,
)


class DummyPSDContext:
    def __init__(self, frequency_offset_hz: np.ndarray, power_linear: np.ndarray) -> None:
        self._psd = PSDResult(
            frequency_offset_hz=frequency_offset_hz,
            power_linear=power_linear,
            power_dbfs_per_hz=np.full_like(power_linear, -100.0),
        )

    def psd(self) -> PSDResult:
        return self._psd


class TestSpectralRolloff:
    """Test 85% and 95% cumulative spectral rolloff frequency offset calculation."""

    def test_case_a_single_dominant_bin_rolloff(self) -> None:
        """Test Case A: Power concentrated at 0 Hz results in both rolloffs at 0.0 Hz."""
        freqs = np.array([-200.0, -100.0, 0.0, 100.0, 200.0], dtype=np.float64)
        powers = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = SpectralFeatureExtractor()

        fset = extractor.extract(ctx)

        r85 = fset.features["spectral.rolloff_85_offset_hz"]
        r95 = fset.features["spectral.rolloff_95_offset_hz"]

        assert r85.valid is True
        assert r85.unit == "Hz"
        assert r85.value == 0.0
        assert isinstance(r85.value, float)
        assert type(r85.value) is float

        assert r95.valid is True
        assert r95.unit == "Hz"
        assert r95.value == 0.0
        assert isinstance(r95.value, float)
        assert type(r95.value) is float

    def test_case_b_uniform_spectrum_rolloff(self) -> None:
        """Test Case B: Uniform power [1, 1, 1, 1] across [0, 100, 200, 300] reaches 85% and 95% at 300 Hz."""
        freqs = np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float64)
        powers = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = SpectralFeatureExtractor()

        fset = extractor.extract(ctx)

        # Cumulative fractions: [0.25, 0.50, 0.75, 1.00]
        # 85% requires fraction >= 0.85 -> index 3 (300 Hz)
        # 95% requires fraction >= 0.95 -> index 3 (300 Hz)
        assert fset.features["spectral.rolloff_85_offset_hz"].value == 300.0
        assert fset.features["spectral.rolloff_95_offset_hz"].value == 300.0

    def test_case_f_one_bin_rolloff_equals_frequency(self) -> None:
        """Test Case F: Single-bin PSD produces rolloff equal to its frequency offset."""
        freqs = np.array([100.0], dtype=np.float64)
        powers = np.array([2.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = SpectralFeatureExtractor()

        fset = extractor.extract(ctx)

        r85 = fset.features["spectral.rolloff_85_offset_hz"]
        r95 = fset.features["spectral.rolloff_95_offset_hz"]

        assert r85.valid is True
        assert r85.value == 100.0
        assert r95.valid is True
        assert r95.value == 100.0

    def test_case_g_unsorted_frequency_axis(self) -> None:
        """Test Case G: Unsorted/uncentered frequency axis is sorted locally and yields identical rolloff."""
        # Uncentered FFT order: [0, 100, 200, -200, -100]
        freqs_unsorted = np.array([0.0, 100.0, 200.0, -200.0, -100.0], dtype=np.float64)
        powers_unsorted = np.array([3.0, 1.0, 1.0, 4.0, 1.0], dtype=np.float64)

        # Centered sorted order: [-200, -100, 0, 100, 200]
        freqs_sorted = np.array([-200.0, -100.0, 0.0, 100.0, 200.0], dtype=np.float64)
        powers_sorted = np.array([4.0, 1.0, 3.0, 1.0, 1.0], dtype=np.float64)

        extractor = SpectralFeatureExtractor()
        fset_unsorted = extractor.extract(DummyPSDContext(freqs_unsorted, powers_unsorted))
        fset_sorted = extractor.extract(DummyPSDContext(freqs_sorted, powers_sorted))

        assert (
            fset_unsorted.features["spectral.rolloff_85_offset_hz"].value
            == fset_sorted.features["spectral.rolloff_85_offset_hz"].value
        )
        assert (
            fset_unsorted.features["spectral.rolloff_95_offset_hz"].value
            == fset_sorted.features["spectral.rolloff_95_offset_hz"].value
        )

    def test_case_h_negative_frequency_spectrum_preserves_negative_rolloff(self) -> None:
        """Test Case H: Spectrum concentrated at negative frequencies preserves negative rolloff offset."""
        freqs = np.array([-300.0, -200.0, -100.0, 0.0, 100.0], dtype=np.float64)
        powers = np.array([5.0, 4.0, 1.0, 0.0, 0.0], dtype=np.float64)
        # Total power = 10.0
        # Cumulative: [-300: 5.0 (50%), -200: 9.0 (90%), -100: 10.0 (100%)]
        # 85% reached at -200 Hz (90% >= 85%) -> -200.0 Hz
        # 95% reached at -100 Hz (100% >= 95%) -> -100.0 Hz
        ctx = DummyPSDContext(freqs, powers)
        extractor = SpectralFeatureExtractor()

        fset = extractor.extract(ctx)

        r85 = fset.features["spectral.rolloff_85_offset_hz"]
        r95 = fset.features["spectral.rolloff_95_offset_hz"]

        assert r85.value == -200.0
        assert r95.value == -100.0

    def test_case_i_rolloff_95_greater_or_equal_rolloff_85(self) -> None:
        """Test Case I: Because cumulative power is monotonic, rolloff 95% must be >= rolloff 85%."""
        freqs = np.array([0.0, 100.0, 200.0], dtype=np.float64)
        powers = np.array([8.5, 1.0, 0.5], dtype=np.float64)
        # Total = 10.0
        # Cumsum: [8.5, 9.5, 10.0]
        # Target 85% = 8.5 -> reached at index 0 (0 Hz)
        # Target 95% = 9.5 -> reached at index 1 (100 Hz)
        ctx = DummyPSDContext(freqs, powers)
        extractor = SpectralFeatureExtractor()

        fset = extractor.extract(ctx)

        r85 = fset.features["spectral.rolloff_85_offset_hz"].value
        r95 = fset.features["spectral.rolloff_95_offset_hz"].value

        assert r85 == 0.0
        assert r95 == 100.0
        assert r95 >= r85

    def test_reject_duplicate_frequency_bins(self) -> None:
        """Verify frequency axis with non-strictly increasing (duplicate) bins raises SpectralFeatureError."""
        freqs = np.array([0.0, 100.0, 100.0, 200.0], dtype=np.float64)
        powers = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = SpectralFeatureExtractor()

        with pytest.raises(SpectralFeatureError, match="strictly increasing distinct values"):
            extractor.extract(ctx)
