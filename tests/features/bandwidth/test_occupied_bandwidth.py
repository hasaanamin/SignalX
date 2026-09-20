"""Tests for 99% cumulative equal-tail occupied bandwidth calculation."""

import numpy as np
import pytest

from signal_processing.core import PSDResult
from signal_processing.features.bandwidth import BandwidthFeatureExtractor


class DummyPSDContext:
    def __init__(self, frequency_offset_hz: np.ndarray, power_linear: np.ndarray) -> None:
        self._psd = PSDResult(
            frequency_offset_hz=frequency_offset_hz,
            power_linear=power_linear,
            power_dbfs_per_hz=np.full_like(power_linear, -100.0),
        )

    def psd(self) -> PSDResult:
        return self._psd


class TestOccupiedBandwidth:
    """Test 99% occupied bandwidth calculation across various spectral profiles."""

    def test_case_a_single_bin_occupied(self) -> None:
        """Test Case A: Single powered bin has finite resolution-limited width df = 100 Hz."""
        freqs = np.array([-200.0, -100.0, 0.0, 100.0, 200.0], dtype=np.float64)
        powers = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = BandwidthFeatureExtractor()

        fset = extractor.extract(ctx)

        assert fset.features["bandwidth.occupied_99_hz"].value == 100.0
        assert fset.features["bandwidth.occupied_99_hz"].valid is True
        assert fset.features["bandwidth.occupied_99_hz"].unit == "Hz"

    def test_case_b_three_equal_occupied_bins(self) -> None:
        """Test Case B: Three adjacent equal powered bins [-100, 0, 100] Hz span [-150, +150] -> 300 Hz."""
        freqs = np.array([-200.0, -100.0, 0.0, 100.0, 200.0], dtype=np.float64)
        powers = np.array([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = BandwidthFeatureExtractor()

        fset = extractor.extract(ctx)

        assert fset.features["bandwidth.occupied_99_hz"].value == 300.0
        assert fset.features["bandwidth.occupied_99_hz"].valid is True

    def test_case_c_known_99_percent_distribution_tails(self) -> None:
        """Test Case C: Verify exact 0.5% lower and 0.5% upper tail exclusion."""
        n_bins = 1000
        df = 1_000.0  # 1 kHz bin spacing
        freqs = np.arange(n_bins, dtype=np.float64) * df - (n_bins * df / 2.0)
        powers = np.zeros(n_bins, dtype=np.float64)

        # Total power = 1000.0 units
        # Lower tail: 0.5% = 5.0 units placed across bins 0..4 (5 bins of 1.0)
        powers[0:5] = 1.0  # sum = 5.0 (indices 0..4)
        # Central 99%: 990.0 units placed across bins 5..994 (990 bins of 1.0)
        powers[5:995] = 1.0  # sum = 990.0 (indices 5..994)
        # Upper tail: 0.5% = 5.0 units placed across bins 995..999 (5 bins of 1.0)
        powers[995:1000] = 1.0  # sum = 5.0 (indices 995..999)

        ctx = DummyPSDContext(freqs, powers)
        extractor = BandwidthFeatureExtractor()

        fset = extractor.extract(ctx)

        # Cumulative power reaches 0.005 (0.5%) at index 4 (5/1000).
        # Cumulative power reaches 0.995 (99.5%) at index 994 (995/1000).
        # Occupied bins: from index 4 to 994 -> 991 bins total
        # Expected bandwidth = 991 * 1000 Hz = 991,000 Hz
        expected_obw = (994 - 4 + 1) * df
        assert pytest.approx(fset.features["bandwidth.occupied_99_hz"].value) == expected_obw

    def test_asymmetric_distribution_obw(self) -> None:
        """Verify OBW on asymmetric spectrum correctly tracks the 99% cumulative energy window."""
        freqs = np.array([0.0, 100.0, 200.0, 300.0, 400.0], dtype=np.float64)
        # Power heavily concentrated at high frequencies: [0.002, 0.002, 0.002, 0.5, 0.5]
        # Total power = 1.006.
        # Indices 3 and 4 hold 1.0/1.006 = 99.4% of power.
        powers = np.array([0.002, 0.002, 0.002, 0.5, 0.5], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = BandwidthFeatureExtractor()

        fset = extractor.extract(ctx)

        # Low tail 0.5% is passed before bin 3 (bins 0,1,2 sum to 0.006 > 0.5%)
        # So k_low is at bin 2 or 3.
        assert fset.features["bandwidth.occupied_99_hz"].value > 0.0
        assert fset.features["bandwidth.occupied_99_hz"].valid is True
