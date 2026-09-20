"""Tests for contiguous half-power (3 dB) bandwidth around the dominant peak."""

import numpy as np

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


class TestThreeDbBandwidth:
    """Test 3 dB contiguous bandwidth calculation around the dominant peak."""

    def test_single_bin_three_db(self) -> None:
        """Single powered bin produces resolution-limited 3 dB bandwidth equal to df = 100 Hz."""
        freqs = np.array([-200.0, -100.0, 0.0, 100.0, 200.0], dtype=np.float64)
        powers = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = BandwidthFeatureExtractor()

        fset = extractor.extract(ctx)

        assert fset.features["bandwidth.three_db_hz"].value == 100.0
        assert fset.features["bandwidth.three_db_hz"].valid is True
        assert fset.features["bandwidth.three_db_hz"].unit == "Hz"

    def test_multi_bin_three_db(self) -> None:
        """Three adjacent bins above half power produce 300 Hz."""
        freqs = np.array([-200.0, -100.0, 0.0, 100.0, 200.0], dtype=np.float64)
        powers = np.array([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = BandwidthFeatureExtractor()

        fset = extractor.extract(ctx)

        assert fset.features["bandwidth.three_db_hz"].value == 300.0
        assert fset.features["bandwidth.three_db_hz"].valid is True

    def test_case_d_dominant_peak_with_sidebands(self) -> None:
        """Test Case D: Peak at 0 Hz with sidebands above and below 0.5."""
        freqs = np.array([-300.0, -200.0, -100.0, 0.0, 100.0, 200.0, 300.0], dtype=np.float64)
        powers = np.array([0.01, 0.1, 0.6, 1.0, 0.6, 0.1, 0.01], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = BandwidthFeatureExtractor()

        fset = extractor.extract(ctx)

        # Peak is 1.0 at 0 Hz. Half-power is 0.5.
        # Contiguous >= 0.5 bins are at -100, 0, +100 Hz (3 bins * 100 Hz = 300 Hz)
        assert fset.features["bandwidth.three_db_hz"].value == 300.0

    def test_case_e_separate_second_peak_not_included(self) -> None:
        """Test Case E: Separate peak above half-power across a sub-threshold gap is strictly excluded."""
        freqs = np.array([-300.0, -200.0, -100.0, 0.0, 100.0, 200.0, 300.0], dtype=np.float64)
        # Dominant peak (1.0) at index 3 (0 Hz), second peak (0.8) at index 1 (-200 Hz).
        # Gap at index 2 (-100 Hz) has power 0.0 < 0.5.
        powers = np.array([0.0, 0.8, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = BandwidthFeatureExtractor()

        fset = extractor.extract(ctx)

        # Must only encompass the single dominant bin at 0 Hz -> 100 Hz width
        assert fset.features["bandwidth.three_db_hz"].value == 100.0

    def test_peak_at_spectrum_boundary(self) -> None:
        """Verify contiguous half-power expansion stops safely at array boundaries."""
        freqs = np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float64)
        # Peak at boundary index 0 (1.0), index 1 (0.6 >= 0.5), index 2 (0.2 < 0.5)
        powers = np.array([1.0, 0.6, 0.2, 0.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = BandwidthFeatureExtractor()

        fset = extractor.extract(ctx)

        assert fset.features["bandwidth.three_db_hz"].value == 200.0

    def test_multiple_equal_maxima(self) -> None:
        """Verify argmax selects the first global maximum deterministically."""
        freqs = np.array([-200.0, -100.0, 0.0, 100.0, 200.0], dtype=np.float64)
        powers = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = BandwidthFeatureExtractor()

        fset = extractor.extract(ctx)

        # First maximum at -100 Hz (isolated bin), 3 dB bandwidth = 100 Hz
        assert fset.features["bandwidth.three_db_hz"].value == 100.0
