"""Tests for BandwidthFeatureExtractor metadata, units, sorting, and error handling."""

import math

import numpy as np
import pytest

from signal_processing.core import FeatureSet, PSDResult
from signal_processing.features.bandwidth import (
    BandwidthFeatureError,
    BandwidthFeatureExtractor,
)


class DummyPSDContext:
    """Minimal duck-typed context providing a psd() method returning a PSDResult."""

    def __init__(
        self,
        frequency_offset_hz: np.ndarray,
        power_linear: np.ndarray,
        power_dbfs_per_hz: np.ndarray | None = None,
    ) -> None:
        if power_dbfs_per_hz is None:
            power_dbfs_per_hz = np.full_like(power_linear, -100.0)
        self._psd = PSDResult(
            frequency_offset_hz=frequency_offset_hz,
            power_linear=power_linear,
            power_dbfs_per_hz=power_dbfs_per_hz,
        )

    def psd(self) -> PSDResult:
        return self._psd


class TestBandwidthExtractor:
    """Test general bandwidth feature extraction mechanics, axis validation, and error handling."""

    def test_featureset_metadata_and_types(self) -> None:
        """Verify extractor returns standard FeatureSet with canonical keys, units, and Python float types."""
        freqs = np.array([-100.0, 0.0, 100.0], dtype=np.float64)
        powers = np.array([0.1, 10.0, 0.1], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = BandwidthFeatureExtractor()

        fset = extractor.extract(ctx)

        assert isinstance(fset, FeatureSet)
        assert fset.extractor == "bandwidth"
        assert fset.extractor_version == "2.0"
        assert len(fset.features) == 2

        # 1. 99% OBW
        fv_obw = fset.features["bandwidth.occupied_99_hz"]
        assert fv_obw.unit == "Hz"
        assert fv_obw.valid is True
        assert fv_obw.reason is None
        assert isinstance(fv_obw.value, float)
        assert type(fv_obw.value) is float
        assert fv_obw.value >= 0.0

        # 2. 3 dB Bandwidth
        fv_3db = fset.features["bandwidth.three_db_hz"]
        assert fv_3db.unit == "Hz"
        assert fv_3db.valid is True
        assert fv_3db.reason is None
        assert isinstance(fv_3db.value, float)
        assert type(fv_3db.value) is float
        assert fv_3db.value >= 0.0

    def test_case_g_unsorted_frequency_axis(self) -> None:
        """Test Case G: Unsorted/uncentered frequency axis is sorted monotonically with identical results."""
        # Uncentered FFT order: [0, 100, 200, -200, -100]
        freqs_unsorted = np.array([0.0, 100.0, 200.0, -200.0, -100.0], dtype=np.float64)
        powers_unsorted = np.array([10.0, 1.0, 0.1, 0.1, 1.0], dtype=np.float64)

        # Equivalent centered order: [-200, -100, 0, 100, 200]
        freqs_sorted = np.array([-200.0, -100.0, 0.0, 100.0, 200.0], dtype=np.float64)
        powers_sorted = np.array([0.1, 1.0, 10.0, 1.0, 0.1], dtype=np.float64)

        extractor = BandwidthFeatureExtractor()
        fset_unsorted = extractor.extract(DummyPSDContext(freqs_unsorted, powers_unsorted))
        fset_sorted = extractor.extract(DummyPSDContext(freqs_sorted, powers_sorted))

        assert fset_unsorted.features["bandwidth.occupied_99_hz"].value == fset_sorted.features["bandwidth.occupied_99_hz"].value
        assert fset_unsorted.features["bandwidth.three_db_hz"].value == fset_sorted.features["bandwidth.three_db_hz"].value

    def test_case_h_nonuniform_axis_rejected(self) -> None:
        """Test Case H: Non-uniform frequency spacing must raise BandwidthFeatureError."""
        freqs_nonuniform = np.array([0.0, 100.0, 205.0, 300.0], dtype=np.float64)
        powers = np.array([1.0, 2.0, 1.0, 0.5], dtype=np.float64)
        ctx = DummyPSDContext(freqs_nonuniform, powers)
        extractor = BandwidthFeatureExtractor()

        with pytest.raises(BandwidthFeatureError, match="uniformly spaced"):
            extractor.extract(ctx)

    def test_duplicate_frequency_bins_rejected(self) -> None:
        """Verify frequency axis with duplicate bins is rejected."""
        freqs_dup = np.array([0.0, 100.0, 100.0, 200.0], dtype=np.float64)
        powers = np.array([1.0, 2.0, 1.0, 0.5], dtype=np.float64)
        ctx = DummyPSDContext(freqs_dup, powers)
        extractor = BandwidthFeatureExtractor()

        with pytest.raises(BandwidthFeatureError, match="strictly increasing distinct values"):
            extractor.extract(ctx)

    def test_one_bin_psd_insufficient_resolution(self) -> None:
        """One-bin PSD has insufficient frequency resolution; returns invalid FeatureValues."""
        freqs = np.array([0.0], dtype=np.float64)
        powers = np.array([10.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = BandwidthFeatureExtractor()

        fset = extractor.extract(ctx)

        obw = fset.features["bandwidth.occupied_99_hz"]
        assert obw.valid is False
        assert math.isnan(obw.value)
        assert obw.unit == "Hz"
        assert "At least two PSD bins are required" in obw.reason

        three_db = fset.features["bandwidth.three_db_hz"]
        assert three_db.valid is False
        assert math.isnan(three_db.value)
        assert three_db.unit == "Hz"
        assert "At least two PSD bins are required" in three_db.reason

    def test_repeated_extraction_statelessness(self) -> None:
        """Repeated extraction produces identical results."""
        freqs = np.array([-100.0, 0.0, 100.0], dtype=np.float64)
        powers = np.array([0.5, 4.0, 0.5], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = BandwidthFeatureExtractor()

        fset1 = extractor.extract(ctx)
        fset2 = extractor.extract(ctx)

        for k in fset1.features:
            assert fset1.features[k].value == fset2.features[k].value
            assert fset1.features[k].unit == fset2.features[k].unit
            assert fset1.features[k].valid == fset2.features[k].valid
            assert fset1.features[k].reason == fset2.features[k].reason

    def test_reject_invalid_context(self) -> None:
        """Verify extractor raises BandwidthFeatureError on None or unusable context."""
        extractor = BandwidthFeatureExtractor()

        with pytest.raises(BandwidthFeatureError, match="Context must not be None"):
            extractor.extract(None)  # type: ignore[arg-type]

        with pytest.raises(BandwidthFeatureError, match="Expected SignalContext"):
            extractor.extract("not_a_context")  # type: ignore[arg-type]

        class BrokenContext:
            def psd(self) -> PSDResult:
                raise RuntimeError("Hardware failure")

        with pytest.raises(BandwidthFeatureError, match="Failed to obtain PSD"):
            extractor.extract(BrokenContext())

    def test_reject_invalid_psd_data(self) -> None:
        """Verify extractor raises BandwidthFeatureError on malformed PSD structures and arrays."""
        extractor = BandwidthFeatureExtractor()

        class MissingAttrContext:
            def psd(self) -> object:
                return object()

        with pytest.raises(BandwidthFeatureError, match="PSDResult must contain"):
            extractor.extract(MissingAttrContext())

        class Unconvertible:
            def __array__(self) -> None:
                raise ValueError("Conversion failed")

        class BadConversionPSD:
            frequency_offset_hz = Unconvertible()
            power_linear = np.array([1.0, 2.0])

        class BadConversionContext:
            def psd(self) -> BadConversionPSD:
                return BadConversionPSD()

        with pytest.raises(BandwidthFeatureError, match="Cannot convert PSD data"):
            extractor.extract(BadConversionContext())

        # Empty array
        class EmptyPSD:
            frequency_offset_hz = np.array([], dtype=np.float64)
            power_linear = np.array([], dtype=np.float64)

        class EmptyContext:
            def psd(self) -> EmptyPSD:
                return EmptyPSD()

        with pytest.raises(BandwidthFeatureError, match="must not be empty"):
            extractor.extract(EmptyContext())

        # Multidimensional
        class MultiDimPSD:
            frequency_offset_hz = np.zeros((2, 2), dtype=np.float64)
            power_linear = np.zeros((2, 2), dtype=np.float64)

        class MultiDimContext:
            def psd(self) -> MultiDimPSD:
                return MultiDimPSD()

        with pytest.raises(BandwidthFeatureError, match="1-dimensional"):
            extractor.extract(MultiDimContext())

        # Length mismatch
        class MismatchedPSD:
            frequency_offset_hz = np.array([10.0, 20.0], dtype=np.float64)
            power_linear = np.array([1.0], dtype=np.float64)

        class MismatchedContext:
            def psd(self) -> MismatchedPSD:
                return MismatchedPSD()

        with pytest.raises(BandwidthFeatureError, match="array length mismatch"):
            extractor.extract(MismatchedContext())

        # Non-numeric / Complex
        class ComplexFreqPSD:
            frequency_offset_hz = np.array([10.0 + 1j, 20.0], dtype=np.complex128)
            power_linear = np.array([1.0, 2.0], dtype=np.float64)

        class ComplexFreqContext:
            def psd(self) -> ComplexFreqPSD:
                return ComplexFreqPSD()

        with pytest.raises(BandwidthFeatureError, match="frequency_offset_hz must be real-valued"):
            extractor.extract(ComplexFreqContext())

        class ComplexPowerPSD:
            frequency_offset_hz = np.array([10.0, 20.0], dtype=np.float64)
            power_linear = np.array([1.0 + 2j, 2.0], dtype=np.complex128)

        class ComplexPowerContext:
            def psd(self) -> ComplexPowerPSD:
                return ComplexPowerPSD()

        with pytest.raises(BandwidthFeatureError, match="power_linear must be real-valued"):
            extractor.extract(ComplexPowerContext())

        # Non-finite (NaN / inf)
        class NanPSD:
            frequency_offset_hz = np.array([10.0, float("nan")], dtype=np.float64)
            power_linear = np.array([1.0, 2.0], dtype=np.float64)

        class NanContext:
            def psd(self) -> NanPSD:
                return NanPSD()

        with pytest.raises(BandwidthFeatureError, match="finite values"):
            extractor.extract(NanContext())

        # Negative power
        class NegativePowerPSD:
            frequency_offset_hz = np.array([10.0, 20.0], dtype=np.float64)
            power_linear = np.array([1.0, -0.5], dtype=np.float64)

        class NegativePowerContext:
            def psd(self) -> NegativePowerPSD:
                return NegativePowerPSD()

        with pytest.raises(BandwidthFeatureError, match="non-negative values"):
            extractor.extract(NegativePowerContext())
