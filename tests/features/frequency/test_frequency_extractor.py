"""Tests for FrequencyFeatureExtractor mathematical accuracy, units, and error handling."""

import numpy as np
import pytest

from signal_processing.core import FeatureSet, PSDResult
from signal_processing.features.frequency import (
    FrequencyFeatureError,
    FrequencyFeatureExtractor,
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


class TestFrequencyExtractor:
    """Test frequency feature extraction on synthetic and analytical PSD profiles."""

    def test_case_a_positive_tone(self) -> None:
        """Test Case A: Symmetric tone centered at +100 kHz produces +100,000 Hz for peak and center."""
        freqs = np.array([80_000.0, 90_000.0, 100_000.0, 110_000.0, 120_000.0], dtype=np.float64)
        powers = np.array([0.01, 0.1, 10.0, 0.1, 0.01], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = FrequencyFeatureExtractor()

        fset = extractor.extract(ctx)

        assert isinstance(fset, FeatureSet)
        assert fset.extractor == "frequency"
        assert fset.extractor_version == "1.0"
        assert len(fset.features) == 2

        # 1. Peak offset
        fv_peak = fset.features["frequency.peak_offset_hz"]
        assert fv_peak.value == 100_000.0
        assert fv_peak.unit == "Hz"
        assert fv_peak.valid is True
        assert fv_peak.reason is None
        assert isinstance(fv_peak.value, float)
        assert type(fv_peak.value) is float

        # 2. Center offset
        fv_center = fset.features["frequency.center_offset_hz"]
        assert pytest.approx(fv_center.value, abs=1e-3) == 100_000.0
        assert fv_center.unit == "Hz"
        assert fv_center.valid is True
        assert fv_center.reason is None
        assert isinstance(fv_center.value, float)
        assert type(fv_center.value) is float

    def test_case_b_negative_tone(self) -> None:
        """Test Case B: Negative baseband tone centered at -100 kHz produces -100,000 Hz."""
        freqs = np.array([-120_000.0, -110_000.0, -100_000.0, -90_000.0, -80_000.0], dtype=np.float64)
        powers = np.array([0.01, 0.1, 10.0, 0.1, 0.01], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = FrequencyFeatureExtractor()

        fset = extractor.extract(ctx)

        assert fset.features["frequency.peak_offset_hz"].value == -100_000.0
        assert pytest.approx(fset.features["frequency.center_offset_hz"].value, abs=1e-3) == -100_000.0

    def test_case_c_amplitude_independence(self) -> None:
        """Test Case C: Scaling linear power by an arbitrary factor preserves identical frequency estimates."""
        freqs = np.array([-50_000.0, 0.0, 50_000.0], dtype=np.float64)
        powers_1 = np.array([0.5, 2.0, 0.5], dtype=np.float64)
        powers_2 = np.array([0.5, 2.0, 0.5], dtype=np.float64) * 0.01

        ctx_1 = DummyPSDContext(freqs, powers_1)
        ctx_2 = DummyPSDContext(freqs, powers_2)
        extractor = FrequencyFeatureExtractor()

        fset_1 = extractor.extract(ctx_1)
        fset_2 = extractor.extract(ctx_2)

        assert fset_1.features["frequency.peak_offset_hz"].value == fset_2.features["frequency.peak_offset_hz"].value
        assert pytest.approx(fset_1.features["frequency.center_offset_hz"].value) == fset_2.features["frequency.center_offset_hz"].value

    def test_case_d_known_synthetic_psd(self) -> None:
        """Test Case D: Known symmetric and asymmetric PSD profiles match analytical formulas."""
        extractor = FrequencyFeatureExtractor()

        # Symmetric around 0 Hz
        freqs_sym = np.array([-200.0, -100.0, 0.0, 100.0, 200.0], dtype=np.float64)
        powers_sym = np.array([0.0, 1.0, 2.0, 1.0, 0.0], dtype=np.float64)
        fset_sym = extractor.extract(DummyPSDContext(freqs_sym, powers_sym))

        assert fset_sym.features["frequency.peak_offset_hz"].value == 0.0
        assert pytest.approx(fset_sym.features["frequency.center_offset_hz"].value) == 0.0

        # Asymmetric 1: [0, 100, 200] Hz with weights [1, 2, 1]
        freqs_asym1 = np.array([0.0, 100.0, 200.0], dtype=np.float64)
        powers_asym1 = np.array([1.0, 2.0, 1.0], dtype=np.float64)
        fset_asym1 = extractor.extract(DummyPSDContext(freqs_asym1, powers_asym1))

        assert fset_asym1.features["frequency.peak_offset_hz"].value == 100.0
        assert pytest.approx(fset_asym1.features["frequency.center_offset_hz"].value) == 100.0

        # Asymmetric 2: [-100, 0, 100, 200] Hz with powers [1, 0, 0, 3]
        # weighted_sum = -100*1 + 200*3 = 500
        # total_power = 4
        # center = 500 / 4 = 125.0 Hz
        # peak = 200.0 Hz
        freqs_asym2 = np.array([-100.0, 0.0, 100.0, 200.0], dtype=np.float64)
        powers_asym2 = np.array([1.0, 0.0, 0.0, 3.0], dtype=np.float64)
        fset_asym2 = extractor.extract(DummyPSDContext(freqs_asym2, powers_asym2))

        assert fset_asym2.features["frequency.peak_offset_hz"].value == 200.0
        assert pytest.approx(fset_asym2.features["frequency.center_offset_hz"].value) == 125.0

    def test_repeated_extraction_statelessness(self) -> None:
        """Repeated extraction on the same context produces identical FeatureSets."""
        freqs = np.array([-10.0, 0.0, 10.0], dtype=np.float64)
        powers = np.array([1.0, 5.0, 2.0], dtype=np.float64)
        ctx = DummyPSDContext(freqs, powers)
        extractor = FrequencyFeatureExtractor()

        fset1 = extractor.extract(ctx)
        fset2 = extractor.extract(ctx)

        for k in fset1.features:
            assert fset1.features[k].value == fset2.features[k].value
            assert fset1.features[k].unit == fset2.features[k].unit
            assert fset1.features[k].valid == fset2.features[k].valid
            assert fset1.features[k].reason == fset2.features[k].reason

    def test_reject_invalid_context(self) -> None:
        """Verify extractor raises FrequencyFeatureError on None or unusable context."""
        extractor = FrequencyFeatureExtractor()

        with pytest.raises(FrequencyFeatureError, match="Context must not be None"):
            extractor.extract(None)  # type: ignore[arg-type]

        with pytest.raises(FrequencyFeatureError, match="Expected SignalContext"):
            extractor.extract("not_a_context")  # type: ignore[arg-type]

        class BrokenContext:
            def psd(self) -> PSDResult:
                raise RuntimeError("DSP pipeline broken")

        with pytest.raises(FrequencyFeatureError, match="Failed to obtain PSD"):
            extractor.extract(BrokenContext())

    def test_reject_invalid_psd_data(self) -> None:
        """Verify extractor raises FrequencyFeatureError on malformed PSD structures and arrays."""
        extractor = FrequencyFeatureExtractor()

        class MissingAttributesPSD:
            pass

        class MissingAttrContext:
            def psd(self) -> MissingAttributesPSD:
                return MissingAttributesPSD()

        with pytest.raises(FrequencyFeatureError, match="PSDResult must contain"):
            extractor.extract(MissingAttrContext())

        class Unconvertible:
            def __array__(self) -> None:
                raise ValueError("Conversion impossible")

        class BadConversionPSD:
            frequency_offset_hz = Unconvertible()
            power_linear = np.array([1.0, 2.0])

        class BadConversionContext:
            def psd(self) -> BadConversionPSD:
                return BadConversionPSD()

        with pytest.raises(FrequencyFeatureError, match="Cannot convert PSD data"):
            extractor.extract(BadConversionContext())

        # Empty array
        class EmptyPSD:
            frequency_offset_hz = np.array([], dtype=np.float64)
            power_linear = np.array([], dtype=np.float64)

        class EmptyContext:
            def psd(self) -> EmptyPSD:
                return EmptyPSD()

        with pytest.raises(FrequencyFeatureError, match="must not be empty"):
            extractor.extract(EmptyContext())

        # Multidimensional
        class MultiDimPSD:
            frequency_offset_hz = np.zeros((2, 2), dtype=np.float64)
            power_linear = np.zeros((2, 2), dtype=np.float64)

        class MultiDimContext:
            def psd(self) -> MultiDimPSD:
                return MultiDimPSD()

        with pytest.raises(FrequencyFeatureError, match="1-dimensional"):
            extractor.extract(MultiDimContext())

        # Length mismatch
        class MismatchedPSD:
            frequency_offset_hz = np.array([10.0, 20.0], dtype=np.float64)
            power_linear = np.array([1.0], dtype=np.float64)

        class MismatchedContext:
            def psd(self) -> MismatchedPSD:
                return MismatchedPSD()

        with pytest.raises(FrequencyFeatureError, match="array length mismatch"):
            extractor.extract(MismatchedContext())

        # Non-numeric / Complex
        class ComplexFreqPSD:
            frequency_offset_hz = np.array([10.0 + 1j, 20.0], dtype=np.complex128)
            power_linear = np.array([1.0, 2.0], dtype=np.float64)

        class ComplexFreqContext:
            def psd(self) -> ComplexFreqPSD:
                return ComplexFreqPSD()

        with pytest.raises(FrequencyFeatureError, match="frequency_offset_hz must be real-valued"):
            extractor.extract(ComplexFreqContext())

        class ComplexPowerPSD:
            frequency_offset_hz = np.array([10.0, 20.0], dtype=np.float64)
            power_linear = np.array([1.0 + 2j, 2.0], dtype=np.complex128)

        class ComplexPowerContext:
            def psd(self) -> ComplexPowerPSD:
                return ComplexPowerPSD()

        with pytest.raises(FrequencyFeatureError, match="power_linear must be real-valued"):
            extractor.extract(ComplexPowerContext())

        # Non-finite (NaN / inf)
        class NanPSD:
            frequency_offset_hz = np.array([10.0, float("nan")], dtype=np.float64)
            power_linear = np.array([1.0, 2.0], dtype=np.float64)

        class NanContext:
            def psd(self) -> NanPSD:
                return NanPSD()

        with pytest.raises(FrequencyFeatureError, match="finite values"):
            extractor.extract(NanContext())

        # Negative power
        class NegativePowerPSD:
            frequency_offset_hz = np.array([10.0, 20.0], dtype=np.float64)
            power_linear = np.array([1.0, -0.5], dtype=np.float64)

        class NegativePowerContext:
            def psd(self) -> NegativePowerPSD:
                return NegativePowerPSD()

        with pytest.raises(FrequencyFeatureError, match="non-negative values"):
            extractor.extract(NegativePowerContext())
