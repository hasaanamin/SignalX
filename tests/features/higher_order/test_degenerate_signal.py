"""Tests for degenerate signals (constant IQ, zero power, and single-sample inputs)."""

import math

import numpy as np

from signal_processing.features.higher_order import HigherOrderFeatureExtractor


class DummyIQContext:
    def __init__(self, signal_iq: np.ndarray) -> None:
        self.signal_iq = signal_iq


class TestDegenerateSignal:
    """Test degenerate signal handling (constant signal, one sample, all-zero signal)."""

    def test_case_f_constant_iq_signal(self) -> None:
        """Test Case F: Constant complex IQ [2+3j, 2+3j, ...] yields centered power 0 and invalid features."""
        iq = np.full(50, 2.0 + 3.0j, dtype=np.complex128)
        ctx = DummyIQContext(iq)
        extractor = HigherOrderFeatureExtractor()

        fset = extractor.extract(ctx)

        for key in [
            "higher_order.normalized_abs_c20",
            "higher_order.normalized_abs_c40",
            "higher_order.normalized_abs_c41",
            "higher_order.normalized_abs_c42",
            "higher_order.normalized_abs_c60",
            "higher_order.normalized_abs_c63",
        ]:
            fv = fset.features[key]
            assert fv.valid is False
            assert math.isnan(fv.value)
            assert fv.unit is None
            assert fv.reason == "Centered signal power is zero"

    def test_case_g_one_sample_signal(self) -> None:
        """Test Case G: Single-sample IQ has insufficient sample count (< 2) and returns invalid features."""
        iq = np.array([1.0 + 1.0j], dtype=np.complex128)
        ctx = DummyIQContext(iq)
        extractor = HigherOrderFeatureExtractor()

        fset = extractor.extract(ctx)

        for key in [
            "higher_order.normalized_abs_c20",
            "higher_order.normalized_abs_c40",
            "higher_order.normalized_abs_c41",
            "higher_order.normalized_abs_c42",
            "higher_order.normalized_abs_c60",
            "higher_order.normalized_abs_c63",
        ]:
            fv = fset.features[key]
            assert fv.valid is False
            assert math.isnan(fv.value)
            assert fv.unit is None
            assert fv.reason == "At least two IQ samples are required"

    def test_all_zero_iq_signal(self) -> None:
        """All-zero IQ produces centered power of 0.0 and invalid features without raising exceptions."""
        iq = np.zeros(100, dtype=np.complex128)
        ctx = DummyIQContext(iq)
        extractor = HigherOrderFeatureExtractor()

        fset = extractor.extract(ctx)

        for key in fset.features:
            fv = fset.features[key]
            assert fv.valid is False
            assert math.isnan(fv.value)
            assert fv.reason == "Centered signal power is zero"

    def test_case_i_random_finite_iq(self) -> None:
        """Test Case I: Random finite IQ produces finite, non-negative, Python float features deterministically."""
        rng = np.random.default_rng(101)
        iq = rng.normal(size=200) + 1j * rng.normal(size=200)

        ctx = DummyIQContext(iq)
        extractor = HigherOrderFeatureExtractor()

        fset1 = extractor.extract(ctx)
        fset2 = extractor.extract(ctx)

        for key in fset1.features:
            fv1 = fset1.features[key]
            fv2 = fset2.features[key]

            assert fv1.valid is True
            assert fv1.unit is None
            assert isinstance(fv1.value, float)
            assert type(fv1.value) is float
            assert math.isfinite(fv1.value)
            assert fv1.value >= 0.0

            assert fv1.value == fv2.value
