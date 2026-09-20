"""Tests for second- and fourth-order cumulant features (C20, C40, C41, C42)."""

import numpy as np
import pytest

from signal_processing.features.higher_order import HigherOrderFeatureExtractor


class DummyIQContext:
    def __init__(self, signal_iq: np.ndarray) -> None:
        self.signal_iq = signal_iq


class TestFourthOrderCumulants:
    """Test normalized second- and fourth-order cumulants."""

    def test_case_a_balanced_bpsk_fourth_order(self) -> None:
        """Test Case A: Balanced BPSK [+1, -1] yields |C20|=1, |C40|=2, |C41|=2, |C42|=2."""
        # 100 symbols of alternating +1 and -1
        iq = np.tile([1.0 + 0.0j, -1.0 + 0.0j], 50).astype(np.complex128)
        ctx = DummyIQContext(iq)
        extractor = HigherOrderFeatureExtractor()

        fset = extractor.extract(ctx)

        c20_fv = fset.features["higher_order.normalized_abs_c20"]
        c40_fv = fset.features["higher_order.normalized_abs_c40"]
        c41_fv = fset.features["higher_order.normalized_abs_c41"]
        c42_fv = fset.features["higher_order.normalized_abs_c42"]

        assert c20_fv.valid is True
        assert c20_fv.unit is None
        assert pytest.approx(c20_fv.value, abs=1e-12) == 1.0

        assert c40_fv.valid is True
        assert c40_fv.unit is None
        assert pytest.approx(c40_fv.value, abs=1e-12) == 2.0

        assert c41_fv.valid is True
        assert c41_fv.unit is None
        assert pytest.approx(c41_fv.value, abs=1e-12) == 2.0

        assert c42_fv.valid is True
        assert c42_fv.unit is None
        assert pytest.approx(c42_fv.value, abs=1e-12) == 2.0

    def test_case_b_balanced_qpsk_fourth_order(self) -> None:
        """Test Case B: Balanced QPSK [1, j, -1, -j] yields |C20|=0, |C40|=1, |C41|=0, |C42|=1."""
        qpsk_symbols = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j], dtype=np.complex128)
        iq = np.tile(qpsk_symbols, 25)
        ctx = DummyIQContext(iq)
        extractor = HigherOrderFeatureExtractor()

        fset = extractor.extract(ctx)

        c20_fv = fset.features["higher_order.normalized_abs_c20"]
        c40_fv = fset.features["higher_order.normalized_abs_c40"]
        c41_fv = fset.features["higher_order.normalized_abs_c41"]
        c42_fv = fset.features["higher_order.normalized_abs_c42"]

        assert c20_fv.valid is True
        assert pytest.approx(c20_fv.value, abs=1e-12) == 0.0

        assert c40_fv.valid is True
        assert pytest.approx(c40_fv.value, abs=1e-12) == 1.0

        assert c41_fv.valid is True
        assert pytest.approx(c41_fv.value, abs=1e-12) == 0.0

        assert c42_fv.valid is True
        assert pytest.approx(c42_fv.value, abs=1e-12) == 1.0

    def test_arbitrary_complex_iq_fourth_order_formula(self) -> None:
        """Verify fourth-order cumulants against independently calculated formulas on arbitrary complex IQ."""
        iq = np.array([1.0 + 2.0j, -3.0 + 1.0j, 2.0 - 1.0j, 0.0 + 2.0j], dtype=np.complex128)
        ctx = DummyIQContext(iq)
        extractor = HigherOrderFeatureExtractor()

        fset = extractor.extract(ctx)

        # Independent manual calculation:
        mu = np.mean(iq)
        z0 = iq - mu
        p = np.mean(np.abs(z0) ** 2)
        u = z0 / np.sqrt(p)

        m20 = np.mean(u ** 2)
        m21 = np.mean(np.abs(u) ** 2)
        m40 = np.mean(u ** 4)
        m41 = np.mean((u ** 3) * np.conj(u))
        m42 = np.mean(np.abs(u) ** 4)

        expected_c20 = abs(m20)
        expected_c40 = abs(m40 - 3.0 * (m20 ** 2))
        expected_c41 = abs(m41 - 3.0 * m20 * m21)
        expected_c42 = abs(m42 - (abs(m20) ** 2) - 2.0 * (m21 ** 2))

        assert pytest.approx(fset.features["higher_order.normalized_abs_c20"].value) == expected_c20
        assert pytest.approx(fset.features["higher_order.normalized_abs_c40"].value) == expected_c40
        assert pytest.approx(fset.features["higher_order.normalized_abs_c41"].value) == expected_c41
        assert pytest.approx(fset.features["higher_order.normalized_abs_c42"].value) == expected_c42
