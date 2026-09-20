"""Tests for sixth-order cumulant features (C60, C63)."""

import numpy as np
import pytest

from signal_processing.features.higher_order import HigherOrderFeatureExtractor


class DummyIQContext:
    def __init__(self, signal_iq: np.ndarray) -> None:
        self.signal_iq = signal_iq


class TestSixthOrderCumulants:
    """Test normalized sixth-order cumulants C60 and C63 with full general formulas."""

    def test_case_a_balanced_bpsk_sixth_order(self) -> None:
        """Test Case A: Balanced BPSK yields |C60|=16 and |C63|=16."""
        iq = np.tile([1.0 + 0.0j, -1.0 + 0.0j], 50).astype(np.complex128)
        ctx = DummyIQContext(iq)
        extractor = HigherOrderFeatureExtractor()

        fset = extractor.extract(ctx)

        c60_fv = fset.features["higher_order.normalized_abs_c60"]
        c63_fv = fset.features["higher_order.normalized_abs_c63"]

        assert c60_fv.valid is True
        assert c60_fv.unit is None
        assert pytest.approx(c60_fv.value, abs=1e-12) == 16.0

        assert c63_fv.valid is True
        assert c63_fv.unit is None
        assert pytest.approx(c63_fv.value, abs=1e-12) == 16.0

    def test_case_b_balanced_qpsk_sixth_order(self) -> None:
        """Test Case B: Balanced QPSK yields |C60|=0 and |C63|=4."""
        qpsk_symbols = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j], dtype=np.complex128)
        iq = np.tile(qpsk_symbols, 25)
        ctx = DummyIQContext(iq)
        extractor = HigherOrderFeatureExtractor()

        fset = extractor.extract(ctx)

        c60_fv = fset.features["higher_order.normalized_abs_c60"]
        c63_fv = fset.features["higher_order.normalized_abs_c63"]

        assert c60_fv.valid is True
        assert pytest.approx(c60_fv.value, abs=1e-12) == 0.0

        assert c63_fv.valid is True
        assert pytest.approx(c63_fv.value, abs=1e-12) == 4.0

    def test_full_c60_and_c63_formula_with_nonzero_third_order_moments(self) -> None:
        """Verify full formula retains M30 and M31 terms when signal has asymmetric distribution."""
        # Asymmetric complex distribution where E[u^3] != 0
        iq = np.array([2.0 + 1.0j, -1.0 - 0.5j, 0.5 + 2.0j, -1.5 - 2.5j, 3.0 + 0.0j], dtype=np.complex128)
        ctx = DummyIQContext(iq)
        extractor = HigherOrderFeatureExtractor()

        fset = extractor.extract(ctx)

        # Independent manual calculation from definitions:
        mu = np.mean(iq)
        z0 = iq - mu
        p = np.mean(np.abs(z0) ** 2)
        u = z0 / np.sqrt(p)

        m20 = np.mean(u ** 2)
        m21 = np.mean(np.abs(u) ** 2)
        m30 = np.mean(u ** 3)
        m31 = np.mean((u ** 2) * np.conj(u))
        m40 = np.mean(u ** 4)
        m41 = np.mean((u ** 3) * np.conj(u))
        m42 = np.mean(np.abs(u) ** 4)
        m60 = np.mean(u ** 6)
        m63 = np.mean(np.abs(u) ** 6)

        # Verify M30 is indeed non-zero for this signal
        assert abs(m30) > 1e-6
        assert abs(m31) > 1e-6

        expected_c60 = abs(
            m60
            - 15.0 * m20 * m40
            - 10.0 * (m30 ** 2)
            + 30.0 * (m20 ** 3)
        )

        expected_c63 = abs(
            m63
            - 9.0 * m21 * m42
            - 3.0 * m20 * np.conj(m41)
            - 3.0 * np.conj(m20) * m41
            - 9.0 * (abs(m31) ** 2)
            - (abs(m30) ** 2)
            + 18.0 * (abs(m20) ** 2) * m21
            + 12.0 * (m21 ** 3)
        )

        assert pytest.approx(fset.features["higher_order.normalized_abs_c60"].value) == expected_c60
        assert pytest.approx(fset.features["higher_order.normalized_abs_c63"].value) == expected_c63
