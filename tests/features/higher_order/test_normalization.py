"""Tests for higher-order cumulant normalization and equivalence to direct post-normalization."""

import numpy as np
import pytest

from signal_processing.features.higher_order import HigherOrderFeatureExtractor


class DummyIQContext:
    def __init__(self, signal_iq: np.ndarray) -> None:
        self.signal_iq = signal_iq


class TestNormalizationEquivalence:
    """Verify equivalence between pre-normalization (u = z0 / sigma) and direct post-normalization."""

    def test_case_h_direct_post_normalization_equivalence(self) -> None:
        """Test Case H: Unit-power pre-normalization produces identical features to direct C_pq / C21^(order/2)."""
        rng = np.random.default_rng(2026)
        # Random complex signal with varied power
        real_part = rng.normal(loc=1.5, scale=3.0, size=200)
        imag_part = rng.normal(loc=-0.5, scale=2.0, size=200)
        iq = real_part + 1j * imag_part

        ctx = DummyIQContext(iq)
        extractor = HigherOrderFeatureExtractor()
        fset = extractor.extract(ctx)

        # Direct post-normalization calculation:
        mu = np.mean(iq)
        z0 = iq - mu

        # Raw moments of z0
        c21_raw = float(np.mean(np.abs(z0) ** 2))
        m20_raw = np.mean(z0 ** 2)
        m21_raw = c21_raw
        m30_raw = np.mean(z0 ** 3)
        m31_raw = np.mean((z0 ** 2) * np.conj(z0))
        m40_raw = np.mean(z0 ** 4)
        m41_raw = np.mean((z0 ** 3) * np.conj(z0))
        m42_raw = np.mean(np.abs(z0) ** 4)
        m60_raw = np.mean(z0 ** 6)
        m63_raw = np.mean(np.abs(z0) ** 6)

        # Raw cumulants
        c20_raw = m20_raw
        c40_raw = m40_raw - 3.0 * (m20_raw ** 2)
        c41_raw = m41_raw - 3.0 * m20_raw * m21_raw
        c42_raw = m42_raw - (abs(m20_raw) ** 2) - 2.0 * (m21_raw ** 2)
        c60_raw = (
            m60_raw
            - 15.0 * m20_raw * m40_raw
            - 10.0 * (m30_raw ** 2)
            + 30.0 * (m20_raw ** 3)
        )
        c63_raw = (
            m63_raw
            - 9.0 * m21_raw * m42_raw
            - 3.0 * m20_raw * np.conj(m41_raw)
            - 3.0 * np.conj(m20_raw) * m41_raw
            - 9.0 * (abs(m31_raw) ** 2)
            - (abs(m30_raw) ** 2)
            + 18.0 * (abs(m20_raw) ** 2) * m21_raw
            + 12.0 * (m21_raw ** 3)
        )

        # Direct division by powers of C21
        expected_c20 = abs(c20_raw) / c21_raw
        expected_c40 = abs(c40_raw) / (c21_raw ** 2)
        expected_c41 = abs(c41_raw) / (c21_raw ** 2)
        expected_c42 = abs(c42_raw) / (c21_raw ** 2)
        expected_c60 = abs(c60_raw) / (c21_raw ** 3)
        expected_c63 = abs(c63_raw) / (c21_raw ** 3)

        assert pytest.approx(fset.features["higher_order.normalized_abs_c20"].value, rel=1e-11) == expected_c20
        assert pytest.approx(fset.features["higher_order.normalized_abs_c40"].value, rel=1e-11) == expected_c40
        assert pytest.approx(fset.features["higher_order.normalized_abs_c41"].value, rel=1e-11) == expected_c41
        assert pytest.approx(fset.features["higher_order.normalized_abs_c42"].value, rel=1e-11) == expected_c42
        assert pytest.approx(fset.features["higher_order.normalized_abs_c60"].value, rel=1e-11) == expected_c60
        assert pytest.approx(fset.features["higher_order.normalized_abs_c63"].value, rel=1e-11) == expected_c63
