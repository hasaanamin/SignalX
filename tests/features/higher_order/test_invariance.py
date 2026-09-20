"""Tests for higher-order cumulant invariance properties (scale, phase rotation, and DC offset)."""

import numpy as np
import pytest

from signal_processing.features.higher_order import HigherOrderFeatureExtractor


class DummyIQContext:
    def __init__(self, signal_iq: np.ndarray) -> None:
        self.signal_iq = signal_iq


@pytest.fixture
def base_iq() -> np.ndarray:
    """Deterministic non-degenerate complex IQ sequence."""
    rng = np.random.default_rng(42)
    re = rng.uniform(-2.0, 2.0, size=300)
    im = rng.uniform(-2.0, 2.0, size=300)
    return re + 1j * im


class TestInvarianceProperties:
    """Test amplitude-scale, constant carrier-phase rotation, and complex DC-offset invariances."""

    def test_case_c_scale_invariance(self, base_iq: np.ndarray) -> None:
        """Test Case C: Multiplying complex signal by arbitrary non-zero real scalar preserves all 6 features."""
        extractor = HigherOrderFeatureExtractor()

        fset_1 = extractor.extract(DummyIQContext(base_iq))
        fset_small = extractor.extract(DummyIQContext(0.2 * base_iq))
        fset_large = extractor.extract(DummyIQContext(7.5 * base_iq))

        for key in fset_1.features:
            val_1 = fset_1.features[key].value
            val_small = fset_small.features[key].value
            val_large = fset_large.features[key].value

            assert pytest.approx(val_small, rel=1e-10) == val_1
            assert pytest.approx(val_large, rel=1e-10) == val_1

    def test_case_d_phase_rotation_invariance(self, base_iq: np.ndarray) -> None:
        """Test Case D: Rotating complex signal by constant phase theta preserves all 6 magnitude features."""
        extractor = HigherOrderFeatureExtractor()

        theta = 1.2345
        rotated_iq = base_iq * np.exp(1j * theta)

        fset_orig = extractor.extract(DummyIQContext(base_iq))
        fset_rot = extractor.extract(DummyIQContext(rotated_iq))

        for key in fset_orig.features:
            val_orig = fset_orig.features[key].value
            val_rot = fset_rot.features[key].value

            assert pytest.approx(val_rot, rel=1e-10) == val_orig

    def test_case_e_dc_offset_invariance(self, base_iq: np.ndarray) -> None:
        """Test Case E: Adding an arbitrary complex DC offset preserves all 6 features."""
        extractor = HigherOrderFeatureExtractor()

        c = 3.2 - 1.7j
        offset_iq = base_iq + c

        fset_orig = extractor.extract(DummyIQContext(base_iq))
        fset_off = extractor.extract(DummyIQContext(offset_iq))

        for key in fset_orig.features:
            val_orig = fset_orig.features[key].value
            val_off = fset_off.features[key].value

            assert pytest.approx(val_off, rel=1e-10) == val_orig

    def test_combined_transformation_invariance(self, base_iq: np.ndarray) -> None:
        """Verify combined scale, phase rotation, and DC offset produces identical features."""
        extractor = HigherOrderFeatureExtractor()

        transformed_iq = 4.2 * base_iq * np.exp(1j * 0.785) + (1.5 - 2.5j)

        fset_orig = extractor.extract(DummyIQContext(base_iq))
        fset_trans = extractor.extract(DummyIQContext(transformed_iq))

        for key in fset_orig.features:
            val_orig = fset_orig.features[key].value
            val_trans = fset_trans.features[key].value

            assert pytest.approx(val_trans, rel=1e-10) == val_orig
