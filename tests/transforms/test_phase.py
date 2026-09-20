"""Tests for phase computation primitive."""

import numpy as np
import pytest

from signal_processing.core import PhaseResult
from signal_processing.transforms.exceptions import TransformError
from signal_processing.transforms.phase import compute_phase


class TestPhaseCalculation:
    """Test phase computation correctness and PhaseResult contract adherence."""

    def test_known_cardinal_phases(self) -> None:
        """Test standard 4-quadrant cardinal phases."""
        iq = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j], dtype=np.complex128)
        res = compute_phase(iq)

        assert isinstance(res, PhaseResult)
        assert len(res.wrapped_rad) == 4
        assert len(res.unwrapped_rad) == 4

        expected = np.array([0.0, np.pi / 2.0, np.pi, -np.pi / 2.0])
        np.testing.assert_allclose(res.wrapped_rad, expected, atol=1e-12)

    def test_phase_unwrapping_across_boundaries(self) -> None:
        """Phase ramping through multiple 2*pi periods must unwrap continuously."""
        # 4 full cycles over 400 samples
        n = np.arange(400)
        continuous_phase = 4.0 * 2.0 * np.pi * n / 400.0  # 0 to 8*pi
        iq = np.exp(1j * continuous_phase).astype(np.complex128)

        res = compute_phase(iq)

        # Wrapped phase should stay in [-pi, pi]
        assert np.all(res.wrapped_rad >= -np.pi)
        assert np.all(res.wrapped_rad <= np.pi)

        # Unwrapped phase should match continuous phase
        np.testing.assert_allclose(res.unwrapped_rad, continuous_phase, atol=1e-10)

    def test_output_contract_and_immutability(self) -> None:
        """PhaseResult arrays must be read-only and non-aliased."""
        iq = np.array([1.0 + 1.0j, -1.0 - 1.0j], dtype=np.complex64)
        res = compute_phase(iq)

        assert res.wrapped_rad.flags.writeable is False
        assert res.unwrapped_rad.flags.writeable is False
        assert np.all(np.isfinite(res.wrapped_rad))
        assert np.all(np.isfinite(res.unwrapped_rad))

        with pytest.raises((ValueError, TypeError)):
            res.wrapped_rad[0] = 0.0  # type: ignore[misc]

        with pytest.raises((ValueError, TypeError)):
            res.unwrapped_rad[0] = 0.0  # type: ignore[misc]

    def test_input_immutability(self) -> None:
        """Input IQ array must not be mutated."""
        iq = np.array([1.0 + 1.0j, 2.0 - 2.0j], dtype=np.complex64)
        iq_copy = iq.copy()
        _ = compute_phase(iq)
        np.testing.assert_array_equal(iq, iq_copy)


class TestPhaseValidation:
    """Test validation of inputs to compute_phase."""

    def test_reject_empty_array(self) -> None:
        with pytest.raises(TransformError, match="at least 1 samples"):
            compute_phase(np.array([], dtype=np.complex64))

    def test_reject_real_array(self) -> None:
        with pytest.raises(TransformError, match="complex numeric"):
            compute_phase(np.array([1.0, 2.0], dtype=np.float64))

    def test_reject_non_finite_iq(self) -> None:
        iq = np.array([1.0 + 0j, np.nan + 0j], dtype=np.complex64)
        with pytest.raises(TransformError, match="non-finite"):
            compute_phase(iq)
