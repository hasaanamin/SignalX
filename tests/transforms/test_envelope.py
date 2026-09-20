"""Tests for envelope computation primitive."""

import numpy as np
import pytest

from signal_processing.transforms.envelope import compute_envelope
from signal_processing.transforms.exceptions import TransformError


class TestEnvelopeCalculation:
    """Test mathematical accuracy and properties of compute_envelope."""

    def test_constant_complex_magnitude(self) -> None:
        """Constant 1.0 + 0j signal must yield envelope of 1.0."""
        iq = np.ones(100, dtype=np.complex64)
        env = compute_envelope(iq)

        assert len(env) == 100
        assert np.isrealobj(env)
        assert np.all(env >= 0.0)
        np.testing.assert_allclose(env, 1.0)

    def test_varying_magnitude(self) -> None:
        """Known complex samples must yield exact euclidean magnitudes sqrt(I^2 + Q^2)."""
        iq = np.array([3.0 + 4.0j, 0.0 + 0.0j, 1.0 - 1.0j, -5.0 + 12.0j], dtype=np.complex128)
        expected = np.array([5.0, 0.0, np.sqrt(2.0), 13.0], dtype=np.float64)

        env = compute_envelope(iq)
        np.testing.assert_allclose(env, expected, rtol=1e-12)

    def test_amplitude_scaled_complex_sinusoid(self) -> None:
        """Complex sinusoid with amplitude 0.5 must yield constant envelope ~0.5."""
        fs = 1_000_000.0
        f0 = 100_000.0
        n = np.arange(200)
        iq = 0.5 * np.exp(1j * 2.0 * np.pi * f0 * n / fs).astype(np.complex64)

        env = compute_envelope(iq)
        assert len(env) == 200
        np.testing.assert_allclose(env, 0.5, atol=1e-6)

    def test_output_properties_and_immutability(self) -> None:
        """Envelope output must be real, non-negative, finite, and read-only."""
        iq = np.array([1.0 + 1.0j, -2.0 + 3.0j], dtype=np.complex64)
        env = compute_envelope(iq)

        assert env.flags.writeable is False
        assert np.all(np.isfinite(env))
        assert np.all(env >= 0.0)

        with pytest.raises((ValueError, TypeError)):
            env[0] = 999.0  # type: ignore[misc]

    def test_input_immutability(self) -> None:
        """Input IQ array must not be mutated or aliased."""
        iq = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)
        iq_copy = iq.copy()

        env = compute_envelope(iq)
        np.testing.assert_array_equal(iq, iq_copy)

        # Output should not share memory with input
        assert not np.shares_memory(iq, env)


class TestEnvelopeValidation:
    """Test rejection of malformed or invalid inputs."""

    def test_reject_empty_array(self) -> None:
        with pytest.raises(TransformError, match="at least 1 samples"):
            compute_envelope(np.array([], dtype=np.complex64))

    def test_reject_multidimensional_array(self) -> None:
        with pytest.raises(TransformError, match="1-dimensional"):
            compute_envelope(np.zeros((10, 2), dtype=np.complex64))

    def test_reject_real_only_array(self) -> None:
        with pytest.raises(TransformError, match="complex numeric"):
            compute_envelope(np.ones(10, dtype=np.float64))

    def test_reject_string_array(self) -> None:
        with pytest.raises(TransformError, match="complex numeric"):
            compute_envelope(np.array(["1.0+2.0j", "3.0+4.0j"]))

    def test_reject_object_array(self) -> None:
        with pytest.raises(TransformError, match="complex numeric"):
            compute_envelope(np.array([1.0 + 2.0j, "bad"], dtype=object))

    def test_reject_nan_in_iq(self) -> None:
        iq = np.array([1.0 + 1.0j, np.nan + 1.0j], dtype=np.complex64)
        with pytest.raises(TransformError, match="non-finite"):
            compute_envelope(iq)

    def test_reject_inf_in_iq(self) -> None:
        iq = np.array([1.0 + 1.0j, np.inf + 0.0j], dtype=np.complex64)
        with pytest.raises(TransformError, match="non-finite"):
            compute_envelope(iq)

    def test_validate_iq_min_samples_constraint(self) -> None:
        from signal_processing.transforms.validation import validate_iq
        with pytest.raises(TransformError, match="min_samples must be >= 1"):
            validate_iq(np.ones(10, dtype=np.complex64), min_samples=0)

