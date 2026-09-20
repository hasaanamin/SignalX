"""Tests for discrete complex autocorrelation primitive."""

import numpy as np
import pytest

from signal_processing.core import AutocorrelationResult
from signal_processing.transforms.autocorrelation import compute_autocorrelation
from signal_processing.transforms.exceptions import TransformError


def _direct_autocorrelation(iq: np.ndarray, max_lag: int) -> np.ndarray:
    """Direct reference implementation of Rxx[k] = sum_{n=0}^{N-1-k} x[n+k] * conj(x[n])."""
    n = len(iq)
    r = np.zeros(max_lag + 1, dtype=np.complex128)
    for k in range(max_lag + 1):
        r[k] = np.sum(iq[k:n] * np.conj(iq[: n - k]))
    return r


class TestAutocorrelationComputation:
    """Test mathematical accuracy, normalization, and properties of compute_autocorrelation."""

    def test_zero_lag_unnormalized_matches_energy(self) -> None:
        """At lag 0, unnormalized autocorrelation must equal total signal energy sum(|x|^2)."""
        iq = np.array([1.0 + 2.0j, 3.0 - 4.0j, -1.0 + 1.0j], dtype=np.complex128)
        expected_energy = float(np.sum(np.abs(iq) ** 2))

        res = compute_autocorrelation(iq, normalized=False)
        assert isinstance(res, AutocorrelationResult)
        assert pytest.approx(res.values[0].real) == expected_energy
        assert pytest.approx(res.values[0].imag, abs=1e-12) == 0.0

    def test_zero_lag_normalized_equals_unity(self) -> None:
        """At lag 0, normalized autocorrelation must be identically 1.0 + 0.0j."""
        iq = np.array([2.0 - 1.0j, 0.5 + 3.0j, -1.0 + 0.0j], dtype=np.complex128)
        res = compute_autocorrelation(iq, normalized=True)

        assert res.values[0] == 1.0 + 0.0j

    def test_comparison_against_direct_reference(self) -> None:
        """FFT-based calculation must numerically match the exact O(N^2) direct definition."""
        rng = np.random.default_rng(123)
        n = 120
        iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)

        max_lag = 45
        expected_r = _direct_autocorrelation(iq, max_lag)

        res_unnorm = compute_autocorrelation(iq, max_lag_samples=max_lag, normalized=False)
        np.testing.assert_allclose(res_unnorm.values, expected_r, rtol=1e-10, atol=1e-10)

        # Normalized comparison
        expected_r_norm = expected_r / expected_r[0]
        res_norm = compute_autocorrelation(iq, max_lag_samples=max_lag, normalized=True)
        np.testing.assert_allclose(res_norm.values, expected_r_norm, rtol=1e-10, atol=1e-10)

    def test_complex_tone_phase_progression(self) -> None:
        """For x[n] = exp(j * omega * n), Rxx[k] / Rxx[0] = ((N - k)/N) * exp(j * omega * k)."""
        n_samples = 200
        n = np.arange(n_samples)
        omega = 2.0 * np.pi * 0.05  # 0.05 normalized frequency
        iq = np.exp(1j * omega * n).astype(np.complex128)

        max_lag = 50
        res = compute_autocorrelation(iq, max_lag_samples=max_lag, normalized=True)

        k = np.arange(max_lag + 1)
        expected_decay = (n_samples - k) / n_samples
        expected_complex = expected_decay * np.exp(1j * omega * k)

        np.testing.assert_allclose(res.values, expected_complex, rtol=1e-10, atol=1e-10)

    def test_max_lag_defaults_and_boundaries(self) -> None:
        """Test default max_lag (N - 1), zero lag (0), and boundary lag (N - 1)."""
        iq = np.array([1.0 + 1.0j, 2.0 - 1.0j, -1.0 + 0.0j], dtype=np.complex64)
        n = len(iq)

        # Default max lag is N - 1
        res_default = compute_autocorrelation(iq)
        assert len(res_default.lag_samples) == n
        assert res_default.lag_samples[-1] == n - 1

        # Lag 0 only
        res_zero = compute_autocorrelation(iq, max_lag_samples=0)
        assert len(res_zero.lag_samples) == 1
        assert res_zero.lag_samples[0] == 0

        # Boundary max_lag = n - 1
        res_full = compute_autocorrelation(iq, max_lag_samples=n - 1)
        assert len(res_full.lag_samples) == n

    def test_output_and_input_immutability(self) -> None:
        """AutocorrelationResult arrays must be read-only and input must not be mutated."""
        iq = np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex64)
        iq_copy = iq.copy()

        res = compute_autocorrelation(iq)

        np.testing.assert_array_equal(iq, iq_copy)
        assert res.lag_samples.flags.writeable is False
        assert res.values.flags.writeable is False

        with pytest.raises((ValueError, TypeError)):
            res.values[0] = 0.0  # type: ignore[misc]

    def test_unnormalized_zero_signal_allowed(self) -> None:
        """Zero signal with normalized=False must succeed and return zeros."""
        iq = np.zeros(10, dtype=np.complex64)
        res = compute_autocorrelation(iq, normalized=False)
        np.testing.assert_array_equal(res.values, 0.0)


class TestAutocorrelationValidation:
    """Test validation and error handling for compute_autocorrelation."""

    def test_reject_zero_energy_signal_when_normalized(self) -> None:
        """Zero-energy signal cannot be normalized and must raise TransformError."""
        iq = np.zeros(50, dtype=np.complex64)
        with pytest.raises(TransformError, match="zero-energy"):
            compute_autocorrelation(iq, normalized=True)

    @pytest.mark.parametrize("bad_lag", [-1, 10, 100, 2.5, "5", True])
    def test_reject_invalid_max_lag(self, bad_lag: object) -> None:
        iq = np.ones(10, dtype=np.complex64)
        with pytest.raises(TransformError):
            compute_autocorrelation(iq, max_lag_samples=bad_lag)  # type: ignore[arg-type]

    def test_reject_non_boolean_normalized(self) -> None:
        iq = np.ones(10, dtype=np.complex64)
        with pytest.raises(TransformError, match="boolean"):
            compute_autocorrelation(iq, normalized="yes")  # type: ignore[arg-type]

    def test_reject_invalid_iq(self) -> None:
        with pytest.raises(TransformError):
            compute_autocorrelation(np.array([], dtype=np.complex64))

        with pytest.raises(TransformError):
            compute_autocorrelation(np.ones(10, dtype=np.float64))
