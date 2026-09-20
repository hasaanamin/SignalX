"""Tests for power computation primitives."""

import numpy as np
import pytest

from signal_processing.transforms.exceptions import TransformError
from signal_processing.transforms.power import (
    compute_mean_power_dbfs,
    compute_mean_power_linear,
    compute_power,
    linear_power_to_dbfs,
)


class TestInstantaneousPower:
    """Test instantaneous linear power computation."""

    def test_constant_signal(self) -> None:
        """Constant 1.0 + 0j signal must yield power of 1.0."""
        iq = np.ones(50, dtype=np.complex64)
        p = compute_power(iq)

        assert len(p) == 50
        assert np.isrealobj(p)
        assert np.all(p >= 0.0)
        np.testing.assert_allclose(p, 1.0)

    def test_amplitude_half_power_quarter(self) -> None:
        """Amplitude 0.5 signal must yield power 0.25."""
        iq = np.full(50, 0.5 + 0.0j, dtype=np.complex64)
        p = compute_power(iq)
        np.testing.assert_allclose(p, 0.25)

    def test_known_complex_values(self) -> None:
        """Verify I^2 + Q^2 on known complex values."""
        iq = np.array([3.0 + 4.0j, 1.0 - 1.0j, 0.0 + 0.0j], dtype=np.complex128)
        expected = np.array([25.0, 2.0, 0.0], dtype=np.float64)
        p = compute_power(iq)
        np.testing.assert_allclose(p, expected, rtol=1e-12)

    def test_output_properties_and_immutability(self) -> None:
        """Power output must be read-only and non-aliased."""
        iq = np.array([1.0 + 2.0j, -1.0 + 0.5j], dtype=np.complex64)
        p = compute_power(iq)

        assert p.flags.writeable is False
        assert np.all(np.isfinite(p))
        assert not np.shares_memory(iq, p)

        with pytest.raises((ValueError, TypeError)):
            p[0] = 100.0  # type: ignore[misc]

    def test_input_immutability(self) -> None:
        """Input IQ array must remain unchanged."""
        iq = np.array([2.0 - 3.0j, 1.0 + 1.0j], dtype=np.complex64)
        iq_copy = iq.copy()
        _ = compute_power(iq)
        np.testing.assert_array_equal(iq, iq_copy)


class TestMeanPower:
    """Test mean linear and dBFS power calculations."""

    def test_mean_power_linear_constant(self) -> None:
        iq = np.ones(100, dtype=np.complex64)
        mean_p = compute_mean_power_linear(iq)
        assert isinstance(mean_p, float)
        assert pytest.approx(mean_p) == 1.0

    def test_mean_power_linear_sinusoid(self) -> None:
        fs = 1_000_000.0
        f0 = 50_000.0
        n = np.arange(1000)
        # 0.5 * exp(j * ...) has constant instantaneous power 0.25
        iq = 0.5 * np.exp(1j * 2.0 * np.pi * f0 * n / fs).astype(np.complex64)
        mean_p = compute_mean_power_linear(iq)
        assert pytest.approx(mean_p, rel=1e-5) == 0.25

    def test_mean_power_dbfs_constant(self) -> None:
        iq = np.ones(100, dtype=np.complex64)
        mean_dbfs = compute_mean_power_dbfs(iq)
        assert isinstance(mean_dbfs, float)
        assert pytest.approx(mean_dbfs, abs=1e-6) == 0.0

    def test_mean_power_dbfs_half_amplitude(self) -> None:
        iq = np.full(100, 0.5 + 0.0j, dtype=np.complex64)
        mean_dbfs = compute_mean_power_dbfs(iq)
        # 10 * log10(0.25) ~= -6.0205999
        expected = 10.0 * np.log10(0.25)
        assert pytest.approx(mean_dbfs, abs=1e-4) == expected

    def test_mean_power_dbfs_zero_signal_flooring(self) -> None:
        iq = np.zeros(100, dtype=np.complex64)
        mean_dbfs = compute_mean_power_dbfs(iq, floor_dbfs=-300.0)
        assert mean_dbfs == -300.0

        mean_dbfs_custom = compute_mean_power_dbfs(iq, floor_dbfs=-150.0)
        assert mean_dbfs_custom == -150.0


class TestLinearPowerToDbfs:
    """Test linear_power_to_dbfs conversion for scalars and arrays."""

    def test_scalar_reference_power(self) -> None:
        assert pytest.approx(linear_power_to_dbfs(1.0)) == 0.0
        assert pytest.approx(linear_power_to_dbfs(0.1)) == -10.0
        assert pytest.approx(linear_power_to_dbfs(0.01)) == -20.0
        assert pytest.approx(linear_power_to_dbfs(0.25), abs=1e-5) == -6.0206

    def test_scalar_zero_flooring(self) -> None:
        val = linear_power_to_dbfs(0.0, floor_dbfs=-300.0)
        assert val == -300.0

        val_custom = linear_power_to_dbfs(0.0, floor_dbfs=-120.0)
        assert val_custom == -120.0

        # Extreme floor underflow check
        val_extreme = linear_power_to_dbfs(0.0, floor_dbfs=-4000.0)
        assert val_extreme == -4000.0

    def test_array_conversion_and_flooring(self) -> None:
        p_arr = np.array([1.0, 0.1, 0.01, 0.0], dtype=np.float64)
        dbfs = linear_power_to_dbfs(p_arr, floor_dbfs=-200.0)

        assert dbfs.flags.writeable is False
        assert np.isrealobj(dbfs)
        assert np.all(np.isfinite(dbfs))
        np.testing.assert_allclose(dbfs[:3], [0.0, -10.0, -20.0])
        assert dbfs[3] == -200.0

        # Extreme floor underflow check on array
        dbfs_extreme = linear_power_to_dbfs(p_arr, floor_dbfs=-4000.0)
        assert dbfs_extreme[3] == -4000.0
        assert np.all(np.isfinite(dbfs_extreme))


    def test_reject_negative_power(self) -> None:
        with pytest.raises(TransformError, match="non-negative"):
            linear_power_to_dbfs(-1.0)

        with pytest.raises(TransformError, match="negative"):
            linear_power_to_dbfs(np.array([1.0, -0.5, 2.0]))

    def test_reject_non_numeric_and_complex(self) -> None:
        with pytest.raises(TransformError):
            linear_power_to_dbfs(1.0 + 2.0j)

        with pytest.raises(TransformError):
            linear_power_to_dbfs("invalid")

        with pytest.raises(TransformError):
            linear_power_to_dbfs(True)

    def test_reject_non_finite_power(self) -> None:
        with pytest.raises(TransformError, match="finite"):
            linear_power_to_dbfs(float("nan"))

        with pytest.raises(TransformError, match="finite"):
            linear_power_to_dbfs(float("inf"))

        with pytest.raises(TransformError, match="non-finite"):
            linear_power_to_dbfs(np.array([1.0, np.nan]))

    def test_reject_invalid_floor_dbfs(self) -> None:
        with pytest.raises(TransformError, match="numeric float"):
            linear_power_to_dbfs(1.0, floor_dbfs=True)  # type: ignore[arg-type]

        with pytest.raises(TransformError, match="finite"):
            linear_power_to_dbfs(1.0, floor_dbfs=float("-inf"))
