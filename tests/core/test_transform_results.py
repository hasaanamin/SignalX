"""Unit tests for transform result contracts (PSDResult, PhaseResult, AutocorrelationResult)."""

import numpy as np
import pytest

from signal_processing.core import (
    AutocorrelationResult,
    ContractValidationError,
    PhaseResult,
    PSDResult,
)

# ---------------------------------------------------------------------------
# PSDResult Tests
# ---------------------------------------------------------------------------

def test_psd_result_valid() -> None:
    """Test valid PSDResult creation and read-only attributes."""
    freq = np.linspace(-500e3, 500e3, 1024)
    p_lin = np.ones(1024)
    p_dbfs = np.zeros(1024)

    psd = PSDResult(
        frequency_offset_hz=freq,
        power_linear=p_lin,
        power_dbfs_per_hz=p_dbfs,
    )

    assert len(psd.frequency_offset_hz) == 1024
    assert len(psd.power_linear) == 1024
    assert len(psd.power_dbfs_per_hz) == 1024

    # Read-only verification
    assert not psd.frequency_offset_hz.flags.writeable
    assert not psd.power_linear.flags.writeable
    assert not psd.power_dbfs_per_hz.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        psd.frequency_offset_hz[0] = 0.0


def test_psd_result_size_mismatch() -> None:
    """Test rejection when array lengths differ."""
    freq = np.linspace(-500e3, 500e3, 1024)
    p_lin = np.ones(1024)
    p_dbfs_short = np.zeros(512)

    with pytest.raises(ContractValidationError, match="lengths must match"):
        PSDResult(
            frequency_offset_hz=freq,
            power_linear=p_lin,
            power_dbfs_per_hz=p_dbfs_short,
        )


def test_psd_result_empty_array() -> None:
    """Test rejection when arrays are empty."""
    with pytest.raises(ContractValidationError, match="empty"):
        PSDResult(
            frequency_offset_hz=np.array([]),
            power_linear=np.array([]),
            power_dbfs_per_hz=np.array([]),
        )


def test_psd_result_non_finite() -> None:
    """Test rejection of NaN or Inf in PSD arrays."""
    freq = np.array([0.0, 100.0, np.nan])
    p_lin = np.array([1.0, 1.0, 1.0])
    p_dbfs = np.array([0.0, 0.0, 0.0])

    with pytest.raises(ContractValidationError, match="non-finite"):
        PSDResult(
            frequency_offset_hz=freq,
            power_linear=p_lin,
            power_dbfs_per_hz=p_dbfs,
        )


# ---------------------------------------------------------------------------
# PhaseResult Tests
# ---------------------------------------------------------------------------

def test_phase_result_valid() -> None:
    """Test valid PhaseResult construction and read-only attributes."""
    wrapped = np.array([-np.pi, 0.0, np.pi])
    unwrapped = np.array([-np.pi, 0.0, np.pi])

    phase = PhaseResult(wrapped_rad=wrapped, unwrapped_rad=unwrapped)

    assert len(phase.wrapped_rad) == 3
    assert len(phase.unwrapped_rad) == 3
    assert not phase.wrapped_rad.flags.writeable
    assert not phase.unwrapped_rad.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        phase.wrapped_rad[0] = 0.0


def test_phase_result_size_mismatch() -> None:
    """Test rejection when wrapped and unwrapped array lengths differ."""
    wrapped = np.array([-np.pi, 0.0, np.pi])
    unwrapped = np.array([-np.pi, 0.0])

    with pytest.raises(ContractValidationError, match="lengths must match"):
        PhaseResult(wrapped_rad=wrapped, unwrapped_rad=unwrapped)


def test_phase_result_non_finite() -> None:
    """Test rejection of non-finite values in PhaseResult."""
    wrapped = np.array([0.0, float("nan")])
    unwrapped = np.array([0.0, 0.0])

    with pytest.raises(ContractValidationError, match="non-finite"):
        PhaseResult(wrapped_rad=wrapped, unwrapped_rad=unwrapped)


# ---------------------------------------------------------------------------
# AutocorrelationResult Tests
# ---------------------------------------------------------------------------

def test_autocorrelation_result_valid() -> None:
    """Test valid AutocorrelationResult construction and read-only attributes."""
    lags = np.arange(-5, 6)
    vals = np.exp(-np.abs(lags))

    ac = AutocorrelationResult(lag_samples=lags, values=vals)

    assert len(ac.lag_samples) == 11
    assert len(ac.values) == 11
    assert not ac.lag_samples.flags.writeable
    assert not ac.values.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        ac.values[0] = 10.0


def test_autocorrelation_result_size_mismatch() -> None:
    """Test rejection when lags and values lengths differ."""
    lags = np.arange(10)
    vals = np.ones(5)

    with pytest.raises(ContractValidationError, match="lengths must match"):
        AutocorrelationResult(lag_samples=lags, values=vals)


def test_autocorrelation_result_empty() -> None:
    """Test rejection when autocorrelation arrays are empty."""
    with pytest.raises(ContractValidationError, match="empty"):
        AutocorrelationResult(lag_samples=np.array([]), values=np.array([]))


def test_autocorrelation_result_non_finite() -> None:
    """Test rejection of non-finite values in AutocorrelationResult."""
    lags = np.array([0.0, 1.0])
    vals = np.array([1.0, float("inf")])

    with pytest.raises(ContractValidationError, match="non-finite"):
        AutocorrelationResult(lag_samples=lags, values=vals)


def test_transform_results_from_lists() -> None:
    """Test transform results can accept convertible lists."""
    psd = PSDResult(
        frequency_offset_hz=[-100.0, 0.0, 100.0],  # type: ignore[arg-type]
        power_linear=[1.0, 2.0, 1.0],  # type: ignore[arg-type]
        power_dbfs_per_hz=[-50.0, -40.0, -50.0],  # type: ignore[arg-type]
    )
    assert len(psd.frequency_offset_hz) == 3

    phase = PhaseResult(
        wrapped_rad=[-1.0, 1.0],  # type: ignore[arg-type]
        unwrapped_rad=[-1.0, 1.0],  # type: ignore[arg-type]
    )
    assert len(phase.wrapped_rad) == 2


def test_transform_results_multidimensional_rejected() -> None:
    """Test rejection when transform arrays are multi-dimensional."""
    with pytest.raises(ContractValidationError, match="1-dimensional"):
        PhaseResult(
            wrapped_rad=np.ones((2, 2)),
            unwrapped_rad=np.ones((2, 2)),
        )

    with pytest.raises(ContractValidationError, match="1-dimensional"):
        AutocorrelationResult(
            lag_samples=np.ones((2, 2)),
            values=np.ones((2, 2)),
        )


def test_psd_result_negative_power_linear_rejected() -> None:
    """Test rejection of negative power_linear values in PSDResult."""
    with pytest.raises(ContractValidationError, match="power_linear"):
        PSDResult(
            frequency_offset_hz=np.array([0.0, 100.0]),
            power_linear=np.array([1.0, -0.01]),
            power_dbfs_per_hz=np.array([-50.0, -60.0]),
        )


def test_psd_result_complex_arrays_rejected() -> None:
    """Test rejection of complex-valued frequency or power arrays in PSDResult."""
    with pytest.raises(ContractValidationError, match="real-valued"):
        PSDResult(
            frequency_offset_hz=np.array([0.0 + 1.0j, 100.0]),
            power_linear=np.array([1.0, 1.0]),
            power_dbfs_per_hz=np.array([-50.0, -60.0]),
        )

    with pytest.raises(ContractValidationError, match="real-valued"):
        PSDResult(
            frequency_offset_hz=np.array([0.0, 100.0]),
            power_linear=np.array([1.0 + 0.5j, 1.0]),
            power_dbfs_per_hz=np.array([-50.0, -60.0]),
        )


@pytest.mark.parametrize("bad_array", [["a", "b"], [object(), object()]])
def test_psd_result_nonnumeric_arrays_rejected(bad_array: object) -> None:
    """Test rejection of nonnumeric/string/object arrays in PSDResult."""
    with pytest.raises(ContractValidationError, match="numeric"):
        PSDResult(
            frequency_offset_hz=bad_array,  # type: ignore[arg-type]
            power_linear=[1.0, 1.0],  # type: ignore[arg-type]
            power_dbfs_per_hz=[-50.0, -50.0],  # type: ignore[arg-type]
        )


def test_phase_result_complex_array_rejected() -> None:
    """Test rejection of complex-valued phase arrays in PhaseResult."""
    with pytest.raises(ContractValidationError, match="real-valued"):
        PhaseResult(
            wrapped_rad=np.array([0.0 + 1.0j, 1.0]),
            unwrapped_rad=np.array([0.0, 1.0]),
        )


@pytest.mark.parametrize("bad_array", [["a", "b"], [object(), object()]])
def test_phase_result_nonnumeric_arrays_rejected(bad_array: object) -> None:
    """Test rejection of nonnumeric phase arrays in PhaseResult."""
    with pytest.raises(ContractValidationError, match="numeric"):
        PhaseResult(
            wrapped_rad=bad_array,  # type: ignore[arg-type]
            unwrapped_rad=[0.0, 1.0],  # type: ignore[arg-type]
        )


def test_autocorrelation_result_integer_lags_and_conversion() -> None:
    """Test that lag_samples must be integer and safely converts Python integer lists."""
    ac = AutocorrelationResult(
        lag_samples=[-2, -1, 0, 1, 2],  # type: ignore[arg-type]
        values=[0.2, 0.5, 1.0, 0.5, 0.2],  # type: ignore[arg-type]
    )
    assert np.issubdtype(ac.lag_samples.dtype, np.integer)
    assert list(ac.lag_samples) == [-2, -1, 0, 1, 2]


@pytest.mark.parametrize("fractional_lags", [[0.5, 1.5], [0.2, 1.7], np.array([0.1, 1.1])])
def test_autocorrelation_result_fractional_lags_rejected(fractional_lags: object) -> None:
    """Test rejection of fractional lag offsets in AutocorrelationResult."""
    with pytest.raises(ContractValidationError, match="fractional"):
        AutocorrelationResult(
            lag_samples=fractional_lags,  # type: ignore[arg-type]
            values=[1.0, 0.5],  # type: ignore[arg-type]
        )


def test_autocorrelation_result_complex_values_accepted() -> None:
    """Test that complex autocorrelation values are valid and accepted."""
    ac = AutocorrelationResult(
        lag_samples=np.array([0, 1]),
        values=np.array([1.0 + 2.0j, 0.5 - 0.5j], dtype=np.complex64),
    )
    assert np.iscomplexobj(ac.values)
    assert ac.values[0] == 1.0 + 2.0j


@pytest.mark.parametrize("bad_array", [["a", "b"], [object(), object()]])
def test_autocorrelation_result_nonnumeric_arrays_rejected(bad_array: object) -> None:
    """Test rejection of nonnumeric arrays in AutocorrelationResult."""
    with pytest.raises(ContractValidationError, match="(integer|numeric)"):
        AutocorrelationResult(
            lag_samples=bad_array,  # type: ignore[arg-type]
            values=[1.0, 1.0],  # type: ignore[arg-type]
        )

    with pytest.raises(ContractValidationError, match="numeric"):
        AutocorrelationResult(
            lag_samples=[0, 1],  # type: ignore[arg-type]
            values=bad_array,  # type: ignore[arg-type]
        )


