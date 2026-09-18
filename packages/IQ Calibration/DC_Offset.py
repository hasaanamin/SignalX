from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np


def _as_complex_array(x: Any, name: str = "iq") -> np.ndarray:
    """Helper to convert various input formats into a 1D complex numpy array."""
    if x is None:
        raise ValueError(f"{name} cannot be None")
    if isinstance(x, tuple) and len(x) == 2:
        i_arr = np.asarray(x[0], dtype=np.float64).ravel()
        q_arr = np.asarray(x[1], dtype=np.float64).ravel()
        if len(i_arr) != len(q_arr):
            raise ValueError(f"{name} I and Q tuple components must have equal length ({len(i_arr)} != {len(q_arr)})")
        return i_arr + 1j * q_arr
    arr = np.asarray(x)
    if arr.size == 0:
        return np.array([], dtype=np.complex128)
    if arr.ndim == 2:
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            return arr[0, :].astype(np.float64) + 1j * arr[1, :].astype(np.float64)
        elif arr.shape[1] == 2:
            return arr[:, 0].astype(np.float64) + 1j * arr[:, 1].astype(np.float64)
    if np.iscomplexobj(arr):
        return arr.ravel()
    return arr.ravel().astype(np.complex128)


def _validate_finite_iq(arr: np.ndarray, name: str = "iq") -> None:
    """Validates that all samples in the complex array are finite."""
    if not np.all(np.isfinite(arr)):
        nan_count = int(np.sum(np.isnan(arr)))
        inf_count = int(np.sum(np.isinf(arr)))
        raise ValueError(f"{name} contains non-finite values: {nan_count} NaNs, {inf_count} Infs out of {len(arr)} samples")


@dataclass
class DCOffsetResult:
    """
    Result container for DC Offset estimation.

    Attributes
    ----------
    dc_i : float
        In-phase DC offset component in Volts (or raw linear units).
    dc_q : float
        Quadrature DC offset component in Volts (or raw linear units).
    dc_complex : complex
        Complex representation (dc_i + 1j * dc_q).
    dc_magnitude : float
        Magnitude of the complex DC offset: sqrt(dc_i^2 + dc_q^2).
    dc_power : float
        Power of the complex DC offset: dc_i^2 + dc_q^2 in Watts (relative to 1 Ohm).
    dc_power_dbfs : float | None
        DC power relative to full-scale / total signal power in dB (10*log10(P_dc / P_total)).
    method : str
        Estimation method used ('mean', 'median', 'trimmed_mean', 'reference_aided').
    valid : bool
        True if the estimate is mathematically sound and uncorrupted.
    confidence : float
        Confidence score in [0.0, 1.0].
    metadata : dict[str, Any]
        Additional diagnostics and processing metadata.
    """
    dc_i: float = 0.0
    dc_q: float = 0.0
    dc_complex: complex = 0.0 + 0.0j
    dc_magnitude: float = 0.0
    dc_power: float = 0.0
    dc_power_dbfs: float | None = None
    method: str = "mean"
    valid: bool = True
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def diagnostic(self) -> str | None:
        return self.metadata.get("diagnostic") or self.metadata.get("reason")


class DCOffsetAnalyzer:
    """
    Direct-Current (DC) Offset estimation module for complex baseband IQ signals.

    Calculates In-Phase (I) and Quadrature (Q) bias, complex DC vector,
    magnitude, and DC power relative to total signal power.
    Does NOT import or depend on any sibling analysis modules.
    """

    def __init__(self) -> None:
        self.result: DCOffsetResult | None = None

    def compute(
        self,
        iq: Any,
        sample_rate_hz: float = 1.0,
        method: str = "mean",
        reference_iq: Any | None = None,
        trim_fraction: float = 0.1,
        **kwargs: Any,
    ) -> DCOffsetResult:
        """
        Estimates DC offset from complex IQ data.

        Parameters
        ----------
        iq : Any
            Input IQ signal (complex array, list, (I, Q) tuple, or 2D array).
        sample_rate_hz : float
            Sampling rate in Hz (default 1.0).
        method : str
            Estimation method: 'mean', 'median', 'trimmed_mean', or 'reference_aided'.
        reference_iq : Any | None
            Clean reference signal for reference-aided estimation.
        trim_fraction : float
            Fraction of values to trim from each end when method='trimmed_mean' (0.0 to 0.49).

        Returns
        -------
        DCOffsetResult
            Calculated DC offset metrics.
        """
        x = _as_complex_array(iq, name="iq")
        _validate_finite_iq(x, name="iq")
        n = len(x)

        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be strictly positive, got {sample_rate_hz}")

        method_clean = str(method).lower().strip()
        i_arr = x.real
        q_arr = x.imag

        total_power = float(np.mean(np.abs(x) ** 2)) if n > 0 else 0.0

        if method_clean == "mean":
            dc_i = float(np.mean(i_arr)) if n > 0 else 0.0
            dc_q = float(np.mean(q_arr)) if n > 0 else 0.0
            confidence = 1.0
            valid = True

        elif method_clean == "median":
            dc_i = float(np.median(i_arr)) if n > 0 else 0.0
            dc_q = float(np.median(q_arr)) if n > 0 else 0.0
            confidence = 0.95
            valid = True

        elif method_clean == "trimmed_mean":
            if not (0.0 <= trim_fraction < 0.5):
                raise ValueError(f"trim_fraction must be in [0.0, 0.5), got {trim_fraction}")
            k = int(np.floor(n * trim_fraction))
            if k > 0 and n > 2 * k:
                i_sorted = np.sort(i_arr)
                q_sorted = np.sort(q_arr)
                dc_i = float(np.mean(i_sorted[k:-k]))
                dc_q = float(np.mean(q_sorted[k:-k]))
            else:
                dc_i = float(np.mean(i_arr)) if n > 0 else 0.0
                dc_q = float(np.mean(q_arr)) if n > 0 else 0.0
            confidence = 0.98
            valid = True

        elif method_clean == "reference_aided":
            if reference_iq is None:
                raise ValueError("reference_iq must be provided when method='reference_aided'")
            ref = _as_complex_array(reference_iq, name="reference_iq")
            _validate_finite_iq(ref, name="reference_iq")
            if len(ref) != n:
                raise ValueError(f"reference_iq length ({len(ref)}) != iq length ({n})")
            diff = x - ref
            dc_i = float(np.mean(diff.real)) if n > 0 else 0.0
            dc_q = float(np.mean(diff.imag)) if n > 0 else 0.0
            confidence = 1.0
            valid = True

        else:
            raise ValueError(f"Unknown DC estimation method '{method}'. Must be 'mean', 'median', 'trimmed_mean', or 'reference_aided'")

        dc_complex = complex(dc_i, dc_q)
        dc_mag = float(np.hypot(dc_i, dc_q))
        dc_pwr = float(dc_mag ** 2)

        eps = 1e-20
        dc_power_dbfs = float(10.0 * np.log10((dc_pwr + eps) / (total_power + eps)))

        meta = {
            "sample_count": n,
            "sample_rate_hz": float(sample_rate_hz),
            "total_signal_power": total_power,
            "residual_ac_power": max(0.0, total_power - dc_pwr),
        }

        self.result = DCOffsetResult(
            dc_i=dc_i,
            dc_q=dc_q,
            dc_complex=dc_complex,
            dc_magnitude=dc_mag,
            dc_power=dc_pwr,
            dc_power_dbfs=dc_power_dbfs,
            method=method_clean,
            valid=valid,
            confidence=confidence,
            metadata=meta,
        )
        return self.result

    def analyze(self, *args: Any, **kwargs: Any) -> DCOffsetResult:
        """Alias for compute()."""
        return self.compute(*args, **kwargs)


DCOffsetEstimator = DCOffsetAnalyzer


def compute_dc_offset(
    iq: Any = None,
    sample_rate_hz: float = 1.0,
    method: str = "mean",
    reference_iq: Any | None = None,
    trim_fraction: float = 0.1,
    iq_data: Any = None,
    **kwargs: Any,
) -> DCOffsetResult:
    """Convenience functional wrapper for DCOffsetAnalyzer."""
    data = iq if iq is not None else iq_data
    sample_rate = kwargs.get("sample_rate", sample_rate_hz)
    analyzer = DCOffsetAnalyzer()
    return analyzer.compute(
        iq=data,
        sample_rate_hz=sample_rate,
        method=method,
        reference_iq=reference_iq,
        trim_fraction=trim_fraction,
        **kwargs,
    )


estimate_dc_offset = compute_dc_offset

__all__ = [
    "DCOffsetAnalyzer",
    "DCOffsetEstimator",
    "DCOffsetResult",
    "compute_dc_offset",
    "estimate_dc_offset",
]

