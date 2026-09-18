from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
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
class QuadratureErrorResult:
    """
    Result container for Quadrature Quality and Non-Orthogonality metric.

    Attributes
    ----------
    iq_correlation : float
        Normalized cross-correlation coefficient between I and Q: cov(I, Q) / (std(I)*std(Q)).
        Ideal = 0.0. Range in [-1.0, 1.0].
    quadrature_error : float
        Magnitude of non-orthogonality: abs(iq_correlation). Range in [0.0, 1.0].
    quadrature_error_deg : float | None
        Angular non-orthogonality error derived from correlation: arcsin(|rho|) * (180/pi).
    orthogonality_factor : float
        Normalized orthogonality index: 1.0 - quadrature_error (1.0 = perfect orthogonality).
    confidence : float
        Confidence score in [0.0, 1.0].
    valid : bool
        True if statistics are well-conditioned.
    metadata : dict[str, Any]
        Covariance matrix and sample count.
    """
    iq_correlation: float = 0.0
    quadrature_error: float = 0.0
    quadrature_error_deg: float | None = 0.0
    orthogonality_factor: float = 1.0
    confidence: float = 1.0
    valid: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def error_deg(self) -> float | None:
        return self.quadrature_error_deg

    @property
    def correlation_coefficient(self) -> float:
        return self.iq_correlation

    @property
    def diagnostic(self) -> str | None:
        return self.metadata.get("diagnostic") or self.metadata.get("reason")


class QuadratureErrorAnalyzer:
    """
    Quadrature Error / Non-Orthogonality analysis calculator.

    Quantifies the statistical non-orthogonality and cross-correlation between
    In-Phase (I) and Quadrature (Q) channels.
    Does NOT import or depend on any sibling analysis modules.
    """

    def __init__(self) -> None:
        self.result: QuadratureErrorResult | None = None

    def compute(
        self,
        iq: Any,
        sample_rate_hz: float = 1.0,
        dc_result: Any | None = None,
        **kwargs: Any,
    ) -> QuadratureErrorResult:
        """
        Calculates I/Q cross-correlation, quadrature error, and orthogonality factor.

        Parameters
        ----------
        iq : Any
            Input complex IQ signal.
        sample_rate_hz : float
            Sampling rate in Hz (default 1.0).
        dc_result : Any | None
            Optional precomputed DC offset result to reuse.

        Returns
        -------
        QuadratureErrorResult
            Non-orthogonality metrics and covariance properties.
        """
        x = _as_complex_array(iq, name="iq")
        _validate_finite_iq(x, name="iq")
        n = len(x)

        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be strictly positive, got {sample_rate_hz}")

        if n < 4:
            self.result = QuadratureErrorResult(
                iq_correlation=0.0,
                quadrature_error=0.0,
                quadrature_error_deg=0.0,
                orthogonality_factor=1.0,
                confidence=0.0,
                valid=False,
                metadata={"reason": "Sample count too small (< 4 samples)"},
            )
            return self.result

        # Remove DC offset
        if dc_result is not None:
            dc_c = getattr(dc_result, "dc_complex", complex(dc_result))
        else:
            dc_c = complex(float(np.mean(x.real)), float(np.mean(x.imag)))

        x_ac = x - dc_c
        i_ac = x_ac.real
        q_ac = x_ac.imag

        std_i = float(np.std(i_ac))
        std_q = float(np.std(q_ac))

        eps = 1e-20
        if std_i < eps or std_q < eps:
            self.result = QuadratureErrorResult(
                iq_correlation=0.0,
                quadrature_error=0.0,
                quadrature_error_deg=0.0,
                orthogonality_factor=1.0,
                confidence=0.0,
                valid=False,
                metadata={"reason": "Zero AC variance in I or Q channel"},
            )
            return self.result

        cov_iq = float(np.mean(i_ac * q_ac))
        corr = float(np.clip(cov_iq / (std_i * std_q), -1.0, 1.0))
        quad_err = float(abs(corr))
        quad_err_deg = float(np.degrees(np.arcsin(np.clip(quad_err, 0.0, 1.0))))
        orth_factor = float(np.clip(1.0 - quad_err, 0.0, 1.0))

        confidence = float(np.clip(1.0 - 5.0 / np.sqrt(n), 0.1, 1.0))

        meta = {
            "covariance_iq": cov_iq,
            "std_i": std_i,
            "std_q": std_q,
            "sample_count": n,
            "sample_rate_hz": float(sample_rate_hz),
        }

        self.result = QuadratureErrorResult(
            iq_correlation=corr,
            quadrature_error=quad_err,
            quadrature_error_deg=quad_err_deg,
            orthogonality_factor=orth_factor,
            confidence=confidence,
            valid=True,
            metadata=meta,
        )
        return self.result

    def analyze(self, *args: Any, **kwargs: Any) -> QuadratureErrorResult:
        """Alias for compute()."""
        return self.compute(*args, **kwargs)


QuadratureErrorEstimator = QuadratureErrorAnalyzer


def compute_quadrature_error(
    iq: Any = None,
    sample_rate_hz: float = 1.0,
    sample_rate: float | None = None,
    dc_result: Any | None = None,
    dc_offset_result: Any | None = None,
    dc_offset: Any | None = None,
    iq_data: Any = None,
    **kwargs: Any,
) -> QuadratureErrorResult:
    """Convenience functional wrapper for QuadratureErrorAnalyzer."""
    data = iq if iq is not None else iq_data
    fs = sample_rate if sample_rate is not None else sample_rate_hz
    dc_res = dc_result if dc_result is not None else (dc_offset_result if dc_offset_result is not None else dc_offset)

    analyzer = QuadratureErrorAnalyzer()
    return analyzer.compute(
        iq=data,
        sample_rate_hz=fs,
        dc_result=dc_res,
        **kwargs,
    )


estimate_quadrature_error = compute_quadrature_error

__all__ = [
    "QuadratureErrorAnalyzer",
    "QuadratureErrorEstimator",
    "QuadratureErrorResult",
    "compute_quadrature_error",
    "estimate_quadrature_error",
]

