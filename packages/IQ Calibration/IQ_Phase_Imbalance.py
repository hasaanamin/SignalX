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
class IQPhaseImbalanceResult:
    """
    Result container for I/Q Phase Imbalance estimation.

    Sign Convention
    ---------------
    Ideal quadrature is exactly 90 degrees (pi/2 radians).
    estimated_quadrature_angle = 90 deg + phase_imbalance_deg.
    Positive phase_imbalance_deg > 0 means the angle between I and Q LOs is > 90 degrees.
    """
    estimated_quadrature_angle_rad: float = np.pi / 2.0
    estimated_quadrature_angle_deg: float = 90.0
    phase_imbalance_rad: float = 0.0
    phase_imbalance_deg: float = 0.0
    confidence: float = 1.0
    valid: bool = True
    method: str = "statistical"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def diagnostic(self) -> str | None:
        return self.metadata.get("diagnostic") or self.metadata.get("reason")


class IQPhaseImbalanceAnalyzer:
    """
    I/Q Phase Imbalance analyzer for complex baseband IQ receivers.

    Estimates the angular non-orthogonality (deviation from ideal 90 degrees)
    between the mixer local oscillators.
    Does NOT import or depend on any sibling analysis modules.
    """

    def __init__(self) -> None:
        self.result: IQPhaseImbalanceResult | None = None

    def compute(
        self,
        iq: Any,
        sample_rate_hz: float = 1.0,
        dc_result: Any | None = None,
        gain_result: Any | None = None,
        method: str = "statistical",
        reference_iq: Any | None = None,
        **kwargs: Any,
    ) -> IQPhaseImbalanceResult:
        """
        Estimates I/Q phase imbalance.

        Parameters
        ----------
        iq : Any
            Input complex IQ signal.
        sample_rate_hz : float
            Sampling rate in Hz (default 1.0).
        dc_result : Any | None
            Optional precomputed DC offset result to reuse.
        gain_result : Any | None
            Optional precomputed gain imbalance result to normalize branch amplitudes.
        method : str
            Estimation method: 'statistical', 'calibration_tone', or 'reference_aided'.
        reference_iq : Any | None
            Clean reference signal for reference-aided estimation.

        Returns
        -------
        IQPhaseImbalanceResult
            Phase deviation metrics and estimated quadrature angle.
        """
        x = _as_complex_array(iq, name="iq")
        _validate_finite_iq(x, name="iq")
        n = len(x)

        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be strictly positive, got {sample_rate_hz}")

        if n < 4:
            self.result = IQPhaseImbalanceResult(
                estimated_quadrature_angle_rad=np.pi / 2.0,
                estimated_quadrature_angle_deg=90.0,
                phase_imbalance_rad=0.0,
                phase_imbalance_deg=0.0,
                confidence=0.0,
                valid=False,
                method=method,
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

        # Gain normalization
        if gain_result is not None:
            g_ratio = float(getattr(gain_result, "gain_ratio", 1.0))
        else:
            p_i = float(np.mean(i_ac ** 2))
            p_q = float(np.mean(q_ac ** 2))
            g_ratio = float(np.sqrt(p_i / (p_q + 1e-20)) if p_q > 1e-20 else 1.0)

        q_norm = q_ac * g_ratio

        method_clean = str(method).lower().strip()
        meta: dict[str, Any] = {
            "dc_removed": dc_c,
            "gain_ratio_applied": g_ratio,
            "sample_count": n,
            "sample_rate_hz": float(sample_rate_hz),
        }

        if method_clean in ("statistical", "blind_covariance"):
            cov_iq = float(np.mean(i_ac * q_norm))
            var_i = float(np.mean(i_ac ** 2))
            var_q = float(np.mean(q_norm ** 2))
            denom = var_i + var_q

            if denom < 1e-20:
                self.result = IQPhaseImbalanceResult(
                    estimated_quadrature_angle_rad=np.pi / 2.0,
                    estimated_quadrature_angle_deg=90.0,
                    phase_imbalance_rad=0.0,
                    phase_imbalance_deg=0.0,
                    confidence=0.0,
                    valid=False,
                    method="statistical",
                    metadata={"reason": "Signal has zero AC power"},
                )
                return self.result

            # Check for non-circular degenerate 1D signals (e.g. pure real BPSK with zero Q energy)
            if var_i < 1e-12 or var_q < 1e-12 or (min(var_i, var_q) / max(var_i, var_q) < 1e-4):
                self.result = IQPhaseImbalanceResult(
                    estimated_quadrature_angle_rad=np.pi / 2.0,
                    estimated_quadrature_angle_deg=90.0,
                    phase_imbalance_rad=0.0,
                    phase_imbalance_deg=0.0,
                    confidence=0.0,
                    valid=False,
                    method="statistical",
                    metadata={"reason": "Non-circular / degenerate constellation without quadrature energy"},
                )
                return self.result

            # sin(phi) = 2 * E[I * Q_norm] / (E[I^2] + E[Q_norm^2])
            sin_phi = float(np.clip(2.0 * cov_iq / denom, -1.0, 1.0))
            phi_rad = float(np.arcsin(sin_phi))
            phi_deg = float(np.degrees(phi_rad))

            quad_angle_rad = float(np.pi / 2.0 + phi_rad)
            quad_angle_deg = float(90.0 + phi_deg)

            confidence = float(np.clip(1.0 - 10.0 / np.sqrt(n), 0.1, 1.0))
            valid = True
            meta["cov_iq"] = cov_iq
            meta["sin_phi"] = sin_phi

        elif method_clean in ("calibration_tone", "tone"):
            cov_iq = float(np.mean(i_ac * q_ac))
            var_i = float(np.mean(i_ac ** 2))
            var_q = float(np.mean(q_ac ** 2))
            denom = np.sqrt(var_i * var_q + 1e-20)

            sin_phi = float(np.clip(cov_iq / denom, -1.0, 1.0))
            phi_rad = float(np.arcsin(sin_phi))
            phi_deg = float(np.degrees(phi_rad))

            quad_angle_rad = float(np.pi / 2.0 + phi_rad)
            quad_angle_deg = float(90.0 + phi_deg)
            confidence = 0.99
            valid = True

        elif method_clean == "reference_aided":
            if reference_iq is None:
                raise ValueError("reference_iq must be provided when method='reference_aided'")
            ref = _as_complex_array(reference_iq, name="reference_iq")
            _validate_finite_iq(ref, name="reference_iq")
            if len(ref) != n:
                raise ValueError(f"reference_iq length ({len(ref)}) != iq length ({n})")

            cross_qi = np.mean(q_ac * ref.real)
            norm_q = np.sqrt(np.mean(q_ac ** 2) * np.mean(ref.real ** 2) + 1e-20)
            sin_phi = float(np.clip(cross_qi / norm_q, -1.0, 1.0))
            phi_rad = float(np.arcsin(sin_phi))
            phi_deg = float(np.degrees(phi_rad))

            quad_angle_rad = float(np.pi / 2.0 + phi_rad)
            quad_angle_deg = float(90.0 + phi_deg)
            confidence = 1.0
            valid = True

        else:
            raise ValueError(f"Unknown phase estimation method '{method}'. Must be 'statistical', 'calibration_tone', or 'reference_aided'")

        self.result = IQPhaseImbalanceResult(
            estimated_quadrature_angle_rad=quad_angle_rad,
            estimated_quadrature_angle_deg=quad_angle_deg,
            phase_imbalance_rad=phi_rad,
            phase_imbalance_deg=phi_deg,
            confidence=confidence,
            valid=valid,
            method=method_clean,
            metadata=meta,
        )
        return self.result

    def analyze(self, *args: Any, **kwargs: Any) -> IQPhaseImbalanceResult:
        """Alias for compute()."""
        return self.compute(*args, **kwargs)


IQPhaseImbalanceEstimator = IQPhaseImbalanceAnalyzer


def compute_phase_imbalance(
    iq: Any = None,
    sample_rate_hz: float = 1.0,
    sample_rate: float | None = None,
    dc_result: Any | None = None,
    dc_offset_result: Any | None = None,
    dc_offset: Any | None = None,
    gain_result: Any | None = None,
    gain_imbalance_result: Any | None = None,
    gain_imbalance: Any | None = None,
    method: str = "statistical",
    reference_iq: Any | None = None,
    iq_data: Any = None,
    **kwargs: Any,
) -> IQPhaseImbalanceResult:
    """Convenience functional wrapper for IQPhaseImbalanceAnalyzer."""
    data = iq if iq is not None else iq_data
    fs = sample_rate if sample_rate is not None else sample_rate_hz
    dc_res = dc_result if dc_result is not None else (dc_offset_result if dc_offset_result is not None else dc_offset)
    gain_res = gain_result if gain_result is not None else (gain_imbalance_result if gain_imbalance_result is not None else gain_imbalance)

    analyzer = IQPhaseImbalanceAnalyzer()
    return analyzer.compute(
        iq=data,
        sample_rate_hz=fs,
        dc_result=dc_res,
        gain_result=gain_res,
        method=method,
        reference_iq=reference_iq,
        **kwargs,
    )


estimate_phase_imbalance = compute_phase_imbalance

__all__ = [
    "IQPhaseImbalanceAnalyzer",
    "IQPhaseImbalanceEstimator",
    "IQPhaseImbalanceResult",
    "compute_phase_imbalance",
    "estimate_phase_imbalance",
]

