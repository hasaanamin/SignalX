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
class IQGainImbalanceResult:
    """
    Result container for I/Q Gain Imbalance estimation.

    Sign Convention
    ---------------
    positive gain_imbalance_db > 0 means the In-Phase (I) branch has larger
    amplitude/power than the Quadrature (Q) branch.
    gain_ratio = g_I / g_Q.
    """
    i_gain_estimate: float = 1.0
    q_gain_estimate: float = 1.0
    gain_ratio: float = 1.0
    gain_imbalance_db: float = 0.0
    correction_gain: float = 1.0
    method: str = "statistical"
    confidence: float = 1.0
    valid: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def amplitude_imbalance(self) -> float:
        return float(self.gain_ratio - 1.0)

    @property
    def diagnostic(self) -> str | None:
        return self.metadata.get("diagnostic") or self.metadata.get("reason")


class IQGainImbalanceAnalyzer:
    """
    I/Q Gain (Amplitude) Imbalance analyzer for complex baseband IQ receivers.

    Quantifies the amplitude scaling mismatch between the In-Phase (I) and
    Quadrature (Q) branches.
    Does NOT import or depend on any sibling analysis modules.
    """

    def __init__(self) -> None:
        self.result: IQGainImbalanceResult | None = None

    def compute(
        self,
        iq: Any,
        sample_rate_hz: float = 1.0,
        dc_result: Any | None = None,
        method: str = "statistical",
        reference_iq: Any | None = None,
        **kwargs: Any,
    ) -> IQGainImbalanceResult:
        """
        Estimates I/Q gain imbalance.

        Parameters
        ----------
        iq : Any
            Input complex IQ signal.
        sample_rate_hz : float
            Sampling rate in Hz (default 1.0).
        dc_result : Any | None
            Optional precomputed DC offset result to reuse.
        method : str
            Estimation method: 'statistical', 'calibration_tone', or 'reference_aided'.
        reference_iq : Any | None
            Clean reference signal for reference-aided estimation.

        Returns
        -------
        IQGainImbalanceResult
            Gain imbalance metrics.
        """
        x = _as_complex_array(iq, name="iq")
        _validate_finite_iq(x, name="iq")
        n = len(x)

        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be strictly positive, got {sample_rate_hz}")

        if n < 4:
            self.result = IQGainImbalanceResult(
                i_gain_estimate=1.0,
                q_gain_estimate=1.0,
                gain_ratio=1.0,
                gain_imbalance_db=0.0,
                correction_gain=1.0,
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

        method_clean = str(method).lower().strip()
        meta: dict[str, Any] = {"dc_removed": dc_c, "sample_count": n, "sample_rate_hz": float(sample_rate_hz)}

        if method_clean in ("statistical", "power", "rms"):
            p_i = float(np.mean(i_ac ** 2))
            p_q = float(np.mean(q_ac ** 2))
            meta["p_i"] = p_i
            meta["p_q"] = p_q

            eps = 1e-20
            if p_i < eps or p_q < eps:
                self.result = IQGainImbalanceResult(
                    i_gain_estimate=float(np.sqrt(p_i)),
                    q_gain_estimate=float(np.sqrt(p_q)),
                    gain_ratio=1.0,
                    gain_imbalance_db=0.0,
                    correction_gain=1.0,
                    confidence=0.0,
                    valid=False,
                    method="statistical",
                    metadata={"reason": "Zero AC power in I or Q branch"},
                )
                return self.result

            g_i = float(np.sqrt(p_i))
            g_q = float(np.sqrt(p_q))
            gain_ratio = float(g_i / g_q)
            gain_db = float(20.0 * np.log10(gain_ratio))
            corr_gain = float(1.0 / gain_ratio)
            confidence = float(np.clip(1.0 - (10.0 / np.sqrt(n)), 0.1, 1.0))
            valid = True

        elif method_clean in ("calibration_tone", "tone"):
            p_i = float(np.mean(i_ac ** 2))
            p_q = float(np.mean(q_ac ** 2))
            eps = 1e-20
            if p_i < eps or p_q < eps:
                self.result = IQGainImbalanceResult(
                    i_gain_estimate=0.0,
                    q_gain_estimate=0.0,
                    gain_ratio=1.0,
                    gain_imbalance_db=0.0,
                    correction_gain=1.0,
                    confidence=0.0,
                    valid=False,
                    method="calibration_tone",
                    metadata={"reason": "Zero AC tone power in I or Q branch"},
                )
                return self.result

            g_i = float(np.sqrt(2.0 * p_i))
            g_q = float(np.sqrt(2.0 * p_q))
            gain_ratio = float(g_i / g_q)
            gain_db = float(20.0 * np.log10(gain_ratio))
            corr_gain = float(1.0 / gain_ratio)
            confidence = 0.99
            valid = True

        elif method_clean == "reference_aided":
            if reference_iq is None:
                raise ValueError("reference_iq must be provided when method='reference_aided'")
            ref = _as_complex_array(reference_iq, name="reference_iq")
            _validate_finite_iq(ref, name="reference_iq")
            if len(ref) != n:
                raise ValueError(f"reference_iq length ({len(ref)}) != iq length ({n})")

            i_ref = ref.real
            q_ref = ref.imag
            p_ref_i = float(np.mean(i_ref ** 2))
            p_ref_q = float(np.mean(q_ref ** 2))
            p_rx_i = float(np.mean(i_ac ** 2))
            p_rx_q = float(np.mean(q_ac ** 2))

            eps = 1e-20
            g_i = float(np.sqrt((p_rx_i + eps) / (p_ref_i + eps)))
            g_q = float(np.sqrt((p_rx_q + eps) / (p_ref_q + eps)))
            gain_ratio = float(g_i / (g_q + eps))
            gain_db = float(20.0 * np.log10(gain_ratio))
            corr_gain = float(1.0 / (gain_ratio + eps))
            confidence = 1.0
            valid = True

        else:
            raise ValueError(f"Unknown gain estimation method '{method}'. Must be 'statistical', 'calibration_tone', or 'reference_aided'")

        self.result = IQGainImbalanceResult(
            i_gain_estimate=g_i,
            q_gain_estimate=g_q,
            gain_ratio=gain_ratio,
            gain_imbalance_db=gain_db,
            correction_gain=corr_gain,
            method=method_clean,
            confidence=confidence,
            valid=valid,
            metadata=meta,
        )
        return self.result

    def analyze(self, *args: Any, **kwargs: Any) -> IQGainImbalanceResult:
        """Alias for compute()."""
        return self.compute(*args, **kwargs)


IQGainImbalanceEstimator = IQGainImbalanceAnalyzer


def compute_gain_imbalance(
    iq: Any = None,
    sample_rate_hz: float = 1.0,
    dc_result: Any | None = None,
    dc_offset_result: Any | None = None,
    dc_offset: Any | None = None,
    method: str = "statistical",
    reference_iq: Any | None = None,
    iq_data: Any = None,
    **kwargs: Any,
) -> IQGainImbalanceResult:
    """Convenience functional wrapper for IQGainImbalanceAnalyzer."""
    data = iq if iq is not None else iq_data
    dc_res = dc_result if dc_result is not None else (dc_offset_result if dc_offset_result is not None else dc_offset)
    sample_rate = kwargs.get("sample_rate", sample_rate_hz)
    analyzer = IQGainImbalanceAnalyzer()
    return analyzer.compute(
        iq=data,
        sample_rate_hz=sample_rate,
        dc_result=dc_res,
        method=method,
        reference_iq=reference_iq,
        **kwargs,
    )


estimate_gain_imbalance = compute_gain_imbalance

__all__ = [
    "IQGainImbalanceAnalyzer",
    "IQGainImbalanceEstimator",
    "IQGainImbalanceResult",
    "compute_gain_imbalance",
    "estimate_gain_imbalance",
]

