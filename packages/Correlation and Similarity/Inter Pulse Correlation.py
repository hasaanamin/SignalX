from __future__ import annotations
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    _pkg_dir = Path(__file__).resolve().parent
    _src_dir = _pkg_dir.parent.parent
    for _p in [str(_src_dir), str(_pkg_dir.parent)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import importlib
    importlib.import_module(f"packages.{_pkg_dir.name}")
    __package__ = f"packages.{_pkg_dir.name}"

from typing import Any
import numpy as np

from .base import BaseCorrelationMetric, CorrelationMetricFactory, InterPulseCorrelationResult
from ._common import _as_iq_array, _extract_sample_rate, _compute_xcorr_core


@CorrelationMetricFactory.register(["inter_pulse_correlation", "interpulse_correlation", "pulse_to_pulse_correlation", "pulse_correlation"])
class InterPulseCorrelationMetric(BaseCorrelationMetric):
    """
    Inter-Pulse Cross-Correlation Stage.
    Consumes segmented complex IQ pulse candidates from the Pulse Detector
    and performs pairwise normalized cross-correlation.

    Mathematical Formulation:
      For two pulse complex IQ vectors p_i[n] and p_j[n]:
        R_ij[h] = xcorr(p_i, p_j)
        rho_ij[h] = R_ij[h] / sqrt(R_ii[0] * R_jj[0])

      Inter-Pulse Correlation Score:
        C_ij = max_h |rho_ij[h]|,   0 <= C_ij <= 1

      Optimal Alignment Lag:
        h_ij = argmax_h |rho_ij[h]|

      Relative Inter-Pulse Phase Offset:
        phi_ij = angle(rho_ij[h_ij])

    Unequal Pulse Length Handling:
      Zero-pads shorter pulse to length of longer pulse before computing
      coefficient-normalized cross-correlation without data truncation.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "inter_pulse_correlation"

    def compute(
        self,
        pulse_a: Any,
        pulse_b: Any = None,
        *,
        pulse_list: list[Any] | None = None,
        sample_rate: float | None = None,
        maxlag: int | None = None,
        method: str = "auto",
        pulse_id_a: int | str = 0,
        pulse_id_b: int | str = 1,
        as_dict: bool = False,
        **kwargs: Any,
    ) -> InterPulseCorrelationResult | dict[str, Any] | np.ndarray:
        """
        Computes pairwise inter-pulse correlation for two pulses or full matrix for a list of pulses.

        Parameters:
            pulse_a: First pulse IQ array or PulseCandidate object.
            pulse_b: Second pulse IQ array or PulseCandidate object.
            pulse_list: Optional list of pulse IQ arrays/PulseCandidates for full pairwise matrix C.
            sample_rate (Fs): Sampling rate in Hz.
            maxlag: Maximum lag search range.
            pulse_id_a: Identifier for pulse A.
            pulse_id_b: Identifier for pulse B.
            as_dict: If True, returns dictionary representation.

        Returns:
            InterPulseCorrelationResult, dictionary, or 2D matrix C_ij.
        """
        # If pulse_list provided or pulse_a is a list of pulses (>2)
        is_pulse_collection = (pulse_list is not None) or (
            isinstance(pulse_a, (list, tuple)) and pulse_b is None and len(pulse_a) > 1 and (
                isinstance(pulse_a[0], (np.ndarray, list, tuple)) or hasattr(pulse_a[0], "iq_samples")
            )
        )
        if is_pulse_collection:
            pulses = pulse_list if pulse_list is not None else list(pulse_a)

            k = len(pulses)
            corr_matrix = np.eye(k, dtype=np.float64)
            lag_matrix = np.zeros((k, k), dtype=np.int64)
            phase_matrix = np.zeros((k, k), dtype=np.float64)

            # Pre-parse pulse IQ arrays
            parsed_pulses = [_as_iq_array(p) for p in pulses]
            fs = _extract_sample_rate(data=parsed_pulses[0] if k > 0 else None, kwargs=kwargs, sample_rate=sample_rate, default_fs=1.0)

            for i in range(k):
                for j in range(i + 1, k):
                    res_ij = _compute_xcorr_core(
                        parsed_pulses[i],
                        parsed_pulses[j],
                        maxlag=maxlag,
                        scaleopt="coeff",
                        method=method,
                        sample_rate=fs,
                    )
                    score = res_ij.peak_magnitude
                    lag = res_ij.peak_lag_samples
                    phi = res_ij.phase_at_peak_rad

                    corr_matrix[i, j] = score
                    corr_matrix[j, i] = score
                    lag_matrix[i, j] = lag
                    lag_matrix[j, i] = -lag
                    phase_matrix[i, j] = phi
                    phase_matrix[j, i] = -phi

            if as_dict:
                return {
                    "correlation_matrix": corr_matrix,
                    "lag_matrix_samples": lag_matrix,
                    "phase_matrix_rad": phase_matrix,
                    "num_pulses": k,
                    "sample_rate": fs,
                }
            return corr_matrix

        # Pairwise pulse correlation between pulse_a and pulse_b
        id_a = getattr(pulse_a, "pulse_id", pulse_id_a)
        id_b = getattr(pulse_b, "pulse_id", pulse_id_b) if pulse_b is not None else pulse_id_b

        fs = _extract_sample_rate(data=pulse_a, kwargs=kwargs, sample_rate=sample_rate, default_fs=1.0)
        p_a = _as_iq_array(pulse_a)
        p_b = _as_iq_array(pulse_b) if pulse_b is not None else p_a

        corr_res = _compute_xcorr_core(
            p_a,
            p_b,
            maxlag=maxlag,
            scaleopt="coeff",
            method=method,
            sample_rate=fs,
        )

        res = InterPulseCorrelationResult(
            pulse_id_a=id_a,
            pulse_id_b=id_b,
            correlation_result=corr_res,
            peak_similarity=corr_res.peak_magnitude,
            peak_lag_samples=corr_res.peak_lag_samples,
            peak_lag_sec=corr_res.peak_lag_sec,
            phase_offset_rad=corr_res.phase_at_peak_rad,
        )

        return res.to_dict() if as_dict else res


def compute_inter_pulse_correlation(
    pulse_a: Any,
    pulse_b: Any = None,
    *,
    pulse_list: list[Any] | None = None,
    sample_rate: float | None = None,
    maxlag: int | None = None,
    method: str = "auto",
    pulse_id_a: int | str = 0,
    pulse_id_b: int | str = 1,
    as_dict: bool = False,
    **kwargs: Any,
) -> InterPulseCorrelationResult | dict[str, Any] | np.ndarray:
    """
    Convenience functional API for inter-pulse correlation computation.
    """
    metric = InterPulseCorrelationMetric()
    return metric.compute(
        pulse_a,
        pulse_b,
        pulse_list=pulse_list,
        sample_rate=sample_rate,
        maxlag=maxlag,
        method=method,
        pulse_id_a=pulse_id_a,
        pulse_id_b=pulse_id_b,
        as_dict=as_dict,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  InterPulseCorrelationMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    metric = InterPulseCorrelationMetric()
    result = metric.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
