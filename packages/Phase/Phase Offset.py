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

import numpy as np
from .base import BasePhaseMetric, PhaseMetricFactory
from ._common import _as_iq_array, _extract_sample_rate


@PhaseMetricFactory.register(["phase_offset", "initial_phase", "carrier_phase_offset"])
class PhaseOffsetMetric(BasePhaseMetric):
    """Blind carrier phase offset and constellation orientation estimator."""

    def __init__(self, **kwargs) -> None:
        """Constructor."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor."""
        super().__del__()

    @property
    def name(self) -> str:
        return "phase_offset"

    def compute(
        self,
        iq_data,
        *,
        sample_rate: float | None = None,
        carrier_freq_hz: float | None = None,
        modulation: str | None = None,
        order: int | None = None,
        sym_order: int | None = None,
        deg: bool = False,
        mode: str = "offset",
        candidate_orders: tuple[int, ...] = (2, 4, 8, 16),
        min_symmetry_score: float = 0.10,
        **kwargs,
    ) -> float | dict:
        # Format IQ data
        z = _as_iq_array(iq_data)
        z = np.asarray(z, dtype=np.complex128).reshape(-1)
        n = z.size
        if n == 0:
            return 0.0

        # Known CW / unmodulated carrier derotation
        if carrier_freq_hz is not None:
            if sample_rate is None:
                try:
                    fs = _extract_sample_rate(iq_data, sample_rate=None, default_fs=None)
                except Exception:
                    fs = None
            else:
                fs = float(sample_rate)

            if fs is None or not np.isfinite(fs) or fs <= 0.0:
                raise ValueError("A valid sample_rate is required when carrier_freq_hz is provided.")

            fc = float(carrier_freq_hz)
            t = np.arange(n, dtype=np.float64) / fs
            derotated = z * np.exp(-1j * 2.0 * np.pi * fc * t)
            coherent_sum = np.mean(derotated)
            phi = float("nan") if np.abs(coherent_sum) <= 1e-15 else float(np.angle(coherent_sum))
            output_phi = float(np.degrees(phi)) if deg else phi

            if str(mode).lower() in ("all", "full", "dict"):
                return {
                    "phase_offset": output_phi,
                    "method": "known_carrier_derotation",
                    "detected_symmetry_order": None,
                    "symmetry_score": None,
                    "symmetry_scores": None,
                    "unit": "degrees" if deg else "radians",
                }
            return output_phi

        # Remove DC and compute phase-only unit phasors
        zc = z - np.mean(z)
        magnitude = np.abs(zc)
        valid = magnitude > 1e-12
        if np.count_nonzero(valid) < 2:
            return 0.0

        u = zc[valid] / magnitude[valid]

        # Candidate symmetry orders
        if sym_order is not None:
            k_values = [int(sym_order)]
            if k_values[0] < 1:
                raise ValueError("sym_order must be a positive integer.")
        else:
            k_values = sorted({int(k) for k in candidate_orders if int(k) > 0})
            if not k_values:
                raise ValueError("candidate_orders must contain at least one positive integer.")

        # Circular moments: M_K = E[u^K]
        scores: dict[int, float] = {}
        orientations: dict[int, float] = {}
        moments: dict[int, complex] = {}

        for k in k_values:
            moment_k = np.mean(u ** k)
            moments[k] = complex(moment_k)
            scores[k] = float(np.abs(moment_k))
            orientations[k] = float("nan") if np.abs(moment_k) <= 1e-15 else float(np.angle(moment_k) / float(k))

        # Select best symmetry order
        if sym_order is not None:
            best_k = int(sym_order)
        else:
            maximum_score = max(scores.values())
            if maximum_score < float(min_symmetry_score):
                result = float("nan")
                if str(mode).lower() in ("all", "full", "dict"):
                    return {
                        "phase_offset": result,
                        "phase_orientation": result,
                        "detected_symmetry_order": None,
                        "symmetry_score": maximum_score,
                        "symmetry_scores": scores,
                        "reliable": False,
                        "method": "blind_rotational_symmetry",
                        "unit": "degrees" if deg else "radians",
                    }
                return result

            relative_threshold = 0.75 * maximum_score
            best_k = k_values[-1]
            for k in k_values:
                if scores[k] >= relative_threshold:
                    best_k = k
                    break

        phi = orientations[best_k]
        if not np.isfinite(phi):
            result = float("nan")
            if str(mode).lower() in ("all", "full", "dict"):
                return {
                    "phase_offset": result,
                    "phase_orientation": result,
                    "detected_symmetry_order": best_k,
                    "symmetry_score": scores[best_k],
                    "symmetry_scores": scores,
                    "reliable": False,
                    "method": "blind_rotational_symmetry",
                    "unit": "degrees" if deg else "radians",
                }
            return result

        # Wrap to fundamental symmetry interval [-pi/K, +pi/K)
        ambiguity_period = 2.0 * np.pi / float(best_k)
        phi = ((phi + ambiguity_period / 2.0) % ambiguity_period - ambiguity_period / 2.0)
        output_phi = float(np.degrees(phi)) if deg else float(phi)

        if str(mode).lower() in ("all", "full", "dict"):
            score = scores[best_k]
            return {
                "phase_offset": output_phi,
                "phase_orientation": output_phi,
                "detected_symmetry_order": best_k,
                "symmetry_score": score,
                "symmetry_scores": scores,
                "reliable": bool(score >= float(min_symmetry_score)),
                "ambiguity_period": float(np.degrees(ambiguity_period)) if deg else float(ambiguity_period),
                "method": "blind_rotational_symmetry",
                "unit": "degrees" if deg else "radians",
            }

        return output_phi


_instance = PhaseOffsetMetric()
phase_offset = _instance.compute

__all__ = ["PhaseOffsetMetric", "phase_offset"]


def main() -> None:
    print("=" * 60)
    print("  PhaseOffsetMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = PhaseOffsetMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
