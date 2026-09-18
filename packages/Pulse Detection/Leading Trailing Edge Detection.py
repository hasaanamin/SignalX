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
from .base import BasePulseDetectionMetric, PulseDetectionFactory
from ._common import _as_signal_array


@PulseDetectionFactory.register(["leading_trailing_edge_detection", "edge_detection", "detect_edges", "edges"])
class LeadingTrailingEdgeDetectionMetric(BasePulseDetectionMetric):
    """
    Stage 5: Sub-Sample Leading & Trailing Edge Detection.
    Refines pulse edges on the instantaneous amplitude envelope |z| using local baseline estimation,
    robust pulse-top levels (avoiding single-spike bias), and fractional-sample linear interpolation
    by searching around the coarse start for the leading edge and around the coarse end for the trailing edge.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "leading_trailing_edge_detection"

    def compute(
        self,
        amplitude_envelope_or_signal: Any,
        boundaries: list[dict[str, Any]] | None = None,
        edge_level_percent: float = 50.0,
        search_margin_samples: int = 8,
        use_derivative_fallback: bool = True,
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """
        Calculates refined leading and trailing edge positions on the amplitude envelope.

        Parameters:
            amplitude_envelope_or_signal: Instantaneous amplitude envelope |z| or raw complex IQ.
            boundaries: Coarse pulse boundaries from Stage 4 PulseBoundariesMetric.
            edge_level_percent: Crossing percentage between local baseline and robust peak (default 50% for FWHM).
            search_margin_samples: Search margin around coarse start and end bounds (default 8).
            use_derivative_fallback: Fallback to slope extremes if crossing is ambiguous.
        """
        # Parse amplitude envelope
        if isinstance(amplitude_envelope_or_signal, dict) and "amplitude_envelope" in amplitude_envelope_or_signal:
            env = np.asarray(amplitude_envelope_or_signal["amplitude_envelope"], dtype=np.float64)
        else:
            arr = _as_signal_array(amplitude_envelope_or_signal)
            env = np.abs(arr).astype(np.float64) if np.iscomplexobj(arr) else arr.astype(np.float64)

        n = env.size
        if n == 0:
            return []

        # Auto-compute boundaries if not provided
        if boundaries is None:
            from . import PulseBoundariesMetric
            boundaries = PulseBoundariesMetric().compute(env, **kwargs)

        if not boundaries:
            return []

        edge_fraction = edge_level_percent / 100.0
        edge_results: list[dict[str, Any]] = []

        for b in boundaries:
            s_coarse = int(b["start_idx"])
            e_coarse = int(b["end_idx"])
            M = int(b.get("integration_delay_samples", 4))

            # -------------------------------------------------------------
            # 1. Local Noise Baseline Estimation
            # -------------------------------------------------------------
            noise_samples = []
            if s_coarse > 4:
                noise_samples.extend(env[max(0, s_coarse - 16) : s_coarse])
            if e_coarse < n - 5:
                noise_samples.extend(env[e_coarse + 1 : min(n, e_coarse + 17)])

            if noise_samples:
                v_local_base = float(np.median(noise_samples))
            else:
                surround = env[max(0, s_coarse - 8) : min(n, e_coarse + 9)]
                v_local_base = float(np.percentile(surround, 10)) if surround.size > 0 else float(np.min(env))

            # -------------------------------------------------------------
            # 2. Robust Pulse Top Level & Peak Index
            # -------------------------------------------------------------
            pulse_slice = env[s_coarse : e_coarse + 1]
            if pulse_slice.size >= 4:
                v_robust_top = float(np.percentile(pulse_slice, 80))
            elif pulse_slice.size > 0:
                v_robust_top = float(np.max(pulse_slice))
            else:
                v_robust_top = float(env[s_coarse])

            if v_robust_top <= v_local_base:
                v_robust_top = float(np.max(env[max(0, s_coarse - 4) : min(n, e_coarse + 5)]))

            peak_offset = int(np.argmax(env[s_coarse : e_coarse + 1]))
            peak_idx = s_coarse + peak_offset
            peak_val = float(env[peak_idx])

            # Target 50% FWHM crossing level
            v_target = v_local_base + edge_fraction * max(1e-6, v_robust_top - v_local_base)

            # -------------------------------------------------------------
            # 3. Leading Edge Search: Search around s_coarse (s_coarse - M - 2 to s_coarse + 8)
            # -------------------------------------------------------------
            leading_edge_frac = float(s_coarse)
            lead_found = False

            lead_search_start = max(0, s_coarse - M - 2)
            lead_search_end = min(n - 1, s_coarse + 8)

            for idx in range(lead_search_start, lead_search_end):
                if env[idx] <= v_target <= env[idx + 1]:
                    denom = float(env[idx + 1] - env[idx])
                    if denom > 1e-12:
                        frac = (v_target - env[idx]) / denom
                        leading_edge_frac = float(idx + frac)
                        lead_found = True
                        break

            if not lead_found:
                leading_edge_frac = float(s_coarse)

            # -------------------------------------------------------------
            # 4. Trailing Edge Search: Search around e_coarse (e_coarse - 8 to e_coarse + 8)
            # -------------------------------------------------------------
            trailing_edge_frac = float(e_coarse)
            trail_found = False

            trail_search_start = max(0, e_coarse - 8)
            trail_search_end = min(n - 1, e_coarse + 8)

            for idx in range(trail_search_start, trail_search_end):
                if env[idx] >= v_target >= env[idx + 1]:
                    denom = float(env[idx] - env[idx + 1])
                    if denom > 1e-12:
                        frac = (env[idx] - v_target) / denom
                        trailing_edge_frac = float(idx + frac)
                        trail_found = True
                        break

            if not trail_found:
                trailing_edge_frac = float(e_coarse)

            # Ensure leading_edge <= trailing_edge
            if trailing_edge_frac < leading_edge_frac:
                trailing_edge_frac = float(e_coarse)
                leading_edge_frac = float(s_coarse)

            # -------------------------------------------------------------
            # 5. Rise Time Calculation (10% to 90% leading transition)
            # -------------------------------------------------------------
            v_10 = v_local_base + 0.10 * (v_robust_top - v_local_base)
            v_90 = v_local_base + 0.90 * (v_robust_top - v_local_base)

            t_10 = leading_edge_frac
            t_90 = leading_edge_frac

            for idx in range(lead_search_start, lead_search_end):
                if env[idx] <= v_10 <= env[idx + 1]:
                    d = float(env[idx + 1] - env[idx])
                    t_10 = float(idx + (v_10 - env[idx]) / d) if d > 1e-12 else float(idx)
                    break

            for idx in range(int(t_10), min(n - 1, lead_search_end + 4)):
                if env[idx] <= v_90 <= env[idx + 1]:
                    d = float(env[idx + 1] - env[idx])
                    t_90 = float(idx + (v_90 - env[idx]) / d) if d > 1e-12 else float(idx)
                    break

            rise_time_samples = max(0.0, t_90 - t_10)

            # Edge confidence
            snr_contrast = (v_robust_top - v_local_base) / max(1e-6, v_local_base)
            confidence = float(np.clip(snr_contrast / (snr_contrast + 1.0), 0.1, 1.0))

            edge_results.append({
                "leading_edge": round(leading_edge_frac, 4),
                "trailing_edge": round(trailing_edge_frac, 4),
                "leading_edge_idx": int(round(leading_edge_frac)),
                "trailing_edge_idx": int(round(trailing_edge_frac)),
                "rise_time_samples": round(rise_time_samples, 4),
                "peak_idx": peak_idx,
                "peak_val": peak_val,
                "robust_top_level": v_robust_top,
                "local_noise_baseline": v_local_base,
                "target_level": v_target,
                "edge_confidence": round(confidence, 3),
                "edge_level_percent": edge_level_percent,
            })

        return edge_results


_instance = LeadingTrailingEdgeDetectionMetric()
leading_trailing_edge_detection = _instance.compute
detect_edges = _instance.compute

__all__ = ["LeadingTrailingEdgeDetectionMetric", "leading_trailing_edge_detection", "detect_edges"]


def main() -> None:
    print("=" * 60)
    print("  LeadingTrailingEdgeDetectionMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = LeadingTrailingEdgeDetectionMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
