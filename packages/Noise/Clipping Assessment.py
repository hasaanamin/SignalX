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
from .base import BaseNoiseMetric, NoiseMetricFactory
from ._common import _as_iq_array, _extract_sample_rate


@NoiseMetricFactory.register(["clipping_assessment", "clipping", "hard_clipping", "adc_overrange", "clip_detector"])
class ClippingAssessmentMetric(BaseNoiseMetric):
    """
    Assesses Hard Peak Clipping and ADC Rail Over-Range:
      - Clipped Sample Count & Percentage (Re, Im, and Complex Magnitude)
      - Maximum Consecutive Rail Run-Length
      - Crest Factor / PAPR Deficit due to Peak Truncation
      - Severity Rating: 'NONE', 'MILD', 'MODERATE', 'SEVERE'
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "clipping_assessment"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        threshold: float | None = None,
        threshold_ratio: float = 0.99,
        **kwargs: Any,
    ) -> dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {"is_clipped": False, "clip_percentage": 0.0, "severity": "NONE"}

        i_ch = np.real(z)
        q_ch = np.imag(z)
        mag = np.abs(z)

        # Determine clipping threshold (absolute voltage or relative to peak)
        max_peak = float(max(np.max(np.abs(i_ch)), np.max(np.abs(q_ch)), np.max(mag)))
        if threshold is not None and threshold > 0:
            v_clip = float(threshold)
        else:
            v_clip = max_peak * threshold_ratio

        # Detect clipped samples on I, Q, and Envelope
        clip_i = np.abs(i_ch) >= v_clip
        clip_q = np.abs(q_ch) >= v_clip
        clip_mag = mag >= v_clip
        clip_any = clip_i | clip_q | clip_mag

        n_clipped = int(np.sum(clip_any))
        clip_pct = float((n_clipped / n) * 100.0)

        # Run-length analysis (longest sequence of consecutive clipped samples)
        max_consecutive = 0
        if n_clipped > 0:
            runs = np.diff(np.where(np.concatenate(([clip_any[0]], clip_any[:-1] != clip_any[1:], [True])))[0])[::2]
            max_consecutive = int(np.max(runs)) if runs.size > 0 else 1

        # PAPR and Crest Factor Deficit
        rms_val = float(np.sqrt(np.mean(mag ** 2)))
        observed_papr_db = float(10.0 * np.log10(max(1e-12, (max_peak ** 2) / max(1e-18, rms_val ** 2))))
        observed_cf = float(max_peak / max(1e-12, rms_val))

        # Severity Classification
        if clip_pct == 0.0 or n_clipped == 0:
            severity = "NONE"
        elif clip_pct < 0.1 and max_consecutive <= 2:
            severity = "MILD"
        elif clip_pct < 2.0 or max_consecutive <= 5:
            severity = "MODERATE"
        else:
            severity = "SEVERE"

        return {
            "is_clipped": bool(n_clipped > 0),
            "clip_count": n_clipped,
            "clip_percentage": clip_pct,
            "clip_threshold": v_clip,
            "max_consecutive_clipped_samples": max_consecutive,
            "clip_count_i": int(np.sum(clip_i)),
            "clip_count_q": int(np.sum(clip_q)),
            "peak_amplitude": max_peak,
            "rms_amplitude": rms_val,
            "observed_papr_db": observed_papr_db,
            "observed_crest_factor": observed_cf,
            "severity": severity,
            "sample_rate": fs,
        }


def clipping_assessment(
    iq_data: Any,
    sample_rate: float | None = None,
    threshold: float | None = None,
    threshold_ratio: float = 0.99,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Functional DSP wrapper for clipping and ADC rail assessment.
    """
    metric = ClippingAssessmentMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, threshold=threshold, threshold_ratio=threshold_ratio, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  ClippingAssessmentMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = ClippingAssessmentMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
