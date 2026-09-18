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


@NoiseMetricFactory.register(["saturation_assessment", "saturation", "compression_assessment", "soft_clipping", "non_linear_compression"])
class SaturationAssessmentMetric(BaseNoiseMetric):
    """
    Assesses Front-End Soft Saturation, Gain Compression, and AM-AM Non-Linearity:
      - Amplitude Kurtosis Compression Shift (flattened envelope tail)
      - Peak-to-RMS Compression Index [0.0 .. 1.0]
      - Estimated Gain Compression (dB)
      - Severity Rating: 'LINEAR', 'MILD_COMPRESSION', 'HEAVILY_SATURATED'
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "saturation_assessment"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        nominal_papr_db: float = 3.01,
        **kwargs: Any,
    ) -> dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {"saturation_index": 0.0, "gain_compression_db": 0.0, "severity": "LINEAR"}

        mag = np.abs(z)
        peak_val = float(np.max(mag))
        rms_val = float(np.sqrt(np.mean(mag ** 2)))
        mean_val = float(np.mean(mag))

        if rms_val <= 0 or peak_val <= 0:
            return {"saturation_index": 0.0, "gain_compression_db": 0.0, "severity": "LINEAR"}

        # 1. Constant modulus check (clean CW or PSK has naturally flat envelope)
        std_mag = float(np.std(mag))
        is_constant_modulus = (std_mag / max(1e-12, mean_val)) < 0.05

        if is_constant_modulus:
            sat_index = 0.0
            gain_compression_db = 0.0
            severity = "LINEAR"
            papr_compression_loss_db = 0.0
            kurtosis_val = 1.5
            tail_fraction = 1.0
            observed_papr_db = 0.0
        else:
            # 2. Observed Crest Factor / PAPR vs Nominal
            observed_papr_db = float(10.0 * np.log10(max(1e-12, (peak_val ** 2) / max(1e-18, rms_val ** 2))))
            papr_compression_loss_db = max(0.0, nominal_papr_db - observed_papr_db)

            # 3. Envelope Kurtosis Analysis
            kurtosis_val = float(np.mean(((mag - mean_val) / std_mag) ** 4))

            # 4. High-Amplitude Tail Probability
            near_peak_threshold = 0.90 * peak_val
            tail_fraction = float(np.mean(mag >= near_peak_threshold))

            # 5. Saturation Index [0.0 .. 1.0]
            tail_excess = max(0.0, tail_fraction - 0.08)
            sat_index = float(np.clip(papr_compression_loss_db / 3.0 + tail_excess * 3.0, 0.0, 1.0))
            gain_compression_db = float(sat_index * 3.0)

            # Severity Classification
            if sat_index < 0.15:
                severity = "LINEAR"
            elif sat_index < 0.50:
                severity = "MILD_COMPRESSION"
            else:
                severity = "HEAVILY_SATURATED"

        return {
            "saturation_index": sat_index,
            "gain_compression_db": gain_compression_db,
            "observed_papr_db": observed_papr_db,
            "papr_compression_loss_db": papr_compression_loss_db,
            "envelope_kurtosis": kurtosis_val,
            "near_peak_sample_fraction": tail_fraction,
            "peak_amplitude": peak_val,
            "rms_amplitude": rms_val,
            "severity": severity,
            "sample_rate": fs,
        }


def saturation_assessment(
    iq_data: Any,
    sample_rate: float | None = None,
    nominal_papr_db: float = 3.01,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Functional DSP wrapper for saturation, soft compression, and AM-AM assessment.
    """
    metric = SaturationAssessmentMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, nominal_papr_db=nominal_papr_db, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  SaturationAssessmentMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = SaturationAssessmentMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
