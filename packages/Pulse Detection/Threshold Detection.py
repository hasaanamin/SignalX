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


@PulseDetectionFactory.register(["threshold_detection", "threshold", "detect_thresholds"])
class ThresholdDetectionMetric(BasePulseDetectionMetric):
    """
    Stage 2: Robust & Adaptive Threshold Detection.
    Estimates streaming background noise floor and noise scale using ordered-statistic CFAR (OS-CFAR)
    on the detection statistic domain, using a 35th-percentile baseline cut to remain immune to high
    pulse occupancy, with theoretical scale bounding (sigma >= mu / sqrt(M)) and zero upward drift.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reset()

    def __del__(self) -> None:
        super().__del__()

    def reset(self) -> None:
        """Resets streaming background estimation state."""
        self._last_noise_floor = 1.0
        self._last_noise_scale = 1.0

    @property
    def name(self) -> str:
        return "threshold_detection"

    def compute(
        self,
        detection_statistic_or_signal: Any,
        k_high: float = 3.2,
        k_low: float = 1.4,
        reference_window_samples: int = 64,
        guard_samples: int = 12,
        window_size: int | None = None,
        integration_window_samples: int = 4,
        adaptive: bool = False,
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Calculates robust High and Low detection thresholds.

        Parameters:
            detection_statistic_or_signal: 1-D detection statistic from Stage 1 or raw complex IQ.
            k_high: Sensitivity multiplier for High trigger threshold (default 2.5).
            k_low: Sensitivity multiplier for Low continuation threshold (default 0.8).
            reference_window_samples: Length of trailing noise reference window (R).
            guard_samples: Guard cells between reference window and test sample (G).
            window_size: Backward compatibility alias for reference_window_samples.
            integration_window_samples: M samples from Stage 1 (default 4).
            adaptive: If True, computes per-sample adaptive thresholds; if False, returns scalar thresholds.
        """
        # Parse input
        if isinstance(detection_statistic_or_signal, dict) and "detection_statistic" in detection_statistic_or_signal:
            det_stat = np.asarray(detection_statistic_or_signal["detection_statistic"], dtype=np.float64)
            M = int(detection_statistic_or_signal.get("integration_window_samples", integration_window_samples))
        else:
            arr = _as_signal_array(detection_statistic_or_signal)
            if np.iscomplexobj(arr):
                from . import EnvelopeExtractionMetric
                env_res = EnvelopeExtractionMetric().compute(arr, return_dict=True, **kwargs)
                det_stat = env_res["detection_statistic"]
                M = int(env_res.get("integration_window_samples", integration_window_samples))
            else:
                det_stat = arr.astype(np.float64)
                M = int(integration_window_samples)

        n = det_stat.size
        if n == 0:
            raise ValueError("Detection statistic must contain at least one sample")

        R = max(16, int(window_size if window_size is not None else reference_window_samples))
        G = max(4, int(guard_samples))
        k_hi = float(k_high)
        k_lo = float(k_low)
        sqrt_M = np.sqrt(max(1, M))

        # 1. Global Robust Baseline (Lower 35th percentile cut: immune to high pulse occupancy)
        p35_glob = np.percentile(det_stat, 35)
        low_glob = det_stat[det_stat <= p35_glob]
        if low_glob.size > 0:
            med_glob = float(np.median(low_glob) / 0.48)
            scale_glob = max(float(1.4826 * np.median(np.abs(low_glob - np.median(low_glob))) / 0.38), med_glob / sqrt_M)
        else:
            med_glob = float(np.median(det_stat))
            scale_glob = max(float(1.4826 * np.median(np.abs(det_stat - med_glob))), med_glob / sqrt_M)

        if scale_glob < 1e-6:
            scale_glob = max(1e-6, med_glob / sqrt_M)

        # 2. Non-Adaptive Scalar Mode
        if not adaptive or n < (R + G):
            return {
                "high_threshold": float(med_glob + k_hi * scale_glob),
                "low_threshold": float(med_glob + k_lo * scale_glob),
                "noise_floor": float(med_glob),
                "noise_scale": float(scale_glob),
                "noise_std": float(scale_glob),  # Backward compatibility alias
                "k_high": k_hi,
                "k_low": k_lo,
                "is_adaptive": False,
            }

        # 3. Causal Trailing Reference Window CFAR with Global Bounding
        noise_floor = np.empty(n, dtype=np.float64)
        noise_scale = np.empty(n, dtype=np.float64)

        for i in range(n):
            if i >= (R + G):
                ref_slice = det_stat[i - G - R : i - G]
                p35_ref = np.percentile(ref_slice, 35)
                low_ref = ref_slice[ref_slice <= p35_ref]
                if low_ref.size > 4:
                    ref_nf = float(np.median(low_ref) / 0.48)
                    ref_ns = max(float(1.4826 * np.median(np.abs(low_ref - np.median(low_ref))) / 0.38), ref_nf / sqrt_M)
                else:
                    ref_nf = med_glob
                    ref_ns = scale_glob
            else:
                ref_nf = med_glob
                ref_ns = scale_glob

            # Bound between 0.5x and 1.8x of robust global baseline
            noise_floor[i] = float(np.clip(ref_nf, med_glob * 0.5, med_glob * 1.8))
            noise_scale[i] = float(np.clip(ref_ns, scale_glob * 0.5, scale_glob * 1.8))

        high_threshold = noise_floor + k_hi * noise_scale
        low_threshold = noise_floor + k_lo * noise_scale

        return {
            "high_threshold": high_threshold,
            "low_threshold": low_threshold,
            "noise_floor": noise_floor,
            "noise_scale": noise_scale,
            "noise_std": noise_scale,  # Backward compatibility alias
            "k_high": k_hi,
            "k_low": k_lo,
            "is_adaptive": True,
        }


_instance = ThresholdDetectionMetric()
threshold_detection = _instance.compute
detect_thresholds = _instance.compute

__all__ = ["ThresholdDetectionMetric", "threshold_detection", "detect_thresholds"]


def main() -> None:
    print("=" * 60)
    print("  ThresholdDetectionMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = ThresholdDetectionMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
