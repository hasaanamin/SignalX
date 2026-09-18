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
from .base import BasePulseDetectorStage, PulseDetectorFactory
from ._common import _moving_median, _moving_mad


@PulseDetectorFactory.register(["threshold_detection", "threshold", "adaptive_threshold", "noise_threshold", "hysteresis_threshold"])
class ThresholdDetectionStage(BasePulseDetectorStage):
    """
    Stage 2: Threshold Detection.
    Estimates the robust local background level N[n] and noise dispersion sigma_N[n] using a moving median and MAD:
      N[n] = median{P_env[n - W/2 : n + W/2]}
      MAD[n] = median{|P_env[n] - N[n]|}
      sigma_N[n] = 1.4826 * MAD[n]
    Generates dual adaptive hysteresis thresholds:
      T_H[n] = N[n] + K_H * sigma_N[n]   (Starts pulse)
      T_L[n] = N[n] + K_L * sigma_N[n]   (Ends pulse)
    where K_H > K_L >= 0.
    MATLAB Reference:
      noiseFloor = movmedian(Penv, W);
      madNoise = movmedian(abs(Penv - noiseFloor), W);
      sigmaN = 1.4826 * madNoise;
      TH = noiseFloor + KH * sigmaN;
      TL = noiseFloor + KL * sigmaN;
    """

    def __init__(self, window_len: int = 101, k_high: float = 5.0, k_low: float = 3.0, **kwargs: Any) -> None:
        cfg = dict(kwargs)
        cfg["window_len"] = window_len
        cfg["k_high"] = k_high
        cfg["k_low"] = k_low
        super().__init__(**cfg)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "threshold_detection"

    def compute(self, data: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Computes dual adaptive hysteresis thresholds from a real power envelope array.

        Parameters:
            data: 1D real power envelope array P_env.
            window_len: Sliding window length for moving median/MAD (default: 101).
            k_high: High threshold multiplier (default: 5.0).
            k_low: Low threshold multiplier (default: 3.0).

        Returns:
            Dictionary with:
              - 'th_high': High threshold array T_H[n]
              - 'th_low': Low threshold array T_L[n]
              - 'noise_floor': Estimated background noise floor N[n]
              - 'sigma_n': Estimated robust noise standard deviation sigma_N[n]
              - 'mad': Median Absolute Deviation array MAD[n]
              - 'k_high': KH factor
              - 'k_low': KL factor
        """
        params = {**self._config, **kwargs}

        # Reject complex IQ - require real envelope
        arr = np.asarray(data)
        if np.iscomplexobj(arr):
            raise TypeError(
                "ThresholdDetectionStage expects a real power envelope, "
                "not complex IQ. Run EnvelopeExtractionStage first."
            )
        envelope = arr.astype(np.float64).ravel()

        if envelope.size == 0:
            empty = np.array([], dtype=np.float64)
            return {"th_high": empty, "th_low": empty, "noise_floor": empty, "sigma_n": empty, "mad": empty}

        # Strict parameter validation (Bug 8)
        raw_w = int(params.get("window_len", 101))
        if raw_w < 1:
            raise ValueError(f"threshold_window must be >= 1, got {raw_w}")

        k_high = float(params.get("k_high", 5.0))
        k_low = float(params.get("k_low", 3.0))
        if not np.isfinite(k_high) or not np.isfinite(k_low):
            raise ValueError("Threshold multipliers must be finite")
        if k_low < 0:
            raise ValueError(f"k_low must be >= 0, got {k_low}")
        if k_high <= k_low:
            raise ValueError(f"k_high ({k_high}) must be greater than k_low ({k_low})")

        w = min(raw_w, envelope.size)
        if w % 2 == 0:
            w -= 1
        w = max(1, w)

        # 1. Background Noise Floor N[n] = movmedian(Penv, W)
        noise_floor = _moving_median(envelope, w)

        # 2. Robust Noise Variation sigma_N[n] = 1.4826 * movmedian(|Penv - N|, W)
        mad, sigma_n = _moving_mad(envelope, w, noise_floor=noise_floor)

        # 3. Dual Adaptive Thresholds: T_H = N + K_H * sigma_N, T_L = N + K_L * sigma_N
        th_high = noise_floor + k_high * sigma_n
        th_low = noise_floor + k_low * sigma_n

        return {
            "th_high": th_high,
            "th_low": th_low,
            "noise_floor": noise_floor,
            "sigma_n": sigma_n,
            "mad": mad,
            "k_high": k_high,
            "k_low": k_low,
            "window_len": w,
            "mean_noise_floor": float(np.mean(noise_floor)),
            "mean_sigma_n": float(np.mean(sigma_n)),
        }


def threshold_detection(data: Any, window_len: int = 101, k_high: float = 5.0, k_low: float = 3.0, **kwargs: Any) -> dict[str, Any]:
    """Functional wrapper for ThresholdDetectionStage."""
    stage = ThresholdDetectionStage(window_len=window_len, k_high=k_high, k_low=k_low, **kwargs)
    return stage.compute(data)


def main() -> None:
    print("=" * 60)
    print("  ThresholdDetectionStage: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    env = np.zeros(1024, dtype=np.float64)
    env[200:400] = 1.0
    stage = ThresholdDetectionStage()
    result = stage.compute(env, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
