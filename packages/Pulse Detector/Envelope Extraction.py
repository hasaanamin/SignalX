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
from ._common import _as_iq_array, _causal_moving_average


@PulseDetectorFactory.register(["envelope_extraction", "envelope", "extract_envelope", "power_envelope"])
class EnvelopeExtractionStage(BasePulseDetectorStage):
    """
    Stage 1: Envelope Extraction.
    Extracts the instantaneous power envelope from conditioned complex IQ data:
      P[n] = |x[n]|^2 = I[n]^2 + Q[n]^2
    Optionally applies a causal moving average smoothing window of length L:
      bar{P}[n] = (1/L) sum_{k=0}^{L-1} P[n-k]
    Maintains raw_power for high-resolution edge refinement and causal detection_envelope for thresholding.
    MATLAB Reference: `P = abs(x).^2; Penv = movmean(P, L);`
    """

    def __init__(self, mode: str = "power", smooth_len: int = 1, **kwargs: Any) -> None:
        cfg = dict(kwargs)
        cfg["mode"] = mode
        cfg["smooth_len"] = smooth_len
        super().__init__(**cfg)
        self._history: np.ndarray | None = None

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "envelope_extraction"

    def reset(self) -> None:
        """Clears causal moving-average history."""
        self._history = None

    def compute(self, data: Any, **kwargs: Any) -> np.ndarray | dict[str, Any]:
        """
        Extracts power envelope from complex IQ data.

        Parameters:
            data: Complex IQ array or dictionary/tuple.
            mode: Must be 'power' (|x|^2).
            smooth_len (L): Window length for causal moving average smoothing (default: 1).
            as_dict: If True, returns full diagnostic dictionary.

        Returns:
            1D array of smoothed detection envelope values (or dict if as_dict=True).
        """
        params = {**self._config, **kwargs}
        z = _as_iq_array(data)

        if z.size == 0:
            empty = np.array([], dtype=np.float64)
            return {"envelope": empty, "raw_power": empty, "detection_envelope": empty} if params.get("as_dict") else empty

        mode = str(params.get("mode", "power")).strip().lower()
        if mode not in {"power"}:
            raise ValueError(f"Unsupported envelope mode: '{mode}'. Pulse detector currently requires envelope_mode='power'.")

        smooth_len = int(params.get("smooth_len", 1))
        if smooth_len < 1:
            raise ValueError(f"smooth_len must be >= 1, got {smooth_len}")

        # 1. Raw Power Envelope: P[n] = |x[n]|^2
        raw_power = np.abs(z) ** 2

        # 2. Causal Moving Mean Smoothing for Detection
        if smooth_len > 1:
            detection_envelope, self._history = _causal_moving_average(raw_power, smooth_len, history=self._history)
        else:
            detection_envelope = raw_power.copy()

        if params.get("as_dict", False):
            return {
                "envelope": detection_envelope,
                "detection_envelope": detection_envelope,
                "raw_power": raw_power,
                "power_raw": raw_power,
                "mode": mode,
                "smooth_len": smooth_len,
                "num_samples": len(z),
                "peak_envelope": float(np.max(detection_envelope)) if detection_envelope.size > 0 else 0.0,
                "mean_envelope": float(np.mean(detection_envelope)) if detection_envelope.size > 0 else 0.0,
            }

        return detection_envelope


def envelope_extraction(data: Any, mode: str = "power", smooth_len: int = 1, as_dict: bool = False, **kwargs: Any) -> np.ndarray | dict[str, Any]:
    """Functional wrapper for EnvelopeExtractionStage."""
    stage = EnvelopeExtractionStage(mode=mode, smooth_len=smooth_len, as_dict=as_dict, **kwargs)
    return stage.compute(data)


def main() -> None:
    print("=" * 60)
    print("  EnvelopeExtractionStage: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = EnvelopeExtractionStage()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
