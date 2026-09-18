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


@PulseDetectionFactory.register(["envelope_extraction", "envelope", "extract_envelope"])
class EnvelopeExtractionMetric(BasePulseDetectionMetric):
    """
    Stage 1: Envelope Extraction & Causal Temporal Energy Integration.
    Extracts instantaneous amplitude envelope |z|, power envelope |z|^2, and computes
    a causal short-time integrated power detection statistic for low-SNR pulse triggering.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._history_power: list[float] = []

    def __del__(self) -> None:
        if hasattr(self, "_history_power"):
            self._history_power.clear()
        super().__del__()

    def reset(self) -> None:
        """Resets streaming history buffer."""
        self._history_power.clear()

    @property
    def name(self) -> str:
        return "envelope_extraction"

    def compute(
        self,
        signal: Any,
        integration_window_samples: int = 4,
        smoothing_window: int | None = None,
        use_energy: bool = True,
        return_dict: bool = True,
        **kwargs: Any
    ) -> dict[str, np.ndarray] | np.ndarray:
        """
        Extracts amplitude envelope, power envelope, and causal integrated detection statistic.

        Parameters:
            signal: Complex baseband IQ array (e.g. complex64, complex128).
            integration_window_samples: Length of causal short-time power integration window (M >= 1).
            smoothing_window: Backward-compatible alias for integration_window_samples.
            use_energy: Kept for API compatibility.
            return_dict: If True, returns dict with all 3 representations; otherwise returns detection_statistic.
        """
        iq = _as_signal_array(signal)
        n = iq.size

        # 1. Instantaneous Amplitude Envelope: abs(iq) = sqrt(I^2 + Q^2)
        # Directly computed without Hilbert transform on complex IQ
        amp_env = np.abs(iq).astype(np.float64)

        # 2. Instantaneous Power Envelope: abs(iq)^2 = I^2 + Q^2
        pwr_env = (amp_env ** 2).astype(np.float64)

        # 3. Causal Short-Time Temporal Energy Integration
        M = max(1, int(smoothing_window if smoothing_window is not None else integration_window_samples))

        if M == 1 or n < 1:
            det_stat = pwr_env.copy()
        else:
            # Efficient causal rolling sum / cumulative update
            # At sample n, det_stat[n] = mean(pwr_env[max(0, n-M+1) : n+1])
            cumsum = np.cumsum(pwr_env)
            det_stat = np.empty(n, dtype=np.float64)
            
            # Warmup: samples 0 to M-1
            warmup_len = min(n, M)
            det_stat[:warmup_len] = cumsum[:warmup_len] / np.arange(1, warmup_len + 1, dtype=np.float64)
            
            # Streaming window: samples M to n-1
            if n > M:
                det_stat[M:] = (cumsum[M:] - cumsum[:-M]) / float(M)

        if not return_dict:
            return det_stat

        return {
            "iq_data": iq,
            "amplitude_envelope": amp_env,
            "raw_envelope": amp_env,
            "power_envelope": pwr_env,
            "energy_envelope": pwr_env,
            "detection_statistic": det_stat,
            "smoothed_envelope": det_stat,
            "integration_window_samples": M,
        }


_instance = EnvelopeExtractionMetric()
envelope_extraction = _instance.compute
extract_envelope = _instance.compute

__all__ = ["EnvelopeExtractionMetric", "envelope_extraction", "extract_envelope"]


def main() -> None:
    print("=" * 60)
    print("  EnvelopeExtractionMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = EnvelopeExtractionMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
