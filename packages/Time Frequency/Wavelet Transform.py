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
from .base import BaseTimeFrequencyMetric, TimeFrequencyMetricFactory, WaveletResult
from ._common import _as_iq_array, _extract_sample_rate, _compute_cwt_core, CWTFilterBank


@TimeFrequencyMetricFactory.register(["wavelet_transform", "cwt", "continuous_wavelet_transform", "wavelet", "morse_cwt"])
class WaveletTransformStage(BaseTimeFrequencyMetric):
    """
    Continuous Wavelet Transform (CWT) Stage matching MATLAB `cwt` definition.
    Uses Generalized Morse Wavelet (gamma=3, time_bandwidth=60, voices_per_octave=10, L1 normalization).

    MATLAB Formulation:
      psi_{s, u}(t) = (1 / s) * psi((t - u) / s)  (L1 normalization)
      W_x(s, u) = integral x(t) * (1 / s) * psi^*((t - u) / s) dt
      Scales: s_j = s_0 * 2^(j / V), V = 10 voices per octave
      Frequencies: f_j = (f_0 / s_j) * F_s

    Returns:
      WaveletResult container with:
        - coefficients [S, N_samples] (complex128)
        - frequencies_hz [S]
        - scales [S]
        - coefficient_energy [S, N_samples] (|W[s, n]|^2)
        - times_sec [N_samples]
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "wavelet_transform"

    def compute(
        self,
        data: Any,
        *,
        sample_rate: float | None = None,
        wavelet_name: str = "morse",
        voices_per_octave: int = 10,
        gamma: float = 3.0,
        time_bandwidth: float = 60.0,
        frequency_limits: tuple[float, float] | None = None,
        min_frequency_hz: float | None = None,
        max_frequency_hz: float | None = None,
        scales: np.ndarray | None = None,
        normalization: str = "L1",
        frequency_mode: str = "analytic_positive",
        sample_offset: int = 0,
        as_dict: bool = False,
        **kwargs: Any,
    ) -> WaveletResult | dict[str, Any] | np.ndarray:
        """
        Executes CWT on complex IQ data.

        Parameters:
            data: Complex IQ array or existing WaveletResult.
            sample_rate (Fs): Sampling rate in Hz.
            wavelet_name: Wavelet type ('morse').
            voices_per_octave (V): Number of scales per octave (default: 10).
            gamma: Morse symmetry parameter (default: 3.0).
            time_bandwidth (P^2): Morse time-bandwidth product (default: 60.0).
            frequency_limits: Optional (f_min, f_max) in Hz.
            min_frequency_hz: Optional lower frequency bound in Hz.
            max_frequency_hz: Optional upper frequency bound in Hz.
            scales: Optional explicit scale array.
            normalization: 'L1' (default matching MATLAB cwt) or 'L2'.
            frequency_mode: 'analytic_positive' (default) or 'two_sided'.
            sample_offset: Global sample offset for streaming/slicing timestamp alignment.
            as_dict: If True, returns dictionary representation.

        Returns:
            WaveletResult container or dictionary.
        """
        if isinstance(data, WaveletResult):
            return data.to_dict() if as_dict else data

        fs = _extract_sample_rate(data=data, kwargs=kwargs, sample_rate=sample_rate, default_fs=1.0)
        res = _compute_cwt_core(
            data,
            sample_rate=fs,
            wavelet_name=wavelet_name,
            voices_per_octave=voices_per_octave,
            gamma=gamma,
            time_bandwidth=time_bandwidth,
            frequency_limits=frequency_limits,
            min_frequency_hz=min_frequency_hz,
            max_frequency_hz=max_frequency_hz,
            scales=scales,
            normalization=normalization,
            frequency_mode=frequency_mode,
            sample_offset=sample_offset,
        )
        return res.to_dict() if as_dict else res


def compute_cwt(
    iq_data: Any,
    *,
    sample_rate: float | None = None,
    wavelet_name: str = "morse",
    voices_per_octave: int = 10,
    gamma: float = 3.0,
    time_bandwidth: float = 60.0,
    frequency_limits: tuple[float, float] | None = None,
    min_frequency_hz: float | None = None,
    max_frequency_hz: float | None = None,
    scales: np.ndarray | None = None,
    normalization: str = "L1",
    frequency_mode: str = "analytic_positive",
    sample_offset: int = 0,
    as_dict: bool = False,
    **kwargs: Any,
) -> WaveletResult | dict[str, Any]:
    """
    Convenience functional API to compute Continuous Wavelet Transform on complex IQ data.
    """
    stage = WaveletTransformStage()
    return stage.compute(
        iq_data,
        sample_rate=sample_rate,
        wavelet_name=wavelet_name,
        voices_per_octave=voices_per_octave,
        gamma=gamma,
        time_bandwidth=time_bandwidth,
        frequency_limits=frequency_limits,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        scales=scales,
        normalization=normalization,
        frequency_mode=frequency_mode,
        sample_offset=sample_offset,
        as_dict=as_dict,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  WaveletTransformStage: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = WaveletTransformStage()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
