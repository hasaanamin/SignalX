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
from ._common import _extract_sample_rate, _compute_cwt_core


@TimeFrequencyMetricFactory.register(["wavelet_energy", "cwt_energy", "scalogram", "wavelet_scalogram", "wavelet_power"])
class WaveletEnergyMetric(BaseTimeFrequencyMetric):
    """
    Wavelet Energy Stage (Scalogram).
    Computes coefficient magnitude squared: E_W[s, n] = |W[s, n]|^2.

    Mathematical Formulations:
      - Wavelet Coefficient Energy Map: E_W[s, n] = |W[s, n]|^2
      - Wavelet Energy per Scale:       E_s[s] = sum_n |W[s, n]|^2
      - Wavelet Energy vs Time:         E_t[n] = sum_s |W[s, n]|^2
      - Total Coefficient Energy:       E_CWT = sum_s sum_n |W[s, n]|^2

    Note:
      Since numerical CWT uses L1 normalization and is an overcomplete/non-orthonormal continuous transform,
      these quantities measure wavelet coefficient energy distribution rather than Parseval energy conservation.

    Accepts:
      - Raw complex IQ (runs common CWT once)
      - Precomputed WaveletResult (reuses precalculated CWT coefficients with zero duplicate transform steps)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "wavelet_energy"

    def compute(
        self,
        data: Any,
        *,
        sample_rate: float | None = None,
        wavelet_name: str = "morse",
        voices_per_octave: int = 10,
        gamma: float = 3.0,
        time_bandwidth: float = 60.0,
        min_frequency_hz: float | None = None,
        max_frequency_hz: float | None = None,
        frequency_limits: tuple[float, float] | None = None,
        scales: np.ndarray | None = None,
        normalization: str = "L1",
        frequency_mode: str = "analytic_positive",
        mode: str = "map",
        as_dict: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | dict[str, Any]:
        """
        Computes wavelet coefficient energy map, energy vs scale, or energy vs time.

        Parameters:
            data: Complex IQ array or existing WaveletResult.
            sample_rate (Fs): Sampling rate in Hz.
            wavelet_name: Wavelet type ('morse').
            voices_per_octave: Voices per octave.
            gamma: Morse gamma.
            time_bandwidth: Morse time-bandwidth.
            min_frequency_hz: Optional lower frequency bound in Hz.
            max_frequency_hz: Optional upper frequency bound in Hz.
            frequency_limits: Optional tuple (f_min, f_max).
            scales: Optional pre-specified scales array.
            normalization: 'L1' or 'L2'.
            frequency_mode: 'analytic_positive' or 'two_sided'.
            mode: 'map' (default, 2D |W|^2), 'scale' (1D E_s[s]), 'time' (1D E_t[n]), or 'total'.
            as_dict: If True, returns complete dictionary with all energy profiles, scales, frequencies, and times.

        Returns:
            2D energy map, 1D energy profile, or canonical dictionary.
        """
        if isinstance(data, WaveletResult):
            wav_res = data
        elif isinstance(data, dict) and "coefficient_energy" in data:
            coeff_energy = np.asarray(data["coefficient_energy"], dtype=np.float64)
            freqs = np.asarray(data.get("frequencies_hz", np.arange(coeff_energy.shape[0])), dtype=np.float64)
            scales = np.asarray(data.get("scales", np.arange(coeff_energy.shape[0])), dtype=np.float64)
            times = np.asarray(data.get("times_sec", np.arange(coeff_energy.shape[1])), dtype=np.float64)
            e_scale = np.sum(coeff_energy, axis=1)
            e_time = np.sum(coeff_energy, axis=0)
            total_e = float(np.sum(coeff_energy))
            if as_dict:
                return {
                    "wavelet_coefficient_energy": coeff_energy,
                    "wavelet_energy_per_scale": e_scale,
                    "wavelet_energy_vs_time": e_time,
                    "total_wavelet_energy": total_e,
                    "frequencies_hz": freqs,
                    "scales": scales,
                    "times_sec": times,
                    "sample_rate": data.get("sample_rate", 1.0),
                }
            mode_clean = str(mode).lower().strip()
            if mode_clean in ("scale", "energy_scale", "freq"):
                return e_scale
            elif mode_clean in ("time", "energy_time", "temporal"):
                return e_time
            elif mode_clean in ("total", "sum"):
                return np.array([total_e])
            return coeff_energy
        else:
            fs = _extract_sample_rate(data=data, kwargs=kwargs, sample_rate=sample_rate, default_fs=1.0)
            wav_res = _compute_cwt_core(
                data,
                sample_rate=fs,
                wavelet_name=wavelet_name,
                voices_per_octave=voices_per_octave,
                gamma=gamma,
                time_bandwidth=time_bandwidth,
                min_frequency_hz=min_frequency_hz,
                max_frequency_hz=max_frequency_hz,
                frequency_limits=frequency_limits,
                scales=scales,
                normalization=normalization,
                frequency_mode=frequency_mode,
            )

        coeff_energy = wav_res.coefficient_energy
        e_scale = wav_res.energy_per_scale
        e_time = wav_res.energy_vs_time
        total_e = wav_res.total_energy

        if as_dict:
            return {
                "wavelet_coefficient_energy": coeff_energy,
                "wavelet_energy_per_scale": e_scale,
                "wavelet_energy_vs_time": e_time,
                "total_wavelet_energy": total_e,
                "frequencies_hz": wav_res.frequencies_hz,
                "scales": wav_res.scales,
                "times_sec": wav_res.times_sec,
                "sample_rate": wav_res.sample_rate,
                "wavelet_name": wav_res.wavelet_name,
                "voices_per_octave": wav_res.voices_per_octave,
            }

        mode_clean = str(mode).lower().strip()
        if mode_clean in ("scale", "energy_scale", "freq"):
            return e_scale
        elif mode_clean in ("time", "energy_time", "temporal"):
            return e_time
        elif mode_clean in ("total", "sum"):
            return np.array([total_e])

        return coeff_energy


def compute_wavelet_energy(
    data: Any,
    *,
    sample_rate: float | None = None,
    wavelet_name: str = "morse",
    voices_per_octave: int = 10,
    gamma: float = 3.0,
    time_bandwidth: float = 60.0,
    min_frequency_hz: float | None = None,
    max_frequency_hz: float | None = None,
    frequency_limits: tuple[float, float] | None = None,
    scales: np.ndarray | None = None,
    normalization: str = "L1",
    frequency_mode: str = "analytic_positive",
    mode: str = "map",
    as_dict: bool = False,
    **kwargs: Any,
) -> np.ndarray | dict[str, Any]:
    """
    Convenience functional API for wavelet coefficient energy computation.
    """
    metric = WaveletEnergyMetric()
    return metric.compute(
        data,
        sample_rate=sample_rate,
        wavelet_name=wavelet_name,
        voices_per_octave=voices_per_octave,
        gamma=gamma,
        time_bandwidth=time_bandwidth,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        frequency_limits=frequency_limits,
        scales=scales,
        normalization=normalization,
        frequency_mode=frequency_mode,
        mode=mode,
        as_dict=as_dict,
        **kwargs,
    )


def main() -> None:
    print("=" * 60)
    print("  WaveletEnergyMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = WaveletEnergyMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
