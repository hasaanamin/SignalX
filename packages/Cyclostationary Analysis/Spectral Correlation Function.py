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
from .base import BaseCyclostationaryMetric, CyclostationaryMetricFactory
from ._common import _as_iq_array, _extract_sample_rate, _compute_fast_scf


@CyclostationaryMetricFactory.register([
    "spectral_correlation_function",
    "scf",
    "cyclic_spectral_density",
    "csd",
    "fam",
    "cyclic_spectrum"
])
class SpectralCorrelationFunctionMetric(BaseCyclostationaryMetric):
    """
    2D Spectral Correlation Function (SCF) / Cyclic Spectral Density S_x(f, alpha).
    Measures cross-spectral density across bifrequency plane (f, alpha):
      S_x^alpha(f) = E{ X(f + alpha/2) X*(f - alpha/2) }
    
    Provides 2D surface exploration and visual heatmap rendering.
    Independent resolution control:
      nfft_f: ordinary spectral-frequency resolution Delta f = Fs / nfft_f
      nfft_alpha: cyclic-frequency resolution Delta alpha = Fs / nfft_alpha
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "spectral_correlation_function"

    def compute(self, data: Any, **kwargs: Any) -> np.ndarray | dict[str, Any]:
        """
        Computes the complete 2D Spectral Correlation Function surface S_x(alpha, f).

        Parameters:
            data: Complex IQ array x[n].
            sample_rate (Fs): Sampling frequency in Hz (default: 1.0).
            nfft / nfft_f: FFT length for spectral frequency resolution (default: 128).
            nfft_alpha: FFT length for cyclic frequency resolution (default: 128).
            overlap: Segment overlap fraction (default: 0.5).
            window: Window type ('hann', 'hamming', 'blackman', 'rect').
            as_dict: If True, returns full dictionary with axes and 1D projections.

        Returns:
            2D complex array of shape [num_alpha, nfft_f] (or dictionary if as_dict=True).
        """
        params = {**self._config, **kwargs}
        z = _as_iq_array(data)
        fs = _extract_sample_rate(params, default=1.0)
        nfft_f = int(params.get("nfft_f", params.get("nfft", params.get("N_FFT", 128))))
        nfft_alpha = int(params.get("nfft_alpha", params.get("nfft", params.get("N_FFT", 128))))
        overlap = float(params.get("overlap", 0.5))
        win_type = str(params.get("window", "hann"))

        if z.size == 0:
            empty2d = np.array([[]], dtype=np.complex128)
            return {"scf_matrix": empty2d, "frequencies": np.array([]), "alpha_grid": np.array([])} if params.get("as_dict") else empty2d

        # Compute 2D SCF matrix [num_alphas, nfft_f]
        scf_matrix, freq_axis, alpha_axis = _compute_fast_scf(
            z, fs=fs, nfft_f=nfft_f, nfft_alpha=nfft_alpha, overlap=overlap, window=win_type
        )

        if params.get("as_dict", False):
            mag_matrix = np.abs(scf_matrix)
            power_db_matrix = 10.0 * np.log10(np.maximum(1e-18, mag_matrix))

            # 1D Projections
            # Maximum Spectral Profile (MSP) across frequency: P(alpha) = max_f |S_x^alpha(f)|
            msp_alpha = np.max(mag_matrix, axis=1)

            # Cyclic Energy Profile (CEP): E(alpha) = sum_f |S_x^alpha(f)|^2
            cep_alpha = np.sum(mag_matrix ** 2, axis=1)

            # Conventional PSD slice at alpha = 0
            zero_alpha_idx = len(alpha_axis) // 2
            psd_slice = np.real(scf_matrix[zero_alpha_idx, :])

            # Global Peak coordinates
            max_idx = np.unravel_index(np.argmax(mag_matrix), mag_matrix.shape)
            peak_alpha = float(alpha_axis[max_idx[0]])
            peak_freq = float(freq_axis[max_idx[1]])
            peak_val = float(mag_matrix[max_idx])

            return {
                "scf_matrix": scf_matrix,
                "magnitude_matrix": mag_matrix,
                "power_db_matrix": power_db_matrix,
                "frequencies": freq_axis,
                "alpha_grid": alpha_axis,
                "max_spectral_profile": msp_alpha,
                "cyclic_energy_profile": cep_alpha,
                "psd_slice": psd_slice,
                "peak_alpha_hz": peak_alpha,
                "peak_frequency_hz": peak_freq,
                "peak_magnitude": peak_val,
                "sample_rate_hz": fs,
                "nfft_f": nfft_f,
                "nfft_alpha": nfft_alpha,
                "nfft": nfft_f,
                "num_alpha_bins": len(alpha_axis),
            }

        return scf_matrix


def spectral_correlation_function(
    data: Any,
    sample_rate: float = 1.0,
    nfft: int = 128,
    overlap: float = 0.5,
    as_dict: bool = False,
    **kwargs: Any
) -> np.ndarray | dict[str, Any]:
    """Functional wrapper for SpectralCorrelationFunctionMetric."""
    metric = SpectralCorrelationFunctionMetric(sample_rate=sample_rate, nfft=nfft, overlap=overlap, as_dict=as_dict, **kwargs)
    return metric.compute(data)


def main() -> None:
    print("=" * 60)
    print("  SpectralCorrelationFunctionMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = SpectralCorrelationFunctionMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
