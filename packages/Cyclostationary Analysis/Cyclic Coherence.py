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
from ._common import _as_iq_array, _extract_sample_rate, _interpolate_psd
import importlib


@CyclostationaryMetricFactory.register([
    "cyclic_coherence",
    "spectral_coherence",
    "coherence_function",
    "scoh",
    "magnitude_squared_coherence"
])
class CyclicCoherenceMetric(BaseCyclostationaryMetric):
    """
    Cyclic Spectral Coherence Function gamma_x^alpha(f) & C_x^alpha(f).
    Normalizes the Cyclic Spectral Density by the auto-PSD of shifted frequency components:
      gamma_x^alpha(f) = S_x^alpha(f) / sqrt( S_x^0(f + alpha/2) * S_x^0(f - alpha/2) )
    
    Coherence conventions:
      Magnitude Coherence: C_x^alpha(f) = |gamma_x^alpha(f)|
      Magnitude-Squared Coherence: C_{x,2}^alpha(f) = |gamma_x^alpha(f)|^2
    
    Evaluates exact floating-point alpha shifts by linear interpolation of the zero-cycle PSD,
    without circular roll wrapping across Nyquist. Out-of-band evaluation points are marked NaN.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "cyclic_coherence"

    def compute(self, data: Any = None, **kwargs: Any) -> np.ndarray | dict[str, Any]:
        """
        Computes the Cyclic Coherence function (2D matrix or 1D slice at candidate alpha).

        Parameters:
            data: Complex IQ array x[n] OR precomputed 2D SCF matrix S_x(alpha, f) or dict.
            alpha: Optional specific candidate alpha in Hz for 1D slice coherence.
            s_alpha: Optional precomputed 1D SCF slice S_x^alpha(f).
            sample_rate (Fs): Sampling frequency in Hz (default: 1.0).
            nfft / nfft_f: FFT length / spectral frequency resolution bins (default: 128).
            coherence_type: 'magnitude_squared' (default) or 'magnitude'.
            eps: Regularizer floor for PSD denominator (default: 1e-12).
            as_dict: If True, returns full dictionary with profiles and peak metrics.

        Returns:
            2D or 1D float64 array representing C_x^alpha(f) in [0, 1]
            (or comprehensive dictionary if as_dict=True).
        """
        params = {**self._config, **kwargs}
        fs = _extract_sample_rate(params, default=1.0)
        eps = float(params.get("eps", 1e-12))
        coherence_type = str(params.get("coherence_type", "magnitude_squared")).lower()
        if coherence_type not in ("magnitude", "magnitude_squared"):
            coherence_type = "magnitude_squared"

        scf_matrix = None
        freq_axis = None
        alpha_axis = None
        single_alpha = params.get("alpha", None)
        s_slice = params.get("s_alpha", kwargs.get("s_alpha", None))

        if isinstance(data, dict) and "scf_matrix" in data:
            scf_matrix = data["scf_matrix"]
            freq_axis = data.get("frequencies")
            alpha_axis = data.get("alpha_grid")
        elif "scf_matrix" in params:
            scf_matrix = params["scf_matrix"]
            freq_axis = params.get("frequencies")
            alpha_axis = params.get("alpha_grid")
        elif data is not None:
            # Compute SCF from raw IQ
            try:
                z = _as_iq_array(data)
                _scf_mod = importlib.import_module(".Spectral Correlation Function", package="packages.Cyclostationary Analysis")
                nfft_f = int(params.get("nfft_f", params.get("nfft", 128)))
                nfft_alpha = int(params.get("nfft_alpha", nfft_f))
                scf_res = _scf_mod.SpectralCorrelationFunctionMetric(
                    sample_rate=fs, nfft_f=nfft_f, nfft_alpha=nfft_alpha, as_dict=True
                ).compute(z)
                scf_matrix = scf_res["scf_matrix"]
                freq_axis = scf_res["frequencies"]
                alpha_axis = scf_res["alpha_grid"]
            except Exception:
                pass

        if scf_matrix is None or scf_matrix.size == 0:
            empty2d = np.array([[]], dtype=np.float64)
            return {"coherence_matrix": empty2d, "max_coherence_profile": np.array([])} if params.get("as_dict") else empty2d

        num_alphas, nfft = scf_matrix.shape
        if alpha_axis is None or len(alpha_axis) != num_alphas:
            alpha_axis = np.linspace(-fs, fs, num_alphas)
        if freq_axis is None or len(freq_axis) != nfft:
            freq_axis = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))

        # 1. Extract Conventional Auto-PSD: S_x^0(f)
        if "psd_0" in params and params["psd_0"] is not None:
            psd_0 = np.maximum(eps, np.asarray(params["psd_0"], dtype=np.float64))
        elif "psd" in params and params["psd"] is not None:
            psd_0 = np.maximum(eps, np.asarray(params["psd"], dtype=np.float64))
        elif isinstance(data, dict) and "psd_slice" in data and data["psd_slice"] is not None:
            psd_0 = np.maximum(eps, np.asarray(data["psd_slice"], dtype=np.float64))
        else:
            zero_alpha_idx = int(np.argmin(np.abs(alpha_axis)))
            psd_0 = np.maximum(eps, np.real(scf_matrix[zero_alpha_idx, :]))

        # 2. Compute Cyclic Coherence using interpolated S_x^0(f +- alpha/2)
        gamma_matrix = np.full((num_alphas, nfft), np.nan, dtype=np.complex128)
        c_matrix = np.full((num_alphas, nfft), np.nan, dtype=np.float64)

        for a_idx, alpha_val in enumerate(alpha_axis):
            f_plus = freq_axis + (alpha_val / 2.0)
            f_minus = freq_axis - (alpha_val / 2.0)

            p_plus = _interpolate_psd(freq_axis, psd_0, f_plus)
            p_minus = _interpolate_psd(freq_axis, psd_0, f_minus)

            valid_pts = np.isfinite(p_plus) & np.isfinite(p_minus) & (p_plus > 0) & (p_minus > 0)
            if np.any(valid_pts):
                denom = np.sqrt(np.maximum(eps, p_plus[valid_pts] * p_minus[valid_pts]))
                gamma_pts = scf_matrix[a_idx, valid_pts] / denom
                
                if coherence_type == "magnitude":
                    c_pts = np.abs(gamma_pts)
                else:
                    c_pts = np.abs(gamma_pts) ** 2

                # Clamp small numerical excursions [0.0, 1.0]
                c_pts = np.clip(c_pts, 0.0, 1.0)
                gamma_matrix[a_idx, valid_pts] = gamma_pts
                c_matrix[a_idx, valid_pts] = c_pts

        # If a single candidate alpha was requested for 1D slice coherence
        if single_alpha is not None:
            target_alpha = float(single_alpha)
            if s_slice is None:
                # If data is raw IQ or if we can compute slice
                try:
                    z = _as_iq_array(data)
                    _sc_mod = importlib.import_module(".Spectral Correlation", package="packages.Cyclostationary Analysis")
                    s_slice = _sc_mod.SpectralCorrelationMetric(
                        alpha=target_alpha, sample_rate=fs, nfft=nfft, as_dict=False
                    ).compute(z)
                except Exception:
                    # Nearest row from scf_matrix
                    a_nearest = int(np.argmin(np.abs(alpha_axis - target_alpha)))
                    s_slice = scf_matrix[a_nearest, :]

            s_slice = np.asarray(s_slice, dtype=np.complex128)
            f_plus = freq_axis + (target_alpha / 2.0)
            f_minus = freq_axis - (target_alpha / 2.0)
            p_plus = _interpolate_psd(freq_axis, psd_0, f_plus)
            p_minus = _interpolate_psd(freq_axis, psd_0, f_minus)

            slice_gamma = np.full(nfft, np.nan, dtype=np.complex128)
            slice_c = np.full(nfft, np.nan, dtype=np.float64)

            valid_pts = np.isfinite(p_plus) & np.isfinite(p_minus) & (p_plus > 0) & (p_minus > 0)
            if np.any(valid_pts):
                denom = np.sqrt(np.maximum(eps, p_plus[valid_pts] * p_minus[valid_pts]))
                gamma_pts = s_slice[valid_pts] / denom
                c_pts = np.abs(gamma_pts) if coherence_type == "magnitude" else (np.abs(gamma_pts) ** 2)
                c_pts = np.clip(c_pts, 0.0, 1.0)
                slice_gamma[valid_pts] = gamma_pts
                slice_c[valid_pts] = c_pts

            if params.get("as_dict", False):
                finite_vals = slice_c[np.isfinite(slice_c)]
                peak_coh = float(np.max(finite_vals)) if finite_vals.size > 0 else 0.0
                return {
                    "alpha_hz": target_alpha,
                    "coherence_slice": slice_c,
                    "complex_coherence_slice": slice_gamma,
                    "frequencies": freq_axis,
                    "peak_coherence": peak_coh,
                    "coherence_type": coherence_type,
                    "sample_rate_hz": fs,
                }
            return slice_c

        if params.get("as_dict", False):
            # 1D Coherence Profiles (ignoring NaNs)
            max_coh_profile = np.nanmax(c_matrix, axis=1)
            max_coh_profile = np.nan_to_num(max_coh_profile, nan=0.0)

            mean_coh_profile = np.nanmean(c_matrix, axis=1)
            mean_coh_profile = np.nan_to_num(mean_coh_profile, nan=0.0)

            # Exclude alpha = 0 to evaluate peak non-zero cyclic coherence
            min_alpha_hz = 0.02 * fs
            valid_mask = np.abs(alpha_axis) >= min_alpha_hz
            if np.any(valid_mask):
                sub_profile = max_coh_profile.copy()
                sub_profile[~valid_mask] = 0.0
                peak_idx = int(np.argmax(sub_profile))
                peak_alpha = float(alpha_axis[peak_idx])
                peak_coherence = float(sub_profile[peak_idx])
            else:
                peak_alpha = 0.0
                peak_coherence = 0.0

            return {
                "coherence_matrix": np.nan_to_num(c_matrix, nan=0.0),
                "raw_coherence_matrix": c_matrix,
                "complex_coherence_matrix": gamma_matrix,
                "max_coherence_profile": max_coh_profile,
                "mean_coherence_profile": mean_coh_profile,
                "alpha_grid": alpha_axis,
                "frequencies": freq_axis,
                "peak_cyclic_coherence": peak_coherence,
                "peak_cyclic_alpha_hz": peak_alpha,
                "coherence_type": coherence_type,
                "sample_rate_hz": fs,
            }

        return np.nan_to_num(c_matrix, nan=0.0)


def cyclic_coherence(
    data: Any = None,
    sample_rate: float = 1.0,
    nfft: int = 128,
    coherence_type: str = "magnitude_squared",
    as_dict: bool = False,
    **kwargs: Any
) -> np.ndarray | dict[str, Any]:
    """Functional wrapper for CyclicCoherenceMetric."""
    metric = CyclicCoherenceMetric(
        sample_rate=sample_rate, nfft=nfft, coherence_type=coherence_type, as_dict=as_dict, **kwargs
    )
    return metric.compute(data)


def main() -> None:
    print("=" * 60)
    print("  CyclicCoherenceMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t)
    metric = CyclicCoherenceMetric()
    result = metric.compute(sample_iq, sample_rate=fs, alpha=100e3)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
