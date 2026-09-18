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
from scipy.signal import find_peaks
from .base import BaseCyclostationaryMetric, CyclostationaryMetricFactory
from ._common import _as_iq_array, _extract_sample_rate, _compute_fine_alpha_profile
import importlib


@CyclostationaryMetricFactory.register([
    "cyclic_spectral_peaks",
    "cyclic_peaks",
    "alpha_peaks",
    "spectral_peaks_alpha",
    "cyclic_peak_detector"
])
class CyclicSpectralPeaksMetric(BaseCyclostationaryMetric):
    """
    Cyclic Spectral Peak Detection & Feature Extraction.
    Extracts candidate cyclic frequencies and corresponding spectral coordinates:
      (f_p, alpha_p) = argmax_{alpha != 0, f} |S_x^alpha(f)|
    
    Operates generically on arbitrary unknown RF IQ, decoupling cyclic frequency resolution
    from spectral FFT length.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "cyclic_spectral_peaks"

    def compute(self, data: Any = None, **kwargs: Any) -> list[dict[str, Any]] | dict[str, Any]:
        """
        Detects significant cyclic peaks from complex IQ, fine alpha profile, or precomputed SCF matrix.

        Parameters:
            data: Complex IQ array x[n] OR precomputed 2D SCF matrix S_x(alpha, f) or dict.
            sample_rate (Fs): Sampling frequency in Hz (default: 1.0).
            num_peaks: Maximum number of dominant peaks to return (default: 5).
            nfft_f: Spectral frequency FFT length (default: 128).
            nfft_alpha: Cyclic frequency FFT length (default: next power of two).
            min_alpha_hz: Minimum cyclic frequency |alpha| > min_alpha to exclude DC.
            snr_threshold_db: Threshold in dB for cyclostationary detection (default: 3.0).
            as_dict: If True, returns full dictionary with profiles and ranked candidate list.

        Returns:
            List of detected candidate dictionaries (or comprehensive dict if as_dict=True).
        """
        params = {**self._config, **kwargs}
        fs = _extract_sample_rate(params, default=1.0)
        num_peaks = int(params.get("num_peaks", 5))
        snr_threshold_db = float(params.get("snr_threshold_db", 3.0))

        # Check if data is a raw IQ array or precomputed dictionary / SCF matrix
        is_raw_iq = False
        z = None
        if isinstance(data, dict):
            if "iq" in data and data["iq"] is not None:
                z = _as_iq_array(data["iq"])
                is_raw_iq = True
            elif "scf_matrix" in data:
                scf_matrix = data["scf_matrix"]
                freq_axis = data.get("frequencies")
                alpha_axis = data.get("alpha_grid")
            else:
                try:
                    z = _as_iq_array(data)
                    is_raw_iq = True
                except Exception:
                    scf_matrix = None
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            try:
                z = _as_iq_array(data)
                is_raw_iq = True
            except Exception:
                z = None
        elif data is not None and not (isinstance(data, np.ndarray) and data.ndim == 2):
            try:
                z = _as_iq_array(data)
                is_raw_iq = True
            except Exception:
                z = None

        if is_raw_iq and z is not None:
            # Authoritative fine cyclic-frequency estimator
            nfft_alpha = params.get("nfft_alpha", None)
            fine_res = _compute_fine_alpha_profile(
                z,
                fs=fs,
                nfft_alpha=nfft_alpha,
                num_candidates=num_peaks,
                snr_threshold_db=snr_threshold_db,
                min_alpha_hz=params.get("min_alpha_hz", None),
            )

            candidates = fine_res["candidates"]
            cyclostationary_detected = fine_res["cyclostationary_detected"]

            # For top candidates, evaluate spectral frequency f_p where |S(f, alpha)| is maximized
            nfft_f = int(params.get("nfft_f", params.get("nfft", 128)))
            _sc_mod = importlib.import_module(".Spectral Correlation", package="packages.Cyclostationary Analysis")

            peaks_list: list[dict[str, Any]] = []
            for cand in candidates:
                alpha_val = cand["signed_alpha_hz"]
                sc_res = _sc_mod.SpectralCorrelationMetric(
                    alpha=alpha_val, sample_rate=fs, nfft=nfft_f, as_dict=True
                ).compute(z)
                f_p = float(sc_res.get("peak_frequency_hz", 0.0))
                pwr_db = float(sc_res.get("peak_power_db", 0.0))

                peaks_list.append({
                    "rank": cand["rank"],
                    "signed_alpha_hz": alpha_val,
                    "absolute_cyclic_rate_hz": cand["absolute_cyclic_rate_hz"],
                    "alpha_hz": alpha_val,
                    "frequency_hz": f_p,
                    "magnitude": cand["score"],
                    "power_db": pwr_db,
                    "cyclic_snr_db": cand["cyclic_snr_db"],
                    "estimated_cyclic_rate_hz": cand["absolute_cyclic_rate_hz"],
                    "estimated_symbol_rate_hz": cand["absolute_cyclic_rate_hz"],  # deprecated alias
                })

            raw_top_alpha = fine_res["raw_primary_alpha_hz"]
            raw_top_rate = fine_res["raw_primary_rate_hz"]
            raw_top_snr = fine_res["raw_cyclic_snr_db"]
            primary_freq = peaks_list[0]["frequency_hz"] if (peaks_list and cyclostationary_detected) else None

            if params.get("as_dict", False):
                return {
                    "peaks": peaks_list,
                    "num_detected_peaks": len(peaks_list),
                    "cyclostationary_detected": cyclostationary_detected,
                    "primary_signed_alpha_hz": fine_res["primary_signed_alpha_hz"],
                    "primary_absolute_rate_hz": fine_res["primary_absolute_rate_hz"],
                    "estimated_cyclic_rate_hz": fine_res["estimated_cyclic_rate_hz"],
                    "primary_alpha_hz": raw_top_alpha,
                    "primary_frequency_hz": primary_freq,
                    "primary_cyclic_snr_db": raw_top_snr,
                    "raw_primary_alpha_hz": raw_top_alpha,
                    "raw_primary_rate_hz": raw_top_rate,
                    "raw_cyclic_snr_db": raw_top_snr,
                    "max_spectral_profile": fine_res["alpha_profile"],
                    "alpha_grid": fine_res["alpha_axis_hz"],
                    "fine_alpha_profile": fine_res["alpha_profile"],
                    "noise_floor_alpha": fine_res["noise_floor"],
                    "alpha_bin_spacing_hz": fine_res["alpha_bin_spacing_hz"],
                    "alpha_observation_resolution_hz": fine_res["alpha_observation_resolution_hz"],
                    "sample_rate_hz": fs,
                }
            return peaks_list

        # Fallback for 2D SCF matrix input
        scf_matrix = None
        if isinstance(data, dict) and "scf_matrix" in data:
            scf_matrix = data["scf_matrix"]
            freq_axis = data.get("frequencies")
            alpha_axis = data.get("alpha_grid")
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            scf_matrix = data
            freq_axis = np.fft.fftshift(np.fft.fftfreq(scf_matrix.shape[1], d=1.0 / fs))
            alpha_axis = np.linspace(-fs, fs, scf_matrix.shape[0])

        if scf_matrix is None or scf_matrix.size == 0:
            return {"peaks": [], "max_spectral_profile": np.array([])} if params.get("as_dict") else []

        n_alphas, nfft = scf_matrix.shape
        if alpha_axis is None or len(alpha_axis) != n_alphas:
            alpha_axis = np.linspace(-fs, fs, n_alphas)
        if freq_axis is None or len(freq_axis) != nfft:
            freq_axis = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))

        mag_matrix = np.abs(scf_matrix)
        p_alpha = np.max(mag_matrix, axis=1)
        e_alpha = np.sum(mag_matrix ** 2, axis=1)

        min_alpha_hz = float(params.get("min_alpha_hz", 0.015 * fs))
        valid_mask = np.abs(alpha_axis) >= min_alpha_hz
        p_search = p_alpha.copy()
        p_search[~valid_mask] = 0.0

        noise_floor_alpha = float(np.median(p_alpha[valid_mask])) if np.any(valid_mask) else 1e-12
        noise_floor_alpha = max(1e-18, noise_floor_alpha)

        peak_indices, _ = find_peaks(
            p_search,
            height=noise_floor_alpha * 1.5,
            distance=max(2, n_alphas // 64)
        )

        if len(peak_indices) > 0:
            sorted_order = np.argsort(p_search[peak_indices])[::-1]
            top_indices = peak_indices[sorted_order[:num_peaks]]
        else:
            top_indices = np.array([], dtype=int)

        peaks_list: list[dict[str, Any]] = []
        for rank, p_idx in enumerate(top_indices, start=1):
            alpha_val = float(alpha_axis[p_idx])
            alpha_mag = float(p_alpha[p_idx])
            alpha_db = float(10.0 * np.log10(max(1e-18, alpha_mag)))

            f_slice = mag_matrix[p_idx, :]
            f_best_idx = int(np.argmax(f_slice))
            freq_val = float(freq_axis[f_best_idx])

            cyclic_snr_db = float(10.0 * np.log10(max(1e-12, alpha_mag / noise_floor_alpha)))

            peaks_list.append({
                "rank": rank,
                "signed_alpha_hz": alpha_val,
                "absolute_cyclic_rate_hz": abs(alpha_val),
                "alpha_hz": alpha_val,
                "frequency_hz": freq_val,
                "magnitude": alpha_mag,
                "power_db": alpha_db,
                "cyclic_snr_db": cyclic_snr_db,
                "estimated_cyclic_rate_hz": abs(alpha_val),
                "estimated_symbol_rate_hz": abs(alpha_val),
            })

        top_alpha = peaks_list[0]["alpha_hz"] if peaks_list else 0.0
        top_freq = peaks_list[0]["frequency_hz"] if peaks_list else 0.0
        top_snr = peaks_list[0]["cyclic_snr_db"] if peaks_list else 0.0
        detected = bool(top_snr >= snr_threshold_db and abs(top_alpha) > 0)

        if params.get("as_dict", False):
            return {
                "peaks": peaks_list,
                "num_detected_peaks": len(peaks_list),
                "cyclostationary_detected": detected,
                "primary_signed_alpha_hz": top_alpha if detected else None,
                "primary_absolute_rate_hz": abs(top_alpha) if detected else None,
                "estimated_cyclic_rate_hz": abs(top_alpha) if detected else None,
                "primary_alpha_hz": top_alpha,
                "primary_frequency_hz": top_freq if detected else None,
                "primary_cyclic_snr_db": top_snr,
                "raw_primary_alpha_hz": top_alpha,
                "raw_primary_rate_hz": abs(top_alpha),
                "raw_cyclic_snr_db": top_snr,
                "max_spectral_profile": p_alpha,
                "energy_profile": e_alpha,
                "alpha_grid": alpha_axis,
                "frequencies": freq_axis,
                "noise_floor_alpha": noise_floor_alpha,
                "sample_rate_hz": fs,
            }

        return peaks_list


def cyclic_spectral_peaks(
    data: Any = None,
    sample_rate: float = 1.0,
    num_peaks: int = 5,
    as_dict: bool = False,
    **kwargs: Any
) -> list[dict[str, Any]] | dict[str, Any]:
    """Functional wrapper for CyclicSpectralPeaksMetric."""
    metric = CyclicSpectralPeaksMetric(sample_rate=sample_rate, num_peaks=num_peaks, as_dict=as_dict, **kwargs)
    return metric.compute(data)


def main() -> None:
    print("=" * 60)
    print("  CyclicSpectralPeaksMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t)
    metric = CyclicSpectralPeaksMetric()
    result = metric.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
