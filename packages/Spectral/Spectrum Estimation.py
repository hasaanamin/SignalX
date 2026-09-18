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
from .base import BaseSpectralMetric, SpectralMetricFactory
from ._common import _as_iq_array, _extract_sample_rate, _compute_welch_psd


@SpectralMetricFactory.register(["spectrum_estimation", "spectral_estimation", "high_res_spectrum", "psd_estimator", "pmusic", "pburg"])
class SpectrumEstimationMetric(BaseSpectralMetric):
    """
    High-Resolution Non-Parametric and Parametric Spectrum Estimation.
    MATLAB Reference: `pwelch`, `pburg`, `pyulear`, `pmusic`.
    Supported methods:
      - 'welch': Averaged modified periodograms (default).
      - 'burg': Autoregressive (AR) maximum entropy spectral estimator via Burg lattice recursion.
      - 'music': Multiple Signal Classification super-resolution subspace pseudo-spectrum.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "spectrum_estimation"

    def _burg_ar_estimation(self, z: np.ndarray, order: int, nfft: int, fs: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Burg Autoregressive (AR) Maximum Entropy Spectrum Estimation.
        """
        n = len(z)
        order = min(order, n - 1)
        ef = np.array(z, dtype=np.complex128)
        eb = np.array(z, dtype=np.complex128)
        a = np.zeros(order + 1, dtype=np.complex128)
        a[0] = 1.0

        p = float(np.mean(np.abs(z) ** 2))

        for k in range(1, order + 1):
            ef_k = ef[k:]
            eb_k = eb[:-k]
            num = -2.0 * np.sum(ef_k * np.conj(eb_k))
            den = np.sum(np.abs(ef_k) ** 2) + np.sum(np.abs(eb_k) ** 2)
            rc = num / max(1e-18, den)  # Reflection coefficient

            # Levinson recursion
            a_prev = a[:k].copy()
            a[k] = rc
            a[1:k] = a_prev[1:] + rc * np.conj(a_prev[1:][::-1])

            # Update forward/backward errors
            ef_new = ef_k + rc * eb_k
            eb_new = eb_k + np.conj(rc) * ef_k
            ef[k:] = ef_new
            eb[:-k] = eb_new
            p *= (1.0 - np.abs(rc) ** 2)

        # Evaluate all-pole frequency response
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
        w = 2.0 * np.pi * freqs / fs
        k_vec = np.arange(order + 1)
        # H(w) = 1 / sum(a[k] * exp(-j * w * k))
        denom = np.zeros(nfft, dtype=np.complex128)
        for idx in range(order + 1):
            denom += a[idx] * np.exp(-1j * w * idx)

        psd_linear = np.abs(p / np.maximum(1e-18, np.abs(denom) ** 2)) / fs
        return freqs, psd_linear

    def _music_pseudospectrum(self, z: np.ndarray, num_signals: int, nfft: int, fs: float) -> tuple[np.ndarray, np.ndarray]:
        """
        MUSIC (Multiple Signal Classification) Subspace Super-Resolution Pseudo-Spectrum.
        """
        n = len(z)
        m = min(n // 2, 64)  # Covariance matrix dimension
        num_snapshots = n - m + 1
        if num_snapshots < m or num_signals >= m:
            # Fallback to Welch
            return _compute_welch_psd(z, fs=fs, nperseg=nfft)

        # 1. Build spatio-temporal data matrix X
        X = np.zeros((m, num_snapshots), dtype=np.complex128)
        for i in range(m):
            X[i, :] = z[i : i + num_snapshots]

        # 2. Sample Covariance Matrix
        R = (X @ np.conj(X.T)) / num_snapshots

        # 3. Eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(R)
        # Sort ascending
        idx = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx]

        # Noise subspace eigenvectors En (smallest m - num_signals)
        En = eigvecs[:, : m - num_signals]
        EnEnH = En @ np.conj(En.T)

        # 4. Evaluate MUSIC steering vectors across frequency grid
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
        w = 2.0 * np.pi * freqs / fs
        m_vec = np.arange(m)

        pseudospectrum = np.zeros(nfft, dtype=np.float64)
        for i, wi in enumerate(w):
            a = np.exp(-1j * wi * m_vec)
            denom = np.real(np.conj(a).T @ EnEnH @ a)
            pseudospectrum[i] = 1.0 / max(1e-12, denom)

        # Normalize peak to 1.0
        pseudospectrum /= max(1e-12, np.max(pseudospectrum))
        return freqs, pseudospectrum

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        method: str = "welch",
        nfft: int = 1024,
        order: int = 32,
        num_signals: int = 2,
        as_dict: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {"frequencies": np.array([]), "spectrum": np.array([])} if as_dict else np.array([])

        method_clean = method.lower().strip()
        if method_clean in ("burg", "ar", "autoregressive", "pburg"):
            freqs, spectrum = self._burg_ar_estimation(z, order=order, nfft=nfft, fs=fs)
        elif method_clean in ("music", "pmusic", "subspace"):
            freqs, spectrum = self._music_pseudospectrum(z, num_signals=num_signals, nfft=nfft, fs=fs)
        else:  # 'welch'
            freqs, spectrum = _compute_welch_psd(z, fs=fs, nperseg=nfft)

        spectrum_db = 10.0 * np.log10(np.maximum(1e-24, spectrum))

        if as_dict:
            return {
                "frequencies": freqs,
                "spectrum_linear": spectrum,
                "spectrum_db": spectrum_db,
                "method": method_clean,
                "sample_rate": fs,
                "nfft": nfft,
            }
        return spectrum


def spectrum_estimation(
    iq_data: Any,
    sample_rate: float | None = None,
    method: str = "welch",
    nfft: int = 1024,
    order: int = 32,
    num_signals: int = 2,
    as_dict: bool = False,
    **kwargs: Any,
) -> np.ndarray | dict[str, Any]:
    """
    Functional DSP wrapper for High-Resolution Spectrum Estimation.
    """
    metric = SpectrumEstimationMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, method=method, nfft=nfft, order=order, num_signals=num_signals, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  SpectrumEstimationMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = SpectrumEstimationMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
