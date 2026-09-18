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
from ._common import _as_iq_array, _extract_sample_rate, _get_window
import importlib


@CyclostationaryMetricFactory.register([
    "spectral_correlation",
    "spectral_correlation_slice",
    "sc_slice",
    "cyclic_spectrum_slice"
])
class SpectralCorrelationMetric(BaseCyclostationaryMetric):
    """
    Spectral Correlation Slice S_x^alpha(f).
    Evaluates the Fourier transform of the cyclic autocorrelation function over lag domain tau:
      S_x^alpha(f) = int_{-infty}^infty R_x^alpha(tau) e^{-j 2pi f tau} dtau
    Discrete Fourier form:
      S_x^alpha[k] = sum_{ell = -L}^L R_x^alpha[ell] w[ell] e^{-j 2pi k ell / N_FFT}
    Evaluated at the exact floating-point candidate cyclic frequency alpha.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "spectral_correlation"

    def compute(self, data: Any, **kwargs: Any) -> np.ndarray | dict[str, Any]:
        """
        Computes the spectral correlation spectrum S_x^alpha(f) at a specific cyclic frequency alpha.

        Parameters:
            data: Complex IQ array x[n] OR precomputed 1D cyclic autocorrelation array R_x^alpha[ell].
            alpha: Cyclic frequency in Hz (floating-point, default: 0.0).
            max_lag (L): Maximum lag in samples (default: 64).
            nfft / nfft_f: FFT length / spectral frequency bins (default: 256).
            sample_rate (Fs): Sampling frequency in Hz (default: 1.0).
            window: Lag window ('hamming', 'hann', 'blackman', 'rect').
            as_dict: If True, returns comprehensive dictionary.

        Returns:
            1D complex array of S_x^alpha(f) (or dictionary if as_dict=True).
        """
        params = {**self._config, **kwargs}
        fs = _extract_sample_rate(params, default=1.0)
        alpha = float(params.get("alpha", 0.0))
        nfft = int(params.get("nfft_f", params.get("nfft", 256)))
        win_type = str(params.get("window", "hamming"))

        # Limit max_lag so n_lags = 2*max_lag + 1 <= nfft to maintain symmetric lag coverage
        requested_lag = int(params.get("max_lag", params.get("L", 64)))
        max_lag = min(requested_lag, max(16, (nfft // 2) - 1))

        # Check if data is already a 1D cyclic autocorrelation array or raw IQ
        arr = np.asarray(data)
        if np.iscomplexobj(arr) and "r_alpha" in kwargs:
            r_alpha = np.asarray(kwargs["r_alpha"], dtype=np.complex128).ravel()
        elif hasattr(data, "size") and data.ndim == 1 and not np.iscomplexobj(data) and "is_r_alpha" in params:
            r_alpha = np.asarray(data, dtype=np.complex128).ravel()
        else:
            # Compute R_x^alpha[ell] from raw IQ data at exact floating-point alpha
            z = _as_iq_array(data)
            _caf_mod = importlib.import_module(".Cyclic Autocorrelation", package="packages.Cyclostationary Analysis")
            caf_stage = _caf_mod.CyclicAutocorrelationMetric(alpha=alpha, max_lag=max_lag, sample_rate=fs)
            r_alpha = caf_stage.compute(z)

        n_lags = len(r_alpha)
        if n_lags == 0:
            empty = np.array([], dtype=np.complex128)
            return {"s_alpha": empty, "frequencies": np.array([])} if params.get("as_dict") else empty

        # Apply lag window w[ell]
        win = _get_window(win_type, n_lags)
        r_windowed = r_alpha * win

        # Zero-pad to nfft frequency bins
        s_alpha = np.fft.fftshift(np.fft.fft(r_windowed, n=nfft)) / fs
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))

        if params.get("as_dict", False):
            mag = np.abs(s_alpha)
            pwr_db = 10.0 * np.log10(np.maximum(1e-18, mag))
            return {
                "s_alpha": s_alpha,
                "frequencies": freqs,
                "magnitude": mag,
                "power_db": pwr_db,
                "alpha_hz": alpha,
                "peak_power_db": float(np.max(pwr_db)) if pwr_db.size > 0 else 0.0,
                "peak_frequency_hz": float(freqs[np.argmax(mag)]) if mag.size > 0 else 0.0,
                "sample_rate_hz": fs,
                "nfft_f": nfft,
                "nfft": nfft,
            }

        return s_alpha


def spectral_correlation(
    data: Any,
    alpha: float = 0.0,
    max_lag: int = 64,
    nfft: int = 256,
    sample_rate: float = 1.0,
    as_dict: bool = False,
    **kwargs: Any
) -> np.ndarray | dict[str, Any]:
    """Functional wrapper for SpectralCorrelationMetric."""
    metric = SpectralCorrelationMetric(alpha=alpha, max_lag=max_lag, nfft=nfft, sample_rate=sample_rate, as_dict=as_dict, **kwargs)
    return metric.compute(data)


def main() -> None:
    print("=" * 60)
    print("  SpectralCorrelationMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t)
    metric = SpectralCorrelationMetric()
    result = metric.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
