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
from .base import BaseNoiseMetric, NoiseMetricFactory
from ._common import _as_iq_array, _extract_sample_rate, _compute_psd, _estimate_fundamental_carrier


@NoiseMetricFactory.register(["signal_quality_metrics", "signal_quality", "fidelity_metrics", "evm_mer_sinad", "signal_metrics"])
class SignalQualityMetrics(BaseNoiseMetric):
    """
    Comprehensive Signal Quality and Modulation Fidelity Suite.
    MATLAB Reference: `comm.EVM`, `comm.MER`, `sinad`, `thd`, `sfdr`, `enob`.
    Computes:
      - EVM (Error Vector Magnitude): RMS % and dB
      - MER (Modulation Error Ratio): dB
      - SINAD (Signal-to-Noise-and-Distortion): dB
      - THD (Total Harmonic Distortion): dB and %
      - SFDR (Spurious-Free Dynamic Range): dBc
      - ENOB (Effective Number of Bits): bits
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "signal_quality_metrics"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        ideal_reference: Any | None = None,
        num_harmonics: int = 5,
        guard_bins: int = 5,
        **kwargs: Any,
    ) -> dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {
                "evm_pct": 0.0, "evm_db": -np.inf, "mer_db": np.inf,
                "sinad_db": 0.0, "thd_db": -np.inf, "sfdr_dbc": 0.0, "enob_bits": 0.0
            }

        total_power = float(np.mean(np.abs(z) ** 2))

        # 2. Spectral Decomposition: SINAD, THD, SFDR, Blind EVM/MER
        freqs, psd = _compute_psd(z, fs=fs)
        df = fs / len(psd)
        peak_idx = int(np.argmax(psd))
        f_fund = float(freqs[peak_idx])

        # Fundamental Power
        k_min = max(0, peak_idx - guard_bins)
        k_max = min(len(psd), peak_idx + guard_bins + 1)
        p_fund = float(np.sum(psd[k_min:k_max]) * df)
        p_fund = max(1e-18, p_fund)

        # 1. Reference Signal & EVM / MER
        if ideal_reference is not None:
            s_ref = _as_iq_array(ideal_reference)
            if s_ref.size != n:
                raise ValueError(f"Reference signal size ({s_ref.size}) must match input signal size ({n}).")
            err = z - s_ref
            p_ref = float(np.mean(np.abs(s_ref) ** 2))
            p_err = float(np.mean(np.abs(err) ** 2))
            evm_rms = float(np.sqrt(p_err / max(1e-18, p_ref)) * 100.0)
            evm_db = float(20.0 * np.log10(max(1e-12, evm_rms / 100.0)))
            mer_db = float(10.0 * np.log10(max(1e-12, p_ref / max(1e-18, p_err))))
        else:
            # Blind spectral EVM/MER estimation
            p_noise_spec = float(np.sum(np.delete(psd, np.arange(k_min, k_max))) * df)
            p_noise_spec = max(1e-18, p_noise_spec)
            evm_rms = float(np.sqrt(p_noise_spec / p_fund) * 100.0)
            evm_db = float(20.0 * np.log10(max(1e-12, evm_rms / 100.0)))
            mer_db = float(10.0 * np.log10(max(1e-12, p_fund / p_noise_spec)))
        # Harmonic Extraction
        p_harmonics = 0.0
        harmonic_powers = []
        for h in range(2, num_harmonics + 1):
            f_h = f_fund * h
            # Fold back to Nyquist range [-fs/2, fs/2]
            f_h_wrapped = (f_h + fs / 2.0) % fs - fs / 2.0
            idx_h = int(np.argmin(np.abs(freqs - f_h_wrapped)))
            kh_min = max(0, idx_h - 2)
            kh_max = min(len(psd), idx_h + 3)
            # Avoid fundamental mainlobe
            if kh_max < k_min or kh_min > k_max:
                p_h = float(np.sum(psd[kh_min:kh_max]) * df)
                p_harmonics += p_h
                harmonic_powers.append(p_h)

        # Total Noise + Distortion
        p_noise_dist = max(1e-18, total_power - p_fund)

        # SINAD: 10 * log10( (P_fund + P_noise_dist) / P_noise_dist )
        sinad_lin = max(1.0, total_power / p_noise_dist)
        sinad_db = float(10.0 * np.log10(sinad_lin))

        # THD: 10 * log10( P_harmonics / P_fund )
        p_harm_clean = max(1e-18, min(p_fund, p_harmonics))
        thd_lin = p_harm_clean / p_fund
        thd_db = float(10.0 * np.log10(max(1e-12, thd_lin)))
        thd_pct = float(np.sqrt(thd_lin) * 100.0)

        # SFDR: Peak Fundamental to Highest Spur (dBc)
        mask = np.ones(len(psd), dtype=bool)
        mask[k_min:k_max] = False
        non_fund_psd = psd[mask]
        if non_fund_psd.size > 0:
            highest_spur = float(np.max(non_fund_psd))
            fund_peak_density = float(np.max(psd[k_min:k_max]))
            sfdr_dbc = float(10.0 * np.log10(max(1.0, fund_peak_density / max(1e-18, highest_spur))))
        else:
            sfdr_dbc = 100.0

        # ENOB: (SINAD_dB - 1.76) / 6.02
        enob_bits = float(max(0.0, (sinad_db - 1.76) / 6.02))

        return {
            "evm_rms_pct": evm_rms,
            "evm_db": evm_db,
            "mer_db": mer_db,
            "sinad_db": sinad_db,
            "thd_db": thd_db,
            "thd_pct": thd_pct,
            "sfdr_dbc": sfdr_dbc,
            "enob_bits": enob_bits,
            "fundamental_freq_hz": f_fund,
            "fundamental_power_dbw": float(10.0 * np.log10(p_fund)),
            "total_power_dbw": float(10.0 * np.log10(max(1e-18, total_power))),
            "sample_rate": fs,
        }


def signal_quality_metrics(
    iq_data: Any,
    sample_rate: float | None = None,
    ideal_reference: Any | None = None,
    num_harmonics: int = 5,
    guard_bins: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Functional DSP wrapper for complete signal quality metrics suite.
    """
    metric = SignalQualityMetrics()
    return metric.compute(iq_data, sample_rate=sample_rate, ideal_reference=ideal_reference, num_harmonics=num_harmonics, guard_bins=guard_bins, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  SignalQualityMetrics: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = SignalQualityMetrics()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
