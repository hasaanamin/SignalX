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
from ._common import _as_iq_array, _extract_sample_rate, _compute_psd


@NoiseMetricFactory.register(["phase_noise", "phase_jitter", "ssb_phase_noise", "l_f"])
class PhaseNoiseMetric(BaseNoiseMetric):
    """
    Computes Phase Noise PSD L(Delta f) in dBc/Hz and Integrated Phase Jitter (deg, ps).
    MATLAB Reference: `comm.PhaseNoise`, phase fluctuation spectral density.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "phase_noise"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        offsets_hz: list[float] | tuple[float, ...] | None = None,
        integration_limits_hz: tuple[float, float] | None = None,
        as_dict: bool = False,
        **kwargs: Any,
    ) -> float | dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n < 4:
            return {"phase_noise_offsets": {}, "integrated_jitter_deg": 0.0} if as_dict else 0.0

        if offsets_hz is None:
            offsets_hz = [1e3, 10e3, 100e3, 1e6]

        # 1. Extract unwrapped phase
        raw_phase = np.unwrap(np.angle(z))
        t = np.arange(n) / fs

        # 2. Remove linear carrier frequency and initial phase offset
        poly = np.polyfit(t, raw_phase, 1)
        phase_detrend = raw_phase - np.polyval(poly, t)

        # 3. Compute Phase Fluctuation PSD S_phi(f)
        nperseg = min(n, 2048)
        nperseg = max(16, 1 << (nperseg.bit_length() - 1))
        freqs, s_phi = _compute_psd(phase_detrend, fs=fs, nperseg=nperseg)

        # Single-Sideband Phase Noise L(f) = 10 * log10(S_phi(f) / 2) [dBc/Hz]
        pos_mask = freqs > 0
        f_pos = freqs[pos_mask]
        s_phi_pos = s_phi[pos_mask]
        l_f = 10.0 * np.log10(np.maximum(1e-24, s_phi_pos / 2.0))

        # Evaluate at requested offset frequencies
        offset_results = {}
        for f_off in offsets_hz:
            if f_pos.size > 0 and f_off <= f_pos[-1]:
                idx = np.argmin(np.abs(f_pos - f_off))
                offset_results[f"{int(f_off)}Hz"] = float(l_f[idx])
            else:
                offset_results[f"{int(f_off)}Hz"] = None

        # 4. Integrated Phase Jitter (in degrees and picoseconds)
        if integration_limits_hz is not None:
            f_low, f_high = integration_limits_hz
        else:
            f_low = f_pos[0] if f_pos.size > 0 else 1.0
            f_high = f_pos[-1] if f_pos.size > 0 else fs / 2.0

        integ_mask = (f_pos >= f_low) & (f_pos <= f_high)
        if np.any(integ_mask):
            df = f_pos[1] - f_pos[0] if len(f_pos) > 1 else 1.0
            # Integrated phase variance sigma_phi^2 = 2 * int L(f) df
            phi_var_rad2 = float(np.sum(s_phi_pos[integ_mask]) * df)
            rms_jitter_rad = float(np.sqrt(max(0.0, phi_var_rad2)))
        else:
            rms_jitter_rad = float(np.std(phase_detrend))

        rms_jitter_deg = float(np.rad2deg(rms_jitter_rad))
        f_carrier = abs(poly[0]) / (2 * np.pi)
        f_ref = max(1.0, f_carrier if f_carrier > 0 else fs / 4.0)
        rms_jitter_sec = float(rms_jitter_rad / (2 * np.pi * f_ref))
        rms_jitter_ps = float(rms_jitter_sec * 1e12)

        if as_dict:
            return {
                "phase_noise_dbc_hz": offset_results,
                "rms_phase_error_deg": rms_jitter_deg,
                "rms_phase_error_rad": rms_jitter_rad,
                "integrated_phase_jitter_ps": rms_jitter_ps,
                "carrier_freq_hz": f_carrier,
                "sample_rate": fs,
                "offset_frequencies": f_pos,
                "l_f_spectrum": l_f,
            }
        return rms_jitter_deg


def phase_noise(
    iq_data: Any,
    sample_rate: float | None = None,
    offsets_hz: list[float] | tuple[float, ...] | None = None,
    integration_limits_hz: tuple[float, float] | None = None,
    as_dict: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """
    Functional DSP wrapper for phase noise and phase jitter calculation.
    """
    metric = PhaseNoiseMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, offsets_hz=offsets_hz, integration_limits_hz=integration_limits_hz, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  PhaseNoiseMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = PhaseNoiseMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
