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


@NoiseMetricFactory.register(["interference_power", "jammer_power", "spur_power", "aci_power", "interference"])
class InterferencePowerMetric(BaseNoiseMetric):
    """
    Computes total discrete Interference / Jammer Power (P_i) excluding desired signal and thermal noise:
      - Detects in-band / adjacent-channel CW interferers and spurs.
      - Estimates total interference power in linear and dBm/dBW.
      - Enumerates detected interferer frequencies and peak amplitudes.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "interference_power"

    def compute(
        self,
        iq_data: Any,
        *,
        sample_rate: float | None = None,
        guard_bins: int = 5,
        spur_threshold_db: float = 6.0,
        as_dict: bool = False,
        **kwargs: Any,
    ) -> float | dict[str, Any]:
        z = _as_iq_array(iq_data)
        fs = _extract_sample_rate(iq_data, sample_rate=sample_rate, default_fs=1.0)
        n = z.size
        if n == 0:
            return {"interference_power_linear": 0.0, "num_interferers": 0} if as_dict else 0.0

        freqs, psd = _compute_psd(z, fs=fs)
        df = fs / len(psd)

        # 1. Mask Fundamental Peak
        peak_idx = int(np.argmax(psd))
        k_min = max(0, peak_idx - guard_bins)
        k_max = min(len(psd), peak_idx + guard_bins + 1)

        mask = np.ones(len(psd), dtype=bool)
        mask[k_min:k_max] = False

        non_signal_psd = psd[mask]
        non_signal_freqs = freqs[mask]

        if non_signal_psd.size == 0:
            return {"interference_power_linear": 0.0, "num_interferers": 0} if as_dict else 0.0

        # 2. Noise floor baseline
        n0_med = float(np.median(non_signal_psd))
        spur_th = n0_med * (10.0 ** (spur_threshold_db / 10.0))

        # 3. Detect spurs exceeding spur threshold
        is_spur = non_signal_psd > spur_th
        p_interf = float(np.sum(non_signal_psd[is_spur]) * df)
        p_interf_dbw = float(10.0 * np.log10(max(1e-18, p_interf)))
        p_interf_dbm = float(p_interf_dbw + 30.0)

        # Identify discrete peaks
        interferers = []
        if np.any(is_spur):
            spur_indices = np.where(is_spur)[0]
            # Group consecutive bins into distinct interferer spurs
            groups = np.split(spur_indices, np.where(np.diff(spur_indices) > 1)[0] + 1)
            for grp in groups:
                if len(grp) > 0:
                    max_bin = grp[np.argmax(non_signal_psd[grp])]
                    f_int = float(non_signal_freqs[max_bin])
                    p_int = float(np.sum(non_signal_psd[grp]) * df)
                    interferers.append({
                        "frequency_hz": f_int,
                        "power_linear": p_int,
                        "power_dbm": float(10.0 * np.log10(max(1e-18, p_int)) + 30.0),
                    })

        if as_dict:
            return {
                "interference_power_linear": p_interf,
                "interference_power_dbw": p_interf_dbw,
                "interference_power_dbm": p_interf_dbm,
                "num_interferers": len(interferers),
                "interferers": interferers,
                "sample_rate": fs,
            }
        return p_interf


def interference_power(
    iq_data: Any,
    sample_rate: float | None = None,
    guard_bins: int = 5,
    spur_threshold_db: float = 6.0,
    as_dict: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """
    Functional DSP wrapper for interference power estimation.
    """
    metric = InterferencePowerMetric()
    return metric.compute(iq_data, sample_rate=sample_rate, guard_bins=guard_bins, spur_threshold_db=spur_threshold_db, as_dict=as_dict, **kwargs)


def main() -> None:
    print("=" * 60)
    print("  InterferencePowerMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = InterferencePowerMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
