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

import numpy as np
from .base import BaseHigherOrderMetric, HigherOrderStatisticsFactory
from ._common import _prepare_zero_mean_iq


@HigherOrderStatisticsFactory.register(["sixth_order_cumulants", "c6x", "6th_order_cumulants"])
class SixthOrderCumulantsMetric(BaseHigherOrderMetric):
    """
    Computes Full General Sixth-Order Cumulants (C60 and C63):
      - C60 = M60 - 15 * M20 * M40 - 10 * (M30)^2 + 30 * (M20)^3
      - C63 = M63 - 9 * M21 * M42 - 3 * M20 * conj(M41) - 3 * conj(M20) * M41 - 9 * |M31|^2 - |M30|^2 + 18 * |M20|^2 * M21 + 12 * (M21)^3
      
    High-order cumulants provide strong constellation boundary separation
    for distinguishing 16-QAM from 64-QAM and PSK modulations.
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes SixthOrderCumulantsMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up SixthOrderCumulantsMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "sixth_order_cumulants"

    def compute(self, iq_data, *, normalize: bool = False, **kwargs) -> dict[str, complex]:
        # Step 1: Center signal to zero mean and obtain average power
        z_zero_mean, power_sigma2 = _prepare_zero_mean_iq(iq_data, normalize=normalize)
        z_conj = np.conj(z_zero_mean)
        
        # Step 2: 2nd-order moments
        m20 = np.mean(z_zero_mean ** 2)
        m21 = np.mean(np.abs(z_zero_mean) ** 2)

        # Step 3: 3rd-order moments
        m30 = np.mean(z_zero_mean ** 3)
        m31 = np.mean((z_zero_mean ** 2) * z_conj)
        
        # Step 4: 4th-order moments
        m40 = np.mean(z_zero_mean ** 4)
        m41 = np.mean((z_zero_mean ** 3) * z_conj)
        m42 = np.mean(np.abs(z_zero_mean) ** 4)
        
        # Step 5: 6th-order moments
        m60 = np.mean(z_zero_mean ** 6)
        m63 = np.mean(np.abs(z_zero_mean) ** 6)
        
        # Step 6: Compute full general 6th-order cumulants
        c60 = m60 - 15.0 * m20 * m40 - 10.0 * (m30 ** 2) + 30.0 * (m20 ** 3)
        c63 = (
            m63
            - 9.0 * m21 * m42
            - 3.0 * m20 * np.conj(m41)
            - 3.0 * np.conj(m20) * m41
            - 9.0 * (np.abs(m31) ** 2)
            - (np.abs(m30) ** 2)
            + 18.0 * (np.abs(m20) ** 2) * m21
            + 12.0 * (m21 ** 3)
        )

        # Numerical cleanup of floating-point residue
        c63_cplx = complex(c63)
        if abs(c63_cplx.imag) < 1e-12 * max(1.0, abs(c63_cplx.real)):
            c63_cplx = complex(c63_cplx.real, 0.0)
        
        return {
            "C60": complex(c60),
            "C63": c63_cplx,
        }


_instance = SixthOrderCumulantsMetric()
sixth_order_cumulants = _instance.compute

__all__ = ["SixthOrderCumulantsMetric", "sixth_order_cumulants"]


def main() -> None:
    print("=" * 60)
    print("  SixthOrderCumulantsMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = SixthOrderCumulantsMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
