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


@HigherOrderStatisticsFactory.register(["normalized_cumulants", "norm_cumulants", "standardized_cumulants"])
class NormalizedCumulantsMetric(BaseHigherOrderMetric):
    """
    Computes standardized scale-invariant cumulants normalized by power C21^(order/2):
      - c20 = C20 / C21
      - c21 = C21 / C21 = 1.0
      - c40 = C40 / (C21)^2
      - c41 = C41 / (C21)^2
      - c42 = C42 / (C21)^2
      - c60 = C60 / (C21)^3
      - c63 = C63 / (C21)^3
    """

    def __init__(self, **kwargs) -> None:
        """Constructor: Initializes NormalizedCumulantsMetric instance."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor: Cleans up NormalizedCumulantsMetric instance resources."""
        super().__del__()

    @property
    def name(self) -> str:
        return "normalized_cumulants"

    def compute(self, iq_data, **kwargs) -> dict[str, complex | float]:
        # Step 1: Center signal to zero mean WITHOUT pre-normalizing
        z0, signal_power = _prepare_zero_mean_iq(iq_data, normalize=False)
        z_conj = np.conj(z0)
        
        # Step 2: 2nd-order moments
        m20 = np.mean(z0 ** 2)
        m21 = np.mean(np.abs(z0) ** 2)

        # Step 3: 3rd-order moments
        m30 = np.mean(z0 ** 3)
        m31 = np.mean((z0 ** 2) * z_conj)
        
        # Step 4: 4th-order moments
        m40 = np.mean(z0 ** 4)
        m41 = np.mean((z0 ** 3) * z_conj)
        m42 = np.mean(np.abs(z0) ** 4)
        
        # Step 5: 6th-order moments
        m60 = np.mean(z0 ** 6)
        m63 = np.mean(np.abs(z0) ** 6)
        
        # Step 6: Full unnormalized cumulants calculation
        c20 = m20
        c21 = m21
        c40 = m40 - 3.0 * (m20 ** 2)
        c41 = m41 - 3.0 * m20 * m21
        c42 = m42 - (np.abs(m20) ** 2) - 2.0 * (m21 ** 2)
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
        if abs(complex(c63).imag) < 1e-12 * max(1.0, abs(complex(c63).real)):
            c63 = complex(complex(c63).real, 0.0)
        
        # Step 7: Post-normalization by power C21 = M21
        c21_pwr = float(np.real(c21))
        if c21_pwr > 0:
            c20_norm = c20 / c21_pwr
            c21_norm = 1.0 + 0.0j
            c40_norm = c40 / (c21_pwr ** 2)
            c41_norm = c41 / (c21_pwr ** 2)
            c42_norm = c42 / (c21_pwr ** 2)
            c60_norm = c60 / (c21_pwr ** 3)
            c63_norm = c63 / (c21_pwr ** 3)
        else:
            c20_norm = c20
            c21_norm = c21
            c40_norm = c40
            c41_norm = c41
            c42_norm = c42
            c60_norm = c60
            c63_norm = c63

        return {
            "c20": complex(c20_norm),
            "c21": complex(c21_norm),
            "c40": complex(c40_norm),
            "c41": complex(c41_norm),
            "c42": complex(c42_norm),
            "c60": complex(c60_norm),
            "c63": complex(c63_norm),
            "abs_c20": float(np.abs(c20_norm)),
            "abs_c40": float(np.abs(c40_norm)),
            "abs_c41": float(np.abs(c41_norm)),
            "abs_c42": float(np.abs(c42_norm)),
            "abs_c60": float(np.abs(c60_norm)),
            "abs_c63": float(np.abs(c63_norm)),
        }


_instance = NormalizedCumulantsMetric()
normalized_cumulants = _instance.compute

__all__ = ["NormalizedCumulantsMetric", "normalized_cumulants"]


def main() -> None:
    print("=" * 60)
    print("  NormalizedCumulantsMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = NormalizedCumulantsMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
