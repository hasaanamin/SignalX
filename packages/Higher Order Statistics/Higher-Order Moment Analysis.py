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


@HigherOrderStatisticsFactory.register(["higher_order_moment_analysis", "hos_analysis", "moment_analysis"])
class HigherOrderMomentAnalysisMetric(BaseHigherOrderMetric):
    """Performs comprehensive Higher-Order Moment & Cumulant Analysis."""

    def __init__(self, **kwargs) -> None:
        """Constructor."""
        super().__init__(**kwargs)

    def __del__(self) -> None:
        """Destructor."""
        super().__del__()

    @property
    def name(self) -> str:
        return "higher_order_moment_analysis"

    def compute(self, iq_data, **kwargs) -> dict:
        # Step 1: Center signal to zero mean WITHOUT pre-normalizing amplitude
        z0, signal_power = _prepare_zero_mean_iq(iq_data, normalize=False)
        z_conj = np.conj(z0)
        
        # Step 2: Compute full zero-mean complex moments
        m20 = np.mean(z0 ** 2)
        m21 = np.mean(np.abs(z0) ** 2)
        m30 = np.mean(z0 ** 3)
        m31 = np.mean((z0 ** 2) * z_conj)
        m40 = np.mean(z0 ** 4)
        m41 = np.mean((z0 ** 3) * z_conj)
        m42 = np.mean(np.abs(z0) ** 4)
        m60 = np.mean(z0 ** 6)
        m63 = np.mean(np.abs(z0) ** 6)
        
        # Step 3: Compute full general unnormalized cumulants
        c20 = m20
        c21 = m21
        c40 = m40 - 3.0 * (m20 ** 2)
        c41 = m41 - 3.0 * m20 * m21
        c42 = m42 - (np.abs(m20) ** 2) - 2.0 * (m21 ** 2)
        
        # Full general 6th-order cumulant formulas (including 3rd-order moment terms)
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

        # Step 4: Perform post-normalization by signal power C21 = M21 = sigma^2
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
        
        # Step 5: Construct normalized modulation feature vector
        feature_vector = [
            float(np.abs(c20_norm)),
            float(np.abs(c40_norm)),
            float(np.abs(c41_norm)),
            float(np.abs(c42_norm)),
            float(np.abs(c60_norm)),
            float(np.abs(c63_norm)),
        ]

        return {
            "signal_power": float(signal_power),
            "moments": {
                "M20": complex(m20),
                "M21": complex(m21),
                "M30": complex(m30),
                "M31": complex(m31),
                "M40": complex(m40),
                "M41": complex(m41),
                "M42": complex(m42),
                "M60": complex(m60),
                "M63": complex(m63),
            },
            "cumulants": {
                "C20": complex(c20),
                "C21": complex(c21),
                "C40": complex(c40),
                "C41": complex(c41),
                "C42": complex(c42),
                "C60": complex(c60),
                "C63": complex(c63),
            },
            "normalized_cumulants": {
                "c20": complex(c20_norm),
                "c21": complex(c21_norm),
                "c40": complex(c40_norm),
                "c41": complex(c41_norm),
                "c42": complex(c42_norm),
                "c60": complex(c60_norm),
                "c63": complex(c63_norm),
            },
            "normalized_features": {
                "abs_c20": feature_vector[0],
                "abs_c40": feature_vector[1],
                "abs_c41": feature_vector[2],
                "abs_c42": feature_vector[3],
                "abs_c60": feature_vector[4],
                "abs_c63": feature_vector[5],
            },
            "feature_vector": feature_vector,
        }


_instance = HigherOrderMomentAnalysisMetric()
higher_order_moment_analysis = _instance.compute

__all__ = ["HigherOrderMomentAnalysisMetric", "higher_order_moment_analysis"]


def main() -> None:
    print("=" * 60)
    print("  HigherOrderMomentAnalysisMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = HigherOrderMomentAnalysisMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
