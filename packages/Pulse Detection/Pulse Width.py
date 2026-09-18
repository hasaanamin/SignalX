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
from .base import BasePulseDetectionMetric, PulseDetectionFactory
from ._common import _as_signal_array


@PulseDetectionFactory.register(["pulse_width", "width", "calculate_pulse_width", "fwhm"])
class PulseWidthMetric(BasePulseDetectionMetric):
    """
    Stage 6: Pulse Width Calculation.
    Computes pulse width in samples, seconds, milliseconds, microseconds, and nanoseconds,
    supporting FWHM (50% amplitude) and fractional-sample edge positions.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "pulse_width"

    def compute(
        self,
        signal_or_edges: Any,
        sample_rate: float = 1.0,
        unit: str = "s",
        edge_level_percent: float = 50.0,
        return_dict: bool = True,
        **kwargs: Any
    ) -> list[dict[str, float]] | float | list[float]:
        """
        Calculates pulse width across detected pulse edges.

        Parameters:
            signal_or_edges: Raw signal array, envelope, or edge list from LeadingTrailingEdgeDetectionMetric.
            sample_rate: Acquisition sampling rate in Hz (default 1.0).
            unit: Output unit for primary width ('s', 'ms', 'us', 'ns', 'samples').
            edge_level_percent: Amplitude crossing percentage (e.g. 50% for FWHM).
            return_dict: If True, returns detailed metrics dict per pulse.
        """
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be strictly positive, got {sample_rate}")

        # Check if already edge records list or empty list
        if isinstance(signal_or_edges, list):
            if len(signal_or_edges) == 0:
                return [] if return_dict else 0.0
            if isinstance(signal_or_edges[0], dict) and "leading_edge" in signal_or_edges[0]:
                edges = signal_or_edges
            else:
                from . import EnvelopeExtractionMetric, LeadingTrailingEdgeDetectionMetric
                env = EnvelopeExtractionMetric().compute(signal_or_edges, return_dict=False)
                edges = LeadingTrailingEdgeDetectionMetric().compute(env, edge_level_percent=edge_level_percent, **kwargs)
        else:
            from . import EnvelopeExtractionMetric, LeadingTrailingEdgeDetectionMetric
            env = EnvelopeExtractionMetric().compute(signal_or_edges, return_dict=False)
            edges = LeadingTrailingEdgeDetectionMetric().compute(env, edge_level_percent=edge_level_percent, **kwargs)

        if not edges:
            return [] if return_dict else 0.0

        widths: list[dict[str, float]] = []
        unit_scale = {
            "s": 1.0,
            "ms": 1e3,
            "us": 1e6,
            "ns": 1e9,
            "samples": sample_rate,
        }.get(unit.lower(), 1.0)

        for e in edges:
            lead = float(e["leading_edge"])
            trail = float(e["trailing_edge"])
            w_samples = max(0.0, trail - lead)
            w_sec = w_samples / float(sample_rate)

            pulse_dict = {
                "width_samples": round(w_samples, 4),
                "width_seconds": float(w_sec),
                "width_ms": float(w_sec * 1e3),
                "width_us": float(w_sec * 1e6),
                "width_ns": float(w_sec * 1e9),
                "fwhm_seconds": float(w_sec) if edge_level_percent == 50.0 else None,
                "leading_edge": lead,
                "trailing_edge": trail,
                "sample_rate": float(sample_rate),
            }
            widths.append(pulse_dict)

        if not return_dict:
            values = [w["width_seconds"] * unit_scale for w in widths]
            return values[0] if len(values) == 1 else values

        return widths


_instance = PulseWidthMetric()
pulse_width = _instance.compute
calculate_pulse_width = _instance.compute

__all__ = ["PulseWidthMetric", "pulse_width", "calculate_pulse_width"]


def main() -> None:
    print("=" * 60)
    print("  PulseWidthMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = PulseWidthMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
