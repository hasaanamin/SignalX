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


@PulseDetectionFactory.register(["pulse_segmentation", "segmentation", "segment_pulses", "detect_and_segment"])
class PulseSegmentationMetric(BasePulseDetectionMetric):
    """
    Stage 7: Pulse Segmentation & Physical SNR Calculation.
    Extracts segmented complex IQ pulse waveforms covering the entire region above the noise threshold,
    calculates true physical IQ signal-to-noise ratio (SNR), pulse energy, and compiles complete pulse descriptors.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "pulse_segmentation"

    def compute(
        self,
        signal: Any,
        sample_rate: float = 1.0,
        integration_window_samples: int = 4,
        k_high: float = 3.2,
        k_low: float = 1.4,
        reference_window_samples: int = 64,
        guard_samples: int = 12,
        confirmation_window_samples: int = 4,
        required_high_hits: int = 2,
        termination_samples: int = 3,
        refractory_samples: int = 2,
        min_duration_samples: int = 8,
        max_duration_samples: int | None = None,
        min_gap_samples: int = 2,
        edge_level_percent: float = 50.0,
        pre_padding: int = 0,
        post_padding: int = 0,
        return_diagnostics: bool = False,
        **kwargs: Any
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Executes the full 7-stage pulse detection pipeline on complex baseband IQ.

        Parameters:
            signal: Complex baseband IQ array (e.g. complex64, complex128) or canonical record.
            sample_rate: Sampling frequency in Hz (default 1.0).
            integration_window_samples: Temporal energy integration length M (default 4).
            k_high: High threshold multiplier (default 2.5).
            k_low: Low threshold multiplier (default 0.8).
            reference_window_samples: CFAR noise reference window length.
            guard_samples: Guard cells to prevent self-masking.
            confirmation_window_samples: Confirmation window length.
            required_high_hits: Required High threshold hits.
            termination_samples: Consecutive Low threshold drop samples to confirm termination.
            refractory_samples: Refractory dead-time after pulse termination.
            min_duration_samples: Minimum pulse duration constraint (default 6).
            max_duration_samples: Maximum pulse duration constraint.
            min_gap_samples: Minimum gap for pulse candidate merging.
            edge_level_percent: Relative amplitude crossing for sub-sample edges (default 50%).
            pre_padding: Lead samples before pulse start in segmented waveform.
            post_padding: Lag samples after pulse end in segmented waveform.
            return_diagnostics: If True, returns (segmented_pulses, diagnostics_dict).
        """
        iq = _as_signal_array(signal)
        n = iq.size
        if n == 0:
            return ([], {}) if return_diagnostics else []

        from . import (
            EnvelopeExtractionMetric,
            ThresholdDetectionMetric,
            PulseDetectionMetric,
            PulseBoundariesMetric,
            LeadingTrailingEdgeDetectionMetric,
            PulseWidthMetric,
        )

        # -------------------------------------------------------------
        # Stage 1: Envelope Extraction & Causal Temporal Integration
        # -------------------------------------------------------------
        env_res = EnvelopeExtractionMetric().compute(
            iq,
            integration_window_samples=integration_window_samples,
            return_dict=True,
            **kwargs
        )
        amp_env = env_res["amplitude_envelope"]
        pwr_env = env_res["power_envelope"]
        det_stat = env_res["detection_statistic"]
        M = env_res["integration_window_samples"]

        # -------------------------------------------------------------
        # Stage 2: Robust Adaptive Threshold Detection
        # -------------------------------------------------------------
        th_res = ThresholdDetectionMetric().compute(
            det_stat,
            k_high=k_high,
            k_low=k_low,
            reference_window_samples=reference_window_samples,
            guard_samples=guard_samples,
            integration_window_samples=M,
            adaptive=True,
            **kwargs
        )
        th_hi = th_res["high_threshold"]
        th_lo = th_res["low_threshold"]
        noise_floor_arr = th_res["noise_floor"]
        noise_scale_arr = th_res["noise_scale"]

        # -------------------------------------------------------------
        # Stage 3: 5-State FSM Pulse Detection
        # -------------------------------------------------------------
        candidates = PulseDetectionMetric().compute(
            det_stat,
            high_threshold=th_hi,
            low_threshold=th_lo,
            confirmation_window_samples=confirmation_window_samples,
            required_high_hits=required_high_hits,
            termination_samples=termination_samples,
            refractory_samples=refractory_samples,
            **kwargs
        )

        # -------------------------------------------------------------
        # Stage 4: Coarse Boundary Determination & Causal Delay Expansion
        # -------------------------------------------------------------
        boundaries = PulseBoundariesMetric().compute(
            det_stat,
            candidates=candidates,
            low_threshold=th_lo,
            integration_window_samples=M,
            min_duration_samples=min_duration_samples,
            max_duration_samples=max_duration_samples,
            min_gap_samples=min_gap_samples,
            **kwargs
        )

        # -------------------------------------------------------------
        # Stage 5: Sub-Sample Leading / Trailing Edge Detection
        # -------------------------------------------------------------
        edges = LeadingTrailingEdgeDetectionMetric().compute(
            amp_env,
            boundaries=boundaries,
            edge_level_percent=edge_level_percent,
            **kwargs
        )

        # -------------------------------------------------------------
        # Stage 6: Pulse Width Calculation
        # -------------------------------------------------------------
        widths = PulseWidthMetric().compute(
            edges,
            sample_rate=sample_rate,
            edge_level_percent=edge_level_percent,
            return_dict=True,
            **kwargs
        )

        # -------------------------------------------------------------
        # Stage 7: Pulse Segmentation & Physical Complex IQ SNR
        # -------------------------------------------------------------
        segmented_pulses: list[dict[str, Any]] = []

        for idx, (b, e, w) in enumerate(zip(boundaries, edges, widths)):
            s_coarse = int(b["start_idx"])
            e_coarse = int(b["end_idx"])
            w_samples = float(w["width_samples"])

            if w_samples < min_duration_samples:
                continue
            if max_duration_samples is not None and w_samples > max_duration_samples:
                continue

            lead_frac = float(e["leading_edge"])
            trail_frac = float(e["trailing_edge"])

            # 1. Segment directly from the ORIGINAL complex IQ stream across full pulse window
            slice_start = max(0, s_coarse - pre_padding)
            slice_end = min(n, e_coarse + 1 + post_padding)
            pulse_iq = iq[slice_start:slice_end]

            # 2. Estimate Physical Noise Power from Local Pulse-Free Complex IQ Samples
            noise_slices = []
            if slice_start > 4:
                noise_slices.append(iq[max(0, slice_start - 32) : slice_start])
            if slice_end < n - 4:
                noise_slices.append(iq[slice_end : min(n, slice_end + 32)])

            if noise_slices:
                local_noise_iq = np.concatenate(noise_slices)
                noise_power = float(np.mean(np.abs(local_noise_iq) ** 2))
            else:
                noise_power = float(np.mean(np.abs(iq) ** 2))

            # 3. Calculate Total and Signal-Only Power on Complex IQ
            mag_sq = np.abs(pulse_iq) ** 2
            pulse_energy = float(np.sum(mag_sq))
            pulse_region_power = float(np.mean(mag_sq)) if mag_sq.size > 0 else 0.0

            if noise_power > 1e-12:
                signal_power = max(1e-12, pulse_region_power - noise_power)
                snr_db = float(10.0 * np.log10(signal_power / noise_power))
            else:
                snr_db = None

            w_sec = float(w["width_seconds"])
            peak_amp = float(e["peak_val"])
            peak_idx = int(e["peak_idx"])

            pulse_record = {
                "pulse_id": len(segmented_pulses) + 1,
                "start": s_coarse,
                "end": e_coarse,
                "coarse_start": s_coarse,
                "coarse_end": e_coarse,
                "slice_start": slice_start,
                "slice_end": slice_end,
                "leading_edge": lead_frac,
                "trailing_edge": trail_frac,
                "leading_edge_time_s": float(lead_frac / sample_rate),
                "trailing_edge_time_s": float(trail_frac / sample_rate),
                "pulse_width_samples": round(w_samples, 4),
                "pulse_width_seconds": w_sec,
                "pulse_width_us": float(w_sec * 1e6),
                "pulse_width_ns": float(w_sec * 1e9),
                "peak_amplitude": peak_amp,
                "peak_index": peak_idx,
                "peak_time_s": float(peak_idx / sample_rate),
                "pulse_energy": pulse_energy,
                "pulse_power": pulse_region_power,
                "noise_power": noise_power,
                "local_snr_db": round(snr_db, 2) if snr_db is not None else None,
                "edge_confidence": e.get("edge_confidence", 1.0),
                "rise_time_samples": e.get("rise_time_samples", 0.0),
                "sample_rate": float(sample_rate),
                "waveform": pulse_iq,  # Complex IQ array
            }
            segmented_pulses.append(pulse_record)

        if return_diagnostics:
            diagnostics = {
                "iq_data": iq,
                "amplitude_envelope": amp_env,
                "power_envelope": pwr_env,
                "detection_statistic": det_stat,
                "noise_floor": noise_floor_arr,
                "noise_scale": noise_scale_arr,
                "high_threshold": th_hi,
                "low_threshold": th_lo,
                "candidates": candidates,
                "boundaries": boundaries,
                "edges": edges,
                "widths": widths,
                "integration_window_samples": M,
            }
            return segmented_pulses, diagnostics

        return segmented_pulses


_instance = PulseSegmentationMetric()
pulse_segmentation = _instance.compute
segment_pulses = _instance.compute
detect_and_segment = _instance.compute

__all__ = ["PulseSegmentationMetric", "pulse_segmentation", "segment_pulses", "detect_and_segment"]


def main() -> None:
    print("=" * 60)
    print("  PulseSegmentationMetric: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = PulseSegmentationMetric()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
