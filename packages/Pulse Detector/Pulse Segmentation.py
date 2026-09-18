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
from .base import BasePulseDetectorStage, PulseDetectorFactory, PulseCandidate
from ._common import _as_iq_array, _extract_sample_rate


@PulseDetectorFactory.register([
    "pulse_segmentation",
    "segmentation",
    "extract_candidates",
    "segment_pulses",
    "pulse_candidate",
    "segment_pulse_candidates"
])
class PulseSegmentationStage(BasePulseDetectorStage):
    """
    Stage 7: Pulse Segmentation.
    Extracts isolated complex IQ segments for all detected and validated pulse events,
    packaging them into definitive PulseCandidate entities:
      pulseIQ = x[n_lead : n_trail + 1]
      TOA = (sample_offset + n_lead) / F_s  (0-based Python timestamping)
      mean_power = mean(|pulseIQ|^2)
      SNR = (mean_power - N) / N
    MATLAB Reference:
      pulseIQ = x(leadingIdx:trailingIdx);
      TOA = (leadingIdx - 1) / Fs;  % MATLAB 1-based index
    """

    def __init__(self, sample_rate: float | None = None, include_rejected: bool = False, **kwargs: Any) -> None:
        cfg = dict(kwargs)
        if sample_rate is not None:
            cfg["sample_rate"] = sample_rate
        cfg["include_rejected"] = include_rejected
        super().__init__(**cfg)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "pulse_segmentation"

    def compute(self, data: Any, **kwargs: Any) -> list[PulseCandidate]:
        """
        Extracts and segments Pulse Candidates from conditioned IQ and validated edge events.

        Parameters:
            data: Complex IQ array x[n].
            validated_events: List of validated pulse edge dictionaries from Stage 6.
            sample_rate (Fs): Sampling frequency in Hz (must be explicitly provided and > 0).
            envelope: Optional 1D envelope array for peak_envelope extraction.
            noise_floor: Optional 1D noise floor array.
            sample_offset: Global sample index offset for streaming TOA computation (default: 0).
            start_candidate_id: Starting integer ID for candidates (default: 1).
            include_rejected: If True, also includes rejected candidate objects flagged with is_valid_pulse=False.

        Returns:
            List of segmented PulseCandidate objects.
        """
        params = {**self._config, **kwargs}
        z = _as_iq_array(data)
        fs = _extract_sample_rate(params)
        events = params.get("validated_events", params.get("events", []))
        envelope = params.get("envelope")
        noise_floor = params.get("noise_floor")
        sample_offset = int(params.get("sample_offset", 0))
        cand_id = int(params.get("start_candidate_id", 1))
        include_rejected = bool(params.get("include_rejected", False))

        if z.size == 0 or not events:
            return []

        # If envelope not provided, compute raw power on the fly
        if envelope is None:
            env_arr = np.abs(z) ** 2
        else:
            env_arr = np.asarray(envelope, dtype=np.float64).ravel()

        if noise_floor is None:
            med = float(np.median(env_arr))
            nf_arr = np.full_like(env_arr, max(1e-18, med))
        else:
            nf_arr = np.asarray(noise_floor, dtype=np.float64).ravel()

        n_samples = z.size
        candidates: list[PulseCandidate] = []

        for ev in events:
            if isinstance(ev, dict):
                n_lead = int(ev.get("n_lead", 0))
                n_trail = int(ev.get("n_trail", 0))
                is_valid = bool(ev.get("is_valid", True))
                rejection = str(ev.get("rejection_reason", ""))
                rise_slope = float(ev.get("rise_slope", 0.0))
                fall_slope = float(ev.get("fall_slope", 0.0))
            elif isinstance(ev, (tuple, list)) and len(ev) >= 2:
                n_lead, n_trail = int(ev[0]), int(ev[1])
                is_valid = True
                rejection = ""
                rise_slope, fall_slope = 0.0, 0.0
            else:
                continue

            if not is_valid and not include_rejected:
                continue

            # BUG 9 Fix: Preserve rejected invalid-edge events without modifying indices or fabricating 1-sample segments
            if rejection == "INVALID_EDGE_ORDER" or n_trail < n_lead:
                abs_n_lead = sample_offset + n_lead
                abs_n_trail = sample_offset + n_trail
                candidate = PulseCandidate(
                    candidate_id=cand_id,
                    start_idx=n_lead,
                    stop_idx=n_trail,
                    start_idx_absolute=abs_n_lead,
                    stop_idx_absolute=abs_n_trail,
                    toa_sec=float(abs_n_lead / fs),
                    pulse_width_sec=0.0,
                    pulse_width_samples=0,
                    peak_envelope=0.0,
                    mean_power=0.0,
                    snr_db=float("-inf"),
                    noise_floor_linear=0.0,
                    noise_floor_db=float("-inf"),
                    rise_slope=rise_slope,
                    fall_slope=fall_slope,
                    iq_samples=np.empty(0, dtype=np.complex128),
                    is_valid_pulse=False,
                    rejection_reason="INVALID_EDGE_ORDER" if not rejection else rejection,
                )
                candidates.append(candidate)
                cand_id += 1
                continue

            # Bound indices within chunk
            n_lead_clamped = max(0, min(n_lead, n_samples - 1))
            n_trail_clamped = max(n_lead_clamped, min(n_trail, n_samples - 1))

            # 0-indexed Python TOA referenced to absolute stream position
            abs_n_lead = sample_offset + n_lead
            abs_n_trail = sample_offset + n_trail
            toa_sec = float(abs_n_lead / fs)

            # Sliced complex IQ
            iq_slice = z[n_lead_clamped : n_trail_clamped + 1].copy()
            pw_samples = n_trail - n_lead + 1
            pw_sec = float(pw_samples / fs)

            # Calculate mean power directly from original segmented complex IQ
            mean_pwr = float(np.mean(np.abs(iq_slice) ** 2)) if iq_slice.size > 0 else 0.0

            # Peak envelope from detection envelope
            env_slice = env_arr[n_lead_clamped : n_trail_clamped + 1]
            peak_env = float(np.max(env_slice)) if env_slice.size > 0 else 0.0

            # Local background noise floor level
            nf_val = float(nf_arr[n_lead_clamped]) if n_lead_clamped < len(nf_arr) else float(np.median(nf_arr))
            nf_val = max(1e-18, nf_val)
            nf_db = float(10.0 * np.log10(nf_val))

            # True SNR: (P_measured - P_noise) / P_noise
            signal_power = mean_pwr - nf_val
            if signal_power > 0 and nf_val > 0:
                snr_linear = signal_power / nf_val
                snr_db = float(10.0 * np.log10(snr_linear))
            else:
                snr_db = float("-inf")

            candidate = PulseCandidate(
                candidate_id=cand_id,
                start_idx=n_lead,
                stop_idx=n_trail,
                start_idx_absolute=abs_n_lead,
                stop_idx_absolute=abs_n_trail,
                toa_sec=toa_sec,
                pulse_width_sec=pw_sec,
                pulse_width_samples=pw_samples,
                peak_envelope=peak_env,
                mean_power=mean_pwr,
                snr_db=snr_db,
                noise_floor_linear=nf_val,
                noise_floor_db=nf_db,
                rise_slope=rise_slope,
                fall_slope=fall_slope,
                iq_samples=iq_slice,
                is_valid_pulse=is_valid,
                rejection_reason=rejection,
            )
            candidates.append(candidate)
            cand_id += 1

        return candidates


def segment_pulses(
    data: Any,
    validated_events: list[dict[str, Any]],
    sample_rate: float,
    envelope: np.ndarray | None = None,
    noise_floor: np.ndarray | None = None,
    sample_offset: int = 0,
    include_rejected: bool = False,
    **kwargs: Any
) -> list[PulseCandidate]:
    """Functional wrapper for PulseSegmentationStage."""
    stage = PulseSegmentationStage(sample_rate=sample_rate, include_rejected=include_rejected, **kwargs)
    return stage.compute(
        data,
        validated_events=validated_events,
        envelope=envelope,
        noise_floor=noise_floor,
        sample_offset=sample_offset
    )


def main() -> None:
    print("=" * 60)
    print("  PulseSegmentationStage: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = PulseSegmentationStage()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
