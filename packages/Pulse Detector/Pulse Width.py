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
import math
import warnings
import numpy as np
from .base import BasePulseDetectorStage, PulseDetectorFactory
from ._common import _extract_sample_rate


@PulseDetectorFactory.register([
    "pulse_width",
    "width_validation",
    "pulse_gate",
    "width_gating",
    "validate_pulse_width",
    "pulse_validator"
])
class PulseWidthStage(BasePulseDetectorStage):
    """
    Stage 6: Pulse Width & Shape Gating.
    Computes true duration and enforces the pulse-shape gate:
      PW = (n_trail - n_lead + 1) / F_s
    Validates candidates against physiological pulse criteria:
      1. Duration Gate: PW_min <= PW <= PW_max
         - Continuous carriers produce PW > PW_max -> REJECT ('PW_TOO_LONG')
         - One-sample noise spikes produce PW < PW_min -> REJECT ('PW_TOO_SHORT')
      2. Slope Gate: S_rise >= S_min and S_fall <= -S_min (rapid rise and fall)
      3. Background Isolation: Pre-pulse and Post-pulse background levels return near local noise floor
    MATLAB Reference:
      pw = (trailingIdx - leadingIdx + 1) / Fs;
      valid = (pw >= PWmin) & (pw <= PWmax);
    """

    def __init__(
        self,
        sample_rate: float | None = None,
        pw_min: float | None = None,
        pw_max: float | None = None,
        pw_min_samples: int = 3,
        pw_max_samples: int | None = None,
        min_rise_slope: float = 0.0,
        background_window_sec: float = 10e-6,
        **kwargs: Any
    ) -> None:
        cfg = dict(kwargs)
        if sample_rate is not None:
            cfg["sample_rate"] = sample_rate
        cfg["pw_min"] = pw_min
        cfg["pw_max"] = pw_max
        cfg["pw_min_samples"] = pw_min_samples
        cfg["pw_max_samples"] = pw_max_samples
        cfg["min_rise_slope"] = min_rise_slope
        cfg["background_window_sec"] = background_window_sec
        super().__init__(**cfg)

    def __del__(self) -> None:
        super().__del__()

    @property
    def name(self) -> str:
        return "pulse_width"

    def compute(self, data: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Validates edge pairs against pulse width limits, slope thresholds, and background isolation.

        Parameters:
            data: List of edge dictionaries from Stage 5 or boundary tuples (n_lead, n_trail).
            sample_rate (Fs): Sampling rate in Hz (must be explicitly provided and > 0).
            pw_min: Minimum physical pulse width in seconds (default: None).
            pw_max: Maximum physical pulse width in seconds (default: None).
            pw_min_samples: Minimum pulse width in samples (default: 3).
            pw_max_samples: Maximum pulse width in samples (default: None).
            min_rise_slope: Minimum leading edge slope S_min in W/s (default: 0.0).
            background_window_sec: Pre/post isolation duration in seconds (default: 10 us).
            envelope: Optional 1D envelope array.
            raw_power: Optional 1D raw power array |x|^2.
            noise_floor: Optional 1D local noise floor array N[n].
            sigma_n: Optional 1D local noise sigma array.

        Returns:
            List of dictionaries with validated candidate metrics.
        """
        params = {**self._config, **kwargs}
        fs = _extract_sample_rate(params)
        edges = list(data) if isinstance(data, (list, tuple)) else []

        if not edges:
            return []

        # Parameter validation (Bug 8)
        pw_min_samples = int(params.get("pw_min_samples", 3))
        if pw_min_samples < 1:
            raise ValueError(f"pw_min_samples must be >= 1, got {pw_min_samples}")

        pw_min = params.get("pw_min")
        if pw_min is not None:
            pw_min = float(pw_min)
            if pw_min <= 0:
                raise ValueError(f"pw_min must be > 0, got {pw_min}")

        pw_max = params.get("pw_max")
        if pw_max is not None:
            pw_max = float(pw_max)
            if pw_max <= 0:
                raise ValueError(f"pw_max must be > 0, got {pw_max}")

        if pw_min is not None and pw_max is not None and pw_max <= pw_min:
            raise ValueError(f"pw_max ({pw_max}) must be greater than pw_min ({pw_min})")

        pw_max_samples = params.get("pw_max_samples")
        if pw_max_samples is not None:
            pw_max_samples = int(pw_max_samples)
            if pw_max_samples < pw_min_samples:
                raise ValueError(f"pw_max_samples ({pw_max_samples}) must be >= pw_min_samples ({pw_min_samples})")

        min_rise_slope = float(params.get("min_rise_slope", 0.0))
        if min_rise_slope < 0:
            raise ValueError(f"min_rise_slope must be >= 0, got {min_rise_slope}")

        bg_window_sec = float(params.get("background_window_sec", 10e-6))
        if bg_window_sec <= 0:
            raise ValueError(f"background_window_sec must be > 0, got {bg_window_sec}")

        n_bg = max(1, int(round(bg_window_sec * fs)))

        # Effective sample limits
        if pw_min is not None:
            effective_min_samples = max(pw_min_samples, int(math.ceil(pw_min * fs)))
        else:
            effective_min_samples = pw_min_samples

        if pw_max is not None:
            effective_max_samples = int(math.floor(pw_max * fs))
        elif pw_max_samples is not None:
            effective_max_samples = pw_max_samples
        else:
            effective_max_samples = None

        # Guard warning for threshold window vs pulse width (Bug 10)
        th_w = params.get("threshold_window", params.get("window_len"))
        if th_w is not None and effective_max_samples is not None:
            if int(th_w) <= 2 * effective_max_samples:
                warnings.warn(
                    f"threshold_window ({th_w}) is too short relative to maximum expected pulse width ({effective_max_samples} samples); "
                    f"pulses may contaminate the background estimate.",
                    UserWarning,
                    stacklevel=2,
                )

        envelope = params.get("envelope")
        raw_power = params.get("raw_power")
        noise_floor = params.get("noise_floor")
        sigma_n = params.get("sigma_n")
        k_bg = float(params.get("k_bg", params.get("k_low", 3.0)))

        validated_list: list[dict[str, Any]] = []

        for item in edges:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                n_lead, n_trail = int(item[0]), int(item[1])
                rise_slope, fall_slope = 0.0, 0.0
            elif isinstance(item, dict):
                n_lead = int(item.get("n_lead", item.get("start_idx", item.get("prelim_start", 0))))
                n_trail = int(item.get("n_trail", item.get("stop_idx", item.get("prelim_stop", 0))))
                rise_slope = float(item.get("rise_slope", 0.0))
                fall_slope = float(item.get("fall_slope", 0.0))
            else:
                continue

            # BUG 12 Fix: Reject reversed edges immediately
            if n_trail < n_lead:
                validated_list.append({
                    "n_lead": n_lead,
                    "n_trail": n_trail,
                    "pulse_width_sec": 0.0,
                    "pulse_width_samples": 0,
                    "rise_slope": rise_slope,
                    "fall_slope": fall_slope,
                    "is_valid": False,
                    "rejection_reason": "INVALID_EDGE_ORDER",
                })
                continue

            pw_samples = n_trail - n_lead + 1
            pw_sec = float(pw_samples / fs)

            # Evaluate Gating Criteria
            is_valid = True
            rejection_reason = ""

            if pw_samples < effective_min_samples:
                is_valid = False
                rejection_reason = "PW_TOO_SHORT"
            elif effective_max_samples is not None and pw_samples > effective_max_samples:
                is_valid = False
                rejection_reason = "PW_TOO_LONG"
            elif min_rise_slope > 0.0 and (rise_slope < min_rise_slope or fall_slope > -min_rise_slope):
                is_valid = False
                rejection_reason = "INSUFFICIENT_EDGE_SLOPE"

            # Two-Sided Background Isolation Check (Bugs 17 & 18)
            pwr_src = raw_power if raw_power is not None else envelope
            if is_valid and pwr_src is not None and noise_floor is not None:
                env_arr = np.asarray(pwr_src, dtype=np.float64).ravel()
                nf_arr = np.asarray(noise_floor, dtype=np.float64).ravel()
                sig_arr = np.asarray(sigma_n, dtype=np.float64).ravel() if sigma_n is not None else None
                smooth_l = int(params.get("smooth_len", 1))

                pre_start = n_lead - n_bg
                post_start = n_trail + 1 + max(0, smooth_l - 1)
                post_stop = post_start + n_bg

                # Only evaluate isolation if complete pre and post baseline windows exist within buffer
                if pre_start >= 0 and post_stop <= len(env_arr):
                    pre_segment = env_arr[pre_start:n_lead]
                    post_segment = env_arr[post_start:post_stop]

                    local_nf = float(nf_arr[n_lead]) if n_lead < len(nf_arr) else float(np.median(nf_arr))
                    local_sigma = float(sig_arr[n_lead]) if sig_arr is not None and n_lead < len(sig_arr) else local_nf * 1.5
                    t_bg = local_nf + k_bg * max(local_sigma, local_nf * 0.5)

                    pre_pwr = float(np.mean(pre_segment))
                    post_pwr = float(np.mean(post_segment))

                    pre_ok = pre_pwr <= t_bg
                    post_ok = post_pwr <= t_bg

                    if not (pre_ok and post_ok):
                        is_valid = False
                        rejection_reason = "CONTINUOUS_ACTIVITY_NO_ISOLATION"

            validated_list.append({
                "n_lead": n_lead,
                "n_trail": n_trail,
                "pulse_width_sec": pw_sec,
                "pulse_width_samples": pw_samples,
                "rise_slope": rise_slope,
                "fall_slope": fall_slope,
                "is_valid": is_valid,
                "rejection_reason": rejection_reason,
            })

        return validated_list


def evaluate_pulse_width(
    edges: list[Any],
    sample_rate: float,
    pw_min: float | None = None,
    pw_max: float | None = None,
    pw_min_samples: int = 3,
    pw_max_samples: int | None = None,
    min_rise_slope: float = 0.0,
    background_window_sec: float = 10e-6,
    **kwargs: Any
) -> list[dict[str, Any]]:
    """Functional wrapper for PulseWidthStage."""
    stage = PulseWidthStage(
        sample_rate=sample_rate,
        pw_min=pw_min,
        pw_max=pw_max,
        pw_min_samples=pw_min_samples,
        pw_max_samples=pw_max_samples,
        min_rise_slope=min_rise_slope,
        background_window_sec=background_window_sec,
        **kwargs
    )
    return stage.compute(edges)


def main() -> None:
    print("=" * 60)
    print("  PulseWidthStage: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = PulseWidthStage()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
