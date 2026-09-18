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
from .base import PulseCandidate, PulseDetectorFactory
from ._common import _as_iq_array, _validate_sample_rate, _causal_moving_average


class StreamingPulseDetector:
    """
    Stateful Real-Time Streaming Pulse Detector.
    Processes continuous streams of IQ sample blocks across chunk boundaries without splitting
    straddled pulses or losing detection/filter state:
      - Preserves causal moving-average envelope memory
      - Preserves hysteresis state machine status
      - Buffers active pulse IQ and power samples across buffer boundaries with full pre/post context
      - Assigns continuous, monotonic global absolute timestamps and sample indices
    """

    def __init__(
        self,
        sample_rate: float | None = None,
        envelope_mode: str = "power",
        smooth_len: int = 1,
        threshold_window: int = 101,
        k_high: float = 5.0,
        k_low: float = 3.0,
        edge_search_window: int = 8,
        pw_min: float | None = None,
        pw_max: float | None = None,
        pw_min_samples: int = 3,
        pw_max_samples: int | None = None,
        start_confirm_samples: int = 2,
        end_confirm_samples: int = 5,
        min_rise_slope: float = 0.0,
        background_window_sec: float = 10e-6,
        include_rejected: bool = False,
    ) -> None:
        # BUG 6 Fix: sample_rate must be explicitly provided
        if sample_rate is None:
            raise ValueError("sample_rate must be explicitly provided.")
        self.sample_rate = _validate_sample_rate(sample_rate)

        mode = str(envelope_mode).strip().lower()
        if mode not in {"power"}:
            raise ValueError(f"Unsupported envelope mode: '{mode}'. Pulse detector requires envelope_mode='power'.")
        self.envelope_mode = mode

        # BUG 8 Fix: Configuration validation
        self.smooth_len = int(smooth_len)
        if self.smooth_len < 1:
            raise ValueError(f"smooth_len must be >= 1, got {self.smooth_len}")

        self.threshold_window = int(threshold_window)
        if self.threshold_window < 1:
            raise ValueError(f"threshold_window must be >= 1, got {self.threshold_window}")

        self.edge_search_window = int(edge_search_window)
        if self.edge_search_window < 1:
            raise ValueError(f"edge_search_window must be >= 1, got {self.edge_search_window}")

        self.pw_min_samples = int(pw_min_samples)
        if self.pw_min_samples < 1:
            raise ValueError(f"pw_min_samples must be >= 1, got {self.pw_min_samples}")

        self.start_confirm_samples = int(start_confirm_samples)
        if self.start_confirm_samples < 1:
            raise ValueError(f"start_confirm_samples must be >= 1, got {self.start_confirm_samples}")

        self.end_confirm_samples = int(end_confirm_samples)
        if self.end_confirm_samples < 1:
            raise ValueError(f"end_confirm_samples must be >= 1, got {self.end_confirm_samples}")

        self.k_high = float(k_high)
        self.k_low = float(k_low)
        if not np.isfinite(self.k_high) or not np.isfinite(self.k_low):
            raise ValueError("Threshold multipliers must be finite")
        if self.k_low < 0:
            raise ValueError(f"k_low must be >= 0, got {self.k_low}")
        if self.k_high <= self.k_low:
            raise ValueError(f"k_high ({self.k_high}) must be greater than k_low ({self.k_low})")

        self.pw_min = float(pw_min) if pw_min is not None else None
        if self.pw_min is not None and self.pw_min <= 0:
            raise ValueError(f"pw_min must be > 0, got {self.pw_min}")

        self.pw_max = float(pw_max) if pw_max is not None else None
        if self.pw_max is not None and self.pw_max <= 0:
            raise ValueError(f"pw_max must be > 0, got {self.pw_max}")

        if self.pw_min is not None and self.pw_max is not None and self.pw_max <= self.pw_min:
            raise ValueError(f"pw_max ({self.pw_max}) must be greater than pw_min ({self.pw_min})")

        self.pw_max_samples = int(pw_max_samples) if pw_max_samples is not None else None
        if self.pw_max_samples is not None and self.pw_max_samples < self.pw_min_samples:
            raise ValueError(f"pw_max_samples ({self.pw_max_samples}) must be >= pw_min_samples ({self.pw_min_samples})")

        self.min_rise_slope = float(min_rise_slope)
        if self.min_rise_slope < 0:
            raise ValueError(f"min_rise_slope must be >= 0, got {self.min_rise_slope}")

        self.background_window_sec = float(background_window_sec)
        if self.background_window_sec <= 0:
            raise ValueError(f"background_window_sec must be > 0, got {self.background_window_sec}")

        self.include_rejected = bool(include_rejected)

        # BUG 10 Fix: Threshold window vs max pulse width guard warning
        effective_max_samples = self.pw_max_samples
        if effective_max_samples is None and self.pw_max is not None:
            effective_max_samples = int(math.floor(self.pw_max * self.sample_rate))
        if effective_max_samples is not None and self.threshold_window <= 2 * effective_max_samples:
            warnings.warn(
                f"threshold_window ({self.threshold_window}) is too short relative to maximum expected pulse width ({effective_max_samples} samples); "
                f"pulses may contaminate the background estimate.",
                UserWarning,
                stacklevel=2,
            )

        # Context length required for edge refinement and background isolation (Bugs 4 & 5)
        n_bg = max(1, int(round(self.background_window_sec * self.sample_rate)))
        self.context_samples = max(self.edge_search_window + 1, n_bg, self.smooth_len)

        # Stage instances
        self.stage_th = PulseDetectorFactory.create("threshold_detection", window_len=self.threshold_window, k_high=self.k_high, k_low=self.k_low)
        self.stage_detect = PulseDetectorFactory.create(
            "pulse_detection",
            start_confirm_samples=self.start_confirm_samples,
            end_confirm_samples=self.end_confirm_samples,
        )
        self.stage_bound = PulseDetectorFactory.create("pulse_boundaries")
        self.stage_edges = PulseDetectorFactory.create("leading_trailing_edge_detection", search_window=self.edge_search_window, sample_rate=self.sample_rate)
        self.stage_width = PulseDetectorFactory.create(
            "pulse_width",
            sample_rate=self.sample_rate,
            pw_min=self.pw_min,
            pw_max=self.pw_max,
            pw_min_samples=self.pw_min_samples,
            pw_max_samples=self.pw_max_samples,
            min_rise_slope=self.min_rise_slope,
            background_window_sec=self.background_window_sec
        )
        self.stage_seg = PulseDetectorFactory.create("pulse_segmentation", sample_rate=self.sample_rate, include_rejected=self.include_rejected)



        # Persistent streaming state
        self.absolute_sample_index: int = 0
        self.next_candidate_id: int = 1
        self.ma_history: np.ndarray | None = None
        self.env_history: np.ndarray | None = None

        # Ring buffers for preceding samples (for pre-context extraction)
        self.recent_iq_history: np.ndarray = np.empty(0, dtype=np.complex128)
        self.recent_raw_power_history: np.ndarray = np.empty(0, dtype=np.float64)
        self.recent_env_history: np.ndarray = np.empty(0, dtype=np.float64)
        self.recent_nf_history: np.ndarray = np.empty(0, dtype=np.float64)
        self.recent_sigma_history: np.ndarray = np.empty(0, dtype=np.float64)

        # Open pulse buffers (accumulates across chunks)
        self.open_pulse_start_abs: int | None = None
        self.open_pulse_pre_context_len: int = 0
        self.open_pulse_iq: list[np.ndarray] = []
        self.open_pulse_raw_power: list[np.ndarray] = []
        self.open_pulse_env: list[np.ndarray] = []
        self.open_pulse_nf: list[np.ndarray] = []
        self.open_pulse_sigma: list[np.ndarray] = []

        # Pending finalization pulse (holding for post-context) (Bug 5)
        self.pending_pulse_dict: dict[str, Any] | None = None

    def reset(self) -> None:
        """Resets all streaming buffers, filter history, and ongoing pulse states."""
        self.absolute_sample_index = 0
        self.next_candidate_id = 1
        self.ma_history = None
        self.env_history = None
        self.stage_detect.reset()
        self.stage_bound.reset()
        self.recent_iq_history = np.empty(0, dtype=np.complex128)
        self.recent_raw_power_history = np.empty(0, dtype=np.float64)
        self.recent_env_history = np.empty(0, dtype=np.float64)
        self.recent_nf_history = np.empty(0, dtype=np.float64)
        self.recent_sigma_history = np.empty(0, dtype=np.float64)
        self.open_pulse_start_abs = None
        self.open_pulse_pre_context_len = 0
        self.open_pulse_iq.clear()
        self.open_pulse_raw_power.clear()
        self.open_pulse_env.clear()
        self.open_pulse_nf.clear()
        self.open_pulse_sigma.clear()
        self.pending_pulse_dict = None

    def _finalize_pulse_with_context(
        self,
        full_iq: np.ndarray,
        full_raw_power: np.ndarray,
        full_env: np.ndarray,
        full_nf: np.ndarray,
        full_sigma: np.ndarray,
        pulse_start_abs: int,
        pre_context_len: int,
        pulse_body_len: int,
    ) -> list[PulseCandidate]:
        """
        Refines, validates, and segments a pulse event with full pre-context and post-context.
        """
        if pulse_body_len <= 0 or len(full_iq) == 0:
            return []

        # Boundary relative to full context buffer
        start_in_full = pre_context_len
        stop_in_full = pre_context_len + pulse_body_len - 1
        boundary = [(start_in_full, stop_in_full)]

        refined = self.stage_edges.compute(
            full_env,
            boundaries=boundary,
            raw_power=full_raw_power,
            sample_rate=self.sample_rate
        )

        validated = self.stage_width.compute(
            refined,
            sample_rate=self.sample_rate,
            envelope=full_env,
            raw_power=full_raw_power,
            noise_floor=full_nf,
            sigma_n=full_sigma,
            smooth_len=self.smooth_len
        )

        if not validated:
            return []

        val = validated[0]
        ref_n_lead = val["n_lead"]
        ref_n_trail = val["n_trail"]
        is_valid = val["is_valid"]
        rejection = val["rejection_reason"]
        rise_slope = val["rise_slope"]
        fall_slope = val["fall_slope"]

        if not is_valid and not self.include_rejected:
            return []

        # Convert back to absolute indices
        abs_start = pulse_start_abs + (ref_n_lead - pre_context_len)
        abs_stop = pulse_start_abs + (ref_n_trail - pre_context_len)
        toa_sec = float(abs_start / self.sample_rate)

        # Slice pulse IQ
        if is_valid and ref_n_trail >= ref_n_lead:
            clamped_lead = max(0, min(ref_n_lead, len(full_iq) - 1))
            clamped_trail = max(clamped_lead, min(ref_n_trail, len(full_iq) - 1))
            iq_slice = full_iq[clamped_lead : clamped_trail + 1].copy()
            pw_samples = ref_n_trail - ref_n_lead + 1
            pw_sec = float(pw_samples / self.sample_rate)
            mean_pwr = float(np.mean(np.abs(iq_slice) ** 2)) if iq_slice.size > 0 else 0.0
            env_slice = full_env[clamped_lead : clamped_trail + 1]
            peak_env = float(np.max(env_slice)) if env_slice.size > 0 else 0.0
            nf_val = float(full_nf[clamped_lead]) if clamped_lead < len(full_nf) else float(np.median(full_nf))
            nf_val = max(1e-18, nf_val)
            nf_db = float(10.0 * np.log10(nf_val))
            signal_power = mean_pwr - nf_val
            snr_db = float(10.0 * np.log10(signal_power / nf_val)) if signal_power > 0 and nf_val > 0 else float("-inf")
        else:
            iq_slice = np.empty(0, dtype=np.complex128)
            pw_samples = 0
            pw_sec = 0.0
            mean_pwr = 0.0
            peak_env = 0.0
            nf_val = 0.0
            nf_db = float("-inf")
            snr_db = float("-inf")

        candidate = PulseCandidate(
            candidate_id=self.next_candidate_id,
            start_idx=ref_n_lead - pre_context_len,
            stop_idx=ref_n_trail - pre_context_len,
            start_idx_absolute=abs_start,
            stop_idx_absolute=abs_stop,
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
        self.next_candidate_id += 1
        return [candidate]

    def process(self, iq_chunk: Any, final_chunk: bool = False) -> list[PulseCandidate]:
        """
        Processes a block/chunk of complex IQ data in real-time streaming mode.

        Parameters:
            iq_chunk: 1D complex IQ array for the current chunk.
            final_chunk: Set to True if this is the final block in stream/file to flush any open pulse.

        Returns:
            List of completed PulseCandidate objects finalized in this chunk.
        """
        z = _as_iq_array(iq_chunk)
        if z.size == 0:
            return []

        chunk_len = len(z)
        chunk_offset = self.absolute_sample_index
        completed_candidates: list[PulseCandidate] = []

        # 1. Raw power & Causal Moving Average
        raw_power = np.abs(z) ** 2
        if self.smooth_len > 1:
            detection_envelope, self.ma_history = _causal_moving_average(raw_power, self.smooth_len, history=self.ma_history)
        else:
            detection_envelope = raw_power.copy()

        # 2. Threshold Detection with continuous multi-chunk background history
        if self.env_history is not None and len(self.env_history) > 0:
            ext_env = np.concatenate([self.env_history, detection_envelope])
            th_dict = self.stage_th.compute(ext_env, window_len=self.threshold_window, k_high=self.k_high, k_low=self.k_low)
            hist_len = len(self.env_history)
            th_high = th_dict["th_high"][hist_len:]
            th_low = th_dict["th_low"][hist_len:]
            noise_floor = th_dict["noise_floor"][hist_len:]
            sigma_n = th_dict["sigma_n"][hist_len:]
            self.env_history = ext_env[-self.threshold_window:].copy()
        else:
            th_dict = self.stage_th.compute(detection_envelope, window_len=self.threshold_window, k_high=self.k_high, k_low=self.k_low)
            th_high = th_dict["th_high"]
            th_low = th_dict["th_low"]
            noise_floor = th_dict["noise_floor"]
            sigma_n = th_dict["sigma_n"]
            self.env_history = detection_envelope[-self.threshold_window:].copy()

        # 3. Complete pending post-context finalization if any (Bug 5)
        if self.pending_pulse_dict is not None:
            pend = self.pending_pulse_dict
            needed = pend["needed_post_samples"]
            avail_take = min(chunk_len, needed)
            pend["post_iq"].append(z[:avail_take])
            pend["post_raw_power"].append(raw_power[:avail_take])
            pend["post_env"].append(detection_envelope[:avail_take])
            pend["post_nf"].append(noise_floor[:avail_take])
            pend["post_sigma"].append(sigma_n[:avail_take])
            pend["needed_post_samples"] -= avail_take

            if pend["needed_post_samples"] <= 0 or final_chunk:
                # Post context collection complete -> finalize pulse
                full_iq = np.concatenate([pend["pre_iq"], pend["body_iq"]] + pend["post_iq"])
                full_raw_pwr = np.concatenate([pend["pre_raw_power"], pend["body_raw_power"]] + pend["post_raw_power"])
                full_env = np.concatenate([pend["pre_env"], pend["body_env"]] + pend["post_env"])
                full_nf = np.concatenate([pend["pre_nf"], pend["body_nf"]] + pend["post_nf"])
                full_sig = np.concatenate([pend["pre_sigma"], pend["body_sigma"]] + pend["post_sigma"])

                cands = self._finalize_pulse_with_context(
                    full_iq,
                    full_raw_pwr,
                    full_env,
                    full_nf,
                    full_sig,
                    pend["pulse_start_abs"],
                    pend["pre_context_len"],
                    pend["pulse_body_len"],
                )
                completed_candidates.extend(cands)
                self.pending_pulse_dict = None

        # 4. Pulse Detection (Hysteresis state machine with persistent debouncing counters)
        was_in_pulse = self.stage_detect.in_pulse_state
        pulse_mask = self.stage_detect.compute(detection_envelope, th_high=th_high, th_low=th_low)


        # 5. Pulse Boundaries
        bound_res = self.stage_bound.compute(
            pulse_mask,
            final_chunk=final_chunk,
            initial_in_pulse=was_in_pulse,
            as_dict=True
        )
        closed_boundaries = bound_res["boundaries"]
        has_open_pulse = bound_res["has_open_pulse"]
        open_start_rel = bound_res["open_pulse_start"]

        # 6. Process closed boundaries in this chunk
        for b_idx, (start_rel, stop_rel) in enumerate(closed_boundaries):
            if start_rel is None:
                # Straddled pulse closing in this chunk (or ending at chunk boundary)
                if stop_rel >= 0:
                    self.open_pulse_iq.append(z[: stop_rel + 1])
                    self.open_pulse_raw_power.append(raw_power[: stop_rel + 1])
                    self.open_pulse_env.append(detection_envelope[: stop_rel + 1])
                    self.open_pulse_nf.append(noise_floor[: stop_rel + 1])
                    self.open_pulse_sigma.append(sigma_n[: stop_rel + 1])

                body_iq = np.concatenate(self.open_pulse_iq) if self.open_pulse_iq else np.empty(0, dtype=np.complex128)
                body_raw_pwr = np.concatenate(self.open_pulse_raw_power) if self.open_pulse_raw_power else np.empty(0, dtype=np.float64)
                body_env = np.concatenate(self.open_pulse_env) if self.open_pulse_env else np.empty(0, dtype=np.float64)
                body_nf = np.concatenate(self.open_pulse_nf) if self.open_pulse_nf else np.empty(0, dtype=np.float64)
                body_sig = np.concatenate(self.open_pulse_sigma) if self.open_pulse_sigma else np.empty(0, dtype=np.float64)

                pulse_start_abs = self.open_pulse_start_abs if self.open_pulse_start_abs is not None else chunk_offset
                pre_context_len = self.open_pulse_pre_context_len
                pulse_body_len = max(0, len(body_iq) - pre_context_len)


                # Collect available post-context
                post_start_idx = max(0, stop_rel + 1)
                available_post = max(0, chunk_len - post_start_idx)
                take_post = min(available_post, self.context_samples)

                post_iq = [z[post_start_idx : post_start_idx + take_post]] if take_post > 0 else []
                post_raw_pwr = [raw_power[post_start_idx : post_start_idx + take_post]] if take_post > 0 else []
                post_env = [detection_envelope[post_start_idx : post_start_idx + take_post]] if take_post > 0 else []
                post_nf = [noise_floor[post_start_idx : post_start_idx + take_post]] if take_post > 0 else []
                post_sig = [sigma_n[post_start_idx : post_start_idx + take_post]] if take_post > 0 else []

                # Reset open pulse tracking
                self.open_pulse_start_abs = None
                self.open_pulse_pre_context_len = 0
                self.open_pulse_iq.clear()
                self.open_pulse_raw_power.clear()
                self.open_pulse_env.clear()
                self.open_pulse_nf.clear()
                self.open_pulse_sigma.clear()
                was_in_pulse = False

                if take_post >= self.context_samples or final_chunk:
                    # Finalize immediately
                    full_iq = np.concatenate([body_iq] + post_iq) if post_iq else body_iq
                    full_raw_pwr = np.concatenate([body_raw_pwr] + post_raw_pwr) if post_raw_pwr else body_raw_pwr
                    full_env = np.concatenate([body_env] + post_env) if post_env else body_env
                    full_nf = np.concatenate([body_nf] + post_nf) if post_nf else body_nf
                    full_sig = np.concatenate([body_sig] + post_sig) if post_sig else body_sig

                    cands = self._finalize_pulse_with_context(
                        full_iq,
                        full_raw_pwr,
                        full_env,
                        full_nf,
                        full_sig,
                        pulse_start_abs,
                        pre_context_len,
                        pulse_body_len,
                    )
                    completed_candidates.extend(cands)
                else:
                    # Hold for remaining post-context in next chunk (Bug 5)
                    self.pending_pulse_dict = {
                        "pre_iq": np.empty(0, dtype=np.complex128),
                        "body_iq": body_iq,
                        "post_iq": post_iq,
                        "pre_raw_power": np.empty(0, dtype=np.float64),
                        "body_raw_power": body_raw_pwr,
                        "post_raw_power": post_raw_pwr,
                        "pre_env": np.empty(0, dtype=np.float64),
                        "body_env": body_env,
                        "post_env": post_env,
                        "pre_nf": np.empty(0, dtype=np.float64),
                        "body_nf": body_nf,
                        "post_nf": post_nf,
                        "pre_sigma": np.empty(0, dtype=np.float64),
                        "body_sigma": body_sig,
                        "post_sigma": post_sig,
                        "pulse_start_abs": pulse_start_abs,
                        "pre_context_len": pre_context_len,
                        "pulse_body_len": pulse_body_len,
                        "needed_post_samples": self.context_samples - take_post,
                    }

            else:
                # Contained entirely within this chunk (or starting in this chunk)
                pulse_start_abs = chunk_offset + start_rel
                pulse_body_len = stop_rel - start_rel + 1

                # Extract pre-context (up to context_samples before start_rel)
                if start_rel >= self.context_samples:
                    pre_slice_start = start_rel - self.context_samples
                    pre_iq = z[pre_slice_start : start_rel]
                    pre_raw_pwr = raw_power[pre_slice_start : start_rel]
                    pre_env = detection_envelope[pre_slice_start : start_rel]
                    pre_nf = noise_floor[pre_slice_start : start_rel]
                    pre_sig = sigma_n[pre_slice_start : start_rel]
                else:
                    needed_from_recent = self.context_samples - start_rel
                    avail_from_recent = min(len(self.recent_iq_history), needed_from_recent)
                    if avail_from_recent > 0:
                        hist_iq = self.recent_iq_history[-avail_from_recent:]
                        hist_raw_pwr = self.recent_raw_power_history[-avail_from_recent:]
                        hist_env = self.recent_env_history[-avail_from_recent:]
                        hist_nf = self.recent_nf_history[-avail_from_recent:]
                        hist_sig = self.recent_sigma_history[-avail_from_recent:]
                        pre_iq = np.concatenate([hist_iq, z[:start_rel]])
                        pre_raw_pwr = np.concatenate([hist_raw_pwr, raw_power[:start_rel]])
                        pre_env = np.concatenate([hist_env, detection_envelope[:start_rel]])
                        pre_nf = np.concatenate([hist_nf, noise_floor[:start_rel]])
                        pre_sig = np.concatenate([hist_sig, sigma_n[:start_rel]])
                    else:
                        pre_iq = z[:start_rel]
                        pre_raw_pwr = raw_power[:start_rel]
                        pre_env = detection_envelope[:start_rel]
                        pre_nf = noise_floor[:start_rel]
                        pre_sig = sigma_n[:start_rel]

                pre_context_len = len(pre_iq)
                body_iq = z[start_rel : stop_rel + 1]
                body_raw_pwr = raw_power[start_rel : stop_rel + 1]
                body_env = detection_envelope[start_rel : stop_rel + 1]
                body_nf = noise_floor[start_rel : stop_rel + 1]
                body_sig = sigma_n[start_rel : stop_rel + 1]

                # Available post-context in this chunk
                post_start_idx = stop_rel + 1
                available_post = max(0, chunk_len - post_start_idx)
                take_post = min(available_post, self.context_samples)

                post_iq = [z[post_start_idx : post_start_idx + take_post]] if take_post > 0 else []
                post_raw_pwr = [raw_power[post_start_idx : post_start_idx + take_post]] if take_post > 0 else []
                post_env = [detection_envelope[post_start_idx : post_start_idx + take_post]] if take_post > 0 else []
                post_nf = [noise_floor[post_start_idx : post_start_idx + take_post]] if take_post > 0 else []
                post_sig = [sigma_n[post_start_idx : post_start_idx + take_post]] if take_post > 0 else []

                if take_post >= self.context_samples or final_chunk:
                    full_iq = np.concatenate([pre_iq, body_iq] + post_iq)
                    full_raw_pwr = np.concatenate([pre_raw_pwr, body_raw_pwr] + post_raw_pwr)
                    full_env = np.concatenate([pre_env, body_env] + post_env)
                    full_nf = np.concatenate([pre_nf, body_nf] + post_nf)
                    full_sig = np.concatenate([pre_sig, body_sig] + post_sig)

                    cands = self._finalize_pulse_with_context(
                        full_iq,
                        full_raw_pwr,
                        full_env,
                        full_nf,
                        full_sig,
                        pulse_start_abs,
                        pre_context_len,
                        pulse_body_len,
                    )
                    completed_candidates.extend(cands)
                else:
                    # Hold for remaining post-context in next chunk (Bug 5)
                    self.pending_pulse_dict = {
                        "pre_iq": pre_iq,
                        "body_iq": body_iq,
                        "post_iq": post_iq,
                        "pre_raw_power": pre_raw_pwr,
                        "body_raw_power": body_raw_pwr,
                        "post_raw_power": post_raw_pwr,
                        "pre_env": pre_env,
                        "body_env": body_env,
                        "post_env": post_env,
                        "pre_nf": pre_nf,
                        "body_nf": body_nf,
                        "post_nf": post_nf,
                        "pre_sigma": pre_sig,
                        "body_sigma": body_sig,
                        "post_sigma": post_sig,
                        "pulse_start_abs": pulse_start_abs,
                        "pre_context_len": pre_context_len,
                        "pulse_body_len": pulse_body_len,
                        "needed_post_samples": self.context_samples - take_post,
                    }

        # 7. Handle trailing open pulse (Bugs 2 & 4)
        if has_open_pulse:
            if open_start_rel is not None:
                # Pulse started in this chunk at open_start_rel
                abs_start = chunk_offset + open_start_rel
                self.open_pulse_start_abs = abs_start

                # Extract pre-context (up to context_samples)
                if open_start_rel >= self.context_samples:
                    pre_slice_start = open_start_rel - self.context_samples
                    pre_iq = z[pre_slice_start : open_start_rel]
                    pre_raw_pwr = raw_power[pre_slice_start : open_start_rel]
                    pre_env = detection_envelope[pre_slice_start : open_start_rel]
                    pre_nf = noise_floor[pre_slice_start : open_start_rel]
                    pre_sig = sigma_n[pre_slice_start : open_start_rel]
                else:
                    needed = self.context_samples - open_start_rel
                    avail = min(len(self.recent_iq_history), needed)
                    if avail > 0:
                        pre_iq = np.concatenate([self.recent_iq_history[-avail:], z[:open_start_rel]])
                        pre_raw_pwr = np.concatenate([self.recent_raw_power_history[-avail:], raw_power[:open_start_rel]])
                        pre_env = np.concatenate([self.recent_env_history[-avail:], detection_envelope[:open_start_rel]])
                        pre_nf = np.concatenate([self.recent_nf_history[-avail:], noise_floor[:open_start_rel]])
                        pre_sig = np.concatenate([self.recent_sigma_history[-avail:], sigma_n[:open_start_rel]])
                    else:
                        pre_iq = z[:open_start_rel]
                        pre_raw_pwr = raw_power[:open_start_rel]
                        pre_env = detection_envelope[:open_start_rel]
                        pre_nf = noise_floor[:open_start_rel]
                        pre_sig = sigma_n[:open_start_rel]

                self.open_pulse_pre_context_len = len(pre_iq)
                self.open_pulse_iq = [pre_iq, z[open_start_rel:].copy()]
                self.open_pulse_raw_power = [pre_raw_pwr, raw_power[open_start_rel:].copy()]
                self.open_pulse_env = [pre_env, detection_envelope[open_start_rel:].copy()]
                self.open_pulse_nf = [pre_nf, noise_floor[open_start_rel:].copy()]
                self.open_pulse_sigma = [pre_sig, sigma_n[open_start_rel:].copy()]
            elif was_in_pulse:
                # BUG 2 Fix: Pulse started in an earlier chunk and continues across this entire chunk
                # Do NOT overwrite open_pulse_start_abs!
                self.open_pulse_iq.append(z.copy())
                self.open_pulse_raw_power.append(raw_power.copy())
                self.open_pulse_env.append(detection_envelope.copy())
                self.open_pulse_nf.append(noise_floor.copy())
                self.open_pulse_sigma.append(sigma_n.copy())
        else:
            self.open_pulse_start_abs = None
            self.open_pulse_pre_context_len = 0
            self.open_pulse_iq.clear()
            self.open_pulse_raw_power.clear()
            self.open_pulse_env.clear()
            self.open_pulse_nf.clear()
            self.open_pulse_sigma.clear()

        # Update recent sample histories
        save_hist = min(chunk_len, self.context_samples)
        self.recent_iq_history = z[-save_hist:].copy()
        self.recent_raw_power_history = raw_power[-save_hist:].copy()
        self.recent_env_history = detection_envelope[-save_hist:].copy()
        self.recent_nf_history = noise_floor[-save_hist:].copy()
        self.recent_sigma_history = sigma_n[-save_hist:].copy()

        self.absolute_sample_index += chunk_len
        return completed_candidates


def main() -> None:
    print("=" * 60)
    print("  Pulse Detector Streaming: Demonstration")
    print("=" * 60)
    print("Streaming engine loaded successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
