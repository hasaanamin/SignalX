"""Event capture assembly and timeline retention management."""

from __future__ import annotations

import numpy as np

from signal_processing.buffering.ring_buffer import CircularIQBuffer
from signal_processing.capture.sidecar import CaptureSidecar
from signal_processing.core.contracts import SignalEvent, StreamInfo, _validate_and_freeze_iq_array


class EventAssembler:
    """Collects and packages detected signal events from circular storage into SignalEvents."""

    def __init__(
        self,
        ring_buffer: CircularIQBuffer,
        pre_trigger_samples: int = 512,
        pre_signal_margin_samples: int = 256,
        post_signal_samples: int = 256,
        min_post_trigger_samples: int = 512,
        max_event_samples: int = 100_000,
        config_version: str = "2.0",
        window_size: int = 0,
        hop_size: int = 0,
        start_persistence: int = 1,
    ) -> None:
        if isinstance(pre_trigger_samples, bool) or not isinstance(pre_trigger_samples, (int, np.integer)) or pre_trigger_samples < 0:
            raise ValueError(f"pre_trigger_samples must be >= 0, got {pre_trigger_samples!r}")
        if isinstance(pre_signal_margin_samples, bool) or not isinstance(pre_signal_margin_samples, (int, np.integer)) or pre_signal_margin_samples < 0:
            raise ValueError(f"pre_signal_margin_samples must be >= 0, got {pre_signal_margin_samples!r}")
        if isinstance(post_signal_samples, bool) or not isinstance(post_signal_samples, (int, np.integer)) or post_signal_samples < 0:
            raise ValueError(f"post_signal_samples must be >= 0, got {post_signal_samples!r}")
        if isinstance(min_post_trigger_samples, bool) or not isinstance(min_post_trigger_samples, (int, np.integer)) or min_post_trigger_samples < 0:
            raise ValueError(f"min_post_trigger_samples must be >= 0, got {min_post_trigger_samples!r}")
        if isinstance(max_event_samples, bool) or not isinstance(max_event_samples, (int, np.integer)) or max_event_samples < 2:
            raise ValueError(f"max_event_samples must be >= 2, got {max_event_samples!r}")

        if (
            pre_trigger_samples >= max_event_samples
            or pre_signal_margin_samples >= max_event_samples
            or min_post_trigger_samples >= max_event_samples
        ):
            raise ValueError(
                f"Incompatible capture configuration: max_event_samples ({max_event_samples}) "
                f"must be strictly greater than requested margins (pre_trigger={pre_trigger_samples}, "
                f"pre_signal_margin={pre_signal_margin_samples}, min_post_trigger={min_post_trigger_samples})"
            )

        if window_size > 0:
            k_on = max(1, int(start_persistence))
            h = max(1, int(hop_size))
            min_span = max(
                pre_trigger_samples + 1,
                (k_on - 1) * h + int(window_size) + pre_signal_margin_samples,
            )
            if max_event_samples < min_span:
                raise ValueError(
                    f"Incompatible capture configuration: max_event_samples ({max_event_samples}) "
                    f"must be at least {min_span} samples to span detection window/margins"
                )

        self._ring = ring_buffer
        self._pre_trigger_samples = int(pre_trigger_samples)
        self._pre_signal_margin_samples = int(pre_signal_margin_samples)
        self._post_signal_samples = int(post_signal_samples)
        self._min_post_trigger_samples = int(min_post_trigger_samples)
        self._max_event_samples = int(max_event_samples)
        self._config_version = str(config_version)

        self._event_counter: int = 0
        self._in_flight: bool = False

        # In-flight event metadata
        self._event_id: str = ""
        self._stream_info: StreamInfo | None = None
        self._stream_epoch: int = 0
        self._first_on_sample: int = 0
        self._confirmation_sample: int = 0
        self._trigger_sample: int = 0
        self._detected_start_sample: int = 0
        self._requested_event_start: int = 0
        self._actual_event_start: int = 0
        self._is_leading_truncated: bool = False

        self._detected_end_sample: int | None = None
        self._end_confirmation_sample: int | None = None
        self._target_end_sample: int | None = None
        self._requested_target_end_sample: int | None = None
        self._collected_end_sample: int = 0
        self._assembled_chunks: list[np.ndarray] = []

        self._noise_provenance: str = ""
        self._noise_valid: bool = True
        self._boundary_uncertainty: int | None = None
        self._detector_mode: str = "full_band"
        self._monitored_band_hz: tuple[float, float] | None = None

        self._anchor_start_sample: int | None = None
        self._anchor_timestamp_ns: int | None = None
        self._known_lost_samples: int = 0
        self._unknown_loss: bool = False
        self._max_duration_reached: bool = False
        self._reopened_count: int = 0

    @property
    def is_in_flight(self) -> bool:
        return self._in_flight

    @property
    def target_end_sample(self) -> int | None:
        return self._target_end_sample

    @property
    def collected_end_sample(self) -> int:
        return self._collected_end_sample

    @property
    def reopened_count(self) -> int:
        return self._reopened_count

    def set_timestamp_anchor(self, start_sample: int, timestamp_ns: int | None) -> None:
        if timestamp_ns is not None:
            self._anchor_start_sample = int(start_sample)
            self._anchor_timestamp_ns = int(timestamp_ns)
        else:
            self._anchor_start_sample = None
            self._anchor_timestamp_ns = None

    def start_capture(
        self,
        stream_info: StreamInfo,
        first_on_sample: int,
        confirmation_sample: int,
        trigger_sample: int,
        detected_start_sample: int,
        detector_mode: str,
        monitored_band_hz: tuple[float, float] | None,
        noise_provenance: str,
        noise_valid: bool,
        boundary_uncertainty: int | None = None,
        decision_horizon: int | None = None,
    ) -> None:
        """Initialize active event capture when segmenter triggers START."""
        self._event_counter += 1
        self._event_id = f"evt_{stream_info.stream_id}_{self._event_counter:04d}"
        self._stream_info = stream_info
        self._stream_epoch = self._ring.stream_generation

        self._first_on_sample = int(first_on_sample)
        self._confirmation_sample = int(confirmation_sample)
        self._trigger_sample = int(trigger_sample)
        self._detected_start_sample = int(detected_start_sample)
        self._detector_mode = str(detector_mode)
        self._monitored_band_hz = monitored_band_hz
        self._noise_provenance = str(noise_provenance)
        self._noise_valid = bool(noise_valid)
        self._boundary_uncertainty = int(boundary_uncertainty) if boundary_uncertainty is not None else None

        self._detected_end_sample = None
        self._end_confirmation_sample = None
        self._target_end_sample = None
        self._requested_target_end_sample = None
        self._max_duration_reached = False
        self._known_lost_samples = 0
        self._unknown_loss = False
        self._assembled_chunks = []
        self._reopened_count = 0

        # Calculate requested leading extent
        req_start = min(
            self._trigger_sample - self._pre_trigger_samples,
            self._detected_start_sample - self._pre_signal_margin_samples,
        )
        self._requested_event_start = req_start

        # Clip history using current generation start / ring oldest sample
        ring_oldest, ring_next = self._ring.available_range()
        min_avail = max(0, ring_oldest)
        earliest_allowed = self._trigger_sample - (self._max_event_samples - 1)
        act_start = max(req_start, earliest_allowed, min_avail)
        self._is_leading_truncated = (act_start > req_start)
        self._actual_event_start = act_start

        # Fetch initial history bounded by logical decision horizon and max_event_samples
        effective_horizon = int(decision_horizon) if decision_horizon is not None else ring_next
        collect_limit = min(ring_next, effective_horizon)

        if collect_limit > act_start:
            max_allowed_end = act_start + self._max_event_samples
            initial_end = min(collect_limit, max_allowed_end)
            history = self._ring.read_range(act_start, initial_end)
            self._assembled_chunks.append(history)
            self._collected_end_sample = initial_end
            if self._collected_end_sample >= max_allowed_end:
                self._max_duration_reached = True
        else:
            self._collected_end_sample = act_start

        self._in_flight = True

    def update_active(self, horizon: int) -> None:
        """Update active event collection with new samples up to logical decision horizon."""
        if not self._in_flight:
            return

        max_allowed_end = self._actual_event_start + self._max_event_samples
        if self._collected_end_sample >= max_allowed_end:
            self._max_duration_reached = True
            return

        end_to_read = min(int(horizon), max_allowed_end)
        if end_to_read > self._collected_end_sample:
            chunk = self._ring.read_range(self._collected_end_sample, end_to_read)
            self._assembled_chunks.append(chunk)
            self._collected_end_sample = end_to_read
            if self._collected_end_sample >= max_allowed_end:
                self._max_duration_reached = True

    def end_capture(
        self,
        detected_end_sample: int,
        end_confirmation_sample: int,
    ) -> None:
        """Register signal end boundary and set target trailing endpoint."""
        if not self._in_flight:
            return

        self._detected_end_sample = int(detected_end_sample)
        self._end_confirmation_sample = int(end_confirmation_sample)

        # Calculate requested target trailing sample
        req_target_end = max(
            self._detected_end_sample + self._post_signal_samples,
            self._trigger_sample + self._min_post_trigger_samples,
        )
        self._requested_target_end_sample = req_target_end

        # Cap by max_event_samples
        max_allowed_end = self._actual_event_start + self._max_event_samples
        if req_target_end > max_allowed_end:
            self._target_end_sample = max_allowed_end
            self._max_duration_reached = True
        else:
            self._target_end_sample = req_target_end

    def reopen(self) -> None:
        """Clear pending end/confirmation/finalization target when signal reopens in POST_TRIGGER."""
        if not self._in_flight:
            return
        self._detected_end_sample = None
        self._end_confirmation_sample = None
        self._target_end_sample = None
        self._requested_target_end_sample = None
        self._reopened_count += 1

    def check_finalization(self) -> CaptureSidecar | None:
        """If target trailing endpoint has been reached, assemble and return CaptureSidecar."""
        if not self._in_flight or self._target_end_sample is None:
            return None

        if self._collected_end_sample >= self._target_end_sample:
            return self._build_event_and_sidecar(
                termination_reason="max_duration" if self._max_duration_reached else "normal",
                is_truncated=self._is_leading_truncated or self._max_duration_reached,
            )

        return None

    def handle_discontinuity(
        self, dropped_samples: int = 0, overflow: bool = False
    ) -> CaptureSidecar | None:
        """Terminate active capture early due to stream discontinuity / dropped samples."""
        if not self._in_flight:
            return None

        self._known_lost_samples += int(dropped_samples)
        if overflow:
            self._unknown_loss = True

        # Collect available trusted old-generation IQ up to stream discontinuity
        avail_end = self._ring.next_write_sample
        max_allowed_end = self._actual_event_start + self._max_event_samples
        target = min(avail_end, max_allowed_end)
        if self._target_end_sample is not None:
            target = min(target, self._target_end_sample)
        if target > self._collected_end_sample:
            self.update_active(target)

        return self._build_event_and_sidecar(
            termination_reason="discontinuity",
            is_truncated=True,
        )

    def flush_eof(self) -> CaptureSidecar | None:
        """Terminate active capture at EOF, preserving available tail samples."""
        if not self._in_flight:
            return None

        # Collect available real IQ up to stream end before finalizing
        avail_end = self._ring.next_write_sample
        max_allowed_end = self._actual_event_start + self._max_event_samples
        target = min(avail_end, max_allowed_end)
        if self._target_end_sample is not None:
            target = min(target, self._target_end_sample)
        if target > self._collected_end_sample:
            self.update_active(target)

        return self._build_event_and_sidecar(
            termination_reason="eof",
            is_truncated=True,
        )

    def _build_event_and_sidecar(
        self,
        termination_reason: str,
        is_truncated: bool,
    ) -> CaptureSidecar:
        assert self._stream_info is not None

        if len(self._assembled_chunks) == 0:
            full_iq = np.empty(0, dtype=np.complex64)
        elif len(self._assembled_chunks) == 1:
            full_iq = self._assembled_chunks[0]
        else:
            full_iq = np.concatenate(self._assembled_chunks)

        act_start = self._actual_event_start
        collected_len = len(full_iq)

        if self._target_end_sample is not None and self._collected_end_sample >= self._target_end_sample:
            act_end = self._target_end_sample
            trim_len = act_end - act_start
            full_iq = full_iq[:trim_len]
        else:
            act_end = act_start + collected_len

        # Detected landmarks without fabricated padding
        det_start = self._detected_start_sample
        trig = self._trigger_sample
        if self._detected_end_sample is not None:
            det_end = min(self._detected_end_sample, act_end)
        else:
            det_end = act_end

        # Strict truthful invariant: no fabricated zeros!
        if act_end <= trig or det_end <= trig or act_end <= act_start:
            raise RuntimeError(
                f"Capture truncated before minimal event landmark invariant: "
                f"act_start={act_start}, trig={trig}, det_end={det_end}, act_end={act_end}"
            )

        event_iq = _validate_and_freeze_iq_array(full_iq)

        # Derive event timestamp from anchor if available
        event_ts_ns: int | None = None
        if self._anchor_start_sample is not None and self._anchor_timestamp_ns is not None:
            offset = act_start - self._anchor_start_sample
            event_ts_ns = self._anchor_timestamp_ns + int(
                round(offset * 1e9 / self._stream_info.sample_rate_hz)
            )

        req_lead = self._trigger_sample - self._requested_event_start
        act_lead = trig - act_start
        req_trail = (self._requested_target_end_sample - det_end) if self._requested_target_end_sample is not None else 0
        act_trail = act_end - det_end

        trunc = is_truncated or (act_lead < req_lead) or (act_trail < req_trail) or self._max_duration_reached
        complete = (not trunc) and (self._known_lost_samples == 0) and (not self._unknown_loss)

        event = SignalEvent(
            event_id=self._event_id,
            stream=self._stream_info,
            iq=event_iq,
            event_start_sample=act_start,
            event_end_sample=act_end,
            detected_start_sample=det_start,
            detected_end_sample=det_end,
            trigger_sample=trig,
            timestamp_ns=event_ts_ns,
            dropped_samples=0,
            complete=complete,
        )

        sidecar = CaptureSidecar(
            stream_epoch=self._stream_epoch,
            event_id=self._event_id,
            config_version=self._config_version,
            detector_mode=self._detector_mode,
            monitored_band_hz=self._monitored_band_hz,
            first_on_sample=self._first_on_sample,
            confirmation_sample=self._confirmation_sample,
            end_confirmation_sample=self._end_confirmation_sample,
            requested_leading_samples=req_lead,
            actual_leading_samples=act_lead,
            requested_trailing_samples=req_trail,
            actual_trailing_samples=act_trail,
            boundary_uncertainty_samples=self._boundary_uncertainty,
            noise_provenance=self._noise_provenance,
            noise_valid=self._noise_valid,
            termination_reason=termination_reason,
            is_truncated=trunc,
            known_lost_samples=self._known_lost_samples,
            unknown_loss=self._unknown_loss,
            feature_status="pending",
            feature_reason=None,
            event=event,
            features=None,
            reopened_count=self._reopened_count,
        )

        # Reset active state
        self._in_flight = False
        self._assembled_chunks = []

        return sidecar

    def reset(self) -> None:
        """Reset assembler state."""
        self._in_flight = False
        self._assembled_chunks = []
        self._stream_info = None
        self._anchor_start_sample = None
        self._anchor_timestamp_ns = None
