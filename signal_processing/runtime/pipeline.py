"""End-to-end streaming capture and feature processing pipeline."""

from __future__ import annotations

import enum
import math
from typing import Literal

import numpy as np

from signal_processing.buffering.ring_buffer import CircularIQBuffer
from signal_processing.capture.assembler import EventAssembler
from signal_processing.capture.sidecar import CaptureSidecar
from signal_processing.context import AnalysisConfig
from signal_processing.core.contracts import IQBlock, StreamInfo
from signal_processing.core.enums import SegmentState, SegmentTransition
from signal_processing.core.exceptions import StreamConsistencyError
from signal_processing.core.interfaces import NoiseEstimatorProtocol
from signal_processing.detection.energy_detector import EnergyDetector
from signal_processing.detection.exceptions import PolicyCompatibilityError, UncalibratedNoiseError
from signal_processing.detection.noise_estimation import (
    GaussianNoiseEstimator,
    NoiseLifecycleState,
    ThresholdPolicy,
)
from signal_processing.runtime.worker import FeatureAnalysisWorker
from signal_processing.segmentation.segmenter import HysteresisSegmenter


class PipelineState(enum.Enum):
    """Lifecycle states for StreamingCapturePipeline."""

    OPEN = "OPEN"
    EOF = "EOF"
    FAULTED = "FAULTED"


class StreamingCapturePipeline:
    """Integrated streaming signal detection, segmentation, and feature extraction pipeline.

    Processes incoming continuous IQBlock streams:
      1. Validates stream metadata and sequence continuity.
      2. Appends raw IQ into preallocated circular ring buffer using bounded chunking.
      3. Performs causal windowed energy detection (full-band or configured-band).
      4. Drives the hysteresis persistence segmentation state machine.
      5. Gathers pre-trigger history and trailing margins into SignalEvents.
      6. Applies capture quality gating and dispatches completed events to feature analysis.
    """

    def __init__(
        self,
        stream_info: StreamInfo,
        ring_capacity: int = 16_384,
        window_size: int = 256,
        hop_size: int = 64,
        pfa_on: float = 1e-4,
        pfa_off: float = 1e-2,
        detector_mode: Literal["full_band", "configured_band"] = "full_band",
        monitored_band_hz: tuple[float, float] | None = None,
        noise_estimator: NoiseEstimatorProtocol | None = None,
        start_persistence: int = 2,
        end_persistence: int = 5,
        pre_trigger_samples: int = 512,
        pre_signal_margin_samples: int = 256,
        post_signal_samples: int = 256,
        min_post_trigger_samples: int = 512,
        max_event_samples: int = 100_000,
        synchronous_analysis: bool = True,
        skip_incomplete_quality: bool = True,
        enable_feature_analysis: bool = True,
        max_worker_queue_size: int = 32,
        analysis_config: AnalysisConfig | None = None,
        threshold_policy: ThresholdPolicy | None = None,
        gain_id: str = "digital_iq_unity",
        require_supported_pfa: bool = False,
        max_completed_per_call: int = 128,
        analysis_function=None,
        result_delivery: str = "polling",
    ) -> None:
        if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)) or window_size < 1:
            raise ValueError(f"window_size must be an integer >= 1, got {window_size!r}")
        if isinstance(hop_size, bool) or not isinstance(hop_size, (int, np.integer)) or not (1 <= hop_size <= window_size):
            raise ValueError(f"hop_size must satisfy 1 <= hop_size <= window_size, got {hop_size!r}")
        if not (0.0 < pfa_on < pfa_off < 1.0):
            raise ValueError("False alarm probabilities must satisfy 0 < pfa_on < pfa_off < 1")
        if isinstance(start_persistence, bool) or not isinstance(start_persistence, (int, np.integer)) or start_persistence < 1:
            raise ValueError(f"start_persistence must be an integer >= 1, got {start_persistence!r}")
        if isinstance(end_persistence, bool) or not isinstance(end_persistence, (int, np.integer)) or end_persistence < 1:
            raise ValueError(f"end_persistence must be an integer >= 1, got {end_persistence!r}")
        if isinstance(ring_capacity, bool) or not isinstance(ring_capacity, (int, np.integer)) or ring_capacity < 1:
            raise ValueError(f"ring_capacity must be an integer >= 1, got {ring_capacity!r}")
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
        if detector_mode not in ("full_band", "configured_band"):
            raise ValueError(f"Unknown detector_mode: {detector_mode!r}. Must be 'full_band' or 'configured_band'")

        # Joint limit validation
        min_event_extent = max(
            pre_trigger_samples + 1,
            (start_persistence - 1) * hop_size + window_size + pre_signal_margin_samples,
        )
        if max_event_samples < min_event_extent:
            raise ValueError(
                f"Incompatible capture configuration: max_event_samples ({max_event_samples}) "
                f"must be at least {min_event_extent} samples to span detector window/margins"
            )

        if type(max_completed_per_call) is not int or max_completed_per_call < 1:
            raise ValueError("max_completed_per_call must be a positive integer")
        self._max_completed_per_call = max_completed_per_call
        self._call_outputs: list[CaptureSidecar] | None = None
        self._stream_info = stream_info
        self._state = PipelineState.OPEN

        self._detector = EnergyDetector(
            window_size=window_size,
            hop_size=hop_size,
            pfa_on=pfa_on,
            pfa_off=pfa_off,
            mode=detector_mode,
            monitored_band_hz=monitored_band_hz,
            sample_rate_hz=stream_info.sample_rate_hz,
            threshold_policy=threshold_policy,
            gain_id=gain_id,
            require_supported_pfa=require_supported_pfa,
        )

        filter_delay = getattr(self._detector, "filter_delay_samples", 0)
        min_required_history = (
            pre_trigger_samples
            + pre_signal_margin_samples
            + start_persistence * hop_size
            + window_size
            + filter_delay
        )
        if ring_capacity <= min_required_history:
            raise ValueError(
                f"ring_capacity ({ring_capacity}) is insufficient for requested retention policy "
                f"(requires strictly more than {min_required_history} samples)"
            )
        self._min_required_history = min_required_history

        self._ring = CircularIQBuffer(capacity=ring_capacity)

        if noise_estimator is not None:
            self._noise_estimator = noise_estimator
        else:
            self._noise_estimator = GaussianNoiseEstimator(
                window_size=window_size,
                pfa_on=pfa_on,
                pfa_off=pfa_off,
            )

        self._segmenter = HysteresisSegmenter(
            start_persistence=start_persistence,
            end_persistence=end_persistence,
            max_event_samples=max_event_samples,
        )

        self._assembler = EventAssembler(
            ring_buffer=self._ring,
            pre_trigger_samples=pre_trigger_samples,
            pre_signal_margin_samples=pre_signal_margin_samples,
            post_signal_samples=post_signal_samples,
            min_post_trigger_samples=min_post_trigger_samples,
            max_event_samples=max_event_samples,
            window_size=window_size,
            hop_size=hop_size,
            start_persistence=start_persistence,
        )

        self._worker = FeatureAnalysisWorker(
            max_queue_size=max_worker_queue_size,
            skip_incomplete_quality=skip_incomplete_quality,
            synchronous=synchronous_analysis,
            analysis_config=analysis_config,
            analysis_function=analysis_function,
            result_delivery=result_delivery,
        )

        self._enable_feature_analysis = bool(enable_feature_analysis)
        self._previous_block: IQBlock | None = None
        self._time_anchor: tuple[int, int] | None = None
        self._noise_admission_sample: int | None = None
        self._pending_noise_buf: np.ndarray = np.empty(0, dtype=np.complex64)
        # Compatibility view: no IQ history is retained here.
        self._pending_noise_iq: tuple = ()
        self._received_end_sample: int | None = None
        self._input_residual_iq: np.ndarray = np.empty(0, dtype=np.complex64)
        self._input_residual_start: int = 0

    def _dispatch_event(self, sidecar: CaptureSidecar) -> CaptureSidecar:
        """Route finalized event through feature worker or return directly if feature analysis is disabled."""
        # Record ownership BEFORE analysis submission: submission itself can fail.
        index = len(self._call_outputs) if self._call_outputs is not None else None
        if index is not None:
            self._call_outputs.append(sidecar)
            if len(self._call_outputs) >= self._max_completed_per_call:
                self._call_outputs[index] = sidecar.with_features(features=None, feature_status="failed",
                    feature_reason="Output capacity reached before analysis dispatch")
                raise RuntimeError("Completed-capture limit reached; recover exception.completed_captures and reset")
        try:
            result = self._analyze_event(sidecar)
        except Exception as exc:
            if index is not None:
                self._call_outputs[index] = sidecar.with_features(
                    features=None, feature_status="failed", feature_reason=f"Analysis dispatch failed: {type(exc).__name__}: {exc}")
            raise
        if index is not None:
            self._call_outputs[index] = result
        return result

    def _analyze_event(self, sidecar: CaptureSidecar) -> CaptureSidecar:
        if not self._enable_feature_analysis:
            return sidecar.with_features(
                features=None,
                feature_status="skipped",
                feature_reason="Feature extraction disabled (capture-only mode)",
            )
        return self._worker.submit(sidecar)

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def ring_buffer(self) -> CircularIQBuffer:
        return self._ring

    @property
    def detector(self) -> EnergyDetector:
        return self._detector

    @property
    def noise_estimator(self) -> NoiseEstimatorProtocol:
        return self._noise_estimator

    @property
    def segmenter(self) -> HysteresisSegmenter:
        return self._segmenter

    @property
    def assembler(self) -> EventAssembler:
        return self._assembler

    @property
    def worker(self) -> FeatureAnalysisWorker:
        return self._worker

    @property
    def feature_worker(self) -> FeatureAnalysisWorker:
        return self._worker

    def process_block(self, block: IQBlock) -> list[CaptureSidecar]:
        """Return captures, or raise the original error with completed_captures.

        The immutable exception tuple owns this call's completed raw outputs.
        Feature callbacks/polling are analysis updates of the same event identity,
        not additional raw events. No recovery queue is retained by the pipeline.
        """
        self._call_outputs = []
        consumed_before = self.consumed_end_sample
        try:
            return self._process_block(block)
        except Exception as exc:
            if self._state != PipelineState.EOF and (not isinstance(exc, StreamConsistencyError)
                    or self.consumed_end_sample != consumed_before or self._call_outputs):
                self._state = PipelineState.FAULTED
            self._attach_recovery(exc)
            raise
        finally:
            self._call_outputs = None

    def _attach_recovery(self, exc: Exception) -> None:
        exc.completed_captures = tuple(self._call_outputs or ())
        exc.consumed_end_sample = self.consumed_end_sample
        exc.received_end_sample = self.received_end_sample

    def _process_block(self, block: IQBlock) -> list[CaptureSidecar]:
        """Process incoming IQBlock chunk through the streaming pipeline."""
        if self._state == PipelineState.EOF:
            raise RuntimeError("Cannot process block: pipeline is at EOF. Call reset() to restart.")
        if self._state == PipelineState.FAULTED:
            raise RuntimeError("Cannot process block: pipeline is in FAULTED state. Call reset() to recover.")

        emitted: list[CaptureSidecar] = []
        transport_block = block

        # IQBlock construction already validates and freezes all received data.
        if not isinstance(block, IQBlock):
            raise StreamConsistencyError('Pipeline input must be an immutable IQBlock')

        # 1. Stream validation before mutating state
        if self._previous_block is None:
            if block.stream.stream_id != self._stream_info.stream_id:
                raise StreamConsistencyError(
                    f"First block stream_id '{block.stream.stream_id}' does not match "
                    f"pipeline stream_id '{self._stream_info.stream_id}'"
                )
            if hasattr(self._stream_info, "source_id") and hasattr(block.stream, "source_id"):
                if block.stream.source_id != self._stream_info.source_id:
                    raise StreamConsistencyError(
                        f"First block source_id '{block.stream.source_id}' does not match "
                        f"pipeline source_id '{self._stream_info.source_id}'"
                    )
            if hasattr(self._stream_info, "channel_id") and hasattr(block.stream, "channel_id"):
                if block.stream.channel_id != self._stream_info.channel_id:
                    raise StreamConsistencyError(
                        f"First block channel_id '{block.stream.channel_id}' does not match "
                        f"pipeline channel_id '{self._stream_info.channel_id}'"
                    )
            if not math.isclose(block.stream.sample_rate_hz, self._stream_info.sample_rate_hz, rel_tol=1e-9):
                raise StreamConsistencyError(
                    f"First block sample rate {block.stream.sample_rate_hz} Hz does not match "
                    f"pipeline sample rate {self._stream_info.sample_rate_hz} Hz"
                )
            if self._stream_info.center_frequency_hz is not None:
                if block.stream.center_frequency_hz is None or not math.isclose(
                    block.stream.center_frequency_hz, self._stream_info.center_frequency_hz, rel_tol=1e-9
                ):
                    raise StreamConsistencyError(
                        f"First block center frequency {block.stream.center_frequency_hz} Hz does not match "
                        f"pipeline center frequency {self._stream_info.center_frequency_hz} Hz"
                    )
        else:
            # Metadata must match ALWAYS, even if loss is declared
            if block.stream.stream_id != self._stream_info.stream_id:
                raise StreamConsistencyError(
                    f"Stream ID mismatch: expected '{self._stream_info.stream_id}', got '{block.stream.stream_id}'"
                )
            if hasattr(self._stream_info, "source_id") and hasattr(block.stream, "source_id"):
                if block.stream.source_id != self._stream_info.source_id:
                    raise StreamConsistencyError(
                        f"Source ID mismatch: expected '{self._stream_info.source_id}', got '{block.stream.source_id}'"
                    )
            if hasattr(self._stream_info, "channel_id") and hasattr(block.stream, "channel_id"):
                if block.stream.channel_id != self._stream_info.channel_id:
                    raise StreamConsistencyError(
                        f"Channel ID mismatch: expected '{self._stream_info.channel_id}', got '{block.stream.channel_id}'"
                    )
            if not math.isclose(block.stream.sample_rate_hz, self._stream_info.sample_rate_hz, rel_tol=1e-9):
                raise StreamConsistencyError(
                    f"Sample rate mismatch: expected {self._stream_info.sample_rate_hz} Hz, got {block.stream.sample_rate_hz} Hz"
                )
            prev_cf = self._previous_block.stream.center_frequency_hz
            curr_cf = block.stream.center_frequency_hz
            if (prev_cf is None) != (curr_cf is None):
                raise StreamConsistencyError(
                    f"Incompatible center frequency configuration: previous={prev_cf}, current={curr_cf}"
                )
            if prev_cf is not None and curr_cf is not None and not math.isclose(prev_cf, curr_cf, rel_tol=1e-9):
                raise StreamConsistencyError("Center frequency mismatch between consecutive blocks")

            # Check sequence ordering
            if block.sequence_number <= self._previous_block.sequence_number:
                raise StreamConsistencyError(
                    f"Duplicate or out-of-order block sequence number: previous="
                    f"{self._previous_block.sequence_number}, current={block.sequence_number}"
                )

            # Check coordinate continuity & declared drops
            expected_start = self._previous_block.end_sample + block.dropped_samples_before
            if not block.overflow:
                if block.start_sample != expected_start:
                    gap = block.start_sample - self._previous_block.end_sample
                    raise StreamConsistencyError(
                        f"Unexplained sample discontinuity: previous block ended at {self._previous_block.end_sample}, "
                        f"current block starts at {block.start_sample} (gap of {gap} samples), "
                        f"but declared dropped_samples_before is {block.dropped_samples_before}"
                    )
                if block.dropped_samples_before == 0 and block.sequence_number != self._previous_block.sequence_number + 1:
                    raise StreamConsistencyError(
                        f"Sequence jump without declared sample drop: previous={self._previous_block.sequence_number}, "
                        f"current={block.sequence_number}"
                    )
            else:
                if block.start_sample < self._previous_block.end_sample:
                    raise StreamConsistencyError("Sample index overlap backwards during overflow")

        # Receiver clock consistency is independent of sample continuity. Keep
        # one original anchor; never accumulate transport-rounding drift. Unknown
        # overflow ends timing trust; an explicit reset is required for clock-only
        # changes. Two rounded integer-ns anchors may differ by at most 1 ns.
        if not block.overflow and block.timestamp_ns is not None and self._time_anchor is not None:
            from signal_processing.core.timebase import sample_offset_ns
            index, timestamp = self._time_anchor
            expected = timestamp + sample_offset_ns(block.start_sample-index, self._stream_info.sample_rate_hz)
            if abs(block.timestamp_ns-expected) > 1:
                raise StreamConsistencyError('Receiver timestamp contradicts the timing epoch (tolerance 1 ns); reset required for clock changes')

        # Check noise estimator status before mutating ring buffer state (B06 / C02 / D06)
        is_calibrating = False
        if hasattr(self._noise_estimator, "state"):
            est_state = self._noise_estimator.state
            if est_state == NoiseLifecycleState.UNCALIBRATED:
                self._state = PipelineState.FAULTED
                raise UncalibratedNoiseError("Cannot process block: noise estimator is uncalibrated")
            elif est_state == NoiseLifecycleState.STALE:
                self._state = PipelineState.FAULTED
                raise UncalibratedNoiseError("Cannot process block: noise estimator is stale")
            elif est_state == NoiseLifecycleState.INVALID:
                self._state = PipelineState.FAULTED
                raise UncalibratedNoiseError("Cannot process block: noise estimator is invalid")
            elif est_state == NoiseLifecycleState.CALIBRATING:
                is_calibrating = True
        elif hasattr(self._noise_estimator, "is_calibrated") and not self._noise_estimator.is_calibrated:
            warmup_blocks = getattr(self._noise_estimator, "_warmup_blocks", 0)
            warmup_samples = getattr(self._noise_estimator, "_warmup_samples", 0)
            if warmup_blocks == 0 and warmup_samples == 0:
                self._state = PipelineState.FAULTED
                raise UncalibratedNoiseError("Cannot process block: noise estimator is uncalibrated")
            is_calibrating = True

        if is_calibrating and getattr(self._noise_estimator, 'frozen', False):
            raise StreamConsistencyError('Frozen pipeline warmup cannot consume input; unfreeze and retry the same unconsumed block')

        # Validate policy / threshold compatibility before ring/previous commit (D06)
        if not is_calibrating and hasattr(self._noise_estimator, "get_estimate"):
            preview_noise = self._noise_estimator.get_estimate(block.stream.stream_id, block.sequence_number)
            if preview_noise.valid and math.isfinite(preview_noise.noise_power_dbfs):
                try:
                    self._detector._calculate_thresholds(preview_noise)
                except PolicyCompatibilityError:
                    self._state = PipelineState.FAULTED
                    raise
                except Exception:
                    self._state = PipelineState.FAULTED
                    raise

        if self._previous_block is None and block.stream.center_frequency_hz is not None:
            self._stream_info = StreamInfo(
                stream_id=self._stream_info.stream_id,
                source_id=getattr(self._stream_info, "source_id", "default"),
                sample_rate_hz=self._stream_info.sample_rate_hz,
                center_frequency_hz=block.stream.center_frequency_hz,
                channel_id=getattr(self._stream_info, "channel_id", "default"),
                dtype=getattr(self._stream_info, "dtype", "complex64"),
            )

        self._received_end_sample = block.end_sample
        if self._noise_admission_sample is None:
            self._noise_admission_sample = block.start_sample

        if block.dropped_samples_before > 0 or block.overflow:
            if is_calibrating and hasattr(self._noise_estimator, "suspend_training"):
                self._noise_estimator.suspend_training()

            # Evaluate the old generation through the same path used at EOF.
            emitted.extend(self._drain_input_residual())

            # Discontinuity: terminate in-flight capture
            discont_sidecar = self._assembler.handle_discontinuity(
                dropped_samples=block.dropped_samples_before,
                overflow=block.overflow,
            )
            if discont_sidecar is not None:
                self._segmenter.finish_event()
                analyzed = self._dispatch_event(discont_sidecar)
                emitted.append(analyzed)

            self._detector.reset()
            if is_calibrating:
                if hasattr(self._noise_estimator, "suspend_training"):
                    self._noise_estimator.suspend_training()
            elif isinstance(self._noise_estimator, GaussianNoiseEstimator):
                pass
            else:
                self._noise_estimator.reset()

            # Raw continuity changed; assembled immutable analysis jobs remain valid.
            self._ring.reset(start_sample=block.start_sample)
            self._segmenter.reset()
            self._pending_noise_buf = np.empty(0, dtype=np.complex64)
            self._pending_noise_iq = ()
            self._noise_admission_sample = block.start_sample
            self._input_residual_iq = np.empty(0, dtype=np.complex64)

        # Update assembler timestamp anchor (discard trust after unknown overflow)
        if block.overflow:
            self._assembler.set_timestamp_anchor(block.start_sample, None)
            self._time_anchor = None
        elif block.timestamp_ns is not None and self._time_anchor is None:
            self._time_anchor = (block.start_sample, block.timestamp_ns)
            self._assembler.set_timestamp_anchor(block.start_sample, block.timestamp_ns)

        if is_calibrating:
            if getattr(self._noise_estimator, "_suspended", False):
                self._ring.write(block.start_sample, block.iq)
                self._previous_block = block
                return []

            # During warmup, consume only authorized prefix without filtering future suffix (D03)
            needed = len(block.iq)
            if hasattr(self._noise_estimator, "remaining_warmup_samples"):
                needed = min(needed, self._noise_estimator.remaining_warmup_samples)
            elif hasattr(self._noise_estimator, "_target_warmup_samples"):
                collected = getattr(self._noise_estimator, "_warmup_samples_collected", 0)
                needed = min(len(block.iq), max(0, self._noise_estimator._target_warmup_samples - collected))
            elif hasattr(self._noise_estimator, "_warmup_samples"):
                collected = getattr(self._noise_estimator, "_warmup_samples_collected", 0)
                needed = min(len(block.iq), max(0, self._noise_estimator._warmup_samples - collected))

            warmup_slice = block.iq[:needed]
            # Authorized training really traverses this path; bind its identity,
            # rather than inferring compatibility from a similar mean power.
            if hasattr(self._noise_estimator, "reference_domain"):
                self._noise_estimator.reference_domain = self._detector.reference_domain
            cal_iq = self._detector.prepare_detection_iq(IQBlock(
                block.stream, block.sequence_number, block.start_sample, block.timestamp_ns, warmup_slice))

            consumed = 0
            if hasattr(self._noise_estimator, "update_samples"):
                consumed = self._noise_estimator.update_samples(cal_iq)
            elif hasattr(self._noise_estimator, "update"):
                warmup_block = IQBlock(
                    stream=block.stream,
                    sequence_number=block.sequence_number,
                    start_sample=block.start_sample,
                    timestamp_ns=block.timestamp_ns,
                    iq=cal_iq,
                    dropped_samples_before=block.dropped_samples_before,
                    overflow=block.overflow,
                )
                self._noise_estimator.update(warmup_block)
                consumed = len(warmup_slice)

            self._ring.write(block.start_sample, block.iq[:consumed])
            self._noise_admission_sample = block.start_sample + consumed

            still_calibrating = False
            if hasattr(self._noise_estimator, "state"):
                still_calibrating = (self._noise_estimator.state == NoiseLifecycleState.CALIBRATING)
            elif hasattr(self._noise_estimator, "is_calibrated"):
                still_calibrating = not self._noise_estimator.is_calibrated

            if still_calibrating or consumed >= len(block.iq):
                self._previous_block = block
                return []

            # Calibration budget reached inside this block!
            block = IQBlock(
                stream=block.stream,
                sequence_number=block.sequence_number,
                start_sample=block.start_sample + consumed,
                timestamp_ns=None,
                iq=block.iq[consumed:],
                dropped_samples_before=0,
                overflow=False,
            )

        if self._noise_admission_sample is None:
            self._noise_admission_sample = block.start_sample

        # Combine block with preceding input residual (C01 / C02)
        if len(self._input_residual_iq) > 0:
            combined_iq = np.concatenate([self._input_residual_iq, block.iq])
            combined_start = self._input_residual_start
            combined_block = IQBlock(block.stream, block.sequence_number, combined_start, None, combined_iq)
        else:
            combined_iq = block.iq
            combined_start = block.start_sample
            combined_block = block

        hop_size = self._detector.hop_size
        num_hops = len(combined_iq) // hop_size
        self._input_residual_iq = np.empty(0, dtype=np.complex64)

        for h_idx in range(num_hops):
            h_start = h_idx * hop_size
            h_end = h_start + hop_size
            hop_block = combined_block.slice_received(h_start, h_end)
            emitted.extend(self._process_detection_chunk(hop_block))

        # Update input residual for remaining samples
        rem_offset = num_hops * hop_size
        if rem_offset < len(combined_iq):
            self._input_residual_iq = combined_iq[rem_offset:].copy()
            self._input_residual_start = combined_start + rem_offset
        else:
            self._input_residual_iq = np.empty(0, dtype=np.complex64)
            self._input_residual_start = combined_start + len(combined_iq)

        self._previous_block = transport_block
        self._sync_pending_age()
        return emitted

    @property
    def consumed_end_sample(self) -> int:
        """Exclusive raw commit horizon; inspect after a FAULTED partial block."""
        return self._ring.next_write_sample

    @property
    def received_end_sample(self) -> int | None:
        """End of the most recently admitted transport block, including unprocessed suffix."""
        return self._received_end_sample

    def _sync_pending_age(self) -> None:
        if hasattr(self._noise_estimator, "set_pending_samples"):
            self._noise_estimator.set_pending_samples(
                len(self._pending_noise_buf) + len(self._input_residual_iq)
            )

    def _process_detection_chunk(self, block: IQBlock) -> list[CaptureSidecar]:
        """One chronological chunk, shared by transport, EOF and discontinuity."""
        emitted: list[CaptureSidecar] = []
        try:
            if getattr(self._noise_estimator, "state", None) == NoiseLifecycleState.STALE:
                raise UncalibratedNoiseError("Cannot process block: noise estimator is stale")
            if hasattr(self._noise_estimator, "get_estimate"):
                noise_est = self._noise_estimator.get_estimate(
                    block.stream.stream_id, block.sequence_number
                )
            else:
                noise_est = self._noise_estimator.update(block)
            # Detector validates prerequisites before filter mutation. Commit raw IQ only
            # after detection succeeds; any subsequent failure requires explicit reset.
            measurements = self._detector.process_windows(block, noise_est)
            self._ring.write_received(block)
        except Exception:
            self._state = PipelineState.FAULTED
            raise

        det_iq = self._detector.last_detection_iq if self._detector.mode == "configured_band" else block.iq
        self._pending_noise_buf = np.concatenate([self._pending_noise_buf, det_iq])
        if hasattr(self._noise_estimator, "set_pending_samples"):
            self._noise_estimator.set_pending_samples(0)

        for meas in measurements:
            horizon = meas.end_sample

            if self._assembler.is_in_flight:
                self._assembler.update_active(horizon)

            target_end = self._assembler.target_end_sample
            if self._assembler.is_in_flight and target_end is not None:
                if target_end < horizon:
                    sidecar = self._assembler.check_finalization()
                    if sidecar is not None:
                        self._segmenter.finish_event()
                        analyzed = self._dispatch_event(sidecar)
                        emitted.append(analyzed)
                elif target_end == horizon and not meas.above_on_threshold:
                    sidecar = self._assembler.check_finalization()
                    if sidecar is not None:
                        self._segmenter.finish_event()
                        analyzed = self._dispatch_event(sidecar)
                        emitted.append(analyzed)

            is_active_or_candidate = (
                self._segmenter.state != SegmentState.IDLE
                or meas.above_on_threshold
                or self._segmenter._on_count > 0
            )

            if hasattr(self._noise_estimator, "freeze"):
                self._noise_estimator.freeze(is_active_or_candidate)

            prev_state = self._segmenter.state
            decision = self._segmenter.update(meas)

            if prev_state == SegmentState.POST_TRIGGER and decision.state == SegmentState.ACTIVE:
                self._assembler.reopen()

            if decision.transition == SegmentTransition.START:
                assert self._segmenter.first_on_sample is not None
                assert self._segmenter.confirmation_sample is not None
                assert self._segmenter.trigger_sample is not None
                assert self._segmenter.detected_start_sample is not None

                self._assembler.start_capture(
                    stream_info=block.stream,
                    first_on_sample=self._segmenter.first_on_sample,
                    confirmation_sample=self._segmenter.confirmation_sample,
                    trigger_sample=self._segmenter.trigger_sample,
                    detected_start_sample=self._segmenter.detected_start_sample,
                    detector_mode=self._detector.mode,
                    monitored_band_hz=self._detector.monitored_band_hz,
                    noise_provenance=type(self._noise_estimator).__name__,
                    noise_valid=noise_est.valid,
                    boundary_uncertainty=None,
                    decision_horizon=horizon,
                )
            elif decision.transition == SegmentTransition.END:
                assert self._segmenter.detected_end_sample is not None
                assert self._segmenter.end_confirmation_sample is not None
                self._assembler.end_capture(
                    detected_end_sample=self._segmenter.detected_end_sample,
                    end_confirmation_sample=self._segmenter.end_confirmation_sample,
                )

            if (
                self._assembler.is_in_flight
                and self._assembler.target_end_sample is not None
                and horizon >= self._assembler.target_end_sample
            ):
                sidecar = self._assembler.check_finalization()
                if sidecar is not None:
                    self._segmenter.finish_event()
                    analyzed = self._dispatch_event(sidecar)
                    emitted.append(analyzed)

            # Exact causal noise admission up to decision horizon (D01)
            count = min(len(self._pending_noise_buf), max(0, horizon - self._noise_admission_sample))
            if count > 0:
                chunk = self._pending_noise_buf[:count]
                self._pending_noise_buf = self._pending_noise_buf[count:].copy()
                self._noise_admission_sample += count
                if is_active_or_candidate:
                    if hasattr(self._noise_estimator, "advance_stream_samples"):
                        self._noise_estimator.advance_stream_samples(count)
                else:
                    if hasattr(self._noise_estimator, "update_samples"):
                        self._noise_estimator.update_samples(chunk)
        self._sync_pending_age()
        return emitted

    def _drain_input_residual(self) -> list[CaptureSidecar]:
        if len(self._input_residual_iq) == 0:
            return []
        block = IQBlock(
            stream=self._stream_info,
            sequence_number=self._previous_block.sequence_number if self._previous_block else 0,
            start_sample=self._input_residual_start, timestamp_ns=None,
            iq=self._input_residual_iq,
        )
        self._input_residual_iq = np.empty(0, dtype=np.complex64)
        return self._process_detection_chunk(block)

    def flush(self) -> list[CaptureSidecar]:
        self._call_outputs = []
        try:
            return self._flush()
        except Exception as exc:
            self._state = PipelineState.FAULTED
            self._attach_recovery(exc)
            raise
        finally:
            self._call_outputs = None

    def _flush(self) -> list[CaptureSidecar]:
        """Evaluate complete real windows and finalize the contiguous received prefix."""
        emitted: list[CaptureSidecar] = []
        if self._state == PipelineState.EOF:
            return emitted
        faulted = self._state == PipelineState.FAULTED
        if not faulted:
            emitted.extend(self._drain_input_residual())
        if hasattr(self._noise_estimator, "set_pending_samples"):
            self._noise_estimator.set_pending_samples(0)
        leftover = len(self._pending_noise_buf)
        if leftover and hasattr(self._noise_estimator, "advance_stream_samples"):
            self._noise_estimator.advance_stream_samples(leftover)
        self._pending_noise_buf = np.empty(0, dtype=np.complex64)
        if self._assembler.is_in_flight:
            sidecar = (self._assembler.handle_discontinuity(reason="processing_failure") if faulted
                       else self._assembler.flush_eof())
            if sidecar is not None:
                self._segmenter.finish_event()
                emitted.append(self._dispatch_event(sidecar))
        if not faulted:
            self._state = PipelineState.EOF
        return emitted

    finalize = flush

    def reset(self) -> None:
        """Reset all components to initial state."""
        self._state = PipelineState.OPEN
        self._ring.reset()
        self._detector.reset()
        self._noise_estimator.reset()
        self._segmenter.reset()
        self._assembler.reset()
        self._worker.reset()
        self._previous_block = None
        self._time_anchor = None
        self._received_end_sample = None
        self._noise_admission_sample = None
        self._pending_noise_buf = np.empty(0, dtype=np.complex64)
        self._pending_noise_iq = ()
        self._input_residual_iq = np.empty(0, dtype=np.complex64)
        self._input_residual_start = 0

    def shutdown(self, drain: bool = True, timeout: float = 5.0) -> None:
        """Shutdown worker thread."""
        self._worker.shutdown(drain=drain, timeout=timeout)
