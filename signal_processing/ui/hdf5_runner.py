"""HDF5 dataset adapter for exercising the real streaming capture pipeline.

The supplied ``subset_test.h5`` contains independent labelled frames, not one
continuous receiver recording.  This harness therefore places selected frames
between generated quiet guards.  In the recommended normalized mode, every
frame is scaled to a common physical noise reference using its nominal SNR
label.  The transformation makes the constructed stream's stationarity
assumption explicit; it does not rewrite the dataset or claim hardware truth.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

from signal_processing.capture.sidecar import CaptureSidecar
from signal_processing.core.contracts import IQBlock, StreamInfo
from signal_processing.detection.noise_estimation import (
    AdaptiveRobustNoiseEstimator,
    GaussianNoiseEstimator,
)
from signal_processing.runtime.pipeline import StreamingCapturePipeline

ProgressCallback = Callable[[float, str], None]


class RunCancelled(RuntimeError):
    """Raised when a caller cancels a dataset run between committed blocks."""


@dataclass(frozen=True)
class DatasetSummary:
    path: Path
    frame_count: int
    frame_length: int
    sample_dtype: str
    modulation_to_id: Mapping[str, int]
    snr_counts: Mapping[int, int]
    modulation_counts: Mapping[str, int]
    channel_counts: Mapping[int, int]
    channel_meaning: str
    source: str
    subset_name: str


@dataclass(frozen=True)
class HarnessConfig:
    sample_rate_hz: float = 1_000_000.0
    center_frequency_hz: float = 0.0
    frame_count: int = 24
    modulation: str | None = None
    snr_db: int | None = None
    channel: int | None = None
    selection_seed: int = 2026
    first_index: int = 0
    random_selection: bool = False
    normalize_frames: bool = True
    noise_power_linear: float = 1e-4
    initial_quiet_samples: int = 4096
    guard_samples: int = 2048
    final_quiet_samples: int = 2048
    transport_block_size: int = 1024
    window_size: int = 256
    hop_size: int = 64
    pfa_on: float = 1e-4
    pfa_off: float = 1e-2
    adaptive_noise: bool = True
    adaptation_window_size: int = 64
    start_persistence: int = 2
    end_persistence: int = 3
    pre_trigger_samples: int = 256
    pre_signal_margin_samples: int = 128
    post_signal_samples: int = 256
    min_post_trigger_samples: int = 256
    max_event_samples: int = 8192
    ring_capacity: int = 16384
    enable_features: bool = True
    max_frames: int = 500
    display_point_limit: int = 60_000

    def __post_init__(self) -> None:
        positive_ints = (
            "frame_count",
            "transport_block_size",
            "window_size",
            "hop_size",
            "adaptation_window_size",
            "start_persistence",
            "end_persistence",
            "max_event_samples",
            "ring_capacity",
            "max_frames",
            "display_point_limit",
        )
        nonnegative_ints = (
            "selection_seed",
            "first_index",
            "initial_quiet_samples",
            "guard_samples",
            "final_quiet_samples",
            "pre_trigger_samples",
            "pre_signal_margin_samples",
            "post_signal_samples",
            "min_post_trigger_samples",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in nonnegative_ints:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.frame_count > self.max_frames:
            raise ValueError(f"frame_count exceeds the bounded UI limit ({self.max_frames})")
        if self.hop_size > self.window_size:
            raise ValueError("hop_size must not exceed window_size")
        if not (math.isfinite(self.sample_rate_hz) and self.sample_rate_hz > 0):
            raise ValueError("sample_rate_hz must be finite and positive")
        if not math.isfinite(self.center_frequency_hz):
            raise ValueError("center_frequency_hz must be finite")
        if not (math.isfinite(self.noise_power_linear) and self.noise_power_linear > 0):
            raise ValueError("noise_power_linear must be finite and positive")
        if not 0 < self.pfa_on < self.pfa_off < 1:
            raise ValueError("require 0 < pfa_on < pfa_off < 1")
        for name in ("random_selection", "normalize_frames", "adaptive_noise", "enable_features"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")


@dataclass(frozen=True)
class FrameTruth:
    dataset_index: int
    start_sample: int
    end_sample: int
    modulation: str
    snr_db: int
    channel: int
    original_power: float
    streamed_power: float


@dataclass(frozen=True)
class EventMatch:
    event_index: int
    truth_index: int | None
    overlap_samples: int
    interval_iou: float


@dataclass(frozen=True)
class PipelineRunResult:
    dataset: DatasetSummary
    config: HarnessConfig
    selected_indices: tuple[int, ...]
    truths: tuple[FrameTruth, ...]
    captures: tuple[CaptureSidecar, ...]
    matches: tuple[EventMatch, ...]
    matched_truths: int
    missed_truths: int
    extra_events: int
    total_stream_samples: int
    elapsed_seconds: float
    throughput_samples_per_second: float
    final_noise_power_linear: float | None
    display_sample_indices: np.ndarray
    display_magnitudes: np.ndarray

    @property
    def detection_rate(self) -> float:
        return self.matched_truths / len(self.truths) if self.truths else 0.0


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "HDF5 UI support requires h5py. Install with: pip install -e .[ui]"
        ) from exc
    return h5py


def _strict_label_array(dataset, name: str, frame_count: int) -> np.ndarray:
    if name not in dataset:
        raise ValueError(f"HDF5 file is missing required dataset {name!r}")
    values = np.asarray(dataset[name][:])
    if values.ndim != 1 or len(values) != frame_count:
        raise ValueError(f"{name} must be a one-dimensional array with {frame_count} rows")
    if values.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer labels")
    return values.astype(np.int64, copy=False)


def inspect_hdf5_dataset(path: str | Path) -> DatasetSummary:
    """Validate and summarize a labelled ``X[..., I/Q]`` HDF5 dataset."""
    h5py = _require_h5py()
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    with h5py.File(file_path, "r") as handle:
        if "X" not in handle:
            raise ValueError("HDF5 file is missing required dataset 'X'")
        samples = handle["X"]
        if samples.ndim != 3 or samples.shape[2] != 2:
            raise ValueError("X must have shape (frames, samples, 2) for I and Q")
        frame_count, frame_length, _ = map(int, samples.shape)
        if frame_count < 1 or frame_length < 1 or samples.dtype.kind not in "fiu":
            raise ValueError("X must contain at least one finite numeric IQ frame")
        y_mod = _strict_label_array(handle, "y_mod", frame_count)
        y_snr = _strict_label_array(handle, "y_snr", frame_count)
        y_chan = _strict_label_array(handle, "y_chan", frame_count)
        try:
            raw_map = json.loads(str(handle.attrs.get("mod2id_json", "{}")))
        except json.JSONDecodeError as exc:
            raise ValueError("mod2id_json is not valid JSON") from exc
        if not isinstance(raw_map, dict) or not raw_map:
            raise ValueError("mod2id_json must define at least one modulation label")
        mod_to_id = {str(k): int(v) for k, v in raw_map.items()}
        if len(set(mod_to_id.values())) != len(mod_to_id):
            raise ValueError("mod2id_json contains duplicate numeric IDs")
        id_to_mod = {value: key for key, value in mod_to_id.items()}
        unknown = sorted(set(map(int, np.unique(y_mod))) - set(id_to_mod))
        if unknown:
            raise ValueError(f"y_mod contains IDs absent from mod2id_json: {unknown}")

        def counts(values: np.ndarray) -> dict[int, int]:
            unique, amount = np.unique(values, return_counts=True)
            return {int(k): int(v) for k, v in zip(unique, amount)}

        mod_counts_raw = counts(y_mod)
        return DatasetSummary(
            path=file_path,
            frame_count=frame_count,
            frame_length=frame_length,
            sample_dtype=str(samples.dtype),
            modulation_to_id=MappingProxyType(mod_to_id),
            snr_counts=MappingProxyType(counts(y_snr)),
            modulation_counts=MappingProxyType(
                {id_to_mod[key]: value for key, value in sorted(mod_counts_raw.items())}
            ),
            channel_counts=MappingProxyType(counts(y_chan)),
            channel_meaning=str(handle.attrs.get("channel_meaning", "unspecified")),
            source=str(handle.attrs.get("source", "unspecified")),
            subset_name=str(handle.attrs.get("subset_name", "unspecified")),
        )


def _select_indices(handle, summary: DatasetSummary, config: HarnessConfig) -> np.ndarray:
    y_mod = np.asarray(handle["y_mod"][:], dtype=np.int64)
    y_snr = np.asarray(handle["y_snr"][:], dtype=np.int64)
    y_chan = np.asarray(handle["y_chan"][:], dtype=np.int64)
    mask = np.ones(summary.frame_count, dtype=bool)
    if config.modulation is not None:
        if config.modulation not in summary.modulation_to_id:
            raise ValueError(f"Unknown modulation {config.modulation!r}")
        mask &= y_mod == summary.modulation_to_id[config.modulation]
    if config.snr_db is not None:
        mask &= y_snr == config.snr_db
    if config.channel is not None:
        mask &= y_chan == config.channel
    candidates = np.flatnonzero(mask)
    candidates = candidates[candidates >= config.first_index]
    if len(candidates) < config.frame_count:
        raise ValueError(
            f"Only {len(candidates)} frames match the filters at/after index "
            f"{config.first_index}; requested {config.frame_count}"
        )
    if config.random_selection:
        rng = np.random.default_rng(config.selection_seed)
        selected = np.sort(rng.choice(candidates, size=config.frame_count, replace=False))
    else:
        selected = candidates[: config.frame_count]
    return selected.astype(np.int64, copy=False)


def _quiet_iq(rng: np.random.Generator, samples: int, noise_power: float) -> np.ndarray:
    if samples == 0:
        return np.empty(0, dtype=np.complex64)
    scale = math.sqrt(noise_power / 2.0)
    return np.asarray(
        (rng.standard_normal(samples) + 1j * rng.standard_normal(samples)) * scale,
        dtype=np.complex64,
    )


def _frame_iq(raw_frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(raw_frame, dtype=np.float32)
    if frame.ndim != 2 or frame.shape[1] != 2:
        raise ValueError("Every X frame must have shape (samples, 2)")
    if not np.all(np.isfinite(frame)):
        raise ValueError("Dataset frame contains NaN or infinity")
    return np.asarray(frame[:, 0] + 1j * frame[:, 1], dtype=np.complex64)


def _match_events(
    captures: tuple[CaptureSidecar, ...], truths: tuple[FrameTruth, ...]
) -> tuple[tuple[EventMatch, ...], int, int, int]:
    candidates: list[tuple[int, float, int, int]] = []
    for event_index, sidecar in enumerate(captures):
        event = sidecar.event
        for truth_index, truth in enumerate(truths):
            overlap = max(
                0,
                min(event.detected_end_sample, truth.end_sample)
                - max(event.detected_start_sample, truth.start_sample),
            )
            if overlap:
                union = (
                    event.detected_end_sample
                    - event.detected_start_sample
                    + truth.end_sample
                    - truth.start_sample
                    - overlap
                )
                candidates.append((overlap, overlap / union, event_index, truth_index))
    used_events: set[int] = set()
    used_truths: set[int] = set()
    accepted: dict[int, EventMatch] = {}
    for overlap, iou, event_index, truth_index in sorted(candidates, reverse=True):
        if event_index in used_events or truth_index in used_truths:
            continue
        used_events.add(event_index)
        used_truths.add(truth_index)
        accepted[event_index] = EventMatch(event_index, truth_index, overlap, iou)
    matches = tuple(
        accepted.get(index, EventMatch(index, None, 0, 0.0)) for index in range(len(captures))
    )
    matched = len(used_truths)
    return matches, matched, len(truths) - matched, len(captures) - len(used_events)


def _progress(callback: ProgressCallback | None, fraction: float, message: str) -> None:
    if callback is not None:
        callback(max(0.0, min(1.0, float(fraction))), message)


def run_hdf5_pipeline(
    path: str | Path,
    config: HarnessConfig | None = None,
    *,
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> PipelineRunResult:
    """Construct a bounded labelled stream and run the production pipeline.

    Cancellation is observed between transport blocks.  It never rewinds or
    fabricates a partially committed block.
    """
    h5py = _require_h5py()
    config = config or HarnessConfig()
    summary = inspect_hdf5_dataset(path)
    id_to_mod = {value: key for key, value in summary.modulation_to_id.items()}
    rng = np.random.default_rng(config.selection_seed)
    chunks: list[np.ndarray] = []
    truths: list[FrameTruth] = []
    position = 0

    _progress(progress, 0.01, "Selecting dataset frames")
    with h5py.File(summary.path, "r") as handle:
        selected = _select_indices(handle, summary, config)
        labels_mod = np.asarray(handle["y_mod"][selected], dtype=np.int64)
        labels_snr = np.asarray(handle["y_snr"][selected], dtype=np.int64)
        labels_chan = np.asarray(handle["y_chan"][selected], dtype=np.int64)
        initial = _quiet_iq(rng, config.initial_quiet_samples, config.noise_power_linear)
        chunks.append(initial)
        position += len(initial)
        for order, dataset_index in enumerate(selected):
            if cancel_event is not None and cancel_event.is_set():
                raise RunCancelled("Dataset run cancelled before pipeline admission")
            frame = _frame_iq(handle["X"][int(dataset_index)])
            original_power = float(np.vdot(frame.astype(np.complex128), frame).real / len(frame))
            if original_power <= 0 or not math.isfinite(original_power):
                raise ValueError(f"Dataset frame {int(dataset_index)} has no finite positive power")
            if config.normalize_frames:
                nominal_ratio = 10.0 ** (float(labels_snr[order]) / 10.0)
                target_total_power = config.noise_power_linear * (1.0 + nominal_ratio)
                frame = np.asarray(
                    frame * math.sqrt(target_total_power / original_power), dtype=np.complex64
                )
            streamed_power = float(np.vdot(frame.astype(np.complex128), frame).real / len(frame))
            start = position
            chunks.append(frame)
            position += len(frame)
            truths.append(
                FrameTruth(
                    dataset_index=int(dataset_index),
                    start_sample=start,
                    end_sample=position,
                    modulation=id_to_mod[int(labels_mod[order])],
                    snr_db=int(labels_snr[order]),
                    channel=int(labels_chan[order]),
                    original_power=original_power,
                    streamed_power=streamed_power,
                )
            )
            quiet = _quiet_iq(rng, config.guard_samples, config.noise_power_linear)
            chunks.append(quiet)
            position += len(quiet)
            _progress(
                progress,
                0.03 + 0.12 * (order + 1) / len(selected),
                f"Prepared frame {order + 1}/{len(selected)}",
            )
        final = _quiet_iq(rng, config.final_quiet_samples, config.noise_power_linear)
        chunks.append(final)
        position += len(final)

    stream_iq = np.concatenate(chunks).astype(np.complex64, copy=False)
    stream_iq.setflags(write=False)
    stream = StreamInfo(
        stream_id="subset-test-ui",
        source_id=summary.path.name,
        sample_rate_hz=config.sample_rate_hz,
        center_frequency_hz=config.center_frequency_hz,
        channel_id="hdf5-test-harness",
    )
    if config.adaptive_noise:
        estimator = AdaptiveRobustNoiseEstimator(
            window_size=config.window_size,
            pfa_on=config.pfa_on,
            pfa_off=config.pfa_off,
            time_constant_s=0.5,
            sample_rate_hz=config.sample_rate_hz,
            initial_noise_power_linear=config.noise_power_linear,
            adaptation_window_size=config.adaptation_window_size,
            max_staleness_samples=max(position * 2, config.window_size),
        )
    else:
        estimator = GaussianNoiseEstimator(
            window_size=config.window_size,
            pfa_on=config.pfa_on,
            pfa_off=config.pfa_off,
            calibrated_noise_power_linear=config.noise_power_linear,
        )

    pipeline = StreamingCapturePipeline(
        stream_info=stream,
        ring_capacity=config.ring_capacity,
        window_size=config.window_size,
        hop_size=config.hop_size,
        pfa_on=config.pfa_on,
        pfa_off=config.pfa_off,
        noise_estimator=estimator,
        start_persistence=config.start_persistence,
        end_persistence=config.end_persistence,
        pre_trigger_samples=config.pre_trigger_samples,
        pre_signal_margin_samples=config.pre_signal_margin_samples,
        post_signal_samples=config.post_signal_samples,
        min_post_trigger_samples=config.min_post_trigger_samples,
        max_event_samples=config.max_event_samples,
        synchronous_analysis=True,
        skip_incomplete_quality=False,
        enable_feature_analysis=config.enable_features,
    )
    captures: list[CaptureSidecar] = []
    blocks = math.ceil(len(stream_iq) / config.transport_block_size)
    started = time.perf_counter()
    try:
        for sequence_number, start in enumerate(
            range(0, len(stream_iq), config.transport_block_size)
        ):
            if cancel_event is not None and cancel_event.is_set():
                raise RunCancelled("Dataset run cancelled after a committed transport block")
            block_iq = stream_iq[start : start + config.transport_block_size]
            captures.extend(
                pipeline.process_block(
                    IQBlock(
                        stream=stream,
                        sequence_number=sequence_number,
                        start_sample=start,
                        timestamp_ns=None,
                        iq=block_iq,
                    )
                )
            )
            if sequence_number % 4 == 0 or sequence_number + 1 == blocks:
                _progress(
                    progress,
                    0.15 + 0.78 * (sequence_number + 1) / blocks,
                    f"Processed block {sequence_number + 1}/{blocks}",
                )
        captures.extend(pipeline.flush())
    finally:
        pipeline.shutdown(drain=True, timeout=10.0)
    elapsed = time.perf_counter() - started
    captures_tuple = tuple(captures)
    truths_tuple = tuple(truths)
    matches, matched, missed, extras = _match_events(captures_tuple, truths_tuple)

    stride = max(1, math.ceil(len(stream_iq) / config.display_point_limit))
    display_indices = np.arange(0, len(stream_iq), stride, dtype=np.int64)
    display_magnitudes = np.abs(stream_iq[::stride]).astype(np.float32, copy=False)
    display_indices.setflags(write=False)
    display_magnitudes.setflags(write=False)
    _progress(progress, 1.0, "Run complete")
    return PipelineRunResult(
        dataset=summary,
        config=config,
        selected_indices=tuple(map(int, selected)),
        truths=truths_tuple,
        captures=captures_tuple,
        matches=matches,
        matched_truths=matched,
        missed_truths=missed,
        extra_events=extras,
        total_stream_samples=len(stream_iq),
        elapsed_seconds=elapsed,
        throughput_samples_per_second=len(stream_iq) / elapsed if elapsed else math.inf,
        final_noise_power_linear=getattr(estimator, "noise_power_linear", None),
        display_sample_indices=display_indices,
        display_magnitudes=display_magnitudes,
    )
