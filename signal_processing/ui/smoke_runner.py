"""Replay positioned detector datasets and compare outcomes with supplied truth.

Annotations are evaluator inputs. Only receiver metadata and the explicitly
authorized noise prefix configure the pipeline; emitter truth is never fed to it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from signal_processing.capture.sidecar import CaptureSidecar
from signal_processing.core.contracts import IQBlock, StreamInfo, NoiseReferenceDomain
from signal_processing.detection.noise_estimation import GaussianNoiseEstimator
from signal_processing.runtime.pipeline import StreamingCapturePipeline
from signal_processing.ui.hdf5_runner import RunCancelled


@dataclass(frozen=True)
class SmokeConfig:
    window_size: int = 256
    hop_size: int = 64
    block_size: int = 1024
    pre_trigger_samples: int = 256
    post_trigger_samples: int = 256
    ring_capacity: int = 16384
    max_event_samples: int = 131072
    start_persistence: int = 2
    end_persistence: int = 3
    pfa_on: float = 1e-4
    pfa_off: float = 1e-2
    minimum_iou: float = 0.2
    enable_features: bool = True
    detector_profile: str = "compatibility"

    def __post_init__(self):
        for name in (
            "window_size",
            "hop_size",
            "block_size",
            "ring_capacity",
            "max_event_samples",
            "start_persistence",
            "end_persistence",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("pre_trigger_samples", "post_trigger_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 1 <= self.hop_size <= self.window_size <= 65536:
            raise ValueError("Require 1 <= H <= N <= 65536")
        if not 0 < self.pfa_on < self.pfa_off < 1:
            raise ValueError("Require 0 < Pfa ON < Pfa OFF < 1")
        if not 0 < self.minimum_iou <= 1:
            raise ValueError("Minimum IoU must be in (0, 1]")
        if type(self.enable_features) is not bool:
            raise ValueError("enable_features must be bool")
        if self.detector_profile not in ("compatibility", "ar1_whitened", "ar1_multires", "ar1_spectral"):
            raise ValueError("Unknown detector profile")
        if self.detector_profile in ("ar1_multires", "ar1_spectral") and (self.window_size != 256 or self.hop_size != 8):
            raise ValueError("AR1 multires candidate requires N=256/H=8, with 32/64/256 subwindows")


@dataclass(frozen=True)
class SmokeRecord:
    record_id: str
    scenario: str
    iq_path: Path
    annotations_path: Path
    masks_path: Path
    iq_sha256: str


@dataclass(frozen=True)
class EventPreview:
    """UI-only coordinates, never a SignalEvent or a reusable analysis input."""

    event_id: str
    detected_start_sample: int
    detected_end_sample: int
    event_start_sample: int
    event_end_sample: int
    trigger_sample: int
    pre_trigger_samples: int
    complete: bool


@dataclass(frozen=True)
class CapturePreview:
    """Analysis metadata without retaining the event's raw IQ."""

    event: EventPreview
    stream_epoch: int
    processing_generation: int | None
    termination_reason: str
    is_truncated: bool
    known_lost_samples: int
    unknown_loss: bool
    noise_valid: bool
    noise_provenance: str
    feature_status: str
    feature_reason: str | None
    features: object
    analysis_result: object
    coarse_detected_start_sample: int | None
    coarse_detected_end_sample: int | None
    boundary_method: str | None


@dataclass(frozen=True)
class SmokeResult:
    record: SmokeRecord
    config: SmokeConfig
    annotations_json: str
    captures: tuple[CaptureSidecar | CapturePreview, ...]
    # (activity index, capture index, IoU), maximum cardinality then maximum IoU.
    matches: tuple[tuple[int, int, float], ...]
    metrics: dict
    display_samples: np.ndarray
    display_magnitude: np.ndarray

    @property
    def annotations(self) -> dict:
        return json.loads(self.annotations_json)


def compact_smoke_result(result: SmokeResult) -> SmokeResult:
    """Drop raw event ownership after exact-IQ checks; preserve export/UI values."""
    captures = []
    for capture in result.captures:
        event = EventPreview(
            **{name: getattr(capture.event, name) for name in EventPreview.__dataclass_fields__}
        )
        captures.append(
            CapturePreview(
                event=event,
                **{
                    name: getattr(capture, name)
                    for name in CapturePreview.__dataclass_fields__
                    if name != "event"
                },
            )
        )
    return replace(result, captures=tuple(captures))


def _local_path(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Dataset path escapes selected folder: {relative}")
    if not target.is_file():
        raise FileNotFoundError(target)
    return target


def inspect_smoke_dataset(path: str | Path) -> tuple[SmokeRecord, ...]:
    root = Path(path).resolve()
    if root.is_file():
        root = root.parent
    metadata = json.loads((root / "dataset.json").read_text(encoding="utf-8-sig"))
    if metadata.get("status") != "COMPLETE":
        raise ValueError("Dataset generation is not COMPLETE")
    records = []
    for line in (root / "manifest.jsonl").read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records.append(
            SmokeRecord(
                str(row["record_id"]),
                str(row["scenario"]),
                _local_path(root, row["iq_file"]),
                _local_path(root, row["annotations_file"]),
                _local_path(root, row["masks_file"]),
                str(row["iq_sha256"]),
            )
        )
    if not records or len(records) > 512:
        raise ValueError("The interactive bench supports 1–512 records per dataset")
    for key in ("records_expected", "records_completed"):
        if key in metadata and metadata[key] != len(records):
            raise ValueError(f"Dataset {key} contradicts manifest count")
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("Duplicate record IDs in manifest")
    return tuple(records)


def match_activity(truth, detected, minimum_iou=0.2):
    """One-to-one maximum-cardinality assignment, with IoU breaking ties."""
    if not truth or not detected:
        return ()
    ious = np.zeros((len(truth), len(detected)))
    for i, (a, b) in enumerate(truth):
        for j, (c, d) in enumerate(detected):
            overlap = max(0, min(b, d) - max(a, c))
            ious[i, j] = overlap / (b - a + d - c - overlap)
    eligible = ious >= minimum_iou
    # A cardinality bonus larger than every possible sum of IoUs.
    score = np.where(eligible, min(ious.shape) + 1 + ious, 0)
    rows, cols = linear_sum_assignment(score, maximize=True)
    return tuple((int(i), int(j), float(ious[i, j])) for i, j in zip(rows, cols) if eligible[i, j])


def _interval(row: dict, size: int) -> tuple[int, int]:
    a, b = row["start_sample"], row["end_sample"]
    if type(a) is not int or type(b) is not int or not 0 <= a < b <= size:
        raise ValueError(f"Invalid half-open truth interval [{a}, {b})")
    return a, b


def _validated_record(record: SmokeRecord):
    annotations = json.loads(record.annotations_path.read_text(encoding="utf-8-sig"))
    if annotations["record_id"] != record.record_id:
        raise ValueError("Manifest/annotation record identity mismatch")
    iq = np.load(record.iq_path, allow_pickle=False, mmap_mode="r")
    if iq.ndim != 1 or iq.dtype != np.complex64 or len(iq) != annotations["num_samples"]:
        raise ValueError("IQ shape/dtype/length contradicts annotations")
    if not 1 <= len(iq) <= 4_000_000 or not np.all(np.isfinite(iq)):
        raise ValueError("Records must contain 1–4,000,000 finite complex64 samples")
    # This dataset hashes contiguous IQ payload bytes, excluding the .npy header.
    digest = hashlib.sha256(memoryview(iq).cast("B")).hexdigest()
    if digest != record.iq_sha256 or digest != annotations["iq_sha256"]:
        raise ValueError("IQ payload checksum does not match supplied manifest and annotations")
    prefix = annotations["calibration_prefix"]
    if prefix.get("noise_only") is not True or prefix["start_sample"] != 0:
        raise ValueError("An explicitly authorized noise-only prefix starting at zero is required")
    settling, training = prefix["settling_samples"], prefix["training_samples"]
    if (
        type(settling) is not int
        or type(training) is not int
        or settling < 0
        or training < 1
        or settling + training != prefix["end_sample"]
        or prefix["end_sample"] >= len(iq)
    ):
        raise ValueError("Invalid calibration settling/training budget")
    if annotations["background"]["model"] not in ("white", "ar1"):
        raise ValueError(
            "Unknown noise model; this comparison supports white and AR1 stress inputs"
        )
    expected_counts = np.zeros(len(iq), dtype=np.int32)
    for emitter in annotations["emitters"]:
        a, b = _interval(emitter, len(iq))
        if a < prefix["end_sample"]:
            raise ValueError("Emitter truth contradicts noise-only calibration prefix")
        expected_counts[a:b] += 1
    truth = np.zeros(len(iq), dtype=bool)
    intervals = []
    for row in annotations["fullband_activity_intervals"]:
        a, b = _interval(row, len(iq))
        if intervals and a <= intervals[-1][1]:
            raise ValueError("Activity intervals must be sorted, disjoint union regions")
        intervals.append((a, b))
        truth[a:b] = True
    if not np.array_equal(truth, expected_counts > 0):
        raise ValueError("Activity truth is not the union of all emitter intervals")
    with np.load(record.masks_path, allow_pickle=False) as masks:
        if not np.array_equal(masks["any_signal"], truth):
            raise ValueError("Supplied any_signal mask contradicts annotations")
        if not np.array_equal(masks["active_emitter_count"], expected_counts):
            raise ValueError("Supplied emitter count mask contradicts annotations")
    return iq, annotations, truth, intervals


def run_smoke_record(
    record: SmokeRecord,
    config: SmokeConfig | None = None,
    *,
    progress: Callable[[float, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> SmokeResult:
    config = config or SmokeConfig()
    iq, annotations, truth, intervals = _validated_record(record)
    prefix = annotations["calibration_prefix"]
    stream = StreamInfo(
        record.record_id,
        record.iq_path.parents[2].name,
        annotations["sample_rate_hz"],
        annotations["receiver_center_frequency_hz"],
    )
    estimator = GaussianNoiseEstimator(
        window_size=config.window_size,
        pfa_on=config.pfa_on,
        pfa_off=config.pfa_off,
        warmup_samples=prefix["training_samples"],
        warmup_settling_samples=prefix["settling_samples"],
    )
    ar1_model = None
    start_at = 0
    profile_id = None
    if config.detector_profile != "compatibility":
        from signal_processing.detection.ar_noise import fit_ar1_noise
        ar1_model = fit_ar1_noise(iq[prefix["settling_samples"]:prefix["end_sample"]], stream.sample_rate_hz)
        domain = NoiseReferenceDomain(stream.sample_rate_hz, preprocessing_id=ar1_model.model_id)
        estimator = GaussianNoiseEstimator(window_size=config.window_size,
            pfa_on=config.pfa_on, pfa_off=config.pfa_off,
            calibrated_noise_power_linear=ar1_model.innovation_power, reference_domain=domain)
        start_at = prefix["end_sample"]
        profile_id = "ar1_candidate_" + hashlib.sha256(json.dumps(
            {"config":asdict(config),"model":ar1_model.model_id},sort_keys=True).encode()).hexdigest()
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
        pre_signal_margin_samples=128,
        post_signal_samples=config.post_trigger_samples,
        min_post_trigger_samples=config.post_trigger_samples,
        max_event_samples=config.max_event_samples,
        synchronous_analysis=True,
        skip_incomplete_quality=False,
        enable_feature_analysis=config.enable_features,
        ar1_model=ar1_model,
        detection_profile_id=profile_id,
        ar1_windows=(32,64,256) if config.detector_profile in ("ar1_multires","ar1_spectral") else (),
        ar1_subbands=8 if config.detector_profile=="ar1_spectral" else 0,
        refine_ar1_boundaries=config.detector_profile in ("ar1_multires","ar1_spectral"),
    )
    captures = []
    started = time.perf_counter()
    try:
        for seq, start in enumerate(range(start_at, len(iq), config.block_size)):
            if cancel_event is not None and cancel_event.is_set():
                raise RunCancelled("Record replay cancelled between transport blocks")
            captures.extend(
                pipeline.process_block(
                    IQBlock(stream, seq, start, None, iq[start : start + config.block_size])
                )
            )
            if len(captures) > 512:
                raise RuntimeError(
                    "Interactive capture limit (512/record) exceeded; reduce workload"
                )
            if progress and seq % 16 == 0:
                progress(start / len(iq), f"{record.record_id}: sample {start:,}/{len(iq):,}")
        captures.extend(pipeline.flush())
        if len(captures) > 512:
            raise RuntimeError("Interactive capture limit (512/record) exceeded at EOF")
    finally:
        pipeline.shutdown(drain=True, timeout=10)
    elapsed = time.perf_counter() - started
    detected = [(c.event.detected_start_sample, c.event.detected_end_sample) for c in captures]
    matches = match_activity(intervals, detected, config.minimum_iou)
    predicted = np.zeros(len(iq), dtype=bool)
    for a, b in detected:
        predicted[a:b] = True
    evaluation = slice(prefix["end_sample"], len(iq))
    tp = int(np.count_nonzero((truth & predicted)[evaluation]))
    fp = int(np.count_nonzero((~truth & predicted)[evaluation]))
    fn = int(np.count_nonzero((truth & ~predicted)[evaluation]))
    emitter_coverage = [
        float(np.mean(predicted[e["start_sample"] : e["end_sample"]]))
        for e in annotations["emitters"]
    ]
    metrics = dict(
        detector_profile=config.detector_profile,
        ar1_model=ar1_model.to_dict() if ar1_model else None,
        detection_profile_id=profile_id,
        noise_model=annotations["background"]["model"],
        sweep_snr_db=annotations.get("sweep_snr_db"),
        repetition=annotations.get("repetition"),
        threshold_assumptions_match=ar1_model is not None or annotations["background"]["model"] == "white",
        ar1_diagnostics_are_not_certification=ar1_model is not None,
        threshold_support=("Fitted AR1 innovations; model only, parameter uncertainty and Pfa NOT certified" if ar1_model is not None else
            "Gaussian model only; no held-out certification" if annotations["background"]["model"] == "white" else
            "MODEL MISMATCH: iid Gaussian thresholds on correlated AR1; stress comparison only"),
        samples=len(iq),
        evaluation_start=prefix["end_sample"],
        evaluation_samples=len(iq) - prefix["end_sample"],
        noise_only_exposure_s=float(np.count_nonzero(~truth[evaluation])) / stream.sample_rate_hz,
        truth_activities=len(intervals),
        truth_emitters=len(annotations["emitters"]),
        captures=len(captures),
        matched=len(matches),
        event_recall=len(matches)/len(intervals) if intervals else None,
        event_precision=len(matches)/len(captures) if captures else None,
        misses=len(intervals) - len(matches),
        extras=len(captures) - len(matches),
        noise_only_events=sum(not np.any(truth[a:b]) for a, b in detected),
        sample_true_positive=tp,
        sample_false_positive=fp,
        sample_false_negative=fn,
        sample_precision=tp / (tp + fp) if tp + fp else None,
        sample_recall=tp / (tp + fn) if tp + fn else None,
        emitter_coverage=emitter_coverage,
        elapsed_seconds=elapsed,
        throughput_samples_per_second=len(iq) / elapsed,
        calibrated_noise_power=estimator.noise_power_linear,
        calibration_training_samples=prefix["training_samples"],
        feature_failures=sum(c.feature_status == "failed" for c in captures),
        exact_capture_iq=all(
            np.array_equal(c.event.iq, iq[c.event.event_start_sample : c.event.event_end_sample])
            for c in captures
        ),
    )
    # Peak envelope avoids hiding short bursts in a strided preview.
    step = max(1, math.ceil(len(iq) / 4096))
    indices = np.arange(0, len(iq), step, dtype=np.int64)
    magnitude = np.maximum.reduceat(np.abs(iq), indices)
    indices.setflags(write=False)
    magnitude.setflags(write=False)
    if progress:
        progress(1.0, f"{record.record_id}: complete")
    return SmokeResult(
        record,
        config,
        json.dumps(annotations),
        tuple(captures),
        matches,
        metrics,
        indices,
        magnitude,
    )


def comparison_rows(result: SmokeResult, include="combined") -> list[dict]:
    """CSV-ready rows. Missed truth, unmatched results and empty records persist."""
    if include not in ("combined", "truth", "results"):
        raise ValueError("include must be combined, truth or results")
    ann = result.annotations
    base = dict(
        record_id=result.record.record_id,
        scenario=result.record.scenario,
        sample_rate_hz=ann["sample_rate_hz"],
        noise_model=ann["background"]["model"],
        sweep_snr_db=ann.get("sweep_snr_db"),
        repetition=ann.get("repetition"),
        threshold_support=result.metrics["threshold_support"],
        iq_sha256=result.record.iq_sha256,
        interval_convention="[start,end), record-local samples",
        evaluation_start_sample=result.metrics["evaluation_start"],
        minimum_iou=result.config.minimum_iou,
        config_json=json.dumps(asdict(result.config)),
    )
    rows = [
        {
            **base,
            "row_type": "record_summary",
            **{key: value for key, value in result.metrics.items() if key != "emitter_coverage"},
        }
    ]
    by_truth = {i: (j, score) for i, j, score in result.matches}
    by_event = {j: (i, score) for i, j, score in result.matches}
    if include != "results":
        for i, emitter in enumerate(ann["emitters"]):
            rows.append(
                dict(
                    base,
                    row_type="emitter_truth",
                    **{
                        f"truth_{key}": value
                        for key, value in emitter.items()
                        if isinstance(value, (str, int, float, bool)) or value is None
                    },
                    truth_annotation_json=json.dumps(emitter),
                    temporal_coverage=result.metrics["emitter_coverage"][i],
                    attribution="temporal coverage only; no emitter classification/separation",
                )
            )
        for i, activity in enumerate(ann["fullband_activity_intervals"]):
            match = by_truth.get(i)
            event = result.captures[match[0]].event if match else None
            rows.append(
                dict(
                    base,
                    row_type="activity_truth",
                    activity_index=i,
                    truth_start_sample=activity["start_sample"],
                    truth_end_sample=activity["end_sample"],
                    contributing_emitter_ids=json.dumps(activity["contributing_emitter_ids"]),
                    match_status="matched" if match else "missed",
                    event_id=event.event_id if event else None,
                    iou=match[1] if match else None,
                    detected_start_sample=event.detected_start_sample if event else None,
                    detected_end_sample=event.detected_end_sample if event else None,
                    onset_error_samples=event.detected_start_sample - activity["start_sample"]
                    if event
                    else None,
                    end_error_samples=event.detected_end_sample - activity["end_sample"]
                    if event
                    else None,
                )
            )
    if include != "truth":
        for j, capture in enumerate(result.captures):
            event = capture.event
            match = by_event.get(j)
            quality = (
                asdict(capture.analysis_result.capture_quality) if capture.analysis_result else None
            )
            row = dict(
                base,
                row_type="detected_event",
                event_id=event.event_id,
                capture_epoch=capture.stream_epoch,
                processing_generation=capture.processing_generation,
                event_start_sample=event.event_start_sample,
                event_end_sample=event.event_end_sample,
                detected_start_sample=event.detected_start_sample,
                detected_end_sample=event.detected_end_sample,
                coarse_detected_start_sample=capture.coarse_detected_start_sample,
                coarse_detected_end_sample=capture.coarse_detected_end_sample,
                boundary_method=capture.boundary_method,
                trigger_sample=event.trigger_sample,
                pre_trigger_samples=event.pre_trigger_samples,
                complete=event.complete,
                termination_reason=capture.termination_reason,
                is_truncated=capture.is_truncated,
                known_lost_samples=capture.known_lost_samples,
                unknown_loss=capture.unknown_loss,
                noise_valid=capture.noise_valid,
                noise_provenance=capture.noise_provenance,
                feature_status=capture.feature_status,
                feature_reason=capture.feature_reason,
                quality_json=json.dumps(quality),
                match_status="matched" if match else "extra",
                activity_index=match[0] if match else None,
                iou=match[1] if match else None,
            )
            if capture.features:
                row["feature_schema_version"] = capture.features.feature_schema_version
                if capture.analysis_result:
                    row["extraction_policy_json"] = json.dumps(
                        capture.analysis_result.policy.to_dict()
                    )
                for name, feature in capture.features.features.items():
                    for suffix, value in (
                        ("value", feature.value if feature.valid else None),
                        ("valid", feature.valid),
                        ("unit", feature.unit),
                        ("reason", feature.reason),
                        ("interpretation", feature.interpretation),
                    ):
                        row[f"{name}.{suffix}"] = value
            rows.append(row)
    return rows


def export_smoke_csv(path: str | Path, results, include="combined") -> None:
    results = tuple(results)
    if not results:
        raise ValueError("No completed records to export")
    # Two passes keep CSV row materialization bounded to one record, even for 480 files.
    fields = list(
        dict.fromkeys(
            key for result in results for row in comparison_rows(result, include) for key in row
        )
    )
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerows(comparison_rows(result, include))
