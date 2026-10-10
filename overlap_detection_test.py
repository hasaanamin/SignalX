#!/usr/bin/env python3
"""Reproducible complex-IQ overlap test for the FYP signal_processing package.

Install: python -m pip install numpy scipy matplotlib
Generate only: python overlap_detection_test.py --plot
From your unpacked package root:
  python overlap_detection_test.py --package-src . --run-pipeline --plot \
      --check-partitions --enforce-integrity
Harder case:
  python overlap_detection_test.py --package-src . --run-pipeline \
      --target-snr-db -10 --noise-model ar1 --output overlap_hard

Writes mixture.npz (IQ plus separated simulation components), mixture.cf32
(little-endian interleaved float32 I/Q), ground_truth.json, optional figure,
and, when requested, pipeline_report.json plus exact captured IQ files.

SNR means active component power / expected FULL-BAND noise power. It is not
post-filter SNR or SINR. Carrier offsets are relative to receiver center.
Noise is never normalized using the test recording. The detector trains only
on a declared noise-only prefix, through its own causal detection path.

This is an offline functional test, not a scheduled real-time benchmark or a
certified Pd/Pfa experiment. Temporal overlap is NOT source identification.
In particular, a co-channel interferer can keep an energy detector ON before
the target starts. Separate target and interferer controls expose that case.
The pipeline adapter was exercised against examples2_repaired_2.3.0.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy import signal


SCRIPT_VERSION = "1.0.0"
DURATION_S = 0.320
TIMESTAMP_ORIGIN_NS = 1_700_000_000_000_000_000  # Synthetic clock, not wall time.
SIGNAL_KEYS = ("target", "cw", "bpsk", "adjacent_qpsk", "chirp", "cochannel_qpsk")


def power(x: np.ndarray) -> float:
    z = np.asarray(x, dtype=np.complex128)
    return float(np.vdot(z, z).real / len(z)) if len(z) else 0.0


def db_ratio(a: float, b: float) -> float | None:
    return 10.0 * math.log10(a / b) if a > 0 and b > 0 else None


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def iq_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x, dtype="<c8").tobytes()).hexdigest()


def rrc_taps(samples_per_symbol: int, beta: float = 0.35, span: int = 10) -> np.ndarray:
    """Unit-energy root-raised-cosine FIR, including both removable limits."""
    t = np.arange(-span * samples_per_symbol // 2,
                  span * samples_per_symbol // 2 + 1, dtype=float) / samples_per_symbol
    h = np.empty_like(t)
    at_zero = np.isclose(t, 0.0, atol=1e-14, rtol=0.0)
    at_limit = np.isclose(np.abs(t), 1.0 / (4.0 * beta), atol=1e-14, rtol=0.0)
    ordinary = ~(at_zero | at_limit)
    u = t[ordinary]
    h[ordinary] = (np.sin(np.pi * u * (1.0 - beta)) +
                   4.0 * beta * u * np.cos(np.pi * u * (1.0 + beta))) / (
                       np.pi * u * (1.0 - (4.0 * beta * u) ** 2))
    h[at_zero] = 1.0 + beta * (4.0 / np.pi - 1.0)
    h[at_limit] = beta / np.sqrt(2.0) * (
        (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta)) +
        (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta)))
    return h / np.sqrt(np.dot(h, h))


def modulated_burst(kind: str, n: int, sps: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Pulse-shaped BPSK/QPSK; crop FIR delay using extra random edge symbols."""
    taps = rrc_taps(sps)
    padding_symbols = 12
    count = math.ceil(n / sps) + 2 * padding_symbols + 1
    if kind == "bpsk":
        symbols = (2 * rng.integers(0, 2, count) - 1).astype(np.complex128)
    elif kind == "qpsk":
        symbols = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, count)))
    else:
        raise ValueError(f"Unknown modulation {kind!r}")
    shaped = signal.upfirdn(taps, symbols, up=sps)
    first = len(taps) // 2 + padding_symbols * sps
    return shaped[first:first + n].copy()


def make_fixture(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict]:
    fs = args.sample_rate
    n = round(DURATION_S * fs)
    noise_power = 10.0 ** (args.noise_dbfs / 10.0)
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(args.seed).spawn(7)]
    noise_rng = streams[0]
    if args.noise_model == "white":
        noise = np.sqrt(noise_power / 2.0) * (
            noise_rng.standard_normal(n) + 1j * noise_rng.standard_normal(n))
    else:
        rho = args.ar1_rho
        # Burn-in, then stationary AR(1) variance = noise_power; no sample rescaling.
        burn = max(4096, math.ceil(math.log(1e-12) / math.log(rho))) if rho else 4096
        innovations = np.sqrt(noise_power * (1.0 - rho ** 2) / 2.0) * (
            noise_rng.standard_normal(n + burn) + 1j * noise_rng.standard_normal(n + burn))
        noise = signal.lfilter([1.0], [1.0, -rho], innovations)[burn:]
    parts = {"noise": noise.astype(np.complex64)}
    parts.update({key: np.zeros(n, dtype=np.complex64) for key in SIGNAL_KEYS})
    truths = []

    def add(key: str, kind: str, start_s: float, end_s: float,
            snr_db: float, offset_fraction: float,
            sps: int = 40, end_offset_fraction: float | None = None) -> None:
        rng = streams[SIGNAL_KEYS.index(key) + 1]
        a, b = round(start_s * fs), round(end_s * fs)
        count = b - a
        offset = offset_fraction * fs
        local_t = np.arange(count, dtype=float) / fs
        phase = float(rng.uniform(-np.pi, np.pi))
        if kind in ("bpsk", "qpsk"):
            z = modulated_burst(kind, count, sps, rng)
            z *= np.exp(1j * (2 * np.pi * offset * (a / fs + local_t) + phase))
        elif kind == "cw":
            z = np.exp(1j * (2 * np.pi * offset * (a / fs + local_t) + phase))
        elif kind == "chirp":
            end_frequency = float(end_offset_fraction) * fs
            slope = (end_frequency - offset) / (count / fs)
            z = np.exp(1j * (phase + 2 * np.pi * (
                offset * (a / fs + local_t) + 0.5 * slope * local_t ** 2)))
        else:
            raise ValueError(kind)
        # Smooth RF burst edges; labels include ramp samples, not idealized flat tops.
        ramp = min(round(0.000128 * fs), count // 8)
        if ramp > 1:
            taper = np.sin(np.linspace(0.0, np.pi / 2.0, ramp)) ** 2
            z[:ramp] *= taper
            z[-ramp:] *= taper[::-1]
        requested_power = noise_power * 10.0 ** (snr_db / 10.0)
        z *= np.sqrt(requested_power / power(z))
        parts[key][a:b] += z.astype(np.complex64)
        measured_power = power(parts[key][a:b])
        truth = dict(id=f"{key}_{sum(t['component'] == key for t in truths) + 1}",
                     component=key, role="target" if key == "target" else "interferer",
                     modulation=kind, start_sample=a, end_sample=b,
                     start_s=a / fs, end_s=b / fs, frequency_offset_hz=offset,
                     requested_snr_fullband_db=snr_db,
                     active_power_linear=measured_power,
                     measured_snr_vs_expected_noise_db=db_ratio(measured_power, noise_power),
                     approximate_bandwidth_hz=(1.35 * fs / sps if kind in ("bpsk", "qpsk") else None))
        if end_offset_fraction is not None:
            truth["end_frequency_offset_hz"] = end_offset_fraction * fs
        truths.append(truth)

    # One target emitter, repeated four times. The last three meet progressively
    # harder temporal/spectral overlaps. All fractions scale with the sample rate.
    for start, end in ((0.065, 0.095), (0.125, 0.155), (0.185, 0.215), (0.245, 0.275)):
        add("target", "qpsk", start, end, args.target_snr_db, 0.08)
    add("cw", "cw", 0.100, 0.285, 25.0, -0.20)
    add("bpsk", "bpsk", 0.145, 0.235, 14.0, 0.24, sps=25)
    add("adjacent_qpsk", "qpsk", 0.120, 0.220, 10.0, 0.135, sps=50)
    add("chirp", "chirp", 0.175, 0.230, 8.0, -0.06, end_offset_fraction=0.18)
    add("cochannel_qpsk", "qpsk", 0.235, 0.295, 6.0, 0.08)
    # Sum in float64/complex128, round ONCE to the complex64 receiver contract.
    parts["iq"] = sum((parts[k].astype(np.complex128)
                       for k in ("noise", *SIGNAL_KEYS)), np.zeros(n, np.complex128)).astype(np.complex64)
    if not all(np.all(np.isfinite(v)) for v in parts.values()):
        raise ValueError("Fixture overflowed; choose finite, practical power levels")
    quiet_end = min(t["start_sample"] for t in truths)
    settling, training = round(0.002 * fs), round(0.032 * fs)
    if settling + training + 4 * args.window_size >= quiet_end:
        raise ValueError("Detection window too long for the declared noise-only prefix")
    band = [0.05 * fs, 0.11 * fs]
    target_sinr = []
    interferers = sum((parts[k].astype(np.complex128) for k in SIGNAL_KEYS[1:]), np.zeros(n, np.complex128))
    for t in (t for t in truths if t["role"] == "target"):
        a, b = t["start_sample"], t["end_sample"]
        target_sinr.append(dict(truth_id=t["id"],
            realized_fullband_snr_db=db_ratio(power(parts["target"][a:b]), power(parts["noise"][a:b])),
            realized_fullband_sinr_db=db_ratio(power(parts["target"][a:b]), power(parts["noise"][a:b] + interferers[a:b]))))
    metadata = dict(script_version=SCRIPT_VERSION, seed=args.seed,
        sample_rate_hz=fs, center_frequency_hz=args.center_frequency,
        num_samples=n, duration_s=n / fs, dtype="complex64",
        raw_file_format="mixture.cf32: little-endian float32 I,Q,I,Q; no header",
        iq_sha256=iq_hash(parts["iq"]), noise_model=args.noise_model,
        ar1_rho=args.ar1_rho if args.noise_model == "ar1" else None,
        expected_noise_power_linear=noise_power, realized_noise_power_linear=power(parts["noise"]),
        noise_dbfs_convention="10*log10(E[abs(I+jQ)**2]); unit complex amplitude is 0 dBFS",
        clipping_or_recording_normalization_applied=False,
        target_monitored_band_offsets_hz=band,
        target_bandwidth_note="RRC alpha=0.35, Rs=Fs/40; finite burst/taper has spectral tails",
        quiet_prefix_end_sample=quiet_end, warmup_settling_samples=settling,
        warmup_training_samples=training, detection_admission_sample=settling + training,
        snr_definition="active component mean power / expected full-band noise power",
        target_fullband_metrics=target_sinr, truths=truths,
        limitations=["Temporal overlap with a target is not identification of that target.",
            "An energy detector may merge simultaneous or closely spaced emitters.",
            "Configured-band filtering cannot identify/separate co-channel emitters.",
            "Colored-noise thresholds are a stress test, not certified Gaussian-model Pfa.",
            "Nominal model Pfa values are not a measured or certified event false-alarm rate.",
            "Offline timing is not scheduled acquisition reliability or a hardware test."])
    return parts, metadata


def intervals_union_length(intervals: list[tuple[int, int]]) -> int:
    total = 0
    right = None
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if right is None or a > right:
            total += b - a
        elif b > right:
            total += b - right
        right = max(b, right) if right is not None else b
    return total


def overlap_fraction(a: int, b: int, spans: list[tuple[int, int]]) -> float:
    return intervals_union_length([(max(a, s), min(b, e)) for s, e in spans]) / (b - a)


def sample_ns(sample: int, fs: float) -> int:
    return TIMESTAMP_ORIGIN_NS + round(Fraction(sample * 1_000_000_000, 1) / Fraction(fs))


def summarize_capture(sc, received: np.ndarray, args: argparse.Namespace) -> dict:
    e = sc.event
    a, b, trigger = e.event_start_sample, e.event_end_sample, e.trigger_sample
    bounds_ok = 0 <= a < b <= len(received)
    exact = bounds_ok and np.array_equal(e.iq, received[a:b])
    pre_available = trigger >= args.pre_trigger
    expected_start = max(0, min(trigger - args.pre_trigger,
                                e.detected_start_sample - args.pre_margin))
    checks = dict(received_iq_exact=bool(exact), capture_bounds_valid=bool(bounds_ok),
        iq_read_only=not e.iq.flags.writeable,
        requested_leading_extent_retained=(a == expected_start),
        pre_trigger_available=pre_available,
        pre_trigger_retained=(trigger - a >= args.pre_trigger) if pre_available else None,
        pre_trigger_iq_exact=(bool(np.array_equal(e.iq[:trigger - a], received[a:trigger])) if bounds_ok else False),
        timestamp_exact=(e.timestamp_ns == sample_ns(a, args.sample_rate)),
        no_declared_loss=(e.dropped_samples == 0 and sc.known_lost_samples == 0 and not sc.unknown_loss),
        complete=bool(e.complete), not_truncated=not sc.is_truncated)
    feature_values = {}
    if sc.features is not None:
        for name, value in sc.features.features.items():
            scalar = float(value.value)
            feature_values[name] = dict(value=scalar if math.isfinite(scalar) else None,
                                        unit=value.unit, valid=bool(value.valid), reason=value.reason)
    feature_ok = (sc.feature_status == "skipped") if args.no_features else (
        sc.feature_status == "ok" and sc.features is not None)
    checks["feature_delivery_ok"] = feature_ok
    return dict(event_id=e.event_id, stream_epoch=sc.stream_epoch,
        event_start_sample=a, event_end_sample=b,
        detected_start_sample=e.detected_start_sample, detected_end_sample=e.detected_end_sample,
        trigger_sample=trigger, confirmation_sample=sc.confirmation_sample,
        pre_trigger_samples_actual=trigger - a,
        actual_leading_samples=sc.actual_leading_samples, actual_trailing_samples=sc.actual_trailing_samples,
        boundary_uncertainty_samples=sc.boundary_uncertainty_samples,
        termination_reason=sc.termination_reason, iq_sha256=iq_hash(e.iq),
        feature_status=sc.feature_status, feature_reason=sc.feature_reason,
        valid_feature_count=sum(v["valid"] for v in feature_values.values()),
        invalid_feature_count=sum(not v["valid"] for v in feature_values.values()),
        features=feature_values, checks=checks)


def capture_fingerprint(rows: list[dict]) -> list[tuple]:
    names = ("event_start_sample", "event_end_sample", "detected_start_sample",
             "detected_end_sample", "trigger_sample", "confirmation_sample",
             "actual_leading_samples", "actual_trailing_samples",
             "boundary_uncertainty_samples", "termination_reason", "iq_sha256")
    return [tuple(r[key] for key in names) for r in rows]


def run_one(received: np.ndarray, case: str, mode: str, args: argparse.Namespace,
            meta: dict, api: dict, random_chunks: bool = False,
            capture_dir: Path | None = None) -> tuple[list[dict], dict]:
    stream = api["StreamInfo"](stream_id=f"overlap-{case}-{mode}", source_id="synthetic-overlap",
        sample_rate_hz=args.sample_rate, center_frequency_hz=args.center_frequency)
    estimator = api["GaussianNoiseEstimator"](window_size=args.window_size,
        pfa_on=args.pfa_on, pfa_off=args.pfa_off,
        warmup_samples=meta["warmup_training_samples"],
        warmup_settling_samples=meta["warmup_settling_samples"])
    pipeline = api["StreamingCapturePipeline"](stream_info=stream,
        detector_mode="full_band" if mode == "full" else "configured_band",
        monitored_band_hz=None if mode == "full" else tuple(meta["target_monitored_band_offsets_hz"]),
        noise_estimator=estimator, window_size=args.window_size, hop_size=args.hop_size,
        pfa_on=args.pfa_on, pfa_off=args.pfa_off,
        ring_capacity=max(16384, 4 * (args.window_size + args.pre_trigger + args.pre_margin)),
        start_persistence=2, end_persistence=3,
        pre_trigger_samples=args.pre_trigger, pre_signal_margin_samples=args.pre_margin,
        post_signal_samples=args.post_margin, min_post_trigger_samples=args.pre_trigger,
        max_event_samples=len(received) + args.pre_trigger + args.window_size,
        synchronous_analysis=True, enable_feature_analysis=not args.no_features,
        require_supported_pfa=False)
    rng = np.random.default_rng(args.seed + 9981)
    captures = []
    blocks = start = 0
    started = time.perf_counter()
    fault = None
    try:
        while start < len(received):
            size = int(rng.integers(1, 3 * args.block_size + 1)) if random_chunks else args.block_size
            end = min(start + size, len(received))
            block = api["IQBlock"](stream=stream, sequence_number=blocks,
                start_sample=start, timestamp_ns=sample_ns(start, args.sample_rate), iq=received[start:end])
            captures.extend(pipeline.process_block(block))
            blocks += 1
            start = end
        captures.extend(pipeline.flush())
    except Exception as exc:
        captures.extend(getattr(exc, "completed_captures", ()))
        fault = dict(type=type(exc).__name__, message=str(exc),
                     consumed_end_sample=getattr(exc, "consumed_end_sample", None),
                     received_end_sample=getattr(exc, "received_end_sample", None))
    finally:
        try:
            pipeline.shutdown(timeout=10.0)
        except Exception as exc:
            shutdown_fault = dict(type=type(exc).__name__, message=str(exc))
            fault = dict(processing_fault=fault, shutdown_fault=shutdown_fault)
    elapsed = time.perf_counter() - started
    rows = [summarize_capture(sc, received, args) for sc in captures]
    if capture_dir is not None:
        capture_dir.mkdir(parents=True, exist_ok=True)
        for i, sc in enumerate(captures):
            filename = f"capture_{i:03d}.npy"
            np.save(capture_dir / filename, sc.event.iq, allow_pickle=False)
            rows[i]["iq_file"] = str((capture_dir / filename).resolve())
    info = dict(blocks=blocks, elapsed_offline_s=elapsed,
                samples_offered=len(received), received_end_sample=pipeline.received_end_sample,
                consumed_end_sample=pipeline.consumed_end_sample,
                threshold_model=pipeline.detector.threshold_model,
                training_reference_power_linear=estimator.noise_power_linear,
                pfa_support_claim="model_only; no independent held-out certification",
                full_received_recording_consumed=(pipeline.received_end_sample == len(received)
                                                   and pipeline.consumed_end_sample == len(received)),
                unique_capture_identities=(len({(sc.stream_epoch, sc.event_id) for sc in captures}) == len(captures)),
                fault=fault)
    return rows, info


def score_case(rows: list[dict], case: str, meta: dict) -> dict:
    present = [t for t in meta["truths"] if (
        case == "mixture" or (case == "target_only" and t["role"] == "target") or
        (case == "interferers_only" and t["role"] == "interferer"))]
    detected_spans = [(r["detected_start_sample"], r["detected_end_sample"]) for r in rows]
    schedule = []
    for truth in (t for t in meta["truths"] if t["role"] == "target"):
        a, b = truth["start_sample"], truth["end_sample"]
        ids = [r["event_id"] for r in rows if r["detected_start_sample"] < b and r["detected_end_sample"] > a]
        new_confirmations = [r["confirmation_sample"] - a for r in rows
                             if a <= r["confirmation_sample"] < b]
        schedule.append(dict(truth_id=truth["id"], target_present=case in ("target_only", "mixture"),
            detected_time_coverage=overlap_fraction(a, b, detected_spans),
            overlapping_event_ids=ids, fresh_confirmation_inside_target=bool(new_confirmations),
            confirmation_delays_samples=new_confirmations,
            target_identity_verified=False))
    for row in rows:
        a, b = row["detected_start_sample"], row["detected_end_sample"]
        row["overlapping_truth_ids"] = [t["id"] for t in present
            if t["start_sample"] < b and t["end_sample"] > a]
    no_emitter_events = sum(not row["overlapping_truth_ids"] for row in rows)
    noise_exposure = (meta["num_samples"] - meta["detection_admission_sample"]) / meta["sample_rate_hz"]
    return dict(capture_count=len(rows), target_schedule_overlap=schedule,
        target_bursts_with_at_least_50pct_time_coverage=(sum(s["detected_time_coverage"] >= 0.5 for s in schedule)
            if case in ("target_only", "mixture") else None),
        target_bursts_with_new_confirmation=(sum(s["fresh_confirmation_inside_target"] for s in schedule)
            if case in ("target_only", "mixture") else None),
        target_bursts_below_50pct_time_coverage=(sum(s["detected_time_coverage"] < 0.5 for s in schedule)
            if case in ("target_only", "mixture") else None),
        target_bursts_without_fresh_confirmation=(sum(not s["fresh_confirmation_inside_target"] for s in schedule)
            if case in ("target_only", "mixture") else None),
        target_identification_or_separation_proven=False,
        events_with_no_emitter_time_overlap=no_emitter_events,
        noise_only_false_event_count=len(rows) if case == "noise_only" else None,
        noise_only_exposure_s=noise_exposure if case == "noise_only" else None,
        noise_only_event_rate_per_s=len(rows) / noise_exposure if case == "noise_only" else None,
        capture_checks_exercised=bool(rows),
        all_capture_integrity_checks_pass=(all(all(v is not False for v in row["checks"].values()) for row in rows)
                                           if rows else None))


def run_pipeline(parts: dict, meta: dict, args: argparse.Namespace, out: Path) -> tuple[dict, bool]:
    if args.package_src:
        root = Path(args.package_src).expanduser().resolve()
        if not (root / "signal_processing" / "__init__.py").is_file():
            raise ValueError("--package-src must be the directory CONTAINING signal_processing/")
        sys.path.insert(0, str(root))
    try:
        core = importlib.import_module("signal_processing.core.contracts")
        detection = importlib.import_module("signal_processing.detection.noise_estimation")
        runtime = importlib.import_module("signal_processing.runtime.pipeline")
        package = importlib.import_module("signal_processing")
    except ImportError as exc:
        raise RuntimeError("Cannot import your package. Run from its root with --package-src . "
                           "or install its wheel. Generation does not need this package.") from exc
    api = dict(StreamInfo=core.StreamInfo, IQBlock=core.IQBlock,
               GaussianNoiseEstimator=detection.GaussianNoiseEstimator,
               StreamingCapturePipeline=runtime.StreamingCapturePipeline)
    targets = (parts["noise"].astype(np.complex128) + parts["target"]).astype(np.complex64)
    interferers = sum((parts[k].astype(np.complex128) for k in ("noise", *SIGNAL_KEYS[1:])),
                     np.zeros(len(parts["iq"]), np.complex128)).astype(np.complex64)
    cases = dict(noise_only=parts["noise"], target_only=targets,
                 interferers_only=interferers, mixture=parts["iq"])
    modes = ("full", "band") if args.mode == "both" else (args.mode,)
    report = dict(script_version=SCRIPT_VERSION, package_version=getattr(package, "__version__", "unknown"),
        package_path=str(Path(package.__file__).resolve()), python=sys.version,
        platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
        fixture_sha256=meta["iq_sha256"],
        pipeline_configuration=dict(window_size=args.window_size, hop_size=args.hop_size,
            block_size=args.block_size, pfa_on=args.pfa_on, pfa_off=args.pfa_off,
            pre_trigger_samples=args.pre_trigger, pre_signal_margin_samples=args.pre_margin,
            post_signal_samples=args.post_margin, features_enabled=not args.no_features,
            synchronous_analysis=True, scheduled_real_time_test=False),
        limitations=meta["limitations"], runs={})
    integrity_ok = True
    for mode in modes:
        for case, iq in cases.items():
            label = f"{mode}/{case}"
            print(f"Running {label} ...", flush=True)
            rows, info = run_one(iq, case, mode, args, meta, api,
                                 capture_dir=out / "captures" / mode / case)
            scored = score_case(rows, case, meta)
            result = dict(runtime=info, summary=scored, events=rows)
            integrity_ok &= (info["fault"] is None and info["full_received_recording_consumed"] and
                             info["unique_capture_identities"] and scored["all_capture_integrity_checks_pass"] is not False)
            if args.check_partitions:
                # Rerun the entire stateful path on IDENTICAL input, fresh state,
                # random non-hop-aligned blocks. Avoid redoing feature extraction.
                check_args = argparse.Namespace(**vars(args))
                check_args.no_features = True
                other, other_info = run_one(iq, case, mode, check_args, meta, api, random_chunks=True)
                check_passed = (other_info["fault"] is None and info["fault"] is None and
                    other_info["full_received_recording_consumed"] and other_info["unique_capture_identities"] and
                    capture_fingerprint(rows) == capture_fingerprint(other) and
                    all(all(v is not False for v in row["checks"].values()) for row in other))
                result["partition_check"] = dict(passed=check_passed,
                    fixed_block_count=info["blocks"], random_block_count=other_info["blocks"],
                    capture_count_fixed=len(rows), capture_count_random=len(other),
                    random_run_fault=other_info["fault"],
                    compared="All capture coordinates, landmarks, termination, exact raw IQ hashes; features excluded")
                integrity_ok &= check_passed
            report["runs"][label] = result
            write_json(out / "pipeline_report.json", report)  # Keep completed cases if a later case fails.
            print(f"  captures={len(rows)}, target-time-covered={scored['target_bursts_with_at_least_50pct_time_coverage']}, "
                  f"fresh-target-confirmations={scored['target_bursts_with_new_confirmation']}, fault={info['fault']}", flush=True)
    report["functional_integrity_checks_pass"] = bool(integrity_ok)
    report["capture_count_verified_total"] = sum(len(r["events"]) for r in report["runs"].values())
    report["detection_qualification_or_freeze_pass"] = None
    write_json(out / "pipeline_report.json", report)
    return report, bool(integrity_ok)


def plot_fixture(parts: dict, meta: dict, report: dict | None, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = meta["sample_rate_hz"]
    iq = parts["iq"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    width = max(8, round(0.0005 * fs))
    smoothed = signal.fftconvolve(np.abs(iq.astype(np.complex128)) ** 2,
                                 np.ones(width) / width, mode="same")
    decimation = max(1, len(iq) // 5000)
    axes[0].plot(np.arange(0, len(iq), decimation) / fs * 1000,
                 10 * np.log10(np.maximum(smoothed[::decimation], 1e-30)), lw=0.8)
    axes[0].set_ylabel("Mean IQ power (dBFS)")
    axes[0].set_title(f"Overlapping RF fixture | target {meta['target_fullband_metrics'][0]['realized_fullband_snr_db']:.1f} dB full-band SNR | {meta['noise_model']} noise")
    nfft = 1024
    f, t, psd = signal.spectrogram(iq, fs=fs, nperseg=nfft, noverlap=3 * nfft // 4,
        detrend=False, return_onesided=False, scaling="density", mode="psd")
    order = np.argsort(f)
    colors = axes[1].pcolormesh(t * 1000, f[order] / 1000,
        10 * np.log10(np.maximum(psd[order], 1e-30)), shading="auto", cmap="magma")
    lo, hi = meta["target_monitored_band_offsets_hz"]
    axes[1].axhline(lo / 1000, color="cyan", ls="--", lw=0.8)
    axes[1].axhline(hi / 1000, color="cyan", ls="--", lw=0.8)
    axes[1].set_ylabel("Baseband offset (kHz)")
    fig.colorbar(colors, ax=axes[1], label="PSD (dBFS/Hz)")
    labels = list(SIGNAL_KEYS)
    for truth in meta["truths"]:
        row = labels.index(truth["component"])
        axes[2].broken_barh([(truth["start_s"] * 1000, (truth["end_s"] - truth["start_s"]) * 1000)],
                           (row - 0.3, 0.6), facecolors="tab:red" if truth["role"] == "target" else "tab:blue")
    if report:
        for key, result in report["runs"].items():
            if not key.endswith("/mixture"):
                continue
            row = len(labels)
            labels.append(key + " detection")
            for event in result["events"]:
                a, b = event["detected_start_sample"], event["detected_end_sample"]
                axes[2].broken_barh([(a / fs * 1000, (b - a) / fs * 1000)],
                                   (row - 0.3, 0.6), facecolors="tab:green")
    axes[2].set_yticks(range(len(labels)), labels)
    axes[2].set_ylim(-0.6, len(labels) - 0.4)
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_title("Ground-truth emitter schedules and detected intervals; overlap is not identification")
    for ax in axes:
        ax.set_xlim(0, meta["duration_s"] * 1000)
    fig.savefig(out / "overlap_fixture.png", dpi=160)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=Path("overlap_results"))
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--sample-rate", type=float, default=1_000_000.0)
    p.add_argument("--center-frequency", type=float, default=915_000_000.0)
    p.add_argument("--noise-dbfs", type=float, default=-40.0)
    p.add_argument("--target-snr-db", type=float, default=0.0)
    p.add_argument("--noise-model", choices=("white", "ar1"), default="white")
    p.add_argument("--ar1-rho", type=float, default=0.8)
    p.add_argument("--package-src", help="Folder containing signal_processing/, e.g. --package-src .")
    p.add_argument("--run-pipeline", action="store_true")
    p.add_argument("--mode", choices=("both", "full", "band"), default="both")
    p.add_argument("--window-size", type=int, default=512)
    p.add_argument("--hop-size", type=int, default=128)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--pre-trigger", type=int, default=2048)
    p.add_argument("--pre-margin", type=int, default=1024)
    p.add_argument("--post-margin", type=int, default=1024)
    p.add_argument("--pfa-on", type=float, default=1e-4)
    p.add_argument("--pfa-off", type=float, default=1e-2)
    p.add_argument("--no-features", action="store_true", help="Capture-only run")
    p.add_argument("--check-partitions", action="store_true", help="Compare fixed vs random transport chunk sizes")
    p.add_argument("--enforce-integrity", action="store_true",
                   help="Exit 1 on faults, corrupt/incomplete capture, failed features or changed partition results; NOT a Pd/Pfa gate")
    p.add_argument("--plot", action="store_true", help="Save waveform, spectrogram and truth/detection timeline (needs matplotlib)")
    args = p.parse_args(argv)
    if args.seed < 0:
        p.error("--seed must be nonnegative")
    if not math.isfinite(args.sample_rate) or not 100_000 <= args.sample_rate <= 10_000_000:
        p.error("--sample-rate must be finite, between 100000 and 10000000 Hz")
    if not math.isfinite(args.center_frequency) or args.center_frequency < 0:
        p.error("--center-frequency must be finite and nonnegative")
    if not math.isfinite(args.noise_dbfs) or not -120 <= args.noise_dbfs <= -20:
        p.error("--noise-dbfs must be between -120 and -20 (no clipping is simulated)")
    if not math.isfinite(args.target_snr_db) or not -40 <= args.target_snr_db <= 30:
        p.error("--target-snr-db must be between -40 and 30")
    if not math.isfinite(args.ar1_rho) or not 0 <= args.ar1_rho < 0.99:
        p.error("--ar1-rho must be finite and in [0, 0.99)")
    if not 1 <= args.hop_size <= args.window_size or args.block_size < 1:
        p.error("Require 1 <= hop-size <= window-size and block-size >= 1")
    if min(args.pre_trigger, args.pre_margin, args.post_margin) < 0:
        p.error("Capture margins must be nonnegative")
    if not 0 < args.pfa_on < args.pfa_off < 1:
        p.error("Require 0 < pfa-on < pfa-off < 1")
    if (args.check_partitions or args.enforce_integrity) and not args.run_pipeline:
        p.error("--check-partitions and --enforce-integrity require --run-pipeline")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Do not silently mix a new test with stale reports/capture files.
    if any((out / name).exists() for name in ("mixture.npz", "mixture.cf32", "pipeline_report.json",
                                             "ground_truth.json", "captures", "overlap_fixture.png")):
        raise ValueError(f"Output already contains a fixture: {out}. Choose a new --output directory.")
    parts, metadata = make_fixture(args)
    np.savez_compressed(out / "mixture.npz", **parts,
                        sample_rate_hz=np.array(args.sample_rate),
                        center_frequency_hz=np.array(args.center_frequency))
    parts["iq"].astype("<c8", copy=False).tofile(out / "mixture.cf32")
    write_json(out / "ground_truth.json", metadata)
    print(f"Generated {metadata['num_samples']} complex IQ samples in {out}", flush=True)
    print("Target: four pulse-shaped QPSK bursts. Interferers: CW, BPSK, adjacent QPSK, chirp, co-channel QPSK.", flush=True)
    report = None
    integrity = True
    if args.run_pipeline:
        report, integrity = run_pipeline(parts, metadata, args, out)
        print(f"Functional capture/feature integrity checks: {integrity}; detection qualification: not claimed.", flush=True)
    if args.plot:
        plot_fixture(parts, metadata, report, out)
    print(f"Results: {out}", flush=True)
    return int(args.enforce_integrity and not integrity)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
