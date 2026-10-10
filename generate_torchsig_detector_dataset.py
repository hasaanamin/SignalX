#!/usr/bin/env python3
"""Generate labeled complex-IQ detector tests using TorchSig 2.2.0 builders.

Install in a separate Python 3.12 environment:
  python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
  python -m pip install torchsig==2.2.0 numpy==2.3.5 scipy==1.17.0 numba==0.62.1 opencv-python==4.11.0.86 lightning==2.5.3

Small test (8 recordings):
  python generate_torchsig_detector_dataset.py generate --output detector_smoke --repeats 1 --snrs 0 --previews 3
Default dataset (240 recordings, about 240 MiB of IQ):
  python generate_torchsig_detector_dataset.py generate --output detector_dataset
White + colored noise:
  python generate_torchsig_detector_dataset.py generate --output detector_stress --noise-models white ar1

Evaluate your current FYP package, from its unpacked project root:
  python generate_torchsig_detector_dataset.py evaluate --dataset detector_dataset --package-src . --output detector_evaluation --check-partitions --enforce-integrity

Waveforms come from TorchSig's public builders. The wrapper controls recording
intervals, frequency shifts, active full-band SNR and noise, so every required
overlap case is reproducible. IQ is exported as one-dimensional complex64 .npy
arrays; the evaluate subcommand streams them through your FYP package.

Every emitter has start/stop samples AND seconds, frequency offsets, modulation,
nominal generator bandwidth, power, SNR, seed and preserved generator metadata.
Aggregate activity intervals and sample masks are also exported. Interval ends
are exclusive. No post-generation peak normalization or ADC clipping is applied.

The built-in evaluator tests FULL-BAND ACTIVITY detection, not emitter identity.
It matches events one-to-one against the union of all emitter intervals. Emitter
coverage is reported separately and does not prove independent detection of an
emitter hidden inside a long activity event. Colored noise and co-channel overlap
are challenge cases, not a promise of certified Pfa or identifiable sources.

Primary API reference: https://github.com/TorchDSP/torchsig/tree/v2.2.0
Release: https://pypi.org/project/torchsig/2.2.0/
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import importlib
from importlib import metadata as package_metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy import signal
from scipy.optimize import linear_sum_assignment


VERSION = "1.0.0"
TORCHSIG_VERSION = "2.2.0"
SCENARIOS = ("noise_only", "single", "sequential", "overlap_separated",
             "simultaneous_separated", "nested_near_far", "cochannel_overlap", "short_bursts")
CLASSES = ("bpsk", "qpsk", "8psk", "16qam", "64qam", "2fsk", "4fsk",
           "2gfsk", "ofdm-64", "ofdm-128", "tone")
DEFAULT_CLASSES = ("bpsk", "qpsk", "16qam", "2fsk", "ofdm-64", "tone")
CLOCK_NS = 1_700_000_000_000_000_000


def jsonable(value):
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite annotation; generation must not silently save invalid labels")
    return value


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(jsonable(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def power(x: np.ndarray) -> float:
    z = np.asarray(x, dtype=np.complex128)
    return float(np.vdot(z, z).real / len(z)) if len(z) else 0.0


def db_ratio(a: float, b: float) -> float | None:
    return 10.0 * math.log10(a / b) if a > 0 and b > 0 else None


def iq_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x, dtype="<c8").tobytes()).hexdigest()


def stable_seed(*values) -> int:
    text = json.dumps(values, separators=(",", ":"), allow_nan=False)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def union_intervals(spans) -> list[tuple[int, int]]:
    result = []
    for a, b in sorted((int(a), int(b)) for a, b in spans if b > a):
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def covered_samples(a: int, b: int, spans) -> int:
    return sum(y - x for x, y in union_intervals((max(a, x), min(b, y)) for x, y in spans))


def timestamp_ns(index: int, fs: float) -> int:
    return CLOCK_NS + round(Fraction(index * 1_000_000_000, 1) / Fraction(float(fs)))


def import_torchsig_builders() -> dict:
    import torchsig
    if torchsig.__version__ != TORCHSIG_VERSION:
        raise RuntimeError(f"This script targets TorchSig {TORCHSIG_VERSION}; installed {torchsig.__version__}. "
                           "Use the pinned installation commands in the file header.")
    from torchsig.signals.builders.constellation import ConstellationSignalGenerator
    from torchsig.signals.builders.fsk import FSKSignalGenerator
    from torchsig.signals.builders.ofdm import OFDMSignalGenerator
    from torchsig.signals.builders.tone import ToneSignalGenerator
    return dict(constellation=ConstellationSignalGenerator, fsk=FSKSignalGenerator,
                ofdm=OFDMSignalGenerator, tone=ToneSignalGenerator)


def build_waveform(kind: str, count: int, bandwidth_hz: int, fs: float,
                   seed: int, builders: dict) -> tuple[np.ndarray, dict]:
    # Generate a longer native waveform and crop its interior, especially for
    # 64/256-sample transients. Do not mistake a builder's padded filter edges
    # for a meaningful very short burst.
    generated_count = max(count, 4096, math.ceil(128 * fs / bandwidth_hz))
    common = dict(seed=seed, sample_rate=fs, bandwidth_min=bandwidth_hz,
                  bandwidth_max=bandwidth_hz,
                  signal_duration_in_samples_min=generated_count,
                  signal_duration_in_samples_max=generated_count)
    if kind == "tone":
        builder = builders["tone"](**common)
    elif kind.startswith("ofdm-"):
        builder = builders["ofdm"](num_subcarriers=int(kind.split("-")[1]), **common)
    elif kind.endswith("gfsk"):
        builder = builders["fsk"](fsk_type="gfsk", constellation_size=int(kind[:-4]), **common)
    elif kind.endswith("fsk"):
        builder = builders["fsk"](fsk_type="fsk", constellation_size=int(kind[:-3]), **common)
    else:
        builder = builders["constellation"](constellation_name=kind, **common)
    builder.validate_metadata_fields()
    native = builder()
    data = np.asarray(native.data)
    if data.ndim != 1 or len(data) != generated_count or not np.iscomplexobj(data) or not np.isfinite(data).all():
        raise RuntimeError(f"TorchSig {kind} returned invalid waveform shape/data")
    crop = (generated_count - count) // 2
    z = data[crop:crop + count].astype(np.complex128)
    if power(z) <= 0:
        raise RuntimeError(f"TorchSig {kind} produced an empty-power crop")
    provenance = dict(builder=type(builder).__name__, torchsig_version=TORCHSIG_VERSION,
        generated_samples=generated_count, interior_crop_start_sample=crop,
        metadata=jsonable(native.get_full_metadata()))
    return z, provenance


def scenario_plan(name: str, n: int, snr_db: float, rng: np.random.Generator) -> list[dict]:
    # time fractions, baseband carrier/Fs, per-emitter full-band SNR.
    if name == "noise_only":
        specs = []
    elif name == "single":
        specs = [(0.30, 0.52, float(rng.choice([-0.22, 0.0, 0.22])), snr_db)]
    elif name == "sequential":
        specs = [(0.23, 0.35, -0.22, snr_db), (0.49, 0.61, 0.0, snr_db), (0.74, 0.86, 0.22, snr_db)]
    elif name == "overlap_separated":
        specs = [(0.25, 0.55, -0.22, snr_db), (0.45, 0.70, 0.22, snr_db), (0.60, 0.84, 0.0, snr_db)]
    elif name == "simultaneous_separated":
        specs = [(0.30, 0.75, f, snr_db) for f in (-0.22, 0.0, 0.22)]
    elif name == "nested_near_far":
        # Strong A stays ON while weaker B and C appear inside it.
        specs = [(0.24, 0.85, -0.22, 18.0), (0.40, 0.53, 0.22, snr_db), (0.63, 0.74, 0.0, snr_db)]
    elif name == "cochannel_overlap":
        specs = [(0.25, 0.78, 0.08, snr_db + 6.0), (0.45, 0.60, 0.08, snr_db)]
    elif name == "short_bursts":
        return [dict(start_sample=round(f * n), end_sample=round(f * n) + duration,
                     center_fraction=offset, requested_snr_db=snr_db)
                for f, duration, offset in zip((0.24, 0.40, 0.57, 0.74),
                    (64, 256, 2048, 4096), (-0.22, 0.0, 0.22, -0.22))]
    else:
        raise ValueError(name)
    return [dict(start_sample=round(a * n), end_sample=round(b * n),
                 center_fraction=f, requested_snr_db=s) for a, b, f, s in specs]


def make_noise(n: int, expected_power: float, model: str, rho: float,
               seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if model == "white":
        x = np.sqrt(expected_power / 2.0) * (
            rng.standard_normal(n) + 1j * rng.standard_normal(n))
    else:
        burn = max(4096, math.ceil(math.log(1e-12) / math.log(rho))) if rho else 4096
        innovations = np.sqrt(expected_power * (1.0 - rho ** 2) / 2.0) * (
            rng.standard_normal(n + burn) + 1j * rng.standard_normal(n + burn))
        x = signal.lfilter([1.0], [1.0, -rho], innovations)[burn:]
    return x.astype(np.complex64)


def make_record(args, record_id: str, scenario: str, noise_model: str,
                snr_db: float, repetition: int, index: int, builders: dict) -> tuple[np.ndarray, dict, dict, dict]:
    n, fs = args.samples, args.sample_rate
    seed = stable_seed(args.seed, scenario, noise_model, float(snr_db), repetition)
    rng = np.random.default_rng(seed)
    noise_power = 10.0 ** (args.noise_dbfs / 10.0)
    noise = make_noise(n, noise_power, noise_model, args.ar1_rho, stable_seed(seed, "noise"))
    combined = noise.astype(np.complex128)
    clean_sum = np.zeros(n, dtype=np.complex128)
    components = {"noise": noise}
    annotations = []
    plan = scenario_plan(scenario, n, snr_db, rng)
    for i, placement in enumerate(plan):
        a, b = placement["start_sample"], placement["end_sample"]
        kind = args.classes[index % len(args.classes)] if i == 0 else str(rng.choice(args.classes))
        width = max(1, round(float(rng.choice(args.bandwidth_fractions)) * fs))
        waveform_seed = stable_seed(seed, "emitter", i)
        z, provenance = build_waveform(kind, b - a, width, fs, waveform_seed, builders)
        # Strictly positive edge taper: keep scheduled edge samples meaningful.
        ramp = min(64, len(z) // 16)
        if ramp > 0:
            edge = np.hanning(2 * ramp + 2)[1:ramp + 1]
            z[:ramp] *= edge
            z[-ramp:] *= edge[::-1]
        f = placement["center_fraction"] * fs
        z *= np.exp(1j * (rng.uniform(-np.pi, np.pi) + 2 * np.pi * f * (a + np.arange(len(z))) / fs))
        support = np.flatnonzero(z != 0)
        if not len(support):
            raise RuntimeError("Empty emitter after taper/frequency placement")
        # Some native builders retain zero padding even in long outputs. Scale
        # over the SAME envelope that the exported start/stop labels describe.
        active_power = power(z[int(support[0]):int(support[-1]) + 1])
        z *= np.sqrt(noise_power * 10 ** (placement["requested_snr_db"] / 10.0) / active_power)
        component = z.astype(np.complex64)
        nonzero = np.flatnonzero(component != 0)
        if len(nonzero) == 0:
            raise RuntimeError("Empty emitter after receiver dtype conversion")
        start, stop = a + int(nonzero[0]), a + int(nonzero[-1]) + 1
        if not 0 <= a < b <= n or not np.isfinite(component).all():
            raise RuntimeError("Invalid placed emitter")
        combined[a:b] += component
        clean_sum[a:b] += component
        components[f"emitter_{i:03d}"] = component
        native_width = float(provenance["metadata"]["bandwidth"])
        active_component = component[start - a:stop - a]
        annotations.append(dict(emitter_id=f"{record_id}:emitter_{i:03d}", modulation=kind,
            start_sample=start, end_sample=stop, start_s=start / fs, end_s=stop / fs,
            start_ms=start / fs * 1000, end_ms=stop / fs * 1000,
            duration_samples=stop - start, duration_s=(stop - start) / fs,
            placement_start_sample=a, placement_end_sample=b, component_array=f"emitter_{i:03d}",
            center_frequency_offset_hz=f, absolute_center_frequency_hz=args.center_frequency + f,
            nominal_bandwidth_hz=native_width,
            nominal_lower_frequency_offset_hz=f - native_width / 2,
            nominal_upper_frequency_offset_hz=f + native_width / 2,
            bandwidth_semantics="TorchSig generator nominal bandwidth; finite gating adds spectral tails; not certified occupied bandwidth",
            requested_snr_fullband_db=placement["requested_snr_db"],
            measured_snr_vs_expected_noise_db=db_ratio(power(active_component), noise_power),
            measured_snr_vs_realized_noise_db=db_ratio(power(active_component), power(noise[start:stop])),
            active_power_linear=power(active_component), edge_ramp_samples=ramp,
            waveform_seed=waveform_seed, torchsig=provenance))
    iq = combined.astype(np.complex64)
    prefix, settling = n // 8, n // 64
    counts = np.zeros(n, dtype=np.uint8)
    for emitter in annotations:
        counts[emitter["start_sample"]:emitter["end_sample"]] += 1
        a, b = emitter["start_sample"], emitter["end_sample"]
        placed_a = emitter["placement_start_sample"]
        comp = components[emitter["component_array"]][a - placed_a:b - placed_a].astype(np.complex128)
        nuisance = noise[a:b].astype(np.complex128) + clean_sum[a:b] - comp
        emitter["realized_fullband_sinr_db"] = db_ratio(power(comp), power(nuisance))
    intervals = union_intervals((e["start_sample"], e["end_sample"]) for e in annotations)
    if not np.array_equal(iq[:prefix], noise[:prefix]) or not np.isfinite(iq).all():
        raise RuntimeError("Noise-only calibration prefix contaminated, or invalid IQ")
    if any(e["start_sample"] < prefix for e in annotations):
        raise RuntimeError("An emitter overlaps the declared noise calibration prefix")
    meta = dict(record_id=record_id, scenario=scenario, record_seed=seed, repetition=repetition,
        sweep_snr_db=snr_db, sample_rate_hz=fs, receiver_center_frequency_hz=args.center_frequency,
        num_samples=n, duration_s=n / fs, receiver_dtype="complex64",
        iq_file="iq.npy", iq_sha256=iq_hash(iq), signal_count=len(annotations), emitters=annotations,
        interval_convention="[start_sample, end_sample), end exclusive; times relative to this recording",
        background=dict(kind="noise", start_sample=0, end_sample=n, start_s=0.0, end_s=n / fs,
            model=noise_model, ar1_rho=args.ar1_rho if noise_model == "ar1" else None,
            expected_fullband_noise_power_linear=noise_power, realized_fullband_noise_power_linear=power(noise)),
        calibration_prefix=dict(start_sample=0, end_sample=prefix, settling_samples=settling,
                                training_samples=prefix - settling, noise_only=True),
        fullband_activity_intervals=[dict(start_sample=a, end_sample=b, start_s=a / fs, end_s=b / fs,
            contributing_emitter_ids=[e["emitter_id"] for e in annotations if e["start_sample"] < b and e["end_sample"] > a])
            for a, b in intervals],
        expected_fullband_activity_event_count=len(intervals),
        cochannel_identity_resolvable_by_energy_detector=False if scenario == "cochannel_overlap" else None,
        clipping_or_composite_normalization_applied=False,
        snr_semantics="Per-emitter active power / expected FULL-BAND noise power; not detector-band SNR or SINR",
        emitter_interval_semantics="Envelope from first to last nonzero placed component sample; internal waveform zeros do not split an emitter")
    return iq, meta, dict(any_signal=counts > 0, active_emitter_count=counts), components


def plot_record(iq: np.ndarray, meta: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = meta["sample_rate_hz"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    f, t, p = signal.spectrogram(iq, fs=fs, nperseg=1024, noverlap=768,
        detrend=False, return_onesided=False, scaling="density", mode="psd")
    order = np.argsort(f)
    image = axes[0].pcolormesh(t * 1000, f[order] / 1000,
        10 * np.log10(np.maximum(p[order], 1e-30)), shading="auto", cmap="magma")
    axes[0].set_ylabel("Offset (kHz)")
    axes[0].set_title(f"{meta['record_id']} | {meta['scenario']} | sweep SNR {meta['sweep_snr_db']} dB")
    fig.colorbar(image, ax=axes[0], label="PSD (dBFS/Hz)")
    labels = []
    for i, emitter in enumerate(meta["emitters"]):
        labels.append(f"{i}: {emitter['modulation']} @ {emitter['center_frequency_offset_hz']/1000:.0f} kHz")
        axes[1].broken_barh([(emitter["start_ms"], emitter["end_ms"] - emitter["start_ms"])],
                           (i - 0.3, 0.6), facecolors=f"C{i % 10}")
    if not labels:
        labels = ["noise only"]
    axes[1].set_yticks(range(len(labels)), labels)
    axes[1].set_ylim(-0.6, len(labels) - 0.4)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_title("All emitter start/stop labels; simultaneous emitters stay separate in ground truth")
    for ax in axes:
        ax.set_xlim(0, meta["duration_s"] * 1000)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def environment() -> dict:
    versions = {}
    for name in ("torchsig", "torch", "torchaudio", "numpy", "scipy", "numba", "lightning", "opencv-python"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.version, platform=platform.platform(), packages=versions)


def empty_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output is not empty: {path}. Choose a new output directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate(args) -> int:
    builders = import_torchsig_builders()
    out = empty_output(args.output)
    total = len(args.scenarios) * len(args.snrs) * len(args.noise_models) * args.repeats
    dataset = dict(script_version=VERSION, purpose="detector evaluation", status="GENERATING",
        torchsig_version=TORCHSIG_VERSION, root_seed=args.seed, sample_rate_hz=args.sample_rate,
        receiver_center_frequency_hz=args.center_frequency, num_samples_per_record=args.samples,
        scenarios=args.scenarios, modulation_classes=args.classes, noise_models=args.noise_models,
        snr_sweep_fullband_db=args.snrs, repetitions_per_cell=args.repeats,
        bandwidth_fractions=args.bandwidth_fractions, noise_dbfs=args.noise_dbfs,
        ar1_rho=args.ar1_rho, records_expected=total, records_completed=0,
        environment=environment(), generation="TorchSig 2.2.0 public waveform builders + controlled placement/noise wrapper",
        limitations=["All emitters are labeled; energy detection cannot establish their identities.",
            "Evaluate full-band activity against the union of all emitter intervals.",
            "Temporal coverage of nested/co-channel emitters does not prove separate detections.",
            "Per-emitter nominal bandwidth is not a certified spectral-support or occupied-bandwidth boundary.",
            "SNR is full-band input SNR; filtered detection-band SNR differs.",
            "AR1 noise challenges Gaussian threshold assumptions; calibration is not Pfa certification.",
            "Synthetic finite data is not receiver validation or a real-time performance guarantee.",
            "Each recording starts a fresh stream at sample zero; reset the detector between files."])
    write_json(out / "dataset.json", dataset)
    signal_fields = ("record_id", "emitter_id", "modulation", "start_sample", "end_sample", "start_s", "end_s",
                     "start_ms", "end_ms", "duration_samples", "center_frequency_offset_hz",
                     "absolute_center_frequency_hz", "nominal_bandwidth_hz", "requested_snr_fullband_db",
                     "measured_snr_vs_expected_noise_db", "realized_fullband_sinr_db")
    started = time.perf_counter()
    print(f"Generating {total} recordings; IQ payload about {total * args.samples * 8 / 2**20:.1f} MiB, plus labels/previews.", flush=True)
    try:
        with (out / "manifest.jsonl").open("w", encoding="utf-8") as manifest, \
                (out / "signals.csv").open("w", newline="", encoding="utf-8") as signal_file:
            csv_writer = csv.DictWriter(signal_file, fieldnames=signal_fields)
            csv_writer.writeheader()
            index = 0
            for noise_model in args.noise_models:
                for snr in args.snrs:
                    for scenario in args.scenarios:
                        for repetition in range(args.repeats):
                            record_id = f"rec_{index:06d}"
                            iq, meta, masks, components = make_record(args, record_id, scenario,
                                noise_model, snr, repetition, index, builders)
                            folder = out / "records" / record_id
                            folder.mkdir(parents=True)
                            np.save(folder / "iq.npy", iq, allow_pickle=False)
                            np.savez_compressed(folder / "truth_masks.npz", **masks)
                            write_json(folder / "annotations.json", meta)
                            if args.save_components:
                                np.savez_compressed(folder / "components.npz", **components)
                            if index < args.previews:
                                plot_record(iq, meta, folder / "preview.png")
                            entry = dict(record_id=record_id, iq_file=f"records/{record_id}/iq.npy",
                                annotations_file=f"records/{record_id}/annotations.json",
                                masks_file=f"records/{record_id}/truth_masks.npz", scenario=scenario,
                                noise_model=noise_model, snr_sweep_db=snr, repetition=repetition,
                                signal_count=meta["signal_count"], iq_sha256=meta["iq_sha256"])
                            manifest.write(json.dumps(entry, allow_nan=False) + "\n")
                            manifest.flush()
                            for emitter in meta["emitters"]:
                                csv_writer.writerow({field: (record_id if field == "record_id" else emitter[field]) for field in signal_fields})
                            signal_file.flush()
                            index += 1
                            dataset["records_completed"] = index
                            if index % 10 == 0 or index == total:
                                write_json(out / "dataset.json", dataset)
                                print(f"  {index}/{total} recordings", flush=True)
    except Exception as exc:
        dataset["status"] = "FAILED"
        dataset["failure"] = dict(type=type(exc).__name__, message=str(exc))
        write_json(out / "dataset.json", dataset)
        raise
    dataset["status"] = "COMPLETE"
    dataset["elapsed_generation_s"] = time.perf_counter() - started
    write_json(out / "dataset.json", dataset)
    (out / "README.md").write_text(
        "# Detector test dataset\n\n"
        "Read `dataset.json` and `manifest.jsonl`. Each record has complex64 `iq.npy`, "
        "`annotations.json` labeling EVERY emitter, and `truth_masks.npz` containing "
        "`any_signal` and `active_emitter_count`. `signals.csv` is the combined emitter table.\n\n"
        "All sample/time ends are exclusive. `background` gives the noise interval. "
        "The calibration prefix is genuinely noise-only; the evaluator trains on it through "
        "the pipeline's causal detection path. Reset all streaming state between records.\n\n"
        "Load IQ with `np.load(record_path / 'iq.npy', allow_pickle=False)`. "
        "Feed contiguous chunks with start_sample equal to their array index. "
        "Keep all emitters in their own ground-truth entries, even during overlap.\n\n"
        "For a full-band activity detector use `fullband_activity_intervals`: overlapping "
        "emitters produce one aggregate activity region. The built-in evaluator reports "
        "one-to-one activity event matches, sample precision/recall, boundary errors, "
        "noise-only false events, and temporal coverage of every emitter. Coverage does "
        "not establish source identity. For frequency-selective testing, nominal bandwidth "
        "is only a generator parameter; account for filter leakage and spectral tails.\n\n"
        "SNR means active emitter power / expected full-band noise power. The script "
        "also reports realized SNR and full-band SINR. It uses TorchSig's actual builders, "
        "with controlled scheduling, frequency shifting and noise addition. No post-mix "
        "normalization, clipping, channel-fading or time-warp impairment is silently added.\n\n"
        "All timing is offline. Data and event-rate observations do not certify rare-event "
        "Pfa, real-noise detection, real-time acquisition or multi-emitter separation.\n",
        encoding="utf-8")
    print(f"Dataset ready: {out}", flush=True)
    return 0


def run_record(iq, meta, args, api, random_chunks=False):
    stream = api["StreamInfo"](meta["record_id"], "torchsig-controlled", meta["sample_rate_hz"],
                               center_frequency_hz=meta["receiver_center_frequency_hz"])
    calibration = meta["calibration_prefix"]
    estimator = api["GaussianNoiseEstimator"](window_size=args.window_size,
        warmup_samples=calibration["training_samples"], warmup_settling_samples=calibration["settling_samples"],
        pfa_on=args.pfa_on, pfa_off=args.pfa_off)
    pipeline = api["StreamingCapturePipeline"](stream_info=stream,
        ring_capacity=max(16384, 4 * (args.window_size + args.pre_trigger + args.pre_margin)),
        window_size=args.window_size, hop_size=args.hop_size, pfa_on=args.pfa_on, pfa_off=args.pfa_off,
        noise_estimator=estimator, detector_mode="full_band", start_persistence=2, end_persistence=3,
        pre_trigger_samples=args.pre_trigger, pre_signal_margin_samples=args.pre_margin,
        post_signal_samples=1024, min_post_trigger_samples=args.pre_trigger,
        max_event_samples=len(iq) + args.pre_trigger + args.window_size,
        enable_feature_analysis=args.features, synchronous_analysis=True, require_supported_pfa=False)
    rng = np.random.default_rng(stable_seed(meta["record_seed"], "chunks"))
    captures = []
    fault = None
    start = sequence = 0
    try:
        while start < len(iq):
            length = int(rng.integers(1, 3 * args.block_size + 1)) if random_chunks else args.block_size
            stop = min(start + length, len(iq))
            block = api["IQBlock"](stream, sequence, start,
                timestamp_ns(start, meta["sample_rate_hz"]), iq[start:stop])
            captures.extend(pipeline.process_block(block))
            start, sequence = stop, sequence + 1
        captures.extend(pipeline.flush())
    except Exception as exc:
        captures.extend(getattr(exc, "completed_captures", ()))
        fault = dict(type=type(exc).__name__, message=str(exc))
    finally:
        try:
            pipeline.shutdown(timeout=10.0)
        except Exception as exc:
            fault = dict(previous=fault, shutdown_type=type(exc).__name__, shutdown_message=str(exc))
    rows = []
    for sc in captures:
        e = sc.event
        a, b = e.event_start_sample, e.event_end_sample
        bounds = 0 <= a < b <= len(iq)
        full_quality = e.complete and not sc.is_truncated
        expected_eof_limit = (sc.termination_reason == "eof" and b == len(iq)
                              and not e.complete and sc.is_truncated)
        checks = dict(raw_iq_exact=bool(bounds and np.array_equal(e.iq, iq[a:b])),
            iq_read_only=not e.iq.flags.writeable,
            quality_flags_consistent=bool(full_quality or expected_eof_limit),
            no_loss=e.dropped_samples == 0 and sc.known_lost_samples == 0 and not sc.unknown_loss,
            timestamp_exact=e.timestamp_ns == timestamp_ns(a, meta["sample_rate_hz"]),
            leading_retention_exact=(a == max(0, min(e.trigger_sample - args.pre_trigger,
                                                    e.detected_start_sample - args.pre_margin))),
            pre_trigger_retained=(e.pre_trigger_samples >= args.pre_trigger if e.trigger_sample >= args.pre_trigger else None),
            feature_delivery_ok=((sc.feature_status == "ok" and sc.features is not None)
                if args.features and full_quality else
                (sc.feature_status == "skipped_quality" and sc.features is None)
                if args.features and expected_eof_limit else
                (sc.feature_status == "skipped" and sc.features is None) if not args.features else False))
        rows.append(dict(event_id=e.event_id, event_start_sample=a, event_end_sample=b,
            detected_start_sample=e.detected_start_sample, detected_end_sample=e.detected_end_sample,
            trigger_sample=e.trigger_sample, confirmation_sample=sc.confirmation_sample,
            termination_reason=sc.termination_reason, iq_sha256=iq_hash(e.iq),
            complete=bool(e.complete), truncated=bool(sc.is_truncated),
            eof_limited_capture=bool(expected_eof_limit),
            feature_status=sc.feature_status, checks=checks))
    integrity = (fault is None and pipeline.received_end_sample == len(iq) and
                 pipeline.consumed_end_sample == len(iq) and
                 len({(sc.stream_epoch, sc.event_id) for sc in captures}) == len(captures) and
                 all(all(value is not False for value in row["checks"].values()) for row in rows))
    return rows, dict(fault=fault, integrity_pass=bool(integrity), capture_checks_exercised=bool(rows),
        all_captures_complete=all(row["complete"] and not row["truncated"] for row in rows) if rows else None,
        eof_limited_capture_count=sum(row["eof_limited_capture"] for row in rows),
        received_end_sample=pipeline.received_end_sample, consumed_end_sample=pipeline.consumed_end_sample,
        noise_reference_linear=estimator.noise_power_linear, threshold_model=pipeline.detector.threshold_model,
        pfa_support="model_only; no held-out certification", block_count=sequence)


def score_activity(meta: dict, rows: list[dict], iou_threshold: float) -> dict:
    n = meta["num_samples"]
    admission = meta["calibration_prefix"]["end_sample"]
    truth = [(v["start_sample"], v["end_sample"]) for v in meta["fullband_activity_intervals"]]
    predicted = [(v["detected_start_sample"], v["detected_end_sample"]) for v in rows]
    matrix = np.zeros((len(truth), len(predicted)))
    for i, (a, b) in enumerate(truth):
        for j, (c, d) in enumerate(predicted):
            intersection = max(0, min(b, d) - max(a, c))
            matrix[i, j] = intersection / ((b - a) + (d - c) - intersection)
    matches = []
    if matrix.size:
        # Maximize match COUNT first, then IoU. One detection cannot be credited
        # as separate detections of several ground-truth aggregate events.
        score = (matrix >= iou_threshold) * (min(matrix.shape) + 1) + matrix
        truth_idx, predicted_idx = linear_sum_assignment(-score)
        for i, j in zip(truth_idx, predicted_idx):
            if matrix[i, j] >= iou_threshold:
                matches.append(dict(truth_activity_index=int(i), detected_event_id=rows[j]["event_id"],
                    iou=float(matrix[i, j]), start_error_samples=predicted[j][0] - truth[i][0],
                    end_error_samples=predicted[j][1] - truth[i][1],
                    start_error_s=(predicted[j][0] - truth[i][0]) / meta["sample_rate_hz"],
                    end_error_s=(predicted[j][1] - truth[i][1]) / meta["sample_rate_hz"]))
    matched_truth = {m["truth_activity_index"] for m in matches}
    matched_events = {m["detected_event_id"] for m in matches}
    truth_count = sum(b - a for a, b in truth)
    pred_union = union_intervals((max(admission, a), min(n, b)) for a, b in predicted)
    pred_count = sum(b - a for a, b in pred_union)
    tp_samples = sum(covered_samples(a, b, pred_union) for a, b in truth)
    fp_samples, fn_samples = pred_count - tp_samples, truth_count - tp_samples
    noise_samples = n - admission - truth_count
    emitters = []
    for emitter in meta["emitters"]:
        a, b = emitter["start_sample"], emitter["end_sample"]
        ids = [row["event_id"] for row, (c, d) in zip(rows, predicted) if c < b and d > a]
        emitters.append(dict(emitter_id=emitter["emitter_id"], modulation=emitter["modulation"],
            start_sample=a, end_sample=b, start_s=emitter["start_s"], end_s=emitter["end_s"],
            temporal_coverage=covered_samples(a, b, pred_union) / (b - a),
            overlapping_activity_event_ids=ids,
            new_confirmation_inside_interval=any(a <= row["confirmation_sample"] < b for row in rows),
            individually_identified=False))
    event_annotations = []
    for row, (a, b) in zip(rows, predicted):
        ids = [e["emitter_id"] for e in meta["emitters"] if e["start_sample"] < b and e["end_sample"] > a]
        region_count = sum(c < b and d > a for c, d in truth)
        event_annotations.append(dict(event_id=row["event_id"], overlapping_emitter_ids=ids,
            contains_multiple_emitters=len(ids) > 1,
            bridges_distinct_truth_activity_regions=region_count > 1,
            no_true_emitter_time_overlap=not ids))
    precision = tp_samples / pred_count if pred_count else None
    recall = tp_samples / truth_count if truth_count else None
    fragmented_regions = [i for i, (a, b) in enumerate(truth)
        if sum(c < b and d > a for c, d in predicted) > 1]
    return dict(scored_object="union of all emitter time intervals; full-band activity",
        event_iou_threshold=iou_threshold, expected_activity_events=len(truth), detected_activity_events=len(rows),
        matched_activity_events=len(matches), missed_activity_events=len(truth) - len(matches),
        unmatched_detected_events=len(rows) - len(matches),
        missed_activity_indices=[i for i in range(len(truth)) if i not in matched_truth],
        unmatched_detected_event_ids=[row["event_id"] for row in rows if row["event_id"] not in matched_events],
        activity_regions_with_multiple_detections=fragmented_regions,
        detections_bridging_separate_activity_regions=sum(e["bridges_distinct_truth_activity_regions"] for e in event_annotations),
        matched_boundaries=matches, tp_samples=tp_samples, fp_samples=fp_samples, fn_samples=fn_samples,
        sample_precision=precision, sample_recall=recall,
        sample_f1=(2 * tp_samples / (2 * tp_samples + fp_samples + fn_samples)
                   if (2 * tp_samples + fp_samples + fn_samples) else None),
        noise_exposure_samples=noise_samples, noise_exposure_s=noise_samples / meta["sample_rate_hz"],
        false_active_sample_fraction=fp_samples / noise_samples if noise_samples else None,
        false_active_fraction_is_independent_window_pfa=False,
        noise_only_false_event_count=len(rows) if not meta["emitters"] else None,
        noise_only_event_rate_per_s=(len(rows) / (noise_samples / meta["sample_rate_hz"]) if not meta["emitters"] else None),
        emitter_coverage=emitters, event_contents=event_annotations,
        per_emitter_identification_or_separation_proven=False)


def fingerprint(rows):
    keys = ("event_start_sample", "event_end_sample", "detected_start_sample", "detected_end_sample",
            "trigger_sample", "confirmation_sample", "termination_reason", "iq_sha256", "complete", "truncated",
            "feature_status")
    return [tuple(row[k] for k in keys) for row in rows]


def evaluate(args) -> int:
    root = args.dataset.expanduser().resolve()
    dataset = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if dataset["status"] != "COMPLETE":
        raise ValueError("Dataset generation is not COMPLETE; do not evaluate an incomplete run")
    if args.package_src:
        source = args.package_src.expanduser().resolve()
        if not (source / "signal_processing" / "__init__.py").is_file():
            raise ValueError("--package-src must contain signal_processing/")
        sys.path.insert(0, str(source))
    core = importlib.import_module("signal_processing.core.contracts")
    detection = importlib.import_module("signal_processing.detection.noise_estimation")
    runtime = importlib.import_module("signal_processing.runtime.pipeline")
    package = importlib.import_module("signal_processing")
    api = dict(StreamInfo=core.StreamInfo, IQBlock=core.IQBlock,
               GaussianNoiseEstimator=detection.GaussianNoiseEstimator,
               StreamingCapturePipeline=runtime.StreamingCapturePipeline)
    out = empty_output(args.output)
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != dataset["records_completed"]:
        raise ValueError("Manifest count disagrees with completed dataset")
    if args.limit:
        records = records[:args.limit]
    totals = dict(records=0, integrity_failures=0, expected_activity_events=0, detected_activity_events=0,
        matched_activity_events=0, missed_activity_events=0, unmatched_detected_events=0,
        tp_samples=0, fp_samples=0, fn_samples=0, noise_only_records=0, noise_only_events=0, noise_only_exposure_s=0.0,
        incomplete_captures=0, eof_limited_captures=0)
    results = []
    for entry in records:
        iq = np.load(root / entry["iq_file"], allow_pickle=False, mmap_mode="r")
        meta = json.loads((root / entry["annotations_file"]).read_text(encoding="utf-8"))
        if iq.ndim != 1 or iq.dtype != np.complex64 or len(iq) != meta["num_samples"] or iq_hash(iq) != meta["iq_sha256"]:
            raise ValueError(f"IQ contract/hash failed for {meta['record_id']}")
        rows, run = run_record(iq, meta, args, api)
        metrics = score_activity(meta, rows, args.event_iou)
        result = dict(record_id=meta["record_id"], scenario=meta["scenario"],
                      noise_model=meta["background"]["model"], snr_sweep_db=meta["sweep_snr_db"],
                      runtime=run, metrics=metrics, detections=rows)
        if args.check_partitions:
            # Compare the same functionality, including features if requested.
            other, other_run = run_record(iq, meta, args, api, random_chunks=True)
            passed = run["integrity_pass"] and other_run["integrity_pass"] and fingerprint(rows) == fingerprint(other)
            result["partition_check"] = dict(passed=bool(passed), random_chunk_runtime=other_run)
            run["integrity_pass"] &= bool(passed)
        write_json(out / f"{meta['record_id']}.json", result)
        results.append(dict(record_id=meta["record_id"], scenario=meta["scenario"],
            noise_model=meta["background"]["model"], snr_sweep_db=meta["sweep_snr_db"],
            report_file=f"{meta['record_id']}.json", integrity_pass=run["integrity_pass"],
            sample_f1=metrics["sample_f1"], missed_activity_events=metrics["missed_activity_events"],
            unmatched_detected_events=metrics["unmatched_detected_events"]))
        totals["records"] += 1
        totals["integrity_failures"] += not run["integrity_pass"]
        totals["incomplete_captures"] += sum(not row["complete"] or row["truncated"] for row in rows)
        totals["eof_limited_captures"] += run["eof_limited_capture_count"]
        for key in ("expected_activity_events", "detected_activity_events", "matched_activity_events",
                    "missed_activity_events", "unmatched_detected_events", "tp_samples", "fp_samples", "fn_samples"):
            totals[key] += metrics[key]
        if not meta["emitters"]:
            totals["noise_only_records"] += 1
            totals["noise_only_events"] += metrics["noise_only_false_event_count"]
            totals["noise_only_exposure_s"] += metrics["noise_exposure_s"]
        print(f"{meta['record_id']} {meta['scenario']}: activity matches {metrics['matched_activity_events']}/{metrics['expected_activity_events']}; "
              f"unmatched detections {metrics['unmatched_detected_events']}; integrity {run['integrity_pass']}", flush=True)
    denominator = 2 * totals["tp_samples"] + totals["fp_samples"] + totals["fn_samples"]
    totals["sample_f1"] = 2 * totals["tp_samples"] / denominator if denominator else None
    totals["noise_only_event_rate_per_s"] = (totals["noise_only_events"] / totals["noise_only_exposure_s"]
                                            if totals["noise_only_exposure_s"] else None)
    report = dict(script_version=VERSION, dataset_path=str(root), package_path=str(Path(package.__file__).resolve()),
        package_version=getattr(package, "__version__", "unknown"), environment=environment(),
        detector_configuration=dict(mode="full_band", window_size=args.window_size, hop_size=args.hop_size,
            block_size=args.block_size, pfa_on=args.pfa_on, pfa_off=args.pfa_off,
            start_persistence=2, end_persistence=3, pre_trigger_samples=args.pre_trigger,
            pre_signal_margin_samples=args.pre_margin, post_signal_samples=1024,
            feature_analysis=args.features, event_iou=args.event_iou),
        totals=totals, records=results, functional_integrity_pass=totals["integrity_failures"] == 0,
        detection_qualification_or_freeze_pass=None, limitations=dataset["limitations"])
    write_json(out / "evaluation.json", report)
    print(f"Evaluation ready: {out}", flush=True)
    return int(args.enforce_integrity and totals["integrity_failures"] > 0)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="Generate TorchSig IQ recordings and ALL-emitter labels")
    g.add_argument("--output", type=Path, required=True)
    g.add_argument("--repeats", type=int, default=5, help="Independent recordings per scenario/SNR/noise cell")
    g.add_argument("--samples", type=int, default=131072)
    g.add_argument("--sample-rate", type=float, default=1_000_000.0)
    g.add_argument("--center-frequency", type=float, default=915_000_000.0)
    g.add_argument("--seed", type=int, default=20261013)
    g.add_argument("--snrs", nargs="+", type=float, default=[-15, -10, -5, 0, 5, 10])
    g.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    g.add_argument("--classes", nargs="+", choices=CLASSES, default=list(DEFAULT_CLASSES))
    g.add_argument("--noise-models", nargs="+", choices=("white", "ar1"), default=["white"])
    g.add_argument("--noise-dbfs", type=float, default=-40.0)
    g.add_argument("--ar1-rho", type=float, default=0.8)
    g.add_argument("--bandwidth-fractions", nargs="+", type=float, default=[0.04, 0.08, 0.12])
    g.add_argument("--save-components", action="store_true", help="Also save clean emitter arrays and noise for diagnostics")
    g.add_argument("--previews", type=int, default=3, help="Save a figure for this many initial records")
    e = sub.add_parser("evaluate", help="Test the current FYP full-band detector and exact raw capture")
    e.add_argument("--dataset", type=Path, required=True)
    e.add_argument("--package-src", type=Path, help="Directory containing signal_processing/")
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--limit", type=int, default=0, help="Maximum records, 0 means all")
    e.add_argument("--window-size", type=int, default=512)
    e.add_argument("--hop-size", type=int, default=128)
    e.add_argument("--block-size", type=int, default=1024)
    e.add_argument("--pre-trigger", type=int, default=2048)
    e.add_argument("--pre-margin", type=int, default=1024)
    e.add_argument("--pfa-on", type=float, default=1e-4)
    e.add_argument("--pfa-off", type=float, default=1e-2)
    e.add_argument("--event-iou", type=float, default=0.5, help="Declared aggregate-activity event matching threshold")
    e.add_argument("--features", action="store_true", help="Also execute the pipeline's real feature extraction")
    e.add_argument("--check-partitions", action="store_true")
    e.add_argument("--enforce-integrity", action="store_true",
                   help="Nonzero on corrupt IQ, inconsistent quality, runtime/chunk faults; truthful EOF truncation is recorded separately")
    a = p.parse_args(argv)
    if a.command == "generate":
        if a.repeats < 1 or a.samples < 32768 or a.previews < 0 or a.seed < 0:
            p.error("Require repeats >= 1, samples >= 32768, previews >= 0 and seed >= 0")
        if not math.isfinite(a.sample_rate) or not 100000 <= a.sample_rate <= 10000000:
            p.error("sample-rate must be finite and between 100000 and 10000000 Hz")
        if not math.isfinite(a.center_frequency) or a.center_frequency < 0:
            p.error("center-frequency must be finite and nonnegative")
        if not math.isfinite(a.noise_dbfs) or not -120 <= a.noise_dbfs <= -20:
            p.error("noise-dbfs must be between -120 and -20")
        if not all(math.isfinite(s) and -30 <= s <= 30 for s in a.snrs):
            p.error("All snrs must be finite and between -30 and 30 dB")
        if not math.isfinite(a.ar1_rho) or not 0 <= a.ar1_rho < 0.99:
            p.error("ar1-rho must be in [0, 0.99)")
        if not all(math.isfinite(b) and 0.01 <= b <= 0.20 for b in a.bandwidth_fractions):
            p.error("bandwidth-fractions must be in [0.01, 0.20] of Fs")
        if any(len(v) != len(set(v)) for v in (a.scenarios, a.classes, a.noise_models, a.snrs)):
            p.error("Do not repeat scenarios, classes, noise models or SNR cells")
    else:
        if a.limit < 0 or not 1 <= a.hop_size <= a.window_size or a.block_size < 1:
            p.error("Require limit >= 0, 1 <= hop-size <= window-size, block-size >= 1")
        if min(a.pre_trigger, a.pre_margin) < 0 or not 0 < a.pfa_on < a.pfa_off < 1:
            p.error("Require nonnegative margins and 0 < pfa-on < pfa-off < 1")
        if not 0 < a.event_iou <= 1:
            p.error("event-iou must be in (0, 1]")
    return a


def main(argv=None) -> int:
    args = parse_args(argv)
    return generate(args) if args.command == "generate" else evaluate(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
