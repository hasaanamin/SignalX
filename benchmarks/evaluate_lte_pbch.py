"""Offline PBCH characterization using external BCH reference codewords.

Synthetic test truth is used by evaluation only, never by the real receiver.
Per-record opportunities are independent. Repeated frames inside a recording
are not treated as independent statistical trials. Not receiver-noise or RF
hardware qualification, and not a scheduled acquisition benchmark.
"""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.stats import beta

from signal_processing.core import IQBlock, StreamInfo
from signal_processing.lte import LTEBroadcastReceiver, LTESyncConfig
from tests.lte_pbch_fixture import frame_fixture, waveform


def evaluate(trials=10, profile="compatibility", snrs=(10., 0., -5.), seed=20261019,
             frames=4, random_quarter=False):
    results = []
    rng = np.random.default_rng(seed)
    for snr in snrs:
        for ports in (1, 2, 4):
            records = complete = verified = returned = wrong = synchronized = reached_pbch = 0
            attempts = 0
            failures = []
            frame_errors = []
            start_time = time.perf_counter()
            for trial in range(trials):
                pci, index = int(rng.integers(504)), int(rng.integers(30))
                onset, cfo = int(rng.integers(128, 1200)), float(rng.uniform(-20000, 20000))
                noise_seed = int(rng.integers(2**31))
                quarter = int(rng.integers(4)) if random_quarter else 0
                iq = waveform(pci, ports=ports, index=index, cfo=cfo, onset=onset,
                              snr_db=snr, seed=noise_seed, frames=frames, first_quarter=quarter)
                stream = StreamInfo("pbch", "independent-synthetic", 2e6, 2.14e9)
                config = LTESyncConfig.low_snr() if profile == "low_snr" else LTESyncConfig()
                receiver = LTEBroadcastReceiver(stream, config)
                mibs = []
                observations = []
                locked = False
                for seq, offset in enumerate(range(0, len(iq), 4093)):
                    batch = receiver.process_block(IQBlock(stream, seq, offset, None, iq[offset:offset+4093]))
                    mibs.extend(batch.mibs)
                    observations.extend(batch.synchronization)
                    locked |= any(o.pci == pci and o.state == "SYNCHRONIZED" for o in batch.synchronization)
                batch = receiver.flush()
                mibs.extend(batch.mibs)
                observations.extend(batch.synchronization)
                locked |= any(o.pci == pci and o.state == "SYNCHRONIZED" for o in batch.synchronization)
                synchronized += locked
                attempts += receiver.snapshot()["pbch_attempted_frames"]
                reached_pbch += receiver.snapshot()["pbch_attempted_frames"] > 0
                seen = set()
                for result in mibs:
                    returned += 1
                    frame = round(float(result.input_frame_start-onset)/20000)
                    error = abs(float(result.input_frame_start-onset-frame*20000))
                    payload, _ = frame_fixture(index, ports, max(0, frame), quarter)
                    okay = (0 <= frame < frames and error <= 2 and result.pci == pci
                        and result.antenna_ports == ports and result.frame_quarter == (frame+quarter) % 4
                        and result.mib.payload_bits == tuple(map(int, payload)))
                    if okay and frame not in seen:
                        seen.add(frame)
                        verified += 1
                        frame_errors.append(error)
                    else:
                        wrong += 1
                records += bool(seen)
                complete += len(seen) == frames
                if not seen and len(failures) < 20:
                    failures.append({"trial": trial, "pci": pci, "index": index, "onset": onset,
                        "cfo": cfo, "noise_seed": noise_seed,
                        "first_quarter": quarter,
                        "observations": [o.to_dict() for o in observations[:16]]})
            lower = float(beta.ppf(.05, records, trials-records+1)) if records else 0.
            results.append({"snr_db": snr, "antenna_ports": ports, "independent_recordings": trials,
                "recordings_with_verified_mib": records, "all_frames_verified": complete,
                "all_four_frames_verified": complete if frames == 4 else None,
                "correctly_synchronized_recordings": synchronized, "recordings_reaching_pbch": reached_pbch,
                "pbch_attempted_frames": attempts,
                "failure_examples": failures,
                "verified_frames": verified, "returned_mibs": returned, "wrong_or_duplicate_mibs": wrong,
                "record_detection_one_sided_95_lower": lower,
                "maximum_verified_frame_error_raw_samples": max(frame_errors, default=None),
                "elapsed_s_including_generation": time.perf_counter()-start_time})
    return {"environment": {"platform": platform.platform(), "python": platform.python_version(),
                "numpy": np.__version__, "scipy": scipy.__version__},
            "fixture": "external srsRAN BCH coding + independent synthetic OFDM/CRS assembly",
            "input_rate_hz": 2000000, "frames_per_recording": frames, "seed": seed,
            "bch_phase": "random 0..3" if random_quarter else "zero",
            "profile": profile,
            "runtime_truth_inputs": False, "noise": "iid complex Gaussian; SNR measured over complete synthetic waveform",
            "scope": "Finite synthetic characterization only; no low-SNR/Pfa/timing/hardware qualification",
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("compatibility", "low_snr"), default="compatibility")
    parser.add_argument("--snr", type=float, help="Single SNR; default retains original 10/0/-5 sweep")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--random-quarter", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or not 1 <= args.frames <= 100:
        parser.error("trials must be positive; frames must be 1..100")
    report = evaluate(args.trials, args.profile, (args.snr,) if args.snr is not None else (10., 0., -5.),
                      args.seed, args.frames, args.random_quarter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 1 if any(r["wrong_or_duplicate_mibs"] for r in report["results"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
