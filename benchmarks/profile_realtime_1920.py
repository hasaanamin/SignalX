"""CPU attribution with real DSP; unpaced results are not acquisition passes."""

import argparse
import cProfile
import pstats
import time
from pathlib import Path

import numpy as np

from signal_processing.core import IQBlock, StreamInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("adaptive", "adaptive_on", "adaptive_noise", "lte", "lte_noise", "lte_tone"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--sync-backend", default="numba")
    parser.add_argument("--no-blind-prefilter",action="store_true")
    parser.add_argument("--tone-hz",type=float,default=0.)
    parser.add_argument("--fft-workers",type=int,choices=(1,2,3,4),default=1)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    stream = StreamInfo("profile-1920", "offline", 1_920_000.0, 2.14e9)
    if args.mode.startswith("adaptive"):
        from benchmarks.process_source import _ScheduledIQ
        from signal_processing.detection import POSITIVE_SNR_IID_1920, AdaptiveRobustNoiseEstimator
        from signal_processing.runtime import StreamingCapturePipeline

        noise = AdaptiveRobustNoiseEstimator(
            window_size=256,
            initial_noise_power_linear=1e-4,
            time_constant_s=0.5,
            sample_rate_hz=stream.sample_rate_hz,
            adaptation_window_size=1024,
        )
        processor = StreamingCapturePipeline(
            stream,
            noise_estimator=noise,
            window_size=256,
            hop_size=64,
            ring_capacity=4096,
            start_persistence=2,
            end_persistence=2,
            pre_trigger_samples=256,
            pre_signal_margin_samples=0,
            post_signal_samples=0,
            min_post_trigger_samples=0,
            max_event_samples=4096,
            fixed_quiet_batch=True,
            enable_feature_analysis=False,
            detection_profile_id=POSITIVE_SNR_IID_1920.profile_id,
            boundary_uncertainty_samples=256,
        )
        generator = _ScheduledIQ(args.block_size)
        iq = np.concatenate(
            [
                generator.generate(
                    start, min(args.block_size, round(args.seconds * 1_920_000) - start)
                ).copy()
                for start in range(0, round(args.seconds * 1_920_000), args.block_size)
            ]
        )
        if args.mode == 'adaptive_on':
            iq.fill(.2+.1j)
        elif args.mode == 'adaptive_noise':
            rng = np.random.default_rng(23483)
            iq = ((rng.normal(size=len(iq))+1j*rng.normal(size=len(iq)))*np.sqrt(5e-5)).astype(np.complex64)
    else:
        from signal_processing.lte import LTEBroadcastReceiver, LTESyncConfig

        processor = LTEBroadcastReceiver(
            stream,
            LTESyncConfig.native(low_snr=True),
            sync_backend=args.sync_backend,
            blind_prefilter=not args.no_blind_prefilter,
            fft_workers=args.fft_workers,
            decode_cadence="bch_block",
            measure_timing=True,
        )
        if args.mode == "lte_noise":
            rng = np.random.default_rng(23483)
            count = round(args.seconds * 1_920_000)
            iq = (rng.normal(size=count) + 1j * rng.normal(size=count)).astype(np.complex64)
        elif args.mode == 'lte_tone':
            count = round(args.seconds*1_920_000)
            iq = np.exp(2j*np.pi*args.tone_hz*np.arange(count)/1_920_000).astype(np.complex64)
        else:
            iq = np.fromfile(args.input, "<c8", count=round(args.seconds * 1_920_000))
    blocks = [
        IQBlock(stream, seq, start, None, iq[start : start + args.block_size])
        for seq, start in enumerate(range(0, len(iq), args.block_size))
    ]
    profiler = cProfile.Profile()
    count = sync = 0
    started = time.perf_counter()
    cpu_started = time.process_time()
    if args.profile:
        profiler.enable()
    for block in blocks:
        result = processor.process_block(block)
        count += len(result) if args.mode.startswith("adaptive") else len(result.mibs)
        if not args.mode.startswith("adaptive"):
            sync += len(result.synchronization)
    result = processor.flush()
    count += len(result) if args.mode.startswith("adaptive") else len(result.mibs)
    if args.profile:
        profiler.disable()
    elapsed = time.perf_counter() - started
    print(
        dict(
            samples=len(iq),
            seconds=elapsed,
            cpu_seconds=time.process_time()-cpu_started,
            samples_per_second=len(iq) / elapsed,
            outputs=count,
            sync=sync,
        )
    )
    if args.mode.startswith("adaptive"):
        processor.shutdown()
    else:
        print(processor.snapshot())
        print(processor.stage_timing)
    if args.profile:
        pstats.Stats(profiler).strip_dirs().sort_stats("tottime").print_stats(35)


if __name__ == "__main__":
    main()
