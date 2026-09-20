"""Executable demonstration of the end-to-end streaming detection, capture, and 56-feature extraction pipeline.

Feeds stationary synthetic noise plus multiple known modulated RF bursts (CW tone, BPSK, QPSK)
through the streaming pipeline and prints detailed landmark timestamps, capture sidecar metadata,
and all 56 extracted scalar features.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from signal_processing.core.contracts import StreamInfo
from signal_processing.detection.noise_estimation import GaussianNoiseEstimator
from signal_processing.runtime.pipeline import StreamingCapturePipeline
from signal_processing.sources.synthetic_source import BurstDescriptor, SyntheticBurstSource


def main() -> None:
    print("=" * 80)
    print("FYP STREAMING RF CAPTURE & FEATURE EXTRACTION PIPELINE DEMONSTRATION")
    print("=" * 80)

    fs = 1_000_000.0  # 1 MS/s
    fc = 915_000_000.0  # 915 MHz ISM center
    stream_info = StreamInfo(
        stream_id="stream-demo-01",
        source_id="synthetic-usrp",
        sample_rate_hz=fs,
        center_frequency_hz=fc,
    )

    # Define scheduled signal bursts across a 50,000 sample stream (50 ms duration)
    bursts = [
        BurstDescriptor(
            start_sample=5_000,
            duration_samples=4_000,  # 4 ms
            snr_db=20.0,
            modulation="cw",
            freq_offset_hz=25_000.0,  # +25 kHz tone
        ),
        BurstDescriptor(
            start_sample=20_000,
            duration_samples=6_000,  # 6 ms
            snr_db=15.0,
            modulation="bpsk",
            freq_offset_hz=-50_000.0,  # -50 kHz BPSK burst
        ),
        BurstDescriptor(
            start_sample=35_000,
            duration_samples=5_000,  # 5 ms
            snr_db=18.0,
            modulation="qpsk",
            freq_offset_hz=100_000.0,  # +100 kHz QPSK burst
        ),
    ]

    noise_power_linear = 1e-4  # -40 dBFS noise floor
    source = SyntheticBurstSource(
        stream_info=stream_info,
        total_samples=50_000,
        block_size=1024,
        noise_power_linear=noise_power_linear,
        bursts=bursts,
        seed=2026,
    )

    print(f"\n[Source] Initialized synthetic stream: {source.total_samples} samples @ {fs/1e6:.1f} MS/s")
    print(f"[Source] Scheduled {len(bursts)} bursts:")
    for idx, b in enumerate(bursts, 1):
        print(f"  Burst #{idx}: start={b.start_sample}, duration={b.duration_samples} samples ({b.duration_samples/fs*1e3:.1f} ms), "
              f"SNR={b.snr_db} dB, mod={b.modulation}, offset={b.freq_offset_hz/1e3:.1f} kHz")

    # Configure detection and circular buffering pipeline
    noise_estimator = GaussianNoiseEstimator(
        window_size=256,
        pfa_on=1e-4,
        pfa_off=1e-2,
        calibrated_noise_power_linear=noise_power_linear,
    )

    pipeline = StreamingCapturePipeline(
        stream_info=stream_info,
        ring_capacity=16_384,
        window_size=256,
        hop_size=64,
        noise_estimator=noise_estimator,
        start_persistence=2,
        end_persistence=3,
        pre_trigger_samples=512,
        pre_signal_margin_samples=256,
        post_signal_samples=256,
        min_post_trigger_samples=512,
        synchronous_analysis=True,
    )

    print("\n[Pipeline] Streaming blocks into pipeline...")
    t_start = time.perf_counter()
    emitted_sidecars = []
    blocks_processed = 0

    while (block := source.read_block()) is not None:
        sidecars = pipeline.process_block(block)
        emitted_sidecars.extend(sidecars)
        blocks_processed += 1

    emitted_sidecars.extend(pipeline.flush())
    t_elapsed = time.perf_counter() - t_start

    print(f"[Pipeline] Finished: processed {blocks_processed} transport blocks in {t_elapsed*1e3:.2f} ms")
    print(f"[Pipeline] Effective ingestion throughput: {source.total_samples / t_elapsed / 1e6:.2f} MS/s")
    print(f"[Pipeline] Emitted {len(emitted_sidecars)} captured events\n")

    # Inspect each captured event and extracted feature vector
    for idx, sidecar in enumerate(emitted_sidecars, 1):
        evt = sidecar.event
        print("=" * 80)
        print(f"CAPTURED EVENT #{idx} [{evt.event_id}]")
        print("=" * 80)
        print(f"  Status / Quality: complete={evt.complete}, truncated={sidecar.is_truncated}, "
              f"termination='{sidecar.termination_reason}', feature_status='{sidecar.feature_status}'")
        print("  Landmarks (samples):")
        print(f"    Event Extent:     [{evt.event_start_sample}, {evt.event_end_sample}) (total {evt.num_samples} samples, {evt.duration_s*1e3:.2f} ms)")
        print(f"    Detected Signal:  [{evt.detected_start_sample}, {evt.detected_end_sample}) ({evt.signal_samples} samples, {evt.signal_duration_s*1e3:.2f} ms)")
        print(f"    Trigger Landmark: {evt.trigger_sample} (first ON sample={sidecar.first_on_sample}, confirmed at {sidecar.confirmation_sample})")
        print(f"    Lead Margin:      {evt.pre_signal_samples} samples (requested={sidecar.requested_leading_samples}, actual={sidecar.actual_leading_samples})")
        print(f"    Trail Margin:     {evt.post_signal_samples} samples (requested={sidecar.requested_trailing_samples}, actual={sidecar.actual_trailing_samples})")

        if sidecar.features is not None:
            fv = sidecar.features
            print(f"\n  Extracted 56 Feature Values (Schema {fv.feature_schema_version}):")
            keys = sorted(fv.features.keys())
            for i in range(0, len(keys), 2):
                k1 = keys[i]
                v1 = fv.features[k1]
                s1 = f"{k1}: {v1.value:.4e} {v1.unit}" if v1.valid else f"{k1}: INVALID ({v1.reason})"
                if i + 1 < len(keys):
                    k2 = keys[i + 1]
                    v2 = fv.features[k2]
                    s2 = f"{k2}: {v2.value:.4e} {v2.unit}" if v2.valid else f"{k2}: INVALID ({v2.reason})"
                    print(f"    {s1:<42} | {s2}")
                else:
                    print(f"    {s1}")
        print()

    print("=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
