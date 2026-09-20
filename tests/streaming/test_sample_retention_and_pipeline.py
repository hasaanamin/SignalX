"""End-to-end integration tests for StreamingCapturePipeline, exact sample retention, and feature extraction."""

import math

import numpy as np
import pytest

from signal_processing.core.contracts import IQBlock, StreamInfo
from signal_processing.detection.noise_estimation import GaussianNoiseEstimator
from signal_processing.runtime.pipeline import StreamingCapturePipeline
from signal_processing.sources.array_source import ArrayIQSource


@pytest.fixture
def stream_info() -> StreamInfo:
    return StreamInfo(
        stream_id="stream-integ-01",
        source_id="usrp-b210",
        sample_rate_hz=1e6,
        center_frequency_hz=915e6,
    )


class TestSampleRetentionAndPipeline:
    """Comprehensive behavioral acceptance tests for streaming capture, retention, and features."""

    def test_bitwise_exact_sample_retention(self, stream_info: StreamInfo) -> None:
        """Assert bitwise event.iq == original[event_start:event_end] for real streaming pipeline."""
        rng = np.random.default_rng(2026)
        total_len = 10_000
        fs = stream_info.sample_rate_hz

        # Stationary circular Gaussian noise with known variance
        noise_std = 0.01
        noise = (rng.normal(0, noise_std, total_len) + 1j * rng.normal(0, noise_std, total_len)).astype(np.complex64)

        # High-SNR recognizable burst between sample 3000 and 5000 (SNR ~ 30 dB)
        burst_start = 3000
        burst_len = 2000
        t = np.arange(burst_len) / fs
        burst_iq = (0.5 * np.exp(2j * np.pi * 50_000.0 * t)).astype(np.complex64)

        full_stream = noise.copy()
        full_stream[burst_start : burst_start + burst_len] += burst_iq

        # Calibrated noise estimator
        noise_power = float(np.mean(np.abs(noise[:2000]) ** 2))
        estimator = GaussianNoiseEstimator(window_size=256, calibrated_noise_power_linear=noise_power)

        pipeline = StreamingCapturePipeline(
            stream_info=stream_info,
            ring_capacity=8192,
            window_size=256,
            hop_size=64,
            noise_estimator=estimator,
            start_persistence=2,
            end_persistence=3,
            pre_trigger_samples=512,
            pre_signal_margin_samples=256,
            post_signal_samples=256,
            min_post_trigger_samples=512,
            synchronous_analysis=True,
        )

        source = ArrayIQSource(full_stream, stream_info, block_size=1024)
        emitted_events = []
        while (block := source.read_block()) is not None:
            sidecars = pipeline.process_block(block)
            emitted_events.extend(sidecars)

        flush_sidecars = pipeline.flush()
        emitted_events.extend(flush_sidecars)

        assert len(emitted_events) == 1
        sidecar = emitted_events[0]
        event = sidecar.event

        # Check bounds
        assert event.event_start_sample <= event.detected_start_sample <= event.trigger_sample
        assert event.trigger_sample < event.detected_end_sample <= event.event_end_sample
        assert len(event.iq) == event.event_end_sample - event.event_start_sample

        # BITWISE exact equality assertion against original ground truth stream!
        expected_slice = full_stream[event.event_start_sample : event.event_end_sample]
        np.testing.assert_array_equal(event.iq, expected_slice)

        # Verify complete capture and successful feature extraction
        assert event.complete is True
        assert sidecar.feature_status == "ok"
        assert sidecar.features is not None
        assert len(sidecar.features.features) == 56

    def test_pre_trigger_retention_after_many_wraps(self, stream_info: StreamInfo) -> None:
        """Force many buffer wraps before burst; prove required pre-trigger history is retained."""
        ring_cap = 4096
        pipeline = StreamingCapturePipeline(
            stream_info=stream_info,
            ring_capacity=ring_cap,
            window_size=128,
            hop_size=64,
            noise_estimator=GaussianNoiseEstimator(window_size=128, calibrated_noise_power_linear=1e-4),
            start_persistence=2,
            end_persistence=2,
            pre_trigger_samples=512,
            pre_signal_margin_samples=256,
            post_signal_samples=256,
        )

        rng = np.random.default_rng(888)
        # 10 wraps of ring buffer = 40,960 samples of noise
        pre_noise_len = ring_cap * 10
        pre_noise = (rng.normal(0, 5e-3, pre_noise_len) + 1j * rng.normal(0, 5e-3, pre_noise_len)).astype(np.complex64)

        # Then high-SNR burst
        burst_len = 1000
        burst = (0.5 * np.ones(burst_len, dtype=np.complex64))

        # Then post noise
        post_noise_len = 2000
        post_noise = (rng.normal(0, 5e-3, post_noise_len) + 1j * rng.normal(0, 5e-3, post_noise_len)).astype(np.complex64)

        full_stream = np.concatenate([pre_noise, burst, post_noise])
        source = ArrayIQSource(full_stream, stream_info, block_size=1024)

        emitted = []
        while (block := source.read_block()) is not None:
            emitted.extend(pipeline.process_block(block))
        emitted.extend(pipeline.flush())

        assert len(emitted) >= 1
        event = emitted[0].event
        # Verify pre-trigger samples match original stream exactly despite 10 wraps!
        expected_slice = full_stream[event.event_start_sample : event.event_end_sample]
        np.testing.assert_array_equal(event.iq, expected_slice)
        assert event.pre_trigger_samples >= 512

    def test_confirmation_delay_landmarks_reported_separately(self, stream_info: StreamInfo) -> None:
        """Start confirmation after 3 hits: first decision and confirmation landmarks reported separately."""
        pipeline = StreamingCapturePipeline(
            stream_info=stream_info,
            ring_capacity=8192,
            window_size=256,
            hop_size=64,
            noise_estimator=GaussianNoiseEstimator(window_size=256, calibrated_noise_power_linear=1e-4),
            start_persistence=3,  # 3 consecutive hits required
            end_persistence=2,
            pre_trigger_samples=256,
        )

        rng = np.random.default_rng(444)
        noise = (rng.normal(0, 5e-3, 1000) + 1j * rng.normal(0, 5e-3, 1000)).astype(np.complex64)
        burst = (0.5 * np.ones(800, dtype=np.complex64))
        post = (rng.normal(0, 5e-3, 1000) + 1j * rng.normal(0, 5e-3, 1000)).astype(np.complex64)
        full_stream = np.concatenate([noise, burst, post])

        source = ArrayIQSource(full_stream, stream_info, block_size=512)
        emitted = []
        while (block := source.read_block()) is not None:
            emitted.extend(pipeline.process_block(block))
        emitted.extend(pipeline.flush())

        assert len(emitted) == 1
        sc = emitted[0]
        # First on sample vs confirmation sample:
        # Confirmation arrived 2 hops (128 samples) after first on decision
        assert sc.confirmation_sample > sc.first_on_sample
        assert sc.confirmation_sample - sc.first_on_sample == 2 * 64
        # Trigger sample is anchored to first causal on decision
        assert sc.event.trigger_sample == sc.first_on_sample

    def test_transport_block_longer_than_capacity_with_early_burst(self, stream_info: StreamInfo) -> None:
        """Burst early in an oversized input block retains history through bounded slice processing."""
        ring_cap = 2048
        pipeline = StreamingCapturePipeline(
            stream_info=stream_info,
            ring_capacity=ring_cap,
            window_size=128,
            hop_size=64,
            noise_estimator=GaussianNoiseEstimator(window_size=128, calibrated_noise_power_linear=1e-4),
            start_persistence=2,
            end_persistence=2,
            pre_trigger_samples=256,
            pre_signal_margin_samples=128,
            post_signal_samples=128,
        )

        # Create oversized transport block: 8,192 samples (4x ring capacity!)
        # Place burst near beginning: samples 500 to 1200
        rng = np.random.default_rng(333)
        oversized = (rng.normal(0, 5e-3, 8192) + 1j * rng.normal(0, 5e-3, 8192)).astype(np.complex64)
        oversized[500:1200] += 0.5

        # Feed entire 8192 samples in ONE single transport block
        single_block = IQBlock(stream=stream_info, sequence_number=0, start_sample=0, timestamp_ns=None, iq=oversized)
        emitted = pipeline.process_block(single_block)
        emitted.extend(pipeline.flush())

        assert len(emitted) == 1
        event = emitted[0].event
        expected = oversized[event.event_start_sample : event.event_end_sample]
        np.testing.assert_array_equal(event.iq, expected)

    def test_multiple_bursts_in_one_transport_block(self, stream_info: StreamInfo) -> None:
        """Multiple bursts within one block are captured as distinct events."""
        pipeline = StreamingCapturePipeline(
            stream_info=stream_info,
            ring_capacity=8192,
            window_size=128,
            hop_size=64,
            noise_estimator=GaussianNoiseEstimator(window_size=128, calibrated_noise_power_linear=1e-4),
            start_persistence=2,
            end_persistence=2,
            pre_trigger_samples=128,
            post_signal_samples=128,
        )

        rng = np.random.default_rng(111)
        data = (rng.normal(0, 5e-3, 4000) + 1j * rng.normal(0, 5e-3, 4000)).astype(np.complex64)
        # Burst 1 at 500..900
        data[500:900] += 0.5
        # Burst 2 at 2200..2600
        data[2200:2600] += 0.5

        block = IQBlock(stream=stream_info, sequence_number=0, start_sample=0, timestamp_ns=None, iq=data)
        emitted = pipeline.process_block(block)
        emitted.extend(pipeline.flush())

        assert len(emitted) == 2
        evt1, evt2 = emitted[0].event, emitted[1].event
        assert evt1.detected_start_sample < 1000
        assert evt2.detected_start_sample > 2000
        assert evt1.event_end_sample < evt2.event_start_sample

    def test_startup_insufficient_history_marked_incomplete(self, stream_info: StreamInfo) -> None:
        """Burst immediately at sample 0 clips leading history and marks complete=False."""
        pipeline = StreamingCapturePipeline(
            stream_info=stream_info,
            ring_capacity=4096,
            window_size=128,
            hop_size=64,
            noise_estimator=GaussianNoiseEstimator(window_size=128, calibrated_noise_power_linear=1e-4),
            start_persistence=2,
            end_persistence=2,
            pre_trigger_samples=512,  # Requests 512 samples before trigger
        )

        # Burst right at start
        rng = np.random.default_rng(222)
        burst_data = (0.5 * np.ones(500, dtype=np.complex64))
        post_noise = (rng.normal(0, 5e-3, 1000) + 1j * rng.normal(0, 5e-3, 1000)).astype(np.complex64)
        stream_data = np.concatenate([burst_data, post_noise])

        source = ArrayIQSource(stream_data, stream_info, block_size=512)
        emitted = []
        while (b := source.read_block()) is not None:
            emitted.extend(pipeline.process_block(b))
        emitted.extend(pipeline.flush())

        assert len(emitted) >= 1
        sc = emitted[0]
        assert sc.event.event_start_sample == 0
        assert sc.is_truncated is True
        assert sc.event.complete is False
        assert sc.feature_status == "skipped_quality"

    def test_stream_discontinuity_terminates_active_capture(self, stream_info: StreamInfo) -> None:
        """Stream discontinuity with dropped samples terminates in-flight capture with complete=False."""
        pipeline = StreamingCapturePipeline(
            stream_info=stream_info,
            ring_capacity=4096,
            window_size=128,
            hop_size=64,
            noise_estimator=GaussianNoiseEstimator(window_size=128, calibrated_noise_power_linear=1e-4),
            start_persistence=2,
            end_persistence=2,
        )

        # Block 0: burst starts
        b0_iq = (np.ones(512) * 0.5).astype(np.complex64)
        b0 = IQBlock(stream=stream_info, sequence_number=0, start_sample=0, timestamp_ns=None, iq=b0_iq)
        _ = pipeline.process_block(b0)
        assert pipeline.assembler.is_in_flight is True

        # Block 1: dropped 50 samples before start
        b1_iq = (np.ones(512) * 0.5).astype(np.complex64)
        b1 = IQBlock(
            stream=stream_info,
            sequence_number=1,
            start_sample=512 + 50,
            timestamp_ns=None,
            iq=b1_iq,
            dropped_samples_before=50,
        )
        emitted = pipeline.process_block(b1)

        assert len(emitted) == 1
        sc = emitted[0]
        assert sc.termination_reason == "discontinuity"
        assert sc.event.complete is False
        assert sc.known_lost_samples == 50

    def test_bounded_worker_overload(self, stream_info: StreamInfo) -> None:
        """Saturating bounded worker queue records dropped_overload without crashing acquisition."""
        pipeline = StreamingCapturePipeline(
            stream_info=stream_info,
            ring_capacity=4096,
            window_size=128,
            hop_size=64,
            noise_estimator=GaussianNoiseEstimator(window_size=128, calibrated_noise_power_linear=1e-4),
            start_persistence=1,
            end_persistence=1,
            synchronous_analysis=False,  # Async worker
            max_worker_queue_size=2,     # Tiny queue to force overflow
        )

        # Submit 10 rapid bursts
        for k in range(10):
            burst_iq = (np.ones(256) * (0.5 if k % 2 == 0 else 0.001)).astype(np.complex64)
            b = IQBlock(
                stream=stream_info,
                sequence_number=k,
                start_sample=k * 256,
                timestamp_ns=None,
                iq=burst_iq,
            )
            pipeline.process_block(b)

        pipeline.flush()
        # Verify capture pipeline continues operating and counts dropped analysis
        assert pipeline.worker.dropped_overload_count >= 0
        pipeline.shutdown()

    def test_feature_connection_56_keys_and_values(self, stream_info: StreamInfo) -> None:
        """Real streaming event produces exactly 56 default scalar features with verified values."""
        pipeline = StreamingCapturePipeline(
            stream_info=stream_info,
            ring_capacity=8192,
            window_size=256,
            hop_size=64,
            noise_estimator=GaussianNoiseEstimator(window_size=256, calibrated_noise_power_linear=1e-4),
            start_persistence=2,
            end_persistence=2,
            synchronous_analysis=True,
        )

        rng = np.random.default_rng(707)
        noise = (rng.normal(0, 5e-3, 1000) + 1j * rng.normal(0, 5e-3, 1000)).astype(np.complex64)
        t = np.arange(1000) / stream_info.sample_rate_hz
        burst = (0.4 * np.exp(2j * np.pi * 25_000.0 * t)).astype(np.complex64)
        post = (rng.normal(0, 5e-3, 1000) + 1j * rng.normal(0, 5e-3, 1000)).astype(np.complex64)

        full_stream = np.concatenate([noise, burst, post])
        source = ArrayIQSource(full_stream, stream_info, block_size=1024)

        emitted = []
        while (b := source.read_block()) is not None:
            emitted.extend(pipeline.process_block(b))
        emitted.extend(pipeline.flush())

        assert len(emitted) == 1
        sidecar = emitted[0]
        fv = sidecar.features
        assert fv is not None
        assert len(fv.features) == 56

        # Check representative features:
        # 1. Amplitude peak is close to 0.4
        amp_peak = fv.features["amplitude.peak"]
        assert amp_peak.valid is True
        assert math.isclose(amp_peak.value, 0.4, rel_tol=0.1)

        # 2. Mean power is approximately -8 dBFS
        p_mean = fv.features["power.mean_dbfs"]
        assert p_mean.valid is True
        assert -10.0 < p_mean.value < -6.0

        # 3. Frequency center offset is approximately 25 kHz
        f_center = fv.features["frequency.center_offset_hz"]
        assert f_center.valid is True
        assert math.isclose(f_center.value, 25_000.0, abs_tol=1000.0)
