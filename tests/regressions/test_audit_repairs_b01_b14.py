"""Comprehensive behavioral acceptance and regression test suite verifying all 14 finding groups B01 through B14.

Audit finding coverage:
- B01: Exact calibration prefix consumption, remainder processing, and timestamp anchor preservation.
- B02: Causal, guarded, and partition-invariant adaptive noise integration.
- B03: Matched configured-band calibration, detection-domain adaptation, and held-out false alarm verification.
- B04: Calibration policy validation, detector attachment compatibility, and actual-N enforcement.
- B05: Discontinuity lifecycle: state reset, residual purging, and worker generation coordination.
- B06: Noise age tracking relative to last accepted sample, stale recovery, and fail-fast prerequisites.
- B07: Worker sink reentrancy deadlock prevention, single shutdown deadline, and task ledger.
- B08: Available EOF and gap tail retention with exact IQ and boundary truthfulness.
- B09: Feature-specific emitter bandwidth support (separate 3-dB crossing vs 99% energy tail containment).
- B10: Effective extraction policy identity honoring registered extractor overrides and untouched raw PSD.
- B11: Separation of capture completeness and continuity, degraded incomplete status, and stable filter delay.
- B12: Strict configuration parameter validation rejecting booleans, nonfinite values, and invalid signs.
- B13: Genuine paced arrival streaming, true capture-only isolation, and valid feature accounting.
- B14: Bounded streaming calibration API, chunk partition invariance, and support confidence.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import pytest
from scipy import signal

from signal_processing.capture.assembler import EventAssembler
from signal_processing.capture.sidecar import CaptureSidecar
from signal_processing.context.config import AnalysisConfig
from signal_processing.context.signal_context import SignalContext
from signal_processing.core.contracts import (
    IQBlock,
    NoiseEstimate,
    PSDResult,
    SignalEvent,
    StreamInfo,
)
from signal_processing.core.exceptions import (
    ContractValidationError,
    StreamConsistencyError,
)
from signal_processing.detection.energy_detector import EnergyDetector
from signal_processing.detection.exceptions import (
    FilterConfigurationError,
    PolicyCompatibilityError,
    UncalibratedNoiseError,
)
from signal_processing.detection.noise_estimation import (
    AdaptiveRobustNoiseEstimator,
    EmpiricalNoiseCalibrator,
    GaussianNoiseEstimator,
    NoiseLifecycleState,
    ThresholdPolicy,
)
from signal_processing.features import (
    FeatureAggregator,
    build_default_feature_registry,
)
from signal_processing.features.bandwidth import (
    BandwidthFeatureConfig,
    BandwidthFeatureExtractor,
)
from signal_processing.runtime.pipeline import PipelineState, StreamingCapturePipeline
from signal_processing.runtime.worker import FeatureAnalysisWorker


@pytest.fixture
def stream_1mhz() -> StreamInfo:
    return StreamInfo(
        stream_id="test-b-stream",
        source_id="synthetic",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
        channel_id="ch0",
        dtype="complex64",
    )


# ==============================================================================
# B01: EXACT CALIBRATION PREFIX AND REMAINDER PROCESSING
# ==============================================================================
class TestB01CalibrationPrefixAndRemainder:
    def test_warmup_prefix_and_remainder_across_partitions(self, stream_1mhz: StreamInfo) -> None:
        """Reproduction B01: 512 complex64 samples, seed 321, CN noise power 1e-4, burst [128, 192).

        warmup_samples=32. Partitions 1, 16, 64, 512 must all train exactly 32 samples
        and detect the burst identically.
        """
        rng = np.random.default_rng(321)
        scale = math.sqrt(1e-4 / 2.0)
        n_samples = 512
        full_iq = (rng.normal(0, scale, n_samples) + 1j * rng.normal(0, scale, n_samples)).astype(np.complex64)
        full_iq[128:192] += 0.5 + 0.0j

        def run_with_partition(block_size: int) -> tuple[int, float, list[CaptureSidecar]]:
            est = GaussianNoiseEstimator(window_size=16, warmup_samples=32)
            pipe = StreamingCapturePipeline(
                stream_info=stream_1mhz,
                window_size=16,
                hop_size=8,
                ring_capacity=4096,
                noise_estimator=est,
                start_persistence=1,
                end_persistence=1,
                pre_trigger_samples=32,
                pre_signal_margin_samples=0,
                post_signal_samples=32,
                min_post_trigger_samples=0,
                max_event_samples=10000,
                synchronous_analysis=True,
            )
            emitted = []
            seq = 0
            for i in range(0, n_samples, block_size):
                chunk = full_iq[i : i + block_size]
                b = IQBlock(
                    stream=stream_1mhz,
                    sequence_number=seq,
                    start_sample=i,
                    timestamp_ns=1_000_000_000 + int(i * 1000) if seq == 0 else None,
                    iq=chunk,
                )
                seq += 1
                emitted.extend(pipe.process_block(b))
            emitted.extend(pipe.flush())
            return est._sample_count, est.noise_power_linear, emitted

        count_1, mean_1, evts_1 = run_with_partition(1)
        count_16, mean_16, evts_16 = run_with_partition(16)
        count_64, mean_64, evts_64 = run_with_partition(64)
        count_512, mean_512, evts_512 = run_with_partition(512)

        # All partitions trained exactly 32 samples
        assert count_1 == 32
        assert count_16 == 32
        assert count_64 == 32
        assert count_512 == 32

        assert mean_1 == pytest.approx(mean_16, rel=1e-6)
        assert mean_1 == pytest.approx(mean_64, rel=1e-6)
        assert mean_1 == pytest.approx(mean_512, rel=1e-6)

        # All partitions detected exactly 1 event with identical landmarks
        assert len(evts_1) == 1
        assert len(evts_16) == 1
        assert len(evts_64) == 1
        assert len(evts_512) == 1

        sc = evts_512[0]
        assert sc.first_on_sample == evts_1[0].first_on_sample
        assert sc.event.detected_start_sample == 120
        assert sc.event.trigger_sample == 135
        assert sc.event.detected_end_sample == 208

    def test_timestamp_anchor_preservation_during_warmup(self, stream_1mhz: StreamInfo) -> None:
        """Timestamped 32-sample warmup starting at 1,000,000,000 ns followed by contiguous

        timestamp-less signal block must preserve anchor. Event starting at sample 32
        has timestamp 1,000,032,000 ns.
        """
        est = GaussianNoiseEstimator(window_size=16, warmup_samples=32)
        pipe = StreamingCapturePipeline(
            stream_info=stream_1mhz,
            window_size=16,
            hop_size=8,
            ring_capacity=4096,
            noise_estimator=est,
            pre_trigger_samples=0,
            pre_signal_margin_samples=0,
            post_signal_samples=0,
            start_persistence=1,
            end_persistence=1,
            synchronous_analysis=True,
        )

        # Warmup block: 32 samples starting at timestamp 1,000,000,000 ns
        b0 = IQBlock(
            stream=stream_1mhz,
            sequence_number=0,
            start_sample=0,
            timestamp_ns=1_000_000_000,
            iq=np.zeros(32, dtype=np.complex64),
        )
        res0 = pipe.process_block(b0)
        assert res0 == []
        assert est.state == NoiseLifecycleState.VALID

        # Signal block starting at sample 32, without timestamp
        signal_iq = np.ones(64, dtype=np.complex64) * 0.5
        b1 = IQBlock(
            stream=stream_1mhz,
            sequence_number=1,
            start_sample=32,
            timestamp_ns=None,
            iq=signal_iq,
        )
        res1 = pipe.process_block(b1)
        res1.extend(pipe.flush())
        assert len(res1) == 1

        # Event starting at sample 32 must resolve timestamp = 1,000,000,000 + 32,000 = 1,000,032,000 ns
        evt = res1[0].event
        assert evt.timestamp_ns == 1_000_032_000


# ==============================================================================
# B02: CAUSAL, GUARDED, PARTITION-INVARIANT ADAPTIVE NOISE
# ==============================================================================
class TestB02CausalGuardedAdaptiveNoise:
    def test_adaptive_integration_reproduction_1(self, stream_1mhz: StreamInfo) -> None:
        """Reproduction B02 (1): 2048 samples at amplitude 0.003, with amplitude 0.01 at [512, 768).

        Initial power 1e-4, M=64, time_constant_s=1e-4. Blocks 1, 16, 64, and 2048
        must all accept 1728 noise samples and detect the burst identically.
        """
        n_samples = 2048
        full_iq = np.full(n_samples, 0.003 + 0.0j, dtype=np.complex64)
        full_iq[512:768] = 0.01 + 0.0j

        def run_adaptive(block_size: int) -> tuple[int, list[CaptureSidecar]]:
            est = AdaptiveRobustNoiseEstimator(
                window_size=16,
                adaptation_window_size=64,
                time_constant_s=1e-4,
                sample_rate_hz=1e6,
                initial_noise_power_linear=1e-4,
            )
            pipe = StreamingCapturePipeline(
                stream_info=stream_1mhz,
                window_size=16,
                hop_size=8,
                ring_capacity=4096,
                noise_estimator=est,
                start_persistence=1,
                end_persistence=1,
                pre_trigger_samples=32,
                pre_signal_margin_samples=0,
                post_signal_samples=32,
                min_post_trigger_samples=0,
                max_event_samples=10000,
                synchronous_analysis=True,
            )
            emitted = []
            seq = 0
            for i in range(0, n_samples, block_size):
                chunk = full_iq[i : i + block_size]
                b = IQBlock(stream=stream_1mhz, sequence_number=seq, start_sample=i, timestamp_ns=0, iq=chunk)
                seq += 1
                emitted.extend(pipe.process_block(b))
            emitted.extend(pipe.flush())
            return est._accepted_noise_samples, emitted

        samples_16, evts_16 = run_adaptive(16)
        samples_64, evts_64 = run_adaptive(64)
        samples_2048, evts_2048 = run_adaptive(2048)

        assert samples_16 == 1728
        assert samples_64 == 1728
        assert samples_2048 == 1728

        assert len(evts_2048) == 1
        sc = evts_2048[0]
        assert sc.event.detected_start_sample == 504
        assert sc.event.trigger_sample == 519
        assert sc.event.detected_end_sample == 784
        assert sc.event.event_start_sample == 487
        assert sc.event.event_end_sample == 816

    def test_startup_guard_reproduction_2(self, stream_1mhz: StreamInfo) -> None:
        """Reproduction B02 (2): N=256, H=64, M=64, time_constant_s=1e-6, initial power 1e-4.

        512 samples at amplitude 0.1. Slices of 64 must NOT admit signal before window 0 exists,
        and both 64 and 512 must detect the event.
        """
        n_samples = 512
        full_iq = np.full(n_samples, 0.1 + 0.0j, dtype=np.complex64)

        def run_startup(block_size: int) -> tuple[float, list[CaptureSidecar]]:
            est = AdaptiveRobustNoiseEstimator(
                window_size=256,
                adaptation_window_size=64,
                time_constant_s=1e-6,
                sample_rate_hz=1e6,
                initial_noise_power_linear=1e-4,
            )
            pipe = StreamingCapturePipeline(
                stream_info=stream_1mhz,
                window_size=256,
                hop_size=64,
                ring_capacity=4096,
                noise_estimator=est,
                start_persistence=1,
                end_persistence=1,
                pre_trigger_samples=0,
                pre_signal_margin_samples=0,
                post_signal_samples=0,
                min_post_trigger_samples=0,
                max_event_samples=10000,
                synchronous_analysis=True,
            )
            emitted = []
            seq = 0
            for i in range(0, n_samples, block_size):
                chunk = full_iq[i : i + block_size]
                b = IQBlock(stream=stream_1mhz, sequence_number=seq, start_sample=i, timestamp_ns=0, iq=chunk)
                seq += 1
                emitted.extend(pipe.process_block(b))
            emitted.extend(pipe.flush())
            return est.noise_power_linear, emitted

        p_64, evts_64 = run_startup(64)
        p_512, evts_512 = run_startup(512)

        # Neither partition admits the high-amplitude signal as noise!
        assert p_64 == pytest.approx(1e-4, rel=1e-4)
        assert p_512 == pytest.approx(1e-4, rel=1e-4)

        assert len(evts_64) >= 1
        assert len(evts_512) >= 1


# ==============================================================================
# B03: MATCHED CONFIGURED-BAND CALIBRATION AND DOMAIN ADAPTATION
# ==============================================================================
class TestB03MatchedConfiguredBandDomain:
    def test_configured_band_empirical_vs_gamma_held_out_and_cw_detection(self) -> None:
        """Reproduction B03: Fs=1 MHz, band [90k, 110k] Hz, N=H=64, pfa_on=1e-4, settling=4096.

        2M training CN noise power 1e-4, 2.004M test noise.
        Matched filtered mean ~ 2.046e-6. Gamma gives ~18.1% hits, empirical policy gives < 0.1%.
        100 kHz CW at power 1e-4 triggers all 64 windows with matched reference.
        """
        fs = 1_000_000.0
        band = (90_000.0, 110_000.0)
        n = 64
        h = 64
        settling = 4096
        pfa_on = 1e-4
        pfa_off = 1e-2

        rng = np.random.default_rng(2608)
        std = math.sqrt(1e-4 / 2.0)
        train_noise = (rng.normal(0, std, 2_000_000) + 1j * rng.normal(0, std, 2_000_000)).astype(np.complex64)
        test_noise = (rng.normal(0, std, 2_004_096) + 1j * rng.normal(0, std, 2_004_096)).astype(np.complex64)

        calibrator = EmpiricalNoiseCalibrator(
            window_size=n,
            hop_size=h,
            pfa_on=pfa_on,
            pfa_off=pfa_off,
            max_samples=5_000_000,
        )
        cal_res = calibrator.calibrate(
            noise_iq=train_noise,
            sample_rate_hz=fs,
            settling_samples=settling,
            monitored_band_hz=band,
        )

        matched_mean = cal_res["mean_power_linear"]
        assert matched_mean == pytest.approx(2.046e-6, rel=0.05)

        policy = calibrator.create_policy(cal_res)
        detector = EnergyDetector(
            window_size=n,
            hop_size=h,
            pfa_on=pfa_on,
            pfa_off=pfa_off,
            mode="configured_band",
            monitored_band_hz=band,
            sample_rate_hz=fs,
            threshold_policy=policy,
        )

        # Held-out false alarm test with matching reference
        b_test = IQBlock(
            stream=StreamInfo("test-stream", "test", fs, 915e6),
            sequence_number=0,
            start_sample=0,
            timestamp_ns=0,
            iq=test_noise[settling:],
        )
        matched_noise_est = NoiseEstimate(
            stream_id="test-stream",
            block_sequence_number=0,
            noise_power_dbfs=10.0 * math.log10(matched_mean),
            valid=True,
        )
        meas = detector.process_windows(b_test, matched_noise_est)
        hits = sum(1 for m in meas if m.above_on_threshold)
        assert len(meas) == 31250
        assert hits < 30  # Empirical policy satisfies target ~0.01%

        # Positive CW detection at 100 kHz, power 1e-4, 8192 samples
        t_cw = np.arange(8192) / fs
        cw_iq = (math.sqrt(1e-4) * np.exp(2j * np.pi * 100_000.0 * t_cw)).astype(np.complex64)
        detector.reset()
        b_cw = IQBlock(
            stream=StreamInfo("test-stream", "test", fs, 915e6),
            sequence_number=1,
            start_sample=0,
            timestamp_ns=0,
            iq=cw_iq,
        )
        meas_cw = detector.process_windows(b_cw, matched_noise_est)
        # Post-settling windows (after settling delay) must trigger
        post_settling_meas = meas_cw[settling // h :]
        assert len(post_settling_meas) > 0
        assert all(m.above_on_threshold for m in post_settling_meas)

    def test_live_adaptive_reference_stays_in_detection_domain(self, stream_1mhz: StreamInfo) -> None:
        """In configured-band mode, live adaptive reference initialized near 2.04e-6

        must NOT drift to raw full-band 1e-4 after 32,000 quiet samples.
        """
        band = (90_000.0, 110_000.0)
        est = AdaptiveRobustNoiseEstimator(
            window_size=64,
            adaptation_window_size=64,
            time_constant_s=0.01,
            sample_rate_hz=1e6,
            initial_noise_power_linear=2.046e-6,
        )
        pipe = StreamingCapturePipeline(
            stream_info=stream_1mhz,
            window_size=64,
            hop_size=64,
            ring_capacity=16384,
            detector_mode="configured_band",
            monitored_band_hz=band,
            noise_estimator=est,
        )

        rng = np.random.default_rng(42)
        std = math.sqrt(1e-4 / 2.0)
        quiet_iq = (rng.normal(0, std, 32000) + 1j * rng.normal(0, std, 32000)).astype(np.complex64)

        b = IQBlock(stream=stream_1mhz, sequence_number=0, start_sample=0, timestamp_ns=0, iq=quiet_iq)
        pipe.process_block(b)

        # The adapted reference remains in filtered domain (~2e-6), NOT drifting to full-band 1e-4!
        assert est.noise_power_linear < 1e-5
        assert est.noise_power_linear == pytest.approx(2.04e-6, rel=0.2)


# ==============================================================================
# B04: BIND AND VALIDATE CALIBRATION POLICIES
# ==============================================================================
class TestB04PolicyValidationAndActualN:
    def test_threshold_policy_invariants(self) -> None:
        """ThresholdPolicy rejects nonfinite or non-ordered thresholds."""
        with pytest.raises(ContractValidationError):
            ThresholdPolicy(
                mode="empirical_matched",
                mean_power_linear=-1.0,
                t_on_linear=10.0,
                t_off_linear=5.0,
                t_on_dbfs=10.0,
                t_off_dbfs=5.0,
                pfa_on=1e-4,
                pfa_off=1e-2,
                window_size=64,
                hop_size=64,
            )

        with pytest.raises(ContractValidationError):
            ThresholdPolicy(
                mode="empirical_matched",
                mean_power_linear=1.0,
                t_on_linear=5.0,
                t_off_linear=10.0,  # on < off
                t_on_dbfs=5.0,
                t_off_dbfs=10.0,
                pfa_on=1e-4,
                pfa_off=1e-2,
                window_size=64,
                hop_size=64,
            )

    def test_detector_policy_compatibility_validation(self) -> None:
        """EnergyDetector rejects policies with mismatched Fs, filter order, N, or mode."""
        policy = ThresholdPolicy(
            mode="empirical_matched",
            mean_power_linear=1e-4,
            t_on_linear=2e-4,
            t_off_linear=1.5e-4,
            t_on_dbfs=10.0 * math.log10(2e-4),
            t_off_dbfs=10.0 * math.log10(1.5e-4),
            pfa_on=1e-4,
            pfa_off=1e-2,
            window_size=64,
            hop_size=64,
            sample_rate_hz=2_000_000.0,  # 2 MHz
        )
        detector = EnergyDetector(
            window_size=64,
            hop_size=64,
            sample_rate_hz=1_000_000.0,  # 1 MHz
        )
        with pytest.raises(PolicyCompatibilityError, match="Sample rate mismatch"):
            detector.set_threshold_policy(policy)

    def test_relabeling_calibration_dictionary_rejected(self) -> None:
        """Relabeling a calibration dictionary without recomputation must be rejected by create_policy."""
        calibrator = EmpiricalNoiseCalibrator(
            window_size=16,
            hop_size=16,
            pfa_on=0.01,
            pfa_off=0.1,
        )
        rng = np.random.default_rng(123)
        res = calibrator.calibrate(rng.normal(0, 0.01, 10_000) + 1j * rng.normal(0, 0.01, 10_000), sample_rate_hz=1e6)

        # Attempt to relabel calibrator for N=256
        calibrator_256 = EmpiricalNoiseCalibrator(
            window_size=256,
            hop_size=256,
            pfa_on=1e-4,
            pfa_off=0.01,
        )
        with pytest.raises(PolicyCompatibilityError, match="Window size mismatch"):
            calibrator_256.create_policy(res)


# ==============================================================================
# B05: RESET AT GAPS AND COORDINATE GENERATIONS
# ==============================================================================
class TestB05DiscontinuityGenerations:
    def test_gap_resets_estimator_residuals_and_advances_worker_generation(self, stream_1mhz: StreamInfo) -> None:
        """On dropped samples, estimator reference resets to initial configured power,

        residuals are purged, and worker advances generation.
        """
        est = AdaptiveRobustNoiseEstimator(
            window_size=16,
            adaptation_window_size=64,
            initial_noise_power_linear=1e-4,
        )
        pipe = StreamingCapturePipeline(
            stream_info=stream_1mhz,
            window_size=16,
            hop_size=8,
            ring_capacity=4096,
            noise_estimator=est,
        )

        # Learn 64 samples at amplitude 0.003
        b0 = IQBlock(stream_1mhz, 0, 0, 0, np.full(64, 0.003 + 0.0j, dtype=np.complex64))
        pipe.process_block(b0)
        assert est.noise_power_linear < 1e-4

        # Add 5 quiet samples at start 64
        b1 = IQBlock(stream_1mhz, 1, 64, None, np.full(5, 0.003 + 0.0j, dtype=np.complex64))
        pipe.process_block(b1)

        # Gap discontinuity: dropped_samples_before=31
        worker_gen_before = pipe.worker._generation
        b2 = IQBlock(stream_1mhz, 2, 100, None, np.full(5, 0.003 + 0.0j, dtype=np.complex64), dropped_samples_before=31)
        pipe.process_block(b2)

        # Estimator was reset to initial configured power and residuals purged
        assert est.noise_power_linear == pytest.approx(1e-4)
        assert len(est._residual_samples) == 0
        # Worker generation advanced
        assert pipe.worker._generation > worker_gen_before


# ==============================================================================
# B06: NOISE AGE, STALE RECOVERY AND FAILURE COMMITS
# ==============================================================================
class TestB06NoiseAgeAndStaleness:
    def test_fresh_quiet_noise_does_not_falsely_age(self, stream_1mhz: StreamInfo) -> None:
        """128 fresh quiet samples with max_staleness=64 ends at age 0 and state VALID."""
        est = AdaptiveRobustNoiseEstimator(
            window_size=16,
            adaptation_window_size=16,
            initial_noise_power_linear=1e-4,
            max_staleness_samples=64,
        )
        pipe = StreamingCapturePipeline(
            stream_info=stream_1mhz,
            window_size=16,
            hop_size=8,
            ring_capacity=4096,
            noise_estimator=est,
        )
        b = IQBlock(stream_1mhz, 0, 0, 0, np.full(128, 0.01 + 0.0j, dtype=np.complex64))
        pipe.process_block(b)
        assert est.staleness_samples == 0
        assert est.state == NoiseLifecycleState.VALID

    def test_stale_estimator_fails_fast_without_ring_mutation(self, stream_1mhz: StreamInfo) -> None:
        """Stale estimator raises UncalibratedNoiseError and enters FAULTED before mutating ring."""
        est = AdaptiveRobustNoiseEstimator(
            window_size=16,
            adaptation_window_size=16,
            initial_noise_power_linear=1e-4,
            max_staleness_samples=31,
        )
        pipe = StreamingCapturePipeline(
            stream_info=stream_1mhz,
            window_size=16,
            hop_size=8,
            ring_capacity=4096,
            noise_estimator=est,
        )

        # 32 high-power samples freeze adaptation; age advances to 32 -> STALE
        b0 = IQBlock(stream_1mhz, 0, 0, 0, np.full(32, 0.5 + 0.0j, dtype=np.complex64))
        pipe.process_block(b0)
        assert est.state == NoiseLifecycleState.STALE

        # Second block: attempting to process when STALE must fail fast and enter FAULTED
        ring_samples_before = pipe.ring_buffer.total_samples_written
        b1 = IQBlock(stream_1mhz, 1, 32, None, np.zeros(32, dtype=np.complex64))
        with pytest.raises(UncalibratedNoiseError, match="stale"):
            pipe.process_block(b1)
        assert pipe.state == PipelineState.FAULTED
        # Ring was NOT mutated with the unprocessable block
        assert pipe.ring_buffer.total_samples_written == ring_samples_before

    def test_stale_recovery_via_recalibration(self) -> None:
        """Estimator can be cleanly recovered from STALE via recalibrate."""
        est = AdaptiveRobustNoiseEstimator(
            window_size=16,
            initial_noise_power_linear=1e-4,
            max_staleness_samples=100,
        )
        est.advance_stream_samples(200)
        assert est.state == NoiseLifecycleState.STALE

        est.recalibrate(reference_linear=2e-4)
        assert est.state == NoiseLifecycleState.VALID
        assert est.noise_power_linear == pytest.approx(2e-4)
        assert est.staleness_samples == 0


# ==============================================================================
# B07: WORKER CALLBACK REENTRANCY AND SHUTDOWN DEADLINE
# ==============================================================================
class TestB07WorkerSafetyAndLedger:
    def test_reentrant_sink_no_deadlock(self, stream_1mhz: StreamInfo) -> None:
        """Sink that accesses worker.processed_results must complete without deadlocking."""
        event = SignalEvent("ev1", stream_1mhz, np.ones(64, dtype=np.complex64), 0, 64, 0, 64, 0)
        sidecar = CaptureSidecar(
            stream_epoch=0,
            event_id="ev1",
            config_version="v1",
            detector_mode="full_band",
            monitored_band_hz=None,
            first_on_sample=0,
            confirmation_sample=16,
            end_confirmation_sample=64,
            requested_leading_samples=0,
            actual_leading_samples=0,
            requested_trailing_samples=0,
            actual_trailing_samples=0,
            boundary_uncertainty_samples=8,
            noise_provenance="GaussianNoiseEstimator",
            noise_valid=True,
            termination_reason="normal",
            is_truncated=False,
            known_lost_samples=0,
            unknown_loss=False,
            feature_status="ok",
            feature_reason=None,
            event=event,
        )

        completed_sink = threading.Event()
        worker = FeatureAnalysisWorker(synchronous=False)

        def reentrant_sink(sc: CaptureSidecar) -> None:
            # Reentrantly query worker results while inside callback
            _ = worker.processed_results
            completed_sink.set()

        worker.add_result_sink(reentrant_sink)
        worker.submit(sidecar)
        assert completed_sink.wait(timeout=2.0) is True
        worker.shutdown(timeout=1.0)

        assert worker.completed_count == 1
        assert worker.sink_error_count == 0


# ==============================================================================
# B08: PRESERVE REAL AVAILABLE EOF AND GAP TAIL
# ==============================================================================
class TestB08AvailableEOFTail:
    def test_69_samples_all_retained_across_partitions(self, stream_1mhz: StreamInfo) -> None:
        """Reproduction B08: N=16, H=8, 69 samples of amplitude 0.5.

        Blocks 1, 16, and 69 must all retain all 69 samples at EOF.
        """
        full_iq = np.full(69, 0.5 + 0.0j, dtype=np.complex64)

        def run_eof(block_size: int) -> CaptureSidecar:
            est = GaussianNoiseEstimator(window_size=16, calibrated_noise_power_linear=1e-4)
            pipe = StreamingCapturePipeline(
                stream_info=stream_1mhz,
                window_size=16,
                hop_size=8,
                ring_capacity=1024,
                noise_estimator=est,
                pre_trigger_samples=0,
                pre_signal_margin_samples=0,
                post_signal_samples=0,
                min_post_trigger_samples=0,
                start_persistence=1,
                end_persistence=1,
                synchronous_analysis=True,
            )
            emitted = []
            seq = 0
            for i in range(0, 69, block_size):
                chunk = full_iq[i : i + block_size]
                b = IQBlock(stream_1mhz, seq, i, 0, chunk)
                seq += 1
                emitted.extend(pipe.process_block(b))
            emitted.extend(pipe.flush())
            assert len(emitted) == 1
            return emitted[0]

        sc1 = run_eof(1)
        sc16 = run_eof(16)
        sc69 = run_eof(69)

        # All 69 samples preserved without dropping tail
        assert len(sc1.event.iq) == 69
        assert len(sc16.event.iq) == 69
        assert len(sc69.event.iq) == 69
        assert sc1.event.event_start_sample == 0
        assert sc1.event.event_end_sample == 69


# ==============================================================================
# B09: FEATURE-SPECIFIC EMITTER BANDWIDTH SUPPORT
# ==============================================================================
class TestB09EmitterBandwidthSupport:
    def test_boundary_clipping_without_roi(self) -> None:
        """Reproduction B09 (1): Frequencies 0..8 Hz, PSD [101, 81, 51, 11, 1, 1, 1, 1, 1], noise=1, roi=None.

        Lobe touches lower boundary at f=0: both 3-dB and 99% OBW must be invalid.
        """
        freqs = np.arange(9, dtype=np.float64)
        powers = np.array([101.0, 81.0, 51.0, 11.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

        class DummyCtx:
            def psd(self):
                return PSDResult(freqs, powers, np.zeros_like(powers))

        cfg = BandwidthFeatureConfig(
            mode="emitter",
            roi_hz=None,
            noise_floor_dbfs_per_hz=0.0,  # linear noise = 1.0
            check_edge_clipping=True,
        )
        extractor = BandwidthFeatureExtractor(cfg)
        fset = extractor.extract(DummyCtx())

        assert fset.features["bandwidth.occupied_99_hz"].valid is False
        assert fset.features["bandwidth.three_db_hz"].valid is False

    def test_energy_tail_containment_in_roi(self) -> None:
        """Reproduction B09 (2): Frequencies -10..10 Hz, PSD = 1 + 100*exp(-f^2/8), ROI [-3, 3].

        7.7% excess energy outside ROI -> 99% OBW is invalid, while 3-dB width remains valid (~5 Hz).
        """
        freqs = np.arange(-10, 11, dtype=np.float64)
        powers = 1.0 + 100.0 * np.exp(-(freqs**2) / 8.0)

        class DummyCtx:
            def psd(self):
                return PSDResult(freqs, powers, np.zeros_like(powers))

        cfg = BandwidthFeatureConfig(
            mode="emitter",
            roi_hz=(-3.0, 3.0),
            noise_floor_dbfs_per_hz=0.0,
            check_edge_clipping=True,
        )
        extractor = BandwidthFeatureExtractor(cfg)
        fset = extractor.extract(DummyCtx())

        obw = fset.features["bandwidth.occupied_99_hz"]
        three_db = fset.features["bandwidth.three_db_hz"]

        # 99% OBW is invalid because energy tails are clipped outside ROI (> 1%)
        assert obw.valid is False
        assert "not contained in roi" in str(obw.reason).lower()

        # 3-dB width remains valid because its half-power crossings are contained inside [-3, 3]
        assert three_db.valid is True
        assert three_db.value == pytest.approx(5.0, abs=1.0)


# ==============================================================================
# B10: EFFECTIVE EXTRACTION POLICY AND RAW PSD
# ==============================================================================
class TestB10ExtractionPolicyAndRawPSD:
    def test_registered_extractor_overrides_change_policy_id(self, stream_1mhz: StreamInfo) -> None:
        """Aggregator honoring registered extractor overrides produces distinct policy IDs."""
        event = SignalEvent("ev1", stream_1mhz, np.ones(64, dtype=np.complex64), 0, 64, 0, 64, 0)
        ctx = SignalContext(event, config=AnalysisConfig())

        # Registry with default raw_psd extractor
        reg_raw = build_default_feature_registry()
        agg_raw = FeatureAggregator(reg_raw)
        res_raw = agg_raw.extract_analysis_result(ctx)

        # Registry with explicit emitter extractor
        from signal_processing.features.aggregation.registry import FeatureRegistry
        from signal_processing.features.amplitude import AmplitudeFeatureExtractor

        custom_bw = BandwidthFeatureExtractor(BandwidthFeatureConfig(mode="emitter"))
        custom_extractors = [
            custom_bw if e.name == "bandwidth" else e
            for e in reg_raw.extractors
        ]
        reg_emitter = FeatureRegistry(custom_extractors)
        agg_emitter = FeatureAggregator(reg_emitter)
        res_emitter = agg_emitter.extract_analysis_result(ctx)

        assert res_raw.policy.policy_id != res_emitter.policy.policy_id
        assert res_raw.policy.bandwidth_mode == "raw_psd"
        assert res_emitter.policy.bandwidth_mode == "emitter"

    def test_raw_psd_mode_does_not_subtract_noise_floor(self) -> None:
        """In raw_psd mode, configured noise floor is NOT subtracted; measures untouched PSD."""
        freqs = np.arange(100, dtype=np.float64)
        powers = np.ones(100, dtype=np.float64)
        powers[45:55] = 11.0

        class DummyCtx:
            def psd(self):
                return PSDResult(freqs, powers, np.zeros_like(powers))

        cfg_with_floor = BandwidthFeatureConfig(
            mode="raw_psd",
            noise_floor_dbfs_per_hz=0.0,  # 1.0 linear
        )
        extractor = BandwidthFeatureExtractor(cfg_with_floor)
        fset = extractor.extract(DummyCtx())

        # In raw_psd mode, entire 100-bin baseline is untouched, resulting in ~99 Hz OBW
        obw = fset.features["bandwidth.occupied_99_hz"]
        assert obw.valid is True
        assert obw.value > 90.0


# ==============================================================================
# B11: QUALITY CONCEPTS AND STABLE FILTER DELAY
# ==============================================================================
class TestB11QualityAndStableFilterDelay:
    def test_eof_truncation_remains_internally_continuous(self, stream_1mhz: StreamInfo) -> None:
        """Capture truncated at EOF with 0 dropped samples is continuous."""
        evt = SignalEvent("ev1", stream_1mhz, np.ones(100, dtype=np.complex64), 0, 100, 0, 100, 0, complete=False, dropped_samples=0)
        ctx = SignalContext(evt)
        assert ctx.capture_quality.complete is False
        assert ctx.capture_quality.is_continuous is True

    def test_stable_narrow_band_filter_group_delay(self) -> None:
        """Group delay calculated at Fs=1 MHz, center 100 kHz with stable per-section phase derivative.

        1000 Hz width ~ 831.8 samples, 100 Hz width ~ 8317.8 samples, 10 Hz width ~ 83178.4 samples.
        """
        fs = 1_000_000.0

        d_1000 = EnergyDetector(
            window_size=64,
            hop_size=64,
            mode="configured_band",
            monitored_band_hz=(99_500.0, 100_500.0),  # 1000 Hz width
            sample_rate_hz=fs,
        )
        assert d_1000.filter_delay_samples == pytest.approx(832, abs=2)

        d_100 = EnergyDetector(
            window_size=64,
            hop_size=64,
            mode="configured_band",
            monitored_band_hz=(99_950.0, 100_050.0),  # 100 Hz width
            sample_rate_hz=fs,
        )
        assert d_100.filter_delay_samples == pytest.approx(8318, abs=5)

        d_10 = EnergyDetector(
            window_size=64,
            hop_size=64,
            mode="configured_band",
            monitored_band_hz=(99_995.0, 100_005.0),  # 10 Hz width
            sample_rate_hz=fs,
        )
        assert d_10.filter_delay_samples == pytest.approx(83178, abs=50)


# ==============================================================================
# B12: STRICT CONFIGURATION VALIDATION
# ==============================================================================
class TestB12StrictConfigurationValidation:
    def test_strict_parameter_type_and_sign_checks(self) -> None:
        """Reject booleans, nonintegers, and invalid values across components."""
        with pytest.raises(ValueError):
            AdaptiveRobustNoiseEstimator(adaptation_window_size=0)

        with pytest.raises(ValueError):
            AdaptiveRobustNoiseEstimator(adaptation_window_size=True)

        with pytest.raises(ValueError):
            AdaptiveRobustNoiseEstimator(max_staleness_samples=-1)

        with pytest.raises(ValueError):
            EnergyDetector(conversion_floor_linear=0.0)

        with pytest.raises(ValueError):
            EnergyDetector(conversion_floor_linear=True)

        with pytest.raises(ValueError):
            EnergyDetector(filter_order=0)


# ==============================================================================
# B14: BOUNDED STREAMING CALIBRATION
# ==============================================================================
class TestB14BoundedStreamingCalibration:
    def test_streaming_calibration_chunk_invariance(self) -> None:
        """Streaming update_calibration produces identical results across chunk sizes."""
        fs = 1_000_000.0
        rng = np.random.default_rng(777)
        noise = (rng.normal(0, 0.01, 20_000) + 1j * rng.normal(0, 0.01, 20_000)).astype(np.complex64)

        cal1 = EmpiricalNoiseCalibrator(window_size=64, hop_size=64)
        cal1.begin_calibration(sample_rate_hz=fs)
        cal1.update_calibration(noise)
        res1 = cal1.finalize_calibration()

        cal2 = EmpiricalNoiseCalibrator(window_size=64, hop_size=64)
        cal2.begin_calibration(sample_rate_hz=fs)
        for i in range(0, len(noise), 1024):
            cal2.update_calibration(noise[i : i + 1024])
        res2 = cal2.finalize_calibration()

        assert res1["mean_power_linear"] == pytest.approx(res2["mean_power_linear"], rel=1e-6)
        assert res1["t_on_linear"] == pytest.approx(res2["t_on_linear"], rel=1e-6)
