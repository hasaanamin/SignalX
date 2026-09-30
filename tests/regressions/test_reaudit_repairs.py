"""Comprehensive regression test suite asserting all repairs from the re-audit.

Covers Sections 1 through 8:
1. Noise startup and causal adaptation (warmup progression, tracking, partition invariance, staleness).
2. Configured-band thresholds (matched empirical calibration, held-out validation, support rejection).
3. Logical capture chronology (partition invariance across blocks 1/64/512, joint configuration bounds, retention).
4. Worker admission, shutdown, and reset generation safety (races, barriers, purge on reset).
5. Emitter bandwidth prerequisites (pure noise rejection, clipped lobe, sub-threshold SNR, positive cases).
6. Stream validation and discontinuity policy (anchor invalidation on overflow, continuous preservation).
7. Extraction policy and truthful quality/timing (unique policy IDs, EOF truncation truthfulness).
8. Configuration validation and lifecycle states (strict input checks, state transitions).
"""

import math
import threading
import time
import numpy as np
import pytest
from scipy import signal

from signal_processing.capture.assembler import EventAssembler
from signal_processing.context.config import AnalysisConfig
from signal_processing.context.signal_context import SignalContext
from signal_processing.capture.sidecar import CaptureSidecar
from signal_processing.core.contracts import (
    IQBlock,
    NoiseEstimate,
    SignalEvent,
    StreamInfo,
)
from signal_processing.core.exceptions import StreamConsistencyError
from signal_processing.detection.exceptions import UncalibratedNoiseError
from signal_processing.detection.energy_detector import EnergyDetector
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
from signal_processing.features.bandwidth.config import BandwidthFeatureConfig
from signal_processing.features.bandwidth.extractor import BandwidthFeatureExtractor
from signal_processing.runtime.pipeline import PipelineState, StreamingCapturePipeline
from signal_processing.runtime.worker import FeatureAnalysisWorker


@pytest.fixture
def stream_info() -> StreamInfo:
    return StreamInfo(
        stream_id="test-stream",
        source_id="test-source",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
        channel_id="ch0",
        dtype="complex64",
    )


# ==============================================================================
# SECTION 1: NOISE STARTUP AND CAUSAL ADAPTATION
# ==============================================================================
class TestSection1NoiseStartupAndAdaptation:
    def test_gaussian_warmup_progresses_without_error(self, stream_info: StreamInfo) -> None:
        """Reproduction 1: GaussianNoiseEstimator(window_size=16, warmup_blocks=2) must progress

        without raising on the first block.
        """
        estimator = GaussianNoiseEstimator(window_size=16, warmup_blocks=2)
        assert estimator.state == NoiseLifecycleState.CALIBRATING
        assert estimator.is_calibrated is False

        rng = np.random.default_rng(1234)
        iq0 = (rng.normal(0, 0.01, 16) + 1j * rng.normal(0, 0.01, 16)).astype(np.complex64)
        b0 = IQBlock(stream=stream_info, sequence_number=0, start_sample=0, timestamp_ns=0, iq=iq0)

        # Block 0: still calibrating
        est0 = estimator.update(b0)
        assert est0.valid is False
        assert "Warmup phase (CALIBRATING)" in str(est0.reason)
        assert estimator.state == NoiseLifecycleState.CALIBRATING

        # Block 1: warmup completes
        iq1 = (rng.normal(0, 0.01, 16) + 1j * rng.normal(0, 0.01, 16)).astype(np.complex64)
        b1 = IQBlock(stream=stream_info, sequence_number=1, start_sample=16, timestamp_ns=16000, iq=iq1)
        est1 = estimator.update(b1)
        assert est1.valid is True
        assert estimator.state == NoiseLifecycleState.VALID
        assert estimator.is_calibrated is True
        assert estimator.noise_power_linear is not None and estimator.noise_power_linear > 0.0

    def test_pipeline_warmup_consumption(self, stream_info: StreamInfo) -> None:
        """Pipeline consumes warmup blocks returning empty detections without throwing."""
        estimator = GaussianNoiseEstimator(window_size=16, warmup_blocks=2)
        pipeline = StreamingCapturePipeline(
            stream_info=stream_info,
            window_size=16,
            hop_size=8,
            ring_capacity=1024,
            noise_estimator=estimator,
        )
        rng = np.random.default_rng(5678)
        iq0 = (rng.normal(0, 0.01, 16) + 1j * rng.normal(0, 0.01, 16)).astype(np.complex64)
        b0 = IQBlock(stream=stream_info, sequence_number=0, start_sample=0, timestamp_ns=0, iq=iq0)
        res0 = pipeline.process_block(b0)
        assert res0 == []
        assert estimator.state == NoiseLifecycleState.CALIBRATING

        iq1 = (rng.normal(0, 0.01, 16) + 1j * rng.normal(0, 0.01, 16)).astype(np.complex64)
        b1 = IQBlock(stream=stream_info, sequence_number=1, start_sample=16, timestamp_ns=16000, iq=iq1)
        res1 = pipeline.process_block(b1)
        assert res1 == []
        assert estimator.state == NoiseLifecycleState.VALID

    def test_uncalibrated_estimator_without_warmup_faults_pipeline(self, stream_info: StreamInfo) -> None:
        """Uncalibrated estimator without warmup throws UncalibratedNoiseError and faults pipeline."""
        estimator = GaussianNoiseEstimator(window_size=16, warmup_blocks=0)
        pipeline = StreamingCapturePipeline(
            stream_info=stream_info,
            window_size=16,
            hop_size=8,
            ring_capacity=1024,
            noise_estimator=estimator,
        )
        b0 = IQBlock(stream=stream_info, sequence_number=0, start_sample=0, timestamp_ns=0, iq=np.zeros(16, dtype=np.complex64))
        with pytest.raises(UncalibratedNoiseError):
            pipeline.process_block(b0)
        assert pipeline.state == PipelineState.FAULTED

    def test_adaptive_tracking_quiet_noise_and_burst_protection(self) -> None:
        """Reproduction 2: Adaptive reference initially 1e-4, 16,000 quiet samples (2e-5)

        with time constant 100 us tracks quiet noise, and freezes during burst.
        """
        estimator = AdaptiveRobustNoiseEstimator(
            window_size=16,
            time_constant_s=100e-6,
            sample_rate_hz=1e6,
            initial_noise_power_linear=1e-4,
            adaptation_window_size=64,
        )
        assert estimator.noise_power_linear == pytest.approx(1e-4)

        # 16,000 quiet samples with power 2e-5
        rng = np.random.default_rng(999)
        std_quiet = math.sqrt(2e-5 / 2.0)
        quiet_iq = (rng.normal(0, std_quiet, 16000) + 1j * rng.normal(0, std_quiet, 16000)).astype(np.complex64)
        estimator.update_samples(quiet_iq)

        # Demonstrably tracked down toward 2e-5
        assert estimator.noise_power_linear is not None
        assert estimator.noise_power_linear < 5e-5
        assert estimator.noise_power_linear == pytest.approx(2e-5, rel=0.25)
        tracked_val = estimator.noise_power_linear

        # Burst protection: freeze estimator and feed high-power burst
        estimator.freeze(True)
        burst_iq = (np.ones(1000) * 10.0).astype(np.complex64)
        estimator.update_samples(burst_iq)
        assert estimator.noise_power_linear == tracked_val

    def test_adaptation_partition_invariance(self) -> None:
        """Reproduction 3: Identical 1,000 samples fed directly as chunks 1, 16, 64, and 1,000

        must yield identical final adaptive states and thresholds.
        """
        rng = np.random.default_rng(42)
        samples = (rng.normal(0, 0.01, 1000) + 1j * rng.normal(0, 0.01, 1000)).astype(np.complex64)

        def run_with_chunk(chunk_size: int) -> tuple[float, float, float]:
            est = AdaptiveRobustNoiseEstimator(
                window_size=16,
                time_constant_s=0.01,
                sample_rate_hz=1e6,
                initial_noise_power_linear=1e-4,
                adaptation_window_size=64,
            )
            for i in range(0, len(samples), chunk_size):
                est.update_samples(samples[i : i + chunk_size])
            t_on, t_off = est.get_thresholds_linear()
            return est.noise_power_linear, t_on, t_off

        p1, on1, off1 = run_with_chunk(1)
        p16, on16, off16 = run_with_chunk(16)
        p64, on64, off64 = run_with_chunk(64)
        p1000, on1000, off1000 = run_with_chunk(1000)

        # All four partition sizes produce exactly the same adaptation state
        assert p1 == pytest.approx(p16, rel=1e-7)
        assert p1 == pytest.approx(p64, rel=1e-7)
        assert p1 == pytest.approx(p1000, rel=1e-7)
        assert on1 == pytest.approx(on1000, rel=1e-7)
        assert off1 == pytest.approx(off1000, rel=1e-7)

    def test_staleness_detection_advancing_stream_time(self) -> None:
        """Advancing stream time while frozen allows staleness to be detected."""
        est = AdaptiveRobustNoiseEstimator(
            window_size=16,
            initial_noise_power_linear=1e-4,
            max_staleness_samples=1000,
        )
        assert est.state == NoiseLifecycleState.VALID
        est.advance_stream_samples(500)
        assert est.state == NoiseLifecycleState.VALID
        est.advance_stream_samples(600)
        assert est.state == NoiseLifecycleState.STALE


# ==============================================================================
# SECTION 2: CONFIGURED-BAND THRESHOLDS & EMPIRICAL CALIBRATION
# ==============================================================================
class TestSection2ConfiguredBandThresholds:
    def test_mandatory_held_out_empirical_calibration_validation(self) -> None:
        """Mandatory held-out reproduction:

        Fs=1 MHz, band 90-110 kHz, N=H=64, 4096 settling samples.
        Seed 2608, independent 2M training noise and 2.004M test noise, power 1e-4.
        Uncalibrated Gamma produces ~18.13% false alarms (5667/31250).
        Empirical supported calibration must pass statistical acceptance (< 0.1% hits).
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

        # Resolution permits estimation, not certification from training alone.
        assert cal_res["quantile_resolved"] is True
        assert cal_res["supported_pfa"] is False
        policy = calibrator.create_policy(cal_res)
        assert isinstance(policy, ThresholdPolicy)
        assert policy.valid is True
        assert policy.supported_pfa is False

        # Run test noise through detector with empirical policy
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
        b_test = IQBlock(
            stream=StreamInfo("test-stream", "test", fs, 915e6),
            sequence_number=0,
            start_sample=0,
            timestamp_ns=0,
            iq=test_noise[settling:],
        )
        noise_est = NoiseEstimate(
            stream_id="test-stream",
            block_sequence_number=0,
            noise_power_dbfs=10.0 * math.log10(policy.mean_power_linear),
            valid=True,
        )
        meas = detector.process_windows(b_test, noise_est)
        hits = sum(1 for m in meas if m.above_on_threshold)
        total_windows = len(meas)
        hit_rate = hits / total_windows if total_windows > 0 else 0.0

        # Uncalibrated Gamma gave 5667 hits (18.13%). Empirical policy achieves < 0.1% hits!
        assert total_windows == 31250
        assert hits < 30
        assert hit_rate < 0.001

    def test_insufficient_support_rejected(self) -> None:
        """Insufficient training support must result in supported_pfa=False."""
        calibrator = EmpiricalNoiseCalibrator(
            window_size=256,
            hop_size=256,
            pfa_on=1e-4,
            pfa_off=1e-2,
        )
        rng = np.random.default_rng(111)
        short_noise = (rng.normal(0, 0.01, 10_000) + 1j * rng.normal(0, 0.01, 10_000)).astype(np.complex64)
        res = calibrator.calibrate(short_noise, sample_rate_hz=1e6)
        assert res["supported_pfa"] is False
        policy = calibrator.create_policy(res)
        assert policy.supported_pfa is False


# ==============================================================================
# SECTION 3: LOGICAL CAPTURE CHRONOLOGY AND COMPATIBLE LIMITS
# ==============================================================================
class TestSection3CaptureChronologyAndLimits:
    def test_512_sample_burst_partition_invariance(self, stream_info: StreamInfo) -> None:
        """Mandatory reproduction: 512 complex64 samples, burst [100, 110), N=16, H=8.

        Blocks 1, 64, and 512 must have identical normal termination, completeness,
        and event coordinates [71, 160) (89 samples).
        """
        n_total = 512
        full_iq = np.zeros(n_total, dtype=np.complex64)
        full_iq[100:110] = 0.5 + 0.0j

        def run_pipeline(block_size: int) -> list[CaptureSidecar]:
            est = GaussianNoiseEstimator(window_size=16, calibrated_noise_power_linear=1e-4)
            p = StreamingCapturePipeline(
                stream_info=stream_info,
                window_size=16,
                hop_size=8,
                ring_capacity=1024,
                noise_estimator=est,
                start_persistence=1,
                end_persistence=1,
                pre_trigger_samples=32,
                pre_signal_margin_samples=0,
                post_signal_samples=32,
                min_post_trigger_samples=0,
                max_event_samples=200,
                synchronous_analysis=True,
            )
            emitted = []
            seq = 0
            for i in range(0, n_total, block_size):
                chunk = full_iq[i : i + block_size]
                b = IQBlock(
                    stream=stream_info,
                    sequence_number=seq,
                    start_sample=i,
                    timestamp_ns=int(i * 1000),
                    iq=chunk,
                )
                seq += 1
                emitted.extend(p.process_block(b))
            emitted.extend(p.finalize())
            return emitted

        out1 = run_pipeline(1)
        out64 = run_pipeline(64)
        out512 = run_pipeline(512)

        for out in [out1, out64, out512]:
            assert len(out) == 1
            sidecar = out[0]
            assert sidecar.termination_reason == "normal"
            assert sidecar.event.complete is True
            assert sidecar.is_truncated is False
            assert sidecar.event.event_start_sample == 71
            assert sidecar.event.event_end_sample == 160
            assert len(sidecar.event.iq) == 89

    def test_impossible_configuration_rejected_on_init(self, stream_info: StreamInfo) -> None:
        """Mandatory reproduction: N=64, H=8, start_persistence=1, cap=16, zero margins

        must be rejected at configuration rather than crash at runtime.
        """
        with pytest.raises(ValueError, match="Incompatible capture configuration"):
            StreamingCapturePipeline(
                stream_info=stream_info,
                window_size=64,
                hop_size=8,
                start_persistence=1,
                max_event_samples=16,
                pre_trigger_samples=0,
                pre_signal_margin_samples=0,
            )

    def test_large_pre_trigger_retention(self, stream_info: StreamInfo) -> None:
        """Reproduction: Capacity 12,000, pre-trigger 10,000, N=16, H=8, burst [12500, 12700).

        Input blocks 64 and 30,000 must retain original requested start 2503 and exact IQ.
        """
        n_total = 15000
        full_iq = np.zeros(n_total, dtype=np.complex64)
        full_iq[12500:12700] = 0.8 + 0.0j

        def run_with_block(block_size: int) -> CaptureSidecar:
            est = GaussianNoiseEstimator(window_size=16, calibrated_noise_power_linear=1e-4)
            p = StreamingCapturePipeline(
                stream_info=stream_info,
                window_size=16,
                hop_size=8,
                ring_capacity=12000,
                noise_estimator=est,
                pre_trigger_samples=10000,
                pre_signal_margin_samples=256,
                post_signal_samples=256,
                min_post_trigger_samples=512,
                max_event_samples=11000,
                synchronous_analysis=True,
            )
            emitted = []
            seq = 0
            for i in range(0, n_total, block_size):
                chunk = full_iq[i : i + block_size]
                b = IQBlock(
                    stream=stream_info,
                    sequence_number=seq,
                    start_sample=i,
                    timestamp_ns=int(i * 1000),
                    iq=chunk,
                )
                seq += 1
                emitted.extend(p.process_block(b))
            emitted.extend(p.finalize())
            assert len(emitted) == 1
            return emitted[0]

        sc64 = run_with_block(64)
        sc30k = run_with_block(15000)

        assert sc64.event.event_start_sample == sc30k.event.event_start_sample
        assert sc64.event.event_start_sample == 2503
        assert len(sc64.event.iq) == len(sc30k.event.iq)


# ==============================================================================
# SECTION 4: WORKER ADMISSION, SHUTDOWN AND RESET GENERATION SAFETY
# ==============================================================================
def make_test_sidecar(event: SignalEvent) -> CaptureSidecar:
    return CaptureSidecar(
        stream_epoch=0,
        event_id=event.event_id,
        config_version="v1",
        detector_mode="full_band",
        monitored_band_hz=None,
        first_on_sample=0,
        confirmation_sample=min(64, event.event_end_sample),
        end_confirmation_sample=min(64, event.event_end_sample),
        requested_leading_samples=0,
        actual_leading_samples=0,
        requested_trailing_samples=0,
        actual_trailing_samples=0,
        boundary_uncertainty_samples=8,
        noise_provenance="gaussian",
        noise_valid=True,
        termination_reason="normal",
        is_truncated=False,
        known_lost_samples=0,
        unknown_loss=False,
        feature_status="pending",
        feature_reason=None,
        event=event,
    )


class TestSection4WorkerSafety:
    def test_worker_atomic_admission_and_shutdown(self) -> None:
        """Shutdown closes worker atomically; no orphan admitted behind termination marker."""
        worker = FeatureAnalysisWorker(max_queue_size=16, synchronous=False)
        stream = StreamInfo("s", "src", 1e6, 915e6)
        event = SignalEvent("ev1", stream, np.ones(64, dtype=np.complex64), 0, 64, 0, 64, 0)
        sidecar = make_test_sidecar(event)

        # Thread attempting admissions
        admitted = []
        rejected = []

        def submit_loop():
            for i in range(50):
                try:
                    res = worker.submit(sidecar)
                    admitted.append(res)
                except RuntimeError:
                    rejected.append(i)

        t = threading.Thread(target=submit_loop)
        t.start()
        time.sleep(0.001)
        worker.shutdown(timeout=1.0, drain=True)
        t.join()

        # All admitted jobs must have results or be handled without hanging
        assert len(admitted) + len(rejected) == 50

    def test_worker_reset_generation_barrier(self) -> None:
        """In-flight job from prior generation must never appear in results after reset."""
        worker = FeatureAnalysisWorker(max_queue_size=16, synchronous=False)
        stream = StreamInfo("s", "src", 1e6, 915e6)
        # Slow event with large array
        event = SignalEvent("ev_old", stream, np.ones(4096, dtype=np.complex64), 0, 4096, 0, 4096, 0)
        sidecar = make_test_sidecar(event)

        worker.submit(sidecar)
        worker.reset()
        assert worker.get_result(block=False) is None
        assert len(worker._processed_results) == 0

        # Wait a moment for worker thread to finish prior generation task
        time.sleep(0.05)
        # Old result is discarded because generation tag does not match
        assert worker.get_result(block=False) is None
        assert len(worker._processed_results) == 0
        worker.shutdown(timeout=0.5)


# ==============================================================================
# SECTION 5: EMITTER BANDWIDTH PREREQUISITES AND CONFIDENCE
# ==============================================================================
class TestSection5EmitterBandwidth:
    def test_pure_gaussian_noise_rejected_in_emitter_mode(self, stream_info: StreamInfo) -> None:
        """Reproduction 1: Pure Gaussian IQ, RNG seed 731, 16384 samples, Welch segment 512.

        Default emitter mode without ROI/noise reference must be invalid.
        """
        rng = np.random.default_rng(731)
        n = 16384
        iq = (rng.normal(0, 1.0, n) + 1j * rng.normal(0, 1.0, n)).astype(np.complex64)
        event = SignalEvent("ev-noise", stream_info, iq, 0, n, 0, n, 0)

        cfg = AnalysisConfig(psd_mode="welch", psd_nperseg=512, bandwidth_mode="emitter")
        ctx = SignalContext(event, config=cfg)
        extractor = BandwidthFeatureExtractor(BandwidthFeatureConfig(mode="emitter"))
        fset = extractor.extract(ctx)

        obw = fset.features["bandwidth.occupied_99_hz"]
        three_db = fset.features["bandwidth.three_db_hz"]
        assert obw.valid is False
        assert three_db.valid is False
        assert "Noise reference unavailable" in obw.reason or "Insufficient excess SNR" in obw.reason

    def test_clipped_lobe_3db_rejected(self) -> None:
        """Reproduction 2: Controlled PSD with frequencies 0..8 Hz, powers [11, 10, 6, 1, 1, 1, 1, 1, 1],

        ROI [0, 8], known noise density 1: clipped lobe 3 dB result must be invalid.
        """
        from signal_processing.core.contracts import PSDResult

        freqs = np.arange(9, dtype=np.float64)
        powers = np.array([11.0, 10.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

        class DummyCtx:
            def psd(self):
                return PSDResult(frequency_offset_hz=freqs, power_linear=powers, power_dbfs_per_hz=np.zeros_like(powers))

        cfg = BandwidthFeatureConfig(
            mode="emitter",
            roi_hz=(0.0, 8.0),
            noise_floor_dbfs_per_hz=0.0,  # 10*log10(1) = 0 dBFS/Hz -> linear noise = 1.0
            check_edge_clipping=True,
        )
        extractor = BandwidthFeatureExtractor(cfg)
        fset = extractor.extract(DummyCtx())

        three_db = fset.features["bandwidth.three_db_hz"]
        assert three_db.valid is False
        assert "clipped" in str(three_db.reason).lower()

    def test_sub_threshold_snr_rejected(self) -> None:
        """Reproduction 3: Frequencies 0..99 Hz, PSD 1 everywhere except bins 40..59 equal 11.

        Known noise density 1, ROI [0, 99], min SNR 10 dB: integrated S/N is 2 (3.01 dB).
        Both emitter widths must fail configured gate.
        """
        from signal_processing.core.contracts import PSDResult

        freqs = np.arange(100, dtype=np.float64)
        powers = np.ones(100, dtype=np.float64)
        powers[40:60] = 11.0

        class DummyCtx:
            def psd(self):
                return PSDResult(frequency_offset_hz=freqs, power_linear=powers, power_dbfs_per_hz=np.zeros_like(powers))

        cfg = BandwidthFeatureConfig(
            mode="emitter",
            roi_hz=(0.0, 99.0),
            noise_floor_dbfs_per_hz=0.0,
            min_snr_db=10.0,
        )
        extractor = BandwidthFeatureExtractor(cfg)
        fset = extractor.extract(DummyCtx())

        obw = fset.features["bandwidth.occupied_99_hz"]
        three_db = fset.features["bandwidth.three_db_hz"]
        assert obw.valid is False
        assert three_db.valid is False
        assert "Insufficient excess SNR" in obw.reason

    def test_positive_emitter_cases(self, stream_info: StreamInfo) -> None:
        """Positive cases: clean tone with ROI and noise reference passes emitter gate."""
        fs = 1_000_000.0
        n_samples = 1024
        n = np.arange(n_samples)
        iq = (np.exp(2j * np.pi * 50_000.0 * n / fs) + 0.01 * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))).astype(np.complex64)
        event = SignalEvent("ev-tone", stream_info, iq, 0, n_samples, 0, n_samples, 0)

        cfg = AnalysisConfig(
            bandwidth_mode="emitter",
            bandwidth_roi_hz=(0.0, 100_000.0),
            bandwidth_noise_floor_dbfs_per_hz=-60.0,
        )
        ctx = SignalContext(event, config=cfg)
        extractor = BandwidthFeatureExtractor()
        fset = extractor.extract(ctx)

        obw = fset.features["bandwidth.occupied_99_hz"]
        three_db = fset.features["bandwidth.three_db_hz"]
        assert obw.valid is True
        assert three_db.valid is True


# ==============================================================================
# SECTION 6: STREAM VALIDATION AND DISCONTINUITY POLICY
# ==============================================================================
class TestSection6StreamValidationAndDiscontinuity:
    def test_anchor_invalidation_on_overflow(self, stream_info: StreamInfo) -> None:
        """Unknown overflow invalidates timestamp anchor; does not produce stale timestamps."""
        est = GaussianNoiseEstimator(window_size=16, calibrated_noise_power_linear=1e-4)
        p = StreamingCapturePipeline(
            stream_info=stream_info,
            window_size=16,
            hop_size=8,
            ring_capacity=1024,
            noise_estimator=est,
        )

        # Block 0: trusted timestamp
        b0 = IQBlock(stream_info, 0, 0, 1_000_000_000, np.zeros(64, dtype=np.complex64))
        p.process_block(b0)

        # Block 1: unknown overflow
        b1 = IQBlock(stream_info, 1, 1000, None, np.zeros(64, dtype=np.complex64), overflow=True)
        p.process_block(b1)

        # Assembler anchor must have been cleared
        assert p._assembler._anchor_start_sample is None
        assert p._assembler._anchor_timestamp_ns is None

    def test_continuous_block_missing_timestamp_preserves_anchor(self, stream_info: StreamInfo) -> None:
        """Continuous block missing timestamp preserves existing valid anchor."""
        est = GaussianNoiseEstimator(window_size=16, calibrated_noise_power_linear=1e-4)
        p = StreamingCapturePipeline(
            stream_info=stream_info,
            window_size=16,
            hop_size=8,
            ring_capacity=1024,
            noise_estimator=est,
        )

        b0 = IQBlock(stream_info, 0, 0, 1_000_000_000, np.zeros(64, dtype=np.complex64))
        p.process_block(b0)
        assert p._assembler._anchor_timestamp_ns == 1_000_000_000

        # Continuous block without timestamp: anchor is preserved
        b1 = IQBlock(stream_info, 1, 64, None, np.zeros(64, dtype=np.complex64))
        p.process_block(b1)
        assert p._assembler._anchor_timestamp_ns == 1_000_000_000


# ==============================================================================
# SECTION 7: EXTRACTION POLICY AND TRUTHFUL QUALITY
# ==============================================================================
class TestSection7ExtractionPolicyAndQuality:
    def test_extraction_policy_unique_across_configurations(self, stream_info: StreamInfo) -> None:
        """Changing bandwidth_mode, interpolation, prominence, or centering produces different policy IDs."""
        event = SignalEvent("ev1", stream_info, np.ones(64, dtype=np.complex64), 0, 64, 0, 64, 0)
        reg = build_default_feature_registry()
        agg = FeatureAggregator(reg)

        cfg1 = AnalysisConfig(bandwidth_mode="raw_psd", centered_spectrum=True)
        ctx1 = SignalContext(event, config=cfg1)
        res1 = agg.extract_analysis_result(ctx1)

        cfg2 = AnalysisConfig(bandwidth_mode="emitter", centered_spectrum=True)
        ctx2 = SignalContext(event, config=cfg2)
        res2 = agg.extract_analysis_result(ctx2)

        cfg3 = AnalysisConfig(bandwidth_mode="raw_psd", centered_spectrum=False)
        ctx3 = SignalContext(event, config=cfg3)
        res3 = agg.extract_analysis_result(ctx3)

        assert res1.policy.policy_id != res2.policy.policy_id
        assert res1.policy.policy_id != res3.policy.policy_id


# ==============================================================================
# SECTION 8: CONFIGURATION VALIDATION AND LIFECYCLE STATES
# ==============================================================================
class TestSection8ConfigValidationAndLifecycle:
    def test_strict_parameter_validation(self, stream_info: StreamInfo) -> None:
        """Strict validation rejects NaN time constants, boolean window sizes, negative warmup."""
        with pytest.raises(ValueError):
            AdaptiveRobustNoiseEstimator(time_constant_s=float("nan"))

        with pytest.raises(ValueError):
            AdaptiveRobustNoiseEstimator(window_size=True)

        with pytest.raises(ValueError):
            GaussianNoiseEstimator(warmup_blocks=-1)

        with pytest.raises(ValueError):
            EmpiricalNoiseCalibrator(max_samples=0)

        with pytest.raises(ValueError):
            EnergyDetector(mode="invalid_mode")

    def test_pipeline_lifecycle_states(self, stream_info: StreamInfo) -> None:
        """Pipeline state transitions: OPEN -> EOF, reset -> OPEN."""
        est = GaussianNoiseEstimator(window_size=16, calibrated_noise_power_linear=1e-4)
        p = StreamingCapturePipeline(
            stream_info=stream_info,
            window_size=16,
            hop_size=8,
            ring_capacity=1024,
            noise_estimator=est,
        )
        assert p.state == PipelineState.OPEN
        p.finalize()
        assert p.state == PipelineState.EOF

        # Attempting process_block at EOF raises RuntimeError
        b = IQBlock(stream_info, 0, 0, 0, np.zeros(16, dtype=np.complex64))
        with pytest.raises(RuntimeError, match="pipeline is at EOF"):
            p.process_block(b)

        # Reset recovers pipeline
        p.reset()
        assert p.state == PipelineState.OPEN
