"""Behavioral tests for EnergyDetector, noise policies, and causal streaming windowing."""

import math

import numpy as np
import pytest
from scipy.stats import gamma

from signal_processing.core.contracts import IQBlock, NoiseEstimate, StreamInfo
from signal_processing.detection.energy_detector import EnergyDetector
from signal_processing.detection.noise_estimation import (
    GaussianNoiseEstimator,
)


@pytest.fixture
def stream_info() -> StreamInfo:
    return StreamInfo(
        stream_id="stream-det-01",
        source_id="synthetic-source",
        sample_rate_hz=1e6,
        center_frequency_hz=1e9,
    )


class TestDetectorAndNoise:
    """Rigorous verification of detection physics, CFAR statistics, and streaming invariance."""

    def test_actual_detection_power_against_float64_oracle(self, stream_info: StreamInfo) -> None:
        """Measured power changes with input power and matches an independent float64 oracle."""
        detector = EnergyDetector(window_size=256, hop_size=64)
        noise_est = NoiseEstimate(
            stream_id=stream_info.stream_id,
            block_sequence_number=0,
            noise_power_dbfs=-60.0,
            valid=True,
        )

        rng = np.random.default_rng(42)
        # Low power noise block (sigma = 1e-4 -> power ~ -77 dBFS)
        noise_iq = (rng.normal(0, 1e-4, 512) + 1j * rng.normal(0, 1e-4, 512)).astype(np.complex64)
        block_low = IQBlock(stream=stream_info, sequence_number=0, start_sample=0, timestamp_ns=None, iq=noise_iq)

        # High power signal block
        sig_iq = (rng.normal(0, 0.5, 512) + 1j * rng.normal(0, 0.5, 512)).astype(np.complex64)
        block_high = IQBlock(stream=stream_info, sequence_number=1, start_sample=512, timestamp_ns=None, iq=sig_iq)

        meas_low = detector.process_windows(block_low, noise_est)
        meas_high = detector.process_windows(block_high, noise_est)

        # Values are not constants
        assert meas_low[0].signal_power_dbfs != meas_high[0].signal_power_dbfs
        assert meas_high[0].signal_power_dbfs > meas_low[0].signal_power_dbfs + 20.0
        assert meas_high[0].above_on_threshold is True
        assert meas_low[0].above_on_threshold is False

        # Independent float64 oracle verification for each window
        for m in meas_low:
            w_iq = noise_iq[m.start_sample : m.end_sample]
            oracle_lin = float(np.mean(np.real(w_iq.astype(np.complex128)) ** 2 + np.imag(w_iq.astype(np.complex128)) ** 2))
            oracle_dbfs = 10.0 * math.log10(oracle_lin)
            assert math.isclose(m.signal_power_dbfs, oracle_dbfs, rel_tol=1e-5, abs_tol=1e-5)

    def test_no_future_leakage(self, stream_info: StreamInfo) -> None:
        """Future samples cannot affect earlier decisions (strict causality)."""
        detector1 = EnergyDetector(window_size=128, hop_size=64)
        detector2 = EnergyDetector(window_size=128, hop_size=64)
        noise_est = NoiseEstimate(
            stream_id=stream_info.stream_id,
            block_sequence_number=0,
            noise_power_dbfs=-60.0,
            valid=True,
        )

        rng = np.random.default_rng(999)
        initial_iq = (rng.normal(0, 1e-3, 512) + 1j * rng.normal(0, 1e-3, 512)).astype(np.complex64)
        block_init = IQBlock(stream=stream_info, sequence_number=0, start_sample=0, timestamp_ns=None, iq=initial_iq)

        # Run detector 1 on initial block only
        meas1 = detector1.process_windows(block_init, noise_est)

        # Run detector 2 on initial block, then immediately feed an enormous burst in block 2
        meas2 = detector2.process_windows(block_init, noise_est)

        burst_iq = (np.ones(512) * 5.0).astype(np.complex64)
        block_burst = IQBlock(stream=stream_info, sequence_number=1, start_sample=512, timestamp_ns=None, iq=burst_iq)
        _ = detector2.process_windows(block_burst, noise_est)

        # Earlier decisions in detector 2 must be identically equal to detector 1
        assert len(meas1) == len(meas2)
        for m1, m2 in zip(meas1, meas2, strict=True):
            assert m1.start_sample == m2.start_sample
            assert m1.end_sample == m2.end_sample
            assert m1.signal_power_dbfs == m2.signal_power_dbfs
            assert m1.above_on_threshold == m2.above_on_threshold

    def test_window_partition_invariance(self, stream_info: StreamInfo) -> None:
        """Partitioning a continuous stream into different block sizes yields identical measurements."""
        rng = np.random.default_rng(101)
        total_len = 1024
        stream_iq = (rng.normal(0, 0.1, total_len) + 1j * rng.normal(0, 0.1, total_len)).astype(np.complex64)
        noise_est = NoiseEstimate(
            stream_id=stream_info.stream_id,
            block_sequence_number=0,
            noise_power_dbfs=-40.0,
            valid=True,
        )

        # Run 1: single large block of 1024 samples
        det1 = EnergyDetector(window_size=128, hop_size=64)
        b1 = IQBlock(stream=stream_info, sequence_number=0, start_sample=0, timestamp_ns=None, iq=stream_iq)
        res1 = det1.process_windows(b1, noise_est)

        # Run 2: irregular chunk sizes: [100, 250, 45, 629]
        det2 = EnergyDetector(window_size=128, hop_size=64)
        chunks = [100, 250, 45, 629]
        res2 = []
        cursor = 0
        for seq, c_len in enumerate(chunks):
            chunk_iq = stream_iq[cursor : cursor + c_len]
            b2 = IQBlock(stream=stream_info, sequence_number=seq, start_sample=cursor, timestamp_ns=None, iq=chunk_iq)
            res2.extend(det2.process_windows(b2, noise_est))
            cursor += c_len

        assert len(res1) == len(res2)
        for m1, m2 in zip(res1, res2, strict=True):
            assert m1.start_sample == m2.start_sample
            assert m1.end_sample == m2.end_sample
            assert math.isclose(m1.signal_power_dbfs, m2.signal_power_dbfs, abs_tol=1e-5)
            assert m1.above_on_threshold == m2.above_on_threshold

    def test_gaussian_gamma_threshold_and_false_alarm_rate(self) -> None:
        """Validate Gamma CFAR threshold formula and test empirical false alarms on holdout noise."""
        N = 256
        sigma2 = 1.0
        pfa_target = 0.01  # 1% false alarm target

        scale = sigma2 / N
        t_on = float(gamma.isf(pfa_target, a=N, scale=scale))

        # Evaluate on 2,000 independent non-overlapping holdout noise windows
        num_windows = 2000
        rng = np.random.default_rng(777)
        noise_samples = (rng.normal(0, math.sqrt(sigma2 / 2), num_windows * N) +
                         1j * rng.normal(0, math.sqrt(sigma2 / 2), num_windows * N)).astype(np.complex64)

        powers = np.mean(
            np.real(noise_samples.reshape(num_windows, N).astype(np.complex128)) ** 2 +
            np.imag(noise_samples.reshape(num_windows, N).astype(np.complex128)) ** 2,
            axis=1,
        )

        false_alarms = int(np.count_nonzero(powers >= t_on))
        measured_pfa = false_alarms / num_windows

        # 99% confidence interval under Binomial(num_windows=2000, p=0.01):
        # Std dev = sqrt(0.01 * 0.99 / 2000) ~= 0.0022. 3*sigma band: [0.003, 0.017]
        assert 0.003 <= measured_pfa <= 0.018, f"Measured Pfa {measured_pfa} outside 3-sigma band of 0.01"

    def test_configured_band_in_band_detection_and_out_of_band_rejection(self, stream_info: StreamInfo) -> None:
        """Configured-band detector responds to in-band signal and rejects out-of-band signal."""
        # Monitored band: 50 kHz to 150 kHz (center = 100 kHz, bandwidth = 100 kHz)
        det = EnergyDetector(
            window_size=256,
            hop_size=64,
            mode="configured_band",
            monitored_band_hz=(50_000.0, 150_000.0),
            sample_rate_hz=stream_info.sample_rate_hz,
        )
        assert det.filter_delay_samples > 0

        noise_est = NoiseEstimate(
            stream_id=stream_info.stream_id,
            block_sequence_number=0,
            noise_power_dbfs=-60.0,
            valid=True,
        )

        n = 1024
        t = np.arange(n) / stream_info.sample_rate_hz

        # In-band tone at 100 kHz
        in_band_iq = (np.exp(2j * np.pi * 100_000.0 * t)).astype(np.complex64)
        b_in = IQBlock(stream=stream_info, sequence_number=0, start_sample=0, timestamp_ns=None, iq=in_band_iq)
        meas_in = det.process_windows(b_in, noise_est)

        # Out-of-band tone at -300 kHz (far outside passband)
        det.reset()
        oob_iq = (np.exp(2j * np.pi * -300_000.0 * t)).astype(np.complex64)
        b_oob = IQBlock(stream=stream_info, sequence_number=0, start_sample=0, timestamp_ns=None, iq=oob_iq)
        meas_oob = det.process_windows(b_oob, noise_est)

        # In-band power must be vastly higher than attenuated out-of-band power (> 20 dB suppression)
        p_in = meas_in[-1].signal_power_dbfs
        p_oob = meas_oob[-1].signal_power_dbfs
        assert p_in - p_oob > 25.0

    def test_noise_estimator_warmup_and_freeze(self, stream_info: StreamInfo) -> None:
        """Noise estimator completes warmup, tracks noise, and freezes during active bursts."""
        estimator = GaussianNoiseEstimator(window_size=256, warmup_blocks=3)
        rng = np.random.default_rng(555)

        # Feed block 0: still in warmup
        b0 = IQBlock(
            stream=stream_info, sequence_number=0, start_sample=0, timestamp_ns=None,
            iq=(rng.normal(0, 0.1, 512) + 1j * rng.normal(0, 0.1, 512)).astype(np.complex64),
        )
        ne0 = estimator.update(b0)
        assert ne0.valid is False
        assert "Warmup" in str(ne0.reason)

        # Feed blocks 1 and 2: warmup completes
        b1 = IQBlock(
            stream=stream_info, sequence_number=1, start_sample=512, timestamp_ns=None,
            iq=(rng.normal(0, 0.1, 512) + 1j * rng.normal(0, 0.1, 512)).astype(np.complex64),
        )
        b2 = IQBlock(
            stream=stream_info, sequence_number=2, start_sample=1024, timestamp_ns=None,
            iq=(rng.normal(0, 0.1, 512) + 1j * rng.normal(0, 0.1, 512)).astype(np.complex64),
        )
        _ = estimator.update(b1)
        ne2 = estimator.update(b2)
        assert ne2.valid is True
        assert estimator.is_calibrated is True
        calibrated_p = estimator.noise_power_linear

        # Freeze estimator and feed high-power burst
        estimator.freeze(True)
        burst_block = IQBlock(
            stream=stream_info, sequence_number=3, start_sample=1536, timestamp_ns=None,
            iq=(np.ones(512) * 10.0).astype(np.complex64),
        )
        _ = estimator.update(burst_block)
        # Noise estimate remains frozen and uncorrupted by burst!
        assert estimator.noise_power_linear == calibrated_p
