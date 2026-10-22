"""Public streaming tests: no designated noise prefix, SNR or ideal reference."""
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.signal import lfilter

from signal_processing.core import IQBlock, StreamInfo
from signal_processing.core.exceptions import StreamConsistencyError
from signal_processing.detection import OnlineNoiseConfig, OnlineNoiseTracker
from signal_processing.detection.exceptions import UncalibratedNoiseError
from signal_processing.runtime.pipeline import StreamingCapturePipeline


def samples(n, rho=0, seed=2233, power=1e-4):
    rng = np.random.default_rng(seed)
    z = (rng.normal(size=n)+1j*rng.normal(size=n))*math.sqrt(power*(1-abs(rho)**2)/2)
    return lfilter([1], [1, -rho], z).astype(np.complex64)


def pipeline(config=None, **kwargs):
    s = StreamInfo("online", "test", 1e6)
    p = StreamingCapturePipeline(s, window_size=64, hop_size=16, ring_capacity=4096,
        pre_trigger_samples=128, pre_signal_margin_samples=64, post_signal_samples=64,
        min_post_trigger_samples=64, start_persistence=2, end_persistence=3, max_event_samples=8192,
        online_noise=config or OnlineNoiseConfig(), enable_feature_analysis=kwargs.pop("features", False), **kwargs)
    return s, p


def feed(p, s, x, size=1024, origin=0):
    out = []
    for seq, i in enumerate(range(0, len(x), size)):
        out.extend(p.process_block(IQBlock(s, seq, origin+i, None, x[i:i+size])))
    out.extend(p.flush())
    return out


@pytest.mark.parametrize("rho,model", [(0, "gaussian"), (0, "ar1"), (.8, "ar1"), (.5+.3j, "ar1")])
def test_online_fits_actual_samples_and_dynamic_thresholds(rho, model):
    s, p = pipeline(OnlineNoiseConfig(model=model, update_frames=1))
    seen = []
    original = p.detector.process_windows
    def observe(block, estimate):
        result = original(block, estimate)
        seen.extend(result)
        return result
    p.detector.process_windows = observe
    x = samples(24000, rho)
    try:
        feed(p, s, x)
        snapshot = p.noise_estimator.snapshot
        assert abs(snapshot.coefficient-rho) < .04
        assert snapshot.innovation_power == pytest.approx(1e-4*(1-abs(rho)**2), rel=.08)
        assert snapshot.revision > 5
        assert seen and seen[0].end_sample > 8192
        assert all(m.threshold_support_method == "model_only" for m in seen)
        assert len({m.on_threshold_linear for m in seen}) > 5
        assert not p.noise_estimator.diagnostics()["supported_pfa"]
    finally:
        p.shutdown()


@pytest.mark.parametrize("size", [1, 7, 16, 173, 1024, 24000])
def test_partition_oracle_raw_iq_pretrigger_and_revision(size):
    x = samples(14000, .8)
    x[9500:9700] += .15*np.exp(.2j*np.arange(200)).astype(np.complex64)
    signature = []
    for part in (1024, size):
        s, p = pipeline()
        original = p.detector.process_windows
        seen = []
        def oracle(block, estimate):
            model = p.noise_estimator.snapshot
            measurements = original(block, estimate)
            for m in measurements:
                a, b = m.start_sample-10**12, m.end_sample-10**12
                raw = x[a-1:b].astype(np.complex128)
                y = (raw[1:]-model.mean)-model.coefficient*(raw[:-1]-model.mean)
                expected = math.fsum(float(v.real)**2+float(v.imag)**2 for v in y)/len(y)
                assert m.signal_power_linear == pytest.approx(expected, rel=1e-12, abs=0)
                seen.append((m.start_sample, m.signal_power_linear, m.on_threshold_linear, m.above_on_threshold))
            return measurements
        p.detector.process_windows = oracle
        try:
            out = feed(p, s, x, part, 10**12)
            target = next(c for c in out if c.event.detected_start_sample < 10**12+9700 and c.event.detected_end_sample > 10**12+9500)
            assert target.event.pre_trigger_samples >= 128
            for c in out:
                np.testing.assert_array_equal(c.event.iq, x[c.event.event_start_sample-10**12:c.event.event_end_sample-10**12])
            signature.append((seen, [(c.event.detected_start_sample, c.event.detected_end_sample) for c in out], p.noise_estimator.snapshot))
        finally:
            p.shutdown()
    assert signature[0] == signature[1]


@pytest.mark.parametrize("data", [np.zeros(12000, np.complex64), np.ones(12000, np.complex64),
                                  np.exp(.1j*np.arange(12000)).astype(np.complex64)])
def test_zero_and_tonal_startup_remain_explicitly_uncertain(data):
    s, p = pipeline()
    try:
        assert feed(p, s, data) == []
        assert p.noise_estimator.snapshot is None
        assert not p.noise_estimator.get_estimate().valid
        assert p.noise_estimator.diagnostics()["state"] == "STARTING"
        assert p.retention_snapshot()["retained_noise_samples"] == 0
    finally:
        p.shutdown()


def test_bootstrap_with_burst_and_freeze_does_not_absorb_active_signal():
    x = samples(20000, .8)
    x[1024:2048] += .2  # No caller identifies this contaminated startup frame.
    x[12000:16000] += .2
    s, p = pipeline()
    revisions = []
    try:
        for i in range(0, len(x), 16):
            p.process_block(IQBlock(s, i//16, i, None, x[i:i+16]))
            if 12500 < i < 15000:
                revisions.append(p.noise_estimator.snapshot.revision)
        p.flush()
        assert len(set(revisions)) == 1
        assert p.noise_estimator.noise_power_linear == pytest.approx(3.6e-5, rel=.12)
    finally:
        p.shutdown()


def test_causal_future_cannot_change_earlier_reference_or_decisions():
    x = samples(12000)
    signatures = []
    for suffix in (np.zeros(1024, np.complex64), np.ones(1024, np.complex64)):
        s, p = pipeline()
        original = p.detector.process_windows
        seen = []
        def record(block, estimate):
            result = original(block, estimate)
            seen.extend((m.end_sample, m.on_threshold_linear, m.signal_power_linear) for m in result if m.end_sample <= len(x))
            return result
        p.detector.process_windows = record
        try:
            feed(p, s, np.concatenate((x, suffix)), 20000)
            signatures.append(seen)
        finally:
            p.shutdown()
    assert signatures[0] == signatures[1]


def test_online_rejects_metadata_atomically_and_reset_relearns():
    s, p = pipeline()
    x = samples(10000)
    try:
        p.process_block(IQBlock(s, 0, 1000, None, x[:9001]))
        state = (p.retention_snapshot(), p.noise_estimator.diagnostics(), p.consumed_end_sample, p.received_end_sample)
        with pytest.raises(StreamConsistencyError):
            p.process_block(IQBlock(replace(s, sample_rate_hz=1000000.0005), 1, 10001, None, x[9001:]))
        assert state == (p.retention_snapshot(), p.noise_estimator.diagnostics(), p.consumed_end_sample, p.received_end_sample)
        p.process_block(IQBlock(s, 1, 10001, None, x[9001:]))
        p.reset()
        assert p.noise_estimator.snapshot is None
        feed(p, s, x)
        assert p.noise_estimator.snapshot is not None
    finally:
        p.shutdown()


def test_gap_restarts_blind_learning_without_splicing_and_features_are_real():
    s, p = pipeline(features=True)
    x = samples(12000, .8)
    x[10500:] += .2
    try:
        out = p.process_block(IQBlock(s, 0, 0, None, x))
        out += p.process_block(IQBlock(s, 1, 12032, None, samples(9000, .8, seed=4), dropped_samples_before=32))
        out += p.flush()
        old = next(c for c in out if c.event.event_start_sample < 12000)
        np.testing.assert_array_equal(old.event.iq, x[old.event.event_start_sample:old.event.event_end_sample])
        assert old.known_lost_samples == 32
        assert p.noise_estimator.received_samples == 9000
        assert p.worker.unfinished_work == 0
    finally:
        p.shutdown()


def test_tracker_is_bounded_and_updates_level_and_correlation():
    cfg = OnlineNoiseConfig(time_constant_s=.002, update_frames=1)
    tracker = OnlineNoiseTracker(1e6, cfg)
    first = samples(16384, .2)
    tracker.observe(first, eligible=True)
    old = tracker.snapshot
    tracker.observe(samples(65536, .7, power=2e-4), eligible=True)
    assert abs(tracker.snapshot.coefficient-.7) < .04
    assert tracker.snapshot.innovation_power == pytest.approx(2e-4*(1-.7**2), rel=.12)
    assert tracker.snapshot.model_id != old.model_id
    assert tracker.diagnostics()["retained_samples"] <= cfg.frame_samples*(cfg.history_frames+1)
    tracker.freeze(True)
    fixed = tracker.snapshot
    tracker.observe(np.ones(10000, np.complex64), eligible=True)
    assert tracker.snapshot is fixed
    tracker.freeze(False)
    tracker.observe(samples(2000), eligible=True)
    assert tracker.snapshot.revision > fixed.revision


def test_stale_reference_is_not_silently_relabelled_valid():
    s, p = pipeline(OnlineNoiseConfig(max_age_samples=512))
    try:
        p.process_block(IQBlock(s, 0, 0, None, samples(8192)))
        with pytest.raises(UncalibratedNoiseError):
            p.process_block(IQBlock(s, 1, 8192, None, np.ones(2048, np.complex64)))
        assert p.state.value == "FAULTED"
    finally:
        p.shutdown()


@pytest.mark.parametrize("field,value", [("frame_samples", True), ("history_frames", 2),
    ("time_constant_s", float("nan")), ("max_age_samples", 0), ("model", "anything")])
def test_online_config_validation(field, value):
    with pytest.raises(ValueError):
        OnlineNoiseConfig(**{field: value})


def test_fixed_support_cannot_be_attached_to_dynamic_model():
    from signal_processing.detection.noise_estimation import GaussianNoiseEstimator
    with pytest.raises(ValueError, match="owns"):
        pipeline(noise_estimator=GaussianNoiseEstimator(calibrated_noise_power_linear=1e-4))
    with pytest.raises(ValueError, match="owns"):
        pipeline(require_supported_pfa=True)


def test_default_update_cadence_and_pause_drop_partial_pairs():
    tracker = OnlineNoiseTracker(1e6, OnlineNoiseConfig())
    tracker.observe(samples(8192), eligible=True)
    assert tracker.snapshot.revision == 1
    tracker.observe(samples(4095, seed=7), eligible=True)
    assert tracker.snapshot.revision == 1
    tracker.observe(samples(1, seed=8), eligible=True)
    assert tracker.snapshot.revision == 2
    tracker.observe(samples(17), eligible=True)
    tracker.observe(samples(10), eligible=False)
    assert tracker.diagnostics()["retained_samples"] == 8192


@pytest.mark.parametrize("scale", [1., 1e-8, 1e8])
def test_blind_estimation_preserves_physical_scale(scale):
    tracker = OnlineNoiseTracker(1e6, OnlineNoiseConfig())
    tracker.observe(samples(8192, .8)*np.float32(scale), eligible=True)
    assert tracker.snapshot.innovation_power/(scale*scale) == pytest.approx(3.6e-5, rel=.08)
    assert abs(tracker.snapshot.coefficient-.8) < .025
