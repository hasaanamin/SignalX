"""Expanded E01-E14 contracts. Original audit scripts remain in audit_verification/evidence."""
import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

from signal_processing.buffering import CircularIQBuffer
from signal_processing.context import AnalysisConfig, SignalContext
from signal_processing.core import ExtractionPolicy, IQBlock, NoiseEstimate, SignalEvent, StreamInfo
from signal_processing.detection import AdaptiveRobustNoiseEstimator, EmpiricalNoiseCalibrator, EnergyDetector, GaussianNoiseEstimator
from signal_processing.detection.exceptions import PolicyCompatibilityError, UncalibratedNoiseError
from signal_processing.features import FeatureAggregator
from signal_processing.features.aggregation import FeatureRegistry, build_default_feature_registry
from signal_processing.features.bandwidth import BandwidthFeatureConfig, BandwidthFeatureExtractor
from signal_processing.runtime import FeatureAnalysisWorker, StreamingCapturePipeline

STREAM = StreamInfo("e-repairs", "test", 1e6)


def block(iq, seq=0, start=0, **kwargs):
    return IQBlock(STREAM, seq, start, None, np.asarray(iq, dtype=np.complex64), **kwargs)


def pipeline(**kwargs):
    options = dict(stream_info=STREAM, window_size=17, hop_size=8, ring_capacity=4096,
                   start_persistence=1, end_persistence=1, pre_trigger_samples=0,
                   pre_signal_margin_samples=0, post_signal_samples=0, min_post_trigger_samples=0,
                   enable_feature_analysis=False,
                   noise_estimator=GaussianNoiseEstimator(window_size=17, calibrated_noise_power_linear=1e-4))
    options.update(kwargs)
    return StreamingCapturePipeline(**options)


@pytest.mark.parametrize("origin", [0, 1, 1000, 10**12])
@pytest.mark.parametrize("partition", [1, 8, 17, 64])
def test_absolute_admission_translation_and_replay(origin, partition):
    est = AdaptiveRobustNoiseEstimator(window_size=17, initial_noise_power_linear=1e-4,
                                      adaptation_window_size=1, time_constant_s=1e-6)
    p = pipeline(noise_estimator=est)
    iq = np.full(64, .003, np.complex64); iq[17:] = .1
    for repeat in range(2):
        events = []
        for seq, start in enumerate(range(0, len(iq), partition)):
            events += p.process_block(block(iq[start:start+partition], seq, origin+start))
        events += p.flush()
        assert len(events) == 1
        event = events[0].event
        np.testing.assert_array_equal(event.iq, iq[event.event_start_sample-origin:event.event_end_sample-origin])
        assert est.noise_power_linear < 1e-4  # No active IQ learned.
        assert p._noise_admission_sample >= origin
        p.reset()
    p.shutdown()


def test_quiet_state_is_bounded_over_many_ring_wraps():
    p = pipeline(window_size=256, hop_size=64)
    for seq in range(512):
        p.process_block(block(np.full(1024, .003), seq, seq*1024))
        assert len(p._pending_noise_buf) <= 256
        assert not p._pending_noise_iq
        assert p.ring_buffer.occupancy <= 4096
    p.shutdown()


@pytest.mark.parametrize("overflow", [False, True])
@pytest.mark.parametrize("origin", [0, 9000])
def test_gap_evaluates_complete_old_window_without_splicing(origin, overflow):
    p = pipeline()
    iq = np.full(17, .5, np.complex64)
    assert not p.process_block(block(iq, start=origin))
    events = p.process_block(block(np.zeros(32), 1, origin+49, dropped_samples_before=32, overflow=overflow))
    assert len(events) == 1
    sidecar = events[0]
    np.testing.assert_array_equal(sidecar.event.iq, iq)
    assert sidecar.termination_reason == "discontinuity"
    assert sidecar.known_lost_samples == 32 and sidecar.unknown_loss == overflow
    p.shutdown()


def test_startup_expiry_is_live_without_training():
    est = AdaptiveRobustNoiseEstimator(window_size=256, initial_noise_power_linear=1e-4,
                                      max_staleness_samples=16)
    p = pipeline(window_size=256, hop_size=64, noise_estimator=est)
    with pytest.raises(UncalibratedNoiseError):
        p.process_block(block(np.full(192, .003)))
    assert est.staleness_samples > 16 and est.total_accepted_samples == 0
    assert p.state.value == "FAULTED"
    p.shutdown()


def policy(n=16, count=20000):
    rng = np.random.default_rng(117)
    iq = ((rng.normal(size=count)+1j*rng.normal(size=count))*.007).astype(np.complex64)
    return EmpiricalNoiseCalibrator(window_size=n, hop_size=8, pfa_on=.01, pfa_off=.1).create_policy(iq, settling_samples=0)


def test_midblock_fault_exposes_consumed_prefix_and_requires_reset():
    q = policy()
    est = AdaptiveRobustNoiseEstimator(window_size=16, initial_noise_power_linear=q.mean_power_linear,
                                      adaptation_window_size=16, time_constant_s=1e-6)
    p = pipeline(window_size=16, noise_estimator=est, threshold_policy=q, pfa_on=.01, pfa_off=.1)
    with pytest.raises(PolicyCompatibilityError):
        p.process_block(block(np.full(64, .003)))
    assert p.state.value == "FAULTED"
    assert p.consumed_end_sample == 16 and p.received_end_sample == 64
    assert p._previous_block is None
    with pytest.raises(RuntimeError):
        p.process_block(block(np.full(64, .003)))
    p.shutdown()


@pytest.mark.parametrize("n", [64, 256, 8192])
@pytest.mark.parametrize("width", [10, 1000, 20000])
def test_configured_window_sizes_and_single_sample(n, width):
    d = EnergyDetector(window_size=n, hop_size=64, mode="configured_band",
                       monitored_band_hz=(100000-width/2, 100000+width/2))
    m = d.process(block([1]), NoiseEstimate(STREAM.stream_id, 0, 0))
    assert 10**(m.on_threshold_dbfs/10) == pytest.approx(-math.log(1e-4), rel=1e-12)
    assert 1-1e-6 <= d._n_eff <= n


def test_domain_identity_and_scale_are_separate():
    q = policy()
    d = EnergyDetector(window_size=16, hop_size=8, pfa_on=.01, pfa_off=.1, threshold_policy=q)
    for scale in (.1, .867, 1.121, 10):
        ref = NoiseEstimate(STREAM.stream_id, 0, 10*math.log10(q.mean_power_linear*scale), reference_domain=q.reference_domain)
        assert d.process(block(np.full(16, .003)), ref)
    wrong = dataclasses.replace(q.reference_domain, gain_id="different_gain")
    with pytest.raises(PolicyCompatibilityError):
        d.process(block(np.full(16, .003)), NoiseEstimate(STREAM.stream_id, 0, -40, reference_domain=wrong))


def test_wrong_input_clock_is_rejected_before_filter_mutation():
    d = EnergyDetector(window_size=16, hop_size=8, mode="configured_band", monitored_band_hz=(90000, 110000))
    wrong_stream = dataclasses.replace(STREAM, sample_rate_hz=2e6)
    before = d._sos_zi.copy()
    with pytest.raises(PolicyCompatibilityError):
        d.process_windows(IQBlock(wrong_stream, 0, 0, None, np.ones(32, np.complex64)), NoiseEstimate(STREAM.stream_id, 0, -40))
    assert d._phase_accumulator == 0
    np.testing.assert_array_equal(before, d._sos_zi)


def sidecar():
    p = pipeline()
    p.process_block(block(np.full(32, .5)))
    result = p.flush()[0]
    p.shutdown()
    return dataclasses.replace(result, stream_epoch=42)


def test_reentrant_reset_suppresses_second_sink_and_preserves_epoch():
    w = FeatureAnalysisWorker(synchronous=True, skip_incomplete_quality=False)
    calls = []
    def first(sc):
        calls.append(sc.stream_epoch)
        w.reset()
    w.add_result_sink(first); w.add_result_sink(lambda sc: calls.append(-1))
    result = w.submit(sidecar())
    assert calls == [42] and result.stream_epoch == 42 and result.processing_generation == 0
    assert w.discarded_old_generation_count == 1
    w.shutdown()


def test_analysis_envelope_retains_loss_calibration_and_capture_epoch():
    w = FeatureAnalysisWorker(synchronous=True, skip_incomplete_quality=False)
    sc = dataclasses.replace(sidecar(), noise_valid=False, known_lost_samples=12, unknown_loss=True)
    result = w.submit(sc)
    q = result.analysis_result.capture_quality
    assert q.noise_valid is False and q.known_lost_samples == 12 and q.unknown_loss
    assert q.capture_epoch == 42 and q.is_continuous and q.dropped_samples == 0
    w.shutdown()


def test_result_queue_suppresses_old_processing_generation():
    w = FeatureAnalysisWorker(synchronous=True)
    w.submit(sidecar())
    w.advance_generation()
    assert w.get_result(block=False) is None
    assert w.discarded_old_generation_count == 1
    w.shutdown()


def test_authorized_settling_is_excluded_and_counted_in_consumed_prefix():
    e = GaussianNoiseEstimator(window_size=16, warmup_samples=16, warmup_settling_samples=8)
    assert e.update_samples(np.full(8, .9, np.complex64)) == 8
    assert e.samples_trained == 0
    assert e.update_samples(np.full(32, .01, np.complex64)) == 16
    assert e.samples_trained == 16 and e.noise_power_linear == pytest.approx(1e-4)


def test_reference_domain_cannot_be_relabelled_without_recalibration():
    q = policy()
    with pytest.raises(ValueError):
        dataclasses.replace(q, monitored_band_hz=(-110000, -90000))


def test_certified_detector_rejects_training_only_and_rescaling():
    q = policy()
    with pytest.raises(PolicyCompatibilityError):
        EnergyDetector(window_size=16, hop_size=8, pfa_on=.01, pfa_off=.1,
                       threshold_policy=q, require_supported_pfa=True)
    q = q.validate_held_out(np.full(1000, q.t_off_linear/2), reference_domain=q.reference_domain,
                           independence_assumption="independent_held_out_windows", evidence_id="test")
    d = EnergyDetector(window_size=16, hop_size=8, pfa_on=.01, pfa_off=.1,
                       threshold_policy=q, require_supported_pfa=True)
    with pytest.raises(PolicyCompatibilityError):
        d.process(block(np.full(16, .003)), NoiseEstimate(STREAM.stream_id, 0,
                  10*math.log10(q.mean_power_linear*2), reference_domain=q.reference_domain))


def test_heavy_tails_preserve_observed_width_but_require_physical_support():
    f = np.arange(-12, 13, dtype=float)
    ctx = SimpleNamespace(psd=lambda: SimpleNamespace(frequency_offset_hz=f, power_linear=1+100/(1+f*f)))
    cfg = BandwidthFeatureConfig(mode="emitter", roi_hz=(-11, 11), noise_floor_dbfs_per_hz=0)
    observed = BandwidthFeatureExtractor(cfg).extract(ctx).features
    assert observed["bandwidth.occupied_99_hz"].valid
    assert observed["bandwidth.occupied_99_hz"].interpretation == "observed_excess_within_roi"
    physical = BandwidthFeatureExtractor(dataclasses.replace(cfg, occupied_power_scope="physical_emitter")).extract(ctx).features
    assert not physical["bandwidth.occupied_99_hz"].valid
    assert physical["bandwidth.three_db_hz"].valid
    supported = BandwidthFeatureExtractor(dataclasses.replace(cfg, occupied_power_scope="physical_emitter",
                       complete_observation=True, support_provenance="caller asserts receiver covers modeled emitter")).extract(ctx).features
    assert supported["bandwidth.occupied_99_hz"].valid


def context():
    iq = np.exp(.1j*np.arange(1024)).astype(np.complex64)
    return SignalContext(SignalEvent("policy-event", STREAM, iq, 0, 1024, 0, 1024, 0), AnalysisConfig())


@pytest.mark.parametrize("change", [dict(auto_noise_estimation=False), dict(multimodal_ratio_threshold=.9),
                                    dict(occupied_power_scope="physical_emitter"), dict(min_snr_db=8.)])
def test_every_effective_bandwidth_control_has_distinct_identity(change):
    cfg = BandwidthFeatureConfig()
    def extract(c):
        return FeatureAggregator(FeatureRegistry([BandwidthFeatureExtractor(c)])).extract_analysis_result(context()).policy
    assert extract(cfg).policy_id != extract(dataclasses.replace(cfg, **change)).policy_id


def test_immutable_canonical_policy_round_trip():
    p = FeatureAggregator(build_default_feature_registry()).extract_analysis_result(context()).policy
    with pytest.raises(TypeError):
        p.extractor_versions["amplitude"] = "invented"
    record = p.to_dict()
    assert ExtractionPolicy.from_dict(record) == p
    record["extractor_versions"]["amplitude"] = "invented"
    with pytest.raises(ValueError):
        ExtractionPolicy.from_dict(record)


def test_training_resolution_is_not_confidence_and_held_out_support_is_explicit():
    q = policy()
    assert q.valid and not q.supported_pfa
    evidence = np.full(1000, q.t_off_linear / 2)
    validated = q.validate_held_out(evidence, reference_domain=q.reference_domain,
                                  independence_assumption="independent_held_out_windows", evidence_id="independent-fixture")
    assert validated.supported_pfa
    assert validated.pfa_upper_on == pytest.approx(1-.025**(1/1000))
    small = q.validate_held_out(evidence[:101], reference_domain=q.reference_domain,
                               independence_assumption="independent_held_out_windows", evidence_id="small-fixture")
    assert not small.supported_pfa
    with pytest.raises(ValueError):
        q.validate_held_out(evidence, reference_domain=q.reference_domain,
                           independence_assumption="adjacent_filtered_windows", evidence_id="correlated")


def test_failed_finalize_can_continue_and_warmup_statistics_are_constant_size():
    c = EmpiricalNoiseCalibrator(window_size=16)
    c.begin_calibration(); c.update_calibration(np.ones(8, np.complex64))
    with pytest.raises(ValueError):
        c.finalize_calibration()
    assert not c._finalized
    c.update_calibration(np.ones(8, np.complex64)); c.finalize_calibration()
    e = GaussianNoiseEstimator(warmup_samples=2048)
    for _ in range(2048):
        e.update_samples(np.full(1, .01, np.complex64))
    assert e.is_calibrated and not hasattr(e, "_warmup_powers")
    assert e.noise_power_linear == pytest.approx(1e-4)


@pytest.mark.parametrize("value", [-1, True, 1.5, float("nan")])
def test_invalid_buffer_reset_is_atomic(value):
    ring = CircularIQBuffer(16)
    ring.write(100, np.ones(8, np.complex64))
    before = ring.available_range(), ring.stream_generation
    with pytest.raises(ValueError):
        ring.reset(value)
    assert (ring.available_range(), ring.stream_generation) == before


def test_bipartite_assignment_accounts_for_duplicate_and_extra_detections():
    from benchmarks.sustained_evaluation import match_intervals
    matched, missed, extras = match_intervals([(0, 10), (20, 30)], [(0, 10), (1, 9), (20, 30), (50, 60)])
    assert len(matched) == 2 and not missed and len(extras) == 2


def test_bounded_acquisition_overload_and_recovery():
    from benchmarks.sustained_evaluation import bounded_acquisition
    r = bounded_acquisition(.15, sample_rate_hz=100000, queue_capacity=2, consumer_pause_seconds=.05)
    assert r["acquisition_queue_peak"] <= 2 and r["dropped_samples"] > 0
    assert r["recovered"] and r["sample_accounting_balanced"]
    assert r["acquisition_unfinished"] == 0 and r["retained_history_samples"] == 0
