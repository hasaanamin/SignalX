"""Behavioral acceptance checks against unchanged signal_processing(2).zip.
These tests execute the implementation; audit failures are expected.
"""
import math, dataclasses, hashlib, threading
from functools import lru_cache
import numpy as np
import pytest
import signal_processing2_deep_probes as h
import signal_processing2_extended_probes as ex
from signal_processing.detection.exceptions import UncalibratedNoiseError,PolicyCompatibilityError
from signal_processing.detection.noise_estimation import NoiseLifecycleState

@lru_cache(None)
def observed(name):
    fn=getattr(h,name,None) or getattr(ex,name)
    return fn()

def test_warmup_exact_prefix_and_remainder_preserved():
    rng=np.random.default_rng(321)
    x=((rng.normal(size=512)+1j*rng.normal(size=512))*math.sqrt(1e-4/2)).astype(np.complex64);x[128:192]+=.5
    rows=[]
    for bs in (1,16,64,512):
        est=h.GaussianNoiseEstimator(window_size=16,warmup_samples=32)
        r=h.run(x,bs,noise_estimator=est)
        assert est.samples_trained==32
        assert est.noise_power_linear==pytest.approx(float(np.mean(abs(x[:32].astype(np.complex128))**2)))
        assert len(r['events'])==1
        rows.append(r['events'])
    assert all(a==rows[0] for a in rows)

def test_stale_age_is_relative_to_last_successful_update():
    rows=observed('age_after_burst')
    assert all(r['age']==0 for r in rows)

def test_staleness_enforced_at_same_logical_sample_prefix():
    rows=observed('staleness_partition')
    assert all(r['error']==rows[0]['error'] for r in rows)
    assert len({r['measurements_processed'] for r in rows})==1

def test_unsupported_zero_power_updates_do_not_keep_reference_fresh():
    rows=observed('no_update_age')
    assert all(r['accepted']==0 for r in rows)
    assert all(r['error'] is not None or r['age']>=128 for r in rows)

def test_non_hop_aligned_blocks_have_identical_capture_and_reference():
    x=np.full(512,.003,np.complex64);x[257:321]=.01
    rows=[]
    for bs in (8,11):
        est=h.AdaptiveRobustNoiseEstimator(window_size=16,initial_noise_power_linear=1e-4,time_constant_s=1e-4,adaptation_window_size=5)
        rows.append(h.run(x,bs,noise_estimator=est,pre_trigger_samples=0,post_signal_samples=0))
    assert rows[0]['events']==rows[1]['events']
    assert rows[0]['accepted']==rows[1]['accepted']
    assert rows[0]['reference']==pytest.approx(rows[1]['reference'],rel=1e-12)

def test_original_aligned_adaptive_and_startup_guard_cases_preserved():
    x=np.full(2048,.003,np.complex64);x[512:768]=.01
    rows=[]
    for bs in (1,16,64,2048):
        est=h.AdaptiveRobustNoiseEstimator(window_size=16,initial_noise_power_linear=1e-4,time_constant_s=1e-4,adaptation_window_size=64)
        rows.append(h.run(x,bs,noise_estimator=est))
    assert all(r['events']==rows[0]['events'] for r in rows)
    assert all(r['accepted']==1728 for r in rows)
    for bs in (64,512):
        est=h.AdaptiveRobustNoiseEstimator(window_size=256,initial_noise_power_linear=1e-4,time_constant_s=1e-6,adaptation_window_size=64)
        r=h.run(np.full(512,.1,np.complex64),bs,noise_estimator=est,window_size=256,hop_size=64,pre_trigger_samples=0)
        assert r['accepted']==0 and len(r['events'])==1

def test_synchronous_worker_delivers_registered_sink():
    r=observed('sync_sink')
    assert r['callbacks']==['e']

def test_overload_outcome_is_delivered_to_registered_sink():
    r=observed('overload_sink')
    assert r['input_drops']==1
    assert r['callbacks'].count('dropped_overload')==1

def test_repeated_shutdown_timeouts_leave_no_orphan_sentinels():
    r=observed('repeated_shutdown')
    assert r['state']=='CLOSED' and not r['thread_alive']
    assert r['unfinished_after_close']==0 and r['queued_after_close']==0

def test_async_sink_reentrancy_and_errors_are_observable():
    calls=[];done=threading.Event()
    w=h.FeatureAnalysisWorker(synchronous=False)
    def sink(sc):calls.append(len(w.processed_results));done.set()
    w.add_result_sink(sink);w.submit(h.sidecar());assert done.wait(2);w.shutdown()
    assert calls==[1] and w._queue.unfinished_tasks==0
    r=observed('throwing_sink')
    assert r['sink_errors']==1 and r['completed']==1 and r['unfinished']==0

def test_atomic_worker_admission_is_preserved():
    r=observed('controlled_worker_admission')
    assert not r['shutdown_completed_before_admission_release']
    assert r['completed']==1 and not r['errors']
    assert r['unfinished']==0 and r['state']=='CLOSED'

def test_positive_input_output_overload_stays_bounded():
    r=observed('input_output_overload')
    assert r['input_drop']==1 and r['output_drop']==2 and r['completed']==2
    assert r['history_size']==1 and r['result_queue']==1 and r['unfinished']==0

def test_filter_and_reference_domain_mismatch_is_not_silently_accepted():
    rows=observed('threshold_reference_identity')
    assert rows[0]['ON_windows']==64
    assert not rows[1]['accepted']

def test_calibration_dict_requires_complete_H_and_OFF_identity():
    rows=observed('policy_missing_identity')
    assert all(not r['accepted'] for r in rows)

def test_streaming_calibration_caps_input_before_filtering():
    r=observed('calibration_streaming_cap')
    assert sum(r['actual_samples_sent_to_filter'])<=r['cap']+r['settling']

def test_calibration_rejects_nonfinite_input_before_commit():
    rows=observed('invalid_calibration_inputs')
    assert not rows[1]['accepted']

def test_policy_requires_valid_boolean_H_and_immutable_band():
    rows=observed('direct_policy_invariants')
    assert all(not r['accepted'] for r in rows[1:])

def test_non_hop_window_count_alone_is_not_statistical_support():
    r=observed('sampling_support')
    assert not r['valid'] or any('confidence' in k or 'precision' in k or 'dependence' in k for k in r['policy_fields'])

def test_valid_contained_99_percent_width_is_not_rejected_by_peak_heuristic():
    r=observed('width_cases')['contained_gaussian']
    assert r['known_energy_outside_ROI_fraction']<.01
    assert r['features']['bandwidth.occupied_99_hz']['valid']

def test_observation_truncation_does_not_certify_physical_emitter_99_percent_width():
    r=observed('width_cases')['clipped_heavy_tails']
    assert r['known_energy_outside_observation_fraction']>.01
    assert not r['features']['bandwidth.occupied_99_hz']['valid']

def test_effective_bandwidth_SNR_configuration_changes_policy_id():
    rows=observed('emitter_policy_collisions')
    assert rows[0]['features']!=rows[1]['features']
    assert rows[0]['policy_id']!=rows[1]['policy_id']

def test_amplitude_ddof_changes_effective_policy_id():
    rows=observed('amplitude_policy_collision')
    assert rows[0]['std']!=rows[1]['std']
    assert rows[0]['policy_id']!=rows[1]['policy_id']

def test_rejected_foreign_stream_cannot_change_adopted_center():
    r=observed('center_adoption_before_validation')
    assert r['first_error']=='StreamConsistencyError'
    assert r['center_after_rejected_foreign_block'] is None
    assert r['valid_original_block_accepted'] is True

def test_gap_does_not_automatically_authorize_training_on_new_signal():
    r=observed('warmup_gap_signal')
    assert r['new_reference_after_gap_signal']==pytest.approx(r['first_noise_reference']) or r['events']

def test_physical_boundary_uncertainty_is_truthful():
    r=observed('timing_bound')
    u=r['event']['boundary_uncertainty']
    assert u is None or u>=max(r['onset_error'],r['end_error'])

def test_public_detector_sampling_rate_and_calibration_band_validation():
    rows=observed('numeric_config_gaps')
    assert all(not r['accepted'] for r in rows)

def test_real_EOF_tail_retention_preserved():
    x=np.full(69,.5,np.complex64)
    for bs in (1,16,69):
        r=h.run(x,bs,pre_trigger_samples=0,post_signal_samples=0)
        e=r['events'][0]
        assert (e['start'],e['end'])==(0,69)
        assert e['iq_sha256']==hashlib.sha256(x.tobytes()).hexdigest()

def test_long_pretrigger_retention_preserved():
    x=np.zeros(30000,np.complex64);x[12500:12700]=.5
    events=[]
    for bs in (64,30000):
        r=h.run(x,bs,ring_capacity=12000,pre_trigger_samples=10000,max_event_samples=20000)
        e=r['events'][0];events.append(e)
        assert e['start']==2503
        assert e['iq_sha256']==hashlib.sha256(x[e['start']:e['end']].tobytes()).hexdigest()
    assert events[0]==events[1]

def test_invalid_detector_noise_rejected_before_filter_state_mutation():
    d=h.EnergyDetector(window_size=16,hop_size=8,mode='configured_band',monitored_band_hz=(90000,110000))
    phase=d._phase_accumulator;zi=d._sos_zi.copy()
    ref=h.NoiseEstimate(h.S.stream_id,0,float('nan'),valid=False,reason='missing calibration')
    with pytest.raises(UncalibratedNoiseError):d.process_windows(h.b(np.ones(25)),ref)
    assert d._phase_accumulator==phase
    np.testing.assert_array_equal(d._sos_zi,zi)

def test_stable_narrow_filter_delay_preserved():
    for width,want in ((1000,832),(100,8318),(10,83178)):
        d=h.EnergyDetector(mode='configured_band',monitored_band_hz=(100000-width/2,100000+width/2))
        assert d.filter_delay_samples==pytest.approx(want,abs=2)

def test_analytic_single_block_actual_N_preserved():
    d=h.EnergyDetector(window_size=256,hop_size=64)
    ref=h.NoiseEstimate(h.S.stream_id,0,0)
    m=d.process(h.b(np.ones(1)),ref)
    assert 10**(m.on_threshold_dbfs/10)==pytest.approx(9.210340371976184)

def test_optional_incomplete_analysis_is_degraded_and_contiguous():
    pipe=h.p(pre_trigger_samples=0,post_signal_samples=0)
    pipe.process_block(h.b(np.full(69,.5)))
    sc=pipe.flush()[0];pipe.shutdown()
    w=h.FeatureAnalysisWorker(synchronous=True,skip_incomplete_quality=False)
    result=w.submit(sc);w.shutdown()
    assert result.feature_status=='degraded'
    assert result.analysis_result.capture_quality.is_continuous
    assert not result.analysis_result.capture_quality.complete

def test_public_INVALID_reference_does_not_leave_partly_committed_OPEN_pipeline():
    class InvalidEstimator:
        state=NoiseLifecycleState.INVALID
        def get_estimate(self,*a):return h.NoiseEstimate(h.S.stream_id,0,float('nan'),valid=False,reason='receiver reference invalid')
        def reset(self):pass
        def freeze(self,*a):pass
        def update_samples(self,*a):pass
    pipe=h.p(noise_estimator=InvalidEstimator())
    before=pipe.ring_buffer.available_range()
    with pytest.raises(UncalibratedNoiseError):pipe.process_block(h.b(np.ones(32)))
    after=pipe.ring_buffer.available_range()
    assert pipe.state.value=='FAULTED' or after==before
    pipe.shutdown()

def test_configured_band_gaussian_warmup_uses_matching_measurement_domain():
    rows=observed('configured_gaussian_warmup')
    assert len(rows[0]['events'])==1
    assert len(rows[1]['events'])==1
    assert rows[1]['reference_after_prefix']==pytest.approx(rows[0]['matched_policy_reference'],rel=.5)

def test_gaussian_recalibration_cannot_enter_permanent_zero_budget_training():
    r=observed('gaussian_recalibration_no_budget')
    assert r['state']!='CALIBRATING' or r['target']>0

def test_supported_policy_binding_checks_and_actual_N_rejection_preserved():
    rng=np.random.default_rng(33)
    x=((rng.normal(size=20000)+1j*rng.normal(size=20000))*.01).astype(np.complex64)
    c=h.EmpiricalNoiseCalibrator(window_size=16,hop_size=16,pfa_on=.01,pfa_off=.1)
    q=c.create_policy(x)
    for changes in ({'sample_rate_hz':2e6},{'window_size':32},{'hop_size':8},{'pfa_on':.001},{'monitored_band_hz':(0,1000),'filter_sos_hash':'different'}):
        args=dict(window_size=16,hop_size=16,pfa_on=.01,pfa_off=.1,threshold_policy=q);args.update(changes)
        if 'filter_sos_hash' in args:
            args.pop('filter_sos_hash');args['mode']='configured_band'
        with pytest.raises(PolicyCompatibilityError):h.EnergyDetector(**args)
    d=h.EnergyDetector(window_size=16,hop_size=16,pfa_on=.01,pfa_off=.1,threshold_policy=q)
    with pytest.raises(Exception,match='cannot be applied to block length'):d.process(h.b(np.ones(1)),h.NoiseEstimate(h.S.stream_id,0,-37))

def test_contiguous_received_prefix_remains_continuous_after_known_external_gap():
    pipe=h.p(pre_trigger_samples=0,post_signal_samples=0)
    received=np.full(69,.5,np.complex64)
    pipe.process_block(h.b(received))
    sc=pipe.process_block(h.b(np.zeros(16,np.complex64),1,100,dropped_samples_before=31))[0]
    pipe.shutdown()
    np.testing.assert_array_equal(sc.event.iq,received)
    assert (sc.event.event_start_sample,sc.event.event_end_sample)==(0,69)
    assert sc.known_lost_samples==31 and not sc.event.complete
    w=h.FeatureAnalysisWorker(synchronous=True,skip_incomplete_quality=False)
    result=w.submit(sc);w.shutdown()
    assert result.analysis_result.capture_quality.is_continuous

