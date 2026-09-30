"""Expanded runtime requirements for examples(2).zip; fresh observations only."""
from functools import lru_cache
from pathlib import Path
import pytest
import fresh_runtime_probes as f
import new_mechanism_probes as previous

@lru_cache(None)
def observed(name):return getattr(f,name)()

def test_current_source_import_target():
    import signal_processing
    assert Path(signal_processing.__file__).resolve().is_relative_to(f.SRC)

def test_quiet_stream_does_not_retain_all_received_IQ():
    rows=observed('retained_history')
    assert rows[-1]['retained_iq_samples']<=rows[-1]['ring_capacity']+256

def test_quiet_stream_state_memory_stabilizes():
    rows=observed('retained_history')
    assert rows[-1]['traced_current_bytes']-rows[0]['traced_current_bytes']<65536

def test_origin_zero_positive_adaptation():
    r=observed('origin_causality')[0]
    assert len(r['events'])==1
    assert all(a['active_samples']==0 for a in r['admitted'])

def test_shifted_origin_preserves_positive_detection():
    rows=observed('origin_causality')
    assert len(rows[1]['events'])==len(rows[0]['events'])==1

def test_shifted_origin_never_admits_future_active_IQ():
    assert all(a['active_samples']==0 for a in observed('origin_causality')[1]['admitted'])

def test_complete_residual_window_before_gap_is_evaluated():
    rows=observed('gap_complete_window')
    assert len(rows[0]['events'])==1
    assert len(rows[1]['events'])==1

def test_EOF_nonmultiple_window_positive():
    assert len(observed('gap_complete_window')[0]['events'])==1

def test_live_startup_reference_ages_before_EOF():
    r=observed('startup_age_before_flush')['before_flush']
    assert r['error'] or r['age']>=192

def test_midblock_policy_error_is_faulted_or_atomic():
    r=observed('midblock_policy_failure')
    assert r['error']=='PolicyCompatibilityError'
    assert r['state']=='FAULTED' or r['ring_after']==r['ring_before']

def test_same_domain_training_variability_is_not_a_domain_mismatch():
    r=observed('matched_reference_variability')
    assert all(c['error']!='PolicyCompatibilityError' for c in r['cases'])

@pytest.mark.parametrize('case',[0,2])
def test_configured_Gamma_target_not_excluded_by_independent_tail_validation(case):
    r=observed('effective_gamma_tails')[case]
    assert r['95pct_binomial_interval'][0]<=r['target']

def test_configured_single_sample_threshold_uses_actual_N():
    r=observed('configured_actual_N_and_large_N')[0]
    assert r['accepted']
    assert r['unit_threshold']==pytest.approx(r['correct_single_sample_exponential_threshold'],rel=1e-12)

def test_configured_valid_large_N_constructs():
    assert observed('configured_actual_N_and_large_N')[1]['accepted']

def test_reset_suppresses_subsequent_old_generation_sink():
    r=observed('callback_reset_between_sinks')
    assert all(c['sink']!='second' for c in r['callbacks'])

def test_worker_preserves_capture_continuity_epoch():
    r=observed('overwritten_capture_epoch')
    assert r['result_epoch']==r['input_epoch']

def test_cancel_retry_positive_liveness_with_longer_deadline():
    r=observed('cancel_second_call_liveness')
    assert r['job_processed'] and r['state']=='CLOSED' and not r['thread_alive'] and r['unfinished']==0

def test_one_extra_training_window_is_not_confidence_support():
    r=observed('support_101_windows')
    assert r['windows']==101
    assert not r['supported']

def test_failed_finalize_does_not_seal_incomplete_training():
    r=observed('finalize_failure_recovery')
    assert not r['sealed'] or r['samples']>=16

def test_one_bin_outside_ROI_does_not_prove_unobserved_heavy_tails_absent():
    r=observed('bandwidth_one_extra_bin')
    assert r['unobserved_known_energy_fraction']>.01
    width = r['features']['bandwidth.occupied_99_hz']
    # E09 permits a useful observed-excess result if it is explicitly identified.
    # Physical-emitter support is tested separately in test_examples2_repairs.
    assert not width['valid'] or width['interpretation'] == 'observed_excess_within_roi'

def test_supported_half_power_positive_despite_unknown_outer_tail():
    assert observed('bandwidth_one_extra_bin')['features']['bandwidth.three_db_hz']['valid']

def test_auto_noise_setting_changes_effective_policy_ID():
    rows=observed('feature_policy_remaining_collisions')[1]['results']
    assert rows[0]['features']!=rows[1]['features']
    assert rows[0]['policy_id']!=rows[1]['policy_id']

def test_policy_versions_record_is_immutable():
    assert not observed('mutable_versions_record')['mutation_allowed']

def test_threshold_linear_db_consistency_is_numerical():
    assert not observed('half_db_policy_contradiction')['accepted']

def test_analysis_quality_represents_calibration_trust():
    rows=observed('quality_calibration_trust')
    assert rows[0]['status']=='ok' and rows[1]['status']=='degraded'
    assert rows[0]['quality']!=rows[1]['quality']

def test_direct_configured_detector_rejects_actual_input_Fs_mismatch():
    assert not observed('wrong_input_Fs')['accepted']

@pytest.mark.parametrize('case',[0,1,2])
def test_buffer_reset_rejects_bad_coordinates(case):
    assert not observed('other_buffer_entrypoints')[case]['accepted']

def test_buffer_failed_reset_does_not_advance_generation():
    r=observed('other_buffer_entrypoints')[3]
    assert not r['accepted'] and r['generation_after']==r['generation_before']

def test_buffer_range_does_not_truncate_fractional_coordinates():
    assert not observed('other_buffer_entrypoints')[4]['accepted']

def test_old_warmup_prefix_causality_is_preserved():
    rows=previous.configured_warmup_future_poison()
    assert rows[0]['first_measurement']==pytest.approx(rows[1]['first_measurement'],abs=1e-6)

def test_old_ROI_equal_to_observation_rejection_is_preserved():
    rows=previous.bandwidth_roi_bypass()
    assert not rows[1]['features']['bandwidth.occupied_99_hz']['valid']

def test_original_pre_callback_reset_barrier_is_preserved():
    assert previous.reset_callback_race()['callbacks']==[]

def test_old_direct_buffer_guards_are_preserved():
    rows=previous.direct_buffer_contracts()
    assert all(not r['accepted'] for r in rows)

def test_empirical_tail_minimal_support_boundary_rejection_is_preserved():
    assert not previous.support_without_reliability()['supported_pfa']

def test_shipped_gamma_reduction_positive_sanity():
    r=observed('effective_gamma_tails')[0]
    assert r['rate']<.001

@pytest.fixture(scope='session',autouse=True)
def write_fresh_observation_log():
    yield
    import json
    names=['retained_history','origin_causality','gap_complete_window','startup_age_before_flush','midblock_policy_failure','matched_reference_variability','effective_gamma_tails','configured_actual_N_and_large_N','callback_reset_between_sinks','overwritten_capture_epoch','cancel_second_call_liveness','support_101_windows','finalize_failure_recovery','bandwidth_one_extra_bin','feature_policy_remaining_collisions','mutable_versions_record','half_db_policy_contradiction','quality_calibration_trust','wrong_input_Fs','other_buffer_entrypoints']
    out={'source':str(f.SRC),'observations':{name:observed(name) for name in names}}
    Path(__file__).with_name('fresh_acceptance_observations.json').write_text(json.dumps(out,indent=2,allow_nan=True))
