"""Fresh requirements beyond the old 37 checks; actual runtime, no saved answers."""
from functools import lru_cache
import dataclasses,math
from pathlib import Path
import numpy as np
import pytest
import new_mechanism_probes as n
import signal_processing2_deep_probes as h

@lru_cache(None)
def observed(name):return getattr(n,name)()

def test_imports_target_current_source():
    import signal_processing
    assert Path(signal_processing.__file__).resolve().is_relative_to(n.SRC)

@pytest.mark.parametrize('case',[0,1,2])
def test_complete_nonmultiple_window_is_evaluated_at_EOF(case):
    r=observed('nonmultiple_eof')[case]
    assert r['direct_windows']==1 and r['direct_ON']==1
    assert len(r['pipeline_events'])==1

def test_aligned_complete_window_positive_path():
    r=observed('nonmultiple_eof')[3]
    assert len(r['pipeline_events'])==1

def test_no_future_signal_admitted_by_earlier_quiet_window():
    rows=observed('future_admission')
    assert not any(a['active_samples'] for a in rows[1]['admitted'])

def test_nonmultiple_adaptation_preserves_positive_signal_detection():
    rows=observed('future_admission')
    assert len(rows[0]['events'])==1
    assert len(rows[1]['events'])==1

@pytest.mark.parametrize('case',[0,1])
def test_age_advances_without_successful_update(case):
    r=observed('quiet_without_successful_update')[case]
    assert r['accepted']==0
    assert r['error'] or r['age']>=r['samples']

def test_age_includes_post_update_residual():
    r=observed('quiet_without_successful_update')[2]
    assert r['accepted']==20
    assert r['age']==4

@pytest.mark.parametrize('case',[1,2])
def test_configured_warmup_processes_each_input_once(case):
    r=observed('configured_warmup_partition')[case]
    assert r['processed_by_measurement_path']==r['input_samples']

def test_configured_warmup_power_partition_invariance():
    assert all(r['maximum_power_difference_db']<1e-6 for r in observed('configured_warmup_partition'))

def test_future_suffix_cannot_change_earlier_window():
    rows=observed('configured_warmup_future_poison')
    assert rows[0]['first_measurement']==pytest.approx(rows[1]['first_measurement'],abs=1e-6)

def test_warmup_uses_filtered_units_positive():
    rows=observed('configured_warmup_partition')
    assert all(1e-6<r['reference']<3e-6 and r['trained']==4096 for r in rows)

def test_filtered_default_model_is_not_permitted_with_grossly_wrong_tail():
    r=observed('filtered_gamma_false_alarm')[0]
    assert r['rate']<=10*r['requested_pfa']

def test_matched_empirical_heldout_positive_sanity():
    r=observed('filtered_gamma_false_alarm')[1]
    assert r['rate']<=10*r['requested_pfa']

def test_raw_reference_rejected_even_when_within_5dB_of_filtered_mean():
    rows=observed('wide_band_raw_reference')
    assert rows[0]['accepted']
    assert not rows[1]['accepted']

def test_pipeline_policy_failure_is_not_partly_committed_OPEN():
    r=observed('pipeline_policy_failure_commit')
    assert r['error']=='PolicyCompatibilityError'
    assert r['state']=='FAULTED' or r['ring_before']==r['ring_after']

@pytest.mark.parametrize('case',[1,2])
def test_ROI_not_wider_than_observation_cannot_certify_unknown_heavy_tails(case):
    r=observed('bandwidth_roi_bypass')[case]
    assert not r['features']['bandwidth.occupied_99_hz']['valid']

def test_heavy_tail_unknown_support_rejected_without_ROI_positive():
    r=observed('bandwidth_roi_bypass')[0]
    assert not r['features']['bandwidth.occupied_99_hz']['valid']
    assert r['features']['bandwidth.three_db_hz']['valid']

def test_edge_clipping_configuration_changes_policy_identity():
    rows=observed('bandwidth_policy_collision')
    assert rows[0]['features']!=rows[1]['features']
    assert rows[0]['policy_id']!=rows[1]['policy_id']

def test_minimal_tail_resolution_is_not_validated_reliability():
    r=observed('support_without_reliability')
    assert r['training_windows']==100
    assert not r['supported_pfa']

def test_calibration_shape_error_does_not_commit_counts():
    r=observed('calibration_lifecycle_and_shape')[0]
    assert not r['accepted']
    assert r['samples']==0 and r['windows']==0

def test_finalization_is_terminal_under_requested_contract():
    r=observed('calibration_lifecycle_and_shape')[1]
    assert not r['accepted'] or r['second_samples']==r['first_samples']

@pytest.mark.parametrize('case',[1,2])
def test_signed_baseband_band_supported_consistently(case):
    rows=observed('signed_band_contracts')
    assert rows[0]['accepted']
    assert rows[case]['accepted']

@pytest.mark.parametrize('case',[0,1,3])
def test_direct_buffer_invalid_inputs_rejected(case):
    assert not observed('direct_buffer_contracts')[case]['accepted']

def test_shutdown_cancel_retry_remains_closable():
    r=observed('shutdown_cancel_retry')
    assert r['state']=='CLOSED' and not r['thread_alive'] and r['unfinished']==0

def test_reset_blocks_old_result_delivery_that_has_not_started():
    r=observed('reset_callback_race')
    assert r['generation_after_reset']>r['generation_before']
    assert r['callbacks']==[]

def test_analysis_quality_preserves_loss_and_termination_distinctions():
    rows=observed('known_unknown_quality')
    assert len({repr(r['analysis_quality']) for r in rows})==3

def test_contiguous_prefix_quality_positive():
    assert all(r['analysis_quality']['is_continuous'] and not r['analysis_quality']['complete'] for r in observed('known_unknown_quality'))

def test_calibration_preprocessing_allocation_respects_consumed_budget():
    r=observed('calibration_allocation_cap')
    assert max(r['finite_validation_input_sizes'])<=r['budget_plus_settling']

def test_full_calibration_identity_requires_sampling_rate():
    c=h.EmpiricalNoiseCalibrator(window_size=16,hop_size=16,pfa_on=.01,pfa_off=.1)
    result=c.calibrate(n.noise(1600),settling_samples=0);del result['sample_rate_hz']
    with pytest.raises(Exception):c.create_policy(result,sample_rate_hz=2e6)

def test_linear_db_consistency_is_numeric_not_one_dB():
    c=h.EmpiricalNoiseCalibrator(window_size=16,hop_size=16,pfa_on=.01,pfa_off=.1)
    q=c.create_policy(n.noise(1600),settling_samples=0)
    with pytest.raises(Exception):dataclasses.replace(q,t_on_dbfs=q.t_on_dbfs+.9,t_off_dbfs=q.t_off_dbfs+.9)

def test_policy_persists_ddof_and_versions_used_in_hash():
    rows=observed('bandwidth_policy_collision')
    assert 'amplitude_ddof' in rows[0]['policy_fields']
    assert 'extractor_versions' in rows[0]['policy_fields']

def test_gap_during_warmup_does_not_authorize_signal_as_noise():
    r=observed('gap_during_training')
    assert r['reference_after_signal']!=pytest.approx(.25)
