"""New acceptance cases for the repaired archive. All observations are computed live."""
from functools import lru_cache
from pathlib import Path
import pytest
import f_revision_probes as f

@lru_cache(None)
def observed(name):return getattr(f,name)()

def test_active_import_target():
    import signal_processing
    assert Path(signal_processing.__file__).resolve().is_relative_to(f.SRC)

def test_gap_feature_delivery_synchronous_control():
    r=observed('async_gap_delivery')[0]
    assert len(r['returned'])==len(r['callbacks'])==1
    assert r['returned'][0]['status']=='degraded'

def test_gap_capture_receives_asynchronous_terminal_result():
    r=observed('async_gap_delivery')[1]
    assert len(r['returned'])==1
    # Explicit terminal cancellation is also legitimate; permanent pending is not.
    assert r['callbacks'] or any(sc['status']!='pending' for sc in r['returned'])

def test_midblock_fault_keeps_completed_prefix_capture_recoverable():
    r=observed('partial_block_emissions')
    assert r['error']=='PolicyCompatibilityError' and r['state']=='FAULTED'
    assert len(r['internally_dispatched'])==1
    assert r['returned'] or r['recovered'] or r['worker_stored']

def test_original_held_out_evidence_is_supported():
    r=observed('held_out_evidence_binding')
    assert r['original_supported'] and r['original_count']==0

def test_threshold_change_invalidates_held_out_evidence():
    r=observed('held_out_evidence_binding')
    assert r['actual_changed_exceedances']/r['validation_windows']>.9
    assert not r['detector_accepts_changed_evidence']

def test_certified_analytic_policy_rejects_or_preserves_fixed_thresholds():
    r=observed('analytic_certified_scale')
    assert r['supported']
    assert r.get('rejected') or r['runtime_on']==pytest.approx(r['calibrated_on'],rel=1e-12)

def test_detector_does_not_construct_a_window_across_declared_loss():
    r=observed('detector_discontinuity')
    assert r['first_count']==0
    assert not r['second']

def test_conversion_floor_does_not_change_on_decision_for_equal_snr():
    r=observed('runtime_conversion_floor')
    assert r[0]['on']==r[1]['on']

def test_failed_result_sink_is_not_an_end_to_end_benchmark_pass():
    r=observed('benchmark_failing_sink')
    assert r['worker_sink_errors']>0
    assert not r['passed']

@pytest.mark.parametrize('partition_index',range(5))
def test_settling_warmup_absolute_origin_and_partition_oracle(partition_index):
    r=observed('gaussian_warmup_partition_oracle')[partition_index]
    assert r['same_as_reference'] and r['exact_iq'] and r['domain_matches']
    assert r['training_samples']==19
    assert r['trained_power']==pytest.approx(9e-6,rel=1e-7)
    assert len(r['events'])==1

@pytest.mark.parametrize('case',range(5))
def test_configured_gaussian_bound_matches_covariance_oracle(case):
    r=observed('continuous_filtered_covariance_oracle')[case]
    assert r['threshold_model']=='stationary_gaussian_quadratic_bound'
    assert r['unit_on']>r['unit_off']>0
    if 'neff_relative_error' in r:
        assert r['neff_relative_error']<1e-10
        assert r['computed_at_least_exact_bound']
    else: assert 1-1e-6<=r['effective_n']<=r['n']
