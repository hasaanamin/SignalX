"""Fresh live acceptance checks; no saved JSON is read as a runtime answer."""
from functools import lru_cache
from pathlib import Path
import dataclasses,math
import numpy as np
import pytest
import g_revision_probes as f

@lru_cache(None)
def observed(name):return getattr(f,name)()

def test_active_source():
    import signal_processing
    assert Path(signal_processing.__file__).resolve().is_relative_to(f.SRC)

@pytest.mark.parametrize('case',range(3))
def test_positive_excess_snr_matches_physical_ratio(case):
    r=observed('scaled_snr')['scales'][case]
    assert r['snr']==pytest.approx(r['physical_snr'],abs=1e-6)

def test_zero_signal_does_not_report_positive_snr():
    r=observed('scaled_snr')
    assert r['zero_signal_below_off']
    assert r['zero_signal_snr']<=0

@pytest.mark.parametrize('case',range(4))
def test_large_dynamic_range_window_power_and_decisions(case):
    r=observed('dynamic_range_partitions');row=r['rows'][case]
    np.testing.assert_allclose(row['powers'],r['oracle_powers'],rtol=1e-9,atol=0)
    assert row['on']==r['rows'][0]['on']

@pytest.mark.parametrize('case',range(3))
def test_calibration_dynamic_range_matches_direct_window_oracle(case):
    r=observed('training_dynamic_range');row=r['rows'][case]
    assert row['maximum_quiet_relative_error']<1e-9
    assert row['t_on']==pytest.approx(r['oracle_on'],rel=1e-9,abs=0)
    assert row['t_off']==pytest.approx(r['oracle_off'],rel=1e-9,abs=0)

def test_synchronous_shutdown_waits_for_admitted_analysis_or_times_out():
    r=observed('sync_shutdown_inflight')['before_release']
    assert r['submit_thread_alive']
    assert r['shutdown_error']=='TimeoutError' and r['state']=='CLOSING'

def test_no_late_publication_after_successful_shutdown():
    r=observed('sync_shutdown_inflight')
    if r['before_release']['state']=='CLOSED' and r['before_release']['shutdown_error'] is None:
        assert not r['after_callbacks'] and r['after_outputs']==0

def test_frozen_calibration_is_explicit_or_can_resume():
    r=observed('frozen_warmup')
    # A documented preflight rejection while frozen is also a safe contract.
    # Silent partial filter advancement followed by failed resume is not.
    assert not r['errors'] or (r['errors'][0]['phase']=='frozen'
                              and r['before']['detector_input_end'] is None
                              and r['before']['ring_range']==(0,0))

def test_valid_backend_result_is_bound_to_submitted_event():
    r=observed('backend_event_mismatch')
    assert r['status']=='failed' or (r['vector_id']==r['submitted_id']
                                    and r['reported_mean_amplitude']==pytest.approx(r['submitted_mean_amplitude'],rel=1e-6))

def test_polling_reset_reservation_and_delivery_positive():
    r=observed('polling_reset_positive')
    assert r['result_status']=='ok' and r['reserved']==0
    assert r['current_generation']==r['result_generation']==1

def test_consistent_timestamp_controls():
    for r in observed('timestamp_consistency'):
        if not r['jump_ns']:
            assert r['error'] is None and r['events']
            assert all(e['timestamp']==e['first_anchor_expected'] for e in r['events'])

def test_inconsistent_timestamp_is_rejected_or_not_extrapolated_across_jump():
    for r in observed('timestamp_consistency'):
        if r['jump_ns']:
            assert r['error'] is not None or all(e['timestamp'] is None or e['timestamp']==e['first_anchor_expected'] for e in r['events'])

def test_async_overload_does_not_run_slow_sink_on_acquisition_caller():
    r=observed('async_overload_sink_blocking')
    assert not r['before_release']['rejected_sink_on_submitter']

def test_async_overload_recovery_positive():
    r=observed('async_overload_sink_blocking')
    assert r['first_status']==r['second_status']=='pending'
    assert r['rejected_status']=='dropped_overload'
    assert r['completed']==2 and r['unfinished']==0 and r['sink_errors']==0

@pytest.mark.parametrize('power', [0.,1e-20,1e-4,1e20])
def test_trusted_measurements_reconstruct_through_public_contract(power):
    d=f.EnergyDetector(window_size=16,hop_size=8)
    m=d.process(f.block(np.full(16,math.sqrt(power))),f.NoiseEstimate(f.S.stream_id,0,-40))
    rebuilt=f.DetectionMeasurement(**dataclasses.asdict(m))
    assert rebuilt==m
