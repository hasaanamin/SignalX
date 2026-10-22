"""The audit harness uses real independently coded IQ and real production DSP."""
import numpy as np

from benchmarks.audit_pipeline_runtime import prepare, lte


def test_streamed_fixture_and_actual_offline_lte_receiver(tmp_path):
    path = tmp_path/"encoded.cf32"
    info = prepare(path, .08, snr=20.)
    assert info["samples"] == 160000
    assert path.stat().st_size == 160000*8
    assert np.all(np.isfinite(np.fromfile(path, "<c8")))
    result = lte(path, "low_snr", scheduled=False)
    assert result["sample_accounting_balanced"]
    assert result["consumed_samples"] == result["total_samples"] == 160000
    assert result["results"]["verified_mibs"] == result["expected_pbch_frames"] == 8
    assert result["protocol_results_correct"] and result["passed"]
    assert not result['realtime_decoding_passed']  # unpaced is never timing qualification
    assert result['stage_timing']['pbch_decode']['calls'] >= 8
    assert result["runner"]["outstanding_deliveries"] == 0
    assert not result["errors"]


def test_native_fixture_scoring_equivalence_and_no_hidden_rate_conversion(tmp_path):
    path = tmp_path/'native.cf32'
    info = prepare(path, .08, snr=20., sample_rate_hz=1_920_000.)
    assert info['samples'] == 153600 and path.stat().st_size == 153600*8
    results = [lte(path, 'low_snr', scheduled=False, sample_rate_hz=1_920_000., scoring_backend=mode)
               for mode in ('inline', 'precomputed')]
    for result in results:
        assert result['consumed_samples'] == result['total_samples'] == 153600
        assert result['receiver']['retained_resampler_samples'] == 0
        assert result['results']['verified_mibs'] == 8
        assert result['passed'] and not result['errors']
        assert not result['realtime_decoding_passed']
        assert result['sample_accounting_balanced']
    assert results[0]['results'] == results[1]['results']


def test_adaptive_recorded_source_uses_real_demand_features_and_accounts_every_sample(tmp_path):
    from benchmarks.replay_adaptive_runtime import prepare as prepare_adaptive,run
    path=tmp_path/'adaptive.cf32'
    assert prepare_adaptive(path,.008)['samples']==8000
    # The scheduled workload starts IN a burst; its first capture correctly
    # lacks pre-trigger IQ and is skipped by the default quality gate. For this
    # genuine positive feature-analysis control provide a received quiet lead.
    # The original scheduled benchmark dataset/workload is not changed.
    np.r_[np.full(1024,.003,np.complex64),np.fromfile(path,'<c8')].tofile(path)
    result=run(path,scheduled=False)
    assert result['sample_accounting_balanced'] and result['acquisition_passed']
    assert result['consumed_samples']==9024 and result['raw_captures']>=2
    assert result['requested_captures']>0
    assert result['outcomes']['ok']>0
    assert result['delivery_passed'] and result['recovered']
    assert not result['sub10ms_lossless_passed']  # Offline never certifies scheduling.
    assert result['unfinished']==result['unresolved']==result['duplicate_results']==0
