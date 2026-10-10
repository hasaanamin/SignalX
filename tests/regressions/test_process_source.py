"""Low-rate accounting controls, not substitutes for the 1 MS/s timing gate."""
import pytest
from benchmarks.sustained_evaluation import bounded_acquisition


@pytest.mark.parametrize('failure', [False,True])
def test_source_terminal_accounting_and_cleanup(failure):
    def sink(sc):
        if failure: raise RuntimeError('intentional required sink failure')
    result=bounded_acquisition(.1,sample_rate_hz=10000,block_size=128,features=True,
        analysis_backend='process',producer_backend='process',result_sink=sink)
    assert result['scheduled_samples']==1000
    assert result['sample_accounting_balanced'] and result['recovered']
    assert result['acquisition_unfinished']==result['worker_unfinished']==0
    assert result['delivery']['unresolved_identities']==result['delivery']['duplicate_results']==0
    if failure:
        assert result['worker_sink_errors'] > 0
        assert not result['analysis_delivery_passed'] and not result['passed']
