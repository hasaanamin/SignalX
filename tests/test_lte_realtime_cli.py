"""Public scheduled CLI uses genuine decoding and enforces a failing deadline."""
import json
import sys

import pytest

from signal_processing.lte.cli import main
from tests.lte_pbch_fixture import waveform


def test_public_process_source_cli_preserves_origin_decodes_and_enforces_deadline(tmp_path,capsys):
    path=tmp_path/'encoded.cf32'
    iq=waveform(42,ports=2,rate=1_920_000.,frames=2,snr_db=20.)
    iq.tofile(path)
    status=main(['--input',str(path),'--sample-rate','1920000','--block-size','65536',
        '--origin',str(10**12),'--scheduled','--source-backend','process','--sync-backend','numpy',
        '--require-mib','--enforce-acquisition','--max-latency-ms','0.000001'])
    records=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    summary=records[-1]
    assert status==1 and summary['acquisition_passed'] and not summary['latency_passed']
    assert summary['processed_samples']==len(iq) and summary['queue_dropped_samples']==0
    assert summary['decoded_mibs']>0 and summary['maximum_latency_ms']>0
    mibs=[r for r in records if r['type']=='mib']
    assert mibs and all(r['pci']==42 for r in mibs)
    assert summary['receiver']['received_input_end']==10**12+len(iq)
    assert summary['source_backend']=='process' and summary['error'] is None
    assert summary['dsp_scheduling'] == 'priority'
    assert summary['dsp_wait'] == 'spin'
    assert summary['thread_relative_priority'] == (2 if sys.platform == 'win32' else None)
    timing = summary['maximum_dsp_latency_block']
    assert timing['latency_ms'] == summary['maximum_dsp_latency_ms']
    assert timing['processing_wall_ms'] > 0 and timing['processing_cpu_ms'] >= 0
    assert timing['before_processing_ms'] >= timing['source_lateness_ms'] >= 0
    assert len(summary['slow_dsp_blocks']) <= 32
    assert not summary['realtime_scheduling']['requested']


@pytest.mark.parametrize('extra',[
    ['--max-latency-ms','10'],['--dsp-wait','spin'],
    ['--scheduled','--source-backend','thread','--dsp-wait','spin']])
def test_deadline_cli_rejects_unpaced_claim_before_starting_input(tmp_path,extra):
    with pytest.raises(SystemExit) as exc:
        main(['--input',str(tmp_path/'not-opened.cf32'),*extra])
    assert exc.value.code==2


@pytest.mark.parametrize('dsp_scheduling',['priority','mmcss'])
@pytest.mark.parametrize('dsp_wait',['spin','block'])
def test_empty_output_still_accounts_for_all_iq_and_enforces_block_latency(tmp_path,capsys,dsp_scheduling,dsp_wait):
    import numpy as np
    path = tmp_path/'before-integration.cf32'
    # Shorter than the low-SNR integration dwell: no synchronization result can
    # be eligible, but every received sample and its latency must still count.
    np.full(16384,.01+.02j,np.complex64).tofile(path)
    status = main(['--input',str(path),'--sample-rate','1920000','--profile','low_snr',
        '--scheduled','--decode-mib','--enforce-acquisition','--max-latency-ms','0.000001',
        '--dsp-scheduling',dsp_scheduling,'--dsp-wait',dsp_wait])
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 1 and records[0]['type'] == 'summary'
    summary = records[0]
    assert status == 1 and summary['acquisition_passed'] and not summary['latency_passed']
    assert summary['processed_samples'] == summary['total_samples'] == 16384
    assert summary['latency_ms_p99'] is not None and summary['maximum_latency_ms'] > 0
    assert summary['publication']['submitted'] == summary['publication']['published'] == 0
    assert summary['publication']['outstanding'] == 0 and not summary['publication']['writer_alive']
    assert summary['dsp_scheduling'] == dsp_scheduling
    assert summary['dsp_wait'] == dsp_wait
    assert summary['realtime_scheduling']['requested'] == (dsp_scheduling == 'mmcss')
