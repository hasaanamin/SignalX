"""Scheduled LTE on/off transitions; raw acquisition and real BCH block coverage."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from fractions import Fraction


def prepare(positive,path):
    import numpy as np
    total = positive.stat().st_size//8
    if positive.stat().st_size % 8 or total % 19200 or not 0 < total <= 600*1_920_000:
        raise ValueError('Expected the finite native-rate encoded reference fixture')
    rng = np.random.default_rng(20261020)
    digest = hashlib.sha256()
    with positive.open('rb') as source,path.open('xb') as destination:
        for start in range(0,total,16384):
            count = min(16384,total-start)
            samples = np.frombuffer(source.read(count*8),'<c8').copy()
            muted = ((np.arange(start,start+count)//3_840_000) % 2) == 1
            noise = (rng.normal(size=count)+1j*rng.normal(size=count)).astype('<c8')
            samples[muted] = noise[muted]
            payload = samples.tobytes()
            destination.write(payload)
            digest.update(payload)
    return dict(path=str(path),samples=total,sha256=digest.hexdigest(),
                positive_input=str(positive),scope='Two seconds encoded LTE / two seconds independent noise, continuously positioned')


def run(path,log,pin_primary_host=False,priority_class='normal',dsp_cpus=None):
    from tests.lte_pbch_fixture import frame_fixture
    frames = path.stat().st_size//(8*19200)
    expected = {frame//4 for frame in range(0,frames,4) if (frame//200) % 2 == 0}
    command = [sys.executable,'-m','signal_processing.lte.cli','--input',str(path.resolve()),
        '--sample-rate','1920000','--profile','low_snr','--scheduled','--enforce-acquisition',
        '--require-mib','--decode-cadence','bch_block','--max-latency-ms','100',
        '--priority-class',priority_class]
    if pin_primary_host:
        command += ['--dsp-cpus','0','1','--source-cpus','2','3']
    if dsp_cpus is not None:
        command += ['--dsp-cpus',*map(str,dsp_cpus)]
    with log.open('x',encoding='utf8') as output:
        code = subprocess.run(command,stdout=output,stderr=subprocess.STDOUT).returncode
    summary,seen,failures = {},set(),[]
    with log.open(encoding='utf8') as output:
        for line in output:
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if item.get('type') == 'summary':
                summary = item
            elif item.get('type') == 'mib':
                coordinate = item['input_frame_start']
                position = Fraction(coordinate['numerator'],coordinate['denominator'])-431
                frame = round(position/19200)
                payload = list(map(int,frame_fixture(4,2,frame)[0]))
                valid = (frame//4 in expected and abs(position-frame*19200) <= 2
                    and item['pci'] == 42 and item['antenna_ports'] == 2
                    and item['frame_quarter'] == frame % 4 and item['crc_valid']
                    and item['mib']['payload_bits'] == payload)
                if valid:
                    seen.add(frame//4)
                else:
                    failures.append(dict(frame=frame,pci=item['pci'],position=float(position)))
            elif item.get('type') == 'observation' and item['state'] == 'SYNCHRONIZED' and item['pci'] != 42:
                failures.append(dict(wrong_confirmed_pci=item['pci']))
    summary.update(command=command,exit_code=code,log=str(log),
        expected_active_bch_blocks=len(expected),verified_active_bch_blocks=len(seen),
        missing_bch_blocks=sorted(expected-seen),validation_failures=failures)
    summary['passed'] = bool(code == 0 and summary.get('acquisition_passed')
        and summary.get('latency_passed') and not failures and seen == expected
        and summary.get('latency_ms_p99') is not None and summary['latency_ms_p99'] <= 50)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--positive-input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pin-primary-host',action='store_true')
    parser.add_argument('--priority-class',choices=('normal','above-normal','high'),default='normal')
    parser.add_argument('--dsp-cpus',type=int,nargs='+')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = (prepare(args.positive_input,args.input) if args.mode == 'prepare'
              else run(args.input,args.output.with_suffix('.cli.log'),args.pin_primary_host,args.priority_class,args.dsp_cpus))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('passed','processed_samples','acquisition_passed','maximum_latency_ms',
          'verified_active_bch_blocks','expected_active_bch_blocks') if key in result}),flush=True)
    return int(args.mode == 'run' and not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
