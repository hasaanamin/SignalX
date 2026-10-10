"""Run every declared AR1 record, preserving failures and full SNR denominators."""
import argparse
import json
from pathlib import Path

from signal_processing.ui.smoke_runner import (
    SmokeConfig, compact_smoke_result, export_smoke_csv, inspect_smoke_dataset, run_smoke_record,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('detector_dataset_complete'))
    parser.add_argument('--profile',choices=('ar1_multires','ar1_spectral'),default='ar1_spectral')
    parser.add_argument('--short-only',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('output/ar1_candidate'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    config=SmokeConfig(window_size=256,hop_size=8,detector_profile=args.profile)
    results=[]
    failures=[]
    with (args.output/'observations.jsonl').open('w',encoding='utf-8') as log:
        for record in inspect_smoke_dataset(args.dataset):
            a=json.loads(record.annotations_path.read_text())
            if a['background']['model']!='ar1' or (args.short_only and a['scenario']!='short_bursts'):
                continue
            try:
                result=compact_smoke_result(run_smoke_record(record,config))
                results.append(result)
                row={'record_id':record.record_id,'scenario':record.scenario,**result.metrics}
            except Exception as exc:
                row={'record_id':record.record_id,'error':f'{type(exc).__name__}: {exc}'}
                failures.append(row)
            log.write(json.dumps(row)+'\n')
            log.flush()
            print(json.dumps(row),flush=True)
    if results:
        export_smoke_csv(args.output/'comparison.csv',results)
    def totals(rows):
        values={key:sum(r.metrics[key] for r in rows) for key in
                ('truth_activities','captures','matched','misses','extras','noise_only_events','feature_failures')}
        values['records']=len(rows)
        values['event_recall']=values['matched']/values['truth_activities'] if values['truth_activities'] else None
        return values
    summary={'profile':args.profile,'failures':failures,'all_snrs':totals(results),
             'snr_0_and_above':totals([r for r in results if r.metrics['sweep_snr_db']>=0]),
             'by_snr':{str(snr):totals([r for r in results if r.metrics['sweep_snr_db']==snr]) for snr in (-15,-10,-5,0,5,10)},
             'claim':'Offline fitted-model characterization; NOT held-out Pfa or real-time qualification'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    # All-SNR >90% is not silently replaced by the easier positive-SNR subset.
    return 1 if failures else (0 if summary['all_snrs']['event_recall'] is not None and summary['all_snrs']['event_recall']>.90 else 2)


if __name__=='__main__':
    raise SystemExit(main())
