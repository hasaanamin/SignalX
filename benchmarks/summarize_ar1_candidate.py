"""Derived reporting from completed live-run rows; never a detector runtime input."""
import csv
import json
from pathlib import Path

import numpy as np


def main():
    root=Path('output/ar1_spectral_all')
    rows=list(csv.DictReader((root/'comparison.csv').open(encoding='utf-8-sig',newline='')))
    activity=[r for r in rows if r['row_type']=='activity_truth']
    events=[r for r in rows if r['row_type']=='detected_event']
    observations=[json.loads(line) for line in (root/'observations.jsonl').read_text().splitlines()]
    def details(truths):
        matched=[r for r in truths if r['match_status']=='matched']
        errors=[abs(int(r[k])) for r in matched for k in ('onset_error_samples','end_error_samples')]
        strict=[]
        for r in matched:
            length=int(r['truth_end_sample'])-int(r['truth_start_sample'])
            tolerance=16 if length<=64 else 32 if length<=256 else None
            if float(r['iou'])>=.5 and (tolerance is None or max(abs(int(r['onset_error_samples'])),abs(int(r['end_error_samples'])))<=tolerance):
                strict.append(r)
        return {'truths':len(truths),'matched':len(matched),
            'recall':len(matched)/len(truths) if truths else None,
            'matched_iou_ge_05':sum(float(r['iou'])>=.5 for r in matched),
            'short_boundary_qualified_other_iou_ge_05':len(strict),
            'absolute_endpoint_error_samples_p50_p95_max':np.percentile(errors,[50,95,100]).tolist() if errors else None}
    out={'all':details(activity),'nonnegative':details([r for r in activity if float(r['sweep_snr_db'])>=0]),
         'short_all':details([r for r in activity if r['scenario']=='short_bursts']),
         'short_nonnegative':details([r for r in activity if r['scenario']=='short_bursts' and float(r['sweep_snr_db'])>=0]),
         'short_by_snr':{str(s):details([r for r in activity if r['scenario']=='short_bursts' and float(r['sweep_snr_db'])==s]) for s in (-15,-10,-5,0,5,10)},
         'short_by_length_nonnegative':{str(n):details([r for r in activity if r['scenario']=='short_bursts' and float(r['sweep_snr_db'])>=0 and int(r['truth_end_sample'])-int(r['truth_start_sample'])==n]) for n in (64,256,2048,4095,4096)},
         'feature_statuses':{s:sum(r['feature_status']==s for r in events) for s in sorted({r['feature_status'] for r in events})},
         'all_raw_captures_exact':all(r['exact_capture_iq'] for r in observations),
         'elapsed_seconds':sum(r['elapsed_seconds'] for r in observations),
         'profile': 'ar1_spectral',
         'strict_metric_note':'IoU >=.5; additionally both endpoint absolute errors <=16 for <=64 samples, <=32 for <=256 samples. Longer bursts: IoU only. Original assignment preserved.',
         'confidence_note':'Characterization of development dataset; no independent-trial confidence or held-out certification claimed.'}
    (root/'boundary_summary.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
