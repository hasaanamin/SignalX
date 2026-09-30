"""Positive filtered-statistic, lifecycle and capture checks with an empirical estimate."""
from pathlib import Path
import sys,json,math,hashlib
import numpy as np
import os
ROOT = Path(__file__).parent
SRC = Path(os.environ.get('SIGNAL_PROCESSING_SRC', ROOT.parent if (ROOT.parent / 'signal_processing').exists() else ROOT / 'signal_processing2_review')).resolve()
sys.path.insert(0, str(SRC))
from signal_processing.core import StreamInfo,IQBlock,NoiseEstimate
from signal_processing.detection import EmpiricalNoiseCalibrator,EnergyDetector,GaussianNoiseEstimator
from signal_processing.runtime import StreamingCapturePipeline
out={}
rng=np.random.default_rng(508)
train=((rng.normal(size=700000)+1j*rng.normal(size=700000))*math.sqrt(1e-4/2)).astype(np.complex64)
cal=EmpiricalNoiseCalibrator(window_size=64,hop_size=64)
reference=cal.calibrate(train,monitored_band_hz=(90000,110000),settling_samples=4096)
rows=[]
for bs in (257,65536,len(train)):
 c=EmpiricalNoiseCalibrator(window_size=64,hop_size=64)
 c.begin_calibration(monitored_band_hz=(90000,110000),settling_samples=4096)
 for start in range(0,len(train),bs):c.update_calibration(train[start:start+bs])
 r=c.finalize_calibration()
 assert r['quantile_resolved'] and not r['supported_pfa']
 for key in ('mean_power_linear','empirical_t_on_linear','empirical_t_off_linear'):
  np.testing.assert_allclose(r[key],reference[key],rtol=1e-7,atol=1e-15)
 rows.append({'block_size':bs,'samples':r['num_samples'],'windows':r['num_windows'],'mean':r['mean_power_linear'],'t_on':r['t_on_linear'],
              'maximum_relative_difference':max(abs(r[key]-reference[key])/reference[key] for key in ('mean_power_linear','empirical_t_on_linear','empirical_t_off_linear'))})
out['empirical_streaming_vs_batch_uncertified']=rows
policy=cal.create_policy(reference)
stream=StreamInfo('filtered','oracle',1e6)
x=((rng.normal(size=12000)+1j*rng.normal(size=12000))*.01).astype(np.complex64)
baseline=None;worst=0;coords=[]
for bs in (1,31,257,4097):
 d=EnergyDetector(window_size=64,hop_size=64,mode='configured_band',monitored_band_hz=(90000,110000),threshold_policy=policy)
 ref=NoiseEstimate(stream.stream_id,0,10*math.log10(policy.mean_power_linear));ms=[]
 for seq,start in enumerate(range(0,len(x),bs)):ms+=d.process_windows(IQBlock(stream,seq,start,None,x[start:start+bs]),ref)
 values=np.array([m.signal_power_dbfs for m in ms]);c=[(m.start_sample,m.end_sample) for m in ms]
 if baseline is None:baseline=(values,c)
 np.testing.assert_allclose(values,baseline[0],rtol=0,atol=1e-9);assert c==baseline[1]
 worst=max(worst,float(np.max(abs(values-baseline[0]))));coords.append({'block_size':bs,'windows':len(ms)})
out['configured_filter_continuity']={'cases':coords,'maximum_db_error':worst}
s=StreamInfo('lifecycle','test',1e6)
p=StreamingCapturePipeline(stream_info=s,window_size=16,hop_size=8,ring_capacity=4096,start_persistence=1,end_persistence=1,
 pre_trigger_samples=0,pre_signal_margin_samples=0,post_signal_samples=0,min_post_trigger_samples=0,
 noise_estimator=GaussianNoiseEstimator(window_size=16,calibrated_noise_power_linear=1e-4),enable_feature_analysis=False)
p.process_block(IQBlock(s,0,0,None,np.full(69,.5,np.complex64)))
events=p.flush();assert len(events)==1 and p.flush()==[]
try:p.process_block(IQBlock(s,1,69,None,np.ones(16,np.complex64)))
except RuntimeError:pass
else:raise AssertionError('EOF allowed input')
p.reset();assert p.state.value=='OPEN' and p.ring_buffer.occupancy==0 and p.worker.processed_results==[]
p.process_block(IQBlock(s,0,0,None,np.full(69,.5,np.complex64)))
new=p.flush();assert len(new)==1;np.testing.assert_array_equal(events[0].event.iq,new[0].event.iq)
p.shutdown()
out['EOF_terminal_reset_replay']={'passed':True,'exact_replayed_iq':True}
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
