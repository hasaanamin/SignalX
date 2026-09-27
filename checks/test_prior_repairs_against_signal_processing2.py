import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import os
ROOT = Path(__file__).resolve().parent
SRC = Path(os.environ.get('SIGNAL_PROCESSING_SRC', ROOT.parent if (ROOT.parent / 'signal_processing').exists() else ROOT / 'signal_processing2_review')).resolve()
sys.path.insert(0, str(SRC))
from signal_processing.core import SignalEvent, StreamInfo
from signal_processing.context import SignalContext
from signal_processing.transforms import compute_phase, compute_instantaneous_frequency
from signal_processing.features.phase import PhaseFeatureExtractor
from signal_processing.features.amplitude import AmplitudeFeatureExtractor, AmplitudeFeatureConfig
from signal_processing.features.correlation import CorrelationFeatureExtractor
from signal_processing.features.aggregation import FeatureAggregator, build_default_feature_registry

def context(x):
 x=np.asarray(x,dtype=np.complex64)
 return SignalContext(SignalEvent('e',StreamInfo('s','src',1e6),x,0,len(x),0,len(x),0))

def test_clean_long_complex64_tone_has_no_artificial_jitter():
 x=np.exp(2j*np.pi*.2*np.arange(1000000)).astype(np.complex64)
 f=compute_instantaneous_frequency(compute_phase(x),1e6)
 assert abs(float(f.mean())-200000)<1
 assert float(f.std())<1

def test_zero_samples_do_not_break_claimed_rotation_invariance():
 x=np.array([1j,0,1j,0,1j],dtype=np.complex64)
 a=PhaseFeatureExtractor().extract(context(x)).features
 b=PhaseFeatureExtractor().extract(context(x*np.exp(.3j))).features
 assert a['phase.circular_resultant_length'].valid==b['phase.circular_resultant_length'].valid
 if a['phase.circular_resultant_length'].valid:
  assert a['phase.circular_resultant_length'].value==pytest.approx(b['phase.circular_resultant_length'].value,abs=1e-6)

def test_zero_correlation_peak_has_no_valid_phase():
 a=CorrelationFeatureExtractor().extract(context([1,0])).features
 assert a['correlation.nonzero_peak_magnitude'].value==0
 assert not a['correlation.nonzero_peak_phase_rad'].valid

def test_ddof_unavailable_statistics_are_invalid_instead_of_contract_crash():
 c=SimpleNamespace(envelope=lambda: np.array([1.]))
 a=AmplitudeFeatureExtractor(AmplitudeFeatureConfig(ddof=1)).extract(c).features
 assert not a['amplitude.std'].valid
 assert not a['amplitude.variance'].valid
 assert a['amplitude.mean'].valid

def test_two_sample_nonzero_event_can_be_aggregated_or_rejected_at_context_boundary():
 # Chosen robust policy: keep the valid event and return invalid unavailable spectral fields.
 # An explicitly documented earlier minimum-length rejection is also a legitimate fix.
 a=FeatureAggregator(build_default_feature_registry()).extract(context([1,1]))
 assert len(a.features)==56
