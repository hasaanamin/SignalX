"""Versioned single-resolution operating envelopes, not universal detectors.

The low-SNR candidate uses a long full-band window; callers supply an independently
validated fixed policy. Construction never manufactures a noise reference or
claims certification from the profile name. Qualification status is external
release evidence for this exact immutable configuration.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from .noise_estimation import GaussianNoiseEstimator, ThresholdPolicy
from .exceptions import PolicyCompatibilityError


@dataclass(frozen=True)
class DetectionProfile:
    name: str = 'long_burst_iid_candidate'
    version: str = '1.0'
    sample_rate_hz: float = 1e6
    window_size: int = 4096
    hop_size: int = 256
    pfa_on: float = 1e-4
    pfa_off: float = 1e-2
    minimum_burst_samples: int = 4096
    design_snr_db: float = -10.0
    noise_assumption: str = 'stationary_iid_circular_complex_gaussian'
    ring_capacity: int = 16384
    pre_trigger_samples: int = 4096
    max_event_samples: int = 65536
    worker_capacity: int = 4

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip() or not isinstance(self.version, str) or not self.version.strip():
            raise ValueError('Profile name/version must be nonblank strings')
        if isinstance(self.design_snr_db, bool) or not isinstance(self.design_snr_db, (int,float)) or not math.isfinite(self.design_snr_db):
            raise ValueError('design_snr_db must be finite')
        for name in ('window_size','hop_size','minimum_burst_samples','ring_capacity',
                     'pre_trigger_samples','max_event_samples','worker_capacity'):
            value = getattr(self,name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not 0 < self.hop_size <= self.window_size:
            raise ValueError('Invalid window/hop')
        if self.minimum_burst_samples < self.window_size or self.max_event_samples < self.minimum_burst_samples+self.pre_trigger_samples+self.window_size:
            raise ValueError('Declared burst and capture cap must accommodate integration and margins')
        if self.ring_capacity <= self.pre_trigger_samples+self.window_size+self.hop_size:
            raise ValueError('Retention must exceed pre-trigger plus decision latency')
        if not 0 < self.pfa_on < self.pfa_off < 1:
            raise ValueError('Invalid PFA ordering')
        if self.sample_rate_hz != 1e6 or self.noise_assumption not in ('stationary_iid_circular_complex_gaussian', 'stationary_circular_complex_gaussian_ar1_0.8'):
            raise ValueError('Only the declared 1 MS/s iid / AR(1)=0.8 candidate models are defined')

    @property
    def profile_id(self):
        data = json.dumps(asdict(self),sort_keys=True,separators=(',',':'),allow_nan=False)
        return 'detection_' + hashlib.sha256(data.encode()).hexdigest()

    def create_pipeline(self, stream, policy: ThresholdPolicy, *, enable_feature_analysis=True,
                        synchronous_analysis=True, result_delivery='polling'):
        """Create the actual causal pipeline using supported absolute thresholds.

        Evidence applies per window, not per event or combined-search false alarms.
        A full-band fixed calibrated reference is explicitly supplied by the caller.
        Epoch age/stationarity remains the caller's external calibration obligation.
        """
        from signal_processing.runtime import StreamingCapturePipeline
        if self.noise_assumption.endswith('ar1_0.8') and policy.mode != 'empirical_matched':
            raise PolicyCompatibilityError('The colored candidate requires a compatible empirical policy, not iid analytic thresholds')
        if (stream.sample_rate_hz != self.sample_rate_hz or not policy.supported_pfa
                or policy.window_size != self.window_size or policy.hop_size != self.hop_size
                or policy.sample_rate_hz != self.sample_rate_hz
                or policy.pfa_on != self.pfa_on or policy.pfa_off != self.pfa_off
                or policy.monitored_band_hz is not None or policy.filter_sos_hash is not None
                or policy.reference_domain is None):
            raise PolicyCompatibilityError('Profile requires supported matching full-band fixed evidence')
        estimator = GaussianNoiseEstimator(window_size=self.window_size,
            calibrated_noise_power_linear=policy.mean_power_linear,
            reference_domain=policy.reference_domain)
        return StreamingCapturePipeline(stream, ring_capacity=self.ring_capacity,
            window_size=self.window_size, hop_size=self.hop_size,
            pfa_on=self.pfa_on, pfa_off=self.pfa_off, noise_estimator=estimator,
            start_persistence=1, end_persistence=1, pre_trigger_samples=self.pre_trigger_samples,
            pre_signal_margin_samples=0, post_signal_samples=0, min_post_trigger_samples=0,
            max_event_samples=self.max_event_samples, max_worker_queue_size=self.worker_capacity,
            threshold_policy=policy, require_supported_pfa=True,
            enable_feature_analysis=enable_feature_analysis, synchronous_analysis=synchronous_analysis,
            result_delivery=result_delivery, detection_profile_id=self.profile_id,
            boundary_uncertainty_samples=self.window_size)


LONG_BURST_IID = DetectionProfile()
LONG_BURST_AR1 = DetectionProfile(name='long_burst_ar1_candidate', window_size=16384,
    hop_size=1024, minimum_burst_samples=32768, ring_capacity=65536,
    pre_trigger_samples=16384, max_event_samples=262144,
    noise_assumption='stationary_circular_complex_gaussian_ar1_0.8')


@dataclass(frozen=True)
class AdaptiveDetectionProfile:
    """Explicit positive-SNR candidate, with causal model-based noise adaptation.

    Not a held-out fixed-threshold certificate. A preceding measured quiet
    reference must be supplied; this profile never learns from future samples.
    Qualification is external evidence for profile_id, not inferred from a name.
    """
    name: str = 'positive_snr_iid_candidate'
    version: str = '1.0'
    sample_rate_hz: float = 1e6
    window_size: int = 256
    hop_size: int = 64
    adaptation_window_size: int = 1024
    time_constant_s: float = .5
    pfa_on: float = 1e-4
    pfa_off: float = 1e-2
    minimum_burst_samples: int = 512
    maximum_qualified_burst_samples: int = 2048
    design_snr_db: float = 0.
    ring_capacity: int = 4096
    pre_trigger_samples: int = 256
    max_event_samples: int = 4096
    persistence: int = 2
    worker_capacity: int = 4
    noise_assumption: str = 'stationary_iid_circular_complex_gaussian'

    def __post_init__(self):
        for field in ('sample_rate_hz','time_constant_s','pfa_on','pfa_off','design_snr_db'):
            value=getattr(self,field)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f'{field} must be finite real')
        for name in ('window_size','hop_size','adaptation_window_size','minimum_burst_samples',
                     'maximum_qualified_burst_samples','ring_capacity','pre_trigger_samples',
                     'max_event_samples','persistence','worker_capacity'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:
                raise ValueError(f'{name} must be a positive integer')
        if (self.sample_rate_hz != 1e6 or not 0<self.hop_size<=self.window_size
                or not 0<self.pfa_on<self.pfa_off<1 or not math.isfinite(self.time_constant_s)
                or self.time_constant_s<=0 or not math.isfinite(self.design_snr_db)
                or self.minimum_burst_samples>self.maximum_qualified_burst_samples
                or self.ring_capacity<=self.pre_trigger_samples+self.window_size+self.hop_size
                or self.max_event_samples<self.maximum_qualified_burst_samples+self.pre_trigger_samples+self.window_size):
            raise ValueError('Invalid adaptive profile timing, thresholds or retention')
        if (not isinstance(self.name,str) or not self.name.strip()
                or not isinstance(self.version,str) or not self.version.strip()
                or self.noise_assumption!='stationary_iid_circular_complex_gaussian'):
            raise ValueError('An explicit iid profile identity is required')

    @property
    def profile_id(self):
        data=json.dumps(asdict(self),sort_keys=True,separators=(',',':'),allow_nan=False)
        return 'adaptive_detection_'+hashlib.sha256(data.encode()).hexdigest()

    def create_pipeline(self, stream, calibrated_mean_power, *, reference_domain=None,
                        enable_feature_analysis=True, synchronous_analysis=True,
                        result_delivery='polling', analysis_backend='thread'):
        from .noise_estimation import AdaptiveRobustNoiseEstimator
        from signal_processing.runtime import StreamingCapturePipeline
        if stream.sample_rate_hz!=self.sample_rate_hz:
            raise PolicyCompatibilityError('Adaptive profile sample rate mismatch')
        estimator=AdaptiveRobustNoiseEstimator(window_size=self.window_size,
            pfa_on=self.pfa_on,pfa_off=self.pfa_off,
            sample_rate_hz=self.sample_rate_hz, initial_noise_power_linear=calibrated_mean_power,
            adaptation_window_size=self.adaptation_window_size,time_constant_s=self.time_constant_s,
            reference_domain=reference_domain)
        return StreamingCapturePipeline(stream,noise_estimator=estimator,
            window_size=self.window_size,hop_size=self.hop_size,ring_capacity=self.ring_capacity,
            pfa_on=self.pfa_on,pfa_off=self.pfa_off,start_persistence=self.persistence,end_persistence=self.persistence,
            pre_trigger_samples=self.pre_trigger_samples,pre_signal_margin_samples=0,
            post_signal_samples=0,min_post_trigger_samples=0,max_event_samples=self.max_event_samples,
            max_worker_queue_size=self.worker_capacity,enable_feature_analysis=enable_feature_analysis,
            synchronous_analysis=synchronous_analysis,result_delivery=result_delivery,analysis_backend=analysis_backend,
            fixed_quiet_batch=True,detection_profile_id=self.profile_id,boundary_uncertainty_samples=self.window_size)


POSITIVE_SNR_IID = AdaptiveDetectionProfile()
