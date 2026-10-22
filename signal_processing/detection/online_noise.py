"""Causal, bounded, blind noise tracking; no truth or quiet-prefix API.

This is a conditional Gaussian/AR1 model, NOT a held-out Pfa certificate.
Startup requires a majority of noise-dominated frames. Persistent Gaussian-like
signals are not identifiable as signals from single-channel IQ alone.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
from scipy.stats import gamma

from signal_processing.core.contracts import NoiseEstimate, NoiseReferenceDomain
from signal_processing.core.exceptions import StreamConsistencyError
from signal_processing.detection.energy_detector import EnergyDetector
from signal_processing.detection.exceptions import PolicyCompatibilityError
from signal_processing.detection.noise_estimation import NoiseLifecycleState


@dataclass(frozen=True)
class OnlineNoiseConfig:
    """Sample-clock controls; defaults at 1 MS/s: 8.192 ms initial learning.

    gaussian assumes iid circular Gaussian IQ. ar1 also estimates a complex lag-1
    coefficient (including approximately zero for white Gaussian noise).
    """
    model: str = "ar1"
    frame_samples: int = 1024
    history_frames: int = 8
    update_frames: int = 4
    time_constant_s: float = 0.02
    max_age_samples: int = 1_000_000
    max_correlation: float = 0.95
    residual_tolerance: float = 0.12
    version: str = "blind_robust_frames_v1"

    def __post_init__(self):
        if self.model not in ("gaussian", "ar1"):
            raise ValueError("Online model must be gaussian or ar1")
        for name, low, high in (("frame_samples", 256, 16384), ("history_frames", 4, 32), ("update_frames", 1, 32),
                                ("max_age_samples", 1, 10**12)):
            v = getattr(self, name)
            if type(v) is not int or not low <= v <= high:
                raise ValueError(f"Invalid {name}")
        for name in ("time_constant_s", "max_correlation", "residual_tolerance"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"Invalid {name}")
        if self.max_correlation >= .98 or self.residual_tolerance >= .5:
            raise ValueError("Unsupported correlation/diagnostic limit")
        if self.version != "blind_robust_frames_v1":
            raise ValueError("Unsupported online estimator semantics")

    @property
    def config_id(self):
        return "online_" + hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def profile_id(self, window_size, hop_size, pfa_on, pfa_off):
        content = dict(noise=asdict(self), window_size=window_size, hop_size=hop_size,
                       pfa_on=pfa_on, pfa_off=pfa_off, statistic="raw_pair_rewhitened_window_v1")
        return "online_profile_" + hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class OnlineNoiseSnapshot:
    coefficient: complex
    mean: complex
    innovation_power: float
    revision: int
    model_id: str
    residual_correlation: float
    residual_impropriety: float


class OnlineNoiseTracker:
    """Robust frame statistics, updated only AFTER the current decision.

    Excluded frames are discarded, never concatenated across active intervals.
    Median frame estimates tolerate isolated high-power bursts, not arbitrary
    occupancy. Freeze cannot distinguish a sustained signal from a noise step.
    """
    def __init__(self, sample_rate_hz: float, config: OnlineNoiseConfig):
        if not isinstance(config, OnlineNoiseConfig):
            raise TypeError("Expected OnlineNoiseConfig")
        if isinstance(sample_rate_hz, bool) or not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("Invalid sample rate")
        self.config, self.sample_rate_hz = config, float(sample_rate_hz)
        self._median_factor = float(gamma.ppf(.5, a=config.frame_samples-1, scale=1/(config.frame_samples-1)))
        self.reset()

    def reset(self):
        self.snapshot = None
        self._frames = deque(maxlen=self.config.history_frames)
        self._partial = np.empty(self.config.frame_samples, np.complex128)
        self._fill = 0
        self._frozen = False
        self._age = 0
        self._pending = 0
        self.received_samples = 0
        self.accepted_samples = 0
        self.rejected_fits = 0
        self.first_reference_received_samples = None
        self._frames_since_fit = 0
        self._estimate_cache = None
        self.reason = "learning: no independently identifiable noise reference yet"
        self.reference_domain = NoiseReferenceDomain(self.sample_rate_hz, preprocessing_id=self.config.config_id)

    @property
    def state(self):
        # STARTING deliberately differs from authorized Gaussian CALIBRATING.
        if self.snapshot is None:
            return "STARTING"
        return NoiseLifecycleState.STALE if self._age+self._pending > self.config.max_age_samples else NoiseLifecycleState.VALID

    @property
    def noise_power_linear(self):
        return None if self.snapshot is None else self.snapshot.innovation_power

    @property
    def frozen(self):
        return self._frozen

    def freeze(self, frozen):
        self._frozen = bool(frozen)
        if frozen:
            self._fill = 0

    def set_pending_samples(self, count):
        self._pending = count

    def get_estimate(self, stream_id="default", block_sequence_number=0):
        valid = self.snapshot is not None and self.state == NoiseLifecycleState.VALID
        key = (stream_id, block_sequence_number, self.snapshot, valid, self.reason, self.accepted_samples)
        if self._estimate_cache is not None and self._estimate_cache[0] == key:
            return self._estimate_cache[1]
        result = NoiseEstimate(stream_id, block_sequence_number,
            10*math.log10(self.snapshot.innovation_power) if valid else float("nan"),
            valid=valid, reason=None if valid else ("online reference stale" if self.snapshot else self.reason),
            reference_domain=self.reference_domain, training_samples=self.accepted_samples,
            uncertainty_method="blind_noise_dominated_frames; model_only; no held_out_support")
        self._estimate_cache = (key, result)
        return result

    def diagnostics(self):
        s = self.snapshot
        return dict(state=self.state.value if isinstance(self.state, NoiseLifecycleState) else self.state,
            reason=self.reason, received_samples=self.received_samples, accepted_samples=self.accepted_samples,
            reference_age_samples=self._age+self._pending, retained_frames=len(self._frames),
            retained_samples=len(self._frames)*self.config.frame_samples+self._fill,
            rejected_fits=self.rejected_fits, revision=s.revision if s else 0,
            first_reference_received_samples=self.first_reference_received_samples,
            coefficient=[s.coefficient.real, s.coefficient.imag] if s else None,
            innovation_power=s.innovation_power if s else None, supported_pfa=False,
            config_id=self.config.config_id)

    def observe(self, iq, *, eligible):
        if not isinstance(iq, np.ndarray) or iq.ndim != 1 or not np.iscomplexobj(iq) or not np.all(np.isfinite(iq)):
            raise ValueError("Expected finite complex IQ")
        self.received_samples += len(iq)
        self._age += len(iq)
        self._pending = 0
        if not eligible or self._frozen:
            self._fill = 0
            return
        offset = 0
        while offset < len(iq):
            n = min(len(iq)-offset, self.config.frame_samples-self._fill)
            self._partial[self._fill:self._fill+n] = iq[offset:offset+n]
            self._fill += n
            offset += n
            if self._fill == self.config.frame_samples:
                self._frames.append(self._partial.copy())
                self._fill = 0
                self.accepted_samples += self.config.frame_samples
                self._frames_since_fit += 1
                if len(self._frames) == self.config.history_frames and self._frames_since_fit >= self.config.update_frames:
                    self._fit()
                    if self.snapshot is not None and self.first_reference_received_samples is None:
                        self.first_reference_received_samples = self.received_samples-(len(iq)-offset)
                    self._frames_since_fit = 0

    def _fit(self):
        frames = np.stack(self._frames)
        means = frames.mean(axis=1)
        centered = frames-means[:, None]
        raw = np.mean(np.abs(centered)**2, axis=1)
        positive = raw > 0
        median = float(np.median(raw))
        # Reject conspicuous signals without assuming generator power or rho.
        good = positive & (raw <= 2*median) & (np.abs(means)**2 <= .1*raw)
        if np.count_nonzero(good) < self.config.history_frames//2+1 or median <= 0:
            self.rejected_fits += 1
            self.reason = "uncertain: insufficient noise-like frames (zero/tonal/occupied input)"
            return
        x = centered[good]
        mu = complex(np.median(means[good].real), np.median(means[good].imag))
        denominator = np.sum(np.abs(x[:, :-1])**2, axis=1)
        coeffs = np.sum(x[:, 1:]*x[:, :-1].conj(), axis=1)/denominator
        a = (complex(np.median(coeffs.real), np.median(coeffs.imag))
             if self.config.model == "ar1" else 0j)
        residual = x[:, 1:]-a*x[:, :-1]
        energy = float(np.sum(np.abs(residual)**2))
        if energy <= 0 or not math.isfinite(energy):
            self.rejected_fits += 1
            self.reason = "uncertain: unusable innovation power"
            return
        corr = max(float(abs(np.sum(residual[:, lag:]*residual[:, :-lag].conj()))/energy) for lag in (1, 2, 4, 8))
        improper = float(abs(np.sum(residual**2))/energy)
        if abs(a) > self.config.max_correlation or max(corr, improper) > self.config.residual_tolerance:
            self.rejected_fits += 1
            self.reason = "uncertain: Gaussian/AR1 residual diagnostics failed"
            return
        power = float(np.median(np.mean(np.abs(residual)**2, axis=1))/self._median_factor)
        old = self.snapshot
        alpha = 1.0 if old is None else -math.expm1(-self.config.frame_samples*self.config.update_frames/(self.sample_rate_hz*self.config.time_constant_s))
        if old is not None:
            a = old.coefficient + alpha*(a-old.coefficient)
            mu = old.mean + alpha*(mu-old.mean)
            # Power must describe the coefficient ACTUALLY used, not the newly
            # fitted but as-yet unapplied coefficient.
            y = (frames[good, 1:]-mu)-a*(frames[good, :-1]-mu)
            target = float(np.median(np.mean(np.abs(y)**2, axis=1))/self._median_factor)
            power = old.innovation_power + alpha*(target-old.innovation_power)
        revision = 1 if old is None else old.revision+1
        identity = hashlib.sha256(repr((self.config.config_id, a, mu, power, revision)).encode()).hexdigest()
        self.snapshot = OnlineNoiseSnapshot(a, mu, power, revision, identity, corr, improper)
        self.reference_domain = NoiseReferenceDomain(self.sample_rate_hz, preprocessing_id=identity)
        self._age = 0
        self.reason = "tracking: conditional noise-dominated Gaussian/AR1 model; not certified"


class OnlineEnergyDetector(EnergyDetector):
    """Re-whiten raw windows under ONE current model; no mixed-model residuals.

    Tracking is coordinated by StreamingCapturePipeline AFTER each hop. Direct
    process()/warmup and external policies are intentionally unsupported here.
    """
    def __init__(self, *, tracker, **kwargs):
        self.tracker = tracker
        super().__init__(**kwargs)

    @property
    def threshold_model(self):
        return "online_noise_gaussian_ar1_model_only"

    def set_threshold_policy(self, policy):
        raise PolicyCompatibilityError("Changing online statistics cannot retain a fixed threshold certificate")

    def process(self, block, noise_estimate):
        raise StreamConsistencyError("Online detection requires the coordinated streaming pipeline")

    def prepare_detection_iq(self, iq):
        raise StreamConsistencyError("Online detection does not accept externally designated warmup")

    def process_windows(self, block, noise_estimate):
        self._validate_continuity(block, "streaming")
        if not block.num_samples:
            return []
        if block.num_samples > self.hop_size:
            raise StreamConsistencyError("Online detector requires causal pipeline hop subdivision")
        model = self.tracker.snapshot
        if model is not None:
            if noise_estimate.reference_domain != self.tracker.reference_domain:
                raise PolicyCompatibilityError("Online reference does not match current model revision")
            self.reference_domain = self.tracker.reference_domain
            thresholds = self._linear_thresholds(noise_estimate)
        # N+1 received raw samples supply N true predecessor pairs. Absolute
        # origin is anchored once, independent of transport partitioning.
        combined = np.concatenate((self._residual_iq, block.iq))
        origin = self._residual_start_sample if len(self._residual_iq) else block.start_sample
        curr = 0
        out = []
        while curr+self.window_size+1 <= len(combined):
            if model is not None:
                raw = combined[curr:curr+self.window_size+1].astype(np.complex128)
                y = (raw[1:]-model.mean)-model.coefficient*(raw[:-1]-model.mean)
                power = float(np.vdot(y, y).real/self.window_size)
                out.append(self._measurement(block, origin+curr+1, origin+curr+1+self.window_size,
                    power, thresholds, "ar1_innovation_power"))
            curr += self.hop_size
        self._residual_iq = combined[curr:].copy()
        self._residual_start_sample = origin+curr
        self._next_window_sample = origin+curr+1
        self._last_detection_iq = block.iq
        self._commit_input(block, "streaming")
        return out
