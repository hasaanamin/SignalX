"""Causal windowed energy detector with full-band and configured-band monitoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal
from scipy.stats import gamma

from signal_processing.core.contracts import DetectionMeasurement, IQBlock, NoiseEstimate, NoiseReferenceDomain
from signal_processing.core.interfaces import DetectorProtocol, Resettable
from signal_processing.core.exceptions import StreamConsistencyError
from signal_processing.core.stream_compatibility import require_stream, same_sample_rate
from signal_processing.detection.exceptions import (
    FilterConfigurationError,
    PolicyCompatibilityError,
    UncalibratedNoiseError,
)
from signal_processing.detection.noise_estimation import ThresholdPolicy
from signal_processing.detection._power import window_mean_power, window_mean_powers


@dataclass(frozen=True)
class _QuietFullBandBatch:
    """Private one-use preparation. It owns no committed stream progress."""

    owner: object
    block: IQBlock
    prior_residual: np.ndarray
    prior_input_end: int | None
    prior_next_window: int
    prior_policy: object
    combined: np.ndarray
    origin: int
    retained_start: int
    last_power: float
    thresholds: tuple


@dataclass
class _FullBandCursor:
    """Call-local numerical lookahead; only commit advances streaming state."""

    owner: object
    block: IQBlock
    combined: np.ndarray
    origin: int
    first: int
    powers: tuple[float, ...]
    index: int
    expected_end: int | None
    expected_residual: np.ndarray
    expected_next: int
    policy: object


class EnergyDetector(DetectorProtocol, Resettable):
    """Causal windowed energy detector.

    Supports:
      1. Full-band monitoring: computes power directly on raw baseband IQ.
      2. Configured-band monitoring: translates a target frequency band to baseband
         with continuous phase tracking, filters out-of-band energy using a causal
         SOS filter with state carried across blocks, and measures in-band energy
         while preserving raw IQ in the ring buffer untouched.
    """

    def __init__(
        self,
        window_size: int = 256,
        hop_size: int = 64,
        pfa_on: float = 1e-4,
        pfa_off: float = 1e-2,
        mode: Literal["full_band", "configured_band"] = "full_band",
        monitored_band_hz: tuple[float, float] | None = None,
        sample_rate_hz: float = 1e6,
        filter_order: int = 4,
        conversion_floor_linear: float = 1e-15,
        threshold_policy: ThresholdPolicy | None = None,
        gain_id: str = "digital_iq_unity",
        require_supported_pfa: bool = False,
        ar1_model=None,
        ar1_windows: tuple[int, ...] = (),
        ar1_subbands: int = 0,
    ) -> None:
        if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)) or window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size!r}")
        if isinstance(hop_size, bool) or not isinstance(hop_size, (int, np.integer)) or not (1 <= hop_size <= window_size):
            raise ValueError(f"hop_size must satisfy 1 <= hop_size <= window_size, got {hop_size!r}")
        if not (0.0 < pfa_on < pfa_off < 1.0):
            raise ValueError("False alarm probabilities must satisfy 0 < pfa_on < pfa_off < 1")
        if mode not in ("full_band", "configured_band"):
            raise ValueError(f"Unknown detector mode: {mode!r}. Must be 'full_band' or 'configured_band'")
        if (
            isinstance(conversion_floor_linear, bool)
            or not isinstance(conversion_floor_linear, (int, float, np.floating, np.integer))
            or not math.isfinite(float(conversion_floor_linear))
            or float(conversion_floor_linear) <= 0.0
        ):
            raise ValueError(f"conversion_floor_linear must be strictly positive and finite, got {conversion_floor_linear!r}")
        if isinstance(filter_order, bool) or not isinstance(filter_order, (int, np.integer)) or filter_order < 1:
            raise ValueError(f"filter_order must be an integer >= 1, got {filter_order!r}")

        if (
            isinstance(sample_rate_hz, bool)
            or not isinstance(sample_rate_hz, (int, float, np.floating, np.integer))
            or not math.isfinite(float(sample_rate_hz))
            or float(sample_rate_hz) <= 0.0
        ):
            raise ValueError(f"sample_rate_hz must be strictly positive and finite, got {sample_rate_hz!r}")

        self._window_size = int(window_size)
        self._hop_size = int(hop_size)
        self._pfa_on = float(pfa_on)
        self._pfa_off = float(pfa_off)
        self._mode = mode
        self._monitored_band_hz = tuple(monitored_band_hz) if monitored_band_hz is not None else None
        self._sample_rate_hz = float(sample_rate_hz)
        self._conversion_floor = float(conversion_floor_linear)
        if ar1_model is not None:
            from .ar_noise import AR1NoiseModel
            if not isinstance(ar1_model, AR1NoiseModel) or ar1_model.sample_rate_hz != float(sample_rate_hz):
                raise ValueError("A fitted AR1 model with matching sample rate is required")
            if mode != "full_band":
                raise ValueError("AR1 whitening plus configured SOS filtering is not yet supported")
        self._ar1_model = ar1_model
        self._ar1_previous = None
        if ar1_windows and (ar1_model is None or threshold_policy is not None):
            raise ValueError("AR1 bank requires its fitted model and currently supports model-only thresholds")
        if (not isinstance(ar1_windows, tuple) or len(ar1_windows)>3
                or len(set(ar1_windows)) != len(ar1_windows)
                or any(type(n) is not int or not hop_size <= n <= window_size for n in ar1_windows)):
            raise ValueError("AR1 bank must contain at most three distinct N between H and primary N")
        self._ar1_windows = ar1_windows
        if type(ar1_subbands) is not int or ar1_subbands not in (0,8) or (ar1_subbands and (32 not in ar1_windows or window_size != 256)):
            raise ValueError("Spectral candidate requires a 256-sample AR1 bank and 8 fixed subbands")
        self._ar1_subbands = ar1_subbands
        tests = len(ar1_windows)+ar1_subbands
        self._ar1_quantiles = {n:(float(gamma.isf(pfa_on/tests,a=n,scale=1/n)),
                                  float(gamma.isf(pfa_off/tests,a=n,scale=1/n)))
                               for n in ar1_windows}
        if ar1_subbands:
            self._ar1_band_bins = tuple((np.arange(c-16,c+16)%256) for c in range(0,256,32))

        self._unit_q_on = float(gamma.isf(self._pfa_on, a=self._window_size, scale=1.0 / self._window_size))
        self._unit_q_off = float(gamma.isf(self._pfa_off, a=self._window_size, scale=1.0 / self._window_size))

        # Stateful streaming buffers
        self._residual_iq: np.ndarray = np.empty(0, dtype=np.complex64)
        self._residual_start_sample: int = 0
        self._next_window_sample: int = 0
        self._phase_accumulator: float = 0.0
        self._input_end: int | None = None
        self._input_stream = None
        self._input_sequence: int | None = None
        self._api_mode: str | None = None
        self._path_samples = 0

        # Filter setup for configured-band
        self._sos: np.ndarray | None = None
        self._sos_zi: np.ndarray | None = None
        self._sos_hash: str | None = None
        self._filter_delay_samples: int = 0
        if self._mode == "configured_band":
            self._init_configured_band_filter(filter_order)

        self.reference_domain = NoiseReferenceDomain(
            self._sample_rate_hz, self._monitored_band_hz, self._sos_hash, gain_id,
            ar1_model.model_id if ar1_model is not None else None,
        )
        self._require_supported_pfa = require_supported_pfa
        self._threshold_policy: ThresholdPolicy | None = None
        if threshold_policy is not None:
            self.set_threshold_policy(threshold_policy)
        elif require_supported_pfa:
            raise PolicyCompatibilityError("Strict supported-Pfa operation requires a held-out policy")

    def set_threshold_policy(self, policy: ThresholdPolicy) -> None:
        """Set and validate an empirical or analytic threshold policy."""
        if self._ar1_windows:
            raise PolicyCompatibilityError("AR1 bank evidence cannot be replaced by a single-window certificate")
        if not isinstance(policy, ThresholdPolicy):
            raise TypeError(f"policy must be a ThresholdPolicy instance, got {type(policy).__name__}")
        if not policy.valid:
            raise PolicyCompatibilityError(f"Threshold policy is marked invalid: {policy.reason}")
        if self._require_supported_pfa and not policy.supported_pfa:
            raise PolicyCompatibilityError(f"Threshold policy has unsupported Pfa: {policy.reason}")
        if policy.window_size != self._window_size:
            raise PolicyCompatibilityError(
                f"Policy window_size ({policy.window_size}) does not match detector window_size ({self._window_size})"
            )
        if policy.hop_size != self._hop_size:
            raise PolicyCompatibilityError(
                f"Policy hop_size ({policy.hop_size}) does not match detector hop_size ({self._hop_size})"
            )
        if not math.isclose(policy.sample_rate_hz, self._sample_rate_hz, rel_tol=1e-12):
            raise PolicyCompatibilityError(
                f"Sample rate mismatch: policy sample_rate_hz ({policy.sample_rate_hz}) does not match detector sample_rate_hz ({self._sample_rate_hz})"
            )
        if not math.isclose(policy.pfa_on, self._pfa_on, rel_tol=1e-7) or not math.isclose(policy.pfa_off, self._pfa_off, rel_tol=1e-7):
            raise PolicyCompatibilityError(
                f"Policy Pfa ({policy.pfa_on}, {policy.pfa_off}) does not match detector Pfa ({self._pfa_on}, {self._pfa_off})"
            )
        if self._mode == "configured_band":
            if policy.monitored_band_hz != self._monitored_band_hz:
                raise PolicyCompatibilityError(
                    f"Policy monitored band ({policy.monitored_band_hz}) does not match detector band ({self._monitored_band_hz})"
                )
            if policy.filter_sos_hash != self._sos_hash:
                raise PolicyCompatibilityError(
                    f"Policy filter SOS hash ({policy.filter_sos_hash}) does not match detector SOS hash ({self._sos_hash})"
                )
        elif self._mode == "full_band":
            if policy.monitored_band_hz is not None or policy.filter_sos_hash is not None:
                raise PolicyCompatibilityError(
                    "Cannot attach a configured-band policy to a full-band detector"
                )
        if policy.reference_domain is not None and policy.reference_domain != self.reference_domain:
            raise PolicyCompatibilityError("Policy reference domain differs from detector path/gain")
        self._threshold_policy = policy

    @property
    def threshold_policy(self) -> ThresholdPolicy | None:
        return self._threshold_policy

    @property
    def sos_hash(self) -> str | None:
        return self._sos_hash

    def _init_configured_band_filter(self, filter_order: int) -> None:
        if self._monitored_band_hz is None:
            raise FilterConfigurationError("monitored_band_hz must be specified for configured_band mode")
        f_min, f_max = float(self._monitored_band_hz[0]), float(self._monitored_band_hz[1])
        nyquist = self._sample_rate_hz / 2.0
        if not (-nyquist <= f_min < f_max <= nyquist):
            raise FilterConfigurationError(
                f"monitored_band_hz ({f_min}, {f_max}) exceeds Nyquist boundaries [-{nyquist}, {nyquist}]"
            )

        f_center = (f_min + f_max) / 2.0
        bw = f_max - f_min
        cutoff_hz = bw / 2.0

        # Normalize cutoff to Nyquist without silent clamping
        wn = cutoff_hz / nyquist
        if wn <= 0.0 or wn >= 1.0:
            raise FilterConfigurationError(
                f"Normalized cutoff frequency {wn} (cutoff {cutoff_hz} Hz / Nyquist {nyquist} Hz) "
                f"must satisfy 0 < wn < 1. Bandwidth {bw} Hz is unsupported."
            )
        self._sos = signal.butter(N=filter_order, Wn=wn, btype="lowpass", output="sos")
        # Initialize causal SOS state with zeros for clean startup
        self._sos_zi = np.zeros((self._sos.shape[0], 2), dtype=np.float64)
        self._f_center_hz = f_center

        # Compute nominal group delay using direct per-section phase derivative at downmixed center (omega=0)
        # sum over SOS of (b1+2*b2)/(b0+b1+b2) - (a1+2*a2)/(a0+a1+a2)
        b_term = (self._sos[:, 1] + 2.0 * self._sos[:, 2]) / (self._sos[:, 0] + self._sos[:, 1] + self._sos[:, 2])
        a_term = (self._sos[:, 4] + 2.0 * self._sos[:, 5]) / (self._sos[:, 3] + self._sos[:, 4] + self._sos[:, 5])
        gd = float(np.sum(b_term - a_term))
        if not math.isfinite(gd) or gd < 0.0:
            raise FilterConfigurationError(f"Calculated filter delay {gd} must be non-negative and finite")
        self._filter_delay_samples = int(round(gd))
        import hashlib
        self._sos_hash = hashlib.sha256(self._sos.tobytes()).hexdigest()[:12]

        # Pole decay determines a converged impulse horizon, rather than N or
        # an arbitrary short cap. Extremely narrow filters use the universal
        # Gaussian correlation bound (sum eigenweights^2 <= 1), explicitly.
        poles = np.concatenate([np.roots(section[3:]) for section in self._sos])
        radius = float(np.max(np.abs(poles)))
        if not 0 < radius < 1:
            raise FilterConfigurationError("Filter must have stable poles")
        length = max(512, int(math.ceil(math.log(1e-15) / math.log(radius))))
        self._worst_case_correlation = length > 262144
        self._impulse_length = 0
        self._impulse_tail_fraction = None
        if self._worst_case_correlation:
            self._covariance_method = "worst_case_gaussian_correlation_bound"
            self._filter_autocorrelation = np.ones(self._window_size)
        else:
            impulse = np.zeros(length)
            impulse[0] = 1.0
            response = signal.sosfilt(self._sos, impulse)
            tail_fraction = float(np.sum(response[-max(32, length // 8):] ** 2) / np.sum(response ** 2))
            if tail_fraction > 1e-12:
                raise FilterConfigurationError("Filter impulse response did not converge")
            autocorr = signal.fftconvolve(response, response[::-1], mode="full")[length - 1:]
            self._filter_autocorrelation = autocorr / autocorr[0]
            self._impulse_length = length
            self._impulse_tail_fraction = tail_fraction
            self._covariance_method = "converged_impulse_autocorrelation"
        self._n_eff = self._effective_n(self._window_size)
        self._unit_q_on, self._unit_q_off = self._configured_quantiles(self._window_size)

    def _ensure_covariance_lags(self, n: int) -> None:
        if len(self._filter_autocorrelation) < n:
            self._filter_autocorrelation = np.pad(
                self._filter_autocorrelation, (0, n - len(self._filter_autocorrelation)),
                constant_values=1.0 if self._worst_case_correlation else 0.0,
            )

    def _effective_n(self, n: int) -> float:
        self._ensure_covariance_lags(n)
        r = self._filter_autocorrelation[:n]
        lags = np.arange(1, len(r))
        variance = (n + 2 * np.sum((n - lags) * r[1:] ** 2)) / n ** 2
        return 1.0 / float(variance)

    def _configured_quantiles(self, n: int) -> tuple[float, float]:
        """Conservative Gaussian quadratic-form bound, not an effective-Gamma tail.

        For X=sum(w_i Exp(1)), sum(w_i)=1:
        P[X >= 1+sqrt(2 sum(w_i^2) t)+max(w_i)t] <= exp(-t).
        The Toeplitz absolute row-sum bounds max(w_i). Impulse convergence
        is checked, or worst-case correlation used; this is conditional on stationary white
        circular-Gaussian input and a correct mean, not receiver certification.
        """
        if n == 1:
            return -math.log(self._pfa_on), -math.log(self._pfa_off)
        self._ensure_covariance_lags(n)
        r = self._filter_autocorrelation[:n]
        variance = 1.0 / self._effective_n(n)
        radius_bound = min(1.0, float(1 + 2 * np.sum(np.abs(r[1:]))) / n)
        def bound(p: float) -> float:
            t = -math.log(p)
            return 1.0 + math.sqrt(2 * variance * t) + radius_bound * t
        return bound(self._pfa_on), bound(self._pfa_off)

    @property
    def threshold_model(self) -> str:
        if self._threshold_policy is not None:
            return self._threshold_policy.mode
        if self._ar1_model is not None:
            return "fitted_ar1_innovation_gamma_union_model" if self._ar1_windows else "fitted_ar1_innovation_gamma_model"
        return "stationary_gaussian_quadratic_bound" if self._mode == "configured_band" else "iid_complex_gaussian_gamma"

    def _validate_input(self, block: IQBlock) -> None:
        if not isinstance(block, IQBlock):
            raise TypeError("Expected a validated immutable IQBlock")
        if not same_sample_rate(block.stream.sample_rate_hz, self._sample_rate_hz):
            raise PolicyCompatibilityError("Input sample rate differs from detector sample rate")
        # IQBlock guarantees finite complex64 storage backed by immutable bytes.
        # Re-scanning every overlapping hop adds no validation at this boundary.

    def _validate_continuity(self, block: IQBlock, api: str) -> None:
        self._validate_input(block)
        if self._api_mode not in (None, api, 'warmup'):
            raise StreamConsistencyError("Do not mix measurement, streaming and raw preparation APIs without reset")
        if block.dropped_samples_before or block.overflow:
            raise StreamConsistencyError("Direct detector discontinuity requires explicit reset and a clean new-epoch block")
        if self._input_end is not None:
            require_stream(block.stream, self._input_stream)
            if block.start_sample != self._input_end:
                raise StreamConsistencyError("Detector input is not contiguous in the same stream; reset required")
            # Repeated sequence numbers allow subdivisions of one transport block.
            # Jumps are legal if sample coordinates remain continuous.
            if block.sequence_number < self._input_sequence:
                raise StreamConsistencyError("Detector sequence numbers must be nondecreasing")

    def _commit_input(self, block: IQBlock, api: str) -> None:
        self._input_end, self._input_stream = block.end_sample, block.stream
        self._input_sequence, self._api_mode = block.sequence_number, api
        self._path_samples += block.num_samples

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def hop_size(self) -> int:
        return self._hop_size

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def monitored_band_hz(self) -> tuple[float, float] | None:
        return self._monitored_band_hz

    @property
    def filter_delay_samples(self) -> int:
        return self._filter_delay_samples

    @property
    def last_detection_iq(self) -> np.ndarray:
        """The most recent detection-domain (mixed and filtered) IQ array."""
        return getattr(self, "_last_detection_iq", np.empty(0, dtype=np.complex64))

    def prepare_detection_iq(self, iq: np.ndarray | IQBlock) -> np.ndarray:
        """Warmup only. IQBlock anchors continuation; uncoordinated arrays require reset before detection."""
        if isinstance(iq, IQBlock):
            self._validate_continuity(iq, 'warmup')
            if not iq.num_samples:
                return iq.iq
            result = self._prepare_detection_iq(iq.iq)
            self._commit_input(iq, 'warmup')
            return result
        if self._api_mode not in (None, 'raw'):
            raise StreamConsistencyError("Raw preparation cannot share a detector with positioned input; reset required")
        if iq.ndim != 1 or not np.iscomplexobj(iq) or not np.all(np.isfinite(iq)):
            raise ValueError("Expected finite complex 1D IQ")
        if not len(iq):
            return iq
        result = self._prepare_detection_iq(iq)
        self._api_mode = 'raw'
        return result

    def _calculate_thresholds(
        self, noise_estimate: NoiseEstimate, window_size: int | None = None
    ) -> tuple[float, float, float, float]:
        """Compatibility preview: physical dB thresholds (never display-floored)."""
        on, off, noise, db = self._linear_thresholds(noise_estimate, window_size)
        return 10*math.log10(on), 10*math.log10(off), noise, db

    def _linear_thresholds(self, noise_estimate: NoiseEstimate, window_size: int | None = None):
        # One-entry identity cache: immutable references commonly repeat for all
        # hops in a transport block. Strict support still checks its live settling
        # horizon on every call. No recording-length cache or rounded-power key.
        n = self._window_size if window_size is None else window_size
        cached = getattr(self, '_threshold_cache', None)
        if (not self._require_supported_pfa and cached is not None
                and cached[0] is noise_estimate and cached[1] is self._threshold_policy
                and cached[2] is self.reference_domain and cached[3] == n):
            return cached[4]
        result = self._uncached_linear_thresholds(noise_estimate, window_size)
        if not self._require_supported_pfa:
            self._threshold_cache = (noise_estimate, self._threshold_policy, self.reference_domain, n, result)
        return result

    def _uncached_linear_thresholds(self, noise_estimate: NoiseEstimate, window_size: int | None = None):
        if not noise_estimate.valid or not math.isfinite(noise_estimate.noise_power_dbfs):
            raise UncalibratedNoiseError(
                f"Cannot compute detection thresholds: noise estimate is invalid or non-finite "
                f"(valid={noise_estimate.valid}, noise_power_dbfs={noise_estimate.noise_power_dbfs})"
            )

        p_noise_dbfs = float(noise_estimate.noise_power_dbfs)
        if self._ar1_model is not None and noise_estimate.reference_domain is None:
            raise PolicyCompatibilityError("AR1 detection requires an explicitly tagged innovation reference")
        if noise_estimate.reference_domain is not None and noise_estimate.reference_domain != self.reference_domain:
            raise PolicyCompatibilityError("Noise reference domain differs from detector path/gain")
        try:
            p_noise_lin = 10.0 ** (p_noise_dbfs / 10.0)
        except OverflowError as exc:
            raise UncalibratedNoiseError("Noise reference exceeds finite float64 power range") from exc
        if p_noise_lin <= 0.0 or not math.isfinite(p_noise_lin):
            raise UncalibratedNoiseError(
                f"Cannot compute detection thresholds: noise power {p_noise_lin} must be strictly positive"
            )

        n = int(window_size) if window_size is not None else self._window_size
        if self._require_supported_pfa:
            policy = self._threshold_policy
            if policy is None or not policy.supported_pfa or n != policy.window_size:
                raise PolicyCompatibilityError("Held-out evidence requires the validated statistic and actual N")
            if self._path_samples < policy.settling_samples:
                raise PolicyCompatibilityError("Held-out policy requires explicit positioned settling/warmup before detection")
            if not math.isclose(p_noise_lin, policy.mean_power_linear, rel_tol=1e-12):
                raise PolicyCompatibilityError("Held-out evidence requires unchanged absolute thresholds/reference; revalidate scaling")
            t_on_lin, t_off_lin = policy.t_on_linear, policy.t_off_linear
        elif self._threshold_policy is not None and self._threshold_policy.mode == "empirical_matched":
            p_ref = self._threshold_policy.mean_power_linear
            if noise_estimate.reference_domain is not None:
                if noise_estimate.reference_domain != self.reference_domain:
                    raise PolicyCompatibilityError("Noise reference domain differs from detector path/gain")
            elif not math.isclose(p_noise_lin, p_ref, rel_tol=1e-12):
                raise PolicyCompatibilityError(
                    "Untagged noise reference may only replay the fixed calibration mean; "
                    "supply reference_domain for compatible scale updates"
                )
            scale_ratio = p_noise_lin / p_ref if p_ref > 0 else 1.0
            if self._require_supported_pfa and not math.isclose(scale_ratio, 1.0, rel_tol=1e-12):
                raise PolicyCompatibilityError("Held-out support applies to fixed calibrated thresholds; revalidate after scale updates")
            t_on_lin = self._threshold_policy.t_on_linear * scale_ratio
            t_off_lin = self._threshold_policy.t_off_linear * scale_ratio
        else:
            n = int(window_size) if window_size is not None else self._window_size
            if n == self._window_size:
                t_on_lin = float(p_noise_lin * self._unit_q_on)
                t_off_lin = float(p_noise_lin * self._unit_q_off)
            else:
                if self._mode == "configured_band":
                    q_on, q_off = self._configured_quantiles(n)
                    t_on_lin, t_off_lin = p_noise_lin * q_on, p_noise_lin * q_off
                else:
                    scale = p_noise_lin / n
                    t_on_lin = float(gamma.isf(self._pfa_on, a=n, scale=scale))
                    t_off_lin = float(gamma.isf(self._pfa_off, a=n, scale=scale))
        if not 0 < t_off_lin < t_on_lin < math.inf:
            raise UncalibratedNoiseError("Physical thresholds underflow, overflow or collapse at this numerical scale")
        if math.log10(t_on_lin) <= math.log10(t_off_lin):
            raise UncalibratedNoiseError("Threshold logarithms are not numerically distinguishable")
        # Policies also accept NumPy scalar inputs. Normalize so decision flags
        # remain strict Python bools in the public measurement contract.
        return float(t_on_lin), float(t_off_lin), p_noise_lin, p_noise_dbfs

    def _measurement(self, block, start, end, power, thresholds, measurement_domain=None):
        on, off, noise, noise_db = thresholds
        if not math.isfinite(power) or power < 0:
            raise ValueError("Unrepresentable physical window power")
        fields = dict(
            stream_id=block.stream.stream_id, block_sequence_number=block.sequence_number,
            start_sample=start, end_sample=end,
            signal_power_dbfs=10*math.log10(power if power > 0 else self._conversion_floor),
            noise_power_dbfs=noise_db,
            snr_estimate_db=10*(math.log10(power-noise)-math.log10(noise)) if power > noise else 0.0,
            snr_valid=power > noise,
            snr_invalid_reason=None if power > noise else 'nonpositive_excess',
            measurement_domain=measurement_domain or ('ar1_innovation_power' if self._ar1_model else 'iq_power'),
            on_threshold_dbfs=10*math.log10(on), off_threshold_dbfs=10*math.log10(off),
            above_on_threshold=power >= on, below_off_threshold=power < off,
            signal_power_linear=power, noise_power_linear=noise,
            on_threshold_linear=on, off_threshold_linear=off,
            threshold_support_method=('held_out_clopper_pearson_bonferroni' if self._require_supported_pfa
                                      else 'uncertified_training_quantile' if self._threshold_policy is not None
                                      and self._threshold_policy.mode == 'empirical_matched' else 'model_only'),
            threshold_evidence_id=self._threshold_policy.evidence_id if self._require_supported_pfa else None)
        # Trusted internal construction: IQBlock validated metadata/coordinates,
        # _linear_thresholds validated positive ordered finite thresholds/reference,
        # and this method validated physical power and derived every diagnostic.
        # Public DetectionMeasurement(...) still performs the full validation.
        # Avoid repeating ~20 scalar type/log checks at every real-time window.
        result = object.__new__(DetectionMeasurement)
        result.__dict__.update(fields)
        return result

    def _prepare_detection_iq(self, iq: np.ndarray) -> np.ndarray:
        """Apply frequency mixing and filtering if configured_band mode is active."""
        if self._ar1_model is not None and len(iq):
            model = self._ar1_model
            x = iq.astype(np.complex128) - model.mean
            previous = np.empty_like(x)
            previous[0] = self._ar1_previous if self._ar1_previous is not None else 0
            previous[1:] = x[:-1]
            y = x - model.coefficient*previous
            if not np.all(np.isfinite(y)):
                raise ValueError("Unrepresentable AR1 residual")
            self._ar1_previous = complex(x[-1])
            return y  # float64 components preserve representable low physical powers
        if self._mode == "full_band" or self._sos is None:
            return iq

        if len(iq) == 0:
            return iq

        # Frequency shift: multiply by exp(-j 2*pi*f_c*t) with phase continuity
        n = len(iq)
        t = np.arange(n, dtype=np.float64) / self._sample_rate_hz
        phase_rot = self._phase_accumulator - 2.0 * np.pi * self._f_center_hz * t
        mixed = iq * np.exp(1j * phase_rot).astype(np.complex64)
        self._phase_accumulator = float((phase_rot[-1] - 2.0 * np.pi * self._f_center_hz / self._sample_rate_hz) % (2.0 * np.pi))

        # Causal SOS filter with state carried across blocks
        if self._sos_zi is None:
            self._sos_zi = np.zeros((self._sos.shape[0], 2), dtype=np.float64)
        filtered, self._sos_zi = signal.sosfilt(self._sos, mixed, zi=self._sos_zi)
        return filtered.astype(np.complex64)

    def process(
        self,
        block: IQBlock,
        noise_estimate: NoiseEstimate,
    ) -> DetectionMeasurement:
        """Process an IQBlock as a single measurement window to conform to DetectorProtocol."""
        self._validate_continuity(block, 'measurement')
        if self._ar1_windows:
            raise StreamConsistencyError("AR1 bank is a streaming-window API; process_windows is required")
        if self._ar1_model is not None and self._ar1_previous is None:
            raise StreamConsistencyError("AR1 direct measurement requires positioned predecessor warmup")
        if block.num_samples == 0:
            raise ValueError("Input IQBlock must not be empty")

        if not noise_estimate.valid or not math.isfinite(noise_estimate.noise_power_dbfs):
            raise UncalibratedNoiseError(
                f"Cannot compute detection thresholds: noise estimate is invalid or non-finite "
                f"(valid={noise_estimate.valid}, noise_power_dbfs={noise_estimate.noise_power_dbfs})"
            )

        if self._threshold_policy is not None and self._threshold_policy.mode == "empirical_matched" and not self._require_supported_pfa:
            if self._threshold_policy.window_size != block.num_samples:
                raise FilterConfigurationError(
                    f"Empirical threshold policy calibrated for N={self._threshold_policy.window_size} "
                    f"cannot be applied to block length N={block.num_samples}"
                )

        thresholds = self._linear_thresholds(
            noise_estimate, window_size=block.num_samples
        )

        det_iq = self._prepare_detection_iq(block.iq)
        det_c128 = det_iq.astype(np.complex128)
        p_lin = float(np.mean(np.real(det_c128) ** 2 + np.imag(det_c128) ** 2))
        result = self._measurement(block, block.start_sample, block.end_sample, p_lin, thresholds)
        self._commit_input(block, 'measurement')
        return result

    def process_windows(
        self,
        block: IQBlock,
        noise_estimate: NoiseEstimate,
    ) -> list[DetectionMeasurement]:
        """Streaming windowed detection generating trailing N-sample measurements with hop H.

        Carries residual samples across blocks to maintain continuous, boundary-independent
        evaluation anchored to absolute source coordinates.
        """
        self._validate_continuity(block, 'streaming')
        if block.num_samples == 0:
            return []
        # Validate prerequisites BEFORE preparing detection IQ or mutating filter/phase state
        if not noise_estimate.valid or not math.isfinite(noise_estimate.noise_power_dbfs):
            raise UncalibratedNoiseError(
                f"Cannot compute detection thresholds: noise estimate is invalid or non-finite "
                f"(valid={noise_estimate.valid}, noise_power_dbfs={noise_estimate.noise_power_dbfs})"
            )
        thresholds = self._linear_thresholds(noise_estimate)

        unavailable_first = self._ar1_model is not None and self._ar1_previous is None
        det_iq = self._prepare_detection_iq(block.iq)
        if unavailable_first:
            det_iq = det_iq[1:]  # No fictitious predecessor enters a measured window.
        self._last_detection_iq = det_iq

        if len(self._residual_iq) == 0:
            combined_iq = det_iq
            start_coord = block.start_sample + int(unavailable_first)
        else:
            combined_iq = np.concatenate([self._residual_iq, det_iq])
            start_coord = self._residual_start_sample

        total_samples = len(combined_iq)

        measurements: list[DetectionMeasurement] = []

        # Direct reductions exclude every sample outside the actual window.
        # Prefix differences are unsafe even in float64 after a large transient.
        combined_c128 = combined_iq.astype(np.complex128)

        # Determine starting window index
        w_start = max(start_coord, self._next_window_sample)
        # Align w_start to hop
        if self._next_window_sample > 0 and w_start > start_coord:
            rel_start = w_start - start_coord
        else:
            rel_start = 0

        curr_rel = rel_start
        while curr_rel + self._window_size <= total_samples:
            p_lin = window_mean_power(combined_c128[curr_rel:curr_rel + self._window_size])

            abs_win_start = start_coord + curr_rel
            abs_win_end = abs_win_start + self._window_size

            if self._ar1_windows:
                candidates = []
                for size in self._ar1_windows:
                    value = window_mean_power(combined_c128[curr_rel+self._window_size-size:curr_rel+self._window_size])
                    qon, qoff = self._ar1_quantiles[size]
                    branch_thresholds = (thresholds[2]*qon, thresholds[2]*qoff, thresholds[2], thresholds[3])
                    candidates.append((value,branch_thresholds,size,'ar1_innovation_power'))
                if self._ar1_subbands:
                    spectrum = np.fft.fft(combined_c128[curr_rel:curr_rel+self._window_size])
                    bin_powers = (spectrum.real**2+spectrum.imag**2)/self._window_size
                    qon,qoff = self._ar1_quantiles[32]
                    band_thresholds = (thresholds[2]*qon,thresholds[2]*qoff,thresholds[2],thresholds[3])
                    for bins in self._ar1_band_bins:
                        candidates.append((float(np.mean(bin_powers[bins])),band_thresholds,self._window_size,'ar1_subband_bin_mean_power'))
                on = [c for c in candidates if c[0] >= c[1][0]]
                selected = max(on,key=lambda c:c[0]/c[1][0]) if on else max(candidates,key=lambda c:c[0]/c[1][1])
                value, branch_thresholds, size, domain = selected
                measurements.append(self._measurement(block, abs_win_end-size, abs_win_end, value, branch_thresholds,domain))
            else:
                measurements.append(self._measurement(block, abs_win_start, abs_win_end, p_lin, thresholds))

            curr_rel += self._hop_size

        # Retain residual unwindowed samples
        retained_start_rel = curr_rel
        if retained_start_rel < total_samples:
            self._residual_iq = combined_iq[retained_start_rel:].copy()
            self._residual_start_sample = start_coord + retained_start_rel
            self._next_window_sample = start_coord + curr_rel
        else:
            self._residual_iq = np.empty(0, dtype=np.complex64)
            self._residual_start_sample = start_coord + total_samples
            self._next_window_sample = start_coord + curr_rel

        self._commit_input(block, 'streaming')
        return measurements

    def _is_quiet_full_band(self, block: IQBlock, noise_estimate: NoiseEstimate) -> bool:
        """Pure lookahead for the pipeline's fixed-reference idle optimization.

        Does not admit samples or mutate residual/filter/reference state. Requires
        all complete windows below OFF (hence also below ON). The caller still
        uses process_windows for actual validation and committed processing.
        """
        return self._prepare_quiet_full_band(block, noise_estimate) is not None

    def _prepare_full_band(self, block: IQBlock) -> _FullBandCursor:
        """Bounded pure preparation for active as well as quiet pipeline input.

        No thresholds, adaptation, filter progress or received/committed clocks
        are advanced. Every reduction uses only its own stored complex64 window.
        Configured filtering intentionally retains the reference execution path.
        """
        if self._mode != 'full_band' or self._ar1_model is not None or block.num_samples > 4096:
            raise ValueError('Full-band preparation requires at most 4096 samples')
        self._validate_continuity(block, 'streaming')
        combined = np.concatenate((self._residual_iq, block.iq)) if len(self._residual_iq) else block.iq
        origin = self._residual_start_sample if len(self._residual_iq) else block.start_sample
        first = max(0, self._next_window_sample-origin)
        values = combined.astype(np.complex128)
        powers = tuple(map(float, window_mean_powers(values[first:], self._window_size, self._hop_size)))
        return _FullBandCursor(self, block, combined, origin, first, powers, 0,
                              self._input_end, self._residual_iq, self._next_window_sample,
                              self._threshold_policy)

    def _commit_full_band(self, cursor, block, noise_estimate, *, quiet=False, active=False, idle=False):
        """Commit only the next received prefix, with the reference valid NOW.

        A failed threshold check leaves the cursor and detector untouched. Future
        IQ in the preparation never enters detector residuals or input clocks.
        Quiet collapsing is authorized only by the pipeline's idle/fixed guard.
        """
        if (cursor.owner is not self or cursor.expected_end != self._input_end
                or cursor.expected_residual is not self._residual_iq
                or cursor.expected_next != self._next_window_sample
                or cursor.policy is not self._threshold_policy):
            raise StreamConsistencyError('Stale or foreign full-band preparation')
        expected = cursor.block.start_sample if cursor.expected_end is None else cursor.expected_end
        if block.start_sample != expected or block.end_sample > cursor.block.end_sample:
            raise StreamConsistencyError('Prepared commit must be the next bounded prefix')
        self._validate_continuity(block, 'streaming')
        if not noise_estimate.valid or not math.isfinite(noise_estimate.noise_power_dbfs):
            raise UncalibratedNoiseError('Cannot commit windows with an invalid noise reference')
        thresholds = self._linear_thresholds(noise_estimate)
        stop = min(len(cursor.powers), max(0, (block.end_sample-cursor.origin-cursor.first-self._window_size)//self._hop_size+1))
        if quiet and any(p >= thresholds[1] for p in cursor.powers[cursor.index:stop]):
            raise ValueError('Cannot collapse nonquiet decisions')
        if idle and any(p >= thresholds[0] for p in cursor.powers[cursor.index:stop]):
            raise ValueError('Cannot collapse above-ON idle decisions')
        if active and any(p < thresholds[1] for p in cursor.powers[cursor.index:stop]):
            raise ValueError('Cannot collapse a below-OFF active decision')
        indices = range(max(cursor.index, stop-1) if quiet or active or idle else cursor.index, stop)
        measurements = [self._measurement(block, cursor.origin+cursor.first+i*self._hop_size,
                        cursor.origin+cursor.first+i*self._hop_size+self._window_size,
                        cursor.powers[i], thresholds) for i in indices]
        retained = cursor.first+stop*self._hop_size
        self._residual_iq = cursor.combined[retained:block.end_sample-cursor.origin].copy()
        self._residual_start_sample = cursor.origin+retained
        self._next_window_sample = self._residual_start_sample
        self._last_detection_iq = block.iq
        self._commit_input(block, 'streaming')
        cursor.index, cursor.expected_end = stop, self._input_end
        cursor.expected_residual, cursor.expected_next = self._residual_iq, self._next_window_sample
        return measurements

    def _prepare_quiet_full_band(self, block: IQBlock, noise_estimate: NoiseEstimate):
        """Compute every eligible quiet power once, without state commitment.

        Used only by the pipeline's bounded, fixed-reference idle fast path.
        Nonquiet input falls back to the ordinary chronological execution path.
        The prepared object is consumed immediately, never cached across calls.
        """
        if self._mode != 'full_band' or self._ar1_model is not None:
            return None
        self._validate_continuity(block, 'streaming')
        if not noise_estimate.valid or not math.isfinite(noise_estimate.noise_power_dbfs):
            raise UncalibratedNoiseError('Cannot prepare windows with an invalid noise reference')
        thresholds = self._linear_thresholds(noise_estimate)
        off = thresholds[1]
        combined = np.concatenate((self._residual_iq, block.iq)) if len(self._residual_iq) else block.iq
        origin = self._residual_start_sample if len(self._residual_iq) else block.start_sample
        first = max(0, self._next_window_sample-origin)
        values = combined.astype(np.complex128)
        count = 0
        for start in range(first, len(values)-self._window_size+1, self._hop_size):
            power = window_mean_power(values[start:start+self._window_size])
            if not math.isfinite(power) or power < 0:
                raise ValueError('Unrepresentable physical window power')
            if power >= off:
                return None
            count += 1
        if not count:
            return None
        return _QuietFullBandBatch(self, block, self._residual_iq, self._input_end,
            self._next_window_sample, self._threshold_policy, combined, origin,
            first+count*self._hop_size, power, thresholds)

    def _commit_quiet_full_band(self, prepared: _QuietFullBandBatch) -> DetectionMeasurement:
        """Commit a validated quiet preparation, returning only its last decision.

        Only an idle segmenter without an in-flight capture may collapse these
        strictly-below-OFF decisions. Public process_windows still returns all
        windows. Reject stale/foreign preparations before any streaming mutation.
        """
        if (prepared.owner is not self or prepared.prior_residual is not self._residual_iq
                or prepared.prior_input_end != self._input_end
                or prepared.prior_next_window != self._next_window_sample
                or prepared.prior_policy is not self._threshold_policy):
            raise StreamConsistencyError('Stale or foreign quiet-window preparation')
        self._validate_continuity(prepared.block, 'streaming')
        start = prepared.origin+prepared.retained_start-self._hop_size
        measurement = self._measurement(prepared.block, start, start+self._window_size,
                                        prepared.last_power, prepared.thresholds)
        self._residual_iq = prepared.combined[prepared.retained_start:].copy()
        self._residual_start_sample = prepared.origin+prepared.retained_start
        self._next_window_sample = self._residual_start_sample
        self._last_detection_iq = prepared.block.iq
        self._commit_input(prepared.block, 'streaming')
        return measurement

    def reset(self) -> None:
        """Reset internal streaming filter and windowing state."""
        self._residual_iq = np.empty(0, dtype=np.complex64)
        self._residual_start_sample = 0
        self._next_window_sample = 0
        self._phase_accumulator = 0.0
        self._ar1_previous = None
        self._input_end = self._input_stream = self._input_sequence = self._api_mode = None
        self._path_samples = 0
        if self._sos is not None:
            self._sos_zi = np.zeros((self._sos.shape[0], 2), dtype=np.float64)
