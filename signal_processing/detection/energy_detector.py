"""Causal windowed energy detector with full-band and configured-band monitoring."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy import signal
from scipy.stats import gamma

from signal_processing.core.contracts import DetectionMeasurement, IQBlock, NoiseEstimate, NoiseReferenceDomain
from signal_processing.core.interfaces import DetectorProtocol, Resettable
from signal_processing.detection.exceptions import (
    FilterConfigurationError,
    PolicyCompatibilityError,
    UncalibratedNoiseError,
)
from signal_processing.detection.noise_estimation import ThresholdPolicy


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

        self._unit_q_on = float(gamma.isf(self._pfa_on, a=self._window_size, scale=1.0 / self._window_size))
        self._unit_q_off = float(gamma.isf(self._pfa_off, a=self._window_size, scale=1.0 / self._window_size))

        # Stateful streaming buffers
        self._residual_iq: np.ndarray = np.empty(0, dtype=np.complex64)
        self._residual_start_sample: int = 0
        self._next_window_sample: int = 0
        self._phase_accumulator: float = 0.0

        # Filter setup for configured-band
        self._sos: np.ndarray | None = None
        self._sos_zi: np.ndarray | None = None
        self._sos_hash: str | None = None
        self._filter_delay_samples: int = 0
        if self._mode == "configured_band":
            self._init_configured_band_filter(filter_order)

        self.reference_domain = NoiseReferenceDomain(
            self._sample_rate_hz, self._monitored_band_hz, self._sos_hash, gain_id
        )
        self._require_supported_pfa = require_supported_pfa
        self._threshold_policy: ThresholdPolicy | None = None
        if threshold_policy is not None:
            self.set_threshold_policy(threshold_policy)

    def set_threshold_policy(self, policy: ThresholdPolicy) -> None:
        """Set and validate an empirical or analytic threshold policy."""
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
        if not math.isclose(policy.sample_rate_hz, self._sample_rate_hz, rel_tol=1e-5):
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
        return "stationary_gaussian_quadratic_bound" if self._mode == "configured_band" else "iid_complex_gaussian_gamma"

    def _validate_input(self, block: IQBlock) -> None:
        if not math.isclose(block.stream.sample_rate_hz, self._sample_rate_hz, rel_tol=1e-12):
            raise PolicyCompatibilityError("Input sample rate differs from detector sample rate")
        if not np.all(np.isfinite(block.iq)):
            raise ValueError("Input IQ must contain only finite samples")

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

    def prepare_detection_iq(self, iq: np.ndarray) -> np.ndarray:
        """Apply frequency downmixing and filtering to convert raw IQ to detection domain."""
        return self._prepare_detection_iq(iq)

    def _calculate_thresholds(
        self, noise_estimate: NoiseEstimate, window_size: int | None = None
    ) -> tuple[float, float, float, float]:
        """Calculate (t_on_dbfs, t_off_dbfs, p_noise_lin, p_noise_dbfs)."""
        if not noise_estimate.valid or not math.isfinite(noise_estimate.noise_power_dbfs):
            raise UncalibratedNoiseError(
                f"Cannot compute detection thresholds: noise estimate is invalid or non-finite "
                f"(valid={noise_estimate.valid}, noise_power_dbfs={noise_estimate.noise_power_dbfs})"
            )

        p_noise_dbfs = float(noise_estimate.noise_power_dbfs)
        if noise_estimate.reference_domain is not None and noise_estimate.reference_domain != self.reference_domain:
            raise PolicyCompatibilityError("Noise reference domain differs from detector path/gain")
        p_noise_lin = 10.0 ** (p_noise_dbfs / 10.0)
        if p_noise_lin <= 0.0 or not math.isfinite(p_noise_lin):
            raise UncalibratedNoiseError(
                f"Cannot compute detection thresholds: noise power {p_noise_lin} must be strictly positive"
            )

        if self._threshold_policy is not None and self._threshold_policy.mode == "empirical_matched":
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
            t_on_dbfs = 10.0 * math.log10(max(t_on_lin, self._conversion_floor))
            t_off_dbfs = 10.0 * math.log10(max(t_off_lin, self._conversion_floor))
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
            t_on_dbfs = 10.0 * math.log10(max(t_on_lin, self._conversion_floor))
            t_off_dbfs = 10.0 * math.log10(max(t_off_lin, self._conversion_floor))

        # Enforce strict hysteresis invariant: on > off
        if t_on_dbfs <= t_off_dbfs:
            t_on_dbfs = t_off_dbfs + 0.1

        return t_on_dbfs, t_off_dbfs, p_noise_lin, p_noise_dbfs

    def _prepare_detection_iq(self, iq: np.ndarray) -> np.ndarray:
        """Apply frequency mixing and filtering if configured_band mode is active."""
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
        self._validate_input(block)
        if block.num_samples == 0:
            raise ValueError("Input IQBlock must not be empty")

        if not noise_estimate.valid or not math.isfinite(noise_estimate.noise_power_dbfs):
            raise UncalibratedNoiseError(
                f"Cannot compute detection thresholds: noise estimate is invalid or non-finite "
                f"(valid={noise_estimate.valid}, noise_power_dbfs={noise_estimate.noise_power_dbfs})"
            )

        if self._threshold_policy is not None and self._threshold_policy.mode == "empirical_matched":
            if self._threshold_policy.window_size != block.num_samples:
                raise FilterConfigurationError(
                    f"Empirical threshold policy calibrated for N={self._threshold_policy.window_size} "
                    f"cannot be applied to block length N={block.num_samples}"
                )

        t_on_dbfs, t_off_dbfs, p_noise_lin, p_noise_dbfs = self._calculate_thresholds(
            noise_estimate, window_size=block.num_samples
        )

        det_iq = self._prepare_detection_iq(block.iq)
        det_c128 = det_iq.astype(np.complex128)
        p_lin = float(np.mean(np.real(det_c128) ** 2 + np.imag(det_c128) ** 2))
        p_dbfs = 10.0 * math.log10(max(p_lin, self._conversion_floor))
        snr_db = 10.0 * math.log10(max(p_lin - p_noise_lin, self._conversion_floor) / p_noise_lin)

        return DetectionMeasurement(
            stream_id=block.stream.stream_id,
            block_sequence_number=block.sequence_number,
            start_sample=block.start_sample,
            end_sample=block.end_sample,
            signal_power_dbfs=p_dbfs,
            noise_power_dbfs=p_noise_dbfs,
            snr_estimate_db=snr_db,
            on_threshold_dbfs=t_on_dbfs,
            off_threshold_dbfs=t_off_dbfs,
            above_on_threshold=p_dbfs >= t_on_dbfs,
            below_off_threshold=p_dbfs < t_off_dbfs,
        )

    def process_windows(
        self,
        block: IQBlock,
        noise_estimate: NoiseEstimate,
    ) -> list[DetectionMeasurement]:
        """Streaming windowed detection generating trailing N-sample measurements with hop H.

        Carries residual samples across blocks to maintain continuous, boundary-independent
        evaluation anchored to absolute source coordinates.
        """
        self._validate_input(block)
        # Validate prerequisites BEFORE preparing detection IQ or mutating filter/phase state
        if not noise_estimate.valid or not math.isfinite(noise_estimate.noise_power_dbfs):
            raise UncalibratedNoiseError(
                f"Cannot compute detection thresholds: noise estimate is invalid or non-finite "
                f"(valid={noise_estimate.valid}, noise_power_dbfs={noise_estimate.noise_power_dbfs})"
            )
        t_on_dbfs, t_off_dbfs, p_noise_lin, p_noise_dbfs = self._calculate_thresholds(noise_estimate)

        det_iq = self._prepare_detection_iq(block.iq)
        self._last_detection_iq = det_iq

        if len(self._residual_iq) == 0:
            combined_iq = det_iq
            start_coord = block.start_sample
        else:
            combined_iq = np.concatenate([self._residual_iq, det_iq])
            start_coord = self._residual_start_sample

        total_samples = len(combined_iq)

        measurements: list[DetectionMeasurement] = []

        # Convert combined IQ to float64 squared magnitude for rolling sum
        combined_c128 = combined_iq.astype(np.complex128)
        mag_sq = np.real(combined_c128) ** 2 + np.imag(combined_c128) ** 2

        # Cumsum prefix array for O(1) window power extraction
        prefix = np.empty(total_samples + 1, dtype=np.float64)
        prefix[0] = 0.0
        np.cumsum(mag_sq, out=prefix[1:])

        # Determine starting window index
        w_start = max(start_coord, self._next_window_sample)
        # Align w_start to hop
        if self._next_window_sample > 0 and w_start > start_coord:
            rel_start = w_start - start_coord
        else:
            rel_start = 0

        curr_rel = rel_start
        while curr_rel + self._window_size <= total_samples:
            p_sum = prefix[curr_rel + self._window_size] - prefix[curr_rel]
            p_lin = float(p_sum / self._window_size)
            p_dbfs = 10.0 * math.log10(max(p_lin, self._conversion_floor))
            snr_db = 10.0 * math.log10(max(p_lin - p_noise_lin, self._conversion_floor) / p_noise_lin)

            abs_win_start = start_coord + curr_rel
            abs_win_end = abs_win_start + self._window_size

            measurements.append(
                DetectionMeasurement(
                    stream_id=block.stream.stream_id,
                    block_sequence_number=block.sequence_number,
                    start_sample=abs_win_start,
                    end_sample=abs_win_end,
                    signal_power_dbfs=p_dbfs,
                    noise_power_dbfs=p_noise_dbfs,
                    snr_estimate_db=snr_db,
                    on_threshold_dbfs=t_on_dbfs,
                    off_threshold_dbfs=t_off_dbfs,
                    above_on_threshold=p_dbfs >= t_on_dbfs,
                    below_off_threshold=p_dbfs < t_off_dbfs,
                )
            )

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

        return measurements

    def reset(self) -> None:
        """Reset internal streaming filter and windowing state."""
        self._residual_iq = np.empty(0, dtype=np.complex64)
        self._residual_start_sample = 0
        self._next_window_sample = 0
        self._phase_accumulator = 0.0
        if self._sos is not None:
            self._sos_zi = np.zeros((self._sos.shape[0], 2), dtype=np.float64)
