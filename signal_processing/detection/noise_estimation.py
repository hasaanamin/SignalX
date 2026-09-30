"""Noise power estimation and statistical CFAR threshold policies."""

from __future__ import annotations

import enum
import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from scipy import signal
from scipy.stats import beta, gamma

from signal_processing.core.contracts import IQBlock, NoiseEstimate, NoiseReferenceDomain
from signal_processing.core.exceptions import ContractValidationError
from signal_processing.core.interfaces import NoiseEstimatorProtocol, Resettable
from signal_processing.detection.exceptions import PolicyCompatibilityError


class NoiseLifecycleState(enum.Enum):
    """Lifecycle states for noise estimation and calibration."""

    UNCALIBRATED = "UNCALIBRATED"
    CALIBRATING = "CALIBRATING"
    VALID = "VALID"
    STALE = "STALE"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ThresholdPolicy:
    """Immutable threshold estimate with separate, conditional support evidence."""

    mode: Literal["analytic_gamma", "empirical_matched"]
    window_size: int
    hop_size: int
    pfa_on: float
    pfa_off: float
    mean_power_linear: float
    t_on_linear: float
    t_off_linear: float
    t_on_dbfs: float
    t_off_dbfs: float
    supported_pfa: bool = False
    sample_rate_hz: float = 1e6
    monitored_band_hz: tuple[float, float] | None = None
    filter_sos_hash: str | None = None
    training_samples: int = 0
    training_windows: int = 0
    policy_version: str = "3.0"
    valid: bool = True
    reason: str | None = None
    reference_domain: NoiseReferenceDomain | None = None
    support_method: str = "uncertified"
    dependence_assumption: str = "unspecified"
    confidence_level: float | None = None
    validation_windows: int = 0
    validation_exceedances_on: int = 0
    validation_exceedances_off: int = 0
    pfa_upper_on: float | None = None
    pfa_upper_off: float | None = None
    evidence_id: str | None = None
    settling_samples: int = 0

    def __post_init__(self) -> None:
        if type(self.valid) is not bool:
            raise ContractValidationError(f"valid must be a strict boolean, got {self.valid!r}")
        if type(self.supported_pfa) is not bool:
            raise ContractValidationError(f"supported_pfa must be a strict boolean, got {self.supported_pfa!r}")
        if self.mode not in ("analytic_gamma", "empirical_matched"):
            raise ContractValidationError(f"mode must be 'analytic_gamma' or 'empirical_matched', got {self.mode!r}")
        if isinstance(self.window_size, bool) or not isinstance(self.window_size, (int, np.integer)) or self.window_size < 1:
            raise ContractValidationError(f"window_size must be an integer >= 1, got {self.window_size!r}")
        if isinstance(self.hop_size, bool) or not isinstance(self.hop_size, (int, np.integer)) or self.hop_size < 1:
            raise ContractValidationError(f"hop_size must be an integer >= 1, got {self.hop_size!r}")
        if self.hop_size > self.window_size:
            raise ContractValidationError(f"hop_size ({self.hop_size}) must be <= window_size ({self.window_size})")
        if (
            isinstance(self.pfa_on, bool)
            or isinstance(self.pfa_off, bool)
            or not (0.0 < float(self.pfa_on) < float(self.pfa_off) < 1.0)
        ):
            raise ContractValidationError(f"Probabilities must satisfy 0 < pfa_on < pfa_off < 1, got pfa_on={self.pfa_on}, pfa_off={self.pfa_off}")
        if (
            isinstance(self.mean_power_linear, bool)
            or not isinstance(self.mean_power_linear, (int, float, np.floating, np.integer))
            or not math.isfinite(float(self.mean_power_linear))
            or float(self.mean_power_linear) <= 0.0
        ):
            raise ContractValidationError(f"mean_power_linear must be strictly positive and finite, got {self.mean_power_linear!r}")
        if (
            isinstance(self.t_on_linear, bool)
            or not isinstance(self.t_on_linear, (int, float, np.floating, np.integer))
            or not math.isfinite(float(self.t_on_linear))
            or float(self.t_on_linear) <= 0.0
        ):
            raise ContractValidationError(f"t_on_linear must be strictly positive and finite, got {self.t_on_linear!r}")
        if (
            isinstance(self.t_off_linear, bool)
            or not isinstance(self.t_off_linear, (int, float, np.floating, np.integer))
            or not math.isfinite(float(self.t_off_linear))
            or float(self.t_off_linear) <= 0.0
        ):
            raise ContractValidationError(f"t_off_linear must be strictly positive and finite, got {self.t_off_linear!r}")
        if float(self.t_on_linear) <= float(self.t_off_linear):
            raise ContractValidationError(f"Threshold invariant violated: t_on_linear ({self.t_on_linear}) must be > t_off_linear ({self.t_off_linear})")
        if (
            not math.isfinite(float(self.t_on_dbfs))
            or not math.isfinite(float(self.t_off_dbfs))
            or float(self.t_on_dbfs) <= float(self.t_off_dbfs)
        ):
            raise ContractValidationError(f"Threshold dBFS invariant violated: t_on_dbfs ({self.t_on_dbfs}) must be > t_off_dbfs ({self.t_off_dbfs})")
        exp_on_dbfs = 10.0 * math.log10(float(self.t_on_linear))
        if not math.isclose(float(self.t_on_dbfs), exp_on_dbfs, rel_tol=0, abs_tol=1e-9):
            raise ContractValidationError(
                f"Contradictory linear/dBFS threshold: t_on_linear={self.t_on_linear} implies ~{exp_on_dbfs:.2f} dBFS, but t_on_dbfs={self.t_on_dbfs}"
            )
        exp_off_dbfs = 10.0 * math.log10(float(self.t_off_linear))
        if not math.isclose(float(self.t_off_dbfs), exp_off_dbfs, rel_tol=0, abs_tol=1e-9):
            raise ContractValidationError(
                f"Contradictory linear/dBFS threshold: t_off_linear={self.t_off_linear} implies ~{exp_off_dbfs:.2f} dBFS, but t_off_dbfs={self.t_off_dbfs}"
            )
        if (
            isinstance(self.sample_rate_hz, bool)
            or not isinstance(self.sample_rate_hz, (int, float, np.floating, np.integer))
            or not math.isfinite(float(self.sample_rate_hz))
            or float(self.sample_rate_hz) <= 0.0
        ):
            raise ContractValidationError(f"sample_rate_hz must be strictly positive and finite, got {self.sample_rate_hz!r}")
        if self.monitored_band_hz is not None:
            if not isinstance(self.monitored_band_hz, tuple):
                raise ContractValidationError(f"monitored_band_hz must be an immutable tuple, got {type(self.monitored_band_hz).__name__}")
            if len(self.monitored_band_hz) != 2:
                raise ContractValidationError("monitored_band_hz must have exactly 2 elements (f_low, f_high)")
            f_low, f_high = self.monitored_band_hz
            nyq = float(self.sample_rate_hz) / 2.0
            if not (-nyq <= f_low < f_high <= nyq):
                raise ContractValidationError(f"monitored_band_hz ({f_low}, {f_high}) outside valid Nyquist interval [-{nyq}, {nyq}]")
        if isinstance(self.training_samples, bool) or not isinstance(self.training_samples, (int, np.integer)) or self.training_samples < 0:
            raise ContractValidationError(f"training_samples must be an integer >= 0, got {self.training_samples!r}")
        if isinstance(self.training_windows, bool) or not isinstance(self.training_windows, (int, np.integer)) or self.training_windows < 0:
            raise ContractValidationError(f"training_windows must be an integer >= 0, got {self.training_windows!r}")
        if self.reference_domain is not None:
            expected = NoiseReferenceDomain(self.sample_rate_hz, self.monitored_band_hz,
                                            self.filter_sos_hash, self.reference_domain.gain_id)
            if self.reference_domain != expected:
                raise ContractValidationError("Reference domain contradicts policy path")
        if self.mode == "empirical_matched" and self.supported_pfa and self.support_method != "held_out_clopper_pearson_bonferroni":
            raise ContractValidationError("Empirical support requires explicit held-out confidence evidence")
        if self.support_method == "held_out_clopper_pearson_bonferroni":
            if self.confidence_level is None or not 0 < self.confidence_level < 1:
                raise ContractValidationError("Invalid confidence level")
            if self.validation_windows < 1 or not self.evidence_id:
                raise ContractValidationError("Held-out support requires evidence and independent windows")
            if self.dependence_assumption != "independent_held_out_windows":
                raise ContractValidationError("Held-out support requires an explicit independence assumption")
            for count in (self.validation_exceedances_on, self.validation_exceedances_off):
                if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or not 0 <= count <= self.validation_windows:
                    raise ContractValidationError("Invalid held-out exceedance count")
            level = 1 - (1 - self.confidence_level) / 2
            for count, upper in ((self.validation_exceedances_on, self.pfa_upper_on), (self.validation_exceedances_off, self.pfa_upper_off)):
                expected = 1.0 if count == self.validation_windows else float(beta.ppf(level, count + 1, self.validation_windows - count))
                if upper is None or not math.isclose(upper, expected, rel_tol=1e-12):
                    raise ContractValidationError("Confidence bound contradicts held-out counts")
            if self.supported_pfa and (self.pfa_upper_on is None or self.pfa_upper_off is None
                                      or self.pfa_upper_on > self.pfa_on or self.pfa_upper_off > self.pfa_off):
                raise ContractValidationError("Certified support contradicts confidence bounds")

    def validate_held_out(
        self, window_powers: np.ndarray, *, reference_domain: NoiseReferenceDomain,
        independence_assumption: str, evidence_id: str, confidence_level: float = 0.95,
    ) -> ThresholdPolicy:
        """Validate fixed thresholds on caller-authorized independent held-out powers.

        No independence is inferred from sample spacing or filter output. Both
        ON/OFF upper bounds have joint confidence at least confidence_level via
        Bonferroni. Bounds apply to these absolute thresholds and this domain;
        rescaling to a newly estimated reference requires new validation.
        """
        if reference_domain != self.reference_domain or self.reference_domain is None:
            raise PolicyCompatibilityError("Held-out evidence must identify the calibration domain")
        if independence_assumption != "independent_held_out_windows":
            raise ValueError("Declare independent_held_out_windows; correlated data cannot certify support")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise ValueError("evidence_id must identify the held-out acquisition")
        if isinstance(confidence_level, bool) or not 0 < confidence_level < 1:
            raise ValueError("confidence_level must lie in (0, 1)")
        powers = np.asarray(window_powers)
        if powers.ndim != 1 or len(powers) == 0 or np.iscomplexobj(powers):
            raise ValueError("Expected nonempty real window powers")
        if not np.all(np.isfinite(powers)) or np.any(powers < 0):
            raise ValueError("Window powers must be finite and nonnegative")
        n = len(powers)
        on, off = int(np.count_nonzero(powers >= self.t_on_linear)), int(np.count_nonzero(powers >= self.t_off_linear))
        level = 1 - (1 - confidence_level) / 2
        def upper(k: int) -> float:
            return 1.0 if k == n else float(beta.ppf(level, k + 1, n - k))
        u_on, u_off = upper(on), upper(off)
        return replace(self, supported_pfa=u_on <= self.pfa_on and u_off <= self.pfa_off,
                       support_method="held_out_clopper_pearson_bonferroni",
                       dependence_assumption=independence_assumption, evidence_id=evidence_id,
                       confidence_level=confidence_level, validation_windows=n,
                       validation_exceedances_on=on, validation_exceedances_off=off,
                       pfa_upper_on=u_on, pfa_upper_off=u_off)


class GaussianNoiseEstimator(NoiseEstimatorProtocol, Resettable):
    """Calibrated Gaussian-reference noise estimator with exact Gamma CFAR thresholds.

    For independent circular complex Gaussian noise with mean sample power sigma2,
    the average power of an N-sample rectangular window follows:
        Gamma(shape=N, scale=sigma2/N).

    Thresholds are derived using the inverse survival function (isf):
        T_on = gamma.isf(pfa_on, a=N, scale=sigma2/N)
        T_off = gamma.isf(pfa_off, a=N, scale=sigma2/N)
    with 0 < pfa_on < pfa_off < 1 ensuring T_on > T_off.
    """

    def __init__(
        self,
        window_size: int = 256,
        pfa_on: float = 1e-4,
        pfa_off: float = 1e-2,
        calibrated_noise_power_linear: float | None = None,
        warmup_blocks: int = 0,
        warmup_samples: int | None = None,
        conversion_floor_linear: float = 1e-15,
        reference_domain: NoiseReferenceDomain | None = None,
        warmup_settling_samples: int = 0,
    ) -> None:
        if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)) or window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size!r}")
        if not (0.0 < pfa_on < pfa_off < 1.0):
            raise ValueError(
                f"Probabilities of false alarm must satisfy 0 < pfa_on < pfa_off < 1, "
                f"got pfa_on={pfa_on}, pfa_off={pfa_off}"
            )
        if isinstance(warmup_blocks, bool) or not isinstance(warmup_blocks, (int, np.integer)) or warmup_blocks < 0:
            raise ValueError(f"warmup_blocks must be an integer >= 0, got {warmup_blocks!r}")
        if warmup_samples is not None:
            if isinstance(warmup_samples, bool) or not isinstance(warmup_samples, (int, np.integer)) or warmup_samples < 0:
                raise ValueError(f"warmup_samples must be an integer >= 0, got {warmup_samples!r}")

        if calibrated_noise_power_linear is not None:
            if (
                isinstance(calibrated_noise_power_linear, bool)
                or calibrated_noise_power_linear <= 0.0
                or not math.isfinite(calibrated_noise_power_linear)
            ):
                raise ValueError("calibrated_noise_power_linear must be strictly positive and finite")

        if (
            isinstance(conversion_floor_linear, bool)
            or not isinstance(conversion_floor_linear, (int, float, np.floating, np.integer))
            or not math.isfinite(float(conversion_floor_linear))
            or float(conversion_floor_linear) <= 0.0
        ):
            raise ValueError(f"conversion_floor_linear must be strictly positive and finite, got {conversion_floor_linear!r}")

        if isinstance(warmup_settling_samples, bool) or not isinstance(warmup_settling_samples, (int, np.integer)) or warmup_settling_samples < 0:
            raise ValueError("warmup_settling_samples must be a nonnegative integer")
        self._warmup_settling_total = int(warmup_settling_samples)
        self._warmup_settling_remaining = self._warmup_settling_total
        self._window_size = int(window_size)
        self.reference_domain = reference_domain
        self._pfa_on = float(pfa_on)
        self._pfa_off = float(pfa_off)
        self._conversion_floor = float(conversion_floor_linear)

        self._initial_noise_power = (
            float(calibrated_noise_power_linear) if calibrated_noise_power_linear is not None else None
        )
        self._noise_power_linear: float | None = self._initial_noise_power
        self._warmup_blocks = int(warmup_blocks)
        if warmup_samples is not None:
            self._warmup_samples = int(warmup_samples)
            self._target_warmup_samples = self._warmup_samples
        else:
            self._warmup_samples = self._warmup_blocks * self._window_size
            self._target_warmup_samples = self._warmup_samples

        self._warmup_power_sum: float = 0.0
        self._warmup_samples_collected: int = 0
        self._frozen: bool = False
        self._suspended: bool = False

        if self._initial_noise_power is not None:
            self._state = NoiseLifecycleState.VALID
        elif self._target_warmup_samples > 0:
            self._state = NoiseLifecycleState.CALIBRATING
        else:
            self._state = NoiseLifecycleState.UNCALIBRATED

    @property
    def window_size(self) -> int:
        """Integration window length N."""
        return self._window_size

    @property
    def pfa_on(self) -> float:
        """False alarm probability for trigger activation."""
        return self._pfa_on

    @property
    def pfa_off(self) -> float:
        """False alarm probability for trigger release."""
        return self._pfa_off

    @property
    def state(self) -> NoiseLifecycleState:
        """Current lifecycle state."""
        return self._state

    @property
    def noise_power_linear(self) -> float | None:
        """Current estimated linear noise power per sample, or None if uncalibrated."""
        return self._noise_power_linear

    @property
    def is_calibrated(self) -> bool:
        """Whether a trustworthy calibrated noise reference is currently available."""
        return (
            self._state == NoiseLifecycleState.VALID
            and self._noise_power_linear is not None
            and math.isfinite(self._noise_power_linear)
            and self._noise_power_linear > 0.0
        )

    def freeze(self, frozen: bool) -> None:
        """Freeze/unfreeze noise adaptation during active signal bursts."""
        self._frozen = bool(frozen)

    def get_thresholds_linear(self) -> tuple[float, float]:
        """Compute (T_on, T_off) in linear power units."""
        if not self.is_calibrated or self._noise_power_linear is None:
            raise RuntimeError("Cannot compute thresholds: noise estimator is uncalibrated")
        scale = self._noise_power_linear / self._window_size
        t_on = float(gamma.isf(self._pfa_on, a=self._window_size, scale=scale))
        t_off = float(gamma.isf(self._pfa_off, a=self._window_size, scale=scale))
        return t_on, t_off

    def get_thresholds_dbfs(self) -> tuple[float, float]:
        """Compute (T_on_dbfs, T_off_dbfs) in dBFS units."""
        t_on_lin, t_off_lin = self.get_thresholds_linear()
        on_dbfs = 10.0 * math.log10(max(t_on_lin, self._conversion_floor))
        off_dbfs = 10.0 * math.log10(max(t_off_lin, self._conversion_floor))
        return on_dbfs, off_dbfs

    @property
    def target_warmup_samples(self) -> int:
        """Target total sample count for warmup calibration."""
        return self._target_warmup_samples

    @property
    def remaining_warmup_samples(self) -> int:
        """Remaining samples needed to complete calibration."""
        if self._state != NoiseLifecycleState.CALIBRATING:
            return 0
        return self._warmup_settling_remaining + max(0, self._target_warmup_samples - self._warmup_samples_collected)

    @property
    def samples_trained(self) -> int:
        """Total number of warmup samples trained so far."""
        return self._warmup_samples_collected

    @property
    def _sample_count(self) -> int:
        """Compatibility property for sample count."""
        return self._warmup_samples_collected

    def get_estimate(self, stream_id: str = "default", block_sequence_number: int = 0) -> NoiseEstimate:
        """Return current noise estimate without ingesting new data."""
        sid = stream_id if (isinstance(stream_id, str) and stream_id.strip()) else "default"
        if not self.is_calibrated or self._noise_power_linear is None:
            target_samples = max(self._target_warmup_samples, 1)
            reason = (
                f"Warmup phase (CALIBRATING): collecting initial noise samples (collected {self._warmup_samples_collected}/"
                f"{target_samples} samples)"
                if self._state == NoiseLifecycleState.CALIBRATING
                else f"Estimator in state {self._state.value}"
            )
            return NoiseEstimate(
                stream_id=sid,
                block_sequence_number=block_sequence_number,
                noise_power_dbfs=float("nan"),
                valid=False,
                reason=reason,
            )

        noise_dbfs = 10.0 * math.log10(max(self._noise_power_linear, self._conversion_floor))
        return NoiseEstimate(
            stream_id=sid,
            block_sequence_number=block_sequence_number,
            noise_power_dbfs=noise_dbfs,
            valid=True,
            reason=None,
            reference_domain=self.reference_domain,
            training_samples=self._warmup_samples_collected,
        )

    def update_samples(self, samples_iq: np.ndarray) -> int:
        """Ingest clean noise samples during warmup training up to the exact authorized budget.

        Returns:
            The number of samples consumed from samples_iq.
        """
        if self._frozen or getattr(self, "_suspended", False) or len(samples_iq) == 0:
            return 0
        if self._state != NoiseLifecycleState.CALIBRATING:
            return 0

        if not isinstance(samples_iq, np.ndarray) or samples_iq.ndim != 1 or not np.iscomplexobj(samples_iq):
            raise ValueError("Training samples must be a complex 1D array")
        budget = self.remaining_warmup_samples
        if not np.all(np.isfinite(samples_iq[:budget])):
            raise ValueError("Training samples must be finite")
        discarded = min(len(samples_iq), self._warmup_settling_remaining)
        self._warmup_settling_remaining -= discarded
        samples_iq = samples_iq[discarded:]

        target_samples = self._target_warmup_samples
        remaining = target_samples - self._warmup_samples_collected
        if remaining <= 0:
            return 0

        n_consume = min(len(samples_iq), remaining)
        chunk = samples_iq[:n_consume]

        c128 = chunk.astype(np.complex128)
        mag_sq = np.real(c128) ** 2 + np.imag(c128) ** 2
        p_sum = float(np.sum(mag_sq))
        if p_sum >= 0.0 and math.isfinite(p_sum):
            self._warmup_power_sum += p_sum
            self._warmup_samples_collected += n_consume

            if self._warmup_samples_collected >= target_samples:
                total_p = self._warmup_power_sum
                total_s = self._warmup_samples_collected
                if total_s > 0:
                    self._noise_power_linear = max(float(total_p / total_s), self._conversion_floor)
                    self._state = NoiseLifecycleState.VALID
        return discarded + n_consume

    def suspend_training(self) -> None:
        """Suspend warmup noise ingestion until explicit authorized recalibration."""
        self._suspended = True

    def update(self, block: IQBlock) -> NoiseEstimate:
        """Update noise estimator with an incoming IQ block."""
        if not self._frozen and not getattr(self, "_suspended", False) and block.num_samples > 0:
            self.update_samples(block.iq)
        return self.get_estimate(block.stream.stream_id, block.sequence_number)

    def recalibrate(self, warmup_samples: int | None = None) -> None:
        """Transition to CALIBRATING state for authorized clean recalibration."""
        if warmup_samples is not None:
            if isinstance(warmup_samples, bool) or not isinstance(warmup_samples, (int, np.integer)) or warmup_samples < 1:
                raise ValueError(f"warmup_samples must be an integer >= 1, got {warmup_samples!r}")
            self._target_warmup_samples = int(warmup_samples)
            self._warmup_samples = int(warmup_samples)
        else:
            self._target_warmup_samples = max(getattr(self, "_target_warmup_samples", 0), self._window_size)
            self._warmup_samples = self._target_warmup_samples
        self._noise_power_linear = None
        self._warmup_power_sum = 0.0
        self._warmup_samples_collected = 0
        self._warmup_settling_remaining = self._warmup_settling_total
        self._frozen = False
        self._suspended = False
        self._state = NoiseLifecycleState.CALIBRATING

    def reset(self) -> None:
        """Reset estimator state."""
        self._noise_power_linear = self._initial_noise_power
        self._warmup_power_sum = 0.0
        self._warmup_samples_collected = 0
        self._warmup_settling_remaining = self._warmup_settling_total
        self._frozen = False
        self._suspended = False
        if self._initial_noise_power is not None:
            self._state = NoiseLifecycleState.VALID
        elif self._target_warmup_samples > 0:
            self._state = NoiseLifecycleState.CALIBRATING
        else:
            self._state = NoiseLifecycleState.UNCALIBRATED


class AdaptiveRobustNoiseEstimator(NoiseEstimatorProtocol, Resettable):
    """Adaptive robust noise estimator using median sample power scaling and guarded updates.

    Under circular complex Gaussian noise, sample squared envelope |x|^2 is Exponential(mean=sigma2).
    The median of an Exponential(1/lambda) distribution is ln(2)/lambda = sigma2 * ln(2).
    Therefore, sigma2 = median(|x|^2) / ln(2).
    This estimator tracks background noise across idle periods and strictly freezes
    adaptation during candidate, active, and trailing signal capture states.
    Uses fixed-size statistic windows with carried residuals to ensure partition-invariant updates.
    """

    def __init__(
        self,
        window_size: int = 256,
        pfa_on: float = 1e-4,
        pfa_off: float = 1e-2,
        time_constant_s: float = 0.5,
        sample_rate_hz: float = 1e6,
        conversion_floor_linear: float = 1e-15,
        initial_noise_power_linear: float | None = None,
        adaptation_window_size: int = 64,
        max_staleness_samples: int = 1_000_000,
        reference_domain: NoiseReferenceDomain | None = None,
    ) -> None:
        if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)) or window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size!r}")
        if not (0.0 < pfa_on < pfa_off < 1.0):
            raise ValueError("Probabilities must satisfy 0 < pfa_on < pfa_off < 1")
        if (
            isinstance(time_constant_s, bool)
            or not isinstance(time_constant_s, (int, float, np.floating, np.integer))
            or not math.isfinite(float(time_constant_s))
            or float(time_constant_s) <= 0.0
        ):
            raise ValueError(f"time_constant_s must be strictly positive and finite, got {time_constant_s!r}")
        if (
            isinstance(sample_rate_hz, bool)
            or not isinstance(sample_rate_hz, (int, float, np.floating, np.integer))
            or not math.isfinite(float(sample_rate_hz))
            or float(sample_rate_hz) <= 0.0
        ):
            raise ValueError(f"sample_rate_hz must be strictly positive and finite, got {sample_rate_hz!r}")
        if initial_noise_power_linear is not None:
            if (
                isinstance(initial_noise_power_linear, bool)
                or initial_noise_power_linear <= 0.0
                or not math.isfinite(initial_noise_power_linear)
            ):
                raise ValueError("initial_noise_power_linear must be strictly positive and finite")

        if (
            isinstance(conversion_floor_linear, bool)
            or not isinstance(conversion_floor_linear, (int, float, np.floating, np.integer))
            or not math.isfinite(float(conversion_floor_linear))
            or float(conversion_floor_linear) <= 0.0
        ):
            raise ValueError(f"conversion_floor_linear must be strictly positive and finite, got {conversion_floor_linear!r}")

        if (
            isinstance(adaptation_window_size, bool)
            or not isinstance(adaptation_window_size, (int, np.integer))
            or adaptation_window_size < 1
        ):
            raise ValueError(f"adaptation_window_size must be an integer >= 1, got {adaptation_window_size!r}")
        if (
            isinstance(max_staleness_samples, bool)
            or not isinstance(max_staleness_samples, (int, np.integer))
            or max_staleness_samples < 1
        ):
            raise ValueError(f"max_staleness_samples must be an integer >= 1, got {max_staleness_samples!r}")

        self._window_size = int(window_size)
        self._pfa_on = float(pfa_on)
        self._pfa_off = float(pfa_off)
        self._time_constant_s = float(time_constant_s)
        self._sample_rate_hz = float(sample_rate_hz)
        self._conversion_floor = float(conversion_floor_linear)
        self._adaptation_window_size = int(adaptation_window_size)
        self._max_staleness_samples = int(max_staleness_samples)
        self.reference_domain = reference_domain

        self._initial_noise_power = (
            float(initial_noise_power_linear) if initial_noise_power_linear is not None else None
        )
        self._noise_power_linear: float | None = self._initial_noise_power
        self._frozen: bool = False
        self._history_powers: list[float] = []

        # Streaming adaptation state
        self._adaptation_residual: np.ndarray = np.empty(0, dtype=np.complex64)
        self._samples_since_last_accepted: int = 0
        self._total_accepted_samples: int = 0
        self._pending_stream_samples: int = 0

        self._unit_q_on = float(gamma.isf(self._pfa_on, a=self._window_size, scale=1.0 / self._window_size))
        self._unit_q_off = float(gamma.isf(self._pfa_off, a=self._window_size, scale=1.0 / self._window_size))

        self._state = (
            NoiseLifecycleState.VALID if self._initial_noise_power is not None else NoiseLifecycleState.UNCALIBRATED
        )

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def state(self) -> NoiseLifecycleState:
        return self._state

    @property
    def noise_power_linear(self) -> float | None:
        return self._noise_power_linear

    @property
    def is_calibrated(self) -> bool:
        return (
            self._state == NoiseLifecycleState.VALID
            and self._noise_power_linear is not None
            and math.isfinite(self._noise_power_linear)
            and self._noise_power_linear > 0.0
        )

    @property
    def staleness_samples(self) -> int:
        """Samples elapsed since the last accepted noise update."""
        return self._samples_since_last_accepted + len(self._adaptation_residual) + self._pending_stream_samples

    @property
    def _accepted_noise_samples(self) -> int:
        """Total accepted noise samples across stream."""
        return self._total_accepted_samples

    @property
    def total_accepted_samples(self) -> int:
        """Total accepted noise samples across stream."""
        return self._total_accepted_samples

    @property
    def _residual_samples(self) -> np.ndarray:
        """Residual unchunked samples held in buffer."""
        return self._adaptation_residual

    def freeze(self, frozen: bool) -> None:
        """Freeze adaptation when a candidate, active, or trailing burst is present."""
        if frozen and not self._frozen:
            # A statistic must never straddle an excluded active interval.
            self._samples_since_last_accepted += len(self._adaptation_residual)
            self._adaptation_residual = np.empty(0, dtype=np.complex64)
        self._frozen = bool(frozen)

    def set_pending_samples(self, num_samples: int) -> None:
        """Account for received IQ awaiting a causal admission decision, without training."""
        if isinstance(num_samples, bool) or not isinstance(num_samples, (int, np.integer)) or num_samples < 0:
            raise ValueError("num_samples must be a non-negative integer")
        self._pending_stream_samples = int(num_samples)
        if self.staleness_samples > self._max_staleness_samples and self._state == NoiseLifecycleState.VALID:
            self._state = NoiseLifecycleState.STALE

    def advance_stream_samples(self, num_samples: int) -> None:
        """Advance logical stream sample count to track staleness even while frozen."""
        if isinstance(num_samples, bool) or not isinstance(num_samples, (int, np.integer)) or num_samples < 0:
            raise ValueError(f"num_samples must be an integer >= 0, got {num_samples!r}")
        self._samples_since_last_accepted += int(num_samples)
        if self.staleness_samples > self._max_staleness_samples and (
            self._state == NoiseLifecycleState.VALID
            or (self._state == NoiseLifecycleState.UNCALIBRATED and self._noise_power_linear is not None)
        ):
            self._state = NoiseLifecycleState.STALE

    def get_thresholds_linear(self) -> tuple[float, float]:
        if not self.is_calibrated or self._noise_power_linear is None:
            raise RuntimeError("Cannot compute thresholds: noise estimator is uncalibrated")
        t_on = float(self._noise_power_linear * self._unit_q_on)
        t_off = float(self._noise_power_linear * self._unit_q_off)
        return t_on, t_off

    def get_thresholds_dbfs(self) -> tuple[float, float]:
        t_on_lin, t_off_lin = self.get_thresholds_linear()
        on_dbfs = 10.0 * math.log10(max(t_on_lin, self._conversion_floor))
        off_dbfs = 10.0 * math.log10(max(t_off_lin, self._conversion_floor))
        return on_dbfs, off_dbfs

    def get_estimate(self, stream_id: str = "default", block_sequence_number: int = 0) -> NoiseEstimate:
        """Return current noise estimate without ingesting new data."""
        sid = stream_id if (isinstance(stream_id, str) and stream_id.strip()) else "default"
        if not self.is_calibrated or self._noise_power_linear is None:
            reason = (
                f"Noise reference is STALE: no accepted updates for {self._samples_since_last_accepted} samples"
                if self._state == NoiseLifecycleState.STALE
                else "Uncalibrated noise estimator"
            )
            return NoiseEstimate(
                stream_id=sid,
                block_sequence_number=block_sequence_number,
                noise_power_dbfs=float("nan"),
                valid=False,
                reason=reason,
            )

        noise_dbfs = 10.0 * math.log10(max(self._noise_power_linear, self._conversion_floor))
        return NoiseEstimate(
            stream_id=sid,
            block_sequence_number=block_sequence_number,
            noise_power_dbfs=noise_dbfs,
            valid=True,
            reason=None,
            reference_domain=self.reference_domain,
        )

    def update_samples(self, samples_iq: np.ndarray) -> None:
        """Update adaptation with confirmed idle noise samples without burst contamination.

        Chunks samples into fixed-size windows with carried residuals to ensure partition invariance.
        """
        if self._frozen or len(samples_iq) == 0:
            return

        if len(self._adaptation_residual) > 0:
            combined = np.concatenate([self._adaptation_residual, samples_iq])
        else:
            combined = samples_iq

        m = self._adaptation_window_size
        n_chunks = len(combined) // m
        if n_chunks == 0:
            self._adaptation_residual = combined.copy()
            if self.staleness_samples > self._max_staleness_samples and (
                self._state == NoiseLifecycleState.VALID
                or (self._state == NoiseLifecycleState.UNCALIBRATED and self._noise_power_linear is not None)
            ):
                self._state = NoiseLifecycleState.STALE
            return

        dt = m / self._sample_rate_hz
        alpha = 1.0 - math.exp(-dt / self._time_constant_s)

        for i in range(n_chunks):
            chunk = combined[i * m : (i + 1) * m]
            c128 = chunk.astype(np.complex128)
            mag_sq = np.real(c128) ** 2 + np.imag(c128) ** 2
            med_p = float(np.median(mag_sq))
            mean_p = float(np.mean(mag_sq))
            sigma2_est = min(med_p / math.log(2.0), mean_p)
            if sigma2_est <= 0.0 or not math.isfinite(sigma2_est):
                self._samples_since_last_accepted += m
                continue

            if self._noise_power_linear is None:
                self._noise_power_linear = sigma2_est
                self._state = NoiseLifecycleState.VALID
            else:
                self._noise_power_linear = (1.0 - alpha) * self._noise_power_linear + alpha * sigma2_est
                if self._state == NoiseLifecycleState.STALE:
                    self._state = NoiseLifecycleState.VALID

            self._total_accepted_samples += m
            self._samples_since_last_accepted = 0

            if len(self._history_powers) >= 100:
                self._history_powers.pop(0)
            self._history_powers.append(self._noise_power_linear)

        rem_start = n_chunks * m
        self._adaptation_residual = combined[rem_start:].copy()
        if self.staleness_samples > self._max_staleness_samples and (
            self._state == NoiseLifecycleState.VALID
            or (self._state == NoiseLifecycleState.UNCALIBRATED and self._noise_power_linear is not None)
        ):
            self._state = NoiseLifecycleState.STALE

    def update(self, block: IQBlock) -> NoiseEstimate:
        if not self._frozen and block.num_samples > 0:
            self.update_samples(block.iq)
        return self.get_estimate(block.stream.stream_id, block.sequence_number)

    def recalibrate(self, reference_linear: float | None = None) -> None:
        """Recalibrate a stale or uncalibrated estimator using authorized clean noise reference."""
        if reference_linear is not None:
            if (
                isinstance(reference_linear, bool)
                or not isinstance(reference_linear, (int, float, np.floating, np.integer))
                or not math.isfinite(float(reference_linear))
                or float(reference_linear) <= 0.0
            ):
                raise ValueError(f"reference_linear must be strictly positive and finite, got {reference_linear!r}")
            self._noise_power_linear = float(reference_linear)
            self._state = NoiseLifecycleState.VALID
        elif self._initial_noise_power is not None:
            self._noise_power_linear = self._initial_noise_power
            self._state = NoiseLifecycleState.VALID
        else:
            self._noise_power_linear = None
            self._state = NoiseLifecycleState.UNCALIBRATED
        self._adaptation_residual = np.empty(0, dtype=np.complex64)
        self._samples_since_last_accepted = 0

        self._pending_stream_samples = 0

    def reset(self) -> None:
        self._noise_power_linear = self._initial_noise_power
        self._frozen = False
        self._history_powers.clear()
        self._adaptation_residual = np.empty(0, dtype=np.complex64)
        self._samples_since_last_accepted = 0
        self._total_accepted_samples = 0
        self._pending_stream_samples = 0
        self._state = (
            NoiseLifecycleState.VALID if self._initial_noise_power is not None else NoiseLifecycleState.UNCALIBRATED
        )


class EmpiricalNoiseCalibrator:
    """Empirical noise calibrator for receiver noise and filtered detection statistics.

    Collects clean training noise passed through the configured detection path (discarding settling),
    computes empirical mean power and tail quantile thresholds with documented confidence intervals,
    and validates sufficient sample count for requested Pfa.
    """

    def __init__(
        self,
        window_size: int = 256,
        hop_size: int | None = None,
        pfa_on: float = 1e-4,
        pfa_off: float = 1e-2,
        max_samples: int = 5_000_000,
    ) -> None:
        if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)) or window_size < 1:
            raise ValueError(f"window_size must be an integer >= 1, got {window_size!r}")
        if hop_size is not None:
            if isinstance(hop_size, bool) or not isinstance(hop_size, (int, np.integer)) or not (1 <= hop_size <= window_size):
                raise ValueError(f"hop_size must satisfy 1 <= hop_size <= window_size, got {hop_size!r}")
        if not (0.0 < pfa_on < pfa_off < 1.0):
            raise ValueError(f"Probabilities must satisfy 0 < pfa_on < pfa_off < 1, got pfa_on={pfa_on}, pfa_off={pfa_off}")
        if isinstance(max_samples, bool) or not isinstance(max_samples, (int, np.integer)) or max_samples < 1:
            raise ValueError(f"max_samples must be an integer >= 1, got {max_samples!r}")

        self.window_size = int(window_size)
        self.hop_size = int(hop_size) if hop_size is not None else self.window_size
        self.pfa_on = float(pfa_on)
        self.pfa_off = float(pfa_off)
        self.max_samples = int(max_samples)

    def begin_calibration(
        self,
        sample_rate_hz: float = 1e6,
        settling_samples: int = 0,
        monitored_band_hz: tuple[float, float] | None = None,
        filter_order: int = 4,
        gain_id: str = "digital_iq_unity",
    ) -> None:
        """Initialize streaming bounded calibration state."""
        if isinstance(sample_rate_hz, bool) or not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be strictly positive and finite")
        if isinstance(settling_samples, bool) or not isinstance(settling_samples, (int, np.integer)) or settling_samples < 0:
            raise ValueError("settling_samples must be an integer >= 0")
        if isinstance(filter_order, bool) or not isinstance(filter_order, (int, np.integer)) or filter_order < 1:
            raise ValueError("filter_order must be an integer >= 1")
        if monitored_band_hz is not None:
            f_min, f_max = float(monitored_band_hz[0]), float(monitored_band_hz[1])
            nyquist = float(sample_rate_hz) / 2.0
            if not (-nyquist <= f_min < f_max <= nyquist):
                raise ValueError(f"monitored_band_hz ({f_min}, {f_max}) outside valid Nyquist interval [-{nyquist}, {nyquist}]")
            if f_max - f_min >= sample_rate_hz:
                raise ValueError("Configured filter bandwidth must be less than sample rate")
        if not isinstance(gain_id, str) or not gain_id.strip():
            raise ValueError("gain_id must be a nonblank string")

        self._cal_sample_rate_hz = float(sample_rate_hz)
        self._cal_settling_remaining = int(settling_samples)
        self._cal_settling_total = int(settling_samples)
        self._cal_monitored_band_hz = tuple(monitored_band_hz) if monitored_band_hz is not None else None
        self._cal_filter_order = int(filter_order)
        self._finalized = False

        self._cal_sos = None
        self._cal_sos_zi = None
        self._cal_sos_hash = None
        self._cal_f_center = 0.0
        self._cal_phase_acc = 0.0

        if monitored_band_hz is not None:
            f_min, f_max = float(monitored_band_hz[0]), float(monitored_band_hz[1])
            self._cal_f_center = (f_min + f_max) / 2.0
            bw = f_max - f_min
            cutoff_hz = bw / 2.0
            nyquist = self._cal_sample_rate_hz / 2.0
            wn = cutoff_hz / nyquist
            self._cal_sos = signal.butter(N=self._cal_filter_order, Wn=wn, btype="lowpass", output="sos")
            self._cal_sos_zi = np.zeros((self._cal_sos.shape[0], 2), dtype=np.float64)
            self._cal_sos_hash = hashlib.sha256(self._cal_sos.tobytes()).hexdigest()[:12]

        self._cal_samples_collected = 0
        self._cal_sample_sum_sq = 0.0
        self._cal_residual_iq = np.empty(0, dtype=np.complex64)
        self._cal_window_powers: list[float] = []
        self._cal_reference_domain = NoiseReferenceDomain(
            self._cal_sample_rate_hz, self._cal_monitored_band_hz, self._cal_sos_hash, gain_id
        )

    def update_calibration(self, chunk: np.ndarray) -> None:
        """Ingest a chunk of training noise samples in a bounded streaming manner."""
        if not hasattr(self, "_cal_samples_collected"):
            raise RuntimeError("Call begin_calibration before update_calibration")
        if getattr(self, "_finalized", False):
            raise RuntimeError("Calibration has been finalized")
        if not isinstance(chunk, np.ndarray) or chunk.ndim != 1:
            raise ValueError(f"Calibration input must be a 1D numpy array, got {getattr(chunk, 'shape', type(chunk))}")
        if not np.iscomplexobj(chunk):
            raise ValueError(f"Calibration input must be complex dtype, got {chunk.dtype}")

        if len(chunk) == 0:
            return

        if self._cal_samples_collected >= self.max_samples:
            return

        # Pre-cap incoming samples by remaining settling + training budget BEFORE filtering/validation
        remaining_needed = self._cal_settling_remaining + (self.max_samples - self._cal_samples_collected)
        if remaining_needed <= 0:
            return
        if len(chunk) > remaining_needed:
            chunk = chunk[:remaining_needed]

        if not np.all(np.isfinite(chunk)):
            raise ValueError("Calibration input contains NaN or Inf values")

        if self._cal_sos is not None:
            n = len(chunk)
            t = np.arange(n, dtype=np.float64) / self._cal_sample_rate_hz
            phase_rot = self._cal_phase_acc - 2.0 * np.pi * self._cal_f_center * t
            mixed = chunk * np.exp(1j * phase_rot).astype(np.complex64)
            self._cal_phase_acc = float((phase_rot[-1] - 2.0 * np.pi * self._cal_f_center / self._cal_sample_rate_hz) % (2.0 * np.pi))
            filtered, self._cal_sos_zi = signal.sosfilt(self._cal_sos, mixed, zi=self._cal_sos_zi)
            valid_chunk = filtered.astype(np.complex64)
        else:
            valid_chunk = chunk

        # Discard settling samples
        if self._cal_settling_remaining > 0:
            if len(valid_chunk) <= self._cal_settling_remaining:
                self._cal_settling_remaining -= len(valid_chunk)
                return
            else:
                valid_chunk = valid_chunk[self._cal_settling_remaining:]
                self._cal_settling_remaining = 0

        # Cap by remaining budget to max_samples
        remaining_budget = self.max_samples - self._cal_samples_collected
        if len(valid_chunk) > remaining_budget:
            valid_chunk = valid_chunk[:remaining_budget]

        if len(valid_chunk) == 0:
            return

        c128 = valid_chunk.astype(np.complex128)
        self._cal_sample_sum_sq += float(np.sum(np.real(c128) ** 2 + np.imag(c128) ** 2))
        self._cal_samples_collected += len(valid_chunk)

        # Sliding/hopping window powers with residual concatenation
        if len(self._cal_residual_iq) > 0:
            combined = np.concatenate([self._cal_residual_iq, valid_chunk])
        else:
            combined = valid_chunk

        w_size = self.window_size
        h_size = self.hop_size
        total_len = len(combined)

        if total_len >= w_size:
            n_win = (total_len - w_size) // h_size + 1
            comb_c128 = combined.astype(np.complex128)
            mag_sq = np.real(comb_c128) ** 2 + np.imag(comb_c128) ** 2
            prefix = np.empty(total_len + 1, dtype=np.float64)
            prefix[0] = 0.0
            np.cumsum(mag_sq, out=prefix[1:])

            for w_idx in range(n_win):
                w_start = w_idx * h_size
                w_end = w_start + w_size
                p_win = (prefix[w_end] - prefix[w_start]) / w_size
                self._cal_window_powers.append(float(p_win))

            rem_start = n_win * h_size
            self._cal_residual_iq = combined[rem_start:].copy()
        else:
            self._cal_residual_iq = combined.copy()

    def finalize_calibration(self) -> dict[str, Any]:
        """Finalize calibration and return result dictionary with full parameter identity."""
        if self._cal_samples_collected < self.window_size:
            raise ValueError(
                f"Insufficient valid calibration samples ({self._cal_samples_collected}) for window size {self.window_size}"
            )

        mean_power = self._cal_sample_sum_sq / self._cal_samples_collected
        if mean_power <= 0.0 or not math.isfinite(mean_power):
            raise ValueError(f"Calibration noise power {mean_power} must be strictly positive and finite")

        w_size = self.window_size
        n_windows = len(self._cal_window_powers)
        win_powers = np.asarray(self._cal_window_powers, dtype=np.float64)

        required_windows_on = int(math.ceil(1.0 / self.pfa_on))
        required_windows_off = int(math.ceil(1.0 / self.pfa_off))
        quantile_resolved = n_windows > required_windows_on
        # Training-window count is resolution, never evidence of independence or
        # a confidence statement about future false alarms.
        supported_pfa = False

        if quantile_resolved and n_windows > 0:
            emp_t_on = float(np.quantile(win_powers, 1.0 - self.pfa_on))
        else:
            emp_t_on = None

        if n_windows >= required_windows_off and n_windows > 0:
            emp_t_off = float(np.quantile(win_powers, 1.0 - self.pfa_off))
        else:
            emp_t_off = None

        scale = mean_power / w_size
        gamma_t_on = float(gamma.isf(self.pfa_on, a=w_size, scale=scale))
        gamma_t_off = float(gamma.isf(self.pfa_off, a=w_size, scale=scale))

        self._finalized = True

        return {
            "mean_power_linear": mean_power,
            "mean_power_dbfs": 10.0 * math.log10(mean_power),
            "num_samples": self._cal_samples_collected,
            "num_windows": n_windows,
            "gamma_t_on_linear": gamma_t_on,
            "gamma_t_off_linear": gamma_t_off,
            "empirical_t_on_linear": emp_t_on,
            "empirical_t_off_linear": emp_t_off,
            "t_on_linear": emp_t_on if emp_t_on is not None else gamma_t_on,
            "t_off_linear": emp_t_off if emp_t_off is not None else gamma_t_off,
            "window_size": self.window_size,
            "hop_size": self.hop_size,
            "pfa_on": self.pfa_on,
            "pfa_off": self.pfa_off,
            "sample_rate_hz": self._cal_sample_rate_hz,
            "supported_pfa": supported_pfa,
            "quantile_resolved": quantile_resolved,
            "reference_domain": self._cal_reference_domain,
            "support_method": "uncertified_training_quantile",
            "dependence_assumption": "unspecified",
            "monitored_band_hz": self._cal_monitored_band_hz,
            "filter_order": self._cal_filter_order,
            "filter_sos_hash": self._cal_sos_hash,
            "settling_samples": self._cal_settling_total,
            "detector_mode": "configured_band" if self._cal_monitored_band_hz is not None else "full_band",
        }

    def calibrate(
        self,
        noise_iq: np.ndarray,
        sample_rate_hz: float = 1e6,
        settling_samples: int = 0,
        monitored_band_hz: tuple[float, float] | None = None,
        filter_order: int = 4,
    ) -> dict[str, Any]:
        """Calibrate on clean training noise."""
        if len(noise_iq) <= settling_samples:
            raise ValueError("Training noise length must exceed settling samples")

        self.begin_calibration(
            sample_rate_hz=sample_rate_hz,
            settling_samples=settling_samples,
            monitored_band_hz=monitored_band_hz,
            filter_order=filter_order,
        )

        # Apply max_samples bound before any large allocations
        max_needed = settling_samples + self.max_samples
        capped_iq = noise_iq[:max_needed]
        chunk_sz = 65536
        for i in range(0, len(capped_iq), chunk_sz):
            self.update_calibration(capped_iq[i : i + chunk_sz])

        return self.finalize_calibration()

    def create_policy(
        self,
        noise_or_cal: np.ndarray | dict[str, Any],
        sample_rate_hz: float = 1e6,
        settling_samples: int = 4096,
        monitored_band_hz: tuple[float, float] | None = None,
        filter_order: int = 4,
    ) -> ThresholdPolicy:
        """Create a validated typed ThresholdPolicy from training noise or calibration dict."""
        if isinstance(noise_or_cal, dict):
            cal = noise_or_cal
            required_keys = ("window_size", "hop_size", "pfa_on", "pfa_off", "sample_rate_hz")
            for k in required_keys:
                if k not in cal:
                    raise PolicyCompatibilityError(f"Calibration dictionary lacks required parameter identity: {k}")
            if cal["window_size"] != self.window_size:
                raise PolicyCompatibilityError(
                    f"Window size mismatch: calibration dict window_size ({cal['window_size']}) does not match calibrator ({self.window_size})"
                )
            if cal["hop_size"] != self.hop_size:
                raise PolicyCompatibilityError(
                    f"Hop size mismatch: calibration dict hop_size ({cal['hop_size']}) does not match calibrator ({self.hop_size})"
                )
            if not math.isclose(cal["pfa_on"], self.pfa_on, rel_tol=1e-7):
                raise PolicyCompatibilityError(
                    f"Pfa_on mismatch: calibration dict pfa_on ({cal['pfa_on']}) does not match calibrator ({self.pfa_on})"
                )
            if not math.isclose(cal["pfa_off"], self.pfa_off, rel_tol=1e-7):
                raise PolicyCompatibilityError(
                    f"Pfa_off mismatch: calibration dict pfa_off ({cal['pfa_off']}) does not match calibrator ({self.pfa_off})"
                )
            if not math.isclose(cal["sample_rate_hz"], sample_rate_hz, rel_tol=1e-7):
                raise PolicyCompatibilityError(
                    f"Sample rate mismatch: calibration dict sample_rate_hz ({cal['sample_rate_hz']}) does not match requested ({sample_rate_hz})"
                )
        else:
            cal = self.calibrate(
                noise_iq=noise_or_cal,
                sample_rate_hz=sample_rate_hz,
                settling_samples=settling_samples,
                monitored_band_hz=monitored_band_hz,
                filter_order=filter_order,
            )

        mean_lin = cal["mean_power_linear"]
        supported = cal["supported_pfa"]
        emp_on = cal["empirical_t_on_linear"]
        emp_off = cal["empirical_t_off_linear"]

        if emp_on is not None and emp_off is not None:
            t_on_lin = emp_on
            t_off_lin = emp_off
            mode: Literal["analytic_gamma", "empirical_matched"] = "empirical_matched"
            valid = True
            reason = None
        else:
            t_on_lin = cal["gamma_t_on_linear"]
            t_off_lin = cal["gamma_t_off_linear"]
            mode = "analytic_gamma"
            valid = False
            reason = f"Insufficient sample support ({cal['num_windows']} windows < {int(1.0/self.pfa_on)} required)"

        t_on_dbfs = 10.0 * math.log10(t_on_lin)
        t_off_dbfs = 10.0 * math.log10(t_off_lin)

        # Enforce strict invariant: on > off
        if t_on_lin <= t_off_lin:
            t_on_lin = t_off_lin * 1.01
            t_on_dbfs = 10.0 * math.log10(t_on_lin)

        return ThresholdPolicy(
            mode=mode,
            window_size=self.window_size,
            hop_size=self.hop_size,
            pfa_on=self.pfa_on,
            pfa_off=self.pfa_off,
            mean_power_linear=mean_lin,
            t_on_linear=t_on_lin,
            t_off_linear=t_off_lin,
            t_on_dbfs=t_on_dbfs,
            t_off_dbfs=t_off_dbfs,
            supported_pfa=supported,
            sample_rate_hz=float(cal.get("sample_rate_hz", sample_rate_hz)),
            monitored_band_hz=tuple(cal["monitored_band_hz"]) if cal.get("monitored_band_hz") is not None else (tuple(monitored_band_hz) if monitored_band_hz is not None else None),
            filter_sos_hash=cal["filter_sos_hash"],
            training_samples=cal["num_samples"],
            training_windows=cal["num_windows"],
            valid=valid,
            reason=reason,
            reference_domain=cal.get("reference_domain"),
            support_method="uncertified_training_quantile",
            dependence_assumption="unspecified",
            settling_samples=int(cal.get("settling_samples", 0)),
        )
