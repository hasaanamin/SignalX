"""Phase domain feature extractor for RF signals."""

import math
from typing import Any

import numpy as np

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, FeatureValue
from signal_processing.features.phase.exceptions import PhaseFeatureError


class PhaseFeatureExtractor:
    """Stateless feature extractor for phase-domain characteristics.

    Extracts the following six canonical scalar features from wrapped and unwrapped phase:
        - phase.circular_resultant_length: Mean resultant length of wrapped phase [None]
        - phase.circular_variance: Circular variance (1 - R) of wrapped phase [None]
        - phase.increment_mean_rad: Mean sample-to-sample unwrapped phase increment [rad]
        - phase.increment_std_rad: Population standard deviation of unwrapped phase increment [rad]
        - phase.increment_rms_rad: Root mean square of unwrapped phase increment [rad]
        - phase.increment_abs_mean_rad: Mean absolute unwrapped phase increment [rad]
    """

    name: str = "phase"
    version: str = "1.0"

    def extract(self, context: SignalContext | Any) -> FeatureSet:
        """Extract phase features from a SignalContext.

        Args:
            context: SignalContext instance (or duck-typed object exposing `mean_power_linear()`
                and `phase()`).

        Returns:
            FeatureSet containing the 6 canonical phase FeatureValues.

        Raises:
            PhaseFeatureError: If context is None, missing required methods, or returns
                malformed PhaseResult structures.
        """
        if context is None:
            raise PhaseFeatureError("Context must not be None")

        if not hasattr(context, "mean_power_linear") or not callable(context.mean_power_linear):
            raise PhaseFeatureError(
                f"Expected SignalContext instance or object with mean_power_linear() method, got {type(context).__name__}"
            )

        if not hasattr(context, "phase") or not callable(context.phase):
            raise PhaseFeatureError(
                f"Expected SignalContext instance or object with phase() method, got {type(context).__name__}"
            )

        # Precheck signal power before evaluating phase
        try:
            p_mean = context.mean_power_linear()
        except Exception as exc:
            raise PhaseFeatureError(f"Failed to obtain mean_power_linear from context: {exc}") from exc

        if (
            not isinstance(p_mean, (int, float, np.floating, np.integer))
            or not math.isfinite(p_mean)
            or p_mean <= 0.0
        ):
            return self._make_zero_power_featureset("Signal has zero power; phase is undefined")

        # Obtain canonical cached PhaseResult
        try:
            phase_res = context.phase()
        except Exception as exc:
            raise PhaseFeatureError(f"Failed to obtain phase from context: {exc}") from exc

        if not hasattr(phase_res, "wrapped_rad") or not hasattr(phase_res, "unwrapped_rad"):
            raise PhaseFeatureError(
                f"PhaseResult must contain wrapped_rad and unwrapped_rad arrays, got {type(phase_res).__name__}"
            )

        try:
            wrapped = np.asarray(phase_res.wrapped_rad)
            unwrapped = np.asarray(phase_res.unwrapped_rad)
        except Exception as exc:
            raise PhaseFeatureError(f"Cannot convert phase data to numpy arrays: {exc}") from exc

        if wrapped.ndim != 1 or unwrapped.ndim != 1 or wrapped.size == 0 or unwrapped.size == 0:
            raise PhaseFeatureError("Phase arrays must be non-empty 1-dimensional arrays")

        if wrapped.size != unwrapped.size:
            raise PhaseFeatureError(
                f"Phase array length mismatch: wrapped_rad has {wrapped.size}, unwrapped_rad has {unwrapped.size}"
            )

        if not (np.issubdtype(wrapped.dtype, np.floating) or np.issubdtype(wrapped.dtype, np.integer)) or np.iscomplexobj(wrapped):
            raise PhaseFeatureError(f"wrapped_rad must be real-valued numeric, got dtype {wrapped.dtype}")

        if not (np.issubdtype(unwrapped.dtype, np.floating) or np.issubdtype(unwrapped.dtype, np.integer)) or np.iscomplexobj(unwrapped):
            raise PhaseFeatureError(f"unwrapped_rad must be real-valued numeric, got dtype {unwrapped.dtype}")

        if not np.all(np.isfinite(wrapped)) or not np.all(np.isfinite(unwrapped)):
            raise PhaseFeatureError("Phase arrays contain non-finite values (NaN or Inf)")


        # Check valid_mask from PhaseResult if available
        if hasattr(phase_res, "valid_mask") and phase_res.valid_mask is not None:
            valid_mask = np.asarray(phase_res.valid_mask, dtype=bool)
        else:
            valid_mask = np.ones(len(wrapped), dtype=bool)

        valid_wrapped = wrapped[valid_mask]
        num_valid_samples = len(valid_wrapped)

        # 1-2. Circular statistics on valid wrapped phase
        if num_valid_samples < 1:
            circ_reason = "No valid phase samples available (amplitude below threshold)"
            fv_r = FeatureValue(value=float("nan"), unit=None, valid=False, reason=circ_reason)
            fv_circ_var = FeatureValue(value=float("nan"), unit=None, valid=False, reason=circ_reason)
        else:
            phasors = np.exp(1j * valid_wrapped.astype(np.float64))
            z_bar = complex(np.mean(phasors))
            r_mag = abs(z_bar)
            r_val = float(np.clip(r_mag, 0.0, 1.0))
            circ_var = float(np.clip(1.0 - r_val, 0.0, 1.0))
            fv_r = FeatureValue(value=r_val, unit=None, valid=True, reason=None)
            fv_circ_var = FeatureValue(value=circ_var, unit=None, valid=True, reason=None)

        # 3-6. Phase increments on valid adjacent pairs
        # An increment between sample n and n+1 is valid ONLY if BOTH n and n+1 are valid
        valid_pairs = valid_mask[:-1] & valid_mask[1:]
        num_valid_pairs = int(np.sum(valid_pairs))

        if num_valid_pairs < 1:
            inc_reason = "At least two phase samples and valid adjacent pairs are required"
            fv_inc_mean = FeatureValue(value=float("nan"), unit="rad", valid=False, reason=inc_reason)
            fv_inc_std = FeatureValue(value=float("nan"), unit="rad", valid=False, reason=inc_reason)
            fv_inc_rms = FeatureValue(value=float("nan"), unit="rad", valid=False, reason=inc_reason)
            fv_inc_abs_mean = FeatureValue(value=float("nan"), unit="rad", valid=False, reason=inc_reason)
        else:
            delta_phi = unwrapped[1:][valid_pairs].astype(np.float64) - unwrapped[:-1][valid_pairs].astype(np.float64)
            # Wrap to [-pi, pi] for adjacent sample increments
            delta_phi = (delta_phi + np.pi) % (2.0 * np.pi) - np.pi
            inc_mean = float(np.mean(delta_phi))
            inc_std = float(np.std(delta_phi, ddof=0))
            inc_rms = float(np.sqrt(np.mean(delta_phi ** 2)))
            inc_abs_mean = float(np.mean(np.abs(delta_phi)))

            fv_inc_mean = FeatureValue(value=inc_mean, unit="rad", valid=True, reason=None)
            fv_inc_std = FeatureValue(value=inc_std, unit="rad", valid=True, reason=None)
            fv_inc_rms = FeatureValue(value=inc_rms, unit="rad", valid=True, reason=None)
            fv_inc_abs_mean = FeatureValue(value=inc_abs_mean, unit="rad", valid=True, reason=None)

        features: dict[str, FeatureValue] = {
            "phase.circular_resultant_length": fv_r,
            "phase.circular_variance": fv_circ_var,
            "phase.increment_mean_rad": fv_inc_mean,
            "phase.increment_std_rad": fv_inc_std,
            "phase.increment_rms_rad": fv_inc_rms,
            "phase.increment_abs_mean_rad": fv_inc_abs_mean,
        }

        return FeatureSet(
            extractor=self.name,
            extractor_version=self.version,
            features=features,
        )

    def _make_zero_power_featureset(self, reason: str) -> FeatureSet:
        """Helper to create a FeatureSet where all 6 phase features are marked invalid."""
        features = {
            "phase.circular_resultant_length": FeatureValue(
                value=float("nan"),
                unit=None,
                valid=False,
                reason=reason,
            ),
            "phase.circular_variance": FeatureValue(
                value=float("nan"),
                unit=None,
                valid=False,
                reason=reason,
            ),
            "phase.increment_mean_rad": FeatureValue(
                value=float("nan"),
                unit="rad",
                valid=False,
                reason=reason,
            ),
            "phase.increment_std_rad": FeatureValue(
                value=float("nan"),
                unit="rad",
                valid=False,
                reason=reason,
            ),
            "phase.increment_rms_rad": FeatureValue(
                value=float("nan"),
                unit="rad",
                valid=False,
                reason=reason,
            ),
            "phase.increment_abs_mean_rad": FeatureValue(
                value=float("nan"),
                unit="rad",
                valid=False,
                reason=reason,
            ),
        }
        return FeatureSet(
            extractor=self.name,
            extractor_version=self.version,
            features=features,
        )
