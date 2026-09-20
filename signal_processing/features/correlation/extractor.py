"""Autocorrelation domain feature extractor for RF signals."""

import math
from typing import Any

import numpy as np

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, FeatureValue
from signal_processing.features.correlation.exceptions import CorrelationFeatureError


class CorrelationFeatureExtractor:
    """Stateless feature extractor for signal autocorrelation characteristics.

    Extracts the following canonical scalar features from the positive-lag normalized autocorrelation:
        - correlation.nonzero_peak_magnitude: Peak normalized autocorrelation magnitude at lag > 0 [None]
        - correlation.nonzero_peak_lag_samples: Discrete lag in samples at the nonzero peak [samples]
        - correlation.nonzero_peak_lag_s: Continuous lag in seconds at the nonzero peak [s]
        - correlation.nonzero_peak_phase_rad: Complex phase angle of autocorrelation at nonzero peak [rad]
        - correlation.mean_nonzero_magnitude: Mean normalized autocorrelation magnitude across lags > 0 [None]
        - correlation.rms_nonzero_magnitude: RMS normalized autocorrelation magnitude across lags > 0 [None]
    """

    name: str = "correlation"
    version: str = "1.0"

    def extract(self, context: SignalContext | Any) -> FeatureSet:
        """Extract correlation features from a SignalContext.

        Args:
            context: SignalContext instance (or duck-typed object exposing `mean_power_linear()`,
                `autocorrelation()`, and `sample_rate_hz`).

        Returns:
            FeatureSet containing the 6 canonical correlation FeatureValues.

        Raises:
            CorrelationFeatureError: If context is None, missing required methods, or returns
                malformed autocorrelation results.
        """
        if context is None:
            raise CorrelationFeatureError("Context must not be None")

        if not hasattr(context, "mean_power_linear") or not callable(context.mean_power_linear):
            raise CorrelationFeatureError(
                f"Expected SignalContext instance or object with mean_power_linear() method, got {type(context).__name__}"
            )

        if not hasattr(context, "autocorrelation") or not callable(context.autocorrelation):
            raise CorrelationFeatureError(
                f"Expected SignalContext instance or object with autocorrelation() method, got {type(context).__name__}"
            )

        if not hasattr(context, "sample_rate_hz"):
            raise CorrelationFeatureError(
                f"Expected SignalContext instance or object with sample_rate_hz attribute, got {type(context).__name__}"
            )

        fs_raw = context.sample_rate_hz
        if (
            not isinstance(fs_raw, (int, float, np.floating, np.integer))
            or not math.isfinite(fs_raw)
            or fs_raw <= 0.0
        ):
            raise CorrelationFeatureError(f"sample_rate_hz must be a positive finite number, got {fs_raw}")
        fs = float(fs_raw)

        # Precheck signal power before requesting autocorrelation
        try:
            p_mean = context.mean_power_linear()
        except Exception as exc:
            raise CorrelationFeatureError(f"Failed to obtain mean_power_linear from context: {exc}") from exc

        if (
            not isinstance(p_mean, (int, float, np.floating, np.integer))
            or not math.isfinite(p_mean)
            or p_mean <= 0.0
        ):
            return self._make_invalid_featureset("Signal has zero power; autocorrelation features are undefined")

        # Obtain canonical cached AutocorrelationResult
        try:
            ac = context.autocorrelation()
        except Exception as exc:
            raise CorrelationFeatureError(f"Failed to obtain autocorrelation from context: {exc}") from exc

        if not hasattr(ac, "lag_samples") or not hasattr(ac, "values"):
            raise CorrelationFeatureError(
                f"AutocorrelationResult must contain lag_samples and values arrays, got {type(ac).__name__}"
            )

        try:
            lags = np.asarray(ac.lag_samples)
            vals = np.asarray(ac.values, dtype=np.complex128)
        except Exception as exc:
            raise CorrelationFeatureError(f"Cannot convert autocorrelation data to numpy arrays: {exc}") from exc

        if lags.ndim != 1 or vals.ndim != 1 or lags.size == 0 or vals.size == 0:
            raise CorrelationFeatureError("Autocorrelation arrays must be non-empty 1-dimensional arrays")

        if lags.size != vals.size:
            raise CorrelationFeatureError(
                f"Autocorrelation array length mismatch: lag_samples has {lags.size}, values has {vals.size}"
            )

        if not np.issubdtype(lags.dtype, np.integer):
            raise CorrelationFeatureError(f"lag_samples must be integer-valued, got dtype {lags.dtype}")

        if not np.all(np.isfinite(lags)) or not np.all(np.isfinite(vals)):
            raise CorrelationFeatureError("Autocorrelation arrays contain non-finite values (NaN or Inf)")

        if np.any(lags < 0):
            raise CorrelationFeatureError("lag_samples must be non-negative (>= 0)")

        if len(lags) > 1 and np.any(np.diff(lags) <= 0):
            raise CorrelationFeatureError("lag_samples must be strictly increasing")

        zero_indices = np.where(lags == 0)[0]
        if len(zero_indices) != 1:
            raise CorrelationFeatureError("Lag 0 must exist exactly once in lag_samples")

        r0 = vals[zero_indices[0]]
        r0_mag = abs(r0)
        if r0_mag <= 0.0 or not math.isfinite(r0_mag):
            return self._make_invalid_featureset("Signal has zero power; autocorrelation features are undefined")

        # Normalize locally so features are amplitude-scale invariant
        rho = vals / r0

        nonzero_mask = lags > 0
        if not np.any(nonzero_mask):
            return self._make_invalid_featureset("At least one positive autocorrelation lag is required")

        lags_nz = lags[nonzero_mask]
        rho_nz = rho[nonzero_mask]
        mag_nz = np.abs(rho_nz)

        # Sanity check normalized magnitude
        if np.any(mag_nz > 1.0 + 1e-4):
            raise CorrelationFeatureError(
                f"Normalized autocorrelation magnitude exceeds 1.0: max={float(np.max(mag_nz)):.6g}"
            )

        # Clamp tiny floating-point rounding excursions
        mag_nz = np.clip(mag_nz, 0.0, 1.0)

        # 1-4. Nonzero peak features
        # If every positive-lag correlation magnitude is zero, no genuine peak exists
        has_positive_peak = bool(np.any(mag_nz > 0.0))

        if not has_positive_peak:
            peak_reason = "No positive-lag correlation peak observed (all positive-lag magnitudes are zero)"
            fv_peak_mag = FeatureValue(value=0.0, unit=None, valid=True, reason=None)
            fv_peak_lag_samples = FeatureValue(
                value=float("nan"),
                unit="samples",
                valid=False,
                reason=peak_reason,
            )
            fv_peak_lag_s = FeatureValue(
                value=float("nan"),
                unit="s",
                valid=False,
                reason=peak_reason,
            )
            fv_peak_phase = FeatureValue(
                value=float("nan"),
                unit="rad",
                valid=False,
                reason=peak_reason,
            )
        else:
            # Deterministic argmax selects smallest lag on tie between genuine nonzero peaks
            i_peak = int(np.argmax(mag_nz))
            peak_mag = float(mag_nz[i_peak])
            peak_lag_samples = float(lags_nz[i_peak])
            peak_lag_s = float(peak_lag_samples / fs)
            peak_phase = float(np.angle(rho_nz[i_peak]))

            fv_peak_mag = FeatureValue(
                value=peak_mag,
                unit=None,
                valid=True,
                reason=None,
            )
            fv_peak_lag_samples = FeatureValue(
                value=peak_lag_samples,
                unit="samples",
                valid=True,
                reason=None,
            )
            fv_peak_lag_s = FeatureValue(
                value=peak_lag_s,
                unit="s",
                valid=True,
                reason=None,
            )
            fv_peak_phase = FeatureValue(
                value=peak_phase,
                unit="rad",
                valid=True,
                reason=None,
            )

        # 5-6. Magnitude statistics across all positive lags (valid zero statistics when all zeros)
        mean_mag = float(np.clip(np.mean(mag_nz), 0.0, 1.0))
        rms_mag = float(np.clip(np.sqrt(np.mean(mag_nz ** 2)), 0.0, 1.0))

        features: dict[str, FeatureValue] = {
            "correlation.nonzero_peak_magnitude": fv_peak_mag,
            "correlation.nonzero_peak_lag_samples": fv_peak_lag_samples,
            "correlation.nonzero_peak_lag_s": fv_peak_lag_s,
            "correlation.nonzero_peak_phase_rad": fv_peak_phase,
            "correlation.mean_nonzero_magnitude": FeatureValue(
                value=mean_mag,
                unit=None,
                valid=True,
                reason=None,
            ),
            "correlation.rms_nonzero_magnitude": FeatureValue(
                value=rms_mag,
                unit=None,
                valid=True,
                reason=None,
            ),
        }

        return FeatureSet(
            extractor=self.name,
            extractor_version=self.version,
            features=features,
        )

    def _make_invalid_featureset(self, reason: str) -> FeatureSet:
        """Helper to create a FeatureSet where all 6 correlation features are marked invalid."""
        features = {
            "correlation.nonzero_peak_magnitude": FeatureValue(
                value=float("nan"),
                unit=None,
                valid=False,
                reason=reason,
            ),
            "correlation.nonzero_peak_lag_samples": FeatureValue(
                value=float("nan"),
                unit="samples",
                valid=False,
                reason=reason,
            ),
            "correlation.nonzero_peak_lag_s": FeatureValue(
                value=float("nan"),
                unit="s",
                valid=False,
                reason=reason,
            ),
            "correlation.nonzero_peak_phase_rad": FeatureValue(
                value=float("nan"),
                unit="rad",
                valid=False,
                reason=reason,
            ),
            "correlation.mean_nonzero_magnitude": FeatureValue(
                value=float("nan"),
                unit=None,
                valid=False,
                reason=reason,
            ),
            "correlation.rms_nonzero_magnitude": FeatureValue(
                value=float("nan"),
                unit=None,
                valid=False,
                reason=reason,
            ),
        }
        return FeatureSet(
            extractor=self.name,
            extractor_version=self.version,
            features=features,
        )
