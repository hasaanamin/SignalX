"""Amplitude and envelope domain feature extractor."""

import math
from typing import Any

import numpy as np

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, FeatureValue
from signal_processing.features.amplitude.config import AmplitudeFeatureConfig
from signal_processing.features.amplitude.exceptions import AmplitudeFeatureError


class AmplitudeFeatureExtractor:
    """Stateless feature extractor for amplitude and envelope domain characteristics.

    Extracts the following canonical scalar features from the signal envelope:
        - amplitude.mean: Arithmetic mean of envelope samples [linear]
        - amplitude.rms: Root-mean-square of envelope samples [linear]
        - amplitude.peak: Maximum instantaneous envelope sample [linear]
        - amplitude.min: Minimum instantaneous envelope sample [linear]
        - amplitude.std: Population standard deviation (ddof=0) [linear]
        - amplitude.variance: Population variance (ddof=0) [linear^2]
        - amplitude.median: Median envelope value [linear]
        - amplitude.peak_to_peak: Peak-to-peak amplitude range (max - min) [linear]
        - amplitude.crest_factor: Ratio of peak to RMS amplitude (peak / rms) [None]
    """

    name: str = "amplitude"
    version: str = "1.0"

    def __init__(self, config: AmplitudeFeatureConfig | None = None) -> None:
        """Initialize AmplitudeFeatureExtractor with optional configuration.

        Args:
            config: Optional AmplitudeFeatureConfig. Defaults to default settings.

        Raises:
            AmplitudeFeatureError: If config is not an AmplitudeFeatureConfig instance.
        """
        if config is None:
            self._config = AmplitudeFeatureConfig()
        elif isinstance(config, AmplitudeFeatureConfig):
            self._config = config
        else:
            raise AmplitudeFeatureError(
                f"Expected AmplitudeFeatureConfig instance or None, got {type(config).__name__}"
            )

    @property
    def config(self) -> AmplitudeFeatureConfig:
        """The immutable configuration for this extractor."""
        return self._config

    def extract(self, context: SignalContext | Any) -> FeatureSet:
        """Extract amplitude features from a SignalContext.

        Args:
            context: SignalContext instance (or duck-typed object exposing `envelope()`).

        Returns:
            FeatureSet containing the 9 canonical amplitude-domain FeatureValues.

        Raises:
            AmplitudeFeatureError: If context is None, missing `envelope()`, or returns
                invalid/empty envelope data.
        """
        if context is None:
            raise AmplitudeFeatureError("Context must not be None")

        if not hasattr(context, "envelope") or not callable(context.envelope):
            raise AmplitudeFeatureError(
                f"Expected SignalContext instance or object with envelope() method, got {type(context).__name__}"
            )


        try:
            env = context.envelope()
        except Exception as exc:
            raise AmplitudeFeatureError(f"Failed to obtain envelope from context: {exc}") from exc

        try:
            arr = np.asarray(env)
        except Exception as exc:
            raise AmplitudeFeatureError(f"Cannot convert envelope to numpy array: {exc}") from exc

        if arr.ndim != 1 or arr.size == 0:
            raise AmplitudeFeatureError(
                f"Envelope must be a non-empty 1-dimensional array, got shape {arr.shape}"
            )

        if not (
            np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)
        ) or np.iscomplexobj(arr):
            raise AmplitudeFeatureError(
                f"Envelope must be a real-valued numeric array, got dtype {arr.dtype}"
            )

        if not np.all(np.isfinite(arr)):
            raise AmplitudeFeatureError("Envelope contains non-finite values (NaN or Inf)")

        if np.any(arr < 0.0):
            raise AmplitudeFeatureError("Envelope contains negative values")

        # Convert to float64 for high numerical precision calculations
        env_f64 = arr.astype(np.float64, copy=False)
        ddof = self._config.ddof

        # 1. Mean
        mean_val = float(np.mean(env_f64))

        # 2. RMS
        rms_val = float(np.sqrt(np.mean(env_f64 ** 2)))

        # 3. Peak
        peak_val = float(np.max(env_f64))

        # 4. Min
        min_val = float(np.min(env_f64))

        # 5. Standard Deviation and 6. Variance
        num_samples = len(env_f64)
        if num_samples <= ddof:
            dof_reason = f"Sample count ({num_samples}) must be strictly greater than ddof ({ddof})"
            std_fv = FeatureValue(value=float("nan"), unit="linear", valid=False, reason=dof_reason)
            var_fv = FeatureValue(value=float("nan"), unit="linear^2", valid=False, reason=dof_reason)
        else:
            std_val = float(np.std(env_f64, ddof=ddof))
            var_val = float(np.var(env_f64, ddof=ddof))
            std_fv = FeatureValue(value=std_val, unit="linear", valid=True, reason=None)
            var_fv = FeatureValue(value=var_val, unit="linear^2", valid=True, reason=None)

        # 7. Median
        median_val = float(np.median(env_f64))

        # 8. Peak-to-Peak
        ptp_val = float(peak_val - min_val)

        # 9. Crest Factor (peak / rms)
        if rms_val > 0.0 and math.isfinite(rms_val):
            cf_val = float(peak_val / rms_val)
            crest_factor_fv = FeatureValue(
                value=cf_val,
                unit=None,
                valid=True,
                reason=None,
            )
        else:
            crest_factor_fv = FeatureValue(
                value=float("nan"),
                unit=None,
                valid=False,
                reason="RMS amplitude is zero",
            )

        features: dict[str, FeatureValue] = {
            "amplitude.mean": FeatureValue(value=mean_val, unit="linear", valid=True),
            "amplitude.rms": FeatureValue(value=rms_val, unit="linear", valid=True),
            "amplitude.peak": FeatureValue(value=peak_val, unit="linear", valid=True),
            "amplitude.min": FeatureValue(value=min_val, unit="linear", valid=True),
            "amplitude.std": std_fv,
            "amplitude.variance": var_fv,
            "amplitude.median": FeatureValue(value=median_val, unit="linear", valid=True),
            "amplitude.peak_to_peak": FeatureValue(value=ptp_val, unit="linear", valid=True),
            "amplitude.crest_factor": crest_factor_fv,
        }

        return FeatureSet(
            extractor=self.name,
            extractor_version=self.version,
            features=features,
        )
