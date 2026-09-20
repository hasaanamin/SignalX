"""Statistical distribution shape feature extractor for RF signal envelope."""

import math
from typing import Any

import numpy as np

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, FeatureValue
from signal_processing.features.statistics.exceptions import StatisticsFeatureError


class StatisticsFeatureExtractor:
    """Stateless feature extractor for signal envelope statistical distribution shape.

    Extracts the following canonical scalar features from the instantaneous amplitude envelope:
        - stats.amplitude.skewness: Third standardized population central moment [None]
        - stats.amplitude.excess_kurtosis: Fourth standardized population central moment minus 3 [None]
        - stats.amplitude.q01: 1st percentile of amplitude envelope [linear]
        - stats.amplitude.q05: 5th percentile of amplitude envelope [linear]
        - stats.amplitude.q25: 25th percentile of amplitude envelope [linear]
        - stats.amplitude.q75: 75th percentile of amplitude envelope [linear]
        - stats.amplitude.q95: 95th percentile of amplitude envelope [linear]
        - stats.amplitude.q99: 99th percentile of amplitude envelope [linear]
        - stats.amplitude.iqr: Interquartile range (q75 - q25) [linear]
    """

    name: str = "statistics"
    version: str = "1.0"

    def extract(self, context: SignalContext | Any) -> FeatureSet:
        """Extract amplitude distribution statistics features from a SignalContext.

        Args:
            context: SignalContext instance (or duck-typed object exposing `envelope()`).

        Returns:
            FeatureSet containing the 9 canonical statistical FeatureValues.

        Raises:
            StatisticsFeatureError: If context is None, missing `envelope()`, or returns
                invalid, non-numeric, non-finite, negative, or empty envelope data.
        """
        if context is None:
            raise StatisticsFeatureError("Context must not be None")

        if not hasattr(context, "envelope") or not callable(context.envelope):
            raise StatisticsFeatureError(
                f"Expected SignalContext instance or object with envelope() method, got {type(context).__name__}"
            )

        try:
            env = context.envelope()
        except Exception as exc:
            raise StatisticsFeatureError(f"Failed to obtain envelope from context: {exc}") from exc

        try:
            arr = np.asarray(env)
        except Exception as exc:
            raise StatisticsFeatureError(f"Cannot convert envelope to numpy array: {exc}") from exc

        if arr.ndim != 1 or arr.size == 0:
            raise StatisticsFeatureError(
                f"Envelope must be a non-empty 1-dimensional array, got shape {arr.shape}"
            )

        if not (
            np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)
        ) or np.iscomplexobj(arr):
            raise StatisticsFeatureError(
                f"Envelope must be a real-valued numeric array, got dtype {arr.dtype}"
            )

        if not np.all(np.isfinite(arr)):
            raise StatisticsFeatureError("Envelope contains non-finite values (NaN or Inf)")

        if np.any(arr < 0.0):
            raise StatisticsFeatureError("Envelope contains negative values")

        # Convert to float64 for high numerical precision population moments and percentiles
        a = arr.astype(np.float64, copy=False)
        n = a.size

        mu = float(np.mean(a))
        diff = a - mu
        m2 = float(np.mean(diff ** 2))

        # 1. Amplitude Skewness
        if n < 3:
            skew_fv = FeatureValue(
                value=float("nan"),
                unit=None,
                valid=False,
                reason="At least 3 samples are required for skewness",
            )
        elif m2 <= 0.0 or not math.isfinite(m2):
            skew_fv = FeatureValue(
                value=float("nan"),
                unit=None,
                valid=False,
                reason="Amplitude variance is zero",
            )
        else:
            m3 = float(np.mean(diff ** 3))
            skew_val = float(m3 / (m2 ** 1.5))
            skew_fv = FeatureValue(
                value=skew_val,
                unit=None,
                valid=True,
                reason=None,
            )

        # 2. Amplitude Excess Kurtosis
        if n < 4:
            kurt_fv = FeatureValue(
                value=float("nan"),
                unit=None,
                valid=False,
                reason="At least 4 samples are required for excess kurtosis",
            )
        elif m2 <= 0.0 or not math.isfinite(m2):
            kurt_fv = FeatureValue(
                value=float("nan"),
                unit=None,
                valid=False,
                reason="Amplitude variance is zero",
            )
        else:
            m4 = float(np.mean(diff ** 4))
            raw_kurtosis = m4 / (m2 ** 2)
            excess_kurt_val = float(raw_kurtosis - 3.0)
            kurt_fv = FeatureValue(
                value=excess_kurt_val,
                unit=None,
                valid=True,
                reason=None,
            )

        # 3-8. Quantiles (using NumPy's standard linear interpolation)
        q_vals = np.percentile(a, [1.0, 5.0, 25.0, 75.0, 95.0, 99.0])
        q01 = float(q_vals[0])
        q05 = float(q_vals[1])
        q25 = float(q_vals[2])
        q75 = float(q_vals[3])
        q95 = float(q_vals[4])
        q99 = float(q_vals[5])

        # 9. Interquartile Range (IQR = q75 - q25)
        iqr_val = float(q75 - q25)

        features: dict[str, FeatureValue] = {
            "stats.amplitude.skewness": skew_fv,
            "stats.amplitude.excess_kurtosis": kurt_fv,
            "stats.amplitude.q01": FeatureValue(
                value=q01,
                unit="linear",
                valid=True,
                reason=None,
            ),
            "stats.amplitude.q05": FeatureValue(
                value=q05,
                unit="linear",
                valid=True,
                reason=None,
            ),
            "stats.amplitude.q25": FeatureValue(
                value=q25,
                unit="linear",
                valid=True,
                reason=None,
            ),
            "stats.amplitude.q75": FeatureValue(
                value=q75,
                unit="linear",
                valid=True,
                reason=None,
            ),
            "stats.amplitude.q95": FeatureValue(
                value=q95,
                unit="linear",
                valid=True,
                reason=None,
            ),
            "stats.amplitude.q99": FeatureValue(
                value=q99,
                unit="linear",
                valid=True,
                reason=None,
            ),
            "stats.amplitude.iqr": FeatureValue(
                value=iqr_val,
                unit="linear",
                valid=True,
                reason=None,
            ),
        }

        return FeatureSet(
            extractor=self.name,
            extractor_version=self.version,
            features=features,
        )
