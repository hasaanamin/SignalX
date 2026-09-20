"""Power-domain feature extractor for RF signal analysis."""

import math
from typing import Any

import numpy as np

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, FeatureValue
from signal_processing.features.power.exceptions import PowerFeatureError
from signal_processing.transforms import linear_power_to_dbfs


class PowerFeatureExtractor:
    """Stateless feature extractor for power-domain signal characteristics.

    Extracts the following canonical scalar features from the signal power:
        - power.mean_linear: Average (mean) instantaneous power [linear]
        - power.mean_dbfs: Average (mean) power relative to full-scale [dBFS]
        - power.peak_linear: Maximum instantaneous power [linear]
        - power.peak_dbfs: Peak instantaneous power relative to full-scale [dBFS]
        - power.min_linear: Minimum instantaneous power [linear]
        - power.std_linear: Population standard deviation of power (ddof=0) [linear]
        - power.variance_linear: Population variance of power (ddof=0) [linear^2]
        - power.median_linear: Median instantaneous power [linear]
        - power.papr_linear: Peak-to-Average Power Ratio (peak / mean) [None]
        - power.papr_db: Peak-to-Average Power Ratio in decibels (10 * log10(PAPR)) [dB]
    """

    name: str = "power"
    version: str = "1.0"

    def extract(self, context: SignalContext | Any) -> FeatureSet:
        """Extract power-domain features from a SignalContext.

        Args:
            context: SignalContext instance (or duck-typed object exposing `power()`,
                and optionally `mean_power_linear()` and `mean_power_dbfs()`).

        Returns:
            FeatureSet containing the 10 canonical power-domain FeatureValues.

        Raises:
            PowerFeatureError: If context is None, missing `power()`, or returns
                invalid/empty power data.
        """
        if context is None:
            raise PowerFeatureError("Context must not be None")

        if not hasattr(context, "power") or not callable(context.power):
            raise PowerFeatureError(
                f"Expected SignalContext instance or object with power() method, got {type(context).__name__}"
            )

        try:
            p = context.power()
        except Exception as exc:
            raise PowerFeatureError(f"Failed to obtain power from context: {exc}") from exc

        try:
            arr = np.asarray(p)
        except Exception as exc:
            raise PowerFeatureError(f"Cannot convert power to numpy array: {exc}") from exc

        if arr.ndim != 1 or arr.size == 0:
            raise PowerFeatureError(
                f"Power must be a non-empty 1-dimensional array, got shape {arr.shape}"
            )

        if not (
            np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)
        ) or np.iscomplexobj(arr):
            raise PowerFeatureError(
                f"Power must be a real-valued numeric array, got dtype {arr.dtype}"
            )

        if not np.all(np.isfinite(arr)):
            raise PowerFeatureError("Power contains non-finite values (NaN or Inf)")

        if np.any(arr < 0.0):
            raise PowerFeatureError("Power contains negative values")

        # Convert to float64 for high numerical precision calculations
        p_f64 = arr.astype(np.float64, copy=False)

        # Retrieve power floor in dBFS if configured on context, otherwise default to -300.0
        floor_dbfs = -300.0
        if hasattr(context, "config") and hasattr(context.config, "power_floor_dbfs"):
            try:
                floor_dbfs = float(context.config.power_floor_dbfs)
            except (TypeError, ValueError):
                floor_dbfs = -300.0

        # 1. Mean Linear Power (reuse context method if available)
        if hasattr(context, "mean_power_linear") and callable(context.mean_power_linear):
            try:
                mean_linear = float(context.mean_power_linear())
            except Exception as exc:
                raise PowerFeatureError(f"Failed to obtain mean_power_linear from context: {exc}") from exc
        else:
            mean_linear = float(np.mean(p_f64))

        # 2. Mean Power dBFS (reuse context method if available)
        if hasattr(context, "mean_power_dbfs") and callable(context.mean_power_dbfs):
            try:
                mean_dbfs = float(context.mean_power_dbfs())
            except Exception as exc:
                raise PowerFeatureError(f"Failed to obtain mean_power_dbfs from context: {exc}") from exc
        else:
            try:
                mean_dbfs = float(linear_power_to_dbfs(mean_linear, floor_dbfs=floor_dbfs))
            except Exception as exc:
                raise PowerFeatureError(f"Failed to convert mean power to dBFS: {exc}") from exc

        # 3. Peak Linear Power
        peak_linear = float(np.max(p_f64))

        # 4. Peak Power dBFS (canonical transform on scalar peak power)
        try:
            peak_dbfs = float(linear_power_to_dbfs(peak_linear, floor_dbfs=floor_dbfs))
        except Exception as exc:
            raise PowerFeatureError(f"Failed to convert peak power to dBFS: {exc}") from exc

        # 5. Minimum Linear Power
        min_linear = float(np.min(p_f64))

        # 6. Standard Deviation (population, ddof=0)
        std_linear = float(np.std(p_f64, ddof=0))

        # 7. Variance (population, ddof=0)
        variance_linear = float(np.var(p_f64, ddof=0))

        # 8. Median Linear Power
        median_linear = float(np.median(p_f64))

        # 9 & 10. PAPR (Peak-to-Average Power Ratio: linear and dB)
        if mean_linear > 0.0 and math.isfinite(mean_linear):
            papr_linear_val = float(peak_linear / mean_linear)
            papr_db_val = float(10.0 * math.log10(papr_linear_val))
            papr_linear_fv = FeatureValue(
                value=papr_linear_val,
                unit=None,
                valid=True,
                reason=None,
            )
            papr_db_fv = FeatureValue(
                value=papr_db_val,
                unit="dB",
                valid=True,
                reason=None,
            )
        else:
            papr_linear_fv = FeatureValue(
                value=float("nan"),
                unit=None,
                valid=False,
                reason="Mean power is zero",
            )
            papr_db_fv = FeatureValue(
                value=float("nan"),
                unit="dB",
                valid=False,
                reason="Mean power is zero",
            )

        features: dict[str, FeatureValue] = {
            "power.mean_linear": FeatureValue(value=mean_linear, unit="linear", valid=True),
            "power.mean_dbfs": FeatureValue(value=mean_dbfs, unit="dBFS", valid=True),
            "power.peak_linear": FeatureValue(value=peak_linear, unit="linear", valid=True),
            "power.peak_dbfs": FeatureValue(value=peak_dbfs, unit="dBFS", valid=True),
            "power.min_linear": FeatureValue(value=min_linear, unit="linear", valid=True),
            "power.std_linear": FeatureValue(value=std_linear, unit="linear", valid=True),
            "power.variance_linear": FeatureValue(value=variance_linear, unit="linear^2", valid=True),
            "power.median_linear": FeatureValue(value=median_linear, unit="linear", valid=True),
            "power.papr_linear": papr_linear_fv,
            "power.papr_db": papr_db_fv,
        }

        return FeatureSet(
            extractor=self.name,
            extractor_version=self.version,
            features=features,
        )
