"""Frequency-domain location feature extractor for RF signal analysis."""

import math
from typing import Any

import numpy as np

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, FeatureValue
from signal_processing.features.frequency.exceptions import FrequencyFeatureError
from signal_processing.transforms.exceptions import TransformError


class FrequencyFeatureExtractor:
    """Stateless feature extractor for complex-baseband frequency location features.

    Extracts the following canonical scalar features from the signal Power Spectral Density (PSD):
        - frequency.peak_offset_hz: Frequency offset of the maximum linear PSD bin [Hz]
        - frequency.center_offset_hz: Power-weighted spectral center (first spectral moment) [Hz]
    """

    name: str = "frequency"
    version: str = "1.0"

    def _make_invalid_featureset(self, reason: str) -> FeatureSet:
        """Create a FeatureSet with all frequency features marked invalid."""
        features: dict[str, FeatureValue] = {
            "frequency.peak_offset_hz": FeatureValue(
                value=float("nan"),
                unit="Hz",
                valid=False,
                reason=reason,
            ),
            "frequency.center_offset_hz": FeatureValue(
                value=float("nan"),
                unit="Hz",
                valid=False,
                reason=reason,
            ),
        }
        return FeatureSet(
            extractor=self.name,
            extractor_version=self.version,
            features=features,
        )

    def extract(self, context: SignalContext | Any) -> FeatureSet:
        """Extract frequency location features from a SignalContext.

        Args:
            context: SignalContext instance (or duck-typed object exposing `psd()`).

        Returns:
            FeatureSet containing the 2 canonical frequency-domain FeatureValues.

        Raises:
            FrequencyFeatureError: If context is None, missing `psd()`, or returns
                invalid PSD data.
        """
        if context is None:
            raise FrequencyFeatureError("Context must not be None")

        if not hasattr(context, "psd") or not callable(context.psd):
            raise FrequencyFeatureError(
                f"Expected SignalContext instance or object with psd() method, got {type(context).__name__}"
            )

        try:
            psd = context.psd()
        except TransformError as exc:
            return self._make_invalid_featureset(f"PSD unavailable: {exc}")
        except Exception as exc:
            raise FrequencyFeatureError(f"Failed to obtain PSD from context: {exc}") from exc

        if not hasattr(psd, "frequency_offset_hz") or not hasattr(psd, "power_linear"):
            raise FrequencyFeatureError(
                f"PSDResult must contain frequency_offset_hz and power_linear arrays, got {type(psd).__name__}"
            )

        try:
            f_arr = np.asarray(psd.frequency_offset_hz)
            p_arr = np.asarray(psd.power_linear)
        except Exception as exc:
            raise FrequencyFeatureError(f"Cannot convert PSD data to numpy arrays: {exc}") from exc

        if f_arr.ndim != 1 or p_arr.ndim != 1:
            raise FrequencyFeatureError(
                f"PSD arrays must be 1-dimensional, got shapes {f_arr.shape} and {p_arr.shape}"
            )

        if f_arr.size == 0 or p_arr.size == 0:
            raise FrequencyFeatureError("PSD arrays must not be empty")

        if f_arr.size != p_arr.size:
            raise FrequencyFeatureError(
                f"PSD array length mismatch: frequency_offset_hz has {f_arr.size} samples, "
                f"power_linear has {p_arr.size} samples"
            )

        if not (
            np.issubdtype(f_arr.dtype, np.floating) or np.issubdtype(f_arr.dtype, np.integer)
        ) or np.iscomplexobj(f_arr):
            raise FrequencyFeatureError(
                f"frequency_offset_hz must be real-valued numeric, got dtype {f_arr.dtype}"
            )

        if not (
            np.issubdtype(p_arr.dtype, np.floating) or np.issubdtype(p_arr.dtype, np.integer)
        ) or np.iscomplexobj(p_arr):
            raise FrequencyFeatureError(
                f"power_linear must be real-valued numeric, got dtype {p_arr.dtype}"
            )

        if not np.all(np.isfinite(f_arr)) or not np.all(np.isfinite(p_arr)):
            raise FrequencyFeatureError("PSD arrays must contain only finite values (no NaN or Inf)")

        if np.any(p_arr < 0.0):
            raise FrequencyFeatureError("power_linear must contain non-negative values (>= 0)")

        # High precision accumulation
        f_f64 = f_arr.astype(np.float64, copy=False)
        p_f64 = p_arr.astype(np.float64, copy=False)

        max_p = float(np.max(p_f64))
        total_p = float(np.sum(p_f64))

        # 1. Peak Frequency Offset (argmax of linear PSD)
        if max_p > 0.0 and math.isfinite(max_p):
            k_peak = int(np.argmax(p_f64))
            peak_hz = float(f_f64[k_peak])
            peak_fv = FeatureValue(
                value=peak_hz,
                unit="Hz",
                valid=True,
                reason=None,
            )
        else:
            peak_fv = FeatureValue(
                value=float("nan"),
                unit="Hz",
                valid=False,
                reason="PSD contains no positive signal power",
            )

        # 2. Spectral Center Frequency Offset (power-weighted first spectral moment)
        if total_p > 0.0 and math.isfinite(total_p):
            weighted_sum = float(np.sum(f_f64 * p_f64, dtype=np.float64))
            center_hz = float(weighted_sum / total_p)
            center_fv = FeatureValue(
                value=center_hz,
                unit="Hz",
                valid=True,
                reason=None,
            )
        else:
            center_fv = FeatureValue(
                value=float("nan"),
                unit="Hz",
                valid=False,
                reason="PSD contains no positive signal power",
            )

        features: dict[str, FeatureValue] = {
            "frequency.peak_offset_hz": peak_fv,
            "frequency.center_offset_hz": center_fv,
        }

        return FeatureSet(
            extractor=self.name,
            extractor_version=self.version,
            features=features,
        )
