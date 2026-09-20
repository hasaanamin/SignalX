"""Spectral-shape feature extractor for RF signal analysis."""

import math
from typing import Any

import numpy as np

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, FeatureValue
from signal_processing.features.spectral.exceptions import SpectralFeatureError
from signal_processing.transforms.exceptions import TransformError


class SpectralFeatureExtractor:
    """Stateless feature extractor for spectral-shape characteristics.

    Extracts the following canonical scalar features from the signal Power Spectral Density (PSD):
        - spectral.entropy: Normalized Shannon spectral entropy [None]
        - spectral.flatness: Spectral flatness (ratio of geometric to arithmetic mean of PSD) [None]
        - spectral.rolloff_85_offset_hz: 85% spectral rolloff frequency offset [Hz]
        - spectral.rolloff_95_offset_hz: 95% spectral rolloff frequency offset [Hz]
    """

    name: str = "spectral"
    version: str = "1.0"

    def _make_invalid_featureset(self, reason: str) -> FeatureSet:
        """Create a FeatureSet with all spectral features marked invalid."""
        features: dict[str, FeatureValue] = {
            "spectral.entropy": FeatureValue(
                value=float("nan"),
                unit=None,
                valid=False,
                reason=reason,
            ),
            "spectral.flatness": FeatureValue(
                value=float("nan"),
                unit=None,
                valid=False,
                reason=reason,
            ),
            "spectral.rolloff_85_offset_hz": FeatureValue(
                value=float("nan"),
                unit="Hz",
                valid=False,
                reason=reason,
            ),
            "spectral.rolloff_95_offset_hz": FeatureValue(
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
        """Extract spectral-shape features from a SignalContext.

        Args:
            context: SignalContext instance (or duck-typed object exposing `psd()`).

        Returns:
            FeatureSet containing the 4 canonical spectral-shape FeatureValues.

        Raises:
            SpectralFeatureError: If context is None, missing `psd()`, or returns
                invalid, malformed, or non-monotonic PSD data.
        """
        if context is None:
            raise SpectralFeatureError("Context must not be None")

        if not hasattr(context, "psd") or not callable(context.psd):
            raise SpectralFeatureError(
                f"Expected SignalContext instance or object with psd() method, got {type(context).__name__}"
            )

        try:
            psd = context.psd()
        except TransformError as exc:
            return self._make_invalid_featureset(f"PSD unavailable: {exc}")
        except Exception as exc:
            raise SpectralFeatureError(f"Failed to obtain PSD from context: {exc}") from exc

        if not hasattr(psd, "frequency_offset_hz") or not hasattr(psd, "power_linear"):
            raise SpectralFeatureError(
                f"PSDResult must contain frequency_offset_hz and power_linear arrays, got {type(psd).__name__}"
            )

        try:
            f_arr = np.asarray(psd.frequency_offset_hz)
            p_arr = np.asarray(psd.power_linear)
        except Exception as exc:
            raise SpectralFeatureError(f"Cannot convert PSD data to numpy arrays: {exc}") from exc

        if f_arr.ndim != 1 or p_arr.ndim != 1:
            raise SpectralFeatureError(
                f"PSD arrays must be 1-dimensional, got shapes {f_arr.shape} and {p_arr.shape}"
            )

        if f_arr.size == 0 or p_arr.size == 0:
            raise SpectralFeatureError("PSD arrays must not be empty")

        if f_arr.size != p_arr.size:
            raise SpectralFeatureError(
                f"PSD array length mismatch: frequency_offset_hz has {f_arr.size} samples, "
                f"power_linear has {p_arr.size} samples"
            )

        if not (
            np.issubdtype(f_arr.dtype, np.floating) or np.issubdtype(f_arr.dtype, np.integer)
        ) or np.iscomplexobj(f_arr):
            raise SpectralFeatureError(
                f"frequency_offset_hz must be real-valued numeric, got dtype {f_arr.dtype}"
            )

        if not (
            np.issubdtype(p_arr.dtype, np.floating) or np.issubdtype(p_arr.dtype, np.integer)
        ) or np.iscomplexobj(p_arr):
            raise SpectralFeatureError(
                f"power_linear must be real-valued numeric, got dtype {p_arr.dtype}"
            )

        if not np.all(np.isfinite(f_arr)) or not np.all(np.isfinite(p_arr)):
            raise SpectralFeatureError("PSD arrays must contain only finite values (no NaN or Inf)")

        if np.any(p_arr < 0.0):
            raise SpectralFeatureError("power_linear must contain non-negative values (>= 0)")

        # Create sorted local view for analysis without mutating PSDResult
        order = np.argsort(f_arr)
        f = f_arr[order].astype(np.float64, copy=False)
        p = p_arr[order].astype(np.float64, copy=False)

        if len(f) > 1 and np.any(np.diff(f) <= 0.0):
            raise SpectralFeatureError(
                "Frequency axis must contain strictly increasing distinct values"
            )

        n_bins = len(p)
        p_total = float(np.sum(p, dtype=np.float64))
        has_positive_power = (p_total > 0.0) and math.isfinite(p_total)

        # 1. Normalized Shannon Spectral Entropy
        if not has_positive_power:
            entropy_fv = FeatureValue(
                value=float("nan"),
                unit=None,
                valid=False,
                reason="PSD contains no positive signal power",
            )
        elif n_bins < 2:
            entropy_fv = FeatureValue(
                value=float("nan"),
                unit=None,
                valid=False,
                reason="At least two PSD bins are required for normalized spectral entropy",
            )
        else:
            prob = p / p_total
            nonzero_prob = prob[prob > 0.0]
            raw_h = float(-np.sum(nonzero_prob * np.log(nonzero_prob)))
            norm_factor = math.log(n_bins)
            entropy_val = float(np.clip(raw_h / norm_factor, 0.0, 1.0))
            entropy_fv = FeatureValue(
                value=entropy_val,
                unit=None,
                valid=True,
                reason=None,
            )

        # 2. Spectral Flatness
        if not has_positive_power:
            flatness_fv = FeatureValue(
                value=float("nan"),
                unit=None,
                valid=False,
                reason="PSD contains no positive signal power",
            )
        elif np.any(p == 0.0):
            flatness_fv = FeatureValue(
                value=0.0,
                unit=None,
                valid=True,
                reason=None,
            )
        else:
            geom_mean = float(np.exp(np.mean(np.log(p))))
            arith_mean = float(np.mean(p))
            flatness_val = float(np.clip(geom_mean / arith_mean, 0.0, 1.0))
            flatness_fv = FeatureValue(
                value=flatness_val,
                unit=None,
                valid=True,
                reason=None,
            )

        # 3. 85% Spectral Rolloff
        if not has_positive_power:
            rolloff_85_fv = FeatureValue(
                value=float("nan"),
                unit="Hz",
                valid=False,
                reason="PSD contains no positive signal power",
            )
        else:
            cumulative = np.cumsum(p, dtype=np.float64)
            target_85 = 0.85 * p_total
            k_85 = int(np.searchsorted(cumulative, target_85, side="left"))
            k_85 = min(k_85, n_bins - 1)
            rolloff_85_val = float(f[k_85])
            rolloff_85_fv = FeatureValue(
                value=rolloff_85_val,
                unit="Hz",
                valid=True,
                reason=None,
            )

        # 4. 95% Spectral Rolloff
        if not has_positive_power:
            rolloff_95_fv = FeatureValue(
                value=float("nan"),
                unit="Hz",
                valid=False,
                reason="PSD contains no positive signal power",
            )
        else:
            cumulative = np.cumsum(p, dtype=np.float64)
            target_95 = 0.95 * p_total
            k_95 = int(np.searchsorted(cumulative, target_95, side="left"))
            k_95 = min(k_95, n_bins - 1)
            rolloff_95_val = float(f[k_95])
            rolloff_95_fv = FeatureValue(
                value=rolloff_95_val,
                unit="Hz",
                valid=True,
                reason=None,
            )

        features: dict[str, FeatureValue] = {
            "spectral.entropy": entropy_fv,
            "spectral.flatness": flatness_fv,
            "spectral.rolloff_85_offset_hz": rolloff_85_fv,
            "spectral.rolloff_95_offset_hz": rolloff_95_fv,
        }

        return FeatureSet(
            extractor=self.name,
            extractor_version=self.version,
            features=features,
        )
