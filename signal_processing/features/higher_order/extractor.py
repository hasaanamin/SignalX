"""Higher-order statistics feature extractor for complex baseband IQ signals."""

import math
from typing import Any

import numpy as np

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, FeatureValue
from signal_processing.features.higher_order.exceptions import HigherOrderFeatureError


class HigherOrderFeatureExtractor:
    """Stateless feature extractor for scale-normalized higher-order complex IQ cumulants.

    Extracts the following canonical scalar features from the complex signal IQ:
        - higher_order.normalized_abs_c20: Magnitude of normalized second-order unconjugated cumulant |C20| [None]
        - higher_order.normalized_abs_c40: Magnitude of normalized fourth-order unconjugated cumulant |C40| [None]
        - higher_order.normalized_abs_c41: Magnitude of normalized fourth-order single-conjugated cumulant |C41| [None]
        - higher_order.normalized_abs_c42: Magnitude of normalized fourth-order conjugate-symmetric cumulant |C42| [None]
        - higher_order.normalized_abs_c60: Magnitude of normalized sixth-order unconjugated cumulant |C60| [None]
        - higher_order.normalized_abs_c63: Magnitude of normalized sixth-order conjugate-symmetric cumulant |C63| [None]
    """

    name: str = "higher_order"
    version: str = "1.0"

    def extract(self, context: SignalContext | Any) -> FeatureSet:
        """Extract higher-order cumulant features from a SignalContext.

        Args:
            context: SignalContext instance (or duck-typed object exposing `signal_iq`).

        Returns:
            FeatureSet containing the 6 canonical higher-order FeatureValues.

        Raises:
            HigherOrderFeatureError: If context is None, missing `signal_iq`, or returns
                invalid, empty, multidimensional, or non-finite IQ data.
        """
        if context is None:
            raise HigherOrderFeatureError("Context must not be None")

        if not hasattr(context, "signal_iq"):
            raise HigherOrderFeatureError(
                f"Expected SignalContext instance or object with signal_iq attribute, got {type(context).__name__}"
            )

        raw_iq = context.signal_iq
        if raw_iq is None:
            raise HigherOrderFeatureError("signal_iq must not be None")

        try:
            z = np.asarray(raw_iq, dtype=np.complex128)
        except Exception as exc:
            raise HigherOrderFeatureError(f"Cannot convert signal_iq to complex array: {exc}") from exc

        if z.ndim != 1 or z.size == 0:
            raise HigherOrderFeatureError(
                f"signal_iq must be a non-empty 1-dimensional array, got shape {z.shape}"
            )

        if not np.all(np.isfinite(z)):
            raise HigherOrderFeatureError("signal_iq contains non-finite values (NaN or Inf)")

        # Case: Insufficient sample count (< 2 samples)
        if z.size < 2:
            return self._make_invalid_featureset("At least two IQ samples are required")

        # Zero-mean centered representation
        mu = np.mean(z)
        z0 = z - mu

        # Centered second-order power: C21 = E[|z0|^2]
        centered_power = float(np.mean(np.abs(z0) ** 2))

        # Case: Degenerate signal (centered power is zero)
        if centered_power <= 0.0 or not math.isfinite(centered_power):
            return self._make_invalid_featureset("Centered signal power is zero")

        # Unit-power pre-normalization for numerical stability across high orders
        sigma = math.sqrt(centered_power)
        u = z0 / sigma

        u_conj = np.conj(u)
        u2 = u * u
        u_abs2 = np.abs(u) ** 2
        u3 = u2 * u
        u_abs4 = u_abs2 * u_abs2
        u4 = u2 * u2
        u6 = u4 * u2
        u_abs6 = u_abs4 * u_abs2

        # Population sample expectations: E[x] = mean(x)
        m20 = np.mean(u2)
        m21 = np.mean(u_abs2)
        m30 = np.mean(u3)
        m31 = np.mean(u2 * u_conj)
        m40 = np.mean(u4)
        m41 = np.mean(u3 * u_conj)
        m42 = np.mean(u_abs4)
        m60 = np.mean(u6)
        m63 = np.mean(u_abs6)

        # Second-order cumulant
        c20 = m20

        # Fourth-order cumulants
        c40 = m40 - 3.0 * (m20 ** 2)
        c41 = m41 - 3.0 * m20 * m21
        c42 = m42 - (np.abs(m20) ** 2) - 2.0 * (m21 ** 2)

        # Full sixth-order cumulants (retaining all validated reference terms)
        c60 = (
            m60
            - 15.0 * m20 * m40
            - 10.0 * (m30 ** 2)
            + 30.0 * (m20 ** 3)
        )
        c63 = (
            m63
            - 9.0 * m21 * m42
            - 3.0 * m20 * np.conj(m41)
            - 3.0 * np.conj(m20) * m41
            - 9.0 * (np.abs(m31) ** 2)
            - (np.abs(m30) ** 2)
            + 18.0 * (np.abs(m20) ** 2) * m21
            + 12.0 * (m21 ** 3)
        )

        abs_c20 = float(abs(c20))
        abs_c40 = float(abs(c40))
        abs_c41 = float(abs(c41))
        abs_c42 = float(abs(c42))
        abs_c60 = float(abs(c60))
        abs_c63 = float(abs(c63))

        values = {
            "higher_order.normalized_abs_c20": abs_c20,
            "higher_order.normalized_abs_c40": abs_c40,
            "higher_order.normalized_abs_c41": abs_c41,
            "higher_order.normalized_abs_c42": abs_c42,
            "higher_order.normalized_abs_c60": abs_c60,
            "higher_order.normalized_abs_c63": abs_c63,
        }

        features: dict[str, FeatureValue] = {}
        for key, val in values.items():
            if math.isfinite(val) and val >= 0.0:
                features[key] = FeatureValue(
                    value=val,
                    unit=None,
                    valid=True,
                    reason=None,
                )
            else:
                features[key] = FeatureValue(
                    value=float("nan"),
                    unit=None,
                    valid=False,
                    reason="Numerical calculation resulted in non-finite value",
                )

        return FeatureSet(
            extractor=self.name,
            extractor_version=self.version,
            features=features,
        )

    def _make_invalid_featureset(self, reason: str) -> FeatureSet:
        """Helper to create a FeatureSet where all 6 features are marked invalid."""
        feature_names = [
            "higher_order.normalized_abs_c20",
            "higher_order.normalized_abs_c40",
            "higher_order.normalized_abs_c41",
            "higher_order.normalized_abs_c42",
            "higher_order.normalized_abs_c60",
            "higher_order.normalized_abs_c63",
        ]
        features = {
            name: FeatureValue(
                value=float("nan"),
                unit=None,
                valid=False,
                reason=reason,
            )
            for name in feature_names
        }
        return FeatureSet(
            extractor=self.name,
            extractor_version=self.version,
            features=features,
        )
