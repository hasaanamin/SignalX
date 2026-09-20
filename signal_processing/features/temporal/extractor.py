"""Temporal-domain feature extractor for RF signal analysis."""

import math
from typing import Any

import numpy as np

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, FeatureValue
from signal_processing.features.temporal.exceptions import TemporalFeatureError


class TemporalFeatureExtractor:
    """Stateless feature extractor for intrinsic temporal characteristics of a detected signal.

    Extracts the following canonical scalar features from event timing metadata:
        - temporal.duration_samples: Active detected signal duration in samples [samples]
        - temporal.duration_s: Active detected signal duration in physical seconds [s]
    """

    name: str = "temporal"
    version: str = "1.0"

    def extract(self, context: SignalContext | Any) -> FeatureSet:
        """Extract temporal features from a SignalContext.

        Args:
            context: SignalContext instance (or duck-typed object exposing timing metadata).

        Returns:
            FeatureSet containing the 2 canonical temporal-domain FeatureValues.

        Raises:
            TemporalFeatureError: If context is None, missing required temporal metadata,
                or yields non-positive or non-finite durations.
        """
        if context is None:
            raise TemporalFeatureError("Context must not be None")

        # 1. Obtain duration in samples
        duration_samples: float | None = None
        try:
            val = context.signal_num_samples
            if callable(val):
                val = val()
            if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
                raise TemporalFeatureError(f"signal_num_samples must be numeric, got {val!r}")
            duration_samples = float(val)
        except AttributeError:
            try:
                ev = context.event
                val = ev.signal_samples
                if callable(val):
                    val = val()
                if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
                    raise TemporalFeatureError(f"event.signal_samples must be numeric, got {val!r}")
                duration_samples = float(val)
            except AttributeError:
                raise TemporalFeatureError(
                    f"Expected SignalContext instance or object with signal_num_samples or event, got {type(context).__name__}"
                )
            except Exception as exc:
                if isinstance(exc, TemporalFeatureError):
                    raise
                raise TemporalFeatureError(
                    f"Failed to obtain signal_samples from context event: {exc}"
                ) from exc
        except Exception as exc:
            if isinstance(exc, TemporalFeatureError):
                raise
            raise TemporalFeatureError(
                f"Failed to obtain signal_num_samples from context: {exc}"
            ) from exc

        if not math.isfinite(duration_samples) or duration_samples <= 0:
            raise TemporalFeatureError(
                f"duration_samples must be a positive finite number, got {duration_samples}"
            )

        # 2. Obtain duration in seconds
        duration_s: float | None = None

        # 2a. Try direct signal_duration_s on context
        try:
            val = context.signal_duration_s
            if callable(val):
                val = val()
            if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
                raise TemporalFeatureError(f"signal_duration_s must be numeric, got {val!r}")
            duration_s = float(val)
        except AttributeError:
            pass
        except Exception as exc:
            if isinstance(exc, TemporalFeatureError):
                raise
            raise TemporalFeatureError(
                f"Failed to obtain signal_duration_s from context: {exc}"
            ) from exc

        # 2b. Try event.signal_duration_s
        if duration_s is None:
            try:
                ev = context.event
                val = ev.signal_duration_s
                if callable(val):
                    val = val()
                if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
                    raise TemporalFeatureError(f"event.signal_duration_s must be numeric, got {val!r}")
                duration_s = float(val)
            except AttributeError:
                pass
            except Exception as exc:
                if isinstance(exc, TemporalFeatureError):
                    raise
                raise TemporalFeatureError(
                    f"Failed to obtain signal_duration_s from context event: {exc}"
                ) from exc

        # 2c. Try sample_rate_hz on context
        if duration_s is None:
            try:
                fs_val = context.sample_rate_hz
                if callable(fs_val):
                    fs_val = fs_val()
                if isinstance(fs_val, bool) or not isinstance(
                    fs_val, (int, float, np.integer, np.floating)
                ):
                    raise TemporalFeatureError(f"sample_rate_hz must be numeric, got {fs_val!r}")
                fs = float(fs_val)
                if not math.isfinite(fs) or fs <= 0:
                    raise TemporalFeatureError(
                        f"sample_rate_hz must be positive and finite, got {fs}"
                    )
                duration_s = duration_samples / fs
            except AttributeError:
                pass
            except Exception as exc:
                if isinstance(exc, TemporalFeatureError):
                    raise
                raise TemporalFeatureError(
                    f"Failed to compute duration_s from sample_rate_hz: {exc}"
                ) from exc

        # 2d. Try event.stream.sample_rate_hz
        if duration_s is None:
            try:
                ev = context.event
                st = ev.stream
                fs_val = st.sample_rate_hz
                if callable(fs_val):
                    fs_val = fs_val()
                if isinstance(fs_val, bool) or not isinstance(
                    fs_val, (int, float, np.integer, np.floating)
                ):
                    raise TemporalFeatureError(
                        f"stream.sample_rate_hz must be numeric, got {fs_val!r}"
                    )
                fs = float(fs_val)
                if not math.isfinite(fs) or fs <= 0:
                    raise TemporalFeatureError(
                        f"sample_rate_hz must be positive and finite, got {fs}"
                    )
                duration_s = duration_samples / fs
            except AttributeError:
                pass
            except Exception as exc:
                if isinstance(exc, TemporalFeatureError):
                    raise
                raise TemporalFeatureError(
                    f"Failed to compute duration_s from stream sample_rate_hz: {exc}"
                ) from exc

        if duration_s is None:
            raise TemporalFeatureError(
                f"Cannot determine signal duration in seconds from context of type {type(context).__name__}"
            )

        if not math.isfinite(duration_s) or duration_s <= 0:
            raise TemporalFeatureError(
                f"duration_s must be a positive finite number, got {duration_s}"
            )

        features: dict[str, FeatureValue] = {
            "temporal.duration_samples": FeatureValue(
                value=float(duration_samples),
                unit="samples",
                valid=True,
                reason=None,
            ),
            "temporal.duration_s": FeatureValue(
                value=float(duration_s),
                unit="s",
                valid=True,
                reason=None,
            ),
        }

        return FeatureSet(
            extractor=self.name,
            extractor_version=self.version,
            features=features,
        )
