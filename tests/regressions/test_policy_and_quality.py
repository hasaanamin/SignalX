"""Regression tests for feature policy recording and capture quality propagation.

Requirements 7 & 8:
- Exact 56 default scalar feature names and units preserved.
- FeatureVector.to_array() exports frozen values array, boolean validity mask, and ordered names tuple.
- Explicit NaN for invalid features; never silent zero substitution.
- ExtractionPolicy records full resolved DSP parameters (nfft, window, psd_mode, sample_rate, etc.).
- Reproduction: spectral entropy variation across nfft=128, 512, 2048 is captured by policy metadata.
- CaptureQuality propagates complete=False, dropped_samples, and continuity status.
- AnalysisResult envelope aggregates FeatureVector, ExtractionPolicy, and CaptureQuality.
"""

import math

import numpy as np
import pytest

from signal_processing.context import AnalysisConfig, SignalContext
from signal_processing.core import (
    AnalysisResult,
    FeatureValue,
    FeatureVector,
    SignalEvent,
    StreamInfo,
)
from signal_processing.features import (
    DEFAULT_EXTRACTOR_NAMES,
    FeatureAggregator,
    build_default_feature_registry,
)


@pytest.fixture
def stream_info() -> StreamInfo:
    return StreamInfo(
        stream_id="stream-policy-test",
        source_id="synthetic",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )


class TestFeaturePolicyAndExport:
    """Validate extraction policy recording and deterministic vector export."""

    def test_default_registry_produces_fifty_six_canonical_features(self, stream_info: StreamInfo) -> None:
        """Default feature registry produces exactly 56 scalar features."""
        n = 512
        iq = np.exp(2j * np.pi * 100_000.0 * np.arange(n) / 1_000_000.0).astype(np.complex64)
        event = SignalEvent(
            event_id="ev-56-keys",
            stream=stream_info,
            iq=iq,
            event_start_sample=0,
            event_end_sample=n,
            detected_start_sample=0,
            detected_end_sample=n,
            trigger_sample=0,
        )

        aggregator = FeatureAggregator(build_default_feature_registry())
        vector = aggregator.extract(SignalContext(event))

        assert len(vector.features) == 56
        assert len(DEFAULT_EXTRACTOR_NAMES) == 10

    def test_vector_to_array_export_with_validity_mask(self, stream_info: StreamInfo) -> None:
        """to_array() returns frozen float array, boolean validity mask, and ordered names."""
        n = 512
        iq = np.exp(2j * np.pi * 100_000.0 * np.arange(n) / 1_000_000.0).astype(np.complex64)
        event = SignalEvent(
            event_id="ev-export",
            stream=stream_info,
            iq=iq,
            event_start_sample=0,
            event_end_sample=n,
            detected_start_sample=0,
            detected_end_sample=n,
            trigger_sample=0,
        )

        aggregator = FeatureAggregator(build_default_feature_registry())
        vector = aggregator.extract(SignalContext(event))

        values, mask, names = vector.to_array()

        assert len(values) == 56
        assert len(mask) == 56
        assert len(names) == 56
        assert values.dtype == np.float64
        assert mask.dtype == bool
        assert values.flags.writeable is False
        assert mask.flags.writeable is False

        # Verify attempting setflags raises ValueError
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag to True"):
            values.setflags(write=True)

        with pytest.raises(ValueError, match="cannot set WRITEABLE flag to True"):
            mask.setflags(write=True)

    def test_invalid_features_exported_as_explicit_nan_not_zero(self) -> None:
        """Invalid FeatureValues are exported as NaN (or custom invalid value), never silent zero."""
        features = {
            "feat_valid": FeatureValue(value=42.0, unit=None, valid=True),
            "feat_invalid": FeatureValue(value=float("nan"), unit=None, valid=False, reason="math error"),
        }
        vec = FeatureVector(
            event_id="ev-nan-export",
            feature_schema_version="1.0",
            features=features,
        )

        vals, mask, names = vec.to_array(["feat_valid", "feat_invalid"])
        assert list(names) == ["feat_valid", "feat_invalid"]
        assert vals[0] == 42.0
        assert bool(mask[0]) is True
        assert math.isnan(vals[1])
        assert bool(mask[1]) is False

        # With custom invalid value -999.0
        vals_custom, _, _ = vec.to_array(["feat_valid", "feat_invalid"], invalid_value=-999.0)
        assert vals_custom[1] == -999.0

    def test_spectral_entropy_configuration_reproduction(self, stream_info: StreamInfo) -> None:
        """Reproduction: same 128-sample tone has different spectral entropy at nfft=128, 512, 2048.

        ExtractionPolicy records nfft explicitly, preserving reproducibility across modeling.
        """
        n = 128
        iq = np.exp(2j * np.pi * 0.1 * np.arange(n)).astype(np.complex64)
        event = SignalEvent(
            event_id="ev-entropy-repro",
            stream=stream_info,
            iq=iq,
            event_start_sample=0,
            event_end_sample=n,
            detected_start_sample=0,
            detected_end_sample=n,
            trigger_sample=0,
        )

        aggregator = FeatureAggregator(build_default_feature_registry())

        # Extract under nfft=128, 512, 2048
        res_128 = aggregator.extract_analysis_result(SignalContext(event, AnalysisConfig(nfft=128)))
        res_512 = aggregator.extract_analysis_result(SignalContext(event, AnalysisConfig(nfft=512)))
        res_2048 = aggregator.extract_analysis_result(SignalContext(event, AnalysisConfig(nfft=2048)))

        # Entropies differ due to FFT zero-padding bin count
        e128 = res_128.feature_vector.features["spectral.entropy"].value
        e512 = res_512.feature_vector.features["spectral.entropy"].value
        e2048 = res_2048.feature_vector.features["spectral.entropy"].value

        assert not np.isclose(e128, e512)
        assert not np.isclose(e512, e2048)

        # Policies explicitly record nfft to prevent unrecorded schema ambiguity
        assert res_128.policy.nfft == 128
        assert res_512.policy.nfft == 512
        assert res_2048.policy.nfft == 2048


class TestCaptureQualityPropagation:
    """Validate capture quality and dropped sample metadata propagation."""

    def test_incomplete_capture_quality_propagation(self, stream_info: StreamInfo) -> None:
        """Event with complete=False and dropped_samples=5 propagates through AnalysisResult."""
        n = 256
        iq = np.ones(n, dtype=np.complex64)
        event = SignalEvent(
            event_id="ev-incomplete",
            stream=stream_info,
            iq=iq,
            event_start_sample=0,
            event_end_sample=n,
            detected_start_sample=0,
            detected_end_sample=n,
            trigger_sample=0,
            dropped_samples=5,
            complete=False,
        )

        ctx = SignalContext(event)
        assert ctx.capture_quality.complete is False
        assert ctx.capture_quality.dropped_samples == 5
        assert ctx.capture_quality.is_continuous is False
        assert len(ctx.capture_quality.warnings) > 0
        assert "dropped_samples=5" in ctx.capture_quality.warnings[0]

        aggregator = FeatureAggregator(build_default_feature_registry())
        analysis_res = aggregator.extract_analysis_result(ctx)

        assert isinstance(analysis_res, AnalysisResult)
        assert analysis_res.capture_quality.complete is False
        assert analysis_res.capture_quality.dropped_samples == 5
        assert analysis_res.capture_quality.is_continuous is False
        assert analysis_res.policy.observation_samples == 256
