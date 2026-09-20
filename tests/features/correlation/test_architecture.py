"""Architectural and contract adherence tests for CorrelationFeatureExtractor."""

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import (
    AutocorrelationResult,
    FeatureSet,
    SignalEvent,
    StreamInfo,
)
from signal_processing.features.correlation import (
    CorrelationFeatureError,
    CorrelationFeatureExtractor,
)


class DummyAC:
    """Unvalidated container to test extractor resilience against malformed upstream objects."""

    def __init__(self, lag_samples: Any, values: Any) -> None:
        self.lag_samples = lag_samples
        self.values = values


class MinimalCorrelationContext:
    """Mock context exposing ONLY autocorrelation(), mean_power_linear(), and sample_rate_hz."""

    def __init__(
        self,
        lag_samples: Any,
        values: Any,
        sample_rate_hz: float = 1_000_000.0,
        mean_power: float = 1.0,
        use_dummy_ac: bool = False,
    ) -> None:
        if use_dummy_ac:
            self._ac = DummyAC(lag_samples=lag_samples, values=values)
        else:
            self._ac = AutocorrelationResult(lag_samples=lag_samples, values=values)
        self._mean_power = mean_power
        self.sample_rate_hz = sample_rate_hz

    def autocorrelation(self) -> AutocorrelationResult:
        return self._ac

    def mean_power_linear(self) -> float:
        return self._mean_power

    @property
    def signal_iq(self) -> Any:
        """Explode if raw IQ is accessed to prove zero raw IQ access."""
        raise AssertionError("CorrelationFeatureExtractor must NOT access context.signal_iq!")


@pytest.fixture
def sample_event() -> SignalEvent:
    """Fixture providing a standard 1 MHz pulse event."""
    stream = StreamInfo(
        stream_id="stream-corr-arch",
        source_id="synthetic-src",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    n = np.arange(256)
    iq = np.exp(1j * 2.0 * np.pi * 50_000.0 * n / 1_000_000.0).astype(np.complex64)

    return SignalEvent(
        event_id="ev-corr-arch-001",
        stream=stream,
        iq=iq,
        event_start_sample=0,
        event_end_sample=256,
        detected_start_sample=16,
        detected_end_sample=240,
        trigger_sample=64,
    )


class TestArchitectureAdherence:
    """Test strict separation of concerns, minimal interface consumption, and exception handling."""

    def test_minimal_context_consumption_and_no_raw_iq_access(self) -> None:
        """Extractor must function with minimal context and never access context.signal_iq."""
        lags = np.array([0, 1, 2, 3], dtype=np.int64)
        values = np.array([1.0 + 0j, 0.5 + 0j, 0.2 + 0j, 0.1 + 0j], dtype=np.complex128)
        ctx = MinimalCorrelationContext(lags, values, sample_rate_hz=1_000_000.0)

        extractor = CorrelationFeatureExtractor()
        fset = extractor.extract(ctx)

        assert isinstance(fset, FeatureSet)
        assert len(fset.features) == 6
        assert fset.features["correlation.nonzero_peak_magnitude"].value == 0.5

    def test_no_unwanted_dsp_transforms_triggered(self, sample_event: SignalEvent) -> None:
        """Extractor must only trigger mean_power_linear and autocorrelation, never FFT, PSD, phase, etc."""
        ctx = SignalContext(sample_event)
        extractor = CorrelationFeatureExtractor()

        assert ctx.cached_keys == frozenset()

        _ = extractor.extract(ctx)

        # Only mean_power_linear and autocorrelation should be cached
        assert "mean_power_linear" in ctx.cached_keys
        assert "autocorrelation" in ctx.cached_keys

        forbidden_cached_transforms = [
            "fft",
            "psd",
            "phase",
            "envelope",
            "instantaneous_frequency",
            "power",
            "mean_power_dbfs",
        ]
        for transform_name in forbidden_cached_transforms:
            assert transform_name not in ctx.cached_keys

    def test_forbidden_keys_not_exposed(self, sample_event: SignalEvent) -> None:
        """Verify that multi-signal, delay, periodicity, or template features are NOT exposed."""
        ctx = SignalContext(sample_event)
        extractor = CorrelationFeatureExtractor()
        fset = extractor.extract(ctx)

        allowed_keys = {
            "correlation.nonzero_peak_magnitude",
            "correlation.nonzero_peak_lag_samples",
            "correlation.nonzero_peak_lag_s",
            "correlation.nonzero_peak_phase_rad",
            "correlation.mean_nonzero_magnitude",
            "correlation.rms_nonzero_magnitude",
        }
        assert set(fset.features.keys()) == allowed_keys

        forbidden_keys = [
            "correlation.delay_samples",
            "correlation.estimated_delay_s",
            "correlation.delay",
            "correlation.finddelay",
            "correlation.period_samples",
            "correlation.period_s",
            "correlation.pri",
            "correlation.prf",
            "correlation.pulse_repetition_period",
            "cross_correlation",
            "inter_pulse_correlation",
            "waveform_similarity",
            "similarity",
            "template_matching",
            "dtw",
            "sequence_similarity",
        ]
        for key in forbidden_keys:
            assert key not in fset.features

    def test_reject_invalid_context_types(self) -> None:
        """Verify CorrelationFeatureError is raised when context is None or lacks required attributes."""
        extractor = CorrelationFeatureExtractor()

        with pytest.raises(CorrelationFeatureError, match="Context must not be None"):
            extractor.extract(None)  # type: ignore[arg-type]

        with pytest.raises(CorrelationFeatureError, match="mean_power_linear"):
            extractor.extract("not_a_context")  # type: ignore[arg-type]

        class MissingAutocorrelation:
            def mean_power_linear(self) -> float:
                return 1.0

        with pytest.raises(CorrelationFeatureError, match="autocorrelation"):
            extractor.extract(MissingAutocorrelation())

        class MissingSampleRate:
            def mean_power_linear(self) -> float:
                return 1.0

            def autocorrelation(self) -> Any:
                return None

        with pytest.raises(CorrelationFeatureError, match="sample_rate_hz"):
            extractor.extract(MissingSampleRate())

        class InvalidSampleRate:
            sample_rate_hz = -100.0

            def mean_power_linear(self) -> float:
                return 1.0

            def autocorrelation(self) -> Any:
                return None

        with pytest.raises(CorrelationFeatureError, match="sample_rate_hz must be a positive finite"):
            extractor.extract(InvalidSampleRate())

    def test_reject_malformed_autocorrelation_results(self) -> None:
        """Verify CorrelationFeatureError is raised for non-compliant AutocorrelationResult structures."""
        extractor = CorrelationFeatureExtractor()

        # 1. Result missing attributes
        class BadResultContext:
            sample_rate_hz = 1000.0

            def mean_power_linear(self) -> float:
                return 1.0

            def autocorrelation(self) -> Any:
                return "not_an_autocorrelation_result"

        with pytest.raises(CorrelationFeatureError, match="must contain lag_samples and values"):
            extractor.extract(BadResultContext())

        # 2. Multi-dimensional arrays
        ctx_2d = MinimalCorrelationContext(
            np.zeros((2, 2), dtype=np.int64),
            np.zeros((2, 2), dtype=np.complex128),
            use_dummy_ac=True,
        )
        with pytest.raises(CorrelationFeatureError, match="1-dimensional"):
            extractor.extract(ctx_2d)

        # 3. Empty arrays
        ctx_empty = MinimalCorrelationContext(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.complex128),
            use_dummy_ac=True,
        )
        with pytest.raises(CorrelationFeatureError, match="non-empty"):
            extractor.extract(ctx_empty)

        # 4. Length mismatch
        ctx_mismatch = MinimalCorrelationContext(
            np.array([0, 1], dtype=np.int64),
            np.array([1.0 + 0j], dtype=np.complex128),
            use_dummy_ac=True,
        )
        with pytest.raises(CorrelationFeatureError, match="array length mismatch"):
            extractor.extract(ctx_mismatch)

        # 5. Non-integer lag dtype
        ctx_float_lags = MinimalCorrelationContext(
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0 + 0j, 0.5 + 0j], dtype=np.complex128),
            use_dummy_ac=True,
        )
        with pytest.raises(CorrelationFeatureError, match="integer-valued"):
            extractor.extract(ctx_float_lags)

        # 6. Non-finite values
        ctx_nan_val = MinimalCorrelationContext(
            np.array([0, 1], dtype=np.int64),
            np.array([1.0 + 0j, float("nan") + 0j], dtype=np.complex128),
            use_dummy_ac=True,
        )
        with pytest.raises(CorrelationFeatureError, match="non-finite values"):
            extractor.extract(ctx_nan_val)

        # 7. Negative lags
        ctx_neg_lag = MinimalCorrelationContext(
            np.array([-1, 0, 1], dtype=np.int64),
            np.array([0.5 + 0j, 1.0 + 0j, 0.5 + 0j], dtype=np.complex128),
            use_dummy_ac=True,
        )
        with pytest.raises(CorrelationFeatureError, match="non-negative"):
            extractor.extract(ctx_neg_lag)

        # 8. Non-increasing lags
        ctx_non_inc = MinimalCorrelationContext(
            np.array([0, 2, 1], dtype=np.int64),
            np.array([1.0 + 0j, 0.5 + 0j, 0.5 + 0j], dtype=np.complex128),
            use_dummy_ac=True,
        )
        with pytest.raises(CorrelationFeatureError, match="strictly increasing"):
            extractor.extract(ctx_non_inc)

        # 9. Missing lag 0
        ctx_missing_zero = MinimalCorrelationContext(
            np.array([1, 2, 3], dtype=np.int64),
            np.array([0.5 + 0j, 0.2 + 0j, 0.1 + 0j], dtype=np.complex128),
            use_dummy_ac=True,
        )
        with pytest.raises(CorrelationFeatureError, match="Lag 0 must exist exactly once"):
            extractor.extract(ctx_missing_zero)

        # 10. Multiple lag 0
        ctx_multiple_zero = MinimalCorrelationContext(
            np.array([0, 0, 1], dtype=np.int64),
            np.array([1.0 + 0j, 1.0 + 0j, 0.5 + 0j], dtype=np.complex128),
            use_dummy_ac=True,
        )
        # diff <= 0 will catch it or zero_indices len != 1
        with pytest.raises(CorrelationFeatureError):
            extractor.extract(ctx_multiple_zero)

    def test_no_reference_package_imports(self) -> None:
        """Inspect source AST of correlation package to ensure no imports of reference code or sibling feature packages."""
        pkg_dir = Path(__file__).resolve().parents[3] / "signal_processing" / "features" / "correlation"

        forbidden_import_substrings = [
            "reference",
            "correlation_and_similarity",
            "Correlation and Similarity",
            "features.phase",
            "features.frequency",
            "features.statistics",
            "features.higher_order",
            "features.amplitude",
            "features.bandwidth",
            "features.spectral",
            "features.power",
            "features.temporal",
        ]

        for py_file in pkg_dir.glob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for forbidden in forbidden_import_substrings:
                            assert forbidden not in alias.name, f"Forbidden import '{alias.name}' in {py_file.name}"
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    for forbidden in forbidden_import_substrings:
                        assert forbidden not in mod, f"Forbidden import from '{mod}' in {py_file.name}"
