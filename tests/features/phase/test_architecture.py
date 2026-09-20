"""Architectural and contract adherence tests for PhaseFeatureExtractor."""

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from signal_processing.context import SignalContext
from signal_processing.core import (
    FeatureSet,
    PhaseResult,
    SignalEvent,
    StreamInfo,
)
from signal_processing.features.phase import (
    PhaseFeatureError,
    PhaseFeatureExtractor,
)


class DummyPhaseResult:
    """Unvalidated container to test extractor resilience against malformed upstream objects."""

    def __init__(self, wrapped_rad: Any, unwrapped_rad: Any) -> None:
        self.wrapped_rad = wrapped_rad
        self.unwrapped_rad = unwrapped_rad


class MinimalPhaseContext:
    """Mock context exposing ONLY phase() and mean_power_linear()."""

    def __init__(
        self,
        wrapped_rad: Any,
        unwrapped_rad: Any,
        mean_power: float = 1.0,
        use_dummy_result: bool = False,
    ) -> None:
        if use_dummy_result:
            self._phase = DummyPhaseResult(wrapped_rad=wrapped_rad, unwrapped_rad=unwrapped_rad)
        else:
            self._phase = PhaseResult(wrapped_rad=wrapped_rad, unwrapped_rad=unwrapped_rad)
        self._mean_power = mean_power

    def phase(self) -> Any:
        return self._phase

    def mean_power_linear(self) -> float:
        return self._mean_power

    @property
    def signal_iq(self) -> Any:
        """Explode if raw IQ is accessed to prove zero raw IQ access."""
        raise AssertionError("PhaseFeatureExtractor must NOT access context.signal_iq!")


@pytest.fixture
def sample_event() -> SignalEvent:
    """Fixture providing a standard 1 MHz pulse event."""
    stream = StreamInfo(
        stream_id="stream-phase-arch",
        source_id="synthetic-src",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )
    n = np.arange(256)
    iq = np.exp(1j * 2.0 * np.pi * 50_000.0 * n / 1_000_000.0).astype(np.complex64)

    return SignalEvent(
        event_id="ev-phase-arch-001",
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
        wrapped = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        unwrapped = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        ctx = MinimalPhaseContext(wrapped, unwrapped)

        extractor = PhaseFeatureExtractor()
        fset = extractor.extract(ctx)

        assert isinstance(fset, FeatureSet)
        assert len(fset.features) == 6
        assert fset.features["phase.increment_mean_rad"].value == pytest.approx(0.1)

    def test_no_unwanted_dsp_transforms_triggered(self, sample_event: SignalEvent) -> None:
        """Extractor must only trigger mean_power_linear and phase, never FFT, PSD, autocorrelation, etc."""
        ctx = SignalContext(sample_event)
        extractor = PhaseFeatureExtractor()

        assert ctx.cached_keys == frozenset()

        _ = extractor.extract(ctx)

        # Only mean_power_linear and phase should be cached
        assert "mean_power_linear" in ctx.cached_keys
        assert "phase" in ctx.cached_keys

        forbidden_cached_transforms = [
            "fft",
            "psd",
            "autocorrelation",
            "envelope",
            "instantaneous_frequency",
            "power",
            "mean_power_dbfs",
        ]
        for transform_name in forbidden_cached_transforms:
            assert transform_name not in ctx.cached_keys

    def test_forbidden_keys_not_exposed(self, sample_event: SignalEvent) -> None:
        """Verify absolute phase, frequency, and modulation/phase-state features are NOT exposed."""
        ctx = SignalContext(sample_event)
        extractor = PhaseFeatureExtractor()
        fset = extractor.extract(ctx)

        allowed_keys = {
            "phase.circular_resultant_length",
            "phase.circular_variance",
            "phase.increment_mean_rad",
            "phase.increment_std_rad",
            "phase.increment_rms_rad",
            "phase.increment_abs_mean_rad",
        }
        assert set(fset.features.keys()) == allowed_keys

        forbidden_keys = [
            "phase.mean_rad",
            "phase.initial_rad",
            "phase.absolute_phase_rad",
            "phase.phase_offset_rad",
            "phase.frequency",
            "phase.cfo",
            "phase.phase_states",
            "phase.constellation",
            "phase.modulation",
            "phase.bpsk",
            "phase.qpsk",
        ]
        for key in forbidden_keys:
            assert key not in fset.features

    def test_reject_invalid_context_types(self) -> None:
        """Verify PhaseFeatureError is raised when context is None or lacks required attributes."""
        extractor = PhaseFeatureExtractor()

        with pytest.raises(PhaseFeatureError, match="Context must not be None"):
            extractor.extract(None)  # type: ignore[arg-type]

        with pytest.raises(PhaseFeatureError, match="mean_power_linear"):
            extractor.extract("not_a_context")  # type: ignore[arg-type]

        class MissingPhase:
            def mean_power_linear(self) -> float:
                return 1.0

        with pytest.raises(PhaseFeatureError, match="phase"):
            extractor.extract(MissingPhase())

    def test_reject_malformed_phase_results(self) -> None:
        """Verify PhaseFeatureError is raised for non-compliant PhaseResult structures."""
        extractor = PhaseFeatureExtractor()

        # 1. Result missing attributes
        class BadResultContext:
            def mean_power_linear(self) -> float:
                return 1.0

            def phase(self) -> Any:
                return "not_a_phase_result"

        with pytest.raises(PhaseFeatureError, match="must contain wrapped_rad and unwrapped_rad"):
            extractor.extract(BadResultContext())

        # 2. Multi-dimensional arrays
        ctx_2d = MinimalPhaseContext(
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
            use_dummy_result=True,
        )
        with pytest.raises(PhaseFeatureError, match="1-dimensional"):
            extractor.extract(ctx_2d)

        # 3. Empty arrays
        ctx_empty = MinimalPhaseContext(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            use_dummy_result=True,
        )
        with pytest.raises(PhaseFeatureError, match="non-empty"):
            extractor.extract(ctx_empty)

        # 4. Length mismatch
        ctx_mismatch = MinimalPhaseContext(
            np.array([0.0, 0.1], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            use_dummy_result=True,
        )
        with pytest.raises(PhaseFeatureError, match="array length mismatch"):
            extractor.extract(ctx_mismatch)

        # 5. Complex phase dtype
        ctx_complex = MinimalPhaseContext(
            np.array([0.0 + 1.0j], dtype=np.complex128),
            np.array([0.0 + 1.0j], dtype=np.complex128),
            use_dummy_result=True,
        )
        with pytest.raises(PhaseFeatureError, match="real-valued"):
            extractor.extract(ctx_complex)

        # 6. Non-finite values
        ctx_nan = MinimalPhaseContext(
            np.array([0.0, float("nan")], dtype=np.float64),
            np.array([0.0, 0.1], dtype=np.float64),
            use_dummy_result=True,
        )
        with pytest.raises(PhaseFeatureError, match="non-finite values"):
            extractor.extract(ctx_nan)

    def test_no_reference_package_imports(self) -> None:
        """Inspect source AST of phase package to ensure no imports of reference code or sibling feature packages."""
        pkg_dir = Path(__file__).resolve().parents[3] / "signal_processing" / "features" / "phase"

        forbidden_import_substrings = [
            "reference",
            "features.correlation",
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
