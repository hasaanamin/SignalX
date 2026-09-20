"""Tests for circular resultant length and circular variance features."""

import numpy as np
import pytest

from signal_processing.core import PhaseResult
from signal_processing.features.phase import PhaseFeatureExtractor


class DummyPhaseContext:
    """Mock context providing phase() and mean_power_linear()."""

    def __init__(self, wrapped_rad: np.ndarray, unwrapped_rad: np.ndarray, mean_power: float = 1.0) -> None:
        self._phase = PhaseResult(wrapped_rad=wrapped_rad, unwrapped_rad=unwrapped_rad)
        self._mean_power = mean_power

    def phase(self) -> PhaseResult:
        return self._phase

    def mean_power_linear(self) -> float:
        return self._mean_power


class TestCircularStatistics:
    """Validate circular resultant length and circular variance formulations."""

    def test_case_a_constant_phase(self) -> None:
        """Test Case A: Constant phase (all zeros) has resultant length 1.0 and variance 0.0."""
        wrapped = np.zeros(100, dtype=np.float64)
        unwrapped = np.zeros(100, dtype=np.float64)
        ctx = DummyPhaseContext(wrapped, unwrapped)

        extractor = PhaseFeatureExtractor()
        fset = extractor.extract(ctx)

        r = fset.features["phase.circular_resultant_length"]
        v = fset.features["phase.circular_variance"]

        assert r.valid is True
        assert r.unit is None
        assert pytest.approx(r.value, abs=1e-12) == 1.0

        assert v.valid is True
        assert v.unit is None
        assert pytest.approx(v.value, abs=1e-12) == 0.0

    def test_case_e_wrap_boundary_circular_statistics(self) -> None:
        """Test Case E: Phases near +179 and -179 degrees are clustered in circular domain (R ~ 1, Var ~ 0)."""
        # Linear statistics would falsely compute enormous variance near the wrap boundary
        angles_deg = np.array([179.0, -179.0, 178.0, -178.0], dtype=np.float64)
        wrapped = np.radians(angles_deg)
        unwrapped = wrapped.copy()
        ctx = DummyPhaseContext(wrapped, unwrapped)

        extractor = PhaseFeatureExtractor()
        fset = extractor.extract(ctx)

        r = fset.features["phase.circular_resultant_length"]
        v = fset.features["phase.circular_variance"]

        assert r.valid is True
        assert r.value > 0.999
        assert v.valid is True
        assert v.value < 0.001
        assert pytest.approx(r.value + v.value, abs=1e-12) == 1.0

    def test_case_f_uniform_circle(self) -> None:
        """Test Case F: Angles uniformly spaced around circle (0, pi/2, pi, -pi/2) have R ~ 0 and Var ~ 1."""
        wrapped = np.array([0.0, np.pi / 2.0, np.pi, -np.pi / 2.0], dtype=np.float64)
        unwrapped = wrapped.copy()
        ctx = DummyPhaseContext(wrapped, unwrapped)

        extractor = PhaseFeatureExtractor()
        fset = extractor.extract(ctx)

        r = fset.features["phase.circular_resultant_length"]
        v = fset.features["phase.circular_variance"]

        assert r.valid is True
        assert pytest.approx(r.value, abs=1e-12) == 0.0

        assert v.valid is True
        assert pytest.approx(v.value, abs=1e-12) == 1.0
