"""Tests for PowerFeatureExtractor mathematical accuracy, units, and error handling."""

import math

import numpy as np
import pytest

from signal_processing.core import FeatureSet
from signal_processing.features.power import (
    PowerFeatureError,
    PowerFeatureExtractor,
)


class DummyPowerContext:
    """Minimal duck-typed context providing power-domain methods."""

    def __init__(
        self,
        power: np.ndarray,
        mean_power_linear: float | None = None,
        mean_power_dbfs: float | None = None,
    ) -> None:
        self._power = power
        self._mean_power_linear = mean_power_linear
        self._mean_power_dbfs = mean_power_dbfs

    def power(self) -> np.ndarray:
        return self._power

    def mean_power_linear(self) -> float:
        if self._mean_power_linear is not None:
            return self._mean_power_linear
        return float(np.mean(self._power))

    def mean_power_dbfs(self) -> float:
        if self._mean_power_dbfs is not None:
            return self._mean_power_dbfs
        mean_p = self.mean_power_linear()
        if mean_p <= 0.0:
            return -300.0
        return float(10.0 * math.log10(mean_p))


class TestPowerExtractor:
    """Test power feature extraction on synthetic power vectors."""

    def test_unit_amplitude_tone(self) -> None:
        """Unit power [1.0, ...] must produce known exact metrics."""
        p = np.ones(100, dtype=np.float64)
        ctx = DummyPowerContext(p)
        extractor = PowerFeatureExtractor()

        fset = extractor.extract(ctx)

        assert isinstance(fset, FeatureSet)
        assert fset.extractor == "power"
        assert fset.extractor_version == "1.0"
        assert len(fset.features) == 10

        assert fset.features["power.mean_linear"].value == 1.0
        assert fset.features["power.mean_linear"].unit == "linear"
        assert fset.features["power.mean_linear"].valid is True

        assert fset.features["power.mean_dbfs"].value == 0.0
        assert fset.features["power.mean_dbfs"].unit == "dBFS"
        assert fset.features["power.mean_dbfs"].valid is True

        assert fset.features["power.peak_linear"].value == 1.0
        assert fset.features["power.peak_linear"].unit == "linear"
        assert fset.features["power.peak_linear"].valid is True

        assert fset.features["power.peak_dbfs"].value == 0.0
        assert fset.features["power.peak_dbfs"].unit == "dBFS"
        assert fset.features["power.peak_dbfs"].valid is True

        assert fset.features["power.min_linear"].value == 1.0
        assert fset.features["power.min_linear"].unit == "linear"
        assert fset.features["power.min_linear"].valid is True

        assert fset.features["power.std_linear"].value == 0.0
        assert fset.features["power.std_linear"].unit == "linear"
        assert fset.features["power.std_linear"].valid is True

        assert fset.features["power.variance_linear"].value == 0.0
        assert fset.features["power.variance_linear"].unit == "linear^2"
        assert fset.features["power.variance_linear"].valid is True

        assert fset.features["power.median_linear"].value == 1.0
        assert fset.features["power.median_linear"].unit == "linear"
        assert fset.features["power.median_linear"].valid is True

        assert fset.features["power.papr_linear"].value == 1.0
        assert fset.features["power.papr_linear"].unit is None
        assert fset.features["power.papr_linear"].valid is True

        assert fset.features["power.papr_db"].value == 0.0
        assert fset.features["power.papr_db"].unit == "dB"
        assert fset.features["power.papr_db"].valid is True

    def test_half_amplitude_tone(self) -> None:
        """Half-amplitude tone (power 0.25) must match expected linear and dBFS values."""
        p = np.full(100, 0.25, dtype=np.float64)
        ctx = DummyPowerContext(p)
        extractor = PowerFeatureExtractor()

        fset = extractor.extract(ctx)

        expected_dbfs = 10.0 * math.log10(0.25)  # ≈ -6.020599913 dBFS

        assert pytest.approx(fset.features["power.mean_linear"].value) == 0.25
        assert pytest.approx(fset.features["power.peak_linear"].value) == 0.25
        assert pytest.approx(fset.features["power.mean_dbfs"].value) == expected_dbfs
        assert pytest.approx(fset.features["power.peak_dbfs"].value) == expected_dbfs
        assert pytest.approx(fset.features["power.min_linear"].value) == 0.25
        assert pytest.approx(fset.features["power.std_linear"].value) == 0.0
        assert pytest.approx(fset.features["power.variance_linear"].value) == 0.0
        assert pytest.approx(fset.features["power.median_linear"].value) == 0.25
        assert pytest.approx(fset.features["power.papr_linear"].value) == 1.0
        assert pytest.approx(fset.features["power.papr_db"].value) == 0.0

    def test_known_varying_power_vector(self) -> None:
        """Known power vector [1.0, 2.0, 3.0, 4.0] must match exact analytical definitions."""
        p = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        ctx = DummyPowerContext(p)
        extractor = PowerFeatureExtractor()

        fset = extractor.extract(ctx)

        # Analytical expectations:
        # mean_linear = (1 + 2 + 3 + 4) / 4 = 2.5
        # mean_dbfs = 10 * log10(2.5) ≈ 3.979400086720376 dBFS
        # peak_linear = 4.0
        # peak_dbfs = 10 * log10(4.0) ≈ 6.020599913279624 dBFS
        # min_linear = 1.0
        # std_linear (ddof=0) = sqrt(1.25)
        # variance_linear (ddof=0) = 1.25
        # median_linear = 2.5
        # papr_linear = 4.0 / 2.5 = 1.6
        # papr_db = 10 * log10(1.6) ≈ 2.041199826559248 dB

        expected_mean = 2.5
        expected_mean_dbfs = 10.0 * math.log10(2.5)
        expected_peak = 4.0
        expected_peak_dbfs = 10.0 * math.log10(4.0)
        expected_min = 1.0
        expected_std = math.sqrt(1.25)
        expected_var = 1.25
        expected_median = 2.5
        expected_papr_lin = 1.6
        expected_papr_db = 10.0 * math.log10(1.6)

        assert pytest.approx(fset.features["power.mean_linear"].value) == expected_mean
        assert pytest.approx(fset.features["power.mean_dbfs"].value) == expected_mean_dbfs
        assert pytest.approx(fset.features["power.peak_linear"].value) == expected_peak
        assert pytest.approx(fset.features["power.peak_dbfs"].value) == expected_peak_dbfs
        assert pytest.approx(fset.features["power.min_linear"].value) == expected_min
        assert pytest.approx(fset.features["power.std_linear"].value) == expected_std
        assert pytest.approx(fset.features["power.variance_linear"].value) == expected_var
        assert pytest.approx(fset.features["power.median_linear"].value) == expected_median
        assert pytest.approx(fset.features["power.papr_linear"].value) == expected_papr_lin
        assert pytest.approx(fset.features["power.papr_db"].value) == expected_papr_db

        # Check population statistics strictly differ from sample statistics (ddof=1)
        sample_std = float(np.std(p, ddof=1))
        assert fset.features["power.std_linear"].value != sample_std

        # Check native Python float types
        for k, fv in fset.features.items():
            assert isinstance(fv.value, float), f"Feature {k} value is not float: {type(fv.value)}"
            assert type(fv.value) is float

    def test_peaky_signal_papr(self) -> None:
        """Signal with one dominant peak must have PAPR > 1 and PAPR_dB > 0."""
        p = np.array([0.1, 0.1, 0.1, 0.1, 10.0], dtype=np.float64)
        ctx = DummyPowerContext(p)
        extractor = PowerFeatureExtractor()

        fset = extractor.extract(ctx)

        papr_lin = fset.features["power.papr_linear"].value
        papr_db = fset.features["power.papr_db"].value

        assert papr_lin > 1.0
        assert papr_db > 0.0
        assert fset.features["power.papr_linear"].valid is True
        assert fset.features["power.papr_db"].valid is True

    def test_reject_invalid_context(self) -> None:
        """Verify strict error handling when context is malformed or invalid."""
        extractor = PowerFeatureExtractor()

        with pytest.raises(PowerFeatureError, match="Context must not be None"):
            extractor.extract(None)  # type: ignore[arg-type]

        with pytest.raises(PowerFeatureError, match="power"):
            extractor.extract("not_a_context")  # type: ignore[arg-type]

    def test_reject_invalid_power_data(self) -> None:
        """Verify extractor rejects empty, multidimensional, non-finite, complex, or negative power arrays."""
        extractor = PowerFeatureExtractor()

        class BrokenContext:
            def power(self) -> np.ndarray:
                raise RuntimeError("Hardware failure")

        with pytest.raises(PowerFeatureError, match="Failed to obtain power"):
            extractor.extract(BrokenContext())

        with pytest.raises(PowerFeatureError, match="non-empty"):
            extractor.extract(DummyPowerContext(np.array([])))

        with pytest.raises(PowerFeatureError, match="1-dimensional"):
            extractor.extract(DummyPowerContext(np.zeros((5, 2))))

        with pytest.raises(PowerFeatureError, match="real-valued"):
            extractor.extract(DummyPowerContext(np.array([1.0 + 2.0j, 3.0])))

        with pytest.raises(PowerFeatureError, match="non-finite"):
            extractor.extract(DummyPowerContext(np.array([1.0, np.nan, 2.0])))

        with pytest.raises(PowerFeatureError, match="non-finite"):
            extractor.extract(DummyPowerContext(np.array([1.0, np.inf, 2.0])))

        with pytest.raises(PowerFeatureError, match="negative"):
            extractor.extract(DummyPowerContext(np.array([1.0, -0.5, 2.0])))

    def test_reject_broken_mean_methods(self) -> None:
        """Verify error handling when context mean power methods raise exceptions."""
        extractor = PowerFeatureExtractor()

        class BrokenMeanLinearContext(DummyPowerContext):
            def mean_power_linear(self) -> float:
                raise RuntimeError("Mean linear computation error")

        with pytest.raises(PowerFeatureError, match="Failed to obtain mean_power_linear"):
            extractor.extract(BrokenMeanLinearContext(np.array([1.0, 2.0])))

        class BrokenMeanDbfsContext(DummyPowerContext):
            def mean_power_dbfs(self) -> float:
                raise RuntimeError("Mean dBFS computation error")

        with pytest.raises(PowerFeatureError, match="Failed to obtain mean_power_dbfs"):
            extractor.extract(BrokenMeanDbfsContext(np.array([1.0, 2.0])))

    def test_minimal_power_only_context(self) -> None:
        """Verify extractor calculates mean linear and mean dBFS directly when context lacks helper methods."""
        class MinimalPowerContext:
            def power(self) -> np.ndarray:
                return np.array([2.0, 8.0], dtype=np.float64)

        extractor = PowerFeatureExtractor()
        fset = extractor.extract(MinimalPowerContext())

        assert fset.features["power.mean_linear"].value == 5.0
        assert pytest.approx(fset.features["power.mean_dbfs"].value) == 10.0 * math.log10(5.0)

    def test_context_with_invalid_config_floor(self) -> None:
        """Verify extractor falls back to default -300.0 dBFS floor when config floor is invalid."""
        class BadConfig:
            power_floor_dbfs = "invalid_floor"

        class ContextWithBadConfig:
            config = BadConfig()

            def power(self) -> np.ndarray:
                return np.array([0.0, 0.0], dtype=np.float64)

            def mean_power_linear(self) -> float:
                return 0.0

            def mean_power_dbfs(self) -> float:
                return -300.0

        extractor = PowerFeatureExtractor()
        fset = extractor.extract(ContextWithBadConfig())

        assert fset.features["power.peak_dbfs"].value == -300.0

    def test_array_conversion_failure(self) -> None:
        """Verify error handling when power() returns an object that fails array conversion."""
        class Unconvertible:
            def __array__(self) -> None:
                raise ValueError("Conversion impossible")

        class UnconvertibleContext:
            def power(self) -> Unconvertible:
                return Unconvertible()

        extractor = PowerFeatureExtractor()
        with pytest.raises(PowerFeatureError, match="Cannot convert power to numpy array"):
            extractor.extract(UnconvertibleContext())

    def test_negative_mean_linear_error(self) -> None:
        """Verify error handling when context mean_power_linear produces negative power."""
        class MockNegativeMeanContext:
            def power(self) -> np.ndarray:
                return np.array([1.0, 2.0], dtype=np.float64)

            def mean_power_linear(self) -> float:
                return -1.0

        extractor = PowerFeatureExtractor()
        with pytest.raises(PowerFeatureError, match="Failed to convert mean power to dBFS"):
            extractor.extract(MockNegativeMeanContext())

    def test_peak_dbfs_conversion_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify error handling when peak power conversion to dBFS fails."""
        import signal_processing.features.power.extractor as ext_mod

        def broken_linear_to_dbfs(*args: object, **kwargs: object) -> float:
            raise RuntimeError("DSP dBFS library failure")

        monkeypatch.setattr(ext_mod, "linear_power_to_dbfs", broken_linear_to_dbfs)

        extractor = PowerFeatureExtractor()
        with pytest.raises(PowerFeatureError, match="Failed to convert (peak|mean) power to dBFS"):
            extractor.extract(DummyPowerContext(np.array([1.0, 2.0])))



