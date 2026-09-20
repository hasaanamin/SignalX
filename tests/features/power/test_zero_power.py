"""Tests for zero-power signal handling and invalid PAPR semantics."""

import math

import numpy as np

from signal_processing.features.power import PowerFeatureExtractor


class DummyPowerContext:
    def __init__(self, power: np.ndarray, floor_dbfs: float = -300.0) -> None:
        self._power = power
        self._floor_dbfs = floor_dbfs

    def power(self) -> np.ndarray:
        return self._power

    def mean_power_linear(self) -> float:
        return float(np.mean(self._power))

    def mean_power_dbfs(self) -> float:
        mean_p = self.mean_power_linear()
        if mean_p <= 0.0:
            return self._floor_dbfs
        return float(10.0 * math.log10(mean_p))


class TestZeroPower:
    """Test extraction on zero-power signals without throwing exceptions."""

    def test_all_zero_power_vector(self) -> None:
        """All-zero signal must produce 0.0 / floor metrics and invalid FeatureValues for PAPR."""
        p = np.zeros(50, dtype=np.float64)
        ctx = DummyPowerContext(p)
        extractor = PowerFeatureExtractor()

        fset = extractor.extract(ctx)

        assert fset.features["power.mean_linear"].value == 0.0
        assert fset.features["power.mean_linear"].valid is True
        assert fset.features["power.mean_linear"].unit == "linear"

        assert fset.features["power.mean_dbfs"].value == -300.0
        assert fset.features["power.mean_dbfs"].valid is True
        assert fset.features["power.mean_dbfs"].unit == "dBFS"

        assert fset.features["power.peak_linear"].value == 0.0
        assert fset.features["power.peak_linear"].valid is True
        assert fset.features["power.peak_linear"].unit == "linear"

        assert fset.features["power.peak_dbfs"].value == -300.0
        assert fset.features["power.peak_dbfs"].valid is True
        assert fset.features["power.peak_dbfs"].unit == "dBFS"

        assert fset.features["power.min_linear"].value == 0.0
        assert fset.features["power.min_linear"].valid is True
        assert fset.features["power.min_linear"].unit == "linear"

        assert fset.features["power.std_linear"].value == 0.0
        assert fset.features["power.std_linear"].valid is True
        assert fset.features["power.std_linear"].unit == "linear"

        assert fset.features["power.variance_linear"].value == 0.0
        assert fset.features["power.variance_linear"].valid is True
        assert fset.features["power.variance_linear"].unit == "linear^2"

        assert fset.features["power.median_linear"].value == 0.0
        assert fset.features["power.median_linear"].valid is True
        assert fset.features["power.median_linear"].unit == "linear"

        # PAPR linear: invalid with explanatory reason and NaN value
        papr_lin = fset.features["power.papr_linear"]
        assert papr_lin.valid is False
        assert math.isnan(papr_lin.value)
        assert papr_lin.unit is None
        assert papr_lin.reason == "Mean power is zero"

        # PAPR dB: invalid with explanatory reason and NaN value
        papr_db = fset.features["power.papr_db"]
        assert papr_db.valid is False
        assert math.isnan(papr_db.value)
        assert papr_db.unit == "dB"
        assert papr_db.reason == "Mean power is zero"
