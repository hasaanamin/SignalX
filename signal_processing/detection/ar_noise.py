"""Bounded, explicitly authorized complex AR(1) calibration.

Diagnostic acceptance is not rare-tail or independence certification. Runtime
receives the fitted model, never generator rho, ideal noise power or burst truth.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AR1NoiseModel:
    coefficient: complex
    mean: complex
    innovation_power: float
    raw_noise_power: float
    sample_rate_hz: float
    training_samples: int
    residual_max_correlation: float
    residual_impropriety: float
    version: str = "ar1_split_fit_v1"

    def __post_init__(self):
        for name in ("coefficient", "mean"):
            z = complex(getattr(self, name))
            if not math.isfinite(z.real) or not math.isfinite(z.imag):
                raise ValueError(f"Nonfinite {name}")
            object.__setattr__(self, name, z)
        if abs(self.coefficient) >= .98:
            raise ValueError("AR1 candidate requires |coefficient| < 0.98")
        for name in ("innovation_power", "raw_noise_power", "sample_rate_hz"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if type(self.training_samples) is not int or not 1024 <= self.training_samples <= 65536:
            raise ValueError("AR1 calibration requires 1024–65536 authorized samples")
        for name in ("residual_max_correlation", "residual_impropriety"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= .08:
                raise ValueError(f"AR1 residual diagnostic failed: {name}")
        if self.version != "ar1_split_fit_v1":
            raise ValueError("Unsupported AR1 model semantics")

    def to_dict(self):
        return {"coefficient": [self.coefficient.real, self.coefficient.imag],
                "mean": [self.mean.real, self.mean.imag], "innovation_power": self.innovation_power,
                "raw_noise_power": self.raw_noise_power, "sample_rate_hz": self.sample_rate_hz,
                "training_samples": self.training_samples,
                "residual_max_correlation": self.residual_max_correlation,
                "residual_impropriety": self.residual_impropriety, "version": self.version}

    @property
    def model_id(self):
        data = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "ar1_" + hashlib.sha256(data.encode()).hexdigest()


def fit_ar1_noise(noise_iq: np.ndarray, sample_rate_hz: float) -> AR1NoiseModel:
    """Fit first half, assess/calibrate residuals on second half of authorized noise.

    The caller removes its declared settling prefix. No future/active suffix is
    read. Input storage is bounded explicitly to 65536 samples, not stream length.
    """
    if (not isinstance(noise_iq, np.ndarray) or noise_iq.ndim != 1
            or not np.iscomplexobj(noise_iq) or not 1024 <= len(noise_iq) <= 65536
            or not np.all(np.isfinite(noise_iq))):
        raise ValueError("Expected 1024–65536 finite authorized complex noise samples")
    x = noise_iq.astype(np.complex128)
    split = len(x)//2
    mu = complex(np.mean(x[:split]))
    u = x[:split] - mu
    power = float(np.vdot(u[:-1], u[:-1]).real)
    if power <= 0:
        raise ValueError("Zero training noise is not valid AR1 calibration")
    coefficient = complex(np.vdot(u[:-1], u[1:])/power)
    residual = (x[split:] - mu) - coefficient*(x[split-1:-1] - mu)
    variance = float(np.vdot(residual, residual).real/len(residual))
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("Invalid innovation power")
    max_corr = max(float(abs(np.vdot(residual[:-lag],residual[lag:])) /
                         ((len(residual)-lag)*variance)) for lag in range(1,9))
    impropriety = float(abs(np.mean(residual*residual))/variance)
    half = len(residual)//2
    p1 = float(np.mean(abs(residual[:half])**2))
    p2 = float(np.mean(abs(residual[half:])**2))
    if not .5 <= p1/p2 <= 2:
        raise ValueError("Nonstationary innovation variance; recalibration required")
    return AR1NoiseModel(coefficient, mu, variance, float(np.mean(abs(x)**2)),
                         sample_rate_hz, len(x), max_corr, impropriety)
