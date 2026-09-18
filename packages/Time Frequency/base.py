from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Type
import numpy as np


@dataclass
class TimeFrequencyResult:
    """
    Standard container for Short-Time Fourier Transform (STFT) and Spectrogram results.
    Shared across all downstream STFT-derived metrics to prevent duplicate computations.
    """
    stft_complex: np.ndarray  # 2D array [K, M_frames], complex128
    power_spectrogram: np.ndarray  # 2D array [K, M_frames], float64 (|X|^2)
    spectrogram_db: np.ndarray  # 2D array [K, M_frames], float64 (10*log10(P + eps))
    frequencies_hz: np.ndarray  # 1D array [K], centered [-Fs/2, Fs/2)
    times_sec: np.ndarray  # 1D array [M_frames], window center timestamps
    sample_rate: float  # Fs in Hz
    hop_samples: int  # Hop size R = M - L
    window_length: int  # Window length M
    nfft: int  # FFT length N_fft
    window_name: str = "hann"
    sample_offset: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Converts result container to a canonical dictionary representation."""
        return {
            "stft_complex": self.stft_complex,
            "power_spectrogram": self.power_spectrogram,
            "spectrogram_db": self.spectrogram_db,
            "frequencies_hz": self.frequencies_hz,
            "times_sec": self.times_sec,
            "sample_rate": self.sample_rate,
            "hop_samples": self.hop_samples,
            "window_length": self.window_length,
            "nfft": self.nfft,
            "window_name": self.window_name,
            "sample_offset": self.sample_offset,
            **self.extra
        }


@dataclass
class WaveletResult:
    """
    Standard container for Continuous Wavelet Transform (CWT) results.
    Shared across all downstream CWT-derived metrics to prevent duplicate computations.
    """
    coefficients: np.ndarray  # 2D array [S, N_samples], complex128 (W[s, n])
    frequencies_hz: np.ndarray  # 1D array [S], pseudo-frequencies in Hz
    scales: np.ndarray  # 1D array [S], wavelet scales
    coefficient_energy: np.ndarray  # 2D array [S, N_samples], float64 (|W[s, n]|^2)
    times_sec: np.ndarray  # 1D array [N_samples], timestamps in seconds
    sample_rate: float  # Fs in Hz
    wavelet_name: str = "morse"
    voices_per_octave: int = 10
    gamma: float = 3.0
    time_bandwidth: float = 60.0
    normalization: str = "L1"
    frequency_mode: str = "analytic_positive"
    sample_offset: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def energy_per_scale(self) -> np.ndarray:
        """Energy versus scale: E_s[s] = sum_n |W[s, n]|^2."""
        return np.sum(self.coefficient_energy, axis=1)

    @property
    def energy_vs_time(self) -> np.ndarray:
        """Energy versus time: E_t[n] = sum_s |W[s, n]|^2."""
        return np.sum(self.coefficient_energy, axis=0)

    @property
    def total_energy(self) -> float:
        """Total coefficient energy: E_CWT = sum_s sum_n |W[s, n]|^2."""
        return float(np.sum(self.coefficient_energy))

    def to_dict(self) -> dict[str, Any]:
        """Converts result container to a canonical dictionary representation."""
        return {
            "coefficients": self.coefficients,
            "frequencies_hz": self.frequencies_hz,
            "scales": self.scales,
            "coefficient_energy": self.coefficient_energy,
            "times_sec": self.times_sec,
            "energy_per_scale": self.energy_per_scale,
            "energy_vs_time": self.energy_vs_time,
            "total_energy": self.total_energy,
            "sample_rate": self.sample_rate,
            "wavelet_name": self.wavelet_name,
            "voices_per_octave": self.voices_per_octave,
            "gamma": self.gamma,
            "time_bandwidth": self.time_bandwidth,
            "normalization": self.normalization,
            "frequency_mode": self.frequency_mode,
            "sample_offset": self.sample_offset,
            **self.extra
        }



class BaseTimeFrequencyMetric(ABC):
    """
    Abstract Base Class for all Time-Frequency Analysis Metrics and Transform Stages.
    Defines the standard lifecycle interface for STFT, spectrogram, frequency trajectory,
    instantaneous bandwidth, TF centroid, TF entropy, CWT, wavelet coefficients, and wavelet energy.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Constructor: Initializes metric configuration and lifecycle state."""
        self._config: dict[str, Any] = dict(kwargs)
        self._is_active: bool = True

    def __del__(self) -> None:
        """Destructor: Releases resources and clears internal configuration."""
        self._is_active = False
        if hasattr(self, "_config") and isinstance(self._config, dict):
            self._config.clear()

    @property
    def is_active(self) -> bool:
        """Indicates if the metric instance is active and ready for computation."""
        return getattr(self, "_is_active", True)

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name for this time-frequency metric or stage."""
        pass

    @property
    def description(self) -> str:
        """Human-readable description of what this metric measures."""
        return self.__doc__ or ""

    @abstractmethod
    def compute(self, data: Any, **kwargs: Any) -> Any:
        """
        Executes the time-frequency measurement or transform on input data.

        Parameters:
            data: Complex IQ array, canonical dictionary, tuple of (I, Q),
                  or precomputed TimeFrequencyResult / WaveletResult.
            **kwargs: Additional computation parameters (e.g. sample_rate, window_length, overlap).

        Returns:
            Computed numerical scalar, 1D array, 2D matrix, TimeFrequencyResult, WaveletResult, or dictionary.
        """
        pass

    def __call__(self, data: Any, **kwargs: Any) -> Any:
        """Allows metric instances to be called directly as functions: metric(iq_data)."""
        return self.compute(data, **kwargs)

    def reset(self) -> None:
        """Resets any internal accumulator or streaming state."""
        pass


class TimeFrequencyMetricFactory:
    """
    Enterprise Factory Pattern for Time-Frequency Analysis Metrics and Stages.
    Maintains a decoupled registry of metric classes and handles lifecycle creation.
    """

    _registry: dict[str, Type[BaseTimeFrequencyMetric]] = {}

    @classmethod
    def register(cls, aliases: str | list[str]) -> Callable[[Type[BaseTimeFrequencyMetric]], Type[BaseTimeFrequencyMetric]]:
        """Decorator to register a concrete BaseTimeFrequencyMetric subclass under one or more alias names."""
        if isinstance(aliases, str):
            aliases = [aliases]

        def decorator(subclass: Type[BaseTimeFrequencyMetric]) -> Type[BaseTimeFrequencyMetric]:
            for alias in aliases:
                normalized = alias.strip().lower().replace(" ", "_").replace("-", "_")
                cls._registry[normalized] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, metric_name: str, **kwargs: Any) -> BaseTimeFrequencyMetric:
        """Factory creation method: Instantiates and returns the requested metric implementation."""
        normalized = metric_name.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized not in cls._registry:
            valid_keys = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown time-frequency metric '{metric_name}'. Registered metrics: [{valid_keys}]")
        return cls._registry[normalized](**kwargs)

    @classmethod
    def list_metrics(cls) -> list[str]:
        """Returns a sorted list of all unique registered time-frequency metric names."""
        return sorted(list(set(cls._registry.keys())))
