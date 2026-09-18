from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Type
import numpy as np


@dataclass
class PulseCandidate:
    """
    Represents a segmented Pulse Candidate extracted from conditioned IQ data.
    Separates short transient/pulsed activity from continuous or slowly varying signals.
    """
    candidate_id: int
    start_idx: int                       # Refined leading edge sample index (n_lead) in local buffer
    stop_idx: int                        # Refined trailing edge sample index (n_trail) in local buffer
    toa_sec: float                       # Time of Arrival in seconds = n_lead_absolute / Fs
    pulse_width_sec: float               # Pulse width in seconds = (n_trail - n_lead + 1) / Fs
    pulse_width_samples: int             # Pulse width in samples = (n_trail - n_lead + 1)
    peak_envelope: float                 # Peak envelope magnitude/power within the pulse
    mean_power: float                    # Mean physical power measured directly from complex IQ: E{|x|^2}
    snr_db: float                        # Estimated SNR (dB) = 10*log10((P_measured - N) / N)
    noise_floor_linear: float            # Local background noise floor level N[n]
    noise_floor_db: float                # Local background noise floor level (dB)
    rise_slope: float                    # Leading edge slope dP/dt at n_lead (W/s)
    fall_slope: float                    # Trailing edge slope dP/dt at n_trail (W/s)
    iq_samples: np.ndarray               # Segmented original complex IQ array x[n_lead:n_trail+1]
    start_idx_absolute: int | None = None  # Global absolute sample index in acquisition stream
    stop_idx_absolute: int | None = None   # Global absolute sample index in acquisition stream
    is_valid_pulse: bool = True          # True if passed pulse-shape gating (PW, slope, background)
    rejection_reason: str = ""           # Reason if rejected (e.g. 'PW_TOO_LONG', 'PW_TOO_SHORT', 'INVALID_EDGE_ORDER')

    def as_dict(self) -> dict[str, Any]:
        """Serializes candidate metrics into a clean dictionary."""
        abs_start = self.start_idx if self.start_idx_absolute is None else self.start_idx_absolute
        abs_stop = self.stop_idx if self.stop_idx_absolute is None else self.stop_idx_absolute
        return {
            "candidate_id": int(self.candidate_id),
            "start_idx": int(self.start_idx),
            "stop_idx": int(self.stop_idx),
            "start_idx_absolute": int(abs_start),
            "stop_idx_absolute": int(abs_stop),
            "toa_sec": float(self.toa_sec),
            "pulse_width_sec": float(self.pulse_width_sec),
            "pulse_width_samples": int(self.pulse_width_samples),
            "peak_envelope": float(self.peak_envelope),
            "mean_power": float(self.mean_power),
            "snr_db": float(self.snr_db),
            "noise_floor_linear": float(self.noise_floor_linear),
            "noise_floor_db": float(self.noise_floor_db),
            "rise_slope": float(self.rise_slope),
            "fall_slope": float(self.fall_slope),
            "num_iq_samples": int(self.iq_samples.size),
            "is_valid_pulse": bool(self.is_valid_pulse),
            "rejection_reason": str(self.rejection_reason),
        }


class BasePulseDetectorStage(ABC):
    """
    Abstract Base Class for all Pulse Detector processing stages.
    Enforces enterprise self-contained classes, factory registration, and clean compute interfaces.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Constructor: Initializes stage parameters and active state."""
        self._config: dict[str, Any] = dict(kwargs)
        self._is_active: bool = True

    def __del__(self) -> None:
        """Destructor: Releases resources and clears configuration."""
        self._is_active = False
        if hasattr(self, "_config") and isinstance(self._config, dict):
            self._config.clear()

    @property
    def is_active(self) -> bool:
        """Indicates whether this detector stage is active and ready."""
        return getattr(self, "_is_active", True)

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name for this pulse detection stage."""
        pass

    @property
    def description(self) -> str:
        """Description of the algorithmic stage."""
        return self.__doc__ or ""

    @abstractmethod
    def compute(self, data: Any, **kwargs: Any) -> Any:
        """
        Executes the processing stage on the input data.

        Parameters:
            data: Input IQ array, envelope series, binary mask, or preliminary boundary list.
            **kwargs: Stage parameters (e.g. sample_rate, window_len, k_high, k_low, pw_min, pw_max).

        Returns:
            Computed envelope array, threshold context, binary mask, refined edges, or pulse candidates.
        """
        pass

    def __call__(self, data: Any, **kwargs: Any) -> Any:
        """Enables stage instances to be invoked directly: stage(data)."""
        return self.compute(data, **kwargs)

    def reset(self) -> None:
        """Resets any internal state, filter memory, or streaming buffers."""
        pass


class PulseDetectorFactory:
    """
    Enterprise Factory Pattern for Pulse Detector Stages.
    Maintains a decoupled registry of stage implementations and handles lifecycle creation.
    """

    _registry: dict[str, Type[BasePulseDetectorStage]] = {}

    @classmethod
    def register(cls, aliases: str | list[str]) -> Callable[[Type[BasePulseDetectorStage]], Type[BasePulseDetectorStage]]:
        """Decorator to register a concrete BasePulseDetectorStage subclass under one or more aliases."""
        if isinstance(aliases, str):
            aliases = [aliases]

        def decorator(subclass: Type[BasePulseDetectorStage]) -> Type[BasePulseDetectorStage]:
            for alias in aliases:
                normalized = alias.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
                cls._registry[normalized] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, stage_name: str, **kwargs: Any) -> BasePulseDetectorStage:
        """Factory creation method: Instantiates and returns requested stage implementation."""
        normalized = stage_name.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        if normalized not in cls._registry:
            valid_keys = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown pulse detector stage '{stage_name}'. Registered stages: [{valid_keys}]")
        return cls._registry[normalized](**kwargs)

    @classmethod
    def list_stages(cls) -> list[str]:
        """Returns sorted list of all unique registered pulse detector stage names."""
        return sorted(list(set(cls._registry.keys())))
