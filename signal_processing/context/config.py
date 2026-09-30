"""Analysis configuration for signal context transforms."""

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from signal_processing.context.exceptions import ContextError


@dataclass(frozen=True)
class AnalysisConfig:
    """Immutable configuration for signal context DSP transforms.

    Attributes:
        nfft: Optional FFT size. If None, defaults to len(signal_iq).
            If specified, must be a strictly positive integer >= len(signal_iq).
        spectral_window: Window name for Fourier analysis ("rectangular", "hann", "hamming", "blackman").
            Default is "hann".
        centered_spectrum: Whether zero frequency is centered in FFT/PSD output.
            Default is True.
        power_floor_dbfs: Logarithmic floor in dBFS applied during power conversions.
            Default is -300.0 dBFS.
        psd_floor_dbfs_per_hz: Logarithmic floor in dBFS/Hz applied during PSD calculations.
            Default is -300.0 dBFS/Hz.
        autocorrelation_max_lag_samples: Optional non-negative maximum lag sample index (0 <= lag < len(signal_iq)).
            If None, defaults to len(signal_iq) - 1.
        autocorrelation_normalized: Whether autocorrelation at lag 0 is normalized to 1.0 + 0j.
            Default is True.
    """

    SUPPORTED_WINDOWS: ClassVar[tuple[str, ...]] = ("rectangular", "hann", "hamming", "blackman")
    SUPPORTED_PSD_MODES: ClassVar[tuple[str, ...]] = ("periodogram", "welch")

    nfft: int | None = None
    spectral_window: str = "hann"
    centered_spectrum: bool = True
    power_floor_dbfs: float = -300.0
    psd_floor_dbfs_per_hz: float = -300.0
    autocorrelation_max_lag_samples: int | None = None
    autocorrelation_normalized: bool = True
    psd_mode: str = "periodogram"
    psd_nperseg: int | None = None
    psd_noverlap: int | None = None
    psd_welch_window: str = "hann"
    phase_amplitude_threshold: float = 0.0
    bandwidth_roi_hz: tuple[float, float] | None = None
    bandwidth_noise_floor_dbfs_per_hz: float | None = None
    bandwidth_dominant_lobe_min_prominence_db: float = 3.0
    bandwidth_mode: str = "raw_psd"
    bandwidth_interpolate_edges: bool = False
    bandwidth_auto_noise_estimation: bool = True
    bandwidth_min_snr_db: float = 3.0
    bandwidth_check_edge_clipping: bool = True
    bandwidth_multimodal_ratio_threshold: float = 0.7
    bandwidth_occupied_power_scope: str = "observed_excess"
    bandwidth_complete_observation: bool = False
    bandwidth_support_provenance: str | None = None

    def __post_init__(self) -> None:
        # Validate nfft
        if self.nfft is not None:
            if isinstance(self.nfft, bool) or not isinstance(self.nfft, (int, np.integer)) or self.nfft <= 0:
                raise ContextError(
                    f"nfft must be a strictly positive integer or None, got {self.nfft!r}"
                )
            object.__setattr__(self, "nfft", int(self.nfft))

        # Validate spectral_window
        if not isinstance(self.spectral_window, str) or not self.spectral_window.strip():
            raise ContextError(
                f"spectral_window must be a non-empty string, got {self.spectral_window!r}"
            )
        norm_win = self.spectral_window.strip().lower()
        if norm_win not in self.SUPPORTED_WINDOWS:
            raise ContextError(
                f"Unsupported spectral_window '{self.spectral_window}'. "
                f"Supported windows: {', '.join(self.SUPPORTED_WINDOWS)}"
            )
        object.__setattr__(self, "spectral_window", norm_win)

        # Validate centered_spectrum
        if not isinstance(self.centered_spectrum, bool):
            raise ContextError(
                f"centered_spectrum must be a boolean, got {type(self.centered_spectrum).__name__}"
            )

        # Validate power_floor_dbfs
        if isinstance(self.power_floor_dbfs, bool) or not isinstance(
            self.power_floor_dbfs, (int, float, np.floating, np.integer)
        ):
            raise ContextError(
                f"power_floor_dbfs must be a numeric float, got {self.power_floor_dbfs!r}"
            )
        p_floor = float(self.power_floor_dbfs)
        if not math.isfinite(p_floor):
            raise ContextError(f"power_floor_dbfs must be finite, got {self.power_floor_dbfs}")
        object.__setattr__(self, "power_floor_dbfs", p_floor)

        # Validate psd_floor_dbfs_per_hz
        if isinstance(self.psd_floor_dbfs_per_hz, bool) or not isinstance(
            self.psd_floor_dbfs_per_hz, (int, float, np.floating, np.integer)
        ):
            raise ContextError(
                f"psd_floor_dbfs_per_hz must be a numeric float, got {self.psd_floor_dbfs_per_hz!r}"
            )
        psd_floor = float(self.psd_floor_dbfs_per_hz)
        if not math.isfinite(psd_floor):
            raise ContextError(f"psd_floor_dbfs_per_hz must be finite, got {self.psd_floor_dbfs_per_hz}")
        object.__setattr__(self, "psd_floor_dbfs_per_hz", psd_floor)

        # Validate autocorrelation_max_lag_samples
        if self.autocorrelation_max_lag_samples is not None:
            if (
                isinstance(self.autocorrelation_max_lag_samples, bool)
                or not isinstance(self.autocorrelation_max_lag_samples, (int, np.integer))
                or self.autocorrelation_max_lag_samples < 0
            ):
                raise ContextError(
                    f"autocorrelation_max_lag_samples must be a non-negative integer or None, "
                    f"got {self.autocorrelation_max_lag_samples!r}"
                )
            object.__setattr__(
                self, "autocorrelation_max_lag_samples", int(self.autocorrelation_max_lag_samples)
            )

        # Validate autocorrelation_normalized
        if not isinstance(self.autocorrelation_normalized, bool):
            raise ContextError(
                f"autocorrelation_normalized must be a boolean, got {type(self.autocorrelation_normalized).__name__}"
            )

        # Validate psd_mode
        if not isinstance(self.psd_mode, str) or self.psd_mode.lower() not in self.SUPPORTED_PSD_MODES:
            raise ContextError(
                f"psd_mode must be one of {self.SUPPORTED_PSD_MODES}, got {self.psd_mode!r}"
            )
        object.__setattr__(self, "psd_mode", self.psd_mode.lower())

        # Validate psd_nperseg
        if self.psd_nperseg is not None:
            if (
                isinstance(self.psd_nperseg, bool)
                or not isinstance(self.psd_nperseg, (int, np.integer))
                or self.psd_nperseg <= 0
            ):
                raise ContextError(f"psd_nperseg must be a strictly positive integer, got {self.psd_nperseg!r}")
            object.__setattr__(self, "psd_nperseg", int(self.psd_nperseg))

        # Validate psd_noverlap
        if self.psd_noverlap is not None:
            if (
                isinstance(self.psd_noverlap, bool)
                or not isinstance(self.psd_noverlap, (int, np.integer))
                or self.psd_noverlap < 0
            ):
                raise ContextError(f"psd_noverlap must be a non-negative integer, got {self.psd_noverlap!r}")
            if self.psd_nperseg is not None and int(self.psd_noverlap) >= int(self.psd_nperseg):
                raise ContextError(
                    f"psd_noverlap ({self.psd_noverlap}) must be strictly less than psd_nperseg ({self.psd_nperseg})"
                )
            object.__setattr__(self, "psd_noverlap", int(self.psd_noverlap))

        # Validate psd_welch_window
        if not isinstance(self.psd_welch_window, str) or self.psd_welch_window.lower() not in self.SUPPORTED_WINDOWS:
            raise ContextError(
                f"psd_welch_window must be one of {self.SUPPORTED_WINDOWS}, got {self.psd_welch_window!r}"
            )
        object.__setattr__(self, "psd_welch_window", self.psd_welch_window.lower())

        # Validate phase_amplitude_threshold
        if (
            isinstance(self.phase_amplitude_threshold, bool)
            or not isinstance(self.phase_amplitude_threshold, (int, float, np.floating, np.integer))
            or self.phase_amplitude_threshold < 0.0
            or not math.isfinite(float(self.phase_amplitude_threshold))
        ):
            raise ContextError(
                f"phase_amplitude_threshold must be a non-negative finite float, got {self.phase_amplitude_threshold!r}"
            )
        object.__setattr__(self, "phase_amplitude_threshold", float(self.phase_amplitude_threshold))

        # Validate bandwidth_roi_hz
        if self.bandwidth_roi_hz is not None:
            if not isinstance(self.bandwidth_roi_hz, (tuple, list)) or len(self.bandwidth_roi_hz) != 2:
                raise ContextError(
                    f"bandwidth_roi_hz must be a 2-tuple of (f_min, f_max) or None, got {self.bandwidth_roi_hz!r}"
                )
            f_min, f_max = float(self.bandwidth_roi_hz[0]), float(self.bandwidth_roi_hz[1])
            if not (math.isfinite(f_min) and math.isfinite(f_max) and f_min < f_max):
                raise ContextError(
                    f"bandwidth_roi_hz requires finite f_min < f_max, got ({f_min}, {f_max})"
                )
            object.__setattr__(self, "bandwidth_roi_hz", (f_min, f_max))

        # Validate bandwidth_noise_floor_dbfs_per_hz
        if self.bandwidth_noise_floor_dbfs_per_hz is not None:
            if (
                isinstance(self.bandwidth_noise_floor_dbfs_per_hz, bool)
                or not isinstance(self.bandwidth_noise_floor_dbfs_per_hz, (int, float, np.floating, np.integer))
                or not math.isfinite(float(self.bandwidth_noise_floor_dbfs_per_hz))
            ):
                raise ContextError(
                    f"bandwidth_noise_floor_dbfs_per_hz must be a finite float or None, got {self.bandwidth_noise_floor_dbfs_per_hz!r}"
                )
            object.__setattr__(
                self, "bandwidth_noise_floor_dbfs_per_hz", float(self.bandwidth_noise_floor_dbfs_per_hz)
            )

        # Validate bandwidth_dominant_lobe_min_prominence_db
        if (
            isinstance(self.bandwidth_dominant_lobe_min_prominence_db, bool)
            or not isinstance(self.bandwidth_dominant_lobe_min_prominence_db, (int, float, np.floating, np.integer))
            or self.bandwidth_dominant_lobe_min_prominence_db < 0.0
            or not math.isfinite(float(self.bandwidth_dominant_lobe_min_prominence_db))
        ):
            raise ContextError(
                f"bandwidth_dominant_lobe_min_prominence_db must be a non-negative finite float, got {self.bandwidth_dominant_lobe_min_prominence_db!r}"
            )
        object.__setattr__(
            self, "bandwidth_dominant_lobe_min_prominence_db", float(self.bandwidth_dominant_lobe_min_prominence_db)
        )

        # Validate bandwidth_mode
        if self.bandwidth_mode not in ("emitter", "raw_psd"):
            raise ContextError(
                f"bandwidth_mode must be 'emitter' or 'raw_psd', got {self.bandwidth_mode!r}"
            )

        # Validate bandwidth_interpolate_edges
        if not isinstance(self.bandwidth_interpolate_edges, bool):
            raise ContextError(
                f"bandwidth_interpolate_edges must be a boolean, got {type(self.bandwidth_interpolate_edges).__name__}"
            )
