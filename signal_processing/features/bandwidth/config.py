"""Bandwidth feature extraction configuration."""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from signal_processing.features.bandwidth.exceptions import BandwidthFeatureError


@dataclass(frozen=True)
class BandwidthFeatureConfig:
    """Immutable configuration for bandwidth feature extraction.

    Attributes:
        roi_hz: Optional baseband frequency Region of Interest (f_min, f_max) in Hz.
        noise_floor_dbfs_per_hz: Optional known noise floor power spectral density in dBFS/Hz.
        dominant_lobe_min_prominence_db: Minimum peak-to-median ratio in dB required to
            declare a valid dominant lobe for 3 dB half-power measurement (default: 3.0 dB).
        auto_noise_estimation: Whether to estimate noise floor from bins outside roi_hz
            when noise_floor_dbfs_per_hz is not explicitly specified (default: True).
        interpolate_edges: Whether to use linear interpolation on cumulative energy percentiles.
        mode: Extraction interpretation mode: "emitter" (verified excess SNR, edge clipping checks,
            and multimodal ambiguity rejection) or "raw_psd" (direct mathematical spectrum descriptor).
        min_snr_db: Minimum required excess SNR in ROI for emitter bandwidth reliability (default: 3.0 dB).
        check_edge_clipping: Whether to verify that the emitter signal is not clipped by ROI boundaries.
        multimodal_ratio_threshold: Prominence ratio threshold to detect ambiguous multi-lobe spectra.
    """

    roi_hz: tuple[float, float] | None = None
    noise_floor_dbfs_per_hz: float | None = None
    dominant_lobe_min_prominence_db: float = 3.0
    auto_noise_estimation: bool = True
    interpolate_edges: bool = False
    mode: Literal["emitter", "raw_psd"] = "raw_psd"
    min_snr_db: float = 3.0
    check_edge_clipping: bool = True
    multimodal_ratio_threshold: float = 0.7

    def __post_init__(self) -> None:
        if self.roi_hz is not None:
            if not isinstance(self.roi_hz, (tuple, list)) or len(self.roi_hz) != 2:
                raise BandwidthFeatureError(
                    f"roi_hz must be a 2-tuple (f_min, f_max) or None, got {self.roi_hz!r}"
                )
            f_min, f_max = float(self.roi_hz[0]), float(self.roi_hz[1])
            if not (math.isfinite(f_min) and math.isfinite(f_max) and f_min < f_max):
                raise BandwidthFeatureError(
                    f"roi_hz requires finite f_min < f_max, got ({f_min}, {f_max})"
                )
            object.__setattr__(self, "roi_hz", (f_min, f_max))

        if self.noise_floor_dbfs_per_hz is not None:
            if (
                isinstance(self.noise_floor_dbfs_per_hz, bool)
                or not isinstance(self.noise_floor_dbfs_per_hz, (int, float, np.floating, np.integer))
                or not math.isfinite(float(self.noise_floor_dbfs_per_hz))
            ):
                raise BandwidthFeatureError(
                    f"noise_floor_dbfs_per_hz must be a finite float or None, got {self.noise_floor_dbfs_per_hz!r}"
                )
            object.__setattr__(
                self, "noise_floor_dbfs_per_hz", float(self.noise_floor_dbfs_per_hz)
            )

        if (
            isinstance(self.dominant_lobe_min_prominence_db, bool)
            or not isinstance(self.dominant_lobe_min_prominence_db, (int, float, np.floating, np.integer))
            or self.dominant_lobe_min_prominence_db < 0.0
            or not math.isfinite(float(self.dominant_lobe_min_prominence_db))
        ):
            raise BandwidthFeatureError(
                f"dominant_lobe_min_prominence_db must be a non-negative finite float, got {self.dominant_lobe_min_prominence_db!r}"
            )
        object.__setattr__(
            self, "dominant_lobe_min_prominence_db", float(self.dominant_lobe_min_prominence_db)
        )

        if not isinstance(self.auto_noise_estimation, bool):
            raise BandwidthFeatureError(
                f"auto_noise_estimation must be a boolean, got {type(self.auto_noise_estimation).__name__}"
            )

        if not isinstance(self.interpolate_edges, bool):
            raise BandwidthFeatureError(
                f"interpolate_edges must be a boolean, got {type(self.interpolate_edges).__name__}"
            )

        if self.mode not in ("emitter", "raw_psd"):
            raise BandwidthFeatureError(
                f"mode must be 'emitter' or 'raw_psd', got {self.mode!r}"
            )

        if (
            isinstance(self.min_snr_db, bool)
            or not isinstance(self.min_snr_db, (int, float, np.floating, np.integer))
            or not math.isfinite(float(self.min_snr_db))
            or self.min_snr_db < 0.0
        ):
            raise BandwidthFeatureError(
                f"min_snr_db must be a non-negative finite float, got {self.min_snr_db!r}"
            )
        object.__setattr__(self, "min_snr_db", float(self.min_snr_db))

        if not isinstance(self.check_edge_clipping, bool):
            raise BandwidthFeatureError(
                f"check_edge_clipping must be a boolean, got {type(self.check_edge_clipping).__name__}"
            )

        if (
            isinstance(self.multimodal_ratio_threshold, bool)
            or not isinstance(self.multimodal_ratio_threshold, (int, float, np.floating, np.integer))
            or not math.isfinite(float(self.multimodal_ratio_threshold))
            or not (0.0 < float(self.multimodal_ratio_threshold) <= 1.0)
        ):
            raise BandwidthFeatureError(
                f"multimodal_ratio_threshold must be a float in (0.0, 1.0], got {self.multimodal_ratio_threshold!r}"
            )
        object.__setattr__(self, "multimodal_ratio_threshold", float(self.multimodal_ratio_threshold))
