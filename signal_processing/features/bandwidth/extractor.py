"""Bandwidth feature extractor for RF signal analysis."""

import math
from dataclasses import replace
from typing import Any

import numpy as np

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, FeatureValue
from signal_processing.features.bandwidth.config import BandwidthFeatureConfig
from signal_processing.features.bandwidth.exceptions import BandwidthFeatureError
from signal_processing.transforms.exceptions import TransformError


class BandwidthFeatureExtractor:
    """Feature extractor for signal bandwidth characteristics.

    Extracts the following canonical scalar features from the signal Power Spectral Density (PSD):
        - bandwidth.occupied_99_hz: 99% cumulative equal-tail occupied bandwidth [Hz]
        - bandwidth.three_db_hz: Contiguous half-power (3 dB) bandwidth around the dominant peak [Hz]
    """

    name: str = "bandwidth"
    version: str = "2.0"

    def __init__(self, config: BandwidthFeatureConfig | None = None) -> None:
        """Initialize BandwidthFeatureExtractor.

        Args:
            config: Optional BandwidthFeatureConfig. If None, extractor resolves
                configuration dynamically from context.config when available.
        """
        if config is not None and not isinstance(config, BandwidthFeatureConfig):
            raise BandwidthFeatureError(
                f"Expected BandwidthFeatureConfig instance or None, got {type(config).__name__}"
            )
        self.config = config
        self._resolved_context_config = None

    def resolve_config(self, context: SignalContext | Any) -> BandwidthFeatureConfig:
        """The single effective configuration used by both extraction and hashing."""
        if self.config is not None:
            return self.config
        from dataclasses import fields
        context_config = getattr(context, "config", None)
        from signal_processing.context import AnalysisConfig
        cached = self._resolved_context_config
        if isinstance(context_config, AnalysisConfig) and cached is not None and cached[0] == context_config:
            return cached[1]
        defaults = BandwidthFeatureConfig()
        resolved = BandwidthFeatureConfig(**{
            f.name: getattr(context_config, "bandwidth_" + f.name, getattr(defaults, f.name))
            for f in fields(defaults)
        })
        if isinstance(context_config, AnalysisConfig):
            self._resolved_context_config = (context_config, resolved)
        return resolved

    def _make_invalid_featureset(self, reason: str) -> FeatureSet:
        """Create a FeatureSet with all features marked invalid."""
        features: dict[str, FeatureValue] = {
            "bandwidth.occupied_99_hz": FeatureValue(
                value=float("nan"),
                unit="Hz",
                valid=False,
                reason=reason,
            ),
            "bandwidth.three_db_hz": FeatureValue(
                value=float("nan"),
                unit="Hz",
                valid=False,
                reason=reason,
            ),
        }
        return FeatureSet(
            extractor=self.name,
            extractor_version=self.version,
            features=features,
        )

    def extract(self, context: SignalContext | Any) -> FeatureSet:
        """Extract bandwidth features from a SignalContext.

        Args:
            context: SignalContext instance (or duck-typed object exposing `psd()`).

        Returns:
            FeatureSet containing the 2 canonical bandwidth FeatureValues.

        Raises:
            BandwidthFeatureError: If context is None, missing `psd()`, or returns
                invalid, malformed, or non-uniformly spaced PSD data.
        """
        if context is None:
            raise BandwidthFeatureError("Context must not be None")

        if not hasattr(context, "psd") or not callable(context.psd):
            raise BandwidthFeatureError(
                f"Expected SignalContext instance or object with psd() method, got {type(context).__name__}"
            )

        # Obtain PSD, handling short event / transform boundary errors as expected insufficient-data conditions
        try:
            psd = context.psd()
        except TransformError as exc:
            return self._make_invalid_featureset(f"PSD unavailable: {exc}")
        except Exception as exc:
            raise BandwidthFeatureError(f"Failed to obtain PSD from context: {exc}") from exc

        if not hasattr(psd, "frequency_offset_hz") or not hasattr(psd, "power_linear"):
            raise BandwidthFeatureError(
                f"PSDResult must contain frequency_offset_hz and power_linear arrays, got {type(psd).__name__}"
            )

        try:
            f_arr = np.asarray(psd.frequency_offset_hz)
            p_arr = np.asarray(psd.power_linear)
        except Exception as exc:
            raise BandwidthFeatureError(f"Cannot convert PSD data to numpy arrays: {exc}") from exc

        if f_arr.ndim != 1 or p_arr.ndim != 1:
            raise BandwidthFeatureError(
                f"PSD arrays must be 1-dimensional, got shapes {f_arr.shape} and {p_arr.shape}"
            )

        if f_arr.size == 0 or p_arr.size == 0:
            raise BandwidthFeatureError("PSD arrays must not be empty")

        if f_arr.size != p_arr.size:
            raise BandwidthFeatureError(
                f"PSD array length mismatch: frequency_offset_hz has {f_arr.size} samples, "
                f"power_linear has {p_arr.size} samples"
            )

        if not (
            np.issubdtype(f_arr.dtype, np.floating) or np.issubdtype(f_arr.dtype, np.integer)
        ) or np.iscomplexobj(f_arr):
            raise BandwidthFeatureError(
                f"frequency_offset_hz must be real-valued numeric, got dtype {f_arr.dtype}"
            )

        if not (
            np.issubdtype(p_arr.dtype, np.floating) or np.issubdtype(p_arr.dtype, np.integer)
        ) or np.iscomplexobj(p_arr):
            raise BandwidthFeatureError(
                f"power_linear must be real-valued numeric, got dtype {p_arr.dtype}"
            )

        if not np.all(np.isfinite(f_arr)) or not np.all(np.isfinite(p_arr)):
            raise BandwidthFeatureError("PSD arrays must contain only finite values (no NaN or Inf)")

        if np.any(p_arr < 0.0):
            raise BandwidthFeatureError("power_linear must contain non-negative values (>= 0)")

        # Case: insufficient frequency resolution (< 2 bins)
        if f_arr.size < 2:
            return self._make_invalid_featureset("At least two PSD bins are required")

        # Sort frequency and power arrays monotonically by frequency
        order = np.argsort(f_arr)
        f = f_arr[order].astype(np.float64, copy=False)
        p = p_arr[order].astype(np.float64, copy=False)

        df_values = np.diff(f)
        if np.any(df_values <= 0.0):
            raise BandwidthFeatureError(
                "Frequency axis must contain strictly increasing distinct values"
            )

        df = float(np.median(df_values))
        if df <= 0.0 or not math.isfinite(df):
            raise BandwidthFeatureError("Frequency bin spacing must be positive and finite")

        max_dev = float(np.max(np.abs(df_values - df)))
        tol = 1e-4 * df + 1e-6
        if max_dev > tol:
            raise BandwidthFeatureError(
                f"Frequency axis must be uniformly spaced, found max deviation of {max_dev:.6g} Hz "
                f"from median spacing of {df:.6g} Hz"
            )

        cfg = self.resolve_config(context)
        mode, roi_hz = cfg.mode, cfg.roi_hz
        noise_floor_dbfs = cfg.noise_floor_dbfs_per_hz
        dominant_lobe_min_prominence_db = cfg.dominant_lobe_min_prominence_db
        auto_noise_estimation = cfg.auto_noise_estimation
        interpolate_edges = cfg.interpolate_edges
        min_snr_db = cfg.min_snr_db
        check_edge_clipping = cfg.check_edge_clipping
        multimodal_ratio_threshold = cfg.multimodal_ratio_threshold

        # Region of interest (ROI) selection
        if roi_hz is not None:
            f_min, f_max = float(roi_hz[0]), float(roi_hz[1])
            roi_mask = (f >= f_min) & (f <= f_max)
            num_roi_bins = int(np.count_nonzero(roi_mask))
            if num_roi_bins < 2:
                return self._make_invalid_featureset(
                    f"Configured ROI [{f_min}, {f_max}] contains {num_roi_bins} bins (at least 2 required)"
                )
        else:
            roi_mask = np.ones(len(f), dtype=bool)

        # Check total PSD power before anything else
        p_raw_total = float(np.sum(p))
        if p_raw_total <= 0.0 or not math.isfinite(p_raw_total):
            return self._make_invalid_featureset("PSD contains no positive signal power")

        # Noise subtraction in linear density units
        # In raw_psd mode: raw PSD is measured untouched (no noise subtraction).
        # In emitter mode: noise floor is subtracted to isolate emitter excess power.
        noise_linear = 0.0
        if mode == "emitter":
            if noise_floor_dbfs is not None:
                noise_linear = float(10.0 ** (noise_floor_dbfs / 10.0))
            elif auto_noise_estimation and roi_hz is not None:
                # Estimate noise from out-of-band bins if at least 8 bins exist outside ROI
                out_of_band = ~roi_mask
                if np.count_nonzero(out_of_band) >= 8:
                    noise_linear = float(np.median(p[out_of_band]))

        # In emitter mode, require noise reference and check excess power over noise reference
        if mode == "emitter":
            if noise_linear <= 0.0:
                return self._make_invalid_featureset("Noise reference unavailable for emitter bandwidth estimation")

            p_roi_raw = p[roi_mask]
            p_roi_total = float(np.sum(p_roi_raw))
            p_noise_total = float(len(p_roi_raw) * noise_linear)
            excess_power = p_roi_total - p_noise_total
            excess_snr_lin = excess_power / p_noise_total if p_noise_total > 0.0 else 0.0
            req_snr_lin = 10.0 ** (min_snr_db / 10.0)

            if excess_power <= 0.0 or excess_snr_lin < req_snr_lin:
                return self._make_invalid_featureset(
                    f"Insufficient excess SNR in ROI ({10.0 * math.log10(max(excess_snr_lin, 1e-15)):.2f} dB < {min_snr_db:.2f} dB)"
                )

        if noise_linear > 0.0:
            p_sub = np.maximum(p - noise_linear, 0.0)
        else:
            p_sub = p

        f_sig = f[roi_mask]
        p_sig = p_sub[roi_mask]

        p_total = float(np.sum(p_sig))
        p_max = float(np.max(p_sig))

        # Check tail containment for 99% occupied emitter bandwidth (B09 & C07)
        obw_tail_clipped = False
        obw_clip_reason: str | None = None
        if mode == "emitter" and check_edge_clipping:
            # 1. If ROI is configured within wider observation, check if > 1% excess energy lies outside ROI
            if roi_hz is not None:
                f_min, f_max = float(roi_hz[0]), float(roi_hz[1])
                if f[0] >= f_min or f[-1] <= f_max:
                    obw_tail_clipped = True
                    obw_clip_reason = "Cannot certify physical emitter 99% containment: observation is truncated at frequency boundary with no wider ROI"
                else:
                    p_excess_all = np.maximum(p - noise_linear, 0.0)
                    total_excess = float(np.sum(p_excess_all))
                    roi_excess = float(np.sum(p_excess_all[roi_mask]))
                    outside_excess = total_excess - roi_excess
                    if total_excess > 0.0 and (outside_excess / total_excess) > 0.01:
                        obw_tail_clipped = True
                        frac_out = (outside_excess / total_excess) * 100.0
                        obw_clip_reason = f"Emitter 99% energy tails not contained in ROI ({frac_out:.1f}% excess energy outside ROI)"
                    else:
                        obw_tail_clipped = False
            else:
                # 2. Observation boundary edge: when roi_hz is None, the observation is truncated at the frequency
                # boundaries without a wider observation window to certify physical emitter 99% containment
                obw_tail_clipped = True
                obw_clip_reason = "Cannot certify physical emitter 99% containment: observation is truncated at frequency boundary with no wider ROI"

        if mode == "emitter" and cfg.occupied_power_scope == "physical_emitter" and not cfg.complete_observation:
            obw_tail_clipped = True
            obw_clip_reason = "Physical occupied bandwidth requires a complete-observation assumption and provenance"

        # 1. 99% Occupied Bandwidth
        if obw_tail_clipped:
            occupied_99_fv = FeatureValue(
                value=float("nan"),
                unit="Hz",
                valid=False,
                reason=obw_clip_reason or "Emitter spectrum clipped by boundary",
            )
        elif p_total > 0.0 and math.isfinite(p_total):
            c = np.cumsum(p_sig, dtype=np.float64)
            c_norm = c / c[-1]
            if interpolate_edges:
                # Low edge (0.005)
                k_low = int(np.searchsorted(c_norm, 0.005, side="left"))
                if k_low == 0:
                    frac_low = 0.005 / c_norm[0] if c_norm[0] > 0 else 0.0
                    f_low = (f_sig[0] - df / 2.0) + frac_low * df
                else:
                    denom = c_norm[k_low] - c_norm[k_low - 1]
                    frac_low = (0.005 - c_norm[k_low - 1]) / denom if denom > 0 else 0.0
                    f_low = (f_sig[k_low - 1] + f_sig[k_low]) / 2.0 + frac_low * df

                # High edge (0.995)
                k_high = int(np.searchsorted(c_norm, 0.995, side="left"))
                if k_high == 0:
                    frac_high = 0.995 / c_norm[0] if c_norm[0] > 0 else 0.0
                    f_high = (f_sig[0] - df / 2.0) + frac_high * df
                else:
                    k_h_prev = min(k_high - 1, len(f_sig) - 1)
                    k_h_curr = min(k_high, len(f_sig) - 1)
                    denom = c_norm[k_h_curr] - c_norm[k_h_prev]
                    frac_high = (0.995 - c_norm[k_h_prev]) / denom if denom > 0 else 0.0
                    f_high = (f_sig[k_h_prev] + f_sig[k_h_curr]) / 2.0 + frac_high * df

                obw_val = max(float(f_high - f_low), df)
            else:
                k_low = int(np.searchsorted(c_norm, 0.005, side="left"))
                k_high = int(np.searchsorted(c_norm, 0.995, side="left"))
                k_low = min(max(k_low, 0), len(f_sig) - 1)
                k_high = min(max(k_high, k_low), len(f_sig) - 1)
                lower_edge_obw = float(f_sig[k_low] - df / 2.0)
                upper_edge_obw = float(f_sig[k_high] + df / 2.0)
                obw_val = float(upper_edge_obw - lower_edge_obw)

            occupied_99_fv = FeatureValue(
                value=obw_val,
                unit="Hz",
                valid=True,
                reason=None,
            )
        else:
            occupied_99_fv = FeatureValue(
                value=float("nan"),
                unit="Hz",
                valid=False,
                reason="PSD contains no positive signal power",
            )

        # 2. 3 dB Bandwidth around the dominant peak (with prominence and ambiguity verification)
        if p_max <= 0.0 or not math.isfinite(p_max):
            three_db_fv = FeatureValue(
                value=float("nan"),
                unit="Hz",
                valid=False,
                reason="PSD contains no positive signal power",
            )
        else:
            k_peak = int(np.argmax(p_sig))
            p_half = p_max / 2.0

            k_left = k_peak
            while k_left > 0 and p_sig[k_left - 1] >= p_half:
                k_left -= 1

            k_right = k_peak
            while k_right < len(p_sig) - 1 and p_sig[k_right + 1] >= p_half:
                k_right += 1

            # Check if 3 dB lobe crossings are clipped by observation or ROI boundary (B09)
            is_3db_clipped = False
            if mode == "emitter" and check_edge_clipping and len(p_sig) >= 2:
                # Assess boundary whether roi_hz is set or not
                if (k_left == 0 and p_sig[0] >= p_half) or (k_right == len(p_sig) - 1 and p_sig[-1] >= p_half):
                    is_3db_clipped = True

            lower_edge_3db = float(f_sig[k_left] - df / 2.0)
            upper_edge_3db = float(f_sig[k_right] + df / 2.0)
            three_db_val = float(upper_edge_3db - lower_edge_3db)

            # Prominence check for flat/OFDM wideband signals
            p_lobe = float(np.sum(p_sig[k_left : k_right + 1]))
            lobe_power_fraction = p_lobe / p_total if p_total > 0.0 else 0.0
            p_median = float(np.median(p_sig))
            prominence_db = 10.0 * math.log10(p_max / p_median) if p_median > 0.0 else float("inf")

            is_broad_spectrum = (occupied_99_fv.valid and occupied_99_fv.value >= 10.0 * df)
            is_minor_fluctuation = (
                lobe_power_fraction < 0.10 or prominence_db < dominant_lobe_min_prominence_db
            ) if is_broad_spectrum else False

            # Multimodal ambiguity check
            is_multimodal = False
            if mode == "emitter" and len(p_sig) >= 3:
                peak_indices = [
                    i
                    for i in range(1, len(p_sig) - 1)
                    if p_sig[i] >= p_half and p_sig[i] >= p_sig[i - 1] and p_sig[i] >= p_sig[i + 1]
                ]
                if len(peak_indices) > 1:
                    for i_idx in range(len(peak_indices) - 1):
                        p1, p2 = peak_indices[i_idx], peak_indices[i_idx + 1]
                        if p2 > p1 + 1:
                            valley_min = float(np.min(p_sig[p1 : p2 + 1]))
                            if valley_min < multimodal_ratio_threshold * p_max:
                                is_multimodal = True
                                break

            if is_3db_clipped:
                three_db_fv = FeatureValue(
                    value=float("nan"),
                    unit="Hz",
                    valid=False,
                    reason="3 dB lobe clipped by observation/ROI boundary",
                )
            elif is_broad_spectrum and is_minor_fluctuation:
                three_db_fv = FeatureValue(
                    value=float("nan"),
                    unit="Hz",
                    valid=False,
                    reason="No distinct dominant spectral lobe observed (peak fluctuation inside wideband spectrum)",
                )
            elif is_multimodal:
                three_db_fv = FeatureValue(
                    value=float("nan"),
                    unit="Hz",
                    valid=False,
                    reason="Ambiguous multimodal spectrum (multiple distinct lobes within 3 dB of peak)",
                )
            else:
                three_db_fv = FeatureValue(
                    value=three_db_val,
                    unit="Hz",
                    valid=True,
                    reason=None,
                )

        features = {
            "bandwidth.occupied_99_hz": replace(
                occupied_99_fv, interpretation=("raw_psd_within_roi" if mode == "raw_psd"
                    else ("physical_emitter_under_declared_support" if cfg.occupied_power_scope == "physical_emitter"
                          else "observed_excess_within_roi"))),
            "bandwidth.three_db_hz": replace(three_db_fv, interpretation="observed_dominant_lobe_half_power"),
        }

        return FeatureSet(
            extractor=self.name,
            extractor_version=self.version,
            features=features,
        )
