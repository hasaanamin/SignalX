"""Feature aggregator coordinating multi-family feature extraction and FeatureVector assembly."""

import hashlib
import json
from typing import Any

from signal_processing.context import SignalContext
from signal_processing.core import (
    AnalysisResult,
    CaptureQuality,
    ExtractionPolicy,
    FeatureSet,
    FeatureValue,
    FeatureVector,
)
from signal_processing.core.exceptions import ContractValidationError
from signal_processing.features.aggregation.exceptions import FeatureAggregationError
from signal_processing.features.aggregation.registry import FeatureRegistry


class FeatureAggregator:
    """Orchestrates multi-extractor feature extraction against a single SignalContext.

    Guarantees:
        - Stateless with respect to events.
        - Shares the exact same SignalContext instance across all registered extractors
          to preserve transform cache reuse.
        - Validates all returned FeatureSets (type, extractor name match, extractor version match,
          non-empty features, FeatureValue types).
        - Prevents partial FeatureVector output on extractor failure (atomic operation).
        - Merges FeatureSets into a single unified FeatureVector using FeatureVector.from_feature_sets.
        - Enforces that duplicate feature keys cause fail-fast aggregation errors.
    """

    def __init__(
        self,
        registry: FeatureRegistry,
        *,
        feature_schema_version: str = "1.0",
    ) -> None:
        """Initialize FeatureAggregator with a FeatureRegistry and schema version.

        Args:
            registry: Configured FeatureRegistry instance.
            feature_schema_version: Schema version string for downstream consumers (default: "1.0").

        Raises:
            FeatureAggregationError: If registry is not a FeatureRegistry or feature_schema_version
                is not a non-blank string.
        """
        if not isinstance(registry, FeatureRegistry):
            raise FeatureAggregationError(
                f"Expected FeatureRegistry instance, got {type(registry).__name__}"
            )

        if not isinstance(feature_schema_version, str) or not feature_schema_version.strip():
            raise FeatureAggregationError(
                f"feature_schema_version must be a non-blank string, got {feature_schema_version!r}"
            )

        self._registry = registry
        self._feature_schema_version = feature_schema_version

    @property
    def registry(self) -> FeatureRegistry:
        """The FeatureRegistry configured for this aggregator."""
        return self._registry

    @property
    def feature_schema_version(self) -> str:
        """The schema version for assembled FeatureVector instances."""
        return self._feature_schema_version

    def extract(self, context: SignalContext | Any) -> FeatureVector:
        """Execute all registered extractors sequentially against a single SignalContext and assemble a FeatureVector.

        Args:
            context: SignalContext instance (must provide non-blank event_id).

        Returns:
            Assembled immutable FeatureVector containing all extracted features.

        Raises:
            FeatureAggregationError: If context is invalid, any extractor fails, returned
                FeatureSets violate invariants, or duplicate feature keys are detected.
        """
        if context is None:
            raise FeatureAggregationError("Context must not be None")

        event_id = getattr(context, "event_id", None)
        if not isinstance(event_id, str) or not event_id.strip():
            raise FeatureAggregationError(
                f"Context must provide a non-blank string 'event_id', got {event_id!r}"
            )

        feature_sets: list[FeatureSet] = []

        # Execute extractors sequentially in deterministic registry order
        for extractor in self._registry.extractors:
            try:
                fset = extractor.extract(context)
            except Exception as exc:
                raise FeatureAggregationError(
                    f"Extractor '{extractor.name}' unexpectedly raised an exception: {exc}"
                ) from exc

            self._validate_feature_set(fset, extractor)
            feature_sets.append(fset)

        # Merge FeatureSets into a unified FeatureVector
        try:
            return FeatureVector.from_feature_sets(
                event_id=event_id,
                feature_schema_version=self._feature_schema_version,
                feature_sets=feature_sets,
            )
        except ContractValidationError as exc:
            raise FeatureAggregationError(f"Failed to assemble FeatureVector: {exc}") from exc

    def extract_analysis_result(self, context: SignalContext | Any) -> AnalysisResult:
        """Extract features and package into an AnalysisResult envelope with policy and capture quality.

        Args:
            context: SignalContext instance (must provide non-blank event_id).

        Returns:
            AnalysisResult containing feature_vector, extraction policy, and capture quality.
        """
        vector = self.extract(context)

        cfg = getattr(context, "config", None)
        sample_rate_hz = getattr(context, "sample_rate_hz", 1.0)
        obs_samples = getattr(context, "signal_num_samples", 0)
        capture_quality = getattr(context, "capture_quality", None)
        if capture_quality is None:
            capture_quality = CaptureQuality(complete=True, dropped_samples=0, is_continuous=True)

        psd_mode = str(getattr(cfg, "psd_mode", "periodogram")).lower()
        if psd_mode == "welch":
            psd_nperseg_val = getattr(cfg, "psd_nperseg", None)
            resolved_nperseg = int(psd_nperseg_val) if psd_nperseg_val is not None else min(int(obs_samples), 1024)
            psd_noverlap_val = getattr(cfg, "psd_noverlap", None)
            resolved_noverlap = int(psd_noverlap_val) if psd_noverlap_val is not None else (resolved_nperseg // 2)
            nfft_val = getattr(cfg, "nfft", None)
            resolved_nfft = int(nfft_val) if nfft_val is not None else resolved_nperseg
            resolved_window = str(getattr(cfg, "psd_welch_window", "hann"))
        else:
            resolved_nperseg = int(obs_samples)
            resolved_noverlap = 0
            nfft_val = getattr(cfg, "nfft", None)
            resolved_nfft = int(nfft_val) if nfft_val is not None else int(obs_samples)
            resolved_window = str(getattr(cfg, "spectral_window", "hann"))

        autocorr_lag_val = getattr(cfg, "autocorrelation_max_lag_samples", None)
        resolved_max_lag = int(autocorr_lag_val) if autocorr_lag_val is not None else max(0, int(obs_samples) - 1)
        resolved_autocorr_norm = bool(getattr(cfg, "autocorrelation_normalized", True))
        resolved_phase_thresh = float(getattr(cfg, "phase_amplitude_threshold", 0.0))
        resolved_centered = bool(getattr(cfg, "centered_spectrum", True))
        resolved_p_floor = float(getattr(cfg, "power_floor_dbfs", -300.0))
        resolved_psd_floor = float(getattr(cfg, "psd_floor_dbfs_per_hz", -300.0))

        # Check registered extractor overrides (B10)
        bw_ext = None
        amp_ext = None
        for ext in self._registry.extractors:
            if getattr(ext, "name", None) == "bandwidth":
                bw_ext = ext
            elif getattr(ext, "name", None) == "amplitude":
                amp_ext = ext

        from signal_processing.features.bandwidth import BandwidthFeatureExtractor
        bw_cfg = (bw_ext or BandwidthFeatureExtractor()).resolve_config(context)
        resolved_bw_mode = bw_cfg.mode
        resolved_bw_interp = bw_cfg.interpolate_edges
        resolved_bw_prom = bw_cfg.dominant_lobe_min_prominence_db
        resolved_bw_roi = list(bw_cfg.roi_hz) if bw_cfg.roi_hz is not None else None
        resolved_bw_noise_floor = bw_cfg.noise_floor_dbfs_per_hz
        resolved_bw_min_snr = bw_cfg.min_snr_db
        resolved_check_edge_clipping = bw_cfg.check_edge_clipping

        resolved_amplitude_ddof = (
            getattr(amp_ext, "_config", None).ddof
            if (amp_ext is not None and getattr(amp_ext, "_config", None) is not None)
            else 0
        )

        canonical_dict = {
            "schema_version": self._feature_schema_version,
            "sample_rate_hz": float(sample_rate_hz),
            "observation_samples": int(obs_samples),
            "psd_mode": psd_mode,
            "nfft": resolved_nfft,
            "spectral_window": resolved_window,
            "psd_nperseg": resolved_nperseg,
            "psd_noverlap": resolved_noverlap,
            "centered_spectrum": resolved_centered,
            "power_floor_dbfs": resolved_p_floor,
            "psd_floor_dbfs_per_hz": resolved_psd_floor,
            "autocorrelation_max_lag_samples": resolved_max_lag,
            "autocorrelation_normalized": resolved_autocorr_norm,
            "phase_amplitude_threshold": resolved_phase_thresh,
            "amplitude_ddof": resolved_amplitude_ddof,
            "bandwidth_mode": resolved_bw_mode,
            "bandwidth_interpolate_edges": resolved_bw_interp,
            "bandwidth_dominant_lobe_min_prominence_db": resolved_bw_prom,
            "bandwidth_roi_hz": resolved_bw_roi,
            "bandwidth_noise_floor_dbfs_per_hz": resolved_bw_noise_floor,
            "bandwidth_min_snr_db": resolved_bw_min_snr,
            "check_edge_clipping": resolved_check_edge_clipping,
            "bandwidth_auto_noise_estimation": bw_cfg.auto_noise_estimation,
            "bandwidth_multimodal_ratio_threshold": bw_cfg.multimodal_ratio_threshold,
            "bandwidth_occupied_power_scope": bw_cfg.occupied_power_scope,
            "bandwidth_complete_observation": bw_cfg.complete_observation,
            "bandwidth_support_provenance": bw_cfg.support_provenance,
            "extractor_versions": {ext.name: ext.version for ext in self._registry.extractors},
            "feature_names": sorted(vector.features.keys()),
        }
        cfg_json = json.dumps(canonical_dict, sort_keys=True)
        policy_hash = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()[:12]
        policy_id = f"policy_{self._feature_schema_version}_{policy_hash}"

        policy = ExtractionPolicy(policy_id=policy_id, **canonical_dict)

        return AnalysisResult(
            feature_vector=vector,
            policy=policy,
            capture_quality=capture_quality,
        )

    @staticmethod
    def _validate_feature_set(fset: Any, extractor: Any) -> None:
        """Validate FeatureSet contract adherence for aggregation."""
        if not isinstance(fset, FeatureSet):
            raise FeatureAggregationError(
                f"Extractor '{extractor.name}' returned invalid result of type {type(fset).__name__}; "
                "expected FeatureSet instance"
            )

        if fset.extractor != extractor.name:
            raise FeatureAggregationError(
                f"Extractor name mismatch: extractor registered as '{extractor.name}' "
                f"returned FeatureSet with extractor='{fset.extractor}'"
            )

        if fset.extractor_version != extractor.version:
            raise FeatureAggregationError(
                f"Extractor version mismatch for '{extractor.name}': registered version='{extractor.version}', "
                f"returned extractor_version='{fset.extractor_version}'"
            )

        if len(fset.features) == 0:
            raise FeatureAggregationError(
                f"Extractor '{extractor.name}' returned an empty FeatureSet"
            )

        for key, val in fset.features.items():
            if not isinstance(val, FeatureValue):
                raise FeatureAggregationError(
                    f"Extractor '{extractor.name}' feature '{key}' has value of type {type(val).__name__}; "
                    "expected FeatureValue instance"
                )
