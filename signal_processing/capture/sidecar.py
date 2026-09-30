"""Runtime envelope sidecar capturing execution metadata, landmarks, and quality."""

from __future__ import annotations

from dataclasses import dataclass

from signal_processing.core.contracts import AnalysisResult, FeatureVector, SignalEvent


@dataclass(frozen=True)
class CaptureSidecar:
    """Immutable runtime sidecar accompanying each extracted SignalEvent.

    Captures landmark sample coordinates, detection provenance, boundary resolution,
    requested vs actual extents, continuity diagnostics, and feature extraction status.
    """

    stream_epoch: int
    event_id: str
    config_version: str
    detector_mode: str
    monitored_band_hz: tuple[float, float] | None
    first_on_sample: int
    confirmation_sample: int
    end_confirmation_sample: int | None
    requested_leading_samples: int
    actual_leading_samples: int
    requested_trailing_samples: int
    actual_trailing_samples: int
    boundary_uncertainty_samples: int | None
    noise_provenance: str
    noise_valid: bool
    termination_reason: str
    is_truncated: bool
    known_lost_samples: int
    unknown_loss: bool
    feature_status: str
    feature_reason: str | None
    event: SignalEvent
    features: FeatureVector | None = None
    analysis_result: AnalysisResult | None = None
    reopened_count: int = 0
    processing_generation: int | None = None

    def with_features(
        self,
        features: FeatureVector | None,
        feature_status: str,
        feature_reason: str | None = None,
        analysis_result: AnalysisResult | None = None,
    ) -> CaptureSidecar:
        """Return a copy of this sidecar updated with feature extraction results."""
        return CaptureSidecar(
            stream_epoch=self.stream_epoch,
            event_id=self.event_id,
            config_version=self.config_version,
            detector_mode=self.detector_mode,
            monitored_band_hz=self.monitored_band_hz,
            first_on_sample=self.first_on_sample,
            confirmation_sample=self.confirmation_sample,
            end_confirmation_sample=self.end_confirmation_sample,
            requested_leading_samples=self.requested_leading_samples,
            actual_leading_samples=self.actual_leading_samples,
            requested_trailing_samples=self.requested_trailing_samples,
            actual_trailing_samples=self.actual_trailing_samples,
            boundary_uncertainty_samples=self.boundary_uncertainty_samples,
            noise_provenance=self.noise_provenance,
            noise_valid=self.noise_valid,
            termination_reason=self.termination_reason,
            is_truncated=self.is_truncated,
            known_lost_samples=self.known_lost_samples,
            unknown_loss=self.unknown_loss,
            feature_status=feature_status,
            feature_reason=feature_reason,
            event=self.event,
            features=features,
            analysis_result=analysis_result if analysis_result is not None else self.analysis_result,
            reopened_count=self.reopened_count,
            processing_generation=self.processing_generation,
        )
