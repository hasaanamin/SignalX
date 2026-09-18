from __future__ import annotations
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    _pkg_dir = Path(__file__).resolve().parent
    _src_dir = _pkg_dir.parent.parent
    for _p in [str(_src_dir), str(_pkg_dir.parent)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import importlib
    importlib.import_module(f"packages.{_pkg_dir.name}")
    __package__ = f"packages.{_pkg_dir.name}"

from typing import Any, Dict, Tuple
import math
import numpy as np

from .base import BaseModulationMetric, ModulationMetricFactory
from ._results import ModulationClassificationResult
from ._validated_helpers import _extract_modulation_features
from ._constellation_helpers import (
    _remove_dc,
    _normalize_complex_rms,
    _estimate_coarse_frequency_offset,
    _apply_frequency_correction,
    _recover_symbol_samples,
    _constellation_cluster_statistics,
)


@ModulationMetricFactory.register(["modulation_type_classification", "modulation_classification", "modulation_classifier", "mod_type", "classifier"])
class ModulationTypeClassification(BaseModulationMetric):
    """
    Modulation Type Classification Module.
    Independently classifies modulation type directly from raw complex IQ.
    Uses multi-domain feature extraction (amplitude, instantaneous frequency, detrended phase,
    spectral, time-frequency, cyclostationary, and constellation cluster geometry).
    
    Supported coarse classes:
      CW, AM, FM, PM, ASK, FSK, PSK, QAM, UNKNOWN
    """

    VALID_CLASSES: Tuple[str, ...] = ("CW", "AM", "FM", "PM", "ASK", "FSK", "PSK", "QAM", "UNKNOWN")

    def __init__(
        self,
        sample_rate: float = 1.0,
        classifier_mode: str = "rules",
        confidence_threshold: float = 0.40,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.classifier_mode = str(classifier_mode).lower().strip()
        self.confidence_threshold = float(confidence_threshold)

    @property
    def name(self) -> str:
        return "modulation_type_classification"

    def compute(
        self,
        iq_data: Any,
        sample_rate: float | None = None,
        **kwargs: Any,
    ) -> ModulationClassificationResult:
        """Executes Modulation Type Classification directly on raw complex IQ."""
        fs = float(sample_rate) if sample_rate is not None else self.sample_rate

        # 1. Feature Extraction via canonical private helper
        feats = _extract_modulation_features(iq_data, sample_rate=fs)

        # 2. Digital Constellation Features
        z_dc, _ = _remove_dc(np.asarray(iq_data))
        z_norm, _ = _normalize_complex_rms(z_dc)
        cfo_hz, _ = _estimate_coarse_frequency_offset(z_norm, sample_rate=fs)
        z_corr = _apply_frequency_correction(z_norm, sample_rate=fs, freq_offset_hz=cfo_hz)
        timing_info = _recover_symbol_samples(z_corr, sample_rate=fs)
        symbols = timing_info["symbol_points"]
        const_stats = _constellation_cluster_statistics(symbols, sample_rate=fs)

        # Merge constellation features into dictionary
        feats["num_symbol_points"] = float(const_stats["num_symbol_points"])
        feats["radial_level_count"] = float(const_stats["radial_level_count"])
        feats["apparent_phase_state_count"] = float(const_stats["apparent_phase_state_count"])
        feats["phase_concentration"] = float(const_stats["phase_concentration"])
        feats["constellation_cluster_count"] = float(const_stats["constellation_cluster_count"])
        feats["cluster_compactness"] = float(const_stats["cluster_compactness"])
        feats["timing_valid"] = 1.0 if timing_info["valid"] else 0.0

        # 3. Deterministic Evidence Scoring
        scores: Dict[str, float] = {c: 0.0 for c in self.VALID_CLASSES}

        env_cv = float(feats["envelope_cv"])
        am_act = float(feats["am_activity"])
        fm_rms = float(feats["fm_rms_deviation_hz"])
        num_f_states = int(feats["num_frequency_states"])
        f_sep = float(feats["frequency_state_separation"])
        pm_rms = float(feats["pm_rms_deviation_rad"])
        cyclo_det = float(feats["cyclo_detected"])
        cyclo_snr = float(feats["cyclo_snr_db"])
        rad_levels = int(feats["radial_level_count"])
        phase_k = int(feats["apparent_phase_state_count"])
        phase_conc = float(feats["phase_concentration"])
        clust_k = int(feats["constellation_cluster_count"])
        spec_flatness = float(feats["spectral_flatness"])
        obw = float(feats["occupied_bandwidth_hz"])

        # Low power / pure noise check
        if feats["envelope_mean"] < 1e-12 or (spec_flatness > 0.85 and env_cv > 0.45 and obw > 0.80 * fs):
            scores["UNKNOWN"] = 0.95

        # CW evidence: flat envelope (env_cv < 0.10), zero FM dev (fm_rms < 25 kHz), zero PM dev (pm_rms < 0.80 rad), narrow obw
        cw_penalties = [
            np.clip(env_cv / 0.10, 0.0, 1.0),
            np.clip(fm_rms / max(25000.0, 0.020 * fs), 0.0, 1.0),
            np.clip(pm_rms / 0.80, 0.0, 1.0),
            np.clip(obw / max(50000.0, 0.050 * fs), 0.0, 1.0),
        ]
        cw_ev = float(np.clip(1.0 - np.max(cw_penalties), 0.0, 1.0))
        scores["CW"] = cw_ev

        # AM evidence: strong periodic envelope variation (env_cv >= 0.12, am_act >= 0.10), low FM dev, narrow obw (< 0.035 * fs)
        am_env_gain = np.clip((env_cv - 0.08) / 0.20, 0.0, 1.0)
        am_penalties = max(
            np.clip(fm_rms / (0.020 * fs), 0.0, 1.0),
            np.clip(obw / (0.035 * fs), 0.0, 1.0),
            0.5 if (rad_levels >= 3 and feats.get("normalized_fourth_moment", 1.0) > 1.4 and cyclo_det == 1.0) else 0.0,
        )
        if num_f_states in (2, 4) and f_sep >= 1.5:
            am_env_gain *= 0.1
        am_ev = float(np.clip(am_env_gain * (1.0 - am_penalties), 0.0, 1.0))
        scores["AM"] = am_ev

        # FM evidence: strictly constant envelope (env_cv < 0.08), strong frequency deviation (fm_rms >= 10 kHz)
        fm_gain = np.clip((fm_rms - 4000.0) / 12000.0, 0.0, 1.0)
        fm_penalty = np.clip(env_cv / 0.10, 0.0, 1.0)
        fm_state_mult = 0.3 if num_f_states in (2, 4) and f_sep >= 1.8 else 1.0
        fm_ev = float(np.clip(fm_gain * (1.0 - fm_penalty) * fm_state_mult, 0.0, 1.0))
        scores["FM"] = fm_ev

        # PM evidence: constant envelope (env_cv < 0.08), moderate PM deviation (0.45 <= pm_rms <= 2.5 rad), low FM deviation, narrow obw
        pm_gain = np.clip((pm_rms - 0.45) / 0.50, 0.0, 1.0) * np.clip((5.0 - pm_rms) / 2.5, 0.0, 1.0)
        pm_penalties = max(
            np.clip(env_cv / 0.10, 0.0, 1.0),
            np.clip(fm_rms / (0.015 * fs), 0.0, 1.0),
            np.clip(obw / (0.030 * fs), 0.0, 1.0),
        )
        pm_ev = float(np.clip(pm_gain * (1.0 - pm_penalties), 0.0, 1.0))
        scores["PM"] = pm_ev

        # FSK evidence: discrete frequency states (K in 2, 4, 8 with f_sep >= 1.2), constant envelope (env_cv < 0.15), wide obw (obw > 0.01 * fs)
        fsk_ev = (
            (1.0 - np.clip(env_cv / 0.15, 0.0, 1.0))
            * (1.0 if num_f_states in (2, 4, 8) and f_sep >= 1.0 else 0.0)
            * np.clip(f_sep / 2.0, 0.0, 1.0)
            * np.clip(obw / (0.01 * fs), 0.0, 1.0)
        )
        scores["FSK"] = float(fsk_ev)

        # ASK evidence: multiple discrete amplitude levels (rad_levels >= 2), wide envelope variation (env_cv >= 0.25), digital bandwidth (obw >= 0.02 * fs)
        ask_ev = (
            np.clip((env_cv - 0.20) / 0.30, 0.0, 1.0)
            * (1.0 if rad_levels >= 2 else 0.0)
            * np.clip(obw / (0.02 * fs), 0.0, 1.0)
            * (1.0 - np.clip(fm_rms / (0.015 * fs), 0.0, 1.0))
            * (0.9 if cyclo_det == 1.0 else 0.5)
        )
        scores["ASK"] = float(ask_ev)

        # PSK evidence: wide digital bandwidth (obw >= 0.03 * fs), envelope transition dips (env_cv >= 0.15), large phase transitions
        psk_ev = (
            np.clip(obw / (0.03 * fs), 0.0, 1.0)
            * np.clip((env_cv - 0.08) / 0.18, 0.0, 1.0)
            * (1.0 if rad_levels <= 2 else 0.2)
            * np.clip((pm_rms - 2.0) / 8.0, 0.0, 1.0)
            * (1.0 if phase_k in (2, 4, 8) and phase_conc >= 0.70 else 0.5)
            * (0.9 if cyclo_det == 1.0 else 0.6)
        )
        if rad_levels >= 3 or feats.get("normalized_fourth_moment", 1.0) > 1.35:
            psk_ev *= 0.1
        scores["PSK"] = float(psk_ev)

        # QAM evidence: multiple radial levels (rad_levels >= 3), high 4th moment (> 1.25), wide digital bandwidth (obw >= 0.03 * fs)
        qam_ev = (
            np.clip(obw / (0.03 * fs), 0.0, 1.0)
            * (1.0 if rad_levels >= 3 else 0.2)
            * np.clip((feats.get("normalized_fourth_moment", 1.0) - 1.15) / 0.30, 0.0, 1.0)
            * (0.9 if cyclo_det == 1.0 else 0.6)
        )
        scores["QAM"] = float(qam_ev)

        # Normalize non-unknown scores
        best_candidate = max((c for c in self.VALID_CLASSES if c != "UNKNOWN"), key=lambda c: scores[c])
        best_score = scores[best_candidate]

        if best_score < self.confidence_threshold or scores["UNKNOWN"] >= 0.8:
            predicted = "UNKNOWN"
            confidence = float(scores["UNKNOWN"]) if scores["UNKNOWN"] >= 0.8 else float(1.0 - best_score)
            is_unknown = True
            reason = "Insufficient feature evidence or high ambiguity."
        else:
            predicted = best_candidate
            confidence = float(np.clip(best_score, 0.4, 0.99))
            is_unknown = False
            reason = f"Classified as {predicted} with score {best_score:.2f}."

        return ModulationClassificationResult(
            predicted_class=predicted,
            confidence=confidence,
            class_scores=scores,
            extracted_features=feats,
            valid_classes=self.VALID_CLASSES,
            unknown=is_unknown,
            reason=reason,
            classifier_mode=self.classifier_mode,
        )


def compute_modulation_type_classification(
    iq_data: Any,
    sample_rate: float = 1.0,
    **kwargs: Any,
) -> ModulationClassificationResult:
    """Functional convenience wrapper for ModulationTypeClassification."""
    return ModulationTypeClassification(sample_rate=sample_rate, **kwargs).compute(iq_data)


def main() -> None:
    print("=" * 60)
    print("  ModulationTypeClassification: Demonstration")
    print("=" * 60)
    fs = 1.0e6
    t = np.arange(1024) / fs
    sample_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.05 * (np.random.randn(1024) + 1j * np.random.randn(1024))
    
    stage = ModulationTypeClassification()
    result = stage.compute(sample_iq, sample_rate=fs)
    print(f"Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
