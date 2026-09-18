from __future__ import annotations
import importlib

_main_mod = importlib.import_module("packages.IQ Calibration.main")
_dc_mod = importlib.import_module("packages.IQ Calibration.DC_Offset")
_gain_mod = importlib.import_module("packages.IQ Calibration.IQ_Gain_Imbalance")
_phase_mod = importlib.import_module("packages.IQ Calibration.IQ_Phase_Imbalance")
_quad_mod = importlib.import_module("packages.IQ Calibration.Quadrature_Error")
_skew_mod = importlib.import_module("packages.IQ Calibration.IQ_Timing_Skew")
_irr_mod = importlib.import_module("packages.IQ Calibration.Image_Rejection")
_leak_mod = importlib.import_module("packages.IQ Calibration.Carrier_Leakage")
_cfo_mod = importlib.import_module("packages.IQ Calibration.Oscillator_Offset")
_cal_mod = importlib.import_module("packages.IQ Calibration.Calibration_Estimation")

# Models & Results
CalibrationInput = _cal_mod.CalibrationInput
DCOffsetResult = _dc_mod.DCOffsetResult
IQGainImbalanceResult = _gain_mod.IQGainImbalanceResult
IQPhaseImbalanceResult = _phase_mod.IQPhaseImbalanceResult
QuadratureErrorResult = _quad_mod.QuadratureErrorResult
IQTimingSkewResult = _skew_mod.IQTimingSkewResult
ImageRejectionResult = _irr_mod.ImageRejectionResult
CarrierLeakageResult = _leak_mod.CarrierLeakageResult
OscillatorOffsetResult = _cfo_mod.OscillatorOffsetResult
CalibrationEstimationResult = _cal_mod.CalibrationEstimationResult
CalibrationResult = _cal_mod.CalibrationResult
CorrectionResult = _cal_mod.CorrectionResult

# Analyzer Classes
DCOffsetAnalyzer = _dc_mod.DCOffsetAnalyzer
DCOffsetEstimator = _dc_mod.DCOffsetEstimator
IQGainImbalanceAnalyzer = _gain_mod.IQGainImbalanceAnalyzer
IQGainImbalanceEstimator = _gain_mod.IQGainImbalanceEstimator
IQPhaseImbalanceAnalyzer = _phase_mod.IQPhaseImbalanceAnalyzer
IQPhaseImbalanceEstimator = _phase_mod.IQPhaseImbalanceEstimator
QuadratureErrorAnalyzer = _quad_mod.QuadratureErrorAnalyzer
QuadratureErrorEstimator = _quad_mod.QuadratureErrorEstimator
IQTimingSkewAnalyzer = _skew_mod.IQTimingSkewAnalyzer
IQTimingSkewEstimator = _skew_mod.IQTimingSkewEstimator
ImageRejectionAnalyzer = _irr_mod.ImageRejectionAnalyzer
ImageRejectionEstimator = _irr_mod.ImageRejectionEstimator
CarrierLeakageAnalyzer = _leak_mod.CarrierLeakageAnalyzer
CarrierLeakageEstimator = _leak_mod.CarrierLeakageEstimator
OscillatorOffsetAnalyzer = _cfo_mod.OscillatorOffsetAnalyzer
OscillatorOffsetEstimator = _cfo_mod.OscillatorOffsetEstimator
CalibrationEstimationAnalyzer = _cal_mod.CalibrationEstimationAnalyzer
CalibrationEstimator = _cal_mod.CalibrationEstimator

# Functional APIs
estimate_dc_offset = _dc_mod.estimate_dc_offset
compute_dc_offset = _dc_mod.compute_dc_offset
estimate_gain_imbalance = _gain_mod.estimate_gain_imbalance
compute_gain_imbalance = _gain_mod.compute_gain_imbalance
estimate_phase_imbalance = _phase_mod.estimate_phase_imbalance
compute_phase_imbalance = _phase_mod.compute_phase_imbalance
estimate_quadrature_error = _quad_mod.estimate_quadrature_error
compute_quadrature_error = _quad_mod.compute_quadrature_error
estimate_timing_skew = _skew_mod.estimate_timing_skew
compute_timing_skew = _skew_mod.compute_timing_skew
estimate_image_rejection = _irr_mod.estimate_image_rejection
compute_image_rejection = _irr_mod.compute_image_rejection
estimate_carrier_leakage = _leak_mod.estimate_carrier_leakage
compute_carrier_leakage = _leak_mod.compute_carrier_leakage
estimate_oscillator_offset = _cfo_mod.estimate_oscillator_offset
compute_oscillator_offset = _cfo_mod.compute_oscillator_offset
estimate_calibration = _cal_mod.estimate_calibration
compute_calibration = _cal_mod.compute_calibration

# Correction & Pipeline
apply_dc_correction = _cal_mod.apply_dc_correction
apply_cfo_correction = _cal_mod.apply_cfo_correction
apply_timing_skew_correction = _cal_mod.apply_timing_skew_correction
apply_iq_imbalance_correction = _cal_mod.apply_iq_imbalance_correction
apply_calibration = _cal_mod.apply_calibration
CalibrationProcessor = _main_mod.CalibrationProcessor
calibrate_iq = _main_mod.calibrate_iq
generate_impaired_iq = _main_mod.generate_impaired_iq

__all__ = [
    "CalibrationInput",
    "DCOffsetResult",
    "IQGainImbalanceResult",
    "IQPhaseImbalanceResult",
    "QuadratureErrorResult",
    "IQTimingSkewResult",
    "ImageRejectionResult",
    "CarrierLeakageResult",
    "OscillatorOffsetResult",
    "CalibrationEstimationResult",
    "CalibrationResult",
    "CorrectionResult",
    "DCOffsetAnalyzer",
    "DCOffsetEstimator",
    "IQGainImbalanceAnalyzer",
    "IQGainImbalanceEstimator",
    "IQPhaseImbalanceAnalyzer",
    "IQPhaseImbalanceEstimator",
    "QuadratureErrorAnalyzer",
    "QuadratureErrorEstimator",
    "IQTimingSkewAnalyzer",
    "IQTimingSkewEstimator",
    "ImageRejectionAnalyzer",
    "ImageRejectionEstimator",
    "CarrierLeakageAnalyzer",
    "CarrierLeakageEstimator",
    "OscillatorOffsetAnalyzer",
    "OscillatorOffsetEstimator",
    "CalibrationEstimationAnalyzer",
    "CalibrationEstimator",
    "estimate_dc_offset",
    "compute_dc_offset",
    "estimate_gain_imbalance",
    "compute_gain_imbalance",
    "estimate_phase_imbalance",
    "compute_phase_imbalance",
    "estimate_quadrature_error",
    "compute_quadrature_error",
    "estimate_timing_skew",
    "compute_timing_skew",
    "estimate_image_rejection",
    "compute_image_rejection",
    "estimate_carrier_leakage",
    "compute_carrier_leakage",
    "estimate_oscillator_offset",
    "compute_oscillator_offset",
    "estimate_calibration",
    "compute_calibration",
    "apply_dc_correction",
    "apply_cfo_correction",
    "apply_timing_skew_correction",
    "apply_iq_imbalance_correction",
    "apply_calibration",
    "CalibrationProcessor",
    "calibrate_iq",
    "generate_impaired_iq",
]
