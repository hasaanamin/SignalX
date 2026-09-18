from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np


def _as_complex_array(x: Any, name: str = "iq") -> np.ndarray:
    """Converts input IQ data to a 1D complex128 numpy array."""
    if isinstance(x, np.ndarray):
        if np.iscomplexobj(x):
            arr = x.astype(np.complex128, copy=False)
        else:
            if x.ndim == 2 and x.shape[0] == 2:
                arr = (x[0, :] + 1j * x[1, :]).astype(np.complex128)
            elif x.ndim == 2 and x.shape[1] == 2:
                arr = (x[:, 0] + 1j * x[:, 1]).astype(np.complex128)
            else:
                arr = x.astype(np.complex128)
    elif isinstance(x, (list, tuple)):
        if len(x) == 2 and isinstance(x[0], (list, tuple, np.ndarray)) and isinstance(x[1], (list, tuple, np.ndarray)):
            arr = (np.asarray(x[0], dtype=np.float64) + 1j * np.asarray(x[1], dtype=np.float64)).astype(np.complex128)
        else:
            arr = np.asarray(x, dtype=np.complex128)
    else:
        arr = np.asarray(x, dtype=np.complex128)

    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


def _validate_finite_iq(arr: np.ndarray, name: str = "iq") -> None:
    """Ensures input complex array contains no NaNs or Infs."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Input '{name}' contains non-finite values (NaN or Inf)")


def _validate_sample_rate(fs: float) -> float:
    """Validates that sample rate is positive and finite."""
    fs_f = float(fs)
    if not np.isfinite(fs_f) or fs_f <= 0.0:
        raise ValueError(f"Sample rate must be positive and finite, got {fs}")
    return fs_f


def _fractional_delay_fft(x: np.ndarray, delay_samples: float) -> np.ndarray:
    """Applies high-fidelity continuous fractional sample delay in frequency domain."""
    n = len(x)
    if abs(delay_samples) < 1e-12 or n == 0:
        return x.copy()

    k = np.fft.fftfreq(n)
    phase_ramp = np.exp(-1j * 2.0 * np.pi * float(delay_samples) * k)
    x_delayed = np.fft.ifft(np.fft.fft(x) * phase_ramp)
    if not np.iscomplexobj(x):
        return np.real(x_delayed)
    return x_delayed


def _compute_psd_welch(
    x: np.ndarray,
    fs: float,
    nfft: int = 2048,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes two-sided Welch Power Spectral Density for complex signals."""
    n = len(x)
    nperseg = min(nfft, n)
    noverlap = int(nperseg * overlap)
    step = nperseg - noverlap

    if step <= 0 or n < nperseg:
        win = np.hanning(n)
        scale = 1.0 / (fs * np.sum(win ** 2))
        spec = np.fft.fftshift(np.fft.fft(x * win, n=nfft))
        psd = (np.abs(spec) ** 2) * scale
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
        return freqs, psd

    win = np.hanning(nperseg)
    win_scale = 1.0 / (fs * np.sum(win ** 2))
    num_segments = (n - noverlap) // step
    psd_accum = np.zeros(nfft, dtype=np.float64)

    for i in range(num_segments):
        start = i * step
        segment = x[start : start + nperseg] * win
        spec = np.fft.fftshift(np.fft.fft(segment, n=nfft))
        psd_accum += (np.abs(spec) ** 2) * win_scale

    psd = psd_accum / float(num_segments)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    return freqs, psd


@dataclass
class CalibrationInput:
    """
    Canonical input descriptor for IQ Calibration analysis and orchestration.
    """
    iq_data: Any = None
    sample_rate: float = 1.0
    expected_frequency_hz: float | None = None
    calibration_tone_frequency_hz: float | None = None
    reference_iq: Any | None = None
    modulation_type: str | None = None
    stream_id: Any | None = None
    source_id: str | int | None = None
    channel_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        iq_data: Any = None,
        sample_rate: float = 1.0,
        expected_frequency_hz: float | None = None,
        calibration_tone_frequency_hz: float | None = None,
        reference_iq: Any | None = None,
        modulation_type: str | None = None,
        stream_id: Any | None = None,
        source_id: str | int | None = None,
        channel_id: int | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if iq_data is None and "iq" in kwargs:
            iq_data = kwargs.pop("iq")
        if expected_frequency_hz is None and "expected_frequency" in kwargs:
            expected_frequency_hz = kwargs.pop("expected_frequency")
        if calibration_tone_frequency_hz is None and "tone_frequency" in kwargs:
            calibration_tone_frequency_hz = kwargs.pop("tone_frequency")

        self.iq_data = iq_data
        self.sample_rate = float(sample_rate)
        self.expected_frequency_hz = expected_frequency_hz
        self.calibration_tone_frequency_hz = calibration_tone_frequency_hz
        self.reference_iq = reference_iq
        self.modulation_type = modulation_type
        self.stream_id = stream_id
        self.source_id = source_id
        self.channel_id = channel_id
        self.metadata = metadata if metadata is not None else {}
        if kwargs:
            self.metadata.update(kwargs)

    @property
    def iq(self) -> Any:
        return self.iq_data


@dataclass
class CalibrationEstimationResult:
    """
    Consolidated calibration state for an IQ receiver or transmission channel.

    Attributes
    ----------
    dc_offset : Any | None
        Result from DC Offset analysis.
    iq_gain_imbalance : Any | None
        Result from I/Q Gain Imbalance analysis.
    iq_phase_imbalance : Any | None
        Result from I/Q Phase Imbalance analysis.
    quadrature_error : Any | None
        Result from Quadrature Error analysis.
    iq_timing_skew : Any | None
        Result from I/Q Timing Skew analysis.
    image_rejection : Any | None
        Result from Image Rejection analysis.
    carrier_leakage : Any | None
        Result from Carrier Leakage analysis.
    oscillator_offset : Any | None
        Result from Oscillator Offset analysis.
    correction_matrix : np.ndarray | None
        2x2 Gram-Schmidt correction transformation matrix H.
    overall_valid : bool
        Overall validity across all critical calibration stages.
    overall_confidence : float
        Mean confidence score across executed calibration modules.
    calibration_method : str
        Strategy used ('reference_aided', 'calibration_tone', 'blind_statistical').
    warnings : list[str]
        List of warning/diagnostic messages.
    metadata : dict[str, Any]
        Diagnostics and computational metadata.
    """
    dc_offset: Any | None = None
    iq_gain_imbalance: Any | None = None
    iq_phase_imbalance: Any | None = None
    quadrature_error: Any | None = None
    iq_timing_skew: Any | None = None
    image_rejection: Any | None = None
    carrier_leakage: Any | None = None
    oscillator_offset: Any | None = None
    correction_matrix: np.ndarray | None = None
    overall_valid: bool = True
    overall_confidence: float = 1.0
    calibration_method: str = "blind_statistical"
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.overall_valid

    @property
    def confidence(self) -> float:
        return self.overall_confidence

    @property
    def gain_imbalance(self) -> Any | None:
        return self.iq_gain_imbalance

    @property
    def phase_imbalance(self) -> Any | None:
        return self.iq_phase_imbalance

    @property
    def timing_skew(self) -> Any | None:
        return self.iq_timing_skew


CalibrationResult = CalibrationEstimationResult


@dataclass
class CorrectionResult:
    """
    Result container for IQ calibration application.

    Attributes
    ----------
    corrected_iq : np.ndarray
        Calibrated complex IQ output array.
    corrections_applied : list[str]
        List of correction stage identifiers executed.
    residual_dc_power : float | None
        Post-correction residual DC power in Watts.
    residual_gain_imbalance_db : float | None
        Post-correction residual gain imbalance in dB.
    residual_phase_imbalance_deg : float | None
        Post-correction residual phase imbalance in degrees.
    residual_cfo_hz : float | None
        Post-correction residual CFO in Hz.
    metadata : dict[str, Any]
        Post-correction diagnostics and metrics.
    """
    corrected_iq: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.complex128))
    corrections_applied: list[str] = field(default_factory=list)
    residual_dc_power: float | None = None
    residual_gain_imbalance_db: float | None = None
    residual_phase_imbalance_deg: float | None = None
    residual_cfo_hz: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def residual_dc_offset_complex(self) -> complex | None:
        if "residual_dc_offset_complex" in self.metadata:
            return complex(self.metadata["residual_dc_offset_complex"])
        if len(self.corrected_iq) > 0:
            return complex(np.mean(self.corrected_iq))
        return None

    @property
    def post_correction_irr_db(self) -> float | None:
        return self.metadata.get("post_correction_irr_db")


def apply_dc_correction(iq_data: Any, dc_complex: complex) -> np.ndarray:
    """Applies DC offset removal: x_corr[n] = x[n] - dc_complex."""
    x = _as_complex_array(iq_data, name="iq_data")
    return np.ascontiguousarray(x - dc_complex)


def apply_cfo_correction(
    iq_data: Any,
    cfo_hz: float,
    sample_rate: float,
    initial_phase_rad: float = 0.0,
) -> np.ndarray:
    """
    Applies Carrier Frequency Offset (CFO) de-rotation:
    x_corr[n] = x[n] * exp(-j * (2*pi * cfo_hz * t + initial_phase_rad)).
    """
    x = _as_complex_array(iq_data, name="iq_data")
    fs = _validate_sample_rate(sample_rate)
    n = len(x)
    t = np.arange(n) / fs
    rotation = np.exp(-1j * (2.0 * np.pi * float(cfo_hz) * t + float(initial_phase_rad)))
    return np.ascontiguousarray(x * rotation)


def apply_timing_skew_correction(
    iq_data: Any,
    timing_skew_samples: float,
    sample_rate: float | None = None,
) -> np.ndarray:
    """
    Applies sub-sample timing skew compensation to the Q branch.
    If Q was delayed by +tau samples, this advances Q by -tau samples.
    """
    x = _as_complex_array(iq_data, name="iq_data")
    i_ch = x.real.copy()
    q_ch = x.imag.copy()
    q_corr = _fractional_delay_fft(q_ch, -float(timing_skew_samples))
    return np.ascontiguousarray(i_ch + 1j * q_corr)


def apply_iq_imbalance_correction(
    iq_data: Any,
    gain_ratio: float,
    phase_imbalance_rad: float,
) -> np.ndarray:
    """
    Applies Gram-Schmidt orthogonalization and amplitude equalization for I/Q imbalance:
    [I_corr, Q_corr]^T = H * [I, Q]^T
    where H = [[1, 0], [-tan(phi), g / cos(phi)]]
    """
    x = _as_complex_array(iq_data, name="iq_data")
    g = float(gain_ratio)
    phi = float(phase_imbalance_rad)

    cos_phi = float(np.cos(phi))
    if abs(cos_phi) < 1e-10 or g < 1e-10:
        return x.copy()

    i_in = x.real
    q_in = x.imag

    i_corr = i_in.copy()
    q_corr = -np.tan(phi) * i_in + (g / cos_phi) * q_in

    return np.ascontiguousarray(i_corr + 1j * q_corr)


def apply_calibration(
    iq_data: Any,
    calibration_result: CalibrationEstimationResult,
    sample_rate: float = 1.0,
    enabled_corrections: list[str] | None = None,
) -> CorrectionResult:
    """
    Executes the recommended full-chain IQ impairment correction pipeline in strict physical order:
    1. DC offset removal
    2. Timing skew compensation
    3. Gain & phase Gram-Schmidt unskewing
    4. Oscillator / CFO de-rotation
    """
    x = _as_complex_array(iq_data, name="iq_data")
    _validate_finite_iq(x, name="iq_data")
    fs = _validate_sample_rate(sample_rate)

    def is_enabled(stage: str) -> bool:
        if enabled_corrections is None:
            return True
        return any(stage.lower() in s.lower() for s in enabled_corrections)

    curr_iq = x.copy()
    applied: list[str] = []

    # 1. DC Removal
    if is_enabled("dc") and calibration_result.dc_offset is not None and getattr(calibration_result.dc_offset, "valid", False):
        dc_val = getattr(calibration_result.dc_offset, "dc_complex", 0.0j)
        if abs(dc_val) > 1e-15:
            curr_iq = apply_dc_correction(curr_iq, dc_val)
            applied.append("dc_removal")

    # 2. Timing Skew Compensation
    if is_enabled("timing_skew") and calibration_result.iq_timing_skew is not None and getattr(calibration_result.iq_timing_skew, "valid", False):
        skew_samples = getattr(calibration_result.iq_timing_skew, "timing_skew_samples", 0.0)
        if abs(skew_samples) > 1e-6:
            curr_iq = apply_timing_skew_correction(curr_iq, skew_samples, fs)
            applied.append("timing_skew_alignment")

    # 3. Gain & Phase / Quadrature Correction (Gram-Schmidt Matrix)
    if is_enabled("gain_phase") or is_enabled("gain") or is_enabled("phase"):
        gain_res = calibration_result.iq_gain_imbalance
        phase_res = calibration_result.iq_phase_imbalance
        if (
            gain_res is not None
            and phase_res is not None
            and getattr(gain_res, "valid", False)
            and getattr(phase_res, "valid", False)
        ):
            g = getattr(gain_res, "gain_ratio", 1.0)
            phi = getattr(phase_res, "phase_imbalance_rad", 0.0)
            if abs(g - 1.0) > 1e-6 or abs(phi) > 1e-6:
                curr_iq = apply_iq_imbalance_correction(curr_iq, g, phi)
                applied.append("iq_gain_phase_unskew")

    # 4. Oscillator / CFO De-rotation
    if is_enabled("cfo") and calibration_result.oscillator_offset is not None and getattr(calibration_result.oscillator_offset, "valid", False):
        cfo = getattr(calibration_result.oscillator_offset, "oscillator_offset_hz", 0.0)
        if abs(cfo) > 1e-9:
            curr_iq = apply_cfo_correction(curr_iq, cfo, fs)
            applied.append("cfo_derotation")

    # Measure post-correction residual metrics
    res_dc_c = complex(np.mean(curr_iq))
    res_dc_pwr = float(np.abs(res_dc_c) ** 2)
    p_i = float(np.mean((curr_iq.real - np.mean(curr_iq.real)) ** 2))
    p_q = float(np.mean((curr_iq.imag - np.mean(curr_iq.imag)) ** 2))
    res_gain_db = float(10.0 * np.log10((p_i + 1e-20) / (p_q + 1e-20))) if (p_i > 1e-20 and p_q > 1e-20) else 0.0

    std_i = float(np.std(curr_iq.real))
    std_q = float(np.std(curr_iq.imag))
    cov_iq = float(np.mean((curr_iq.real - np.mean(curr_iq.real)) * (curr_iq.imag - np.mean(curr_iq.imag))))
    res_phase_deg = float(np.degrees(np.arcsin(np.clip(cov_iq / (std_i * std_q + 1e-20), -1.0, 1.0)))) if (std_i > 1e-20 and std_q > 1e-20) else 0.0

    meta: dict[str, Any] = {
        "input_sample_count": len(x),
        "output_sample_count": len(curr_iq),
        "initial_power": float(np.mean(np.abs(x) ** 2)),
        "corrected_power": float(np.mean(np.abs(curr_iq) ** 2)),
        "residual_dc_offset_complex": res_dc_c,
    }

    # Evaluate post-correction IRR if desired frequency is available
    irr_res = calibration_result.image_rejection
    if irr_res is not None and getattr(irr_res, "desired_frequency_hz", None) is not None:
        freqs, psd = _compute_psd_welch(curr_iq, fs, nfft=min(2048, len(curr_iq)))
        df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        dc_mask = np.abs(freqs) < (3.0 * df)
        masked_psd = psd.copy()
        masked_psd[dc_mask] = 0.0
        peak_idx = int(np.argmax(masked_psd))
        f_des = float(freqs[peak_idx])
        f_img = -f_des
        bw = float(max(5.0 * df, 0.05 * abs(f_des) if abs(f_des) > 1e-3 else 0.05 * fs))
        des_mask = (freqs >= (f_des - bw / 2.0)) & (freqs <= (f_des + bw / 2.0))
        img_mask = (freqs >= (f_img - bw / 2.0)) & (freqs <= (f_img + bw / 2.0))
        p_des = float(np.sum(psd[des_mask]) * df) if np.any(des_mask) else 1e-20
        p_img = float(np.sum(psd[img_mask]) * df) if np.any(img_mask) else 1e-20
        post_irr_db = float(10.0 * np.log10((p_des + 1e-20) / (p_img + 1e-20)))
        meta["post_correction_irr_db"] = post_irr_db

    return CorrectionResult(
        corrected_iq=curr_iq,
        corrections_applied=applied,
        residual_dc_power=res_dc_pwr,
        residual_gain_imbalance_db=res_gain_db,
        residual_phase_imbalance_deg=res_phase_deg,
        residual_cfo_hz=None,
        metadata=meta,
    )


class CalibrationEstimationAnalyzer:
    """
    Calibration Estimation and Consolidation Engine.

    Aggregates module results across all 8 impairment categories, derives the 2x2
    Gram-Schmidt unskewing matrix, evaluates overall system validity and confidence,
    and applies physical-order calibration corrections to IQ data.
    """

    def __init__(self) -> None:
        self.result: CalibrationEstimationResult | None = None

    def compute(
        self,
        calibration_input: CalibrationInput | Any = None,
        sample_rate: float | None = None,
        expected_frequency_hz: float | None = None,
        dc_offset: Any | None = None,
        iq_gain_imbalance: Any | None = None,
        iq_phase_imbalance: Any | None = None,
        quadrature_error: Any | None = None,
        iq_timing_skew: Any | None = None,
        image_rejection: Any | None = None,
        carrier_leakage: Any | None = None,
        oscillator_offset: Any | None = None,
        calibration_tone_frequency_hz: float | None = None,
        reference_iq: Any | None = None,
        expected_frequency: float | None = None,
        calibration_tone_frequency: float | None = None,
        iq: Any = None,
        warnings: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> CalibrationEstimationResult:
        """
        Synthesizes consolidated CalibrationEstimationResult from computed individual impairment estimates.
        """
        meta = metadata.copy() if metadata is not None else {}
        warn_list = list(warnings) if warnings is not None else []

        # Derive 2x2 Gram-Schmidt correction transformation matrix H
        # [I_corr, Q_corr]^T = H * [I, Q]^T
        # H = [[1, 0], [-tan(phi), g / cos(phi)]]
        if (
            iq_gain_imbalance is not None
            and iq_phase_imbalance is not None
            and getattr(iq_gain_imbalance, "valid", False)
            and getattr(iq_phase_imbalance, "valid", False)
        ):
            g = float(getattr(iq_gain_imbalance, "gain_ratio", 1.0))
            phi = float(getattr(iq_phase_imbalance, "phase_imbalance_rad", 0.0))
            cos_phi = float(np.cos(phi))
            if abs(cos_phi) > 1e-10 and g > 1e-10:
                h11 = 1.0
                h12 = 0.0
                h21 = float(-np.tan(phi))
                h22 = float(g / cos_phi)
                corr_matrix = np.array([[h11, h12], [h21, h22]], dtype=np.float64)
            else:
                corr_matrix = np.eye(2, dtype=np.float64)
        else:
            corr_matrix = np.eye(2, dtype=np.float64)

        # Aggregate confidence and validity
        all_results = [
            dc_offset,
            iq_gain_imbalance,
            iq_phase_imbalance,
            quadrature_error,
            iq_timing_skew,
            image_rejection,
            carrier_leakage,
            oscillator_offset,
        ]
        active_results = [r for r in all_results if r is not None]
        confidences = [getattr(r, "confidence", 1.0) for r in active_results]
        overall_conf = float(np.mean(confidences)) if confidences else 1.0

        critical_results = [r for r in [dc_offset, iq_gain_imbalance, iq_phase_imbalance, oscillator_offset] if r is not None]
        overall_valid = all(getattr(r, "valid", True) for r in critical_results) if critical_results else True

        cal_strategy = (
            "reference_aided" if reference_iq is not None
            else ("calibration_tone" if (calibration_tone_frequency_hz is not None or calibration_tone_frequency is not None) else "blind_statistical")
        )

        res = CalibrationEstimationResult(
            dc_offset=dc_offset,
            iq_gain_imbalance=iq_gain_imbalance,
            iq_phase_imbalance=iq_phase_imbalance,
            quadrature_error=quadrature_error,
            iq_timing_skew=iq_timing_skew,
            image_rejection=image_rejection,
            carrier_leakage=carrier_leakage,
            oscillator_offset=oscillator_offset,
            correction_matrix=corr_matrix,
            overall_valid=overall_valid,
            overall_confidence=overall_conf,
            calibration_method=cal_strategy,
            warnings=warn_list,
            metadata=meta,
        )
        self.result = res
        return res

    def analyze(self, *args: Any, **kwargs: Any) -> CalibrationEstimationResult:
        """Alias for compute()."""
        return self.compute(*args, **kwargs)

    def apply_calibration(
        self,
        iq_data: Any,
        sample_rate: float = 1.0,
        calibration_result: CalibrationEstimationResult | None = None,
        enabled_corrections: list[str] | None = None,
    ) -> CorrectionResult:
        """Applies calibration corrections using latest or supplied calibration result."""
        res = calibration_result if calibration_result is not None else self.result
        if res is None:
            raise ValueError("No calibration result available to apply. Run compute() first or provide calibration_result.")
        return apply_calibration(
            iq_data=iq_data,
            calibration_result=res,
            sample_rate=sample_rate,
            enabled_corrections=enabled_corrections,
        )


CalibrationEstimator = CalibrationEstimationAnalyzer


def compute_calibration(
    calibration_input: CalibrationInput | Any = None,
    sample_rate: float | None = None,
    expected_frequency_hz: float | None = None,
    dc_offset: Any | None = None,
    iq_gain_imbalance: Any | None = None,
    iq_phase_imbalance: Any | None = None,
    quadrature_error: Any | None = None,
    iq_timing_skew: Any | None = None,
    image_rejection: Any | None = None,
    carrier_leakage: Any | None = None,
    oscillator_offset: Any | None = None,
    calibration_tone_frequency_hz: float | None = None,
    reference_iq: Any | None = None,
    expected_frequency: float | None = None,
    calibration_tone_frequency: float | None = None,
    iq: Any = None,
    warnings: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> CalibrationEstimationResult:
    """
    Functional API to compute consolidated CalibrationEstimationResult.
    """
    analyzer = CalibrationEstimationAnalyzer()
    return analyzer.compute(
        calibration_input=calibration_input,
        sample_rate=sample_rate,
        expected_frequency_hz=expected_frequency_hz,
        dc_offset=dc_offset,
        iq_gain_imbalance=iq_gain_imbalance,
        iq_phase_imbalance=iq_phase_imbalance,
        quadrature_error=quadrature_error,
        iq_timing_skew=iq_timing_skew,
        image_rejection=image_rejection,
        carrier_leakage=carrier_leakage,
        oscillator_offset=oscillator_offset,
        calibration_tone_frequency_hz=calibration_tone_frequency_hz,
        reference_iq=reference_iq,
        expected_frequency=expected_frequency,
        calibration_tone_frequency=calibration_tone_frequency,
        iq=iq,
        warnings=warnings,
        metadata=metadata,
        **kwargs,
    )


estimate_calibration = compute_calibration

__all__ = [
    "CalibrationInput",
    "CalibrationEstimationResult",
    "CalibrationResult",
    "CorrectionResult",
    "CalibrationEstimationAnalyzer",
    "CalibrationEstimator",
    "compute_calibration",
    "estimate_calibration",
    "apply_dc_correction",
    "apply_cfo_correction",
    "apply_timing_skew_correction",
    "apply_iq_imbalance_correction",
    "apply_calibration",
]


def main() -> None:
    print("=" * 60)
    print("  Calibration Estimation Module Demonstration")
    print("=" * 60)
    fs = 1e6
    n = 4096

    analyzer = CalibrationEstimationAnalyzer()
    res = analyzer.compute()
    assert res is analyzer.result

    print(f"Overall Valid: {res.overall_valid}, Confidence: {res.overall_confidence:.2f}")
    print(f"Correction Matrix H:\n{res.correction_matrix}")
    print("=" * 60)


if __name__ == "__main__":
    main()
