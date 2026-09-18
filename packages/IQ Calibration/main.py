from __future__ import annotations
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    _pkg_dir = Path(__file__).resolve().parent
    _src_dir = _pkg_dir.parent.parent
    for _p in [str(_src_dir), str(_pkg_dir.parent)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    __package__ = f"packages.{_pkg_dir.name}"

import sys
import threading
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Any
import numpy as np


# Import all 9 calibration modules via relative imports
from .DC_Offset import DCOffsetAnalyzer, DCOffsetResult, compute_dc_offset
from .IQ_Gain_Imbalance import IQGainImbalanceAnalyzer, IQGainImbalanceResult, compute_gain_imbalance
from .IQ_Phase_Imbalance import IQPhaseImbalanceAnalyzer, IQPhaseImbalanceResult, compute_phase_imbalance
from .Quadrature_Error import QuadratureErrorAnalyzer, QuadratureErrorResult, compute_quadrature_error
from .IQ_Timing_Skew import IQTimingSkewAnalyzer, IQTimingSkewResult, compute_timing_skew
from .Image_Rejection import ImageRejectionAnalyzer, ImageRejectionResult, compute_image_rejection
from .Carrier_Leakage import CarrierLeakageAnalyzer, CarrierLeakageResult, compute_carrier_leakage
from .Oscillator_Offset import OscillatorOffsetAnalyzer, OscillatorOffsetResult, compute_oscillator_offset
from .Calibration_Estimation import (
    CalibrationEstimationAnalyzer,
    CalibrationEstimationResult,
    CalibrationResult,
    CorrectionResult,
    CalibrationInput,
    apply_dc_correction,
    apply_cfo_correction,
    apply_timing_skew_correction,
    apply_iq_imbalance_correction,
    apply_calibration,
)

_CALIBRATION_INPUT_FIELDS = {f.name for f in fields(CalibrationInput)}


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


def _validate_sample_rate(fs: float) -> float:
    """Validates that sample rate is positive and finite."""
    fs_f = float(fs)
    if not np.isfinite(fs_f) or fs_f <= 0.0:
        raise ValueError(f"Sample rate must be positive and finite, got {fs}")
    return fs_f


@dataclass
class _StreamCalibrationState:
    """Maintains cross-frame temporal continuity and EWMA tracking for a single logical stream."""
    stream_key: Any = None
    running_dc: complex = 0.0 + 0.0j
    phase_accumulator_rad: float = 0.0
    last_calibration: CalibrationResult | None = None
    frames_processed: int = 0
    total_samples_processed: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class CalibrationProcessor:
    """
    Real-time streaming coordinator and multi-channel calibration processor.

    Instantiates and manages all 9 independent calibration analyzers, coordinates
    data and intermediate results across analyzers, maintains thread-safe temporal
    state (EWMA DC tracking, continuous CFO phase accumulator), and applies corrections.
    """

    def __init__(
        self,
        streaming_enabled: bool = True,
        ewma_alpha: float = 0.1,
        dc_alpha: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.streaming_enabled = bool(streaming_enabled)
        alpha = dc_alpha if dc_alpha is not None else ewma_alpha
        self.ewma_alpha = float(np.clip(alpha, 0.001, 1.0))
        self._stream_states: dict[Any, _StreamCalibrationState] = {}
        self._global_lock = threading.Lock()

        # Instantiate all 9 analyzers
        self.dc_analyzer = DCOffsetAnalyzer()
        self.gain_analyzer = IQGainImbalanceAnalyzer()
        self.phase_analyzer = IQPhaseImbalanceAnalyzer()
        self.quadrature_analyzer = QuadratureErrorAnalyzer()
        self.timing_skew_analyzer = IQTimingSkewAnalyzer()
        self.image_rejection_analyzer = ImageRejectionAnalyzer()
        self.carrier_leakage_analyzer = CarrierLeakageAnalyzer()
        self.oscillator_offset_analyzer = OscillatorOffsetAnalyzer()
        self.calibration_analyzer = CalibrationEstimationAnalyzer()

    @property
    def _states(self) -> dict[Any, _StreamCalibrationState]:
        return self._stream_states

    def reset(self, stream_id: Any = None) -> None:
        """Resets persistent state for all streams or a specific stream_id."""
        with self._global_lock:
            if stream_id is not None:
                if stream_id in self._stream_states:
                    del self._stream_states[stream_id]
            else:
                self._stream_states.clear()

    def get_stream_state(self, stream_id: Any) -> Any:
        """Returns the internal StreamState object for a stream_id, or None if absent."""
        with self._global_lock:
            return self._stream_states.get(stream_id)

    def process(
        self,
        calibration_input: CalibrationInput | Any,
        sample_rate: float | None = None,
        expected_frequency_hz: float | None = None,
        apply_live_correction: bool = True,
        enabled_modules: list[str] | None = None,
        apply_correction: bool | None = None,
        dc_method: str = "mean",
        gain_method: str = "statistical",
        phase_method: str = "statistical",
        timing_skew_method: str = "cross_spectrum_slope",
        cfo_method: str = "tone_peak",
        **kwargs: Any,
    ) -> tuple[CalibrationResult, CorrectionResult | None]:
        """
        Processes a frame of IQ data through the orchestrated calibration chain.
        """
        if apply_correction is not None:
            apply_live_correction = apply_correction

        if isinstance(calibration_input, CalibrationInput):
            x = _as_complex_array(calibration_input.iq_data, name="calibration_input.iq_data")
            fs = _validate_sample_rate(calibration_input.sample_rate)
            f_exp = calibration_input.expected_frequency_hz if expected_frequency_hz is None else expected_frequency_hz
            f_tone = calibration_input.calibration_tone_frequency_hz
            ref_iq = calibration_input.reference_iq
            stream_id = calibration_input.stream_id
            source_id = calibration_input.source_id
            channel_id = calibration_input.channel_id
            meta = calibration_input.metadata.copy()
        else:
            x = _as_complex_array(calibration_input, name="iq_data")
            fs = _validate_sample_rate(sample_rate if sample_rate is not None else 1.0)
            f_exp = expected_frequency_hz
            f_tone = kwargs.get("calibration_tone_frequency_hz") or kwargs.get("tone_frequency")
            ref_iq = kwargs.get("reference_iq")
            stream_id = kwargs.get("stream_id")
            source_id = kwargs.get("source_id")
            channel_id = kwargs.get("channel_id")
            meta = {}

        # Determine logical stream key
        if stream_id is not None:
            stream_key = stream_id
            is_anonymous = False
        elif source_id is not None or channel_id is not None:
            stream_key = (source_id, channel_id)
            is_anonymous = False
        else:
            stream_key = None
            is_anonymous = True

        state: _StreamCalibrationState | None = None
        if self.streaming_enabled and not is_anonymous:
            with self._global_lock:
                if stream_key not in self._stream_states:
                    self._stream_states[stream_key] = _StreamCalibrationState(stream_key=stream_key)
                state = self._stream_states[stream_key]

        lock_context = state.lock if state is not None else threading.Lock()

        with lock_context:
            def should_run(name: str) -> bool:
                if enabled_modules is None:
                    return True
                return any(name.lower() == m.lower() or name.lower() in m.lower() for m in enabled_modules)

            warnings: list[str] = []

            # 1. DC Offset Analyzer
            res_dc: DCOffsetResult | None = None
            if should_run("dc_offset") or should_run("dc"):
                res_dc = self.dc_analyzer.compute(
                    iq=x,
                    method=dc_method,
                    reference_iq=ref_iq if dc_method == "reference_aided" else None,
                )

            # 2. Oscillator Offset Analyzer (CFO)
            res_cfo: OscillatorOffsetResult | None = None
            if should_run("oscillator_offset") or should_run("cfo"):
                f_target = f_tone if f_tone is not None else f_exp
                res_cfo = self.oscillator_offset_analyzer.compute(
                    iq=x,
                    sample_rate=fs,
                    expected_frequency_hz=f_target,
                    method=cfo_method if f_target is not None else ("reference_aided" if ref_iq is not None else "autocorrelation"),
                    reference_iq=ref_iq,
                )
                if not res_cfo.valid and res_cfo.metadata.get("reason"):
                    warnings.append(f"CFO: {res_cfo.metadata['reason']}")

            # 3. I/Q Timing Skew Analyzer
            res_skew: IQTimingSkewResult | None = None
            if should_run("timing_skew") or should_run("skew"):
                res_skew = self.timing_skew_analyzer.compute(
                    iq=x,
                    sample_rate=fs,
                    method=timing_skew_method,
                    dc_offset=res_dc,
                    reference_iq=ref_iq,
                )
                if not res_skew.valid and res_skew.diagnostic:
                    warnings.append(f"Timing Skew: {res_skew.diagnostic}")

            # 4. I/Q Gain Imbalance Analyzer
            res_gain: IQGainImbalanceResult | None = None
            if should_run("gain_imbalance") or should_run("gain"):
                res_gain = self.gain_analyzer.compute(
                    iq=x,
                    sample_rate=fs,
                    dc_offset=res_dc,
                    reference_iq=ref_iq,
                    method=gain_method,
                )
                if not res_gain.valid and res_gain.metadata.get("reason"):
                    warnings.append(f"Gain Imbalance: {res_gain.metadata['reason']}")

            # 5. I/Q Phase Imbalance Analyzer
            res_phase: IQPhaseImbalanceResult | None = None
            if should_run("phase_imbalance") or should_run("phase"):
                res_phase = self.phase_analyzer.compute(
                    iq=x,
                    sample_rate=fs,
                    dc_offset=res_dc,
                    gain_imbalance=res_gain,
                    reference_iq=ref_iq,
                    method=phase_method,
                )
                if not res_phase.valid and res_phase.metadata.get("reason"):
                    warnings.append(f"Phase Imbalance: {res_phase.metadata['reason']}")

            # 6. Quadrature Error Analyzer
            res_quad: QuadratureErrorResult | None = None
            if should_run("quadrature_error") or should_run("quadrature"):
                res_quad = self.quadrature_analyzer.compute(
                    iq=x,
                    dc_offset=res_dc,
                )

            # 7. Image Rejection Ratio Analyzer
            res_irr: ImageRejectionResult | None = None
            if should_run("image_rejection") or should_run("irr"):
                f_sig = f_tone if f_tone is not None else f_exp
                res_irr = self.image_rejection_analyzer.compute(
                    iq=x,
                    sample_rate=fs,
                    expected_signal_frequency_hz=f_sig,
                    oscillator_offset=res_cfo,
                    gain_imbalance=res_gain,
                    phase_imbalance=res_phase,
                )
                if not res_irr.valid and res_irr.metadata.get("reason"):
                    warnings.append(f"Image Rejection: {res_irr.metadata['reason']}")

            # 8. Carrier Leakage Analyzer
            res_leakage: CarrierLeakageResult | None = None
            if should_run("carrier_leakage") or should_run("leakage"):
                res_leakage = self.carrier_leakage_analyzer.compute(
                    iq=x,
                    sample_rate=fs,
                    dc_offset=res_dc,
                )

            # 9. Calibration Estimation Analyzer (Consolidation)
            cal_res = self.calibration_analyzer.compute(
                calibration_input=calibration_input,
                sample_rate=fs,
                expected_frequency_hz=f_exp,
                dc_offset=res_dc,
                iq_gain_imbalance=res_gain,
                iq_phase_imbalance=res_phase,
                quadrature_error=res_quad,
                iq_timing_skew=res_skew,
                image_rejection=res_irr,
                carrier_leakage=res_leakage,
                oscillator_offset=res_cfo,
                calibration_tone_frequency_hz=f_tone,
                reference_iq=ref_iq,
                warnings=warnings,
                metadata=meta,
            )

            # Update streaming EWMA DC state if tracking
            if state is not None and cal_res.dc_offset is not None and cal_res.dc_offset.valid:
                frame_dc = cal_res.dc_offset.dc_complex
                if state.frames_processed == 0:
                    state.running_dc = frame_dc
                else:
                    state.running_dc = (1.0 - self.ewma_alpha) * state.running_dc + self.ewma_alpha * frame_dc

            # Apply live streaming correction with continuous phase accumulator
            corr_res: CorrectionResult | None = None
            if apply_live_correction:
                curr_iq = x.copy()
                applied: list[str] = []

                # A. DC Removal (using running EWMA DC if stateful, else frame DC)
                dc_to_remove = state.running_dc if state is not None else (
                    cal_res.dc_offset.dc_complex if cal_res.dc_offset else 0.0j
                )
                if abs(dc_to_remove) > 1e-15:
                    curr_iq = apply_dc_correction(curr_iq, dc_to_remove)
                    applied.append("streaming_ewma_dc_removal" if state is not None else "dc_removal")

                # B. Timing Skew Compensation
                if cal_res.iq_timing_skew is not None and cal_res.iq_timing_skew.valid:
                    skew_s = cal_res.iq_timing_skew.timing_skew_samples
                    if abs(skew_s) > 1e-6:
                        curr_iq = apply_timing_skew_correction(curr_iq, skew_s, fs)
                        applied.append("timing_skew_alignment")

                # C. Gain & Phase Gram-Schmidt Matrix
                if (
                    cal_res.iq_gain_imbalance is not None
                    and cal_res.iq_phase_imbalance is not None
                    and cal_res.iq_gain_imbalance.valid
                    and cal_res.iq_phase_imbalance.valid
                ):
                    g = cal_res.iq_gain_imbalance.gain_ratio
                    phi = cal_res.iq_phase_imbalance.phase_imbalance_rad
                    if abs(g - 1.0) > 1e-6 or abs(phi) > 1e-6:
                        curr_iq = apply_iq_imbalance_correction(curr_iq, g, phi)
                        applied.append("iq_gain_phase_unskew")

                # D. Continuous CFO de-rotation across frame boundaries
                if cal_res.oscillator_offset is not None and cal_res.oscillator_offset.valid:
                    cfo = cal_res.oscillator_offset.oscillator_offset_hz
                    if abs(cfo) > 1e-9:
                        init_phase = state.phase_accumulator_rad if state is not None else 0.0
                        curr_iq = apply_cfo_correction(curr_iq, cfo, fs, initial_phase_rad=init_phase)
                        applied.append("streaming_cfo_derotation")

                        if state is not None:
                            delta_phase = 2.0 * np.pi * float(cfo) * (len(x) / fs)
                            state.phase_accumulator_rad = float((state.phase_accumulator_rad + delta_phase) % (2.0 * np.pi))

                meta_corr: dict[str, Any] = {
                    "stream_key": stream_key,
                    "frames_processed": state.frames_processed if state else 1,
                    "residual_dc_offset_complex": complex(np.mean(curr_iq)),
                }

                corr_res = CorrectionResult(
                    corrected_iq=curr_iq,
                    corrections_applied=applied,
                    residual_dc_power=float(np.abs(np.mean(curr_iq)) ** 2),
                    metadata=meta_corr,
                )

            # Update stream tracking
            if state is not None:
                state.last_calibration = cal_res
                state.frames_processed += 1
                state.total_samples_processed += len(x)

            return cal_res, corr_res


def calibrate_iq(
    calibration_input: CalibrationInput | dict[str, Any] | None = None,
    enabled_modules: list[str] | None = None,
    iq: np.ndarray | None = None,
    sample_rate: float = 1.0,
    apply_corrections: bool = True,
    **kwargs: Any,
) -> tuple[CalibrationResult, CorrectionResult | None]:
    """
    Stateless batch / one-shot entry point for IQ Calibration.
    """
    if calibration_input is None:
        inp = CalibrationInput(iq_data=iq, sample_rate=sample_rate, **kwargs)
    elif isinstance(calibration_input, dict):
        merged = {**calibration_input, **kwargs}
        inp = CalibrationInput(**merged)
    else:
        inp = calibration_input

    processor = CalibrationProcessor()
    return processor.process(inp, enabled_modules=enabled_modules, apply_correction=apply_corrections)



def generate_impaired_iq(
    clean_signal: np.ndarray,
    sample_rate: float,
    dc_offset: complex = 0.0 + 0.0j,
    gain_imbalance_db: float = 0.0,
    phase_imbalance_deg: float = 0.0,
    timing_skew_samples: float = 0.0,
    cfo_hz: float = 0.0,
    snr_db: float | None = None,
    dc_i: float | None = None,
    dc_q: float | None = None,
    noise_snr_db: float | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Generates realistic impaired IQ signals for testing and demonstration.
    """
    x = np.asarray(clean_signal, dtype=np.complex128).ravel()
    fs = float(sample_rate)
    n = len(x)

    if dc_i is not None or dc_q is not None:
        dc_val = complex(dc_i or 0.0, dc_q or 0.0)
    else:
        dc_val = dc_offset

    noise_db = noise_snr_db if noise_snr_db is not None else snr_db

    # 1. Carrier Frequency Offset (RF carrier drift prior to baseband downconversion)
    if abs(cfo_hz) > 1e-12:
        t = np.arange(n) / fs
        x_cfo = x * np.exp(1j * 2.0 * np.pi * cfo_hz * t)
    else:
        x_cfo = x

    # 2. Quadrature Downconversion with Gain & Phase Imbalance
    g = 10.0 ** (gain_imbalance_db / 20.0)
    phi = np.radians(phase_imbalance_deg)

    i_in = x_cfo.real
    q_in = x_cfo.imag

    i_tx = i_in * np.sqrt(g)
    q_tx = (q_in * np.cos(phi) + i_in * np.sin(phi)) / np.sqrt(g)

    # 3. Inter-channel Timing Skew
    if abs(timing_skew_samples) > 1e-12:
        k = np.fft.fftfreq(n)
        phase_ramp = np.exp(-1j * 2.0 * np.pi * float(timing_skew_samples) * k)
        q_tx = np.real(np.fft.ifft(np.fft.fft(q_tx) * phase_ramp))

    impaired = i_tx + 1j * q_tx

    # 4. DC Offset
    impaired = impaired + dc_val

    # 5. AWGN
    if noise_db is not None:
        p_sig = float(np.mean(np.abs(impaired) ** 2))
        p_noise = p_sig / (10.0 ** (noise_db / 10.0))
        noise = (np.random.randn(n) + 1j * np.random.randn(n)) * np.sqrt(p_noise / 2.0)
        impaired = impaired + noise


    ground_truth = {
        "dc_offset": dc_val,
        "gain_imbalance_db": gain_imbalance_db,
        "phase_imbalance_deg": phase_imbalance_deg,
        "timing_skew_samples": timing_skew_samples,
        "cfo_hz": cfo_hz,
        "snr_db": noise_db,
    }

    return impaired, ground_truth


__all__ = [
    "calibrate_iq",
    "CalibrationProcessor",
    "apply_calibration",
    "CalibrationInput",
    "CalibrationResult",
    "CorrectionResult",
    "generate_impaired_iq",
    "DCOffsetAnalyzer",
    "IQGainImbalanceAnalyzer",
    "IQPhaseImbalanceAnalyzer",
    "QuadratureErrorAnalyzer",
    "IQTimingSkewAnalyzer",
    "ImageRejectionAnalyzer",
    "CarrierLeakageAnalyzer",
    "OscillatorOffsetAnalyzer",
    "CalibrationEstimationAnalyzer",
]


def main() -> None:
    print("=" * 70)
    print("  IQ Calibration: Standalone Pipeline & Demonstration")
    print("=" * 70)

    fs = 10e6
    n = 32768
    t = np.arange(n) / fs
    f_tone = 1.25e6

    ideal = np.exp(1j * 2 * np.pi * f_tone * t)

    impaired_iq, ground_truth = generate_impaired_iq(
        ideal,
        sample_rate=fs,
        dc_i=0.15,
        dc_q=0.10,
        gain_imbalance_db=1.5,
        phase_imbalance_deg=4.0,
        cfo_hz=5000.0,
        noise_snr_db=40.0,
    )

    print(f"Sampling Rate: {fs / 1e6:.1f} MHz, Record Length: {n} samples")
    print("Applied Ground Truth Impairments:")
    for k, v in ground_truth.items():
        print(f"  - {k}: {v}")

    cal_input = CalibrationInput(
        iq=impaired_iq,
        sample_rate=fs,
        expected_frequency=f_tone,
        source_id="SDR_FrontEnd_Alpha",
        channel_id=0,
    )

    cal_result, corr_result = calibrate_iq(cal_input, apply_corrections=True)

    print("\n" + "-" * 70)
    print("Estimated Calibration Results:")
    print("-" * 70)
    print(f"Overall Valid: {cal_result.valid}")
    if cal_result.dc_offset:
        print(f"Estimated DC Offset: I={cal_result.dc_offset.dc_i:.5f}, Q={cal_result.dc_offset.dc_q:.5f}")
    if cal_result.gain_imbalance:
        print(f"Estimated Gain Imbalance: {cal_result.gain_imbalance.gain_imbalance_db:.3f} dB (ratio: {cal_result.gain_imbalance.gain_ratio:.4f})")
    if cal_result.phase_imbalance:
        print(f"Estimated Phase Imbalance: {cal_result.phase_imbalance.phase_imbalance_deg:.3f} deg")
    if cal_result.quadrature_error:
        print(f"Estimated Quadrature Error: {cal_result.quadrature_error.error_deg:.3f} deg (corr: {cal_result.quadrature_error.correlation_coefficient:.4f})")
    if cal_result.oscillator_offset:
        print(f"Estimated Oscillator Offset: {cal_result.oscillator_offset.frequency_offset_hz:.2f} Hz (CFO: {cal_result.oscillator_offset.cfo_hz:.2f} Hz)")
    if cal_result.image_rejection:
        print(f"Pre-correction Image Rejection: {cal_result.image_rejection.image_rejection_db:.2f} dB")
    if cal_result.carrier_leakage:
        print(f"Pre-correction Carrier Leakage: {cal_result.carrier_leakage.carrier_leakage_dbc:.2f} dBc")

    if corr_result is not None:
        print("\n" + "-" * 70)
        print("Post-Correction Performance:")
        print("-" * 70)
        print(f"Post-correction DC Offset: {corr_result.residual_dc_offset_complex}")
        if corr_result.post_correction_irr_db is not None:
            print(f"Post-correction Image Rejection: {corr_result.post_correction_irr_db:.2f} dB")

    print("\n" + "=" * 70)
    print("  Demonstration Completed Successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
