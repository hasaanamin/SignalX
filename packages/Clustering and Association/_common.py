from __future__ import annotations
from typing import Any
import math
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import linear_sum_assignment
import importlib

from .base import IQEvent, PulseClusterResult


def _get_pulse_detector() -> Any:
    """Dynamic accessor for canonical Pulse Detector package."""
    return importlib.import_module("packages.Pulse_Detector")


def _get_amplitude() -> Any:
    """Dynamic accessor for canonical Amplitude package."""
    return importlib.import_module("packages.Amplitude")


def _get_spectral() -> Any:
    """Dynamic accessor for canonical Spectral package."""
    return importlib.import_module("packages.Spectral")


def _get_frequency() -> Any:
    """Dynamic accessor for canonical Frequency package."""
    return importlib.import_module("packages.Frequency")


def _get_noise() -> Any:
    """Dynamic accessor for canonical Noise package."""
    return importlib.import_module("packages.Noise")


FEATURE_NAMES: tuple[str, ...] = (
    "spectral_centroid_hz",
    "bandwidth_hz",
    "duration_sec",
    "mean_power_db",
    "spectral_entropy",
    "spectral_flatness",
    "chirp_slope_hz_per_sec",
)

DEFAULT_FEATURE_WEIGHTS: dict[str, float] = {
    "spectral_centroid_hz": 2.0,
    "bandwidth_hz": 1.5,
    "duration_sec": 2.0,
    "mean_power_db": 0.2,
    "spectral_entropy": 0.8,
    "spectral_flatness": 0.8,
    "chirp_slope_hz_per_sec": 2.0,
    "carrier_freq_hz": 2.0,
    "snr_db": 0.2,
    "peak_power_db": 0.2,
}


def _as_iq_array(data: Any) -> np.ndarray:
    """
    Standardizes input into a 1D complex128 NumPy array.
    Supports complex ndarray, tuple/list of (I, Q), canonical dictionary, or structured arrays.
    Rejects NaN / Inf values and invalid non-array types.
    """
    if data is None:
        raise ValueError("Input IQ data cannot be None.")

    if hasattr(data, "iq_samples"):
        return _as_iq_array(data.iq_samples)

    if isinstance(data, dict):
        if "iq" in data and data["iq"] is not None:
            return _as_iq_array(data["iq"])
        if "I" in data and "Q" in data:
            i_arr = np.asarray(data["I"], dtype=np.float64).ravel()
            q_arr = np.asarray(data["Q"], dtype=np.float64).ravel()
            min_len = min(len(i_arr), len(q_arr))
            arr = i_arr[:min_len] + 1j * q_arr[:min_len]
            _validate_finite(arr)
            return arr.astype(np.complex128)
        if "real" in data and "imag" in data:
            r_arr = np.asarray(data["real"], dtype=np.float64).ravel()
            im_arr = np.asarray(data["imag"], dtype=np.float64).ravel()
            min_len = min(len(r_arr), len(im_arr))
            arr = r_arr[:min_len] + 1j * im_arr[:min_len]
            _validate_finite(arr)
            return arr.astype(np.complex128)
        raise ValueError(f"Dictionary missing recognized IQ keys ('iq', 'I'/'Q', 'real'/'imag'). Found: {list(data.keys())}")

    if isinstance(data, (tuple, list)):
        if len(data) == 2 and not np.iscomplexobj(data[0]):
            try:
                i_arr = np.asarray(data[0], dtype=np.float64).ravel()
                q_arr = np.asarray(data[1], dtype=np.float64).ravel()
                min_len = min(len(i_arr), len(q_arr))
                arr = i_arr[:min_len] + 1j * q_arr[:min_len]
                _validate_finite(arr)
                return arr.astype(np.complex128)
            except Exception:
                pass
        arr = np.asarray(data)
        if arr.dtype.kind == "V":
            names = arr.dtype.names or ()
            if "I" in names and "Q" in names:
                arr = (arr["I"].astype(np.float64) + 1j * arr["Q"].astype(np.float64)).ravel()
                _validate_finite(arr)
                return arr.astype(np.complex128)
        if np.iscomplexobj(arr):
            _validate_finite(arr)
            return arr.astype(np.complex128).ravel()
        if arr.ndim == 2 and arr.shape[1] == 2:
            arr = (arr[:, 0].astype(np.float64) + 1j * arr[:, 1].astype(np.float64)).ravel()
            _validate_finite(arr)
            return arr.astype(np.complex128)
        if arr.ndim == 2 and arr.shape[0] == 2:
            arr = (arr[0, :].astype(np.float64) + 1j * arr[1, :].astype(np.float64)).ravel()
            _validate_finite(arr)
            return arr.astype(np.complex128)
        _validate_finite(arr)
        return arr.astype(np.complex128).ravel()

    if isinstance(data, np.ndarray):
        if data.dtype.kind == "V":
            names = data.dtype.names or ()
            if "I" in names and "Q" in names:
                arr = (data["I"].astype(np.float64) + 1j * data["Q"].astype(np.float64)).ravel()
                _validate_finite(arr)
                return arr.astype(np.complex128)
        if np.iscomplexobj(data):
            _validate_finite(data)
            return data.astype(np.complex128).ravel()
        if data.ndim == 2 and data.shape[1] == 2:
            arr = (data[:, 0].astype(np.float64) + 1j * data[:, 1].astype(np.float64)).ravel()
            _validate_finite(arr)
            return arr.astype(np.complex128)
        if data.ndim == 2 and data.shape[0] == 2:
            arr = (data[0, :].astype(np.float64) + 1j * data[1, :].astype(np.float64)).ravel()
            _validate_finite(arr)
            return arr.astype(np.complex128)
        _validate_finite(data)
        return data.astype(np.complex128).ravel()

    raise TypeError(f"Unsupported IQ data type '{type(data).__name__}'.")


def _validate_finite(arr: np.ndarray) -> None:
    """Validates that array contains no NaN or Inf values."""
    if arr.size > 0 and (not np.all(np.isfinite(arr))):
        raise ValueError("Input IQ data contains NaN or Inf values.")


class RawIQAdapter:
    """
    Explicit adapter converting raw IQ into canonical IQEvent pulse records
    by orchestrating the canonical application packages:
      - Pulse Detector (pulse segmentation & boundary extraction)
      - Amplitude (pulse power)
      - Spectral (spectral centroid, bandwidth, entropy, flatness)
      - Frequency (carrier frequency, chirp slope / rate)
      - Noise (SNR, noise floor)
    """

    def __init__(self, sample_rate: float = 1.0, fs: float | None = None, **default_kwargs: Any):
        self.sample_rate = float(fs if fs is not None else sample_rate)
        self.default_kwargs = default_kwargs

    def extract(self, iq_data: Any, **kwargs: Any) -> list[IQEvent]:
        merged = {**self.default_kwargs, **kwargs}
        return self.extract_events(iq_data, sample_rate=self.sample_rate, **merged)

    def __call__(self, iq_data: Any, **kwargs: Any) -> list[IQEvent]:
        return self.extract(iq_data, **kwargs)

    @classmethod
    def extract_events(
        cls,
        iq_data: Any,
        sample_rate: float = 1.0,
        threshold_k: float = 3.5,
        min_event_samples: int = 16,
        merge_gap_samples: int = 8,
        return_iq_segments: bool = False,
        fs: float | None = None,
        **kwargs: Any,
    ) -> list[IQEvent]:
        effective_fs = float(fs if fs is not None else sample_rate)
        effective_fs = effective_fs if effective_fs > 0 else 1.0
        iq = _as_iq_array(iq_data)
        n_pts = len(iq)
        if n_pts == 0:
            return []

        pd_pkg = _get_pulse_detector()
        th_window = max(101, min(n_pts if n_pts % 2 == 1 else n_pts - 1, 2001))
        k_high = float(threshold_k) if threshold_k > 0 else 3.5
        k_low = max(k_high * 0.6, 1.5)
        pw_min = max(int(min_event_samples), 1)
        edge_win = max(int(merge_gap_samples), 2)

        try:
            pulse_candidates = pd_pkg.detect_pulse_candidates(
                iq_data=iq,
                sample_rate=effective_fs,
                threshold_window=th_window,
                k_high=k_high,
                k_low=k_low,
                pw_min_samples=pw_min,
                edge_search_window=edge_win,
                include_rejected=True,
            )
        except Exception:
            pulse_candidates = []

        events: list[IQEvent] = []
        if isinstance(pulse_candidates, list) and len(pulse_candidates) > 0:
            for pc in pulse_candidates:
                if getattr(pc, "rejection_reason", "") in ("INVALID_EDGE_ORDER", "PW_TOO_SHORT"):
                    continue
                if pc.pulse_width_samples < pw_min:
                    continue
                ev = cls._pulse_candidate_to_event(pc, fs=effective_fs, return_iq_segments=return_iq_segments, full_iq=iq)
                events.append(ev)

        return events

    @classmethod
    def _pulse_candidate_to_event(
        cls,
        candidate: Any,
        fs: float,
        return_iq_segments: bool = False,
        full_iq: np.ndarray | None = None,
    ) -> IQEvent:
        """Converts a single PulseCandidate into an IQEvent using canonical packages for measurements."""
        dur_sec = float(getattr(candidate, "pulse_width_sec", 1e-4))
        toa_sec = float(getattr(candidate, "toa_sec", 0.0))
        start_s = int(getattr(candidate, "start_idx", int(toa_sec * fs)))
        stop_s = int(getattr(candidate, "stop_idx", int((toa_sec + dur_sec) * fs)))

        iq_seg = getattr(candidate, "iq_samples", None)
        has_iq = iq_seg is not None and len(iq_seg) > 0 and np.iscomplexobj(iq_seg)

        # Power: prefer existing candidate power or Amplitude package
        if hasattr(candidate, "mean_power") and candidate.mean_power is not None:
            mean_p = float(candidate.mean_power)
            pwr_db = 10.0 * math.log10(max(mean_p, 1e-300))
        elif has_iq:
            amp_pkg = _get_amplitude()
            mean_p = float(amp_pkg.average_power(iq_seg))
            pwr_db = 10.0 * math.log10(max(mean_p, 1e-300))
        else:
            mean_p = 1.0
            pwr_db = 0.0

        # Boundary pulse adjustment
        if full_iq is not None and len(full_iq) > 0:
            pwr_0 = float(np.abs(full_iq[0]) ** 2)
            pwr_end = float(np.abs(full_iq[-1]) ** 2)
            ref_pwr = max(mean_p, 1e-6)
            if start_s <= 16 and pwr_0 >= 0.20 * ref_pwr:
                start_s = 0
                toa_sec = 0.0
                dur_sec = float((stop_s - start_s + 1) / fs)
            if stop_s >= len(full_iq) - 16 and pwr_end >= 0.20 * ref_pwr:
                stop_s = len(full_iq) - 1
                dur_sec = float((stop_s - start_s + 1) / fs)
            if has_iq and (start_s == 0 or stop_s == len(full_iq) - 1):
                iq_seg = full_iq[start_s : stop_s + 1]

        # Spectral & Frequency features from canonical packages
        if has_iq:
            spec_pkg = _get_spectral()
            freq_pkg = _get_frequency()

            centroid_hz = float(spec_pkg.spectral_centroid(iq_seg, sample_rate=fs))
            bw_hz = float(spec_pkg.bandwidth(iq_seg, sample_rate=fs))
            s_entropy = float(spec_pkg.spectral_entropy(iq_seg, sample_rate=fs, normalize=True))
            s_flatness = float(spec_pkg.spectral_flatness(iq_seg, sample_rate=fs))

            carrier_fc = float(freq_pkg.center_frequency(iq_seg, sample_rate=fs))
            slope_hz = float(freq_pkg.frequency_slope(iq_seg, sample_rate=fs))
            slope_valid = bool(abs(slope_hz) > 1.0)
        else:
            centroid_hz = float(getattr(candidate, "spectral_centroid_hz", getattr(candidate, "frequency_hz", 0.0)))
            carrier_fc = float(getattr(candidate, "carrier_freq_hz", centroid_hz))
            bw_hz = float(getattr(candidate, "bandwidth_hz", 1e5))
            s_entropy = float(getattr(candidate, "spectral_entropy", 0.5))
            s_flatness = float(getattr(candidate, "spectral_flatness", 0.5))
            slope_hz = float(getattr(candidate, "chirp_slope_hz_per_sec", 0.0))
            slope_valid = bool(abs(slope_hz) > 0.0)

        snr_db = float(getattr(candidate, "snr_db", 0.0)) if hasattr(candidate, "snr_db") else None
        peak_pwr_db = float(getattr(candidate, "peak_power_db", 0.0)) if hasattr(candidate, "peak_power_db") else None

        return IQEvent(
            start_sample=start_s,
            end_sample=stop_s,
            start_time_sec=toa_sec,
            end_time_sec=toa_sec + dur_sec,
            duration_sec=dur_sec,
            mean_power_db=pwr_db,
            spectral_centroid_hz=centroid_hz,
            bandwidth_hz=bw_hz,
            spectral_entropy=s_entropy,
            spectral_flatness=s_flatness,
            chirp_slope_hz_per_sec=slope_hz,
            chirp_slope_valid=slope_valid,
            carrier_freq_hz=carrier_fc,
            peak_power_db=peak_pwr_db,
            snr_db=snr_db,
            raw_iq=iq_seg if return_iq_segments else None,
        )


def detect_iq_events(
    iq: np.ndarray,
    sample_rate: float,
    threshold_k: float = 3.5,
    min_event_samples: int = 16,
    merge_gap_samples: int = 8,
) -> list[tuple[int, int]]:
    """
    Finds pulse candidate boundary regions above threshold using canonical Pulse Detector.
    Returns: list of (start_sample, end_sample) tuples.
    """
    fs = float(sample_rate) if sample_rate > 0 else 1.0
    iq_arr = _as_iq_array(iq)
    if len(iq_arr) == 0:
        return []

    events = RawIQAdapter.extract_events(
        iq_data=iq_arr,
        sample_rate=fs,
        threshold_k=threshold_k,
        min_event_samples=min_event_samples,
        merge_gap_samples=merge_gap_samples,
    )
    return [(ev.start_sample, ev.end_sample) for ev in events]


def extract_event_features(
    iq_segment: np.ndarray,
    sample_rate: float,
) -> tuple[float, float, float, float, float, float, float, bool]:
    """
    Extracts RF/waveform features from an isolated complex IQ event using canonical analysis packages:
      - Pulse width / duration from segment length
      - Amplitude package for mean_power_db
      - Spectral package for spectral_centroid_hz, bandwidth_hz, spectral_entropy, spectral_flatness
      - Frequency package for chirp_slope_hz_per_sec
    """
    n_pts = len(iq_segment)
    if n_pts == 0:
        return 0.0, -300.0, 0.0, 0.0, 0.0, 0.0, 0.0, False

    fs = float(sample_rate) if sample_rate > 0 else 1.0
    duration_sec = n_pts / fs

    amp_pkg = _get_amplitude()
    spec_pkg = _get_spectral()
    freq_pkg = _get_frequency()

    mean_power = float(amp_pkg.average_power(iq_segment))
    mean_power_db = 10.0 * math.log10(max(mean_power, 1e-300))

    spectral_centroid_hz = float(spec_pkg.spectral_centroid(iq_segment, sample_rate=fs))
    bandwidth_hz = float(spec_pkg.bandwidth(iq_segment, sample_rate=fs))
    spectral_entropy = float(spec_pkg.spectral_entropy(iq_segment, sample_rate=fs, normalize=True))
    spectral_flatness = float(spec_pkg.spectral_flatness(iq_segment, sample_rate=fs))

    chirp_slope_hz_per_sec = float(freq_pkg.frequency_slope(iq_segment, sample_rate=fs))
    chirp_slope_valid = bool(abs(chirp_slope_hz_per_sec) > 1.0)

    return (
        duration_sec,
        mean_power_db,
        spectral_centroid_hz,
        bandwidth_hz,
        spectral_entropy,
        spectral_flatness,
        chirp_slope_hz_per_sec,
        chirp_slope_valid,
    )


def extract_events_from_input(
    data: Any,
    sample_rate: float = 1.0,
    threshold_k: float = 3.5,
    min_event_samples: int = 16,
    merge_gap_samples: int = 8,
    return_iq_segments: bool = False,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    fs: float | None = None,
    **kwargs: Any,
) -> tuple[list[IQEvent], np.ndarray, tuple[str, ...]]:
    """
    Polymorphic input parser converting characterized pulse records or raw IQ into a uniform
    list of IQEvent and 2D feature matrix.
    Supports:
      1. List of IQEvent objects (direct reuse, zero DSP recomputation)
      2. List of PulseCandidate objects (from Pulse Detector)
      3. List of dictionaries with PDW/pulse attributes
      4. 2D NumPy array of precomputed features (N x D)
      5. Raw complex IQ (adapts via RawIQAdapter delegating to Pulse Detector, Amplitude, Spectral, Frequency)
    """
    effective_fs = float(fs if fs is not None else sample_rate)
    fs = effective_fs if effective_fs > 0 else 1.0

    if data is None:
        raise ValueError("Input data cannot be None.")

    # 1. Check if input is a 2D precomputed numerical feature array
    if isinstance(data, np.ndarray):
        if np.iscomplexobj(data):
            pass  # Raw IQ: process below via RawIQAdapter
        elif data.ndim == 2 and data.shape[0] == 2 and data.shape[1] > 16:
            pass  # Raw IQ (2, N) format: process below via RawIQAdapter
        elif data.ndim == 2:
            _validate_finite(data)
            n_rows, n_cols = data.shape
            feat_names = feature_names[:n_cols] if n_cols <= len(feature_names) else tuple(f"feat_{i}" for i in range(n_cols))
            events: list[IQEvent] = []
            for i in range(n_rows):
                dur = float(data[i, 2]) if n_cols > 2 else 1e-4
                pwr = float(data[i, 3]) if n_cols > 3 else 0.0
                fc = float(data[i, 0]) if n_cols > 0 else 0.0
                bw = float(data[i, 1]) if n_cols > 1 else 1e5
                ev = IQEvent(
                    start_sample=int(i * 1000),
                    end_sample=int(i * 1000 + dur * fs),
                    start_time_sec=float(i * (dur + 1e-3)),
                    end_time_sec=float(i * (dur + 1e-3) + dur),
                    duration_sec=dur,
                    mean_power_db=pwr,
                    spectral_centroid_hz=fc,
                    bandwidth_hz=bw,
                    spectral_entropy=0.5,
                    spectral_flatness=0.5,
                    chirp_slope_hz_per_sec=0.0,
                    chirp_slope_valid=False,
                    carrier_freq_hz=fc,
                    is_synthetic_time=True,
                )
                events.append(ev)
            return events, data.astype(np.float64), feat_names

    # 2. Check if input is a list of objects / dicts
    if isinstance(data, list) and len(data) > 0:
        first_item = data[0]

        # 2a. List of IQEvent (already characterized: ZERO DSP recomputation)
        if isinstance(first_item, IQEvent):
            events = list(data)
            raw_feats = _events_to_feature_matrix(events, feature_names)
            return events, raw_feats, feature_names

        # 2b. List of PulseCandidate (from Pulse Detector)
        if hasattr(first_item, "toa_sec") and hasattr(first_item, "pulse_width_sec"):
            events = []
            for item in data:
                has_fc = hasattr(item, "carrier_freq_hz") or hasattr(item, "frequency_hz") or hasattr(item, "spectral_centroid_hz")
                iq_seg = getattr(item, "iq_samples", None)
                if not has_fc and iq_seg is not None and len(iq_seg) > 0 and np.iscomplexobj(iq_seg):
                    ev = RawIQAdapter._pulse_candidate_to_event(item, fs=fs, return_iq_segments=return_iq_segments)
                else:
                    dur_sec = float(getattr(item, "pulse_width_sec", 1e-4))
                    mean_p = getattr(item, "mean_power", None)
                    pwr_db = 10.0 * math.log10(max(float(mean_p), 1e-300)) if mean_p is not None else 0.0
                    f_cent = getattr(item, "spectral_centroid_hz", getattr(item, "frequency_hz", None))
                    f_cent_val = float(f_cent) if f_cent is not None else 0.0
                    f_carrier = getattr(item, "carrier_freq_hz", None)
                    f_carrier_val = float(f_carrier) if f_carrier is not None else (f_cent_val if f_cent is not None else None)
                    bw_hz = float(item.bandwidth_hz) if hasattr(item, "bandwidth_hz") and item.bandwidth_hz is not None else None
                    s_ent = float(item.spectral_entropy) if hasattr(item, "spectral_entropy") and item.spectral_entropy is not None else None
                    s_flat = float(item.spectral_flatness) if hasattr(item, "spectral_flatness") and item.spectral_flatness is not None else None
                    c_slope = float(getattr(item, "chirp_slope_hz_per_sec", 0.0))
                    slope_valid = abs(c_slope) > 0.0
                    toa = float(getattr(item, "toa_sec", 0.0))
                    start_s = int(getattr(item, "start_idx", int(toa * fs)))
                    stop_s = int(getattr(item, "stop_idx", int((toa + dur_sec) * fs)))

                    ev = IQEvent(
                        start_sample=start_s,
                        end_sample=stop_s,
                        start_time_sec=toa,
                        end_time_sec=toa + dur_sec,
                        duration_sec=dur_sec,
                        mean_power_db=pwr_db,
                        spectral_centroid_hz=f_cent_val,
                        bandwidth_hz=bw_hz if bw_hz is not None else 1e5,
                        spectral_entropy=s_ent if s_ent is not None else 0.5,
                        spectral_flatness=s_flat if s_flat is not None else 0.5,
                        chirp_slope_hz_per_sec=c_slope,
                        chirp_slope_valid=slope_valid,
                        carrier_freq_hz=f_carrier_val,
                        snr_db=float(getattr(item, "snr_db", 0.0)) if hasattr(item, "snr_db") else None,
                        raw_iq=iq_seg if return_iq_segments else None,
                    )
                events.append(ev)

            raw_feats = _events_to_feature_matrix(events, feature_names)
            return events, raw_feats, feature_names

        # 2c. List of dicts (PDW / Characterized pulses: ZERO DSP recomputation)
        if isinstance(first_item, dict):
            events = []
            for i, d in enumerate(data):
                dur = float(d.get("duration_sec", d.get("pulse_width_sec", d.get("pw_sec", 1e-4))))
                is_synthetic_t = False
                if "start_time_sec" in d:
                    toa = float(d["start_time_sec"])
                elif "toa_sec" in d:
                    toa = float(d["toa_sec"])
                elif "toa" in d:
                    toa = float(d["toa"])
                else:
                    toa = float(i * 1e-3)
                    is_synthetic_t = True

                carrier_f = None
                if "carrier_freq_hz" in d or "carrier_frequency_hz" in d or "rf_freq_hz" in d or "rf_frequency_hz" in d:
                    carrier_f = float(d.get("carrier_freq_hz", d.get("carrier_frequency_hz", d.get("rf_freq_hz", d.get("rf_frequency_hz")))))

                if "spectral_centroid_hz" in d:
                    centroid_f = float(d["spectral_centroid_hz"])
                elif "frequency_hz" in d or "fc_hz" in d:
                    centroid_f = float(d.get("frequency_hz", d.get("fc_hz")))
                elif carrier_f is not None:
                    centroid_f = carrier_f
                else:
                    centroid_f = 0.0

                if carrier_f is None and ("spectral_centroid_hz" in d or "frequency_hz" in d or "fc_hz" in d):
                    carrier_f = centroid_f

                bw = float(d["bandwidth_hz"]) if "bandwidth_hz" in d else (float(d["bw_hz"]) if "bw_hz" in d else 1e5)
                pwr = float(d["mean_power_db"]) if "mean_power_db" in d else (float(d["power_db"]) if "power_db" in d else 0.0)
                ent = float(d["spectral_entropy"]) if "spectral_entropy" in d else 0.5
                flat = float(d["spectral_flatness"]) if "spectral_flatness" in d else 0.5
                slope = float(d.get("chirp_slope_hz_per_sec", d.get("chirp_rate_hz_per_sec", 0.0)))
                peak_pwr = float(d["peak_power_db"]) if "peak_power_db" in d else None
                snr_val = float(d["snr_db"]) if "snr_db" in d else None

                ev = IQEvent(
                    start_sample=int(d.get("start_sample", int(toa * fs))),
                    end_sample=int(d.get("end_sample", int((toa + dur) * fs))),
                    start_time_sec=toa,
                    end_time_sec=toa + dur,
                    duration_sec=dur,
                    mean_power_db=pwr,
                    spectral_centroid_hz=centroid_f,
                    bandwidth_hz=bw,
                    spectral_entropy=ent,
                    spectral_flatness=flat,
                    chirp_slope_hz_per_sec=slope,
                    chirp_slope_valid=abs(slope) > 0.0,
                    carrier_freq_hz=carrier_f,
                    peak_power_db=peak_pwr,
                    snr_db=snr_val,
                    is_synthetic_time=is_synthetic_t,
                )
                events.append(ev)
            raw_feats = _events_to_feature_matrix(events, feature_names)
            return events, raw_feats, feature_names

    # 3. Handle Raw IQ: Delegate to RawIQAdapter
    events = RawIQAdapter.extract_events(
        iq_data=data,
        sample_rate=fs,
        threshold_k=threshold_k,
        min_event_samples=min_event_samples,
        merge_gap_samples=merge_gap_samples,
        return_iq_segments=return_iq_segments,
    )
    raw_feats = _events_to_feature_matrix(events, feature_names)
    return events, raw_feats, feature_names


def _events_to_feature_matrix(
    events: list[IQEvent],
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> np.ndarray:
    """Extracts 2D NumPy feature array matching feature_names ordering."""
    n_events = len(events)
    n_feat = len(feature_names)
    if n_events == 0:
        return np.empty((0, n_feat), dtype=np.float64)

    mat = np.zeros((n_events, n_feat), dtype=np.float64)
    for i, ev in enumerate(events):
        for j, name in enumerate(feature_names):
            val = None
            if name in ("carrier_freq_hz", "spectral_centroid_hz"):
                if ev.carrier_freq_hz is not None:
                    val = ev.carrier_freq_hz
                elif ev.spectral_centroid_hz is not None:
                    val = ev.spectral_centroid_hz
            elif hasattr(ev, name) and getattr(ev, name) is not None:
                val = getattr(ev, name)
            elif name == "bandwidth_hz":
                val = ev.bandwidth_hz
            elif name in ("duration_sec", "pulse_width_sec"):
                val = ev.duration_sec
            elif name in ("mean_power_db", "power_db"):
                val = ev.mean_power_db
            elif name == "spectral_entropy":
                val = ev.spectral_entropy
            elif name == "spectral_flatness":
                val = ev.spectral_flatness
            elif name in ("chirp_slope_hz_per_sec", "chirp_rate_hz_per_sec"):
                val = ev.chirp_slope_hz_per_sec
            elif name in ("peak_power_db", "peak_pwr_db"):
                val = ev.peak_power_db
            elif name == "snr_db":
                val = ev.snr_db

            mat[i, j] = float(val) if val is not None else 0.0
    return mat


def normalize_features(
    raw_features: np.ndarray,
    sample_rate: float = 1.0,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> np.ndarray:
    """
    Robust feature normalization using median and MAD:
        z_tilde_{ij} = (z_{ij} - median(z_j)) / (1.4826 * MAD(z_j) + eps_j)
    """
    n_events, n_features = raw_features.shape
    if n_events == 0:
        return np.empty((0, n_features), dtype=np.float64)

    if n_events == 1:
        return np.zeros((1, n_features), dtype=np.float64)

    fs = float(sample_rate) if sample_rate > 0 else 1.0
    feature_eps = {
        "spectral_centroid_hz": max(2000.0, 0.01 * fs),
        "bandwidth_hz": max(2000.0, 0.01 * fs),
        "duration_sec": max(1e-6, 20.0 / fs),
        "mean_power_db": 5.0,
        "spectral_entropy": 0.15,
        "spectral_flatness": 0.15,
        "chirp_slope_hz_per_sec": max(1e7, 0.01 * (fs ** 2) / 1000.0),
        "carrier_freq_hz": max(2000.0, 0.01 * fs),
        "snr_db": 3.0,
        "peak_power_db": 5.0,
    }

    norm_features = np.zeros_like(raw_features, dtype=np.float64)

    for j in range(n_features):
        name = feature_names[j] if j < len(feature_names) else f"feat_{j}"
        col = raw_features[:, j]
        valid = col[np.isfinite(col)]
        if len(valid) == 0:
            norm_features[:, j] = 0.0
            continue
        med = float(np.median(valid))
        mad = float(np.median(np.abs(valid - med)))
        eps_val = feature_eps.get(name, 1e-6)
        scale = 1.4826 * mad + eps_val
        if scale <= 1e-15:
            scale = eps_val
        norm_features[:, j] = (col - med) / scale

    return norm_features


def resolve_feature_weights(
    feature_names: tuple[str, ...],
    custom_weights: dict[str, float] | np.ndarray | None = None,
) -> np.ndarray:
    """Resolves 1D feature weight array matching feature_names ordering."""
    n_features = len(feature_names)
    weights = np.ones(n_features, dtype=np.float64)

    for idx, name in enumerate(feature_names):
        if name in DEFAULT_FEATURE_WEIGHTS:
            weights[idx] = DEFAULT_FEATURE_WEIGHTS[name]

    if custom_weights is not None:
        if isinstance(custom_weights, dict):
            for idx, name in enumerate(feature_names):
                if name in custom_weights:
                    weights[idx] = float(custom_weights[name])
        elif isinstance(custom_weights, (list, tuple, np.ndarray)):
            w_arr = np.asarray(custom_weights, dtype=np.float64).ravel()
            if len(w_arr) == n_features:
                weights = w_arr
            else:
                raise ValueError(f"Length of custom weights array ({len(w_arr)}) does not match feature count ({n_features}).")

    return np.maximum(weights, 0.0)


def compute_weighted_distance_matrix(
    norm_features: np.ndarray,
    weights: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """Computes pairwise weighted distance matrix."""
    n_events = len(norm_features)
    if n_events <= 1:
        return np.zeros((n_events, n_events), dtype=np.float64)

    m = metric.lower().strip()
    sqrt_w = np.sqrt(np.maximum(weights, 0.0))
    weighted_pts = norm_features * sqrt_w[None, :]

    if m == "manhattan":
        diff = np.abs(weighted_pts[:, None, :] - weighted_pts[None, :, :])
        return np.sum(diff, axis=-1)
    elif m == "cosine":
        norms = np.linalg.norm(weighted_pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        normalized = weighted_pts / norms
        cosine_sim = np.clip(np.dot(normalized, normalized.T), -1.0, 1.0)
        return 1.0 - cosine_sim
    else:  # euclidean / default
        diff = weighted_pts[:, None, :] - weighted_pts[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        return np.sqrt(np.maximum(dist_sq, 0.0))


def perform_pulse_clustering(
    events: list[IQEvent],
    raw_features: np.ndarray,
    norm_features: np.ndarray,
    feature_names: tuple[str, ...],
    feature_weights: np.ndarray,
    distance_threshold: float = 1.5,
    n_clusters: int | None = None,
    linkage_method: str = "average",
    sample_rate: float = 1.0,
) -> PulseClusterResult:
    """Performs hierarchical agglomerative clustering on extracted normalized RF pulse features."""
    n_events = len(events)
    n_feat = len(feature_names)

    if n_events == 0:
        return PulseClusterResult(
            events=[],
            cluster_labels=np.empty(0, dtype=int),
            num_clusters=0,
            feature_names=feature_names,
            raw_features=np.empty((0, n_feat), dtype=np.float64),
            normalized_features=np.empty((0, n_feat), dtype=np.float64),
            cluster_centroids=np.empty((0, n_feat), dtype=np.float64),
            cluster_spreads=np.empty((0, n_feat), dtype=np.float64),
            cluster_sizes=np.empty(0, dtype=int),
            confidence=np.empty(0, dtype=np.float64),
            sample_rate=sample_rate,
            distance_threshold=distance_threshold,
            linkage=linkage_method,
            feature_weights=feature_weights,
        )

    if n_events == 1:
        return PulseClusterResult(
            events=events,
            cluster_labels=np.array([0], dtype=int),
            num_clusters=1,
            feature_names=feature_names,
            raw_features=raw_features,
            normalized_features=norm_features,
            cluster_centroids=raw_features.copy(),
            cluster_spreads=np.zeros((1, n_feat), dtype=np.float64),
            cluster_sizes=np.array([1], dtype=int),
            confidence=np.array([1.0], dtype=np.float64),
            sample_rate=sample_rate,
            distance_threshold=distance_threshold,
            linkage=linkage_method,
            feature_weights=feature_weights,
        )

    dist_mat = compute_weighted_distance_matrix(norm_features, feature_weights)
    condensed_dist = squareform(dist_mat, checks=False)

    linkage_tree = linkage(condensed_dist, method=linkage_method)

    if n_clusters is not None and n_clusters > 0:
        raw_labels = fcluster(linkage_tree, t=int(n_clusters), criterion="maxclust")
    else:
        raw_labels = fcluster(linkage_tree, t=float(distance_threshold), criterion="distance")

    unique_raw = []
    for lbl in raw_labels:
        if lbl not in unique_raw:
            unique_raw.append(lbl)

    label_map = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(unique_raw)}
    cluster_labels = np.array([label_map[lbl] for lbl in raw_labels], dtype=int)
    num_clusters = len(unique_raw)

    cluster_centroids = np.zeros((num_clusters, n_feat), dtype=np.float64)
    cluster_spreads = np.zeros((num_clusters, n_feat), dtype=np.float64)
    cluster_sizes = np.zeros(num_clusters, dtype=int)
    norm_centroids = np.zeros((num_clusters, n_feat), dtype=np.float64)

    for c in range(num_clusters):
        c_mask = cluster_labels == c
        c_raw = raw_features[c_mask]
        c_norm = norm_features[c_mask]
        cluster_sizes[c] = len(c_raw)
        cluster_centroids[c] = np.mean(c_raw, axis=0)
        norm_centroids[c] = np.mean(c_norm, axis=0)
        if len(c_raw) > 1:
            cluster_spreads[c] = np.std(c_raw, axis=0)
        else:
            cluster_spreads[c] = 0.0

    confidence = np.ones(n_events, dtype=np.float64)
    if num_clusters > 1:
        sqrt_w = np.sqrt(np.maximum(feature_weights, 0.0))
        for i in range(n_events):
            c_own = cluster_labels[i]
            z_i = norm_features[i] * sqrt_w

            mu_own = norm_centroids[c_own] * sqrt_w
            d_own = float(np.linalg.norm(z_i - mu_own))

            d_other = float("inf")
            for c_other in range(num_clusters):
                if c_other != c_own:
                    mu_other = norm_centroids[c_other] * sqrt_w
                    d_c = float(np.linalg.norm(z_i - mu_other))
                    if d_c < d_other:
                        d_other = d_c

            max_d = max(d_other, d_own, 1e-6)
            conf = (d_other - d_own) / max_d
            confidence[i] = float(np.clip(0.5 * (conf + 1.0), 0.0, 1.0))

    return PulseClusterResult(
        events=events,
        cluster_labels=cluster_labels,
        num_clusters=num_clusters,
        feature_names=feature_names,
        raw_features=raw_features,
        normalized_features=norm_features,
        cluster_centroids=cluster_centroids,
        cluster_spreads=cluster_spreads,
        cluster_sizes=cluster_sizes,
        confidence=confidence,
        sample_rate=sample_rate,
        distance_threshold=distance_threshold,
        linkage=linkage_method,
        feature_weights=feature_weights,
    )


def analyze_pri_sequence(toas: np.ndarray | list[float]) -> dict[str, Any]:
    """
    Computes Pulse Repetition Interval (PRI) metrics from pulse arrival times (TOA).
    Classifies PRI modulation mode: 'constant', 'staggered', 'jittered', 'agile'.
    """
    arr = np.sort(np.asarray(toas, dtype=np.float64))
    if len(arr) < 2:
        return {
            "pri_mean_sec": 0.0,
            "pri_std_sec": 0.0,
            "pri_min_sec": 0.0,
            "pri_max_sec": 0.0,
            "pri_jitter_percent": 0.0,
            "pri_mode": "constant" if len(arr) == 1 else "unknown",
            "pri_sequence": [],
            "num_pulses": len(arr),
        }

    pris = np.diff(arr)
    valid_pris = pris[pris > 0]
    if len(valid_pris) == 0:
        return {
            "pri_mean_sec": 0.0,
            "pri_std_sec": 0.0,
            "pri_min_sec": 0.0,
            "pri_max_sec": 0.0,
            "pri_jitter_percent": 0.0,
            "pri_mode": "unknown",
            "pri_sequence": [],
            "num_pulses": len(arr),
        }

    pri_mean = float(np.mean(valid_pris))
    pri_std = float(np.std(valid_pris))
    pri_min = float(np.min(valid_pris))
    pri_max = float(np.max(valid_pris))
    jitter_pct = float((pri_std / max(pri_mean, 1e-15)) * 100.0)

    # PRI Mode Classification
    pri_mode = "constant"
    if jitter_pct < 2.0:
        pri_mode = "constant"
    elif len(valid_pris) >= 4:
        # Check for staggered PRI: small set of recurring discrete values
        quantized = np.round(valid_pris, decimals=6)
        unique_vals, counts = np.unique(quantized, return_counts=True)
        if 2 <= len(unique_vals) <= 8 and np.all(counts >= 2):
            pri_mode = "staggered"
        elif jitter_pct <= 30.0:
            pri_mode = "jittered"
        else:
            pri_mode = "agile"
    elif jitter_pct <= 30.0:
        pri_mode = "jittered"
    else:
        pri_mode = "agile"

    return {
        "pri_mean_sec": pri_mean,
        "pri_std_sec": pri_std,
        "pri_min_sec": pri_min,
        "pri_max_sec": pri_max,
        "pri_jitter_percent": jitter_pct,
        "pri_mode": pri_mode,
        "pri_sequence": [float(p) for p in valid_pris],
        "num_pulses": len(arr),
    }


def hungarian_matching(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes global optimal bipartite matching using Munkres / Hungarian algorithm."""
    if cost_matrix.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


def kmeans_clustering(
    features: np.ndarray,
    k: int,
    max_iter: int = 100,
    n_init: int = 10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Standard K-Means with K-Means++ initialization."""
    n_samples, n_features = features.shape
    if n_samples == 0:
        return np.empty(0, dtype=int), np.empty((0, n_features)), 0.0
    if k >= n_samples:
        labels = np.arange(n_samples, dtype=int)
        centroids = features.copy()
        return labels, centroids, 0.0

    rng = np.random.default_rng(random_state)
    best_inertia = float("inf")
    best_labels = np.zeros(n_samples, dtype=int)
    best_centroids = np.zeros((k, n_features), dtype=np.float64)

    for _ in range(n_init):
        # K-Means++ init
        centroids = np.zeros((k, n_features), dtype=np.float64)
        c0_idx = rng.integers(0, n_samples)
        centroids[0] = features[c0_idx]
        for c_i in range(1, k):
            diffs = features[:, None, :] - centroids[:c_i][None, :, :]
            dists_sq = np.min(np.sum(diffs ** 2, axis=-1), axis=1)
            sum_dists = np.sum(dists_sq)
            if sum_dists <= 1e-12:
                next_idx = int(rng.integers(0, n_samples))
            else:
                probs = dists_sq / sum_dists
                probs = probs / np.sum(probs)
                next_idx = int(rng.choice(n_samples, p=probs))
            centroids[c_i] = features[next_idx]

        labels = np.zeros(n_samples, dtype=int)
        for _ in range(max_iter):
            # Assignment
            diff = features[:, None, :] - centroids[None, :, :]
            dists_sq = np.sum(diff ** 2, axis=-1)
            new_labels = np.argmin(dists_sq, axis=1)

            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            # Update centroids
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    centroids[c] = np.mean(features[mask], axis=0)
                else:
                    centroids[c] = features[rng.integers(0, n_samples)]

        # Calculate inertia
        diff = features - centroids[labels]
        inertia = float(np.sum(diff ** 2))

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids

    return best_labels, best_centroids, best_inertia


def gmm_clustering(
    features: np.ndarray,
    k: int,
    max_iter: int = 100,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Gaussian Mixture Model clustering with Expectation-Maximization."""
    n_samples, n_features = features.shape
    if n_samples == 0:
        return np.empty(0, dtype=int), np.empty((0, n_features)), 0.0
    if k >= n_samples:
        return np.arange(n_samples, dtype=int), features.copy(), 0.0

    # Initialize with K-Means
    labels, means, _ = kmeans_clustering(features, k, max_iter=20, random_state=random_state)
    weights = np.array([np.mean(labels == c) for c in range(k)], dtype=np.float64)
    weights = np.maximum(weights, 1e-4)
    weights /= np.sum(weights)

    covs = np.zeros((k, n_features, n_features), dtype=np.float64)
    reg_cov = 1e-4 * np.eye(n_features)
    for c in range(k):
        mask = labels == c
        if np.sum(mask) > 1:
            covs[c] = np.cov(features[mask], rowvar=False) + reg_cov
        else:
            covs[c] = np.eye(n_features)

    responsibilities = np.zeros((n_samples, k), dtype=np.float64)

    for _ in range(max_iter):
        # E-step
        for c in range(k):
            diff = features - means[c]
            try:
                inv_cov = np.linalg.pinv(covs[c])
                det_cov = max(np.linalg.det(covs[c]), 1e-15)
                norm_const = 1.0 / math.sqrt(((2.0 * math.pi) ** n_features) * det_cov)
                exponent = -0.5 * np.sum(diff * (diff @ inv_cov), axis=1)
                responsibilities[:, c] = weights[c] * norm_const * np.exp(np.clip(exponent, -50.0, 50.0))
            except Exception:
                responsibilities[:, c] = weights[c]

        resp_sum = np.sum(responsibilities, axis=1, keepdims=True)
        resp_sum = np.maximum(resp_sum, 1e-15)
        responsibilities /= resp_sum

        # M-step
        n_c = np.sum(responsibilities, axis=0)
        n_c = np.maximum(n_c, 1e-8)
        weights = n_c / n_samples
        for c in range(k):
            means[c] = np.sum(responsibilities[:, c:c+1] * features, axis=0) / n_c[c]
            diff = features - means[c]
            weighted_diff = responsibilities[:, c:c+1] * diff
            covs[c] = (weighted_diff.T @ diff) / n_c[c] + reg_cov

    final_labels = np.argmax(responsibilities, axis=1)
    diff = features - means[final_labels]
    inertia = float(np.sum(diff ** 2))
    return final_labels, means, inertia


def kmedoids_clustering(
    features: np.ndarray,
    k: int,
    max_iter: int = 100,
    n_init: int = 10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Robust Partitioning Around Medoids (PAM) with K-medoids++ initialization and multiple restarts.
    Minimizes true L1/L2 medoid sum objective J = sum_i d(x_i, m_{c_i}).
    """
    n_samples, n_features = features.shape
    if n_samples == 0:
        return np.empty(0, dtype=int), np.empty((0, n_features), dtype=np.float64), 0.0

    diff = features[:, None, :] - features[None, :, :]
    dist_mat = np.sqrt(np.maximum(np.sum(diff ** 2, axis=-1), 0.0))

    if k <= 1 or n_samples == 1:
        labels = np.zeros(n_samples, dtype=int)
        sum_dists = np.sum(dist_mat, axis=1)
        best_medoid = int(np.argmin(sum_dists))
        medoid_arr = features[best_medoid : best_medoid + 1]
        obj = float(sum_dists[best_medoid])
        return labels, medoid_arr, obj

    if k >= n_samples:
        labels = np.arange(n_samples, dtype=int)
        centroids = features.copy()
        return labels, centroids, 0.0

    rng = np.random.default_rng(random_state)
    best_cost = float("inf")
    best_labels = np.zeros(n_samples, dtype=int)
    best_medoids = np.zeros(k, dtype=int)

    num_restarts = max(1, int(n_init))

    for restart in range(num_restarts):
        # 1. K-medoids++ Initialization
        medoid_indices = [int(rng.integers(0, n_samples))]
        for _ in range(1, k):
            sub_dists = dist_mat[:, medoid_indices]
            min_dists = np.min(sub_dists, axis=1)
            probs = min_dists ** 2
            sum_probs = np.sum(probs)
            if sum_probs <= 1e-12:
                unselected = [idx for idx in range(n_samples) if idx not in medoid_indices]
                next_medoid = int(rng.choice(unselected)) if len(unselected) > 0 else int(rng.integers(0, n_samples))
            else:
                probs = probs / sum_probs
                next_medoid = int(rng.choice(n_samples, p=probs))
            medoid_indices.append(next_medoid)

        medoid_indices = np.array(medoid_indices, dtype=int)
        labels = np.zeros(n_samples, dtype=int)

        # 2. Iterative PAM Refinement
        for _ in range(max_iter):
            # Assignment
            dists_to_medoids = dist_mat[:, medoid_indices]
            new_labels = np.argmin(dists_to_medoids, axis=1)

            changed = False
            # Within-cluster medoid optimization
            for c in range(k):
                cluster_members = np.where(new_labels == c)[0]
                if len(cluster_members) > 0:
                    sub_dm = dist_mat[np.ix_(cluster_members, cluster_members)]
                    best_cand = cluster_members[np.argmin(np.sum(sub_dm, axis=1))]
                    if best_cand != medoid_indices[c]:
                        medoid_indices[c] = best_cand
                        changed = True

            # Fast PAM Swap Pass across non-medoid points
            current_cost = np.sum(np.min(dist_mat[:, medoid_indices], axis=1))
            non_medoid_set = [idx for idx in range(n_samples) if idx not in medoid_indices]
            if len(non_medoid_set) > 0:
                swap_eval_pool = non_medoid_set if len(non_medoid_set) <= 40 else list(rng.choice(non_medoid_set, size=40, replace=False))
                for cand in swap_eval_pool:
                    for c_idx in range(k):
                        trial_medoids = medoid_indices.copy()
                        trial_medoids[c_idx] = cand
                        trial_cost = np.sum(np.min(dist_mat[:, trial_medoids], axis=1))
                        if trial_cost < current_cost - 1e-9:
                            medoid_indices = trial_medoids
                            current_cost = trial_cost
                            changed = True
                            break
                    if changed:
                        break

            if not changed and np.array_equal(new_labels, labels):
                break
            labels = new_labels

        # Calculate final medoid cost J = sum_i d(x_i, m_{c_i})
        final_dists = dist_mat[:, medoid_indices]
        final_labels = np.argmin(final_dists, axis=1)
        final_cost = float(np.sum(np.min(final_dists, axis=1)))

        if final_cost < best_cost:
            best_cost = final_cost
            best_labels = final_labels
            best_medoids = medoid_indices.copy()

    centroids = features[best_medoids]
    return best_labels, centroids, best_cost


def dbscan_clustering(
    features: np.ndarray,
    eps: float | None = None,
    min_samples: int = 3,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray, float] | tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """
    Robust Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
    Supports automatic stability plateau and cluster cohesion evaluation on the k-distance curve.
    Returns: (labels with -1 for noise/outlier pulses, core_indices, eps_used)
    """
    n_samples = len(features)
    if n_samples == 0:
        if return_diagnostics:
            return np.empty(0, dtype=int), np.empty(0, dtype=int), 0.0, {"auto_eps_method": "empty"}
        return np.empty(0, dtype=int), np.empty(0, dtype=int), 0.0

    diff = features[:, None, :] - features[None, :, :]
    dist_mat = np.sqrt(np.maximum(np.sum(diff ** 2, axis=-1), 0.0))

    auto_method = "manual"
    k_val = max(1, min(min_samples, n_samples - 1))
    candidate_eps_list: list[float] = []

    if eps is not None and eps > 0:
        eps_used = float(eps)
    else:
        # Automatic Epsilon Estimation via Multi-Scale Stability Plateau Analysis
        sorted_dists = np.sort(dist_mat, axis=1)
        k_dists = np.sort(sorted_dists[:, k_val])

        # Exclude extreme isolated background noise (> 5.0 in normalized units) to analyze inlier curve
        inlier_k = k_dists[k_dists <= 5.0]
        if len(inlier_k) == 0:
            inlier_k = k_dists

        cand_grid = np.linspace(max(0.05, float(inlier_k[0])), min(2.0, float(inlier_k[-1]) + 0.1), 200)
        candidate_eps_list = [float(round(c, 4)) for c in cand_grid]

        cluster_counts: list[int] = []
        noise_counts: list[int] = []
        sil_scores: list[float] = []

        for c_eps in cand_grid:
            core_mask = np.sum(dist_mat <= c_eps, axis=1) >= min_samples
            labels_trial = np.full(n_samples, -1, dtype=int)
            visited = np.zeros(n_samples, dtype=bool)
            cid_count = 0
            for i in range(n_samples):
                if visited[i]:
                    continue
                visited[i] = True
                if not core_mask[i]:
                    continue
                labels_trial[i] = cid_count
                q = list(np.where(dist_mat[i] <= c_eps)[0])
                while q:
                    nbr = q.pop(0)
                    if not visited[nbr]:
                        visited[nbr] = True
                        if core_mask[nbr]:
                            q.extend(list(np.where(dist_mat[nbr] <= c_eps)[0]))
                    if labels_trial[nbr] == -1:
                        labels_trial[nbr] = cid_count
                cid_count += 1

            n_n = int(np.sum(labels_trial == -1))
            cluster_counts.append(cid_count)
            noise_counts.append(n_n)

            sil = 0.5
            if cid_count >= 2 and (n_samples - n_n) >= 4:
                valid_pts = features[labels_trial >= 0]
                valid_lbls = labels_trial[labels_trial >= 0]
                if len(np.unique(valid_lbls)) >= 2:
                    sil = float(np.clip(compute_silhouette_score(valid_pts, valid_lbls), 0.0, 1.0))
            elif cid_count == 1:
                sil = 0.8
            sil_scores.append(sil)

        c_counts_arr = np.array(cluster_counts)
        n_counts_arr = np.array(noise_counts)
        s_scores_arr = np.array(sil_scores)

        # Extract contiguous clustering stability plateaus
        plateaus = []
        curr_k = c_counts_arr[0]
        curr_n = n_counts_arr[0]
        start_idx = 0

        for i in range(1, len(cand_grid)):
            if c_counts_arr[i] != curr_k or n_counts_arr[i] != curr_n:
                if curr_k >= 1:
                    p_span = cand_grid[i - 1] - cand_grid[start_idx]
                    mean_sil = float(np.mean(s_scores_arr[start_idx:i]))
                    n_clustered = n_samples - curr_n
                    clustered_ratio = n_clustered / n_samples
                    mid_eps = float(0.5 * (cand_grid[start_idx] + cand_grid[i - 1]))
                    plateaus.append({
                        "k": curr_k,
                        "noise": curr_n,
                        "span": p_span,
                        "mean_sil": mean_sil,
                        "clustered_ratio": clustered_ratio,
                        "mid_eps": mid_eps,
                    })
                curr_k = c_counts_arr[i]
                curr_n = n_counts_arr[i]
                start_idx = i

        if curr_k >= 1:
            p_span = cand_grid[-1] - cand_grid[start_idx]
            mean_sil = float(np.mean(s_scores_arr[start_idx:]))
            n_clustered = n_samples - curr_n
            clustered_ratio = n_clustered / n_samples
            mid_eps = float(0.5 * (cand_grid[start_idx] + cand_grid[-1]))
            plateaus.append({
                "k": curr_k,
                "noise": curr_n,
                "span": p_span,
                "mean_sil": mean_sil,
                "clustered_ratio": clustered_ratio,
                "mid_eps": mid_eps,
            })

        best_p = plateaus[0] if plateaus else {"mid_eps": 1.0}
        best_score = -1e9
        for p in plateaus:
            score = (p["span"] * 10.0) + (p["clustered_ratio"] * 10.0) + ((p["mean_sil"] ** 2) * 15.0) - 1.0 * abs(p["mid_eps"] - 1.0)
            if score > best_score:
                best_score = score
                best_p = p

        eps_used = float(best_p["mid_eps"])
        auto_method = "stability_plateau_silhouette"

    # Final DBSCAN execution with selected eps
    labels = np.full(n_samples, -1, dtype=int)
    core_mask = np.zeros(n_samples, dtype=bool)
    neighbors_list = []

    for i in range(n_samples):
        nbrs = np.where(dist_mat[i] <= eps_used)[0]
        neighbors_list.append(nbrs)
        if len(nbrs) >= min_samples:
            core_mask[i] = True

    cluster_id = 0
    visited = np.zeros(n_samples, dtype=bool)

    for i in range(n_samples):
        if visited[i]:
            continue
        visited[i] = True

        if not core_mask[i]:
            continue

        labels[i] = cluster_id
        queue = list(neighbors_list[i])

        while queue:
            nbr = queue.pop(0)
            if not visited[nbr]:
                visited[nbr] = True
                if core_mask[nbr]:
                    queue.extend(neighbors_list[nbr])
            if labels[nbr] == -1:
                labels[nbr] = cluster_id

        cluster_id += 1

    core_indices = np.where(core_mask)[0]

    if return_diagnostics:
        diag = {
            "eps_used": float(eps_used),
            "auto_eps_method": auto_method,
            "k_distance_k": int(k_val),
            "candidate_eps": candidate_eps_list,
        }
        return labels, core_indices, float(eps_used), diag

    return labels, core_indices, float(eps_used)


def compute_silhouette_score(features: np.ndarray, labels: np.ndarray) -> float:
    """Computes mean silhouette score across partitioned clusters."""
    n_samples = len(features)
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]
    valid_features = features[valid_mask]

    unique_labels = np.unique(valid_labels)
    if len(unique_labels) < 2 or len(valid_features) < 3:
        return 0.0

    diff = valid_features[:, None, :] - valid_features[None, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=-1))

    scores = np.zeros(len(valid_features), dtype=np.float64)
    for i, lbl in enumerate(valid_labels):
        same_mask = valid_labels == lbl
        same_mask[i] = False
        if np.any(same_mask):
            a_i = float(np.mean(dist_mat[i, same_mask]))
        else:
            a_i = 0.0

        b_i = float("inf")
        for other_lbl in unique_labels:
            if other_lbl != lbl:
                other_mask = valid_labels == other_lbl
                mean_other = float(np.mean(dist_mat[i, other_mask]))
                if mean_other < b_i:
                    b_i = mean_other

        max_ab = max(a_i, b_i, 1e-15)
        scores[i] = (b_i - a_i) / max_ab

    return float(np.mean(scores))
