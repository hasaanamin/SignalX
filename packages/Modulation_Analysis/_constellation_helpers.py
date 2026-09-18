from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
from scipy import interpolate
from scipy.stats import skew, kurtosis
from scipy.cluster.vq import kmeans2

from ._common import (
    _as_iq_array,
    _extract_sample_rate,
    _validate_sample_rate,
    _safe_divide,
    _complex_power,
    _amplitude_valid_mask,
    _circular_difference,
    _circular_mean,
    _safe_rms_normalize,
)
from ._results import SymbolRateCandidate
from ._validated_helpers import (
    _validated_cyclic_frequency_candidates,
    _estimate_phase_states,
    _fit_gmm_1d,
    _instantaneous_frequency,
)


def _remove_dc(iq_data: np.ndarray) -> Tuple[np.ndarray, complex]:
    """Removes DC offset from complex IQ signal if statistically significant: x_c[n] = x[n] - E[x]."""
    if len(iq_data) == 0:
        return iq_data.copy(), 0.0 + 0.0j
    dc_val = complex(np.mean(iq_data))
    n = len(iq_data)
    std_val = float(np.std(iq_data))
    se = std_val / math.sqrt(n) if n > 0 else 0.0
    if abs(dc_val) > 3.0 * se:
        return iq_data - dc_val, dc_val
    return iq_data.copy(), 0.0 + 0.0j


def _normalize_complex_rms(iq_data: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
    """Normalizes complex IQ array to unit average power: g = sqrt(E[|x|^2]), x_N = x / g."""
    return _safe_rms_normalize(iq_data, eps=eps)


def _estimate_coarse_frequency_offset(
    iq_data: np.ndarray,
    sample_rate: float = 1.0,
    method: str = "auto",
) -> Tuple[float, str]:
    """
    Estimates coarse carrier frequency offset (CFO).
    """
    fs = _validate_sample_rate(sample_rate)
    n = len(iq_data)
    if n < 32 or method == "none":
        return 0.0, "none"

    m_clean = str(method).lower().strip()
    if m_clean == "psk8":
        candidates_m = [8]
    elif m_clean == "psk4":
        candidates_m = [4]
    elif m_clean == "psk2":
        candidates_m = [2]
    elif m_clean == "cw" or m_clean == "direct":
        candidates_m = [1]
    else:
        candidates_m = [8, 4, 2, 1]

    nfft = max(4096, 1 << (min(n, 4096) - 1).bit_length())
    win = np.hanning(min(n, nfft))
    win_len = len(win)

    best_cfo = 0.0
    best_snr = 0.0
    best_method = "none"

    for m in candidates_m:
        z_seg = iq_data[:win_len]
        y_m = (z_seg ** m) * win
        fft_m = np.abs(np.fft.fftshift(np.fft.fft(y_m, n=nfft)))
        freqs_m = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))

        peak_idx = int(np.argmax(fft_m))
        peak_val = float(fft_m[peak_idx])

        # Local floor around peak (excluding 3 bins on each side)
        local_bins = np.concatenate([
            fft_m[max(0, peak_idx - 30):max(0, peak_idx - 3)],
            fft_m[min(nfft, peak_idx + 4):min(nfft, peak_idx + 31)],
        ])
        local_floor = float(np.median(local_bins)) if len(local_bins) > 0 else float(np.median(fft_m))
        prominence = peak_val / max(1e-12, local_floor)

        if prominence > best_snr and prominence > 3.5:
            best_snr = prominence
            # Parabolic sub-bin interpolation on FFT peak
            if 0 < peak_idx < nfft - 1:
                y0 = float(fft_m[peak_idx - 1])
                y1 = float(fft_m[peak_idx])
                y2 = float(fft_m[peak_idx + 1])
                denom = y0 - 2.0 * y1 + y2
                delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
                delta = float(np.clip(delta, -0.5, 0.5))
                f_peak = float(freqs_m[peak_idx] + delta * (fs / nfft))
            else:
                f_peak = float(freqs_m[peak_idx])

            best_cfo = f_peak / float(m)
            best_method = f"m_power_{m}"

    return best_cfo, best_method


def _apply_frequency_correction(
    iq_data: np.ndarray,
    sample_rate: float,
    freq_offset_hz: float,
) -> np.ndarray:
    """Applies frequency de-rotation: x_corr[n] = x[n] * exp(-j 2pi f_offset n / Fs)."""
    if len(iq_data) == 0 or abs(freq_offset_hz) < 1e-9:
        return iq_data.copy()
    fs = _validate_sample_rate(sample_rate)
    n = len(iq_data)
    t = np.arange(n, dtype=np.float64) / fs
    rot = np.exp(-1j * 2.0 * np.pi * freq_offset_hz * t)
    return iq_data * rot


def _estimate_symbol_rate(
    iq_data: Any,
    sample_rate: float = 1.0,
    min_rate_baud: float | None = None,
    max_rate_baud: float | None = None,
) -> Dict[str, Any]:
    """
    Canonical symbol rate estimator reusing the validated cyclostationary fine alpha engine
    and FSK transition clock recovery for multi-modulation support (PSK, QAM, ASK, FSK).
    """
    z = _as_iq_array(iq_data)
    fs = _validate_sample_rate(sample_rate)
    n = len(z)

    if n < 32:
        return {
            "symbol_rate_baud": None,
            "samples_per_symbol": None,
            "candidates": [],
            "confidence": 0.0,
            "valid": False,
            "reason": "Insufficient samples for symbol rate estimation.",
        }

    # CW Gate: check if signal is unmodulated CW carrier
    env = np.abs(z)
    mean_env = float(np.mean(env))
    std_env = float(np.std(env))
    env_cv = std_env / max(1e-12, mean_env)
    
    fm_info = _instantaneous_frequency(z, sample_rate=fs)
    fi = fm_info["instantaneous_frequency_hz"]
    std_fi = float(np.std(fi[np.isfinite(fi)])) if np.any(np.isfinite(fi)) else 0.0
    
    if env_cv < 0.02 and std_fi < 500.0:
        return {
            "symbol_rate_baud": None,
            "samples_per_symbol": None,
            "candidates": [],
            "confidence": 0.0,
            "valid": False,
            "reason": "CW unmodulated carrier detected; no digital symbol modulation.",
        }

    # 1. Standard Cyclostationary fine alpha profile
    cyclo_res = _validated_cyclic_frequency_candidates(
        z,
        sample_rate=fs,
        min_alpha_hz=min_rate_baud,
        num_candidates=5,
    )

    candidates_raw = cyclo_res.get("candidates", [])
    symbol_candidates: List[SymbolRateCandidate] = []

    for c in candidates_raw:
        rate = float(c["absolute_cyclic_rate_hz"])
        score = float(c["score"])
        alpha_hz = float(c["signed_alpha_hz"])

        if max_rate_baud is not None and rate > max_rate_baud:
            continue
        if min_rate_baud is not None and rate < min_rate_baud:
            continue

        symbol_candidates.append(
            SymbolRateCandidate(
                rate_baud=rate,
                score=score,
                source_cyclic_frequency_hz=alpha_hz,
                source="fine_alpha_profile",
            )
        )

    cyclic_detected = cyclo_res.get("cyclostationary_detected", False)
    snr_db = cyclo_res.get("raw_cyclic_snr_db", 0.0)

    # 2. Check for FSK frequency transition clock line if constant envelope with frequency activity
    fsk_rate = None
    fsk_conf = 0.0
    if env_cv < 0.15 and std_fi > 5000.0:
        df = np.abs(np.diff(fi))
        th = np.median(df) + 2.5 * np.std(df)
        trans_indices = np.where(df > th)[0]
        dwell_intervals = np.diff(trans_indices) if len(trans_indices) > 4 else np.array([])
        
        if len(dwell_intervals) >= 5:
            p10_dwell = float(np.percentile(dwell_intervals, 10))
            nfft = 131072
            win = np.hanning(len(df))
            df_win = (df - np.mean(df)) * win
            fft_mag = np.abs(np.fft.rfft(df_win, n=nfft))
            freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
            
            search_mask = (freqs >= 40e3) & (freqs <= 2.5e6)
            sub_mag = fft_mag.copy()
            sub_mag[~search_mask] = 0.0
            
            median_mag = float(np.median(fft_mag[search_mask])) if np.any(search_mask) else 1e-12
            peak_mag = float(np.max(sub_mag))
            peak_snr = (peak_mag / max(1e-12, median_mag)) ** 2
            
            if peak_snr >= 10.0:
                top_indices = np.argsort(sub_mag)[-20:][::-1]
                top_freqs = freqs[top_indices]
                
                best_f0 = None
                best_score = -1.0
                
                for cand_f in top_freqs:
                    if cand_f < 40e3 or cand_f > 2.5e6:
                        continue
                    for sub in [1.0, 0.5, 1.0 / 3.0, 0.25]:
                        f_test = cand_f * sub
                        if f_test < 40e3 or f_test > 1.5e6:
                            continue
                        
                        sub_score = 0.0
                        for h in [1, 2, 3]:
                            h_freq = f_test * h
                            if h_freq <= freqs[-1]:
                                bin_idx = int(round(h_freq / (fs / nfft)))
                                if 0 <= bin_idx < len(fft_mag):
                                    sub_score += float(np.max(fft_mag[max(0, bin_idx - 2):min(len(fft_mag), bin_idx + 3)]))
                                    
                        sps_test = fs / f_test
                        ratio = p10_dwell / sps_test
                        if 0.70 <= ratio <= 1.40:
                            sub_score *= 4.0
                        elif ratio < 0.70:
                            sub_score *= 0.1
                        elif ratio >= 1.8:
                            sub_score *= 0.2
                            
                        if sub_score > best_score:
                            best_score = sub_score
                            best_f0 = f_test
                            
                if best_f0 is not None and best_score > 0:
                    bin_idx = int(round(best_f0 / (fs / nfft)))
                    y0 = float(fft_mag[bin_idx - 1])
                    y1 = float(fft_mag[bin_idx])
                    y2 = float(fft_mag[bin_idx + 1])
                    denom = y0 - 2.0 * y1 + y2
                    delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
                    fsk_rate = (bin_idx + delta) * (fs / nfft)
                    fsk_conf = float(np.clip(10.0 * np.log10(peak_snr) / 25.0 + 0.5, 0.5, 0.98))

    # If cyclic profile had high confidence and standard PSK/QAM/ASK
    if cyclic_detected and snr_db >= 8.0 and symbol_candidates:
        top_cand = symbol_candidates[0]
        top_rate = top_cand.rate_baud
        
        # If signal is constant envelope FSK and FSK clock recovery found high confidence rate
        if env_cv < 0.10 and fsk_rate is not None and fsk_conf > 0.9:
            top_rate = fsk_rate
            conf = fsk_conf
            reason = f"FSK transition clock peak detected at {top_rate:.2f} Baud."
        else:
            conf = float(np.clip((snr_db - 8.0) / 15.0 + 0.5, 0.4, 0.98))
            reason = f"Cyclic peak detected at {top_rate:.2f} Baud (cyclic SNR: {snr_db:.1f} dB)."
            
        sps = fs / max(1e-12, top_rate)
        return {
            "symbol_rate_baud": top_rate,
            "samples_per_symbol": sps,
            "candidates": symbol_candidates,
            "confidence": conf,
            "valid": True,
            "reason": reason,
        }
        
    elif fsk_rate is not None:
        sps = fs / max(1e-12, fsk_rate)
        return {
            "symbol_rate_baud": fsk_rate,
            "samples_per_symbol": sps,
            "candidates": symbol_candidates,
            "confidence": fsk_conf,
            "valid": True,
            "reason": f"FSK transition clock peak detected at {fsk_rate:.2f} Baud.",
        }

    return {
        "symbol_rate_baud": None,
        "samples_per_symbol": None,
        "candidates": symbol_candidates,
        "confidence": 0.0,
        "valid": False,
        "reason": "No prominent cyclostationary or transition symbol timing peak detected.",
    }


def _recover_symbol_samples(
    iq_data: np.ndarray,
    sample_rate: float = 1.0,
    symbol_rate_baud: float | None = None,
    samples_per_symbol: float | None = None,
    timing_grid_points: int = 16,
) -> Dict[str, Any]:
    """
    Fractional symbol synchronization and timing recovery.
    """
    fs = _validate_sample_rate(sample_rate)
    n = len(iq_data)

    if samples_per_symbol is not None and samples_per_symbol > 1.0:
        sps = float(samples_per_symbol)
        sym_rate = fs / sps
    elif symbol_rate_baud is not None and symbol_rate_baud > 0:
        sym_rate = float(symbol_rate_baud)
        sps = fs / sym_rate
    else:
        rate_info = _estimate_symbol_rate(iq_data, sample_rate=fs)
        if rate_info["valid"] and rate_info["symbol_rate_baud"] is not None:
            sym_rate = float(rate_info["symbol_rate_baud"])
            sps = float(rate_info["samples_per_symbol"])
        else:
            return {
                "symbol_points": iq_data.copy(),
                "symbol_times_sec": np.arange(n, dtype=np.float64) / fs,
                "timing_offset_samples": 0.0,
                "samples_per_symbol": 1.0,
                "symbol_rate_baud": fs,
                "confidence": 0.0,
                "valid": False,
            }

    if sps < 1.05 or n < int(sps * 4):
        return {
            "symbol_points": iq_data.copy(),
            "symbol_times_sec": np.arange(n, dtype=np.float64) / fs,
            "timing_offset_samples": 0.0,
            "samples_per_symbol": sps,
            "symbol_rate_baud": sym_rate,
            "confidence": 0.1,
            "valid": False,
        }

    time_indices = np.arange(n, dtype=np.float64)
    interp_i = interpolate.interp1d(time_indices, np.real(iq_data), kind="linear", bounds_error=False, fill_value="extrapolate")
    interp_q = interpolate.interp1d(time_indices, np.imag(iq_data), kind="linear", bounds_error=False, fill_value="extrapolate")

    best_n0 = 0.0
    best_score = float("inf")
    best_symbols = np.array([], dtype=np.complex128)
    best_times = np.array([], dtype=np.float64)

    n0_candidates = np.linspace(0.0, sps, timing_grid_points, endpoint=False)

    for n0 in n0_candidates:
        max_k = int(math.floor((n - 1 - n0) / sps))
        if max_k < 4:
            continue
        k_indices = np.arange(max_k + 1, dtype=np.float64)
        sample_locs = n0 + k_indices * sps

        sym_i = interp_i(sample_locs)
        sym_q = interp_q(sample_locs)
        symbols = sym_i + 1j * sym_q

        sym_pwr = np.abs(symbols) ** 2
        mean_pwr = np.mean(sym_pwr)
        if mean_pwr > 1e-12:
            norm_pwr = sym_pwr / mean_pwr
            score = float(np.mean((norm_pwr - 1.0) ** 2))
        else:
            score = float("inf")

        if score < best_score:
            best_score = score
            best_n0 = float(n0)
            best_symbols = symbols
            best_times = sample_locs / fs

    if len(best_symbols) == 0:
        best_symbols = iq_data
        best_times = time_indices / fs

    conf = float(np.clip(1.0 / (1.0 + best_score), 0.3, 0.95))

    return {
        "symbol_points": best_symbols,
        "symbol_times_sec": best_times,
        "timing_offset_samples": best_n0,
        "samples_per_symbol": sps,
        "symbol_rate_baud": sym_rate,
        "confidence": conf,
        "valid": len(best_symbols) >= 8,
    }


def _estimate_common_phase_offset(
    symbol_points: np.ndarray,
    psk_order: int = 4,
) -> Tuple[np.ndarray, float]:
    """
    Estimates common constellation rotation theta_0 via M-th power average:
      theta_0 = (1 / M) angle( E[z^M] )
    """
    if len(symbol_points) == 0:
        return symbol_points.copy(), 0.0

    m = max(1, int(psk_order))
    z_m = symbol_points ** m
    mean_zm = complex(np.mean(z_m))
    if abs(mean_zm) < 1e-12:
        return symbol_points.copy(), 0.0

    theta_0 = float(np.angle(mean_zm) / float(m))
    z_rot = symbol_points * np.exp(-1j * theta_0)
    return z_rot, theta_0


def _constellation_cluster_statistics(
    symbol_points: np.ndarray,
    sample_rate: float = 1.0,
    reference_constellation: np.ndarray | None = None,
) -> Dict[str, Any]:
    """
    Extracts comprehensive constellation features from synchronized symbol samples using SciPy/NumPy.
    """
    n_syms = len(symbol_points)
    if n_syms < 4:
        return {
            "num_symbol_points": n_syms,
            "mean_complex": 0.0 + 0.0j,
            "mean_power": 0.0,
            "normalized_fourth_moment": 0.0,
            "circularity_coefficient": 0.0,
            "mean_radius": 0.0,
            "radius_std": 0.0,
            "radius_cv": 0.0,
            "radius_skewness": 0.0,
            "radius_kurtosis": 0.0,
            "radial_level_count": 1,
            "apparent_phase_state_count": 1,
            "phase_concentration": 0.0,
            "phase_occupancy_uniformity": 0.0,
            "iq_covariance": np.eye(2),
            "covariance_eigenvalues": np.array([1.0, 1.0]),
            "covariance_axis_ratio": 1.0,
            "constellation_cluster_count": 1,
            "cluster_compactness": 0.0,
            "minimum_cluster_separation": 0.0,
            "evm_rms": None,
            "evm_percent": None,
            "evm_valid": False,
            "confidence": 0.0,
            "valid": False,
        }

    z = symbol_points
    mean_c = complex(np.mean(z))
    pwr = float(np.mean(_complex_power(z)))
    eps = 1e-12

    k4 = float(np.mean(np.abs(z) ** 4) / max(eps, pwr ** 2))
    rho_c = float(np.abs(np.mean(z ** 2)) / max(eps, pwr))

    r = np.abs(z)
    mean_r = float(np.mean(r))
    std_r = float(np.std(r))
    cv_r = float(std_r / max(eps, mean_r))
    skew_r = float(skew(r)) if (n_syms >= 8 and std_r > 1e-6) else 0.0
    kurt_r = float(kurtosis(r)) if (n_syms >= 8 and std_r > 1e-6) else 0.0

    # Phase states estimation (supporting 1, 2, 4, 8, 16)
    phase_info = _estimate_phase_states(z, k_candidates=(1, 2, 4, 8, 16))
    app_phase_k = int(phase_info["num_phase_states"])
    concs = phase_info["concentrations"]
    phase_conc = float(np.mean(concs)) if len(concs) > 0 else 0.0
    occs = phase_info["occupancies"]
    uniform_occ = float(1.0 / app_phase_k) if app_phase_k > 0 else 1.0
    phase_occ_unif = float(1.0 - np.sum(np.abs(occs - uniform_occ)) / 2.0) if len(occs) > 0 else 1.0

    # Subsample for statistical fitting if large
    if n_syms > 6000:
        z_sub = z[:6000]
        r_sub = r[:6000]
        n_sub = 6000
    else:
        z_sub = z
        r_sub = r
        n_sub = n_syms

    # 1. Constellation cluster count & radial level count
    is_psk = (cv_r < 0.10) or (app_phase_k in (2, 4, 8, 16) and cv_r < 0.14)
    if is_psk:
        best_clust_k = app_phase_k
        best_radial_k = 1
    else:
        # Marginal I/Q estimation for QAM / ASK
        # Fine alignment of symbol samples for Cartesian grid analysis
        z_grid = z_sub
        if len(z_grid) >= 32:
            fine_cfo, _ = _estimate_coarse_frequency_offset(z_grid, sample_rate=1.0, method="auto")
            z_grid = _apply_frequency_correction(z_grid, sample_rate=1.0, freq_offset_hz=fine_cfo)
            z_grid, _ = _estimate_common_phase_offset(z_grid, psk_order=4)

        i_vals = np.real(z_grid)
        q_vals = np.imag(z_grid)
        std_i = float(np.std(i_vals))
        std_q = float(np.std(q_vals))

        # I levels
        best_i_k = 1
        best_i_bic = float("inf")
        for k in (1, 2, 4, 8):
            try:
                m, v, w, bic_i, _ = _fit_gmm_1d(i_vals, k=k)
                if np.min(w) < (0.30 / k):
                    continue
                if k > 1:
                    sort_m = np.sort(m)
                    min_d = np.min(np.diff(sort_m))
                    if min_d < 0.18 * std_i:
                        continue
                if bic_i < best_i_bic - 20.0:
                    best_i_bic = bic_i
                    best_i_k = k
            except Exception:
                continue

        # Q levels (check if 1D ASK where Q variance is near 0)
        if np.var(q_vals) < 1e-4 or (std_q / max(1e-12, std_i)) < 0.08:
            best_q_k = 1
        else:
            best_q_k = 1
            best_q_bic = float("inf")
            for k in (1, 2, 4, 8):
                try:
                    m, v, w, bic_q, _ = _fit_gmm_1d(q_vals, k=k)
                    if np.min(w) < (0.30 / k):
                        continue
                    if k > 1:
                        sort_m = np.sort(m)
                        min_d = np.min(np.diff(sort_m))
                        if min_d < 0.18 * std_q:
                            continue
                    if bic_q < best_q_bic - 20.0:
                        best_q_bic = bic_q
                        best_q_k = k
                except Exception:
                    continue

        best_clust_k = best_i_k * best_q_k

        # Theoretical radial level count from constellation grid geometry
        if best_q_k == 1:
            best_radial_k = max(1, best_i_k // 2)
        else:
            i_half = [2 * j + 1 for j in range(max(1, best_i_k // 2))]
            q_half = [2 * j + 1 for j in range(max(1, best_q_k // 2))]
            radii_sq = {ii**2 + qq**2 for ii in i_half for qq in q_half}
            best_radial_k = max(1, len(radii_sq))

    # IQ Covariance
    i_all = np.real(z)
    q_all = np.imag(z)
    cov_mat = np.cov(i_all, q_all)
    if cov_mat.ndim != 2 or cov_mat.shape != (2, 2):
        cov_mat = np.eye(2)
    eigvals = np.linalg.eigvalsh(cov_mat)
    eigvals = np.sort(np.maximum(0.0, eigvals))[::-1]
    axis_ratio = float(eigvals[0] / max(eps, eigvals[1])) if len(eigvals) >= 2 else 1.0

    # Cluster compactness and separation
    compactness = float(np.var(z)) / max(1.0, float(best_clust_k))
    min_sep = 2.0 / math.sqrt(max(1.0, float(best_clust_k)))

    # EVM computation (reference-aligned)
    evm_valid = False
    evm_rms = None
    evm_percent = None

    if reference_constellation is not None and len(reference_constellation) > 0:
        ref_pts = np.asarray(reference_constellation, dtype=np.complex128)
        scale = math.sqrt(float(np.mean(np.abs(ref_pts) ** 2)) / max(eps, pwr))
        z_scaled = z * scale
        best_evm = float("inf")
        for test_rot in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            z_rot = z_scaled * np.exp(1j * test_rot)
            diffs = z_rot[:, None] - ref_pts[None, :]
            closest_idx = np.argmin(np.abs(diffs), axis=1)
            ideal = ref_pts[closest_idx]
            err = float(np.mean(np.abs(z_rot - ideal) ** 2))
            if err < best_evm:
                best_evm = err
        evm_val = math.sqrt(best_evm / max(eps, float(np.mean(np.abs(ref_pts) ** 2))))
        evm_rms = float(evm_val)
        evm_percent = float(evm_val * 100.0)
        evm_valid = True

    conf = float(np.clip(phase_conc, 0.4, 0.98)) if app_phase_k > 1 else 0.6

    return {
        "num_symbol_points": n_syms,
        "mean_complex": mean_c,
        "mean_power": pwr,
        "normalized_fourth_moment": k4,
        "circularity_coefficient": rho_c,
        "mean_radius": mean_r,
        "radius_std": std_r,
        "radius_cv": cv_r,
        "radius_skewness": skew_r,
        "radius_kurtosis": kurt_r,
        "radial_level_count": best_radial_k,
        "apparent_phase_state_count": app_phase_k,
        "phase_concentration": phase_conc,
        "phase_occupancy_uniformity": phase_occ_unif,
        "iq_covariance": cov_mat,
        "covariance_eigenvalues": eigvals,
        "covariance_axis_ratio": axis_ratio,
        "constellation_cluster_count": best_clust_k,
        "cluster_compactness": compactness,
        "minimum_cluster_separation": min_sep,
        "evm_rms": evm_rms,
        "evm_percent": evm_percent,
        "evm_valid": evm_valid,
        "confidence": conf,
        "valid": n_syms >= 8,
    }
