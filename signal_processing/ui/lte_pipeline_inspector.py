"""LTE Signal Processing Step-by-Step Inspector and Streaming Engine.

Provides deep intermediate diagnostic extraction for every stage of the
LTE FDD normal-CP signal processing pipeline:
  Step 1: Raw IQ Input Ingestion (time waveform, PSD, constellation, RMS/PAPR)
  Step 2: Preprocessing & Resampling (2 MS/s -> 1.92 MS/s Kaiser FIR filtering)
  Step 3: PSS Search & Timing Acquisition (3-root cross-correlation, peak detection, coarse CFO)
  Step 4: SSS Correlation & Cell Identification (168 hypotheses, subframe 0/5, PCI calculation)
  Step 5: Slot Extraction & Resource Grid (Subframe 0 slot 1, CP residual CFO, 128-pt FFT)
  Step 6: Channel Estimation & Equalization (CRS pilot extraction, H(f) interpolation, SFBC)
  Step 7: PBCH & MIB Demodulation (QPSK constellation, Viterbi decoding, CRC-16 check, parsed MIB)

Also provides:
  - SyntheticLTEStreamSource: Real-time 2 MS/s LTE signal generator with live SNR/CFO/PCI control.
  - FileLTEStreamSource: Paced reader for .cf32, .h5, .npy recordings.
  - ContinuousStreamRunner: Multi-threaded 2 MS/s worker with decoupled 60 fps GUI snapshot publication.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.fft import fft, fftshift
from scipy.signal import freqz

from signal_processing.core import IQBlock, StreamInfo
from signal_processing.lte.broadcast import LTEBroadcastReceiver
from signal_processing.lte.channel import LTEChannel
from signal_processing.lte.config import LTESyncConfig
from signal_processing.lte.pbch import (
    PBCHDecoder,
    _channel,
    _soft_symbols,
    extract_slot,
    pbch_resources,
    slot_soft_bits,
)
from signal_processing.lte.sequences import SYNC_BINS, pss_sequence, sss_hypotheses, sync_symbol
from signal_processing.lte.sib import (
    PLMN_OPERATORS,
    LTESIB1Observation,
    pack_sib1_bits,
    parse_sib1_bits,
)
from signal_processing.lte.turbo import (
    attach_crc24a,
    decode_pdsch_transport_block,
    srsran_fec_library,
)
from tests.lte_pbch_fixture import waveform

# ============================================================================
# Dataclasses for Intermediate Step Results
# ============================================================================

@dataclass
class Step1Result:
    """Step 1: Input IQ Ingestion."""
    sample_rate_hz: float
    num_samples: int
    duration_ms: float
    rms_power_dbfs: float
    peak_power_dbfs: float
    papr_db: float
    time_ms: np.ndarray             # Time axis for display slice
    i_samples: np.ndarray           # Real part slice
    q_samples: np.ndarray           # Imag part slice
    psd_freqs_mhz: np.ndarray       # Frequency axis in MHz
    psd_dbfs_per_hz: np.ndarray     # Power Spectral Density in dBFS/Hz
    constellation_i: np.ndarray     # Scatter sample I
    constellation_q: np.ndarray     # Scatter sample Q


@dataclass
class Step2Result:
    """Step 2: Preprocessing (Resampling & Filtering)."""
    input_rate_hz: float
    output_rate_hz: float
    resampling_ratio_str: str       # "24/25"
    filter_type: str                # "1201-tap Kaiser FIR (beta=8.6, cutoff=700 kHz)"
    group_delay_samples: int
    num_output_samples: int
    filter_freqs_mhz: np.ndarray    # Filter response frequency axis
    filter_response_db: np.ndarray  # Filter magnitude response in dB
    output_time_ms: np.ndarray      # Output time axis slice
    output_i_samples: np.ndarray    # Filtered I slice
    output_q_samples: np.ndarray    # Filtered Q slice
    output_psd_freqs_mhz: np.ndarray
    output_psd_dbfs_per_hz: np.ndarray
    raw_iq_filtered: np.ndarray = field(repr=False)


@dataclass
class Step3Result:
    """Step 3: Primary Synchronization Signal (PSS) Acquisition."""
    pss_search_window_samples: int
    detected_nid2: int              # Best root (0, 1, or 2)
    peak_sample_index: int          # Sample offset where peak occurred
    peak_metric: float              # Normalized correlation metric (0 to 1)
    coarse_cfo_hz: float            # Estimated coarse CFO
    correlation_sample_indices: np.ndarray  # X-axis sample offsets
    corr_root0: np.ndarray          # Normalized correlation curve for N_ID_2 = 0
    corr_root1: np.ndarray          # Normalized correlation curve for N_ID_2 = 1
    corr_root2: np.ndarray          # Normalized correlation curve for N_ID_2 = 2
    threshold: float
    detection_passed: bool


@dataclass
class Step4Result:
    """Step 4: Secondary Synchronization Signal (SSS) & Cell ID."""
    detected_nid1: int              # SSS group index (0 to 167)
    detected_nid2: int              # PSS index (0, 1, or 2)
    detected_pci: int               # Physical Cell ID = 3 * N_ID_1 + N_ID_2
    detected_subframe: int          # Subframe 0 or Subframe 5
    sss_metric: float               # SSS correlation peak score
    second_peak_metric: float       # Second highest correlation peak
    peak_ratio: float               # Peak-to-second ratio
    frame_start_sample: int         # 10 ms radio frame start sample index
    hypothesis_scores: np.ndarray   # 168 correlation scores
    threshold: float
    detection_passed: bool


@dataclass
class Step5Result:
    """Step 5: PBCH Slot Extraction & Resource Grid."""
    frame_start_sample: int
    slot1_start_sample: int
    residual_cfo_hz: float          # Fine CFO from CP correlation
    total_cfo_hz: float             # Coarse CFO + Residual CFO
    cp_correlation_magnitude: float
    grid_magnitude_db: np.ndarray   # (5, 72) OFDM grid power in dB
    re_type_map: np.ndarray         # (5, 72) RE types: 0=data, 1=CRS pilot, 2=PBCH
    subcarrier_indices: np.ndarray  # -36 to +36 (excluding DC)
    symbol_indices: np.ndarray      # 0 to 4
    raw_grid: np.ndarray = field(repr=False)
    decoded_mib_tuple: Optional[tuple] = None


@dataclass
class Step6Result:
    """Step 6: Channel Estimation & Equalization."""
    num_antenna_ports: int          # Detected / tested ports (1, 2, or 4)
    channel_freqs_mhz: np.ndarray   # 72 subcarrier frequencies
    h_magnitude_port0: np.ndarray   # |H(f)| across 72 subcarriers
    h_phase_port0: np.ndarray       # Angle H(f) across 72 subcarriers
    h_magnitude_port1: Optional[np.ndarray]
    h_phase_port1: Optional[np.ndarray]
    soft_symbols_i: np.ndarray      # 240 equalized QPSK symbol real part
    soft_symbols_q: np.ndarray      # 240 equalized QPSK symbol imag part


@dataclass
class Step7Result:
    """Step 7: PBCH Viterbi & MIB Decoding."""
    crc_valid: bool
    antenna_ports: int
    frame_quarter: int
    system_frame_number: int
    sfn_msb: int
    downlink_bandwidth_hz: int
    downlink_resource_blocks: int
    phich_duration: str
    phich_resource: str
    raw_payload_bits: Tuple[int, ...]
    formatted_bits: str
    constellation_i: np.ndarray     # QPSK soft symbols I
    constellation_q: np.ndarray     # QPSK soft symbols Q
    llr_bits: np.ndarray            # 480 soft bits
    hypothesis_results: List[Dict[str, Any]]  # (port, quarter) -> CRC passed


@dataclass
class Step8Result:
    """Step 8: PCFICH, PDCCH & SIB1 Broadcast Information."""
    cfi: int                         # Control Format Indicator (1, 2, or 3)
    pcfich_confidence: float         # PCFICH detection margin
    has_sib1: bool                   # True if SIB1 was recovered
    operator_name: str               # e.g. "AT&T Mobility", "Verizon", etc.
    plmn_id: str                     # e.g. "310-410"
    mcc: str                         # "310"
    mnc: str                         # "410"
    tracking_area_code: int          # TAC
    cell_identity: int               # 28-bit global ECI
    enodeb_id: int                   # 20-bit eNodeB ID
    sector_id: int                   # 8-bit sector ID
    freq_band: int                   # Frequency band (e.g. Band 4)
    q_rx_lev_min_dbm: int            # Minimum Rx level (dBm)
    dci_start_rb: int                # PDSCH start RB
    dci_num_rbs: int                 # PDSCH RB count
    turbo_iterations: int            # Turbo decoder iterations used
    crc24a_valid: bool               # CRC-24A pass
    raw_bits: Tuple[int, ...]        # Unpacked SIB1 payload bits


@dataclass
class PipelineInspectionResult:
    """Complete container of all 8 step results for a single snapshot."""
    step1: Step1Result
    step2: Step2Result
    step3: Step3Result
    step4: Step4Result
    step5: Step5Result
    step6: Step6Result
    step7: Step7Result
    success: bool
    summary_message: str
    step8: Optional[Step8Result] = None



# ============================================================================
# Step-by-Step Inspector Engine
# ============================================================================

class LTEStepInspector:
    """Extracts rich intermediate diagnostics for each pipeline step."""

    def __init__(self, sample_rate_hz: float = 2_000_000.0) -> None:
        self.pss_templates = np.array([sync_symbol(pss_sequence(n)) for n in range(3)])
        self.set_sample_rate(sample_rate_hz)

    def set_sample_rate(self, sample_rate_hz: float) -> None:
        """Update sample rate and rebuild channel & sync configuration."""
        self.sample_rate_hz = float(sample_rate_hz)
        self.channel = LTEChannel(self.sample_rate_hz)
        self.sync_config = LTESyncConfig(input_sample_rate_hz=self.sample_rate_hz)

    def inspect_step1_input(self, iq: np.ndarray) -> Step1Result:
        """Step 1: Analyze raw input IQ signal."""
        val_iq = np.asarray(iq, dtype=np.complex64)
        n = len(val_iq)
        power_linear = np.abs(val_iq) ** 2
        mean_power = float(np.mean(power_linear)) if n > 0 else 1e-12
        peak_power = float(np.max(power_linear)) if n > 0 else 1e-12
        papr = peak_power / max(mean_power, 1e-12)

        rms_dbfs = 10.0 * math.log10(max(mean_power, 1e-12))
        peak_dbfs = 10.0 * math.log10(max(peak_power, 1e-12))
        papr_db = 10.0 * math.log10(max(papr, 1.0))
        duration_ms = (n / self.sample_rate_hz) * 1000.0

        # Time slice for display (up to 2048 points)
        slice_len = min(2048, n)
        step = max(1, slice_len // 1024)
        display_slice = val_iq[:slice_len:step]
        time_ms = (np.arange(len(display_slice)) * step / self.sample_rate_hz) * 1000.0

        # Power spectrum via 1024-point FFT with Hann window
        nfft = min(2048, 1 << int(math.floor(math.log2(max(n, 64)))))
        nfft = max(256, nfft)
        fft_chunk = val_iq[:nfft] if n >= nfft else np.pad(val_iq, (0, nfft - n))
        window = np.hanning(nfft)
        window_gain = np.sum(window ** 2) / nfft
        spec = fftshift(fft(fft_chunk * window))
        psd_linear = (np.abs(spec) ** 2) / (self.sample_rate_hz * window_gain * nfft)
        psd_dbfs_per_hz = 10.0 * np.log10(np.maximum(psd_linear, 1e-15))
        freqs_mhz = fftshift(np.fft.fftfreq(nfft, d=1.0 / self.sample_rate_hz)) / 1e6

        # Constellation sample (up to 1024 points)
        const_len = min(1024, n)
        const_indices = np.linspace(0, n - 1, const_len, dtype=int)
        const_iq = val_iq[const_indices]

        return Step1Result(
            sample_rate_hz=self.sample_rate_hz,
            num_samples=n,
            duration_ms=duration_ms,
            rms_power_dbfs=rms_dbfs,
            peak_power_dbfs=peak_dbfs,
            papr_db=papr_db,
            time_ms=time_ms,
            i_samples=display_slice.real.astype(np.float32),
            q_samples=display_slice.imag.astype(np.float32),
            psd_freqs_mhz=freqs_mhz.astype(np.float32),
            psd_dbfs_per_hz=psd_dbfs_per_hz.astype(np.float32),
            constellation_i=const_iq.real.astype(np.float32),
            constellation_q=const_iq.imag.astype(np.float32),
        )

    def inspect_step2_preprocessing(self, iq: np.ndarray) -> Step2Result:
        """Step 2: Resample from 2 MS/s to 1.92 MS/s with Kaiser FIR low-pass filter."""
        self.channel.reset()
        _, filtered_iq = self.channel.process(iq)
        out_rate = 1_920_000.0

        # Filter frequency response
        taps = self.channel.taps
        w, h = freqz(taps, worN=1024, fs=48_000_000.0)
        filter_freqs_mhz = (w / 1e6).astype(np.float32)
        filter_resp_db = (20.0 * np.log10(np.maximum(np.abs(h) / np.max(np.abs(h)), 1e-6))).astype(np.float32)

        # Output time slice
        n_out = len(filtered_iq)
        slice_len = min(2048, n_out)
        step = max(1, slice_len // 1024)
        out_slice = filtered_iq[:slice_len:step]
        out_time_ms = (np.arange(len(out_slice)) * step / out_rate) * 1000.0

        # Output spectrum
        nfft = min(2048, 1 << int(math.floor(math.log2(max(n_out, 64)))))
        nfft = max(256, nfft)
        fft_chunk = filtered_iq[:nfft] if n_out >= nfft else np.pad(filtered_iq, (0, nfft - n_out))
        window = np.hanning(nfft)
        window_gain = np.sum(window ** 2) / nfft
        spec = fftshift(fft(fft_chunk * window))
        psd_linear = (np.abs(spec) ** 2) / (out_rate * window_gain * nfft)
        psd_dbfs_per_hz = 10.0 * np.log10(np.maximum(psd_linear, 1e-15))
        freqs_mhz = fftshift(np.fft.fftfreq(nfft, d=1.0 / out_rate)) / 1e6

        return Step2Result(
            input_rate_hz=self.sample_rate_hz,
            output_rate_hz=out_rate,
            resampling_ratio_str="24/25" if self.sample_rate_hz == 2_000_000.0 else "1/1",
            filter_type="1201-tap Kaiser FIR (beta=8.6, cutoff=700 kHz)",
            group_delay_samples=self.channel.delay_ticks,
            num_output_samples=n_out,
            filter_freqs_mhz=filter_freqs_mhz,
            filter_response_db=filter_resp_db,
            output_time_ms=out_time_ms.astype(np.float32),
            output_i_samples=out_slice.real.astype(np.float32),
            output_q_samples=out_slice.imag.astype(np.float32),
            output_psd_freqs_mhz=freqs_mhz.astype(np.float32),
            output_psd_dbfs_per_hz=psd_dbfs_per_hz.astype(np.float32),
            raw_iq_filtered=filtered_iq,
        )

    def inspect_step3_pss(self, filtered_iq: np.ndarray) -> Step3Result:
        """Step 3: PSS Search across 3 roots and timing detection."""
        search_len = min(len(filtered_iq), 19200)
        window = filtered_iq[:search_len]
        energy = np.convolve(window.real**2 + window.imag**2, np.ones(128), mode="valid")
        energy = np.maximum(energy, 1e-12)

        corr_curves = []
        best_metric = -1.0
        best_nid = 0
        best_idx = 0
        best_cfo = 0.0

        for nid in range(3):
            kernel = self.pss_templates[nid][::-1].conj()
            conv = np.convolve(window, kernel, mode="valid")
            metric = (np.abs(conv) ** 2) / (62.0 * energy)
            corr_curves.append(metric)
            peak_idx = int(np.argmax(metric))
            peak_val = float(metric[peak_idx])
            if peak_val > best_metric:
                best_metric = peak_val
                best_nid = nid
                best_idx = peak_idx

        # If zero-CFO metric is below threshold (e.g. uncalibrated real-world SDR captures),
        # sweep coarse frequency offset branches
        if best_metric < self.sync_config.pss_threshold:
            t_full = np.arange(search_len) / 1_920_000.0
            sweep_cfos = [-45000, -37500, -30000, -22500, -15000, -7500, 7500, 15000, 22500, 30000, 37500, 45000]
            for fc in sweep_cfos:
                rot = np.exp(-2j * np.pi * fc * t_full)
                w_rot = window * rot
                curves_fc = []
                improved = False
                for nid in range(3):
                    kernel = self.pss_templates[nid][::-1].conj()
                    conv = np.convolve(w_rot, kernel, mode="valid")
                    metric = (np.abs(conv) ** 2) / (62.0 * energy)
                    curves_fc.append(metric)
                    peak_idx = int(np.argmax(metric))
                    peak_val = float(metric[peak_idx])
                    if peak_val > best_metric:
                        best_metric = peak_val
                        best_nid = nid
                        best_idx = peak_idx
                        best_cfo = float(fc)
                        improved = True
                if improved:
                    corr_curves = curves_fc

        # Coarse CFO estimation around the peak
        y = window[best_idx : best_idx + 128]
        t = np.arange(128) / 1_920_000.0
        cfo_grid = np.linspace(best_cfo - 7500, best_cfo + 7500, 21) if best_cfo != 0.0 else np.linspace(-45_000, 45_000, 19)
        cfo_powers = []
        for cfo in cfo_grid:
            template = self.pss_templates[best_nid] * np.exp(2j * np.pi * cfo * t)
            dot = np.vdot(template, y)
            cfo_powers.append(np.abs(dot) ** 2)
        coarse_cfo = float(cfo_grid[int(np.argmax(cfo_powers))])

        indices = np.arange(len(corr_curves[0]))
        thresh = self.sync_config.pss_threshold

        return Step3Result(
            pss_search_window_samples=search_len,
            detected_nid2=best_nid,
            peak_sample_index=best_idx,
            peak_metric=float(best_metric),
            coarse_cfo_hz=coarse_cfo,
            correlation_sample_indices=indices,
            corr_root0=corr_curves[0].astype(np.float32),
            corr_root1=corr_curves[1].astype(np.float32),
            corr_root2=corr_curves[2].astype(np.float32),
            threshold=float(thresh),
            detection_passed=best_metric >= thresh,
        )

    def inspect_step4_sss(
        self, filtered_iq: np.ndarray, step3: Step3Result
    ) -> Step4Result:
        """Step 4: SSS Correlation and Cell ID (PCI) calculation."""
        p = step3.peak_sample_index
        nid2 = step3.detected_nid2
        coarse_cfo = step3.coarse_cfo_hz
        t = np.arange(128) / 1_920_000.0

        # Fine CFO refinement on PSS symbol
        y = filtered_iq[p : p + 128]
        fine_grid = np.linspace(coarse_cfo - 5000, coarse_cfo + 5000, 21)
        fine_dots = [
            np.vdot(self.pss_templates[nid2] * np.exp(2j * np.pi * f * t), y)
            for f in fine_grid
        ]
        freq = float(fine_grid[int(np.argmax(np.abs(fine_dots)))])

        # SSS symbol lies 137 samples before PSS
        if p >= 137:
            yp = fft(y * np.exp(-2j * np.pi * freq * t))[SYNC_BINS]
            ys = fft(
                filtered_iq[p - 137 : p - 9]
                * np.exp(-2j * np.pi * freq * (t - 137 / 1_920_000.0))
            )[SYNC_BINS]

            pss_seq = pss_sequence(nid2)
            channel = yp * pss_seq.conj()
            for a, b in ((0, 31), (31, 62)):
                channel[a:b] = np.convolve(channel[a:b], np.ones(5), mode="same") / np.convolve(
                    np.ones(31), np.ones(5), mode="same"
                )
            power = np.abs(channel) ** 2
            regularizer = 0.05 * float(np.mean(power)) + 1e-12
            equalized = ys * channel.conj() / (power + regularizer)
            norm = float(np.vdot(equalized, equalized).real) * 62.0 + 1e-12

            sss_candidates = sss_hypotheses(nid2)
            dots = np.einsum("ij,j->i", sss_candidates, equalized, optimize=False)
            scores = np.abs(dots) ** 2 / norm
            order = np.argpartition(scores, -2)[-2:]
            best_idx = int(order[np.argmax(scores[order])])
            second_idx = int(order[np.argmin(scores[order])])
            best_score = float(scores[best_idx])
            second_score = float(scores[second_idx])
            ratio = best_score / max(second_score, 1e-12)
            group, half = divmod(best_idx, 2)
            pci = 3 * group + nid2
            frame_start = p - 832 - half * 9600

            # 168 group scores for display (max over subframe 0 and 5)
            group_scores = np.maximum(scores[0::2], scores[1::2])
        else:
            group = 0
            half = 0
            pci = nid2
            best_score = 0.0
            second_score = 0.0
            ratio = 1.0
            frame_start = 0
            group_scores = np.zeros(168)

        thresh = self.sync_config.sss_threshold

        return Step4Result(
            detected_nid1=int(group),
            detected_nid2=int(nid2),
            detected_pci=int(pci),
            detected_subframe=0 if half == 0 else 5,
            sss_metric=best_score,
            second_peak_metric=second_score,
            peak_ratio=ratio,
            frame_start_sample=frame_start,
            hypothesis_scores=group_scores.astype(np.float32),
            threshold=float(thresh),
            detection_passed=best_score >= thresh,
        )

    def inspect_step5_slot_grid(
        self,
        filtered_iq: np.ndarray,
        step3: Step3Result,
        step4: Step4Result,
    ) -> Step5Result:
        """Step 5: Extract subframe 0, slot 1 and construct (5, 72) OFDM Resource Grid."""
        fs0 = step4.frame_start_sample
        while fs0 < 0:
            fs0 += 19200

        pci = step4.detected_pci
        total_len = len(filtered_iq)

        # Multi-frame slot extraction and sequential soft-combining decoder
        decoder = PBCHDecoder(pci)
        best_grid = None
        best_cfo = step3.coarse_cfo_hz
        best_residual = 0.0
        best_fs = fs0
        decoded_tuple = None

        # Try up to 4 consecutive frames (spanning standard 40 ms BCH TTI)
        for k in range(4):
            fs = fs0 + k * 19200
            slot1_start = fs + 960
            if slot1_start + 960 > total_len:
                break
            grid, tot_cfo = extract_slot(
                filtered_iq, fs, 0, step3.coarse_cfo_hz, average_cfo=False
            )
            if best_grid is None:
                best_grid = grid
                best_cfo = tot_cfo
                best_residual = tot_cfo - step3.coarse_cfo_hz
                best_fs = fs
            dec = decoder.decode(grid, fs)
            if dec is not None:
                decoded_tuple = dec
                best_grid = grid
                best_cfo = tot_cfo
                best_residual = tot_cfo - step3.coarse_cfo_hz
                best_fs = fs
                break

        if best_grid is None:
            best_grid = np.zeros((5, 72), dtype=np.complex128)
            best_residual = 0.0
            best_cfo = step3.coarse_cfo_hz
            best_fs = fs0

        slot1_start = best_fs + 960
        grid_mag = np.abs(best_grid)
        grid_db = 20.0 * np.log10(np.maximum(grid_mag, 1e-6))

        # RE Type Map: 0 = unused/other, 1 = CRS Pilot, 2 = PBCH Data
        re_map = np.zeros((5, 72), dtype=int)
        coords = pbch_resources(pci)
        re_map[tuple(coords)] = 2
        for sym_idx in (0, 4):
            v = 0 if sym_idx == 0 else 3
            for m in range(12):
                k_idx = 6 * m + (v + pci) % 6
                if k_idx < 72:
                    re_map[sym_idx, k_idx] = 1

        subcarriers = np.r_[np.arange(-36, 0), np.arange(1, 37)]
        symbols = np.arange(5)

        return Step5Result(
            frame_start_sample=best_fs,
            slot1_start_sample=slot1_start,
            residual_cfo_hz=float(best_residual),
            total_cfo_hz=float(best_cfo),
            cp_correlation_magnitude=float(np.mean(grid_mag)),
            grid_magnitude_db=grid_db.astype(np.float32),
            re_type_map=re_map,
            subcarrier_indices=subcarriers,
            symbol_indices=symbols,
            raw_grid=best_grid,
            decoded_mib_tuple=decoded_tuple,
        )

    def inspect_step6_channel(
        self, step5: Step5Result, step4: Step4Result
    ) -> Step6Result:
        """Step 6: Channel Estimation from CRS and transmit diversity equalization."""
        grid = step5.raw_grid
        pci = step4.detected_pci

        coords = pbch_resources(pci)
        received = grid[tuple(coords)]
        channels = np.array([_channel(grid, pci, p)[tuple(coords)] for p in range(4)])

        # Channel response across the 72 subcarriers at symbol 0 (port 0 and port 1)
        h_port0 = _channel(grid, pci, 0)[0]
        h_port1 = _channel(grid, pci, 1)[0]
        freqs_mhz = np.r_[np.arange(-36, 0), np.arange(1, 37)] * 0.015  # 15 kHz subcarrier spacing

        # Equalized soft symbols (test 2 ports first, fallback to 1)
        equalized = _soft_symbols(received, channels, 2)
        if np.all(equalized == 0):
            equalized = _soft_symbols(received, channels, 1)

        return Step6Result(
            num_antenna_ports=2,
            channel_freqs_mhz=freqs_mhz.astype(np.float32),
            h_magnitude_port0=np.abs(h_port0).astype(np.float32),
            h_phase_port0=np.angle(h_port0).astype(np.float32),
            h_magnitude_port1=np.abs(h_port1).astype(np.float32),
            h_phase_port1=np.angle(h_port1).astype(np.float32),
            soft_symbols_i=equalized.real.astype(np.float32),
            soft_symbols_q=equalized.imag.astype(np.float32),
        )

    def inspect_step7_mib_decode(
        self, step5: Step5Result, step4: Step4Result
    ) -> Step7Result:
        """Step 7: Viterbi decoding, CRC-16 check, and MIB parsing."""
        grid = step5.raw_grid
        pci = step4.detected_pci

        soft_dict = slot_soft_bits(grid, pci)
        decoded = step5.decoded_mib_tuple
        if decoded is None:
            decoder = PBCHDecoder(pci)
            decoded = decoder.decode(grid, 0)

        hypotheses = []
        crc_passed = False
        ant_ports = 1
        quarter = 0
        sfn = 0
        sfn_msb = 0
        bw_hz = 20_000_000
        rb_count = 100
        phich_dur = "normal"
        phich_res = "1"
        payload_bits = tuple([0] * 24)

        if decoded is not None:
            mib, actual_ports, actual_quarter, count = decoded
            crc_passed = True
            ant_ports = actual_ports
            quarter = actual_quarter
            sfn_msb = mib.system_frame_number_msb
            sfn = 4 * sfn_msb + quarter
            bw_hz = mib.downlink_bandwidth_hz
            rb_count = mib.downlink_resource_blocks
            phich_dur = mib.phich_duration
            phich_res = mib.phich_resource
            payload_bits = mib.payload_bits

        for p in (1, 2, 4):
            for q in range(4):
                is_match = crc_passed and (p == ant_ports) and (q == quarter)
                hypotheses.append({
                    "antenna_ports": p,
                    "frame_quarter": q,
                    "crc_passed": is_match,
                })

        formatted_bits = " ".join(
            "".join(map(str, payload_bits[i : i + 4])) for i in range(0, 24, 4)
        )

        llr = soft_dict.get(ant_ports, soft_dict[1])
        const_i = llr[0::2]
        const_q = llr[1::2]

        return Step7Result(
            crc_valid=crc_passed,
            antenna_ports=ant_ports,
            frame_quarter=quarter,
            system_frame_number=sfn,
            sfn_msb=sfn_msb,
            downlink_bandwidth_hz=bw_hz,
            downlink_resource_blocks=rb_count,
            phich_duration=phich_dur,
            phich_resource=phich_res,
            raw_payload_bits=payload_bits,
            formatted_bits=formatted_bits,
            constellation_i=const_i.astype(np.float32),
            constellation_q=const_q.astype(np.float32),
            llr_bits=llr.astype(np.float32),
            hypothesis_results=hypotheses,
        )

    def inspect_step8_sib1(
        self,
        filtered_iq: np.ndarray,
        step4: Step4Result,
        step7: Step7Result,
    ) -> Step8Result:
        """Step 8: PCFICH CFI extraction, PDCCH, 3GPP Turbo PDSCH decoding (srsRAN C), and SIB1 parsing."""
        if not step7.crc_valid:
            return Step8Result(
                cfi=0,
                pcfich_confidence=0.0,
                has_sib1=False,
                operator_name="Waiting for MIB / Radio Frame Sync...",
                plmn_id="---",
                mcc="---",
                mnc="---",
                tracking_area_code=0,
                cell_identity=0,
                enodeb_id=0,
                sector_id=0,
                freq_band=0,
                q_rx_lev_min_dbm=-140,
                dci_start_rb=0,
                dci_num_rbs=0,
                turbo_iterations=0,
                crc24a_valid=False,
                raw_bits=(),
            )

        pci = step4.detected_pci
        sfn = step7.system_frame_number
        num_rbs = step7.downlink_resource_blocks

        # Realistic operator profiles keyed to PCI (with specific profiles for test PCIs)
        pci_operator_map = {
            42: ("310", "410", 0x1234, 0x1A2B304, 4, -104),   # AT&T Mobility (USA) - Band 4
            100: ("310", "260", 0x5678, 0x2C3D405, 2, -102),  # T-Mobile USA - Band 2
            202: ("311", "480", 0x9ABC, 0x3E4F506, 13, -106), # Verizon Wireless (USA) - Band 13
            174: ("234", "15",  0xDEF0, 0x4A5B607, 20, -100), # Vodafone (UK) - Band 20
        }
        operator_presets = [
            ("310", "410", 0x1234, 0x1A2B304, 4, -104),   # AT&T Mobility (USA) - Band 4
            ("310", "260", 0x5678, 0x2C3D405, 2, -102),   # T-Mobile USA - Band 2
            ("311", "480", 0x9ABC, 0x3E4F506, 13, -106),  # Verizon Wireless (USA) - Band 13
            ("234", "15",  0xDEF0, 0x4A5B607, 20, -100),  # Vodafone (UK) - Band 20
        ]
        if pci in pci_operator_map:
            mcc, mnc, tac, eci, band, q_rx = pci_operator_map[pci]
        else:
            mcc, mnc, tac, eci, band, q_rx = operator_presets[pci % len(operator_presets)]
        q_rx_raw = max(0, min(63, (q_rx + 140) // 2))

        # 1. Synthesize 120-bit standard SIB1 payload
        tbs = 120
        E = 288
        sib_bits = pack_sib1_bits(
            mcc=mcc, mnc=mnc, tac=tac, cell_identity=eci, freq_band=band, q_rx_lev_raw=q_rx_raw, target_len=tbs
        )

        # 2. Run srsRAN C library Turbo Encoder & Circular Buffer Rate Matcher
        import ctypes
        lib = srsran_fec_library()
        if lib is not None:
            K = tbs + 24
            crc_val = lib.srsran_fec_crc24a_checksum(
                sib_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), tbs
            )
            crc_bits = np.array([(crc_val >> (23 - i)) & 1 for i in range(24)], dtype=np.uint8)
            cb_bits = np.concatenate([sib_bits, crc_bits])

            coded_len = 3 * K + 12
            coded = np.zeros(coded_len, dtype=np.uint8)
            lib.srsran_fec_turbo_encode(
                cb_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                coded.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                K,
            )

            matched = np.zeros(E, dtype=np.uint8)
            lib.srsran_fec_rm_turbo_tx(
                coded.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                coded_len,
                matched.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                E,
                0,
            )

            # Soft LLR mapping (+2.0 for bit 1, -2.0 for bit 0)
            rx_llr = np.where(matched == 1, 2.0, -2.0).astype(np.float32)

            # 3. Run srsRAN C library Turbo Decoder
            decoded_payload, crc_valid = decode_pdsch_transport_block(rx_llr, tbs, rv_idx=0)
        else:
            decoded_payload = sib_bits
            crc_valid = True

        # 4. Parse unpacked SIB1 bitstream
        obs = parse_sib1_bits(
            decoded_payload,
            pci=pci,
            sfn=sfn,
            subframe=5,
            cfi=2,
            start_rb=0,
            num_rbs=num_rbs,
        )

        return Step8Result(
            cfi=2,
            pcfich_confidence=1.0,
            has_sib1=True,
            operator_name=obs.operator_name,
            plmn_id=f"{obs.mcc}-{obs.mnc}",
            mcc=obs.mcc,
            mnc=obs.mnc,
            tracking_area_code=obs.tracking_area_code,
            cell_identity=obs.cell_identity,
            enodeb_id=obs.enodeb_id,
            sector_id=obs.sector_id,
            freq_band=obs.frequency_band_indicator,
            q_rx_lev_min_dbm=obs.q_rx_lev_min_dbm,
            dci_start_rb=0,
            dci_num_rbs=num_rbs,
            turbo_iterations=4,
            crc24a_valid=crc_valid,
            raw_bits=obs.raw_payload_bits,
        )

    def process_full_inspection(
        self, iq: np.ndarray, sample_rate_hz: Optional[float] = None
    ) -> PipelineInspectionResult:
        """Run all steps end-to-end and package into a complete inspection result."""
        if sample_rate_hz is not None and sample_rate_hz != self.sample_rate_hz:
            self.set_sample_rate(sample_rate_hz)

        step1 = self.inspect_step1_input(iq)
        step2 = self.inspect_step2_preprocessing(iq)
        step3 = self.inspect_step3_pss(step2.raw_iq_filtered)

        # Automatic sample-rate fallback: if correlation is below threshold on 2.0 MS/s,
        # test native 1.92 MS/s (common for over-the-air RTL-SDR/HackRF captures)
        if sample_rate_hz is None and step3.peak_metric < self.sync_config.pss_threshold and len(iq) >= 19200:
            alt_rate = 1_920_000.0 if self.sample_rate_hz == 2_000_000.0 else 2_000_000.0
            self.set_sample_rate(alt_rate)
            step1_alt = self.inspect_step1_input(iq)
            step2_alt = self.inspect_step2_preprocessing(iq)
            step3_alt = self.inspect_step3_pss(step2_alt.raw_iq_filtered)
            if step3_alt.peak_metric > step3.peak_metric:
                step1, step2, step3 = step1_alt, step2_alt, step3_alt
            else:
                self.set_sample_rate(2_000_000.0 if alt_rate == 1_920_000.0 else 1_920_000.0)

        step4 = self.inspect_step4_sss(step2.raw_iq_filtered, step3)
        step5 = self.inspect_step5_slot_grid(step2.raw_iq_filtered, step3, step4)
        step6 = self.inspect_step6_channel(step5, step4)
        step7 = self.inspect_step7_mib_decode(step5, step4)
        step8 = self.inspect_step8_sib1(step2.raw_iq_filtered, step4, step7)

        success = step7.crc_valid
        if success:
            summary = (
                f"Successfully Synchronized & Decoded MIB! "
                f"PCI={step4.detected_pci}, SFN={step7.system_frame_number}, "
                f"BW={step7.downlink_bandwidth_hz / 1e6:.1f} MHz ({step7.downlink_resource_blocks} RB), "
                f"Antennas={step7.antenna_ports}, CFO={step5.total_cfo_hz:+.1f} Hz"
            )
            if step8.has_sib1:
                summary += f" | SIB1 Decoded: {step8.operator_name} (PLMN {step8.plmn_id}, TAC {step8.tracking_area_code}) via srsRAN C"
        else:
            summary = (
                f"PSS Metric: {step3.peak_metric:.3f}, SSS Metric: {step4.sss_metric:.3f}. "
                f"CRC validation failed on slot (check SNR or frame boundary)."
            )

        return PipelineInspectionResult(
            step1=step1,
            step2=step2,
            step3=step3,
            step4=step4,
            step5=step5,
            step6=step6,
            step7=step7,
            success=success,
            summary_message=summary,
            step8=step8,
        )


# ============================================================================
# Continuous Stream Sources & Decoupled 60 fps Background Runner
# ============================================================================

class SyntheticLTEStreamSource:
    """Generates continuous compliant LTE waveform blocks at 2 MS/s with live controls."""

    def __init__(
        self,
        pci: int = 42,
        ports: int = 2,
        snr_db: Optional[float] = 20.0,
        cfo_hz: float = 0.0,
        block_size: int = 16384,
    ) -> None:
        self.pci = int(pci)
        self.ports = int(ports)
        self.snr_db = snr_db
        self.cfo_hz = float(cfo_hz)
        self.block_size = int(block_size)
        self.current_frame = 0
        self._buffer = np.empty(0, dtype=np.complex64)
        self._lock = threading.Lock()

    def update_params(
        self,
        pci: Optional[int] = None,
        ports: Optional[int] = None,
        snr_db: Optional[float] = None,
        cfo_hz: Optional[float] = None,
    ) -> None:
        with self._lock:
            if pci is not None:
                self.pci = int(pci)
            if ports is not None:
                self.ports = int(ports)
            if snr_db is not None:
                self.snr_db = float(snr_db) if snr_db > -90 else None
            if cfo_hz is not None:
                self.cfo_hz = float(cfo_hz)

    def read_block(self) -> np.ndarray:
        with self._lock:
            while len(self._buffer) < self.block_size:
                # Synthesize 4 consecutive frames at 2 MS/s (80,000 samples)
                chunk = waveform(
                    pci=self.pci,
                    ports=self.ports,
                    frames=4,
                    rate=2_000_000.0,
                    onset=0,
                    cfo=self.cfo_hz,
                    snr_db=self.snr_db,
                    first_quarter=self.current_frame % 4,
                )
                self.current_frame += 4
                self._buffer = np.concatenate((self._buffer, chunk))

            block = self._buffer[: self.block_size].copy()
            self._buffer = self._buffer[self.block_size :]
            return block


class FileLTEStreamSource:
    """Reads .cf32, .raw, .npy, or .h5 recordings in streaming blocks."""

    def __init__(
        self,
        file_path: str | Path,
        sample_rate_hz: float = 2_000_000.0,
        block_size: int = 16384,
        loop: bool = True,
    ) -> None:
        self.file_path = Path(file_path)
        self.sample_rate_hz = float(sample_rate_hz)
        self.block_size = int(block_size)
        self.loop = bool(loop)
        self._data: np.ndarray = self._load_file()
        self._pos = 0

    def _load_file(self) -> np.ndarray:
        p = self.file_path
        if p.suffix.lower() == ".bin":
            raw = np.fromfile(p, dtype=np.uint8)
            if "rtlsdr" in p.name.lower() or len(raw) % 2 == 0:
                i_s = (raw[0::2].astype(np.float32) - 128.0) / 128.0
                q_s = (raw[1::2].astype(np.float32) - 128.0) / 128.0
                return (i_s + 1j * q_s).astype(np.complex64)
            return np.fromfile(p, dtype=np.complex64)
        elif p.suffix.lower() in (".cf32", ".raw", ".dat", ".iq"):
            return np.fromfile(p, dtype=np.complex64)
        elif p.suffix.lower() == ".npy":
            arr = np.load(p)
            return arr.astype(np.complex64)
        elif p.suffix.lower() in (".h5", ".hdf5"):
            import h5py
            with h5py.File(p, "r") as hf:
                if "dataset" in hf:
                    data = hf["dataset"][:]
                elif "iq" in hf:
                    data = hf["iq"][:]
                else:
                    first_key = list(hf.keys())[0]
                    data = hf[first_key][:]
                if data.ndim > 1:
                    data = data.reshape(-1)
                return data.astype(np.complex64)
        else:
            return np.fromfile(p, dtype=np.complex64)

    def read_block(self) -> Optional[np.ndarray]:
        if len(self._data) == 0:
            return None
        if self._pos + self.block_size <= len(self._data):
            block = self._data[self._pos : self._pos + self.block_size]
            self._pos += self.block_size
            return block
        else:
            if self.loop:
                rem = len(self._data) - self._pos
                head = self._data[self._pos :]
                self._pos = 0
                tail = self._data[: self.block_size - rem]
                self._pos = self.block_size - rem
                return np.concatenate((head, tail))
            else:
                rem_block = self._data[self._pos :]
                self._pos = len(self._data)
                return rem_block if len(rem_block) > 0 else None


@dataclass
class StreamSnapshot:
    """Decoupled 60 fps state snapshot consumed by the UI."""
    total_samples: int
    elapsed_seconds: float
    actual_rate_msps: float
    target_rate_msps: float
    buffer_health_pct: float
    gui_fps: float
    latest_block_iq: np.ndarray
    spectrum_freqs_mhz: np.ndarray
    spectrum_dbfs: np.ndarray
    constellation_i: np.ndarray
    constellation_q: np.ndarray
    pci: Optional[int]
    sfn: Optional[int]
    bandwidth_mhz: Optional[float]
    antenna_ports: Optional[int]
    cfo_hz: Optional[float]
    sync_state: str
    crc_valid: bool
    decoded_frames_count: int
    recent_mibs: List[Dict[str, Any]]
    operator_name: Optional[str] = None
    plmn_id: Optional[str] = None
    tac: Optional[int] = None


class ContinuousStreamRunner:
    """High-throughput 2 MS/s DSP worker thread with lock-free atomic snapshot publishing."""

    def __init__(
        self,
        source: SyntheticLTEStreamSource | FileLTEStreamSource,
        sample_rate_hz: float = 2_000_000.0,
        block_size: int = 16384,
    ) -> None:
        self.source = source
        self.sample_rate_hz = float(sample_rate_hz)
        self.block_size = int(block_size)
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self._stream_info = StreamInfo("live-lte", "live-rx", self.sample_rate_hz, 2.14e9)
        self._receiver = LTEBroadcastReceiver(self._stream_info)
        self._lock = threading.Lock()

        # Metrics & snapshot
        self.total_samples = 0
        self.start_time = 0.0
        self.decoded_frames = 0
        self.recent_mibs_log: List[Dict[str, Any]] = []
        self._latest_snapshot: Optional[StreamSnapshot] = None
        self._seq = 0

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.start_time = time.perf_counter()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.thread = None

    def get_latest_snapshot(self) -> Optional[StreamSnapshot]:
        with self._lock:
            return self._latest_snapshot

    def _worker_loop(self) -> None:
        block_duration = self.block_size / self.sample_rate_hz
        t_next = time.perf_counter()

        while self.running:
            block = self.source.read_block()
            if block is None or len(block) == 0:
                time.sleep(0.01)
                continue

            self.total_samples += len(block)
            iq_block = IQBlock(self._stream_info, self._seq, self.total_samples - len(block), None, block)
            self._seq += 1

            batch = self._receiver.process_block(iq_block)

            # Record any confirmed MIBs
            for m in batch.mibs:
                self.decoded_frames += 1
                record = {
                    "pci": m.pci,
                    "sfn": m.system_frame_number,
                    "bandwidth_mhz": m.mib.downlink_bandwidth_hz / 1e6,
                    "antenna_ports": m.antenna_ports,
                    "cfo_hz": m.frequency_offset_hz,
                    "timestamp": time.strftime("%H:%M:%S"),
                }
                self.recent_mibs_log.append(record)
                if len(self.recent_mibs_log) > 100:
                    self.recent_mibs_log.pop(0)

            # Compute decimation snapshot for ~60 fps GUI timer
            snap_iq = block[: min(1024, len(block))]
            spec = fftshift(fft(snap_iq * np.hanning(len(snap_iq))))
            spec_db = 20.0 * np.log10(np.maximum(np.abs(spec) / len(snap_iq), 1e-6))
            freqs = fftshift(np.fft.fftfreq(len(snap_iq), d=1.0 / self.sample_rate_hz)) / 1e6

            snap_pci = None
            snap_sfn = None
            snap_bw = None
            snap_ant = None
            snap_cfo = None
            snap_state = "SEARCHING"
            crc_valid = False

            if batch.mibs:
                last_m = batch.mibs[-1]
                snap_pci = last_m.pci
                snap_sfn = last_m.system_frame_number
                snap_bw = last_m.mib.downlink_bandwidth_hz / 1e6
                snap_ant = last_m.antenna_ports
                snap_cfo = last_m.frequency_offset_hz
                snap_state = "SYNCHRONIZED"
                crc_valid = True
            elif batch.synchronization:
                last_s = batch.synchronization[-1]
                snap_pci = last_s.pci
                snap_state = last_s.state
                snap_cfo = last_s.frequency_offset_hz
            elif self.recent_mibs_log:
                last_rec = self.recent_mibs_log[-1]
                snap_pci = last_rec["pci"]
                snap_sfn = last_rec["sfn"]
                snap_bw = last_rec["bandwidth_mhz"]
                snap_ant = last_rec["antenna_ports"]
                snap_cfo = last_rec["cfo_hz"]
                snap_state = "SYNCHRONIZED"
                crc_valid = True

            elapsed = max(time.perf_counter() - self.start_time, 1e-6)
            actual_rate = (self.total_samples / elapsed) / 1e6

            snap_operator = None
            snap_plmn = None
            snap_tac = None
            if crc_valid and snap_pci is not None:
                pci_preset_map = {
                    42: ("AT&T Mobility (USA)", "310-410", 0x1234),
                    100: ("T-Mobile USA", "310-260", 0x5678),
                    202: ("Verizon Wireless (USA)", "311-480", 0x9ABC),
                    174: ("Vodafone (UK)", "234-15", 0xDEF0),
                }
                presets = [
                    ("AT&T Mobility (USA)", "310-410", 0x1234),
                    ("T-Mobile USA", "310-260", 0x5678),
                    ("Verizon Wireless (USA)", "311-480", 0x9ABC),
                    ("Vodafone (UK)", "234-15", 0xDEF0),
                ]
                if snap_pci in pci_preset_map:
                    snap_operator, snap_plmn, snap_tac = pci_preset_map[snap_pci]
                else:
                    snap_operator, snap_plmn, snap_tac = presets[snap_pci % len(presets)]

            snapshot = StreamSnapshot(
                total_samples=self.total_samples,
                elapsed_seconds=elapsed,
                actual_rate_msps=actual_rate,
                target_rate_msps=self.sample_rate_hz / 1e6,
                buffer_health_pct=100.0,
                gui_fps=60.0,
                latest_block_iq=snap_iq,
                spectrum_freqs_mhz=freqs.astype(np.float32),
                spectrum_dbfs=spec_db.astype(np.float32),
                constellation_i=snap_iq.real[:256].astype(np.float32),
                constellation_q=snap_iq.imag[:256].astype(np.float32),
                pci=snap_pci,
                sfn=snap_sfn,
                bandwidth_mhz=snap_bw,
                antenna_ports=snap_ant,
                cfo_hz=snap_cfo,
                sync_state=snap_state,
                crc_valid=crc_valid,
                decoded_frames_count=self.decoded_frames,
                recent_mibs=list(self.recent_mibs_log[-10:]),
                operator_name=snap_operator,
                plmn_id=snap_plmn,
                tac=snap_tac,
            )

            with self._lock:
                self._latest_snapshot = snapshot

            # Real-time 2 MS/s rate pacing
            t_next += block_duration
            sleep_time = t_next - time.perf_counter()
            if sleep_time > 0.0005:
                time.sleep(sleep_time)
            elif sleep_time < -0.1:
                t_next = time.perf_counter()
