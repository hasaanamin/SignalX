"""Regression tests for Welch PSD estimation and reliable bandwidth measurements.

Requirement 6:
- Explicit Periodogram vs Welch PSD modes with window-energy normalization and two-sided frequencies.
- Full-event FFT vs Welch segment FFT.
- Controlled 100 kHz band-limited signal in 1 MHz capture at Clean, 20 dB, and 10 dB SNR.
- Accuracy tolerance: max(5% of true 99% OBW, 2 effective resolution elements).
- Dominant-peak 3 dB half-power width marked invalid on OFDM-like wideband spectrum.
- Reliable 3 dB bandwidth on signals with genuine dominant lobes.
"""

import math

import numpy as np
import pytest

from signal_processing.context import AnalysisConfig, SignalContext
from signal_processing.core import SignalEvent, StreamInfo
from signal_processing.features.bandwidth import (
    BandwidthFeatureConfig,
    BandwidthFeatureExtractor,
)
from signal_processing.transforms import (
    compute_welch_psd,
)


@pytest.fixture
def stream_1mhz() -> StreamInfo:
    return StreamInfo(
        stream_id="stream-bw-psd",
        source_id="synthetic",
        sample_rate_hz=1_000_000.0,
        center_frequency_hz=915_000_000.0,
    )


class TestPSDModesAndNormalization:
    """Validate Periodogram and Welch PSD calculation and normalization."""

    def test_welch_psd_vs_direct_independent_calculation(self) -> None:
        """Validate Welch PSD against an independent direct calculation."""
        fs = 1_000_000.0
        n_samples = 2048
        nperseg = 256
        noverlap = 128
        n = np.arange(n_samples)
        # 100 kHz complex tone
        iq = np.exp(2j * np.pi * 100_000.0 * n / fs).astype(np.complex64)

        psd_res = compute_welch_psd(
            iq,
            sample_rate_hz=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hann",
        )

        assert len(psd_res.frequency_offset_hz) == nperseg
        assert len(psd_res.power_linear) == nperseg
        assert np.all(psd_res.power_linear >= 0.0)

        # Independent manual calculation
        step = nperseg - noverlap
        n_segs = (n_samples - noverlap) // step
        win = np.hanning(nperseg).astype(np.float64)
        u = float(np.sum(win ** 2))

        seg_accum = np.zeros(nperseg, dtype=np.float64)
        for i in range(n_segs):
            seg_iq = iq[i * step : i * step + nperseg].astype(np.complex128) * win
            seg_fft = np.fft.fftshift(np.fft.fft(seg_iq, n=nperseg))
            seg_accum += (np.abs(seg_fft) ** 2) / (fs * u)

        expected_psd = seg_accum / float(n_segs)
        np.testing.assert_allclose(psd_res.power_linear, expected_psd, rtol=1e-5)

    def test_context_psd_mode_selection(self, stream_1mhz: StreamInfo) -> None:
        """SignalContext selects Welch PSD when configured with psd_mode='welch'."""
        iq = np.ones(1024, dtype=np.complex64)
        event = SignalEvent(
            event_id="ev-welch-ctx",
            stream=stream_1mhz,
            iq=iq,
            event_start_sample=0,
            event_end_sample=1024,
            detected_start_sample=0,
            detected_end_sample=1024,
            trigger_sample=0,
        )

        cfg_welch = AnalysisConfig(psd_mode="welch", psd_nperseg=256, psd_noverlap=128)
        ctx_welch = SignalContext(event, cfg_welch)
        psd_welch = ctx_welch.psd()

        # Welch PSD resolution equals nperseg
        assert len(psd_welch.frequency_offset_hz) == 256

        cfg_periodogram = AnalysisConfig(psd_mode="periodogram")
        ctx_periodogram = SignalContext(event, cfg_periodogram)
        psd_periodogram = ctx_periodogram.psd()

        # Periodogram PSD resolution equals signal length
        assert len(psd_periodogram.frequency_offset_hz) == 1024


class TestReliableBandwidthMeasurements:
    """Validate occupied bandwidth and half-power width across controlled SNR and wideband scenarios."""

    @staticmethod
    def generate_bandlimited_noise(
        fs: float,
        n_samples: int,
        target_bw_hz: float,
        snr_db: float | None = None,
        seed: int = 42,
    ) -> tuple[np.ndarray, float]:
        """Generate a controlled band-limited complex baseband signal with known true 99% OBW."""
        rng = np.random.default_rng(seed)
        # Generate white noise in frequency domain and brick-wall bandlimit
        freqs = np.fft.fftfreq(n_samples, d=1.0 / fs)
        spec = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
        # Keep within [-target_bw_hz/2, +target_bw_hz/2]
        band_mask = np.abs(freqs) <= (target_bw_hz / 2.0)
        spec[~band_mask] = 0.0

        clean_time = np.fft.ifft(spec)
        # Normalize clean signal power to 1.0
        p_clean = np.mean(np.abs(clean_time) ** 2)
        clean_time = clean_time / np.sqrt(p_clean)

        true_obw_hz = target_bw_hz * 0.99

        if snr_db is None:
            return clean_time.astype(np.complex64), true_obw_hz

        # Add AWGN noise
        noise_p = 10.0 ** (-snr_db / 10.0)
        noise = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) * np.sqrt(noise_p / 2.0)
        noisy_signal = clean_time + noise
        return noisy_signal.astype(np.complex64), true_obw_hz

    def test_controlled_obw_clean_20db_10db(self, stream_1mhz: StreamInfo) -> None:
        """Controlled 100 kHz band-limited signal in 1 MHz capture (N=16384).

        Validate reliable OBW at clean, 20 dB, and 10 dB SNR with ROI [-75 kHz, +75 kHz].
        Tolerance: max(5% of true OBW, 2 effective resolution elements).
        """
        fs = 1_000_000.0
        n_samples = 16384
        target_bw = 100_000.0
        df_eff = fs / n_samples
        roi = (-75_000.0, 75_000.0)

        for snr in [None, 20.0, 10.0]:
            iq, true_obw = self.generate_bandlimited_noise(
                fs=fs,
                n_samples=n_samples,
                target_bw_hz=target_bw,
                snr_db=snr,
            )
            event = SignalEvent(
                event_id=f"ev-bw-{snr}",
                stream=stream_1mhz,
                iq=iq,
                event_start_sample=0,
                event_end_sample=n_samples,
                detected_start_sample=0,
                detected_end_sample=n_samples,
                trigger_sample=0,
            )

            # Known noise floor or auto estimation outside ROI
            noise_floor_dbfs = -10.0 * np.log10(10.0 ** (snr / 10.0) * fs) if snr is not None else None
            bw_cfg = BandwidthFeatureConfig(
                roi_hz=roi,
                noise_floor_dbfs_per_hz=noise_floor_dbfs,
                interpolate_edges=True,
            )
            extractor = BandwidthFeatureExtractor(bw_cfg)
            fset = extractor.extract(SignalContext(event))

            measured_obw = fset.features["bandwidth.occupied_99_hz"].value
            assert fset.features["bandwidth.occupied_99_hz"].valid is True

            # Tolerance: max(5% of true 99% OBW, 2 effective resolution elements)
            tol = max(0.05 * true_obw, 2.0 * df_eff)
            err = abs(measured_obw - true_obw)
            assert err <= tol, f"At SNR={snr} dB, error {err:.1f} Hz exceeded tolerance {tol:.1f} Hz (measured {measured_obw:.1f}, true {true_obw:.1f})"

    def test_dominant_peak_half_power_invalid_on_wideband_ofdm(self, stream_1mhz: StreamInfo) -> None:
        """On a 100 kHz wideband random/OFDM signal, dominant peak 3 dB half-power width

        must NOT claim a misleading ~15-30 Hz bandwidth; it must be marked invalid.
        """
        fs = 1_000_000.0
        n_samples = 16384
        target_bw = 100_000.0
        iq, _ = self.generate_bandlimited_noise(fs=fs, n_samples=n_samples, target_bw_hz=target_bw, snr_db=20.0)

        event = SignalEvent(
            event_id="ev-ofdm-3db",
            stream=stream_1mhz,
            iq=iq,
            event_start_sample=0,
            event_end_sample=n_samples,
            detected_start_sample=0,
            detected_end_sample=n_samples,
            trigger_sample=0,
        )

        extractor = BandwidthFeatureExtractor()
        fset = extractor.extract(SignalContext(event))

        # 99% OBW is valid and roughly spans the wideband signal
        assert fset.features["bandwidth.occupied_99_hz"].valid is True
        assert fset.features["bandwidth.occupied_99_hz"].value > 80_000.0

        # 3 dB half-power measurement is marked invalid because no dominant lobe exists!
        three_db = fset.features["bandwidth.three_db_hz"]
        assert three_db.valid is False
        assert math.isnan(three_db.value)
        assert "no distinct dominant spectral lobe observed" in three_db.reason.lower()

    def test_dominant_peak_half_power_valid_on_distinct_tone(self, stream_1mhz: StreamInfo) -> None:
        """On a clean tone with a genuine dominant lobe, 3 dB bandwidth is valid."""
        fs = 1_000_000.0
        n_samples = 1024
        n = np.arange(n_samples)
        iq = np.exp(2j * np.pi * 50_000.0 * n / fs).astype(np.complex64)

        event = SignalEvent(
            event_id="ev-tone-3db",
            stream=stream_1mhz,
            iq=iq,
            event_start_sample=0,
            event_end_sample=n_samples,
            detected_start_sample=0,
            detected_end_sample=n_samples,
            trigger_sample=0,
        )

        extractor = BandwidthFeatureExtractor()
        fset = extractor.extract(SignalContext(event))

        three_db = fset.features["bandwidth.three_db_hz"]
        assert three_db.valid is True
        assert three_db.value > 0.0
