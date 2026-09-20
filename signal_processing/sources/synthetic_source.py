"""Synthetic streaming IQ source with controllable noise and scheduled signal bursts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from signal_processing.core.contracts import IQBlock, StreamInfo


@dataclass(frozen=True)
class BurstDescriptor:
    """Specification of a synthetic signal burst."""

    start_sample: int
    duration_samples: int
    snr_db: float
    modulation: Literal["cw", "bpsk", "qpsk", "chirp", "bandlimited_noise"] = "cw"
    freq_offset_hz: float = 0.0
    chirp_bandwidth_hz: float = 100_000.0
    bandwidth_hz: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.start_sample, bool) or not isinstance(self.start_sample, (int, np.integer)) or self.start_sample < 0:
            raise ValueError(f"start_sample must be an integer >= 0, got {self.start_sample!r}")
        if isinstance(self.duration_samples, bool) or not isinstance(self.duration_samples, (int, np.integer)) or self.duration_samples <= 0:
            raise ValueError(f"duration_samples must be an integer > 0, got {self.duration_samples!r}")
        if isinstance(self.snr_db, bool) or not isinstance(self.snr_db, (int, float, np.floating, np.integer)) or not math.isfinite(float(self.snr_db)):
            raise ValueError(f"snr_db must be a finite float, got {self.snr_db!r}")
        if isinstance(self.freq_offset_hz, bool) or not isinstance(self.freq_offset_hz, (int, float, np.floating, np.integer)) or not math.isfinite(float(self.freq_offset_hz)):
            raise ValueError(f"freq_offset_hz must be a finite float, got {self.freq_offset_hz!r}")
        if self.modulation not in ("cw", "bpsk", "qpsk", "chirp", "bandlimited_noise"):
            raise ValueError(f"Unknown modulation '{self.modulation}'. Supported: 'cw', 'bpsk', 'qpsk', 'chirp', 'bandlimited_noise'")
        bw = self.bandwidth_hz if self.bandwidth_hz is not None else self.chirp_bandwidth_hz
        if isinstance(bw, bool) or not isinstance(bw, (int, float, np.floating, np.integer)) or not math.isfinite(float(bw)) or bw <= 0.0:
            raise ValueError(f"bandwidth must be a strictly positive finite float, got {bw!r}")


class SyntheticBurstSource:
    """Streaming IQ source generating circular complex Gaussian noise and scheduled bursts.

    Enables deterministic, reproducible testing of continuous streaming detection,
    segmentation, and circular buffer retention without SDR hardware.
    """

    def __init__(
        self,
        stream_info: StreamInfo,
        total_samples: int = 100_000,
        block_size: int = 1024,
        noise_power_linear: float = 1.0,
        bursts: list[BurstDescriptor] | None = None,
        seed: int = 42,
        start_sample: int = 0,
        start_timestamp_ns: int | None = None,
    ) -> None:
        if not isinstance(stream_info, StreamInfo):
            raise TypeError(f"stream_info must be a StreamInfo instance, got {type(stream_info)}")
        if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
            raise ValueError(f"block_size must be an integer >= 1, got {block_size!r}")
        if isinstance(total_samples, bool) or not isinstance(total_samples, (int, np.integer)) or total_samples < 1:
            raise ValueError(f"total_samples must be an integer >= 1, got {total_samples!r}")
        if (
            isinstance(noise_power_linear, bool)
            or not isinstance(noise_power_linear, (int, float, np.floating, np.integer))
            or noise_power_linear <= 0.0
            or not math.isfinite(float(noise_power_linear))
        ):
            raise ValueError(f"noise_power_linear must be a positive finite float, got {noise_power_linear!r}")
        if isinstance(start_sample, bool) or not isinstance(start_sample, (int, np.integer)) or start_sample < 0:
            raise ValueError(f"start_sample must be an integer >= 0, got {start_sample!r}")

        self._stream_info = stream_info
        self._total_samples = int(total_samples)
        self._block_size = int(block_size)
        self._noise_power_linear = float(noise_power_linear)
        self._bursts = list(bursts or [])
        self._seed = int(seed)
        self._start_sample = int(start_sample)
        self._start_timestamp_ns = int(start_timestamp_ns) if start_timestamp_ns is not None else None

        self._cursor: int = 0
        self._sequence_number: int = 0
        self._data: np.ndarray | None = None
        self._generate_full_stream()

    @property
    def stream_info(self) -> StreamInfo:
        """Stream metadata describing this source."""
        return self._stream_info

    @property
    def total_samples(self) -> int:
        """Total number of samples generated."""
        return self._total_samples

    @property
    def bursts(self) -> list[BurstDescriptor]:
        """List of scheduled burst descriptors."""
        return list(self._bursts)

    def _generate_full_stream(self) -> None:
        rng = np.random.default_rng(self._seed)
        n = self._total_samples
        fs = self._stream_info.sample_rate_hz

        # Stationary circular complex Gaussian noise: CN(0, sigma2)
        std_1d = math.sqrt(self._noise_power_linear / 2.0)
        noise = (rng.normal(0.0, std_1d, n) + 1j * rng.normal(0.0, std_1d, n)).astype(np.complex64)

        signal = np.zeros(n, dtype=np.complex128)

        for b in self._bursts:
            b_start = b.start_sample
            b_end = min(b_start + b.duration_samples, n)
            if b_start >= n or b_end <= b_start:
                continue
            b_len = b_end - b_start
            t = np.arange(b_len, dtype=np.float64) / fs

            # Signal power: P_sig = P_noise * 10^(SNR/10)
            sig_power_linear = self._noise_power_linear * (10.0 ** (b.snr_db / 10.0))
            amp = math.sqrt(sig_power_linear)

            if b.modulation == "cw":
                burst_iq = amp * np.exp(2j * np.pi * b.freq_offset_hz * t)
            elif b.modulation == "bpsk":
                num_syms = max(1, b_len // 16)
                syms = rng.choice([-1.0, 1.0], size=num_syms)
                sym_seq = np.repeat(syms, int(math.ceil(b_len / num_syms)))[:b_len]
                burst_iq = amp * sym_seq * np.exp(2j * np.pi * b.freq_offset_hz * t)
            elif b.modulation == "qpsk":
                num_syms = max(1, b_len // 16)
                syms = (rng.choice([-1.0, 1.0], size=num_syms) + 1j * rng.choice([-1.0, 1.0], size=num_syms)) / math.sqrt(2.0)
                sym_seq = np.repeat(syms, int(math.ceil(b_len / num_syms)))[:b_len]
                burst_iq = amp * sym_seq * np.exp(2j * np.pi * b.freq_offset_hz * t)
            elif b.modulation == "chirp":
                bw = float(b.bandwidth_hz if b.bandwidth_hz is not None else b.chirp_bandwidth_hz)
                f0 = b.freq_offset_hz - bw / 2.0
                k_rate = bw / (b.duration_samples / fs)
                burst_iq = amp * np.exp(2j * np.pi * (f0 * t + 0.5 * k_rate * t**2))
            elif b.modulation == "bandlimited_noise":
                bw = float(b.bandwidth_hz if b.bandwidth_hz is not None else b.chirp_bandwidth_hz)
                raw_noise = rng.normal(0.0, 1.0, b_len) + 1j * rng.normal(0.0, 1.0, b_len)
                # Frequency domain spectral shaping / bandpass filter
                spec = np.fft.fft(raw_noise)
                freqs = np.fft.fftfreq(b_len, d=1.0 / fs)
                half_bw = bw / 2.0
                mask = np.abs(freqs) <= half_bw
                spec_filtered = spec * mask
                shaped = np.fft.ifft(spec_filtered)
                p_shaped = float(np.mean(np.abs(shaped) ** 2))
                if p_shaped > 0.0:
                    shaped = (shaped / math.sqrt(p_shaped)) * amp
                else:
                    shaped = np.zeros(b_len, dtype=np.complex128)
                burst_iq = shaped * np.exp(2j * np.pi * b.freq_offset_hz * t)
            else:
                raise ValueError(f"Unsupported modulation '{b.modulation}'")

            signal[b_start:b_end] += burst_iq

        total_iq = (noise + signal).astype(np.complex64)
        self._data = total_iq

    def read_block(self) -> IQBlock | None:
        """Read and return next sequential IQBlock, or None on EOF."""
        if self._data is None or self._cursor >= self._total_samples:
            return None

        end_idx = min(self._cursor + self._block_size, self._total_samples)
        block_iq = self._data[self._cursor : end_idx]
        block_start_sample = self._start_sample + self._cursor

        block_ts_ns: int | None = None
        if self._start_timestamp_ns is not None:
            offset_s = self._cursor / self._stream_info.sample_rate_hz
            block_ts_ns = self._start_timestamp_ns + int(round(offset_s * 1e9))

        block = IQBlock(
            stream=self._stream_info,
            sequence_number=self._sequence_number,
            start_sample=block_start_sample,
            timestamp_ns=block_ts_ns,
            iq=block_iq,
            dropped_samples_before=0,
            overflow=False,
        )

        self._cursor = end_idx
        self._sequence_number += 1
        return block

    def reset(self) -> None:
        """Reset stream position and sequence counter to 0."""
        self._cursor = 0
        self._sequence_number = 0
