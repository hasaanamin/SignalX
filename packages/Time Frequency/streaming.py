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

import math
import warnings
from typing import Any
import numpy as np

from .base import TimeFrequencyResult, WaveletResult
from ._common import _as_iq_array, _get_window, CWTFilterBank, calculate_morse_support_samples


class StreamingSTFT:
    """
    Real-Time Stateful Streaming Short-Time Fourier Transform (STFT) Engine matching MATLAB `dsp.STFT`.
    Maintains persistent overlap buffer across arbitrary chunk boundaries.
    Streaming a continuous recording chunk-by-chunk produces mathematically identical STFT frames
    and timestamps compared to batch processing.

    Architecture:
      previous unconsumed samples + new IQ chunk -> working buffer
      While len(working_buffer) >= window_length:
        X[k, m] = fftshift(fft(x_m * w, N_fft))
        t_m = (n_{start, m} + (M - 1)/2) / F_s
        advance working_buffer by hop size R = M - L (not by M)
      Retain remainder for next input chunk.

    Downstream Metrics:
      Provides zero-recalculation streaming outputs for:
        - Spectrogram (|X|^2)
        - Frequency Trajectory (first conditional spectral moment)
        - Instantaneous Bandwidth (second centralized conditional spectral moment)
        - Time-Frequency Entropy (Shannon entropy per frame)
        - Time-Frequency Centroid accumulator
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        window_length: int = 256,
        overlap: float | int = 0.75,
        nfft: int | None = None,
        window: str | np.ndarray = "hann",
    ) -> None:
        self.sample_rate = float(sample_rate)
        if self.sample_rate <= 0 or not np.isfinite(self.sample_rate):
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")

        self.window_length = int(window_length)
        if self.window_length <= 0:
            raise ValueError(f"window_length must be > 0, got {window_length}")

        if isinstance(overlap, float):
            if not (0.0 <= overlap < 1.0):
                raise ValueError(f"Float overlap fraction must be in [0, 1), got {overlap}")
            self.overlap_samples = int(round(self.window_length * overlap))
        else:
            self.overlap_samples = int(overlap)
            if not (0 <= self.overlap_samples < self.window_length):
                raise ValueError(f"Integer overlap samples must satisfy 0 <= overlap < window_length ({self.window_length}), got {overlap}")

        self.hop_samples = max(1, self.window_length - self.overlap_samples)

        if nfft is None:
            self.nfft = max(self.window_length, 256)
        else:
            self.nfft = int(nfft)
            if self.nfft < self.window_length:
                raise ValueError(f"nfft ({self.nfft}) must be >= window_length ({self.window_length})")

        self.window_vector, self.window_name = _get_window(window, self.window_length)

        # Precompute centered frequency axis [-Fs/2, Fs/2)
        k_indices = np.arange(self.nfft)
        self.frequencies_hz = (k_indices - (self.nfft // 2)) * (self.sample_rate / self.nfft)

        # Persistent streaming state
        self.overlap_buffer: np.ndarray = np.empty(0, dtype=np.complex128)
        self.absolute_sample_index: int = 0  # Global sample index of stream start
        self.next_stft_frame_start: int = 0  # Absolute sample index of next STFT frame start
        self.total_frames_emitted: int = 0

        # Accumulators for stream-wide metrics (e.g. TF centroid)
        self._accum_weighted_time: float = 0.0
        self._accum_weighted_freq: float = 0.0
        self._accum_total_energy: float = 0.0

    @property
    def buffered_sample_count(self) -> int:
        """Returns the number of unconsumed buffered samples currently held."""
        return len(self.overlap_buffer)

    def reset(self) -> None:
        """Resets all streaming buffers, index tracking, and accumulators."""
        self.overlap_buffer = np.empty(0, dtype=np.complex128)
        self.absolute_sample_index = 0
        self.next_stft_frame_start = 0
        self.total_frames_emitted = 0
        self._accum_weighted_time = 0.0
        self._accum_weighted_freq = 0.0
        self._accum_total_energy = 0.0

    def process(self, chunk: Any, *, final_chunk: bool = False, eps: float = 1e-18) -> TimeFrequencyResult:
        """
        Processes an incoming chunk of complex IQ samples in real time.

        Parameters:
            chunk: New complex IQ array chunk.
            final_chunk: If True, indicates stream termination.
            eps: Epsilon for log calculations.

        Returns:
            TimeFrequencyResult containing newly completed STFT frames for this chunk.
        """
        z_chunk = _as_iq_array(chunk) if chunk is not None and len(chunk) > 0 else np.empty(0, dtype=np.complex128)

        # Combine unconsumed overlap buffer with new incoming samples
        if len(self.overlap_buffer) > 0:
            working_buffer = np.concatenate([self.overlap_buffer, z_chunk])
        else:
            working_buffer = z_chunk

        n_work = len(working_buffer)
        m_win = self.window_length
        hop_r = self.hop_samples

        if n_work < m_win:
            # Not enough samples for a complete window; hold in buffer
            self.overlap_buffer = working_buffer.copy()
            return TimeFrequencyResult(
                stft_complex=np.empty((self.nfft, 0), dtype=np.complex128),
                power_spectrogram=np.empty((self.nfft, 0), dtype=np.float64),
                spectrogram_db=np.empty((self.nfft, 0), dtype=np.float64),
                frequencies_hz=self.frequencies_hz,
                times_sec=np.empty(0, dtype=np.float64),
                sample_rate=self.sample_rate,
                hop_samples=hop_r,
                window_length=m_win,
                nfft=self.nfft,
                window_name=self.window_name,
                sample_offset=self.next_stft_frame_start,
            )

        num_frames = (n_work - m_win) // hop_r + 1
        shape = (num_frames, m_win)
        strides = (working_buffer.strides[0] * hop_r, working_buffer.strides[0])
        frames = np.lib.stride_tricks.as_strided(working_buffer, shape=shape, strides=strides)

        # Window & centered FFT
        windowed = frames * self.window_vector
        stft_raw = np.fft.fft(windowed, n=self.nfft, axis=1)
        stft_centered = np.fft.fftshift(stft_raw, axes=1)
        stft_complex = stft_centered.T  # Shape: [nfft, num_frames]

        # Timestamps corresponding to window centers: t_m = (n_start,m + (M - 1)/2) / Fs
        frame_start_indices = self.next_stft_frame_start + np.arange(num_frames) * hop_r
        times_sec = (frame_start_indices + (m_win - 1) / 2.0) / self.sample_rate

        # Update persistent streaming indices
        consumed_samples = num_frames * hop_r
        self.next_stft_frame_start += consumed_samples
        self.overlap_buffer = working_buffer[consumed_samples:].copy()
        self.total_frames_emitted += num_frames

        # Linear power & dB spectrogram
        power_spectrogram = np.abs(stft_complex) ** 2
        spectrogram_db = 10.0 * np.log10(power_spectrogram + eps)

        # Update streaming accumulators for global centroid
        if power_spectrogram.size > 0:
            frame_energies = np.sum(power_spectrogram, axis=0)  # [num_frames]
            bin_energies = np.sum(power_spectrogram, axis=1)    # [nfft]
            self._accum_weighted_time += float(np.sum(times_sec * frame_energies))
            self._accum_weighted_freq += float(np.sum(self.frequencies_hz * bin_energies))
            self._accum_total_energy += float(np.sum(frame_energies))

        return TimeFrequencyResult(
            stft_complex=stft_complex,
            power_spectrogram=power_spectrogram,
            spectrogram_db=spectrogram_db,
            frequencies_hz=self.frequencies_hz,
            times_sec=times_sec,
            sample_rate=self.sample_rate,
            hop_samples=hop_r,
            window_length=m_win,
            nfft=self.nfft,
            window_name=self.window_name,
            sample_offset=self.next_stft_frame_start - consumed_samples,
        )

    def frequency_trajectory_from_result(
        self, tf_result: TimeFrequencyResult, trajectory_mode: str = "tfmoment", eps: float = 1e-18
    ) -> dict[str, Any]:
        """Derives frequency trajectory from already-computed STFT result without altering stream state."""
        p_linear = tf_result.power_spectrogram
        if p_linear.size == 0 or p_linear.shape[1] == 0:
            return {"frequency_trajectory_hz": np.empty(0, dtype=np.float64), "times_sec": tf_result.times_sec}

        mode_clean = str(trajectory_mode).lower().strip().replace("-", "_")
        if mode_clean in ("ridge", "peak", "argmax"):
            max_idx = np.argmax(p_linear, axis=0)
            f_inst = self.frequencies_hz[max_idx]
        else:
            den = np.sum(p_linear, axis=0) + eps
            num = np.sum(self.frequencies_hz[:, None] * p_linear, axis=0)
            f_inst = num / den

        return {
            "frequency_trajectory_hz": f_inst,
            "times_sec": tf_result.times_sec,
            "sample_rate": self.sample_rate,
        }

    def instantaneous_bandwidth_from_result(
        self, tf_result: TimeFrequencyResult, scale_factor: float | None = None, eps: float = 1e-18
    ) -> dict[str, Any]:
        """Derives instantaneous bandwidth from already-computed STFT result without altering stream state."""
        p_linear = tf_result.power_spectrogram
        if p_linear.size == 0 or p_linear.shape[1] == 0:
            return {
                "instantaneous_bandwidth_hz": np.empty(0, dtype=np.float64),
                "frequency_std_hz": np.empty(0, dtype=np.float64),
                "times_sec": tf_result.times_sec,
            }

        c_scale = float(scale_factor) if scale_factor is not None else math.sqrt(4.0 * math.pi)
        den = np.sum(p_linear, axis=0) + eps
        num = np.sum(self.frequencies_hz[:, None] * p_linear, axis=0)
        f_inst = num / den

        diff_f = self.frequencies_hz[:, None] - f_inst[None, :]
        var_f = np.sum((diff_f ** 2) * p_linear, axis=0) / den
        sigma_f = np.sqrt(np.maximum(0.0, var_f))
        ibw = c_scale * sigma_f

        return {
            "instantaneous_bandwidth_hz": ibw,
            "frequency_std_hz": sigma_f,
            "instantaneous_frequency_hz": f_inst,
            "times_sec": tf_result.times_sec,
            "sample_rate": self.sample_rate,
            "scale_factor": c_scale,
        }

    def entropy_from_result(
        self, tf_result: TimeFrequencyResult, eps: float = 1e-18
    ) -> dict[str, Any]:
        """Derives normalized and raw Shannon spectral entropy from STFT result without altering stream state."""
        p_linear = tf_result.power_spectrogram
        if p_linear.size == 0 or p_linear.shape[1] == 0:
            return {
                "tf_entropy_normalized": np.empty(0, dtype=np.float64),
                "tf_entropy_raw": np.empty(0, dtype=np.float64),
                "times_sec": tf_result.times_sec,
            }

        n_bins = p_linear.shape[0]
        max_entropy = math.log2(n_bins) if n_bins > 1 else 1.0

        sum_p = np.sum(p_linear, axis=0, keepdims=True) + eps
        prob = p_linear / sum_p
        h_raw = -np.sum(prob * np.log2(np.maximum(1e-300, prob)), axis=0)
        h_norm = np.clip(h_raw / max_entropy, 0.0, 1.0)

        return {
            "tf_entropy_normalized": h_norm,
            "tf_entropy_raw": h_raw,
            "times_sec": tf_result.times_sec,
        }

    def process_all(
        self,
        chunk: Any,
        *,
        final_chunk: bool = False,
        trajectory_mode: str = "tfmoment",
        scale_factor: float | None = None,
        eps: float = 1e-18,
    ) -> dict[str, Any]:
        """
        Ingests IQ chunk exactly ONCE and produces STFT frames plus all downstream metrics
        (Spectrogram, Trajectory, Instantaneous Bandwidth, Entropy) safely.
        """
        tf_res = self.process(chunk, final_chunk=final_chunk, eps=eps)
        traj = self.frequency_trajectory_from_result(tf_res, trajectory_mode=trajectory_mode, eps=eps)
        ibw = self.instantaneous_bandwidth_from_result(tf_res, scale_factor=scale_factor, eps=eps)
        ent = self.entropy_from_result(tf_res, eps=eps)

        return {
            "stft": tf_res,
            "spectrogram": tf_res.power_spectrogram,
            "spectrogram_db": tf_res.spectrogram_db,
            "trajectory": traj,
            "frequency_trajectory": traj,
            "instantaneous_bandwidth": ibw,
            "entropy": ent,
            "times_sec": tf_res.times_sec,
            "frequencies_hz": self.frequencies_hz,
        }

    def process_frequency_trajectory(self, chunk: Any, trajectory_mode: str = "tfmoment", eps: float = 1e-18) -> dict[str, Any]:
        """Processes chunk and yields newly completed frequency trajectory frames."""
        if isinstance(chunk, TimeFrequencyResult):
            return self.frequency_trajectory_from_result(chunk, trajectory_mode=trajectory_mode, eps=eps)
        tf_res = self.process(chunk, eps=eps)
        return self.frequency_trajectory_from_result(tf_res, trajectory_mode=trajectory_mode, eps=eps)

    def frequency_trajectory(self, chunk: Any, trajectory_mode: str = "tfmoment", eps: float = 1e-18) -> dict[str, Any]:
        """Alias for process_frequency_trajectory."""
        return self.process_frequency_trajectory(chunk, trajectory_mode=trajectory_mode, eps=eps)

    def process_instantaneous_bandwidth(self, chunk: Any, scale_factor: float | None = None, eps: float = 1e-18) -> dict[str, Any]:
        """Processes chunk and yields newly completed instantaneous bandwidth frames."""
        if isinstance(chunk, TimeFrequencyResult):
            return self.instantaneous_bandwidth_from_result(chunk, scale_factor=scale_factor, eps=eps)
        tf_res = self.process(chunk, eps=eps)
        return self.instantaneous_bandwidth_from_result(tf_res, scale_factor=scale_factor, eps=eps)

    def instantaneous_bandwidth(self, chunk: Any, scale_factor: float | None = None, eps: float = 1e-18) -> dict[str, Any]:
        """Alias for process_instantaneous_bandwidth."""
        return self.process_instantaneous_bandwidth(chunk, scale_factor=scale_factor, eps=eps)

    def process_time_frequency_entropy(self, chunk: Any, eps: float = 1e-18) -> dict[str, Any]:
        """Processes chunk and yields newly completed time-frequency Shannon entropy frames."""
        if isinstance(chunk, TimeFrequencyResult):
            return self.entropy_from_result(chunk, eps=eps)
        tf_res = self.process(chunk, eps=eps)
        return self.entropy_from_result(tf_res, eps=eps)

    def entropy(self, chunk: Any, eps: float = 1e-18) -> dict[str, Any]:
        """Alias for process_time_frequency_entropy."""
        return self.process_time_frequency_entropy(chunk, eps=eps)

    def get_accumulated_centroid(self) -> dict[str, float]:
        """Returns the global 2D time-frequency centroid across all streamed samples processed so far."""
        if self._accum_total_energy <= 0.0:
            return {"time_centroid_sec": 0.0, "frequency_centroid_hz": 0.0, "total_energy": 0.0}
        return {
            "time_centroid_sec": self._accum_weighted_time / self._accum_total_energy,
            "frequency_centroid_hz": self._accum_weighted_freq / self._accum_total_energy,
            "total_energy": self._accum_total_energy,
        }


class StreamingCWT:
    """
    Real-Time Stateful Block-Streaming Continuous Wavelet Transform (CWT) Engine.
    Uses an overlap-save context buffer architecture and precomputed CWTFilterBank
    to eliminate chunk boundary distortion and edge artifacts.

    Architecture:
      previous context buffer + new chunk -> CWT block
      -> compute CWT via precomputed CWTFilterBank
      -> discard edge-affected boundary coefficients
      -> emit valid center coefficients and advance stream tracking
    """

    def __init__(
        self,
        block_size: int = 1024,
        context_samples: int | str = "auto",
        sample_rate: float = 1.0,
        wavelet_name: str = "morse",
        voices_per_octave: int = 10,
        gamma: float = 3.0,
        time_bandwidth: float = 60.0,
        frequency_limits: tuple[float, float] | None = None,
        min_frequency_hz: float | None = None,
        max_frequency_hz: float | None = None,
        scales: np.ndarray | None = None,
        normalization: str = "L1",
        frequency_mode: str = "analytic_positive",
        support_tolerance: float = 1e-6,
    ) -> None:
        self.sample_rate = float(sample_rate)
        if self.sample_rate <= 0 or not np.isfinite(self.sample_rate):
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")

        self.wavelet_name = str(wavelet_name).lower().strip()
        if self.wavelet_name not in ("morse",):
            raise ValueError(f"Unsupported wavelet_name '{wavelet_name}'. Expected one of: 'morse'.")

        self.voices_per_octave = int(voices_per_octave)
        if self.voices_per_octave < 1:
            raise ValueError(f"voices_per_octave must be >= 1, got {voices_per_octave}")

        self.gamma = float(gamma)
        if self.gamma <= 0 or not np.isfinite(self.gamma):
            raise ValueError(f"gamma must be > 0, got {gamma}")

        self.time_bandwidth = float(time_bandwidth)
        if self.time_bandwidth <= 0 or not np.isfinite(self.time_bandwidth):
            raise ValueError(f"time_bandwidth must be > 0, got {time_bandwidth}")

        self.normalization = str(normalization).upper().strip()
        if self.normalization not in ("L1", "L2"):
            raise ValueError(f"Unsupported normalization '{normalization}'. Expected one of: 'L1', 'L2'.")

        self.frequency_mode = str(frequency_mode).lower().strip()
        if self.frequency_mode not in ("analytic_positive", "two_sided"):
            raise ValueError(f"Unsupported frequency_mode '{frequency_mode}'. Expected one of: 'analytic_positive', 'two_sided'.")

        # Determine target minimum frequency for context sizing
        if min_frequency_hz is not None:
            f_min_target = float(min_frequency_hz)
        elif frequency_limits is not None:
            f_min_target = float(frequency_limits[0])
        else:
            f_min_target = max(self.sample_rate / max(1024, int(block_size)) * 2.0, 5000.0)

        self.required_context_samples = calculate_morse_support_samples(
            self.sample_rate, f_min_target, self.gamma, self.time_bandwidth, support_tolerance
        )

        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")

        if isinstance(context_samples, str) and context_samples.lower().strip() == "auto":
            self.context_samples = max(64, self.required_context_samples)
            self.block_size = max(self.block_size, self.context_samples)
        else:
            self.context_samples = int(context_samples)
            if self.context_samples < 0 or self.context_samples >= self.block_size:
                raise ValueError(f"context_samples ({self.context_samples}) must be >= 0 and < block_size ({self.block_size})")
            if self.context_samples < self.required_context_samples:
                warnings.warn(
                    f"StreamingCWT: provided context_samples ({self.context_samples}) is less than recommended support "
                    f"({self.required_context_samples} samples) for min_frequency_hz={f_min_target:.1f}Hz; "
                    "edge/inter-block error may exceed tolerance.",
                    UserWarning,
                    stacklevel=2,
                )

        self.actual_context_samples = self.context_samples
        # Total processing FFT block length
        self.total_block_len = self.block_size + 2 * self.context_samples

        # Precompute CWT filter bank once for the fixed block length
        self.filter_bank = CWTFilterBank(
            signal_length=self.total_block_len,
            sample_rate=self.sample_rate,
            wavelet_name=self.wavelet_name,
            voices_per_octave=self.voices_per_octave,
            gamma=self.gamma,
            time_bandwidth=self.time_bandwidth,
            frequency_limits=frequency_limits,
            min_frequency_hz=min_frequency_hz,
            max_frequency_hz=max_frequency_hz,
            scales=scales,
            normalization=self.normalization,
            frequency_mode=self.frequency_mode,
        )

        # Persistent streaming state
        self._input_buffer: np.ndarray = np.empty(0, dtype=np.complex128)
        self._prev_context: np.ndarray = np.zeros(self.context_samples, dtype=np.complex128)
        self.absolute_sample_index: int = 0
        self._is_finalized: bool = False

    @property
    def frequencies_hz(self) -> np.ndarray:
        """Returns the pseudo-frequency grid in Hz."""
        return self.filter_bank.frequencies_hz

    @property
    def scales(self) -> np.ndarray:
        """Returns the scale vector."""
        return self.filter_bank.scales

    @property
    def cwt_context_buffer(self) -> np.ndarray:
        return self._input_buffer

    @property
    def buffered_sample_count(self) -> int:
        """Returns the number of un-emitted buffered samples currently held."""
        return len(self._input_buffer)

    def reset(self) -> None:
        """Clears all streaming history and sample offsets."""
        self._input_buffer = np.empty(0, dtype=np.complex128)
        self._prev_context = np.zeros(self.context_samples, dtype=np.complex128)
        self.absolute_sample_index = 0
        self._is_finalized = False

    def process(self, chunk: Any, *, final_chunk: bool = False) -> WaveletResult:
        """
        Processes an incoming chunk of complex IQ samples using block overlap-save CWT.
        Emits valid center coefficients without boundary edge artifacts.
        """
        z_chunk = _as_iq_array(chunk) if chunk is not None and len(chunk) > 0 else np.empty(0, dtype=np.complex128)
        if len(z_chunk) > 0:
            if len(self._input_buffer) > 0:
                self._input_buffer = np.concatenate([self._input_buffer, z_chunk])
            else:
                self._input_buffer = z_chunk

        ctx = self.context_samples
        hop = self.block_size
        tot = self.total_block_len

        emitted_coeffs_list = []
        emitted_energy_list = []
        emitted_times_list = []

        # Process all available full blocks where input_buffer has hop + ctx samples
        while len(self._input_buffer) >= (hop + ctx):
            block = np.concatenate([self._prev_context, self._input_buffer[: hop + ctx]])
            wav_block = self.filter_bank.compute_cwt(block, sample_offset=self.absolute_sample_index)

            valid_coeffs = wav_block.coefficients[:, ctx : ctx + hop]
            valid_energy = wav_block.coefficient_energy[:, ctx : ctx + hop]
            valid_times = (self.absolute_sample_index + np.arange(hop)) / self.sample_rate

            emitted_coeffs_list.append(valid_coeffs)
            emitted_energy_list.append(valid_energy)
            emitted_times_list.append(valid_times)

            self.absolute_sample_index += hop
            self._prev_context = self._input_buffer[hop - ctx : hop].copy()
            self._input_buffer = self._input_buffer[hop:]

        # Handle single-call final flush if final_chunk is True
        if final_chunk:
            self._is_finalized = True
            rem_len = len(self._input_buffer)
            while rem_len > 0:
                cur_valid_len = min(rem_len, hop)
                pre = self._prev_context
                center_and_post = self._input_buffer[: cur_valid_len + ctx]
                needed_pad = tot - len(pre) - len(center_and_post)
                if needed_pad > 0:
                    pad_right = np.zeros(needed_pad, dtype=np.complex128)
                    block = np.concatenate([pre, center_and_post, pad_right])
                else:
                    block = np.concatenate([pre, center_and_post])

                wav_block = self.filter_bank.compute_cwt(block, sample_offset=self.absolute_sample_index)
                valid_coeffs = wav_block.coefficients[:, ctx : ctx + cur_valid_len]
                valid_energy = wav_block.coefficient_energy[:, ctx : ctx + cur_valid_len]
                valid_times = (self.absolute_sample_index + np.arange(cur_valid_len)) / self.sample_rate

                emitted_coeffs_list.append(valid_coeffs)
                emitted_energy_list.append(valid_energy)
                emitted_times_list.append(valid_times)

                self.absolute_sample_index += cur_valid_len
                if cur_valid_len >= ctx:
                    self._prev_context = self._input_buffer[cur_valid_len - ctx : cur_valid_len].copy()
                else:
                    self._prev_context = np.concatenate([self._prev_context[cur_valid_len:], self._input_buffer[:cur_valid_len]])
                self._input_buffer = self._input_buffer[cur_valid_len:]
                rem_len = len(self._input_buffer)

            self._prev_context = np.zeros(self.context_samples, dtype=np.complex128)

        if emitted_coeffs_list:
            all_coeffs = np.concatenate(emitted_coeffs_list, axis=1)
            all_energy = np.concatenate(emitted_energy_list, axis=1)
            all_times = np.concatenate(emitted_times_list)
        else:
            all_coeffs = np.empty((len(self.filter_bank.scales), 0), dtype=np.complex128)
            all_energy = np.empty((len(self.filter_bank.scales), 0), dtype=np.float64)
            all_times = np.empty(0, dtype=np.float64)

        return WaveletResult(
            coefficients=all_coeffs,
            frequencies_hz=self.filter_bank.frequencies_hz,
            scales=self.filter_bank.scales,
            coefficient_energy=all_energy,
            times_sec=all_times,
            sample_rate=self.sample_rate,
            wavelet_name=self.wavelet_name,
            voices_per_octave=self.voices_per_octave,
            gamma=self.gamma,
            time_bandwidth=self.time_bandwidth,
            normalization=self.normalization,
            frequency_mode=self.frequency_mode,
            sample_offset=self.absolute_sample_index - len(all_times),
        )


def main() -> None:
    print("=" * 60)
    print("  Time Frequency Streaming: Demonstration")
    print("=" * 60)
    print("Streaming engine loaded successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
