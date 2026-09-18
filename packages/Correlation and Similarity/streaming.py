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

from typing import Any
from collections import deque
import numpy as np

from .base import (
    CorrelationResult,
    TemplateMatchResult,
    InterPulseCorrelationResult,
    SequenceSimilarityResult,
)
from ._common import (
    _as_iq_array,
    _compute_xcorr_core,
    _compute_findsignal_core,
    _compute_dtw_core,
)


class StreamingCorrelation:
    """
    Real-Time Stateful Streaming Correlation Engine.
    Computes sliding window autocorrelation or cross-correlation across arbitrary chunk boundaries
    matching continuous batch processing.

    Architecture:
      retained overlap buffer + new chunk -> working buffer
      While len(working_buffer) >= W (analysis_window_samples):
        R_xx^(m)[h] = xcorr(x_m, maxlag=L)
        advance working_buffer by H (hop_samples)
      Retain remainder for next chunk.

    Guarantees:
      Streaming arbitrary chunk sizes produces mathematically equivalent windowed correlations
      and timestamps compared to full batch processing.
    """

    def __init__(
        self,
        analysis_window_samples: int = 256,
        hop_samples: int = 64,
        max_lag_samples: int | None = 64,
        sample_rate: float = 1.0,
        scaleopt: str = "coeff",
        method: str = "auto",
    ) -> None:
        self.sample_rate = float(sample_rate)
        if self.sample_rate <= 0 or not np.isfinite(self.sample_rate):
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")

        self.window_samples = int(analysis_window_samples)
        if self.window_samples <= 0:
            raise ValueError(f"analysis_window_samples must be > 0, got {analysis_window_samples}")

        self.hop_samples = int(hop_samples)
        if self.hop_samples <= 0:
            raise ValueError(f"hop_samples must be > 0, got {hop_samples}")

        self.max_lag = int(max_lag_samples) if max_lag_samples is not None else (self.window_samples - 1)
        self.scaleopt = scaleopt
        self.method = method

        # Persistent streaming state
        self.overlap_buffer_x: np.ndarray = np.empty(0, dtype=np.complex128)
        self.overlap_buffer_y: np.ndarray = np.empty(0, dtype=np.complex128)
        self.next_window_start: int = 0
        self.total_windows_emitted: int = 0
        self.correlation_history: list[CorrelationResult] = []

    def reset(self) -> None:
        """Resets all internal streaming buffers and state."""
        self.overlap_buffer_x = np.empty(0, dtype=np.complex128)
        self.overlap_buffer_y = np.empty(0, dtype=np.complex128)
        self.next_window_start = 0
        self.total_windows_emitted = 0
        self.correlation_history.clear()

    def process(
        self,
        chunk_x: Any,
        chunk_y: Any = None,
        *,
        final_chunk: bool = False,
    ) -> list[CorrelationResult]:
        """
        Processes incoming chunk(s) in real time and returns newly completed CorrelationResults.

        Parameters:
            chunk_x: New IQ chunk for signal x.
            chunk_y: Optional new IQ chunk for signal y (if None, computes autocorrelation).
            final_chunk: If True, indicates end of stream.

        Returns:
            List of CorrelationResult objects for windows completed in this chunk.
        """
        zx = _as_iq_array(chunk_x) if chunk_x is not None and len(chunk_x) > 0 else np.empty(0, dtype=np.complex128)
        is_cross = (chunk_y is not None)
        zy = _as_iq_array(chunk_y) if is_cross and len(chunk_y) > 0 else np.empty(0, dtype=np.complex128)

        # Merge with unconsumed overlap buffer
        if len(self.overlap_buffer_x) > 0:
            work_x = np.concatenate([self.overlap_buffer_x, zx])
        else:
            work_x = zx

        if is_cross:
            if len(self.overlap_buffer_y) > 0:
                work_y = np.concatenate([self.overlap_buffer_y, zy])
            else:
                work_y = zy
        else:
            work_y = work_x

        w_len = self.window_samples
        h_len = self.hop_samples
        min_work = min(len(work_x), len(work_y))

        results: list[CorrelationResult] = []

        if min_work < w_len:
            self.overlap_buffer_x = work_x.copy()
            if is_cross:
                self.overlap_buffer_y = work_y.copy()
            return results

        # Process complete windows
        start_local = 0
        while (start_local + w_len) <= min_work:
            win_x = work_x[start_local : start_local + w_len]
            win_y = work_y[start_local : start_local + w_len] if is_cross else None

            corr_res = _compute_xcorr_core(
                win_x,
                win_y,
                maxlag=self.max_lag,
                scaleopt=self.scaleopt,
                method=self.method,
                sample_rate=self.sample_rate,
            )

            # Adjust timestamps to reflect absolute stream sample time
            abs_start = self.next_window_start + start_local
            corr_res.extra["window_index"] = self.total_windows_emitted
            corr_res.extra["absolute_start_sample"] = abs_start
            corr_res.extra["absolute_start_sec"] = float(abs_start / self.sample_rate)

            results.append(corr_res)
            self.correlation_history.append(corr_res)
            self.total_windows_emitted += 1
            start_local += h_len

        # Retain remainder for next chunk
        self.overlap_buffer_x = work_x[start_local:].copy()
        if is_cross:
            self.overlap_buffer_y = work_y[start_local:].copy()
        self.next_window_start += start_local

        return results


class StreamingTemplateMatcher:
    """
    Real-Time Stateful Streaming Template Matcher matching MATLAB `findsignal`.
    Retains (M - 1) context samples across chunk boundaries to ensure templates
    crossing chunk boundaries are detected without dropping or duplicates.

    Architecture:
      retained context (M - 1 samples) + new chunk -> working buffer
      scan newly available M-sample windows -> compute distance
      emit non-overlapping best matches with exact global sample indices.
    """

    def __init__(
        self,
        template: Any,
        sample_rate: float = 1.0,
        metric: str = "squared",
        normalization: str = "none",
        distance_threshold: float | None = None,
    ) -> None:
        self.template = _as_iq_array(template)
        self.m_len = len(self.template)
        if self.m_len == 0:
            raise ValueError("Template cannot be empty.")

        self.sample_rate = float(sample_rate)
        self.metric = metric
        self.normalization = normalization
        self.distance_threshold = float(distance_threshold) if distance_threshold is not None else None

        # Persistent streaming state
        self.context_buffer: np.ndarray = np.empty(0, dtype=np.complex128)
        self.absolute_sample_index: int = 0
        self.last_reported_match_stop: int = -1

    def reset(self) -> None:
        """Resets streaming context and index tracking."""
        self.context_buffer = np.empty(0, dtype=np.complex128)
        self.absolute_sample_index = 0
        self.last_reported_match_stop = -1

    def process(self, chunk: Any, *, final_chunk: bool = False) -> list[TemplateMatchResult]:
        """
        Processes an incoming chunk in real time and returns newly detected template matches.
        """
        z_chunk = _as_iq_array(chunk) if chunk is not None and len(chunk) > 0 else np.empty(0, dtype=np.complex128)
        if len(self.context_buffer) > 0:
            work_buf = np.concatenate([self.context_buffer, z_chunk])
            work_start_abs = self.absolute_sample_index - len(self.context_buffer)
        else:
            work_buf = z_chunk
            work_start_abs = self.absolute_sample_index

        n_work = len(work_buf)
        m = self.m_len

        if n_work < m:
            self.context_buffer = work_buf.copy()
            self.absolute_sample_index += len(z_chunk)
            return []

        # Run template matching on complete working buffer
        res = _compute_findsignal_core(
            work_buf,
            self.template,
            metric=self.metric,
            normalization=self.normalization,
            sample_rate=self.sample_rate,
        )

        matches: list[TemplateMatchResult] = []

        # Check if best match is valid and newly available
        abs_best_start = work_start_abs + res.best_start_sample
        abs_best_stop = work_start_abs + res.best_stop_sample

        is_new = (abs_best_start > self.last_reported_match_stop)
        threshold_met = (self.distance_threshold is None or res.minimum_distance <= self.distance_threshold)

        if is_new and threshold_met:
            match_res = TemplateMatchResult(
                best_start_sample=abs_best_start,
                best_stop_sample=abs_best_stop,
                best_start_sec=float(abs_best_start / self.sample_rate),
                best_stop_sec=float(abs_best_stop / self.sample_rate),
                minimum_distance=res.minimum_distance,
                distance_profile=res.distance_profile,
                matched_segment=res.matched_segment,
                template_method="matlab_findsignal",
                metric=self.metric,
                normalization=self.normalization,
                sample_rate=self.sample_rate,
            )
            matches.append(match_res)
            self.last_reported_match_stop = abs_best_stop

        # Update persistent context: retain last M - 1 samples
        if m > 1:
            self.context_buffer = work_buf[-(m - 1) :].copy()
        else:
            self.context_buffer = np.empty(0, dtype=np.complex128)

        self.absolute_sample_index += len(z_chunk)
        return matches


class StreamingInterPulseCorrelation:
    """
    Real-Time Event-Driven Inter-Pulse Correlation Engine.
    Maintains a rolling history of the last K completed pulses from the Pulse Detector
    and calculates rolling pairwise cross-correlation scores.
    """

    def __init__(
        self,
        history_len: int = 16,
        sample_rate: float = 1.0,
        maxlag: int | None = None,
        method: str = "auto",
    ) -> None:
        self.history_len = int(history_len)
        self.sample_rate = float(sample_rate)
        self.maxlag = maxlag
        self.method = method
        self.pulse_history: deque[Any] = deque(maxlen=self.history_len)
        self.total_pulses_received: int = 0

    def reset(self) -> None:
        """Clears rolling pulse history."""
        self.pulse_history.clear()
        self.total_pulses_received = 0

    def process_pulse(self, pulse: Any) -> dict[str, Any]:
        """
        Receives a newly completed PulseCandidate and computes pairwise correlation
        against previously observed pulses in history.

        Returns:
            Dictionary containing pairwise similarities against previous pulses
            and updated rolling correlation matrix.
        """
        new_iq = _as_iq_array(pulse)
        pulse_id = getattr(pulse, "pulse_id", self.total_pulses_received)
        self.total_pulses_received += 1

        pairwise_scores: list[dict[str, Any]] = []

        for idx, prev_pulse in enumerate(reversed(self.pulse_history)):
            prev_iq = _as_iq_array(prev_pulse)
            prev_id = getattr(prev_pulse, "pulse_id", f"prev_{idx+1}")

            corr_res = _compute_xcorr_core(
                new_iq,
                prev_iq,
                maxlag=self.maxlag,
                scaleopt="coeff",
                method=self.method,
                sample_rate=self.sample_rate,
            )

            pairwise_scores.append({
                "compared_pulse_id": prev_id,
                "history_offset": idx + 1,
                "similarity": corr_res.peak_magnitude,
                "peak_lag_samples": corr_res.peak_lag_samples,
                "peak_lag_sec": corr_res.peak_lag_sec,
                "phase_offset_rad": corr_res.phase_at_peak_rad,
            })

        self.pulse_history.append(pulse)

        # Build rolling pairwise matrix C for current history
        k = len(self.pulse_history)
        c_matrix = np.eye(k, dtype=np.float64)
        history_list = list(self.pulse_history)

        for i in range(k):
            for j in range(i + 1, k):
                p_i = _as_iq_array(history_list[i])
                p_j = _as_iq_array(history_list[j])
                res_ij = _compute_xcorr_core(
                    p_i,
                    p_j,
                    maxlag=self.maxlag,
                    scaleopt="coeff",
                    method=self.method,
                    sample_rate=self.sample_rate,
                )
                c_matrix[i, j] = res_ij.peak_magnitude
                c_matrix[j, i] = res_ij.peak_magnitude

        return {
            "new_pulse_id": pulse_id,
            "total_pulses_received": self.total_pulses_received,
            "pairwise_scores": pairwise_scores,
            "rolling_correlation_matrix": c_matrix,
            "history_size": k,
        }


class StreamingSequenceSimilarity:
    """
    Real-Time Event-Driven Pulse Sequence Similarity & DTW Engine.
    Buffers incoming pulse feature vectors into a rolling sequence of length K
    and performs DTW against a stored reference template sequence.
    """

    def __init__(
        self,
        reference_sequence: Any,
        sequence_length: int = 16,
        sequence_hop: int = 1,
        metric: str = "euclidean",
        normalize_features: bool = True,
    ) -> None:
        self.ref_seq = np.asarray(reference_sequence, dtype=np.float64)
        if self.ref_seq.ndim == 1:
            self.ref_seq = self.ref_seq[:, None]

        self.sequence_length = int(sequence_length)
        self.sequence_hop = int(sequence_hop)
        self.metric = metric
        self.normalize_features = normalize_features

        self.feature_buffer: deque[np.ndarray] = deque(maxlen=self.sequence_length * 2)
        self.pulses_since_last_eval: int = 0
        self.total_pulses_received: int = 0

    def reset(self) -> None:
        """Clears feature buffer and streaming state."""
        self.feature_buffer.clear()
        self.pulses_since_last_eval = 0
        self.total_pulses_received = 0

    def process_pulse_features(self, feature_vector: Any) -> SequenceSimilarityResult | None:
        """
        Appends a newly extracted pulse feature vector and evaluates DTW
        whenever a complete sequence of length K is ready.
        """
        f_arr = np.asarray(feature_vector, dtype=np.float64).ravel()
        self.feature_buffer.append(f_arr)
        self.total_pulses_received += 1
        self.pulses_since_last_eval += 1

        if len(self.feature_buffer) >= self.sequence_length and self.pulses_since_last_eval >= self.sequence_hop:
            # Extract newest K pulse features
            recent_features = np.array(list(self.feature_buffer)[-self.sequence_length :])
            res = _compute_dtw_core(
                recent_features,
                self.ref_seq,
                metric=self.metric,
                normalize_features=self.normalize_features,
            )
            self.pulses_since_last_eval = 0
            return res

        return None


def main() -> None:
    print("=" * 60)
    print("  Correlation and Similarity Streaming: Demonstration")
    print("=" * 60)
    print("Streaming engine loaded successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
