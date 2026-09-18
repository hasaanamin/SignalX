import threading
from dataclasses import dataclass, field, fields
from typing import Any
import numpy as np

from .models import (
    TimingInput,
    TimeAnalysisResult,
    TOAResult,
    SignalDurationResult,
    TimeOffsetResult,
    InterPulseIntervalResult,
    PRIResult,
    PRFResult,
    JitterResult,
    BurstTimingResult,
    TimingPeriodicityResult,
    PulseDensityResult,
)
from ._common import compute_gap_aware_diff, compute_gap_aware_diff_ns
from .TOA import TimeOfArrival
from .Signal_Duration import SignalDuration
from .Time_Offset import TimeOffset
from .Inter_Pulse_Interval import InterPulseInterval
from .PRI import PulseRepetitionInterval
from .PRF import PulseRepetitionFrequency
from .Jitter import TimingJitter
from .Burst_Timing import BurstTiming
from .Periodicity import TimingPeriodicity
from .Pulse_Density import PulseDensity

_TIMING_INPUT_FIELDS = {f.name for f in fields(TimingInput)}


@dataclass
class _StreamTimingState:
    """Maintains cross-frame continuity and rolling history for a single logical stream."""
    stream_key: Any = None
    last_toa_ns: int | None = None
    last_toa_sec: float | None = None
    timing_reference: str | None = None
    recent_ipi_sec: list[float] = field(default_factory=list)
    recent_ipi_ns: list[int] = field(default_factory=list)
    max_ipi_history_len: int = 512
    frames_processed: int = 0
    total_pulses_observed: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class TimeAnalysisProcessor:
    """
    Central coordinator and orchestrator for Time and Timing Analysis.

    Responsible for:
    1. Inspecting TimingInput to determine which analysis modules can run.
    2. Instantiating and executing applicable analysis modules independently.
    3. Passing intermediate output data between downstream modules (e.g. TOA -> IPI -> PRI -> PRF -> Jitter)
       without introducing inter-module code dependencies.
    4. Managing cross-frame streaming state per logical stream (stream_id, source_id, signal_id, channel_id).
    5. Aggregating all individual typed results into a unified TimeAnalysisResult.
    """

    def __init__(self, streaming_enabled: bool = True) -> None:
        self.streaming_enabled = bool(streaming_enabled)
        self._stream_states: dict[Any, _StreamTimingState] = {}
        self._global_lock = threading.Lock()

        # Instantiate analysis engines independently
        self.toa_module = TimeOfArrival()
        self.duration_module = SignalDuration()
        self.offset_module = TimeOffset()
        self.ipi_module = InterPulseInterval()
        self.pri_module = PulseRepetitionInterval()
        self.prf_module = PulseRepetitionFrequency()
        self.jitter_module = TimingJitter()
        self.burst_module = BurstTiming()
        self.periodicity_module = TimingPeriodicity()
        self.density_module = PulseDensity()

    def reset(self) -> None:
        """Resets all stream states."""
        with self._global_lock:
            self._stream_states.clear()

    def reset_stream(
        self,
        stream_id: Any = None,
        source_id: str | int | None = None,
        source_ip: str | None = None,
        source_port: int | None = None,
        signal_id: int | None = None,
        channel_id: int | None = None,
    ) -> None:
        """
        Resets streaming state for a specific stream_id or (source_id/ip/port, signal_id, channel_id).
        """
        with self._global_lock:
            if stream_id is not None:
                if stream_id in self._stream_states:
                    del self._stream_states[stream_id]

            if (
                source_id is not None
                or source_ip is not None
                or source_port is not None
                or signal_id is not None
                or channel_id is not None
            ):
                keys_to_del = []
                for k in self._stream_states:
                    if isinstance(k, tuple):
                        if len(k) == 3:
                            src_k, sig_k, ch_k = k[0], k[1], k[2]
                            if isinstance(src_k, tuple) and len(src_k) == 3:
                                sid_m = (source_id is None or src_k[0] == source_id)
                                ip_m = (source_ip is None or src_k[1] == source_ip)
                                port_m = (source_port is None or src_k[2] == source_port)
                                src_match = sid_m and ip_m and port_m
                            else:
                                src_match = (source_id is None or src_k == source_id)
                            sig_match = (signal_id is None or sig_k == signal_id)
                            ch_match = (channel_id is None or ch_k == channel_id)
                            if src_match and sig_match and ch_match:
                                keys_to_del.append(k)
                        elif len(k) == 2:
                            sig_match = (signal_id is None or k[0] == signal_id)
                            ch_match = (channel_id is None or k[1] == channel_id)
                            if sig_match and ch_match:
                                keys_to_del.append(k)
                for k in keys_to_del:
                    del self._stream_states[k]

    def get_stream_state(self, stream_id: Any) -> Any:
        """Returns the internal StreamState object for a stream_id, or None if absent."""
        with self._global_lock:
            return self._stream_states.get(stream_id)

    def process(
        self,
        timing_input: TimingInput | dict[str, Any],
        enabled_modules: list[str] | None = None,
        enable_streaming: bool | None = None,
    ) -> TimeAnalysisResult:
        """
        Executes orchestrated time and timing analysis on the given TimingInput.

        Parameters
        ----------
        timing_input : TimingInput | dict[str, Any]
            Upstream timing configuration and extracted feature arrays.
        enabled_modules : list[str] | None
            Optional list of specific module names to execute (e.g. ['toa', 'pri', 'prf']).
            If None, executes all modules whose required inputs are satisfied.
        enable_streaming : bool | None
            Optional override for streaming state maintenance across calls.

        Returns
        -------
        TimeAnalysisResult
            Unified container holding results from all executed modules.
        """
        # Convert dict to TimingInput dataclass if necessary, preserving metadata and all fields
        if isinstance(timing_input, dict):
            inp = TimingInput(**{k: v for k, v in timing_input.items() if k in _TIMING_INPUT_FIELDS})
        else:
            inp = timing_input

        # Stream key precedence: explicit stream_id -> (source_id/transport_id, signal_id, channel_id) -> None (anonymous)
        if inp.stream_id is not None:
            stream_key: Any = inp.stream_id
            is_anonymous = False
        elif (
            inp.source_id is not None
            or inp.source_ip is not None
            or inp.source_port is not None
            or inp.signal_id is not None
            or inp.channel_id is not None
        ):
            if inp.source_ip is not None or inp.source_port is not None:
                src_ident = (inp.source_id, inp.source_ip, inp.source_port)
            else:
                src_ident = inp.source_id
            stream_key = (src_ident, inp.signal_id, inp.channel_id)
            is_anonymous = False
        else:
            stream_key = None
            is_anonymous = True

        use_streaming = self.streaming_enabled if enable_streaming is None else bool(enable_streaming)
        stream_state: _StreamTimingState | None = None

        if use_streaming:
            if is_anonymous:
                # Anonymous streams do NOT retain cross-call persistent state
                stream_state = _StreamTimingState(stream_key=None)
            else:
                with self._global_lock:
                    if stream_key not in self._stream_states:
                        self._stream_states[stream_key] = _StreamTimingState(stream_key=stream_key)
                    stream_state = self._stream_states[stream_key]

        lock_context = stream_state.lock if stream_state is not None else threading.Lock()

        with lock_context:
            is_streaming_update = bool(stream_state and stream_state.frames_processed > 0)

            # Module execution filters
            def should_run(name: str) -> bool:
                return (enabled_modules is None) or (name.lower() in [m.lower() for m in enabled_modules])

            executed: list[str] = []
            summary: dict[str, Any] = {
                "has_hardware_timestamp": inp.hardware_timestamp_ns is not None,
                "sample_rate_hz": inp.sample_rate_hz,
                "source_id": inp.source_id,
                "channel_id": inp.channel_id,
                "signal_id": inp.signal_id,
                "stream_id": stream_key,
                "is_streaming": is_streaming_update,
            }
            if isinstance(inp.metadata, dict):
                summary.update(inp.metadata)

            # 1. TOA (Time of Arrival)
            res_toa: TOAResult | None = None
            has_indices = inp.pulse_start_indices is not None and len(inp.pulse_start_indices) > 0
            has_toas = inp.pulse_toas is not None and len(inp.pulse_toas) > 0
            has_toas_ns = (inp.pulse_toas_ns is not None and len(inp.pulse_toas_ns) > 0) or (
                inp.pulse_toas_absolute_ns is not None and len(inp.pulse_toas_absolute_ns) > 0
            )
            has_toas_rel = inp.pulse_toas_relative_sec is not None and len(inp.pulse_toas_relative_sec) > 0

            has_current_toa_data = bool(has_indices or has_toas or has_toas_ns or has_toas_rel)

            if should_run("toa") and has_current_toa_data:
                res_toa = self.toa_module.analyze(
                    pulse_start_indices=inp.pulse_start_indices,
                    sample_rate_hz=inp.sample_rate_hz,
                    hardware_timestamp_ns=inp.hardware_timestamp_ns,
                    pulse_toas=inp.pulse_toas,
                    pulse_toas_ns=inp.pulse_toas_ns,
                    pulse_toas_absolute_ns=inp.pulse_toas_absolute_ns,
                    pulse_toas_relative_sec=inp.pulse_toas_relative_sec,
                    toa_reference=inp.toa_reference,
                    processing_delay_sec=inp.processing_delay_sec,
                    already_delay_corrected=inp.toa_already_delay_corrected,
                )
                executed.append("TOA")

            # Extract active TOAs and timing reference
            active_toas: np.ndarray | None = None
            active_toas_ns: list[int] | None = None

            if res_toa is not None:
                active_toas = res_toa.toas_sec
                if res_toa.toas_ns is not None:
                    active_toas_ns = [int(x) for x in res_toa.toas_ns]
                current_timing_ref = res_toa.timing_reference
            elif has_toas_ns:
                p_ns = inp.pulse_toas_absolute_ns if inp.pulse_toas_absolute_ns is not None else inp.pulse_toas_ns
                if p_ns is not None:
                    active_toas_ns = [int(x) for x in p_ns]
                    active_toas = np.array([float(x) / 1e9 for x in active_toas_ns], dtype=np.float64)
                current_timing_ref = inp.toa_reference or "externally_absolute"
            elif has_toas:
                active_toas = np.asarray(inp.pulse_toas, dtype=np.float64).reshape(-1)
                current_timing_ref = inp.toa_reference or "capture_relative"
            elif has_toas_rel:
                active_toas = np.asarray(inp.pulse_toas_relative_sec, dtype=np.float64).reshape(-1)
                current_timing_ref = inp.toa_reference or "capture_relative"
            else:
                # No TOA data in current call: preserve explicit reference or stream's existing timing reference
                if inp.toa_reference is not None:
                    current_timing_ref = inp.toa_reference
                elif stream_state is not None and stream_state.timing_reference is not None:
                    current_timing_ref = stream_state.timing_reference
                elif inp.matched_event_times_ns is not None:
                    current_timing_ref = "externally_absolute"
                else:
                    current_timing_ref = "capture_relative"

            # 2. Signal Duration
            res_duration: SignalDurationResult | None = None
            has_durs = inp.pulse_durations is not None and len(inp.pulse_durations) > 0
            has_ends = inp.pulse_end_indices is not None and len(inp.pulse_end_indices) > 0

            if should_run("signal_duration") and (has_durs or (has_indices and has_ends and inp.sample_rate_hz is not None)):
                res_duration = self.duration_module.analyze(
                    pulse_start_indices=inp.pulse_start_indices,
                    pulse_end_indices=inp.pulse_end_indices,
                    sample_rate_hz=inp.sample_rate_hz,
                    pulse_durations=inp.pulse_durations,
                )
                executed.append("Signal_Duration")

            # 3. Time Offset
            res_offset: TimeOffsetResult | None = None
            if should_run("time_offset"):
                offset_ref = inp.toa_reference or current_timing_ref
                if inp.matched_event_times_ns is not None and len(inp.matched_event_times_ns) > 0:
                    res_offset = self.offset_module.analyze(
                        matched_pairs_ns=inp.matched_event_times_ns,
                        timing_reference=offset_ref if offset_ref != "capture_relative" else "externally_absolute",
                    )
                    executed.append("Time_Offset")
                elif inp.matched_event_times is not None:
                    res_offset = self.offset_module.analyze(
                        matched_pairs=inp.matched_event_times,
                        timing_reference=offset_ref,
                    )
                    executed.append("Time_Offset")

            # 4. Inter-Pulse Interval (IPI) with Cross-Frame Streaming Continuity & Reference Tracking
            res_ipi: InterPulseIntervalResult | None = None
            prev_toa_ns: int | None = None
            prev_toa_sec: float | None = None

            if stream_state is not None and has_current_toa_data:
                # Check for timing reference incompatibility across frames
                if stream_state.timing_reference is not None and stream_state.timing_reference != current_timing_ref:
                    # Discontinuity due to reference change: clear previous TOA and rolling history
                    stream_state.last_toa_ns = None
                    stream_state.last_toa_sec = None
                    stream_state.recent_ipi_sec.clear()
                    stream_state.recent_ipi_ns.clear()

                stream_state.timing_reference = current_timing_ref

                # Absolute TOA continuity: valid if current reference is absolute and active_toas_ns exists
                if current_timing_ref in {"hardware_absolute", "externally_absolute", "GPS-derived", "PTP-derived"}:
                    if active_toas_ns is not None and len(active_toas_ns) > 0:
                        prev_toa_ns = stream_state.last_toa_ns
                        # Out-of-order / stream restart handling: break continuity if current starts before previous
                        if prev_toa_ns is not None and active_toas_ns[0] <= prev_toa_ns:
                            prev_toa_ns = None
                            stream_state.recent_ipi_sec.clear()
                            stream_state.recent_ipi_ns.clear()
                else:
                    # Capture-relative continuity: ONLY bridge if caller explicitly declares continuous origin
                    is_rel_continuous = (
                        inp.relative_time_continuous is True
                        or inp.metadata.get("relative_time_continuous") is True
                        or inp.metadata.get("continuous_sample_index") is not None
                        or inp.metadata.get("stream_sample_offset") is not None
                        or inp.provenance.get("relative_time_continuous") is True
                    )
                    if is_rel_continuous and active_toas is not None and len(active_toas) > 0:
                        prev_toa_sec = stream_state.last_toa_sec
                        if prev_toa_sec is not None and active_toas[0] <= prev_toa_sec:
                            prev_toa_sec = None
                            stream_state.recent_ipi_sec.clear()
                            stream_state.recent_ipi_ns.clear()
                    else:
                        prev_toa_sec = None

            if should_run("inter_pulse_interval"):
                if active_toas_ns is not None and len(active_toas_ns) > 0:
                    if len(active_toas_ns) > 1 or prev_toa_ns is not None:
                        res_ipi = self.ipi_module.analyze(
                            toas_ns=active_toas_ns,
                            previous_final_toa_ns=prev_toa_ns,
                            timing_reference=current_timing_ref,
                        )
                        executed.append("Inter_Pulse_Interval")
                elif active_toas is not None and len(active_toas) > 0:
                    if len(active_toas) > 1 or prev_toa_sec is not None:
                        res_ipi = self.ipi_module.analyze(
                            toas=active_toas,
                            previous_final_toa_sec=prev_toa_sec,
                            timing_reference=current_timing_ref,
                        )
                        executed.append("Inter_Pulse_Interval")

                # Update rolling IPI history in stream state for Periodicity accumulation (PRESERVING gaps/NaNs)
                if stream_state is not None and res_ipi is not None and len(res_ipi.intervals_sec) > 0:
                    for v in res_ipi.intervals_sec:
                        stream_state.recent_ipi_sec.append(float(v))
                    if len(stream_state.recent_ipi_sec) > stream_state.max_ipi_history_len:
                        stream_state.recent_ipi_sec = stream_state.recent_ipi_sec[-stream_state.max_ipi_history_len:]

                    if res_ipi.intervals_ns is not None:
                        stream_state.recent_ipi_ns.extend(res_ipi.intervals_ns)
                        if len(stream_state.recent_ipi_ns) > stream_state.max_ipi_history_len:
                            stream_state.recent_ipi_ns = stream_state.recent_ipi_ns[-stream_state.max_ipi_history_len:]

            # 5. PRI (Pulse Repetition Interval)
            res_pri: PRIResult | None = None
            if should_run("pri"):
                if res_ipi is not None and len(res_ipi.intervals_sec) > 0:
                    res_pri = self.pri_module.analyze(intervals=res_ipi.intervals_sec)
                    executed.append("PRI")
                elif active_toas_ns is not None and len(active_toas_ns) > 1:
                    res_pri = self.pri_module.analyze(toas_ns=active_toas_ns)
                    executed.append("PRI")
                elif active_toas is not None and len(active_toas) > 1:
                    res_pri = self.pri_module.analyze(toas=active_toas)
                    executed.append("PRI")

            # 6. PRF (Pulse Repetition Frequency)
            res_prf: PRFResult | None = None
            if should_run("prf"):
                if res_pri is not None and len(res_pri.pri_sequence_sec) > 0:
                    res_prf = self.prf_module.analyze(pri_sequence=res_pri.pri_sequence_sec)
                    executed.append("PRF")
                elif res_ipi is not None and len(res_ipi.intervals_sec) > 0:
                    res_prf = self.prf_module.analyze(pri_sequence=res_ipi.intervals_sec)
                    executed.append("PRF")
                elif active_toas_ns is not None and len(active_toas_ns) > 1:
                    pri_seq = compute_gap_aware_diff_ns(active_toas_ns)
                    if len(pri_seq) > 0:
                        res_prf = self.prf_module.analyze(pri_sequence=pri_seq)
                        executed.append("PRF")
                elif active_toas is not None and len(active_toas) > 1:
                    pri_seq = compute_gap_aware_diff(active_toas)
                    if len(pri_seq) > 0:
                        res_prf = self.prf_module.analyze(pri_sequence=pri_seq)
                        executed.append("PRF")

            # 7. Jitter
            res_jitter: JitterResult | None = None
            if should_run("jitter"):
                active_intervals: np.ndarray | None = None
                nom_pri: float | None = None

                if res_pri is not None:
                    active_intervals = res_pri.pri_sequence_sec
                    nom_pri = res_pri.nominal_pri_sec
                elif res_ipi is not None:
                    active_intervals = res_ipi.intervals_sec
                    nom_pri = res_ipi.median_interval_sec
                elif active_toas_ns is not None and len(active_toas_ns) > 1:
                    active_intervals = compute_gap_aware_diff_ns(active_toas_ns)
                    nom_pri = float(np.median(active_intervals)) if len(active_intervals) > 0 else None
                elif active_toas is not None and len(active_toas) > 1:
                    active_intervals = compute_gap_aware_diff(active_toas)
                    nom_pri = float(np.median(active_intervals)) if len(active_intervals) > 0 else None

                if active_intervals is not None and len(active_intervals) > 0:
                    res_jitter = self.jitter_module.analyze(
                        intervals=active_intervals,
                        nominal_pri_sec=nom_pri,
                    )
                    executed.append("Jitter")

            # 8. Burst Timing
            res_burst: BurstTimingResult | None = None
            if should_run("burst_timing") and (active_toas is not None or active_toas_ns is not None) and inp.burst_labels is not None:
                pulse_durs = res_duration.durations_sec if res_duration is not None else inp.pulse_durations
                res_burst = self.burst_module.analyze(
                    toas=active_toas,
                    burst_labels=inp.burst_labels,
                    pulse_durations=pulse_durs,
                    toas_ns=active_toas_ns,
                    timing_reference=current_timing_ref,
                )
                executed.append("Burst_Timing")

            # 9. Periodicity (supports batch sequences or accumulated rolling IPI history)
            res_periodicity: TimingPeriodicityResult | None = None
            if should_run("periodicity"):
                seq_to_test: np.ndarray | None = None
                if res_pri is not None and len(res_pri.pri_sequence_sec) >= 6:
                    seq_to_test = res_pri.pri_sequence_sec
                elif res_burst is not None and len(res_burst.burst_repetition_intervals_sec) >= 6:
                    seq_to_test = res_burst.burst_repetition_intervals_sec
                elif res_ipi is not None and len(res_ipi.intervals_sec) >= 6:
                    seq_to_test = res_ipi.intervals_sec
                elif stream_state is not None and len(stream_state.recent_ipi_sec) >= 6:
                    # Use bounded rolling IPI history for streaming periodicity
                    seq_to_test = np.array(stream_state.recent_ipi_sec, dtype=np.float64)

                if seq_to_test is not None:
                    res_periodicity = self.periodicity_module.analyze(timing_sequence=seq_to_test)
                    executed.append("Periodicity")

            # 10. Pulse Density
            res_density: PulseDensityResult | None = None
            if should_run("pulse_density"):
                has_obs_window = (inp.observation_duration_sec is not None) or (
                    inp.observation_start is not None and inp.observation_end is not None
                )
                if has_obs_window:
                    p_cnt = len(active_toas) if active_toas is not None else (len(inp.pulse_start_indices) if inp.pulse_start_indices is not None else None)
                    if active_toas is not None or p_cnt is not None:
                        p_durs = res_duration.durations_sec if res_duration is not None else inp.pulse_durations
                        res_density = self.density_module.analyze(
                            toas=active_toas,
                            pulse_count=p_cnt,
                            observation_duration_sec=inp.observation_duration_sec,
                            observation_start=inp.observation_start,
                            observation_end=inp.observation_end,
                            pulse_durations=p_durs,
                        )
                        executed.append("Pulse_Density")

            # Update stream continuity state ONLY if TOA data was present
            if stream_state is not None and has_current_toa_data:
                if active_toas_ns is not None and len(active_toas_ns) > 0:
                    stream_state.last_toa_ns = active_toas_ns[-1]
                if active_toas is not None and len(active_toas) > 0:
                    stream_state.last_toa_sec = float(active_toas[-1])
                stream_state.frames_processed += 1
                stream_state.total_pulses_observed += len(active_toas) if active_toas is not None else 0

            summary["total_pulses_analyzed"] = len(active_toas) if active_toas is not None else 0
            summary["modules_executed_count"] = len(executed)

            return TimeAnalysisResult(
                timing_input_summary=summary,
                executed_modules=executed,
                stream_id=stream_key,
                is_streaming_update=is_streaming_update,
                toa=res_toa,
                signal_duration=res_duration,
                time_offset=res_offset,
                inter_pulse_interval=res_ipi,
                pri=res_pri,
                prf=res_prf,
                jitter=res_jitter,
                burst_timing=res_burst,
                periodicity=res_periodicity,
                pulse_density=res_density,
                metadata=inp.metadata.copy(),
            )


__all__ = ["TimeAnalysisProcessor"]


def main() -> None:
    print("=" * 60)
    print("  TimeAnalysisProcessor Demonstration")
    print("=" * 60)
    fs = 1e6
    pulse_indices = np.arange(0, 10000, 1000)
    inp = TimingInput(
        sample_rate_hz=fs,
        pulse_start_indices=pulse_indices,
        hardware_timestamp_ns=1718000000000000000,
        observation_duration_sec=0.01,
    )
    processor = TimeAnalysisProcessor()
    res = processor.process(inp)
    print(f"Executed modules: {res.executed_modules}")
    if res.toa:
        print(f"TOA Reference: {res.toa.timing_reference}")
    if res.pri:
        print(f"Nominal PRI: {res.pri.nominal_pri_sec * 1e6:.1f} us")
    if res.prf:
        print(f"Nominal PRF: {res.prf.nominal_prf_hz:.1f} Hz")
    print("=" * 60)


if __name__ == "__main__":
    main()
