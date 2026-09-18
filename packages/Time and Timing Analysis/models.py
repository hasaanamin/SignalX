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

from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np

# Unsigned 64-bit integer boundaries for hardware timestamps
UINT64_MIN: int = 0
# Unsigned 64-bit integer boundaries for hardware timestamps
UINT64_MIN: int = 0
UINT64_MAX: int = 0xFFFFFFFFFFFFFFFF  # 18446744073709551615


@dataclass
class TimingInput:
    """
    Standard upstream timing input container.

    Fields are optional depending on the analysis requested.
    The package does NOT parse UDP datagrams; upstream Acquisition and Pulse Detection
    populate these structured fields.
    """
    sample_rate_hz: float | None = None
    hardware_timestamp_ns: int | None = None
    pulse_start_indices: np.ndarray | Sequence[int] | None = None
    pulse_end_indices: np.ndarray | Sequence[int] | None = None
    pulse_toas: np.ndarray | Sequence[float] | None = None
    pulse_toas_ns: np.ndarray | Sequence[int] | None = None
    pulse_toas_absolute_ns: np.ndarray | Sequence[int] | None = None
    pulse_toas_relative_sec: np.ndarray | Sequence[float] | None = None
    toa_reference: str | None = None  # "hardware_absolute" | "capture_relative" | "externally_absolute" | ...
    pulse_durations: np.ndarray | Sequence[float] | None = None
    burst_labels: np.ndarray | Sequence[int | str] | None = None
    matched_event_times: Sequence[tuple[float, float]] | tuple[np.ndarray, np.ndarray] | None = None
    matched_event_times_ns: Sequence[tuple[int, int] | list[int]] | list[Any] | None = None
    relative_time_continuous: bool | None = None
    frame_start_absolute_sample_index: int | None = None
    toa_already_delay_corrected: bool = False
    observation_start: float | None = None
    observation_end: float | None = None
    observation_duration_sec: float | None = None
    source_id: str | int | tuple[Any, ...] | None = None
    source_ip: str | None = None
    source_port: int | None = None
    channel_id: int | None = None
    signal_id: int | None = None
    stream_id: tuple[Any, ...] | str | Any | None = None
    processing_delay_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validates configuration and rejects ambiguous input combinations."""
        # 1. Hardware timestamp boundaries & type checking
        if self.hardware_timestamp_ns is not None:
            if isinstance(self.hardware_timestamp_ns, (bool, np.bool_)):
                raise TypeError(f"hardware_timestamp_ns cannot be a boolean (got {self.hardware_timestamp_ns})")
            if isinstance(self.hardware_timestamp_ns, str):
                raise TypeError(f"hardware_timestamp_ns must be an integer, got str '{self.hardware_timestamp_ns}'")
            if isinstance(self.hardware_timestamp_ns, (float, np.floating)):
                raise TypeError(f"hardware_timestamp_ns must be an integer, got float {self.hardware_timestamp_ns}")
            if not isinstance(self.hardware_timestamp_ns, (int, np.integer)):
                raise TypeError(f"hardware_timestamp_ns must be an integer, got {type(self.hardware_timestamp_ns).__name__}")
            hw = int(self.hardware_timestamp_ns)
            if hw < UINT64_MIN or hw > UINT64_MAX:
                raise ValueError(f"hardware_timestamp_ns ({hw}) outside valid uint64 range [0, {UINT64_MAX}]")

        # 2. Check elements of integer ns arrays
        for name, arr in [
            ("pulse_toas_absolute_ns", self.pulse_toas_absolute_ns),
            ("pulse_toas_ns", self.pulse_toas_ns),
        ]:
            if arr is not None:
                if isinstance(arr, (list, tuple)):
                    for x in arr:
                        if isinstance(x, (bool, np.bool_)):
                            raise TypeError(f"Elements of {name} cannot be boolean values (got {x})")
                        if isinstance(x, (float, np.floating)):
                            raise TypeError(f"Elements of {name} must be integral integers, got float {x}")
                        if not isinstance(x, (int, np.integer)):
                            raise TypeError(f"Elements of {name} must be integer-like, got {type(x).__name__}")
                elif isinstance(arr, np.ndarray):
                    if arr.dtype == bool:
                        raise TypeError(f"{name} cannot be boolean numpy array")
                    if np.issubdtype(arr.dtype, np.floating):
                        raise TypeError(f"{name} cannot be floating-point numpy array")
                    for x in arr.reshape(-1):
                        if isinstance(x, (bool, np.bool_)):
                            raise TypeError(f"Elements of {name} cannot be boolean values (got {x})")
                        if isinstance(x, (float, np.floating)):
                            raise TypeError(f"Elements of {name} must be integral integers, got float {x}")
                        if not isinstance(x, (int, np.integer)):
                            raise TypeError(f"Elements of {name} must be integer-like, got {type(x).__name__}")

        # 3. Check matched_event_times_ns elements
        if self.matched_event_times_ns is not None:
            for pair in self.matched_event_times_ns:
                if not isinstance(pair, (tuple, list, np.ndarray)):
                    raise ValueError(f"Each element of matched_event_times_ns must be a pair (got {type(pair).__name__})")
                if len(pair) != 2:
                    raise ValueError(f"Each matched pair in matched_event_times_ns must have length 2 (got {len(pair)})")
                for x in pair:
                    if isinstance(x, (bool, np.bool_)):
                        raise TypeError(f"Elements of matched_event_times_ns cannot be boolean values (got {x})")
                    if isinstance(x, (float, np.floating)):
                        raise TypeError(f"Elements of matched_event_times_ns must be integral integers, got float {x}")
                    if not isinstance(x, (int, np.integer)):
                        raise TypeError(f"Elements of matched_event_times_ns must be integer-like, got {type(x).__name__}")

        # 4. Check pulse_start_indices and pulse_end_indices elements
        for name, arr in [
            ("pulse_start_indices", self.pulse_start_indices),
            ("pulse_end_indices", self.pulse_end_indices),
        ]:
            if arr is not None:
                if isinstance(arr, (list, tuple)):
                    for x in arr:
                        if isinstance(x, (bool, np.bool_)):
                            raise TypeError(f"Elements of {name} cannot be boolean values (got {x})")
                        if isinstance(x, (float, np.floating)):
                            raise TypeError(f"Elements of {name} must be integral integers, got float {x}")
                        if not isinstance(x, (int, np.integer)):
                            raise TypeError(f"Elements of {name} must be integer-like, got {type(x).__name__}")
                elif isinstance(arr, np.ndarray):
                    if arr.dtype == bool:
                        raise TypeError(f"{name} cannot be boolean numpy array")
                    if np.issubdtype(arr.dtype, np.floating):
                        raise TypeError(f"{name} cannot be floating-point numpy array")
                    for x in arr.reshape(-1):
                        if isinstance(x, (bool, np.bool_)):
                            raise TypeError(f"Elements of {name} cannot be boolean values (got {x})")
                        if isinstance(x, (float, np.floating)):
                            raise TypeError(f"Elements of {name} must be integral integers, got float {x}")
                        if not isinstance(x, (int, np.integer)):
                            raise TypeError(f"Elements of {name} must be integer-like, got {type(x).__name__}")

        # 5. Check processing_delay_sec
        if self.processing_delay_sec is not None:
            if isinstance(self.processing_delay_sec, (bool, np.bool_)):
                raise TypeError(f"processing_delay_sec cannot be boolean (got {self.processing_delay_sec})")
            p_del = float(self.processing_delay_sec)
            if not np.isfinite(p_del):
                raise ValueError(f"processing_delay_sec must be finite, got {p_del}")
            if p_del < 0.0:
                raise ValueError(f"processing_delay_sec must be non-negative (>= 0), got {p_del}")

        # 6. Reject ambiguous direct TOA combinations
        abs_provided = (self.pulse_toas_absolute_ns is not None) or (self.pulse_toas_ns is not None)
        rel_provided = (self.pulse_toas_relative_sec is not None) or (self.pulse_toas is not None)
        if abs_provided and rel_provided:
            raise ValueError(
                "Ambiguous direct TOA input: both absolute nanoseconds and relative seconds supplied simultaneously."
            )

        # 7. Explicit reference validation & contradiction checks
        if self.toa_reference is not None:
            valid_refs = {
                "hardware_absolute",
                "capture_relative",
                "externally_absolute",
                "coherent_relative",
                "GPS-derived",
                "PTP-derived",
            }
            if self.toa_reference not in valid_refs:
                raise ValueError(f"Invalid toa_reference '{self.toa_reference}'. Must be one of {valid_refs}")

            if (
                self.toa_reference == "hardware_absolute"
                and self.hardware_timestamp_ns is None
                and rel_provided
                and not abs_provided
            ):
                raise ValueError(
                    "Contradictory timing reference: 'hardware_absolute' declared but only relative TOAs supplied and no hardware timestamp exists."
                )

            if self.toa_reference == "capture_relative" and abs_provided and not rel_provided:
                raise ValueError(
                    "Contradictory timing reference: 'capture_relative' declared but absolute nanosecond TOAs supplied."
                )

        # 8. Stream ID default assignment
        if self.stream_id is None:
            if (
                self.source_id is not None
                or self.source_ip is not None
                or self.source_port is not None
                or self.signal_id is not None
                or self.channel_id is not None
            ):
                if self.source_ip is not None or self.source_port is not None:
                    src_ident = (self.source_id, self.source_ip, self.source_port)
                else:
                    src_ident = self.source_id
                self.stream_id = (src_ident, self.signal_id, self.channel_id)

    @property
    def provenance(self) -> dict[str, Any]:
        """Convenience property accessing metadata provenance."""
        return self.metadata.get("provenance", {})

    @classmethod
    def from_upstream(
        cls,
        acquisition_record: dict[str, Any] | None = None,
        pulse_detection_result: dict[str, Any] | Any | None = None,
        **kwargs: Any,
    ) -> TimingInput:
        """Constructs TimingInput from canonical upstream records."""
        return build_timing_input(
            acquisition_record=acquisition_record,
            pulse_detection_result=pulse_detection_result,
            **kwargs,
        )


def build_timing_input(
    acquisition_record: dict[str, Any] | None = None,
    pulse_detection_result: dict[str, Any] | Any | None = None,
    **kwargs: Any,
) -> TimingInput:
    """
    Boundary adapter that maps upstream Acquisition records and Pulse Detection
    results into a strictly typed TimingInput container with verified provenance.

    Parameters
    ----------
    acquisition_record : dict[str, Any] | None
        Canonical Acquisition and Processing dictionary.
    pulse_detection_result : dict[str, Any] | Any | None
        Upstream pulse detection results dictionary, candidate list, or summary object.
    **kwargs : Any
        Explicit overrides.

    Returns
    -------
    TimingInput
        Fully mapped TimingInput with provenance metadata.
    """
    if pulse_detection_result is None:
        pulse_detection_result = (
            kwargs.get("pulse_candidates")
            or kwargs.get("candidates")
            or kwargs.get("pulse_detection_results")
        )
    prov: dict[str, Any] = {}
    fs: float | None = kwargs.get("sample_rate_hz")
    hw_ts_ns: Any = kwargs.get("hardware_timestamp_ns")
    p_starts: Any = kwargs.get("pulse_start_indices")
    p_ends: Any = kwargs.get("pulse_end_indices")
    p_durs: Any = kwargs.get("pulse_durations")
    p_toas: Any = kwargs.get("pulse_toas")
    p_toas_abs_ns: Any = kwargs.get("pulse_toas_absolute_ns")
    p_toas_rel_sec: Any = kwargs.get("pulse_toas_relative_sec")
    toa_ref: str | None = kwargs.get("toa_reference")
    burst_lbls: Any = kwargs.get("burst_labels")
    src_id: Any = kwargs.get("source_id")
    src_ip: str | None = kwargs.get("source_ip")
    src_port: int | None = kwargs.get("source_port")
    if src_id is None and (src_ip is not None or src_port is not None):
        src_id = (src_ip, src_port) if src_port is not None else src_ip
    ch_id: int | None = kwargs.get("channel_id")
    sig_id: int | None = kwargs.get("signal_id")
    st_id: Any = kwargs.get("stream_id")
    frame_start_idx: int | None = kwargs.get("frame_start_absolute_sample_index")
    already_delay_corr: bool = bool(kwargs.get("toa_already_delay_corrected", False))

    proc_delay_arg = kwargs.get("processing_delay_sec")
    if proc_delay_arg is not None:
        proc_delay_sec = float(proc_delay_arg)
        explicit_proc_delay = True
    else:
        proc_delay_sec = 0.0
        explicit_proc_delay = False

    obs_dur: float | None = kwargs.get("observation_duration_sec")
    obs_start: float | None = kwargs.get("observation_start")
    obs_end: float | None = kwargs.get("observation_end")
    matched_evs: Any = kwargs.get("matched_event_times")
    matched_evs_ns: Any = kwargs.get("matched_event_times_ns")
    rel_cont: bool | None = kwargs.get("relative_time_continuous")
    meta: dict[str, Any] = kwargs.get("metadata", {}).copy()

    # 1. Map Acquisition record
    if acquisition_record is not None and isinstance(acquisition_record, dict):
        if fs is None:
            fs = acquisition_record.get("sample_rate_hz") or acquisition_record.get("sample_rate")
            if fs is not None:
                prov["sample_rate_source"] = "acquisition_record"

        if frame_start_idx is None:
            for k in ["frame_start_absolute_sample_index", "frame_start_sample_index", "sample_offset", "start_sample_index", "global_sample_index"]:
                if k in acquisition_record and acquisition_record[k] is not None:
                    frame_start_idx = int(acquisition_record[k])
                    prov["frame_start_sample_index"] = frame_start_idx
                    break

        # Hardware timestamp extraction (pass through without blind coercion to preserve validation)
        if hw_ts_ns is None:
            if "hardware_timestamp_ns" in acquisition_record and acquisition_record["hardware_timestamp_ns"] is not None:
                hw_ts_ns = acquisition_record["hardware_timestamp_ns"]
                prov["timestamp_source"] = "acquisition_hardware_timestamp_ns"
                prov["timing_reference"] = "hardware_absolute"
            else:
                raw_ts = acquisition_record.get("timestamp")
                ts_unit = acquisition_record.get("timestamp_unit", "ns")
                rec_meta = acquisition_record.get("metadata", {})
                ts_source = rec_meta.get("timestamp_source") if isinstance(rec_meta, dict) else None

                if raw_ts is not None and ts_source == "hardware":
                    if ts_unit == "ns":
                        hw_ts_ns = raw_ts
                    elif ts_unit == "s" and isinstance(raw_ts, (int, float, np.number)):
                        hw_ts_ns = int(round(float(raw_ts) * 1e9))
                    else:
                        hw_ts_ns = raw_ts
                    prov["timestamp_source"] = "hardware"
                    prov["timing_reference"] = "hardware_absolute"
                elif raw_ts is not None and ts_unit == "ns":
                    hw_ts_ns = raw_ts
                    prov["timestamp_source"] = "acquisition_timestamp_ns"
                    prov["timing_reference"] = "hardware_absolute"
                else:
                    prov["timestamp_source"] = "none"
                    prov["timing_reference"] = "capture_relative"

        # Source / Signal / Channel / Stream IDs
        transport_meta = acquisition_record.get("metadata", {}).get("transport", {}) if isinstance(acquisition_record.get("metadata"), dict) else {}
        if src_id is None:
            src_id = acquisition_record.get("source_id")
            if src_id is None and isinstance(transport_meta, dict):
                if "source_id" in transport_meta and transport_meta["source_id"] is not None:
                    src_id = transport_meta["source_id"]
                elif "source_ip" in transport_meta and transport_meta["source_ip"] is not None:
                    rec_ip = transport_meta["source_ip"]
                    rec_port = transport_meta.get("source_port")
                    src_id = (rec_ip, rec_port) if rec_port is not None else rec_ip
                    src_ip = rec_ip
                    src_port = rec_port
        if sig_id is None:
            sig_id = acquisition_record.get("signal_id")
            if sig_id is None and isinstance(transport_meta, dict):
                sig_id = transport_meta.get("signal_id")
        if ch_id is None:
            ch_id = acquisition_record.get("channel_id")
            if ch_id is None and isinstance(transport_meta, dict):
                ch_id = transport_meta.get("channel_id")
        if st_id is None:
            st_id = acquisition_record.get("stream_id")
            if st_id is None and isinstance(transport_meta, dict):
                st_id = transport_meta.get("stream_id")

        # Check for filter group delay in processing provenance only if not explicitly supplied
        if not explicit_proc_delay:
            rec_meta = acquisition_record.get("metadata", {}) if isinstance(acquisition_record.get("metadata"), dict) else {}
            # Direct group delay fields
            if "group_delay_s" in acquisition_record and acquisition_record["group_delay_s"] is not None:
                proc_delay_sec = float(acquisition_record["group_delay_s"])
                prov["processing_delay_source"] = "acquisition_filter"
            elif "group_delay_sec" in acquisition_record and acquisition_record["group_delay_sec"] is not None:
                proc_delay_sec = float(acquisition_record["group_delay_sec"])
                prov["processing_delay_source"] = "acquisition_filter"
            elif "group_delay_samples" in acquisition_record and acquisition_record["group_delay_samples"] is not None and fs is not None and fs > 0:
                proc_delay_sec = float(acquisition_record["group_delay_samples"]) / float(fs)
                prov["processing_delay_source"] = "acquisition_filter"
            elif isinstance(rec_meta, dict):
                proc_entries = rec_meta.get("processing", {})
                if isinstance(proc_entries, dict):
                    if "group_delay_s" in proc_entries and proc_entries["group_delay_s"] is not None:
                        proc_delay_sec = float(proc_entries["group_delay_s"])
                        prov["processing_delay_source"] = "acquisition_filter"
                    elif "group_delay_sec" in proc_entries and proc_entries["group_delay_sec"] is not None:
                        proc_delay_sec = float(proc_entries["group_delay_sec"])
                        prov["processing_delay_source"] = "acquisition_filter"
                    elif "group_delay_samples" in proc_entries and proc_entries["group_delay_samples"] is not None and fs is not None and fs > 0:
                        proc_delay_sec = float(proc_entries["group_delay_samples"]) / float(fs)
                        prov["processing_delay_source"] = "acquisition_filter"
                elif isinstance(proc_entries, list):
                    for entry in proc_entries:
                        if isinstance(entry, dict) and entry.get("operation") in {"filter", "filtering"}:
                            if "group_delay_s" in entry and entry["group_delay_s"] is not None:
                                proc_delay_sec = float(entry["group_delay_s"])
                                prov["processing_delay_source"] = "acquisition_filter"
                                break
                            elif "group_delay_sec" in entry and entry["group_delay_sec"] is not None:
                                proc_delay_sec = float(entry["group_delay_sec"])
                                prov["processing_delay_source"] = "acquisition_filter"
                                break
                            elif "group_delay_samples" in entry and entry["group_delay_samples"] is not None and fs is not None and fs > 0:
                                proc_delay_sec = float(entry["group_delay_samples"]) / float(fs)
                                prov["processing_delay_source"] = "acquisition_filter"
                                break

        # Pass quality flags
        if "quality_flags" in acquisition_record:
            meta["acquisition_quality_flags"] = set(acquisition_record["quality_flags"])

        # Infer observation duration from record sample count if not provided
        if obs_dur is None and obs_start is None and obs_end is None:
            s_count = acquisition_record.get("sample_count")
            if s_count is not None and fs is not None and fs > 0:
                obs_dur = float(s_count) / float(fs)
                prov["observation_window_source"] = "acquisition_record_length"

    # 2. Map Pulse Detection results (duck-typed support for PulseCandidate objects, summary dicts, candidate lists)
    if pulse_detection_result is not None:
        candidates_list: list[Any] | None = None
        p_dict: dict[str, Any] = {}

        if isinstance(pulse_detection_result, (list, tuple)):
            candidates_list = list(pulse_detection_result)
        elif isinstance(pulse_detection_result, dict):
            p_dict = pulse_detection_result
            if "candidates" in p_dict and isinstance(p_dict["candidates"], (list, tuple)):
                candidates_list = list(p_dict["candidates"])
            elif "candidate_dicts" in p_dict and isinstance(p_dict["candidate_dicts"], (list, tuple)):
                candidates_list = list(p_dict["candidate_dicts"])
        else:
            # Check for candidate summary attributes
            if hasattr(pulse_detection_result, "candidates") and isinstance(pulse_detection_result.candidates, (list, tuple)):
                candidates_list = list(pulse_detection_result.candidates)
            elif hasattr(pulse_detection_result, "candidate_dicts") and isinstance(pulse_detection_result.candidate_dicts, (list, tuple)):
                candidates_list = list(pulse_detection_result.candidate_dicts)
            elif hasattr(pulse_detection_result, "__dict__"):
                p_dict = pulse_detection_result.__dict__

        if candidates_list is not None and len(candidates_list) > 0:
            extracted_starts_abs: list[Any] = []
            extracted_starts_local: list[Any] = []
            extracted_ends: list[Any] = []
            extracted_durs: list[float] = []
            extracted_toas: list[float] = []
            extracted_bursts: list[Any] = []

            has_any_abs = False
            for c in candidates_list:
                def get_cand_val(key: str, fallback_key: str | None = None) -> Any:
                    val = None
                    if isinstance(c, dict):
                        val = c.get(key)
                        if val is None and fallback_key is not None:
                            val = c.get(fallback_key)
                    else:
                        val = getattr(c, key, None)
                        if val is None and fallback_key is not None:
                            val = getattr(c, fallback_key, None)
                    return val

                s_abs = get_cand_val("start_idx_absolute")
                s_loc = get_cand_val("start_idx")
                e_abs = get_cand_val("stop_idx_absolute", "stop_idx")

                if s_abs is not None:
                    has_any_abs = True
                    extracted_starts_abs.append(int(s_abs))
                else:
                    extracted_starts_abs.append(None)

                if s_loc is not None:
                    extracted_starts_local.append(int(s_loc))
                elif s_abs is not None and frame_start_idx is not None:
                    extracted_starts_local.append(int(s_abs) - int(frame_start_idx))
                else:
                    extracted_starts_local.append(None)

                if e_abs is not None:
                    extracted_ends.append(int(e_abs))
                else:
                    extracted_ends.append(None)

                pw_s = get_cand_val("pulse_width_sec")
                if pw_s is not None:
                    extracted_durs.append(float(pw_s))
                else:
                    pw_samp = get_cand_val("pulse_width_samples")
                    if pw_samp is not None and fs is not None and fs > 0:
                        extracted_durs.append(float(pw_samp) / float(fs))
                    else:
                        extracted_durs.append(np.nan)

                toa_val = get_cand_val("toa_sec")
                if toa_val is not None:
                    extracted_toas.append(float(toa_val))
                else:
                    extracted_toas.append(np.nan)

                b_id = get_cand_val("burst_id", "cluster_id")
                if b_id is None:
                    b_id = get_cand_val("burst_label", "cluster_label")
                extracted_bursts.append(b_id)

            # Preserve exact row alignment across all candidate arrays
            if p_starts is None:
                # If frame_start_idx is present and s_abs exists, compute local starts for frame
                if frame_start_idx is not None and has_any_abs:
                    chosen_starts = [
                        (int(s) - int(frame_start_idx)) if s is not None else None
                        for s in extracted_starts_abs
                    ]
                else:
                    chosen_starts = (
                        extracted_starts_abs if has_any_abs else extracted_starts_local
                    )
                if any(x is not None for x in chosen_starts):
                    p_starts = np.array(
                        [np.nan if x is None else x for x in chosen_starts], dtype=np.float64
                    )
                    if not np.any(np.isnan(p_starts)):
                        p_starts = p_starts.astype(np.int64)
                    prov["pulse_boundary_source"] = "pulse_candidates"

            if p_ends is None and any(x is not None for x in extracted_ends):
                p_ends = np.array(
                    [np.nan if x is None else x for x in extracted_ends], dtype=np.float64
                )
                if not np.any(np.isnan(p_ends)):
                    p_ends = p_ends.astype(np.int64)

            if p_durs is None and any(np.isfinite(x) for x in extracted_durs):
                p_durs = np.array(extracted_durs, dtype=np.float64)

            if p_toas is None and any(np.isfinite(x) for x in extracted_toas):
                p_toas = np.array(extracted_toas, dtype=np.float64)

            if burst_lbls is None and any(x is not None for x in extracted_bursts):
                burst_lbls = extracted_bursts

            # If global absolute start indices exist and hardware timestamp is provided,
            # calculate integer-ns absolute TOAs with correct origin
            if (
                p_toas_abs_ns is None
                and has_any_abs
                and hw_ts_ns is not None
                and fs is not None
                and fs > 0
                and not isinstance(hw_ts_ns, (bool, np.bool_))
            ):
                f_start = frame_start_idx if frame_start_idx is not None else 0
                calc_abs_ns: list[int] = []
                for s in extracted_starts_abs:
                    if s is not None:
                        delta_s = (int(s) - int(f_start)) / float(fs)
                        calc_abs_ns.append(int(hw_ts_ns) + int(round(delta_s * 1e9)))
                if len(calc_abs_ns) == len(candidates_list):
                    p_toas_abs_ns = calc_abs_ns
                    prov["toa_origin"] = "frame_hw_ts_plus_local_sample_offset"

            # If global sample index is used without hardware timestamp, mark relative time continuous
            if rel_cont is None and has_any_abs and hw_ts_ns is None:
                rel_cont = True
                prov["relative_time_continuous"] = True

        # Fallback to direct dict mappings if candidates_list was not used or incomplete
        if p_starts is None:
            for k in ["pulse_start_indices", "start_indices", "leading_edges", "leading_edge_indices", "pulse_starts", "start_idx_absolute", "start_idx"]:
                if k in p_dict and p_dict[k] is not None:
                    p_starts = p_dict[k]
                    prov["pulse_boundary_source"] = "pulse_detection_result"
                    break

        if p_ends is None:
            for k in ["pulse_end_indices", "end_indices", "trailing_edges", "trailing_edge_indices", "pulse_ends", "stop_idx_absolute", "stop_idx"]:
                if k in p_dict and p_dict[k] is not None:
                    p_ends = p_dict[k]
                    break

        if p_durs is None:
            for k in ["pulse_durations", "durations", "pulse_widths", "pulse_durations_sec", "pulse_width_sec"]:
                if k in p_dict and p_dict[k] is not None:
                    p_durs = p_dict[k]
                    break

        if p_toas is None and p_toas_abs_ns is None and p_toas_rel_sec is None:
            for k in ["pulse_toas", "toas", "pulse_toas_sec", "toa_sec"]:
                if k in p_dict and p_dict[k] is not None:
                    p_toas = p_dict[k]
                    break
            for k in ["pulse_toas_absolute_ns", "pulse_toas_ns", "toas_ns"]:
                if k in p_dict and p_dict[k] is not None:
                    p_toas_abs_ns = p_dict[k]
                    break

        if burst_lbls is None:
            for k in ["burst_labels", "cluster_labels", "burst_ids", "burst_id"]:
                if k in p_dict and p_dict[k] is not None:
                    burst_lbls = p_dict[k]
                    break

    meta["provenance"] = prov

    final_stream_id = st_id
    if final_stream_id is None and (src_id is not None or sig_id is not None or ch_id is not None):
        final_stream_id = (src_id, sig_id, ch_id)

    final_toas_ns = p_toas_abs_ns if p_toas_abs_ns is not None else kwargs.get("pulse_toas_ns")
    final_toas_rel = p_toas_rel_sec if p_toas_rel_sec is not None else (p_toas if final_toas_ns is None else None)

    return TimingInput(
        sample_rate_hz=fs,
        hardware_timestamp_ns=hw_ts_ns,
        pulse_start_indices=p_starts,
        pulse_end_indices=p_ends,
        pulse_toas=final_toas_rel,
        pulse_toas_ns=final_toas_ns,
        pulse_toas_absolute_ns=p_toas_abs_ns,
        pulse_toas_relative_sec=p_toas_rel_sec if final_toas_ns is None else None,
        toa_reference=toa_ref or prov.get("timing_reference"),
        pulse_durations=p_durs,
        burst_labels=burst_lbls,
        matched_event_times=matched_evs,
        matched_event_times_ns=matched_evs_ns,
        relative_time_continuous=rel_cont,
        frame_start_absolute_sample_index=frame_start_idx,
        toa_already_delay_corrected=already_delay_corr,
        observation_start=obs_start,
        observation_end=obs_end,
        observation_duration_sec=obs_dur,
        source_id=src_id,
        channel_id=ch_id,
        signal_id=sig_id,
        stream_id=final_stream_id,
        processing_delay_sec=proc_delay_sec,
        metadata=meta,
    )


@dataclass
class TOAResult:
    """Time of Arrival result object."""
    toas_sec: np.ndarray
    toas_ns: np.ndarray | list[int] | None = None
    timing_reference: str = "capture_relative"  # "hardware_absolute" | "capture_relative" | "externally_absolute"
    sample_count: int = 0
    time_unit: str = "s"
    sample0_hardware_timestamp_ns: int | None = None
    group_delay_corrected_sec: float = 0.0
    processing_delay_applied: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalDurationResult:
    """Pulse and signal duration result object."""
    durations_sec: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    pulse_count: int = 0
    mean_duration_sec: float | None = None
    median_duration_sec: float | None = None
    min_duration_sec: float | None = None
    max_duration_sec: float | None = None
    std_duration_sec: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeOffsetResult:
    """Matched events time offset result object."""
    offsets_sec: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    offsets_ns: np.ndarray | list[int] | None = None
    pair_count: int = 0
    mean_offset_sec: float | None = None
    median_offset_sec: float | None = None
    std_offset_sec: float | None = None
    min_offset_sec: float | None = None
    max_offset_sec: float | None = None
    timing_reference: str = "capture_relative"  # "hardware_absolute" | "capture_relative" | "coherent_relative"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterPulseIntervalResult:
    """Inter-Pulse Interval result object."""
    intervals_sec: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    intervals_ns: np.ndarray | list[int] | None = None
    timing_reference: str = "capture_relative"
    valid_intervals_count: int = 0
    gap_count: int = 0
    total_pairs_count: int = 0
    mean_interval_sec: float | None = None
    median_interval_sec: float | None = None
    min_interval_sec: float | None = None
    max_interval_sec: float | None = None
    std_interval_sec: float | None = None
    cross_frame_connected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)



@dataclass
class PRIResult:
    """Pulse Repetition Interval result object."""
    pri_sequence_sec: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    pri_count: int = 0
    nominal_pri_sec: float | None = None
    mean_pri_sec: float | None = None
    median_pri_sec: float | None = None
    min_pri_sec: float | None = None
    max_pri_sec: float | None = None
    std_pri_sec: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PRFResult:
    """Pulse Repetition Frequency result object."""
    prf_sequence_hz: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    prf_count: int = 0
    nominal_prf_hz: float | None = None
    mean_prf_hz: float | None = None
    median_prf_hz: float | None = None
    min_prf_hz: float | None = None
    max_prf_hz: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JitterResult:
    """Timing jitter and deviation result object."""
    jitter_deviations_sec: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    sample_count: int = 0
    std_jitter_sec: float | None = None
    rms_jitter_sec: float | None = None
    peak_to_peak_jitter_sec: float | None = None
    relative_jitter_percent: float | None = None
    nominal_pri_sec: float | None = None
    jitter_distribution: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BurstTimingResult:
    """Burst timing analysis result object."""
    num_bursts: int = 0
    burst_ids: list[Any] = field(default_factory=list)
    burst_start_toas: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    burst_end_toas: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    burst_start_toas_ns: list[int] | np.ndarray | None = None
    burst_end_toas_ns: list[int] | np.ndarray | None = None
    timing_reference: str = "capture_relative"
    burst_durations_sec: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    inter_burst_intervals_sec: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    burst_repetition_intervals_sec: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    burst_pulse_counts: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    unassigned_pulse_count: int = 0
    mean_burst_duration_sec: float | None = None
    mean_bri_sec: float | None = None
    mean_ibi_sec: float | None = None
    mean_pulses_per_burst: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)



@dataclass
class TimingPeriodicityResult:
    """Timing sequence periodicity analysis result object."""
    is_periodic: bool = False
    fundamental_period_sec: float | None = None
    fundamental_period_samples: int | None = None
    periodicity_confidence: float = 0.0
    periodic_pattern: list[float] | None = None
    harmonic_lags: list[int] = field(default_factory=list)
    harmonic_periods_samples: list[int] = field(default_factory=list)
    harmonic_periods_sec: list[float] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PulseDensityResult:
    """Pulse density and activity rate result object."""
    pulse_density_pps: float = 0.0
    pulse_count: int = 0
    in_window_pulse_count: int = 0
    excluded_event_count: int = 0
    observation_duration_sec: float = 0.0
    duty_cycle: float | None = None
    instantaneous_density_pps: np.ndarray | None = None
    time_bins_sec: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeAnalysisResult:
    """
    Unified aggregate result for all Time and Timing Analysis modules.
    """
    timing_input_summary: dict[str, Any] = field(default_factory=dict)
    executed_modules: list[str] = field(default_factory=list)
    stream_id: tuple[int, int] | str | None = None
    is_streaming_update: bool = False
    toa: TOAResult | None = None
    signal_duration: SignalDurationResult | None = None
    time_offset: TimeOffsetResult | None = None
    inter_pulse_interval: InterPulseIntervalResult | None = None
    pri: PRIResult | None = None
    prf: PRFResult | None = None
    jitter: JitterResult | None = None
    burst_timing: BurstTimingResult | None = None
    periodicity: TimingPeriodicityResult | None = None
    pulse_density: PulseDensityResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ipi(self) -> InterPulseIntervalResult | None:
        return self.inter_pulse_interval

    @property
    def duration(self) -> SignalDurationResult | None:
        return self.signal_duration

    @property
    def offset(self) -> TimeOffsetResult | None:
        return self.time_offset

    @property
    def burst(self) -> BurstTimingResult | None:
        return self.burst_timing

    @property
    def bursts(self) -> BurstTimingResult | None:
        return self.burst_timing

    @property
    def density(self) -> PulseDensityResult | None:
        return self.pulse_density



__all__ = [
    "UINT64_MIN",
    "UINT64_MAX",
    "TimingInput",
    "build_timing_input",
    "TOAResult",
    "SignalDurationResult",
    "TimeOffsetResult",
    "InterPulseIntervalResult",
    "PRIResult",
    "PRFResult",
    "JitterResult",
    "BurstTimingResult",
    "TimingPeriodicityResult",
    "PulseDensityResult",
    "TimeAnalysisResult",
]


def main() -> None:
    print("=" * 60)
    print("  Time and Timing Analysis: Models")
    print("=" * 60)
    inp = TimingInput(sample_rate_hz=1e6, hardware_timestamp_ns=1000000)
    print(f"Sample TimingInput: {inp}")
    print("=" * 60)


if __name__ == "__main__":
    main()
