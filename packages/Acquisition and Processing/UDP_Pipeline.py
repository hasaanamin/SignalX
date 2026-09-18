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

import copy
import importlib
import struct
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .UDP_Acquisition import (
    UDP_IQ_HEADER_FMT,
    UDPIQPacket,
    parse_udp_iq_packet,
    udp_packet_to_canonical_iq,
)
from .UDP_Receiver import IQFrame
from .Validation import validate_iq
from .Calibration import calibrate_iq

# Import modules using robust relative import / importlib
_dc_rem = importlib.import_module(".DC Removal", package=__package__ or "packages.Acquisition and Processing")
_norm = importlib.import_module(".Peak Normalization", package=__package__ or "packages.Acquisition and Processing")
_filt = importlib.import_module(".Filtering", package=__package__ or "packages.Acquisition and Processing")
_resamp = importlib.import_module(".Resampling", package=__package__ or "packages.Acquisition and Processing")

remove_dc = _dc_rem.remove_dc
normalize_iq = _norm.normalize_iq
filter_iq = _filt.filter_iq
resample_iq = _resamp.resample_iq
decimate_iq = _resamp.decimate_iq


@dataclass
class AcquisitionProcessingConfig:
    """
    Configuration options for the Acquisition Processing Pipeline.

    Stages:
    -------
    1. validate : bool
        If True, validates IQ integrity (finite complex samples, consistent metadata).
    2. calibration_enabled : bool
        If True, applies linear IQ calibration (gain, phase, DC offset, 2x2 matrix).
    3. dc_removal_enabled : bool
        If True, applies block or reference DC offset removal.
    4. filtering_enabled : bool
        If True, applies FIR filtering using stateful streaming continuity per channel.
    5. resampling_enabled : bool
        If True, applies polyphase resampling or decimation.
    6. normalization_enabled : bool
        MUST DEFAULT FALSE. Normalization alters absolute amplitude information
        needed by downstream modulation, power, and pulse analysis packages.
    """
    validate: bool = True

    calibration_enabled: bool = False
    calibration_options: dict[str, Any] = field(default_factory=dict)

    dc_removal_enabled: bool = False
    dc_removal_options: dict[str, Any] = field(default_factory=dict)

    filtering_enabled: bool = False
    filtering_options: dict[str, Any] = field(default_factory=dict)

    resampling_enabled: bool = False
    resampling_options: dict[str, Any] = field(default_factory=dict)

    normalization_enabled: bool = False  # Critical: MUST DEFAULT OFF
    normalization_options: dict[str, Any] = field(default_factory=dict)


class AcquisitionPipeline:
    """
    High-level acquisition processing pipeline.

    Receives decoded canonical IQ records or assembled IQFrames, applies configured
    acquisition-stage conditioning stages, maintains per-channel streaming filter
    states, and emits validated canonical IQ records for downstream analysis packages.

    Design Principles:
    ------------------
    - Zero downstream leakage: Downstream packages only see canonical IQ records/frames.
    - Streaming continuity: FIR filter states ('zf' -> 'zi') are maintained per
      (signal_id, channel_id) pair with zero_phase=False.
    - No inter-channel state crosstalk: Each channel maintains independent DSP filter state.
    - Processing provenance: Every enabled stage records its operations and parameters
      under metadata['processing'].
    """

    def __init__(
        self,
        config: AcquisitionProcessingConfig | None = None,
        sample_rate: float | None = None,
    ) -> None:
        self.config = config or AcquisitionProcessingConfig()
        self.default_sample_rate = float(sample_rate) if sample_rate is not None else None

        # Filter states isolated per (signal_id, channel_id)
        self._filter_states: dict[tuple[int, int], np.ndarray] = {}

    def reset_filter_states(
        self,
        signal_id: int | None = None,
        channel_id: int | None = None,
    ) -> None:
        """
        Resets FIR filter state. If signal_id/channel_id are given, resets only that stream;
        otherwise resets all channel filter states.
        """
        if signal_id is None and channel_id is None:
            self._filter_states.clear()
        else:
            keys_to_del = [
                k for k in self._filter_states
                if (signal_id is None or k[0] == signal_id) and (channel_id is None or k[1] == channel_id)
            ]
            for k in keys_to_del:
                del self._filter_states[k]

    def process_record(
        self,
        record: dict[str, Any],
        signal_id: int = 0,
        channel_id: int = 1,
    ) -> dict[str, Any]:
        """
        Applies configured acquisition processing to an individual canonical IQ record.

        Parameters
        ----------
        record : dict[str, Any]
            Canonical Acquisition and Processing IQ record.
        signal_id : int
            Signal stream ID for filter state isolation.
        channel_id : int
            Channel ID for filter state isolation.

        Returns
        -------
        dict[str, Any]
            Processed canonical IQ record.
        """
        out = copy.deepcopy(record)

        # 1. Input Validation
        if self.config.validate:
            val_res = validate_iq(out)
            if not val_res.get("valid", True):
                errs = val_res.get("errors", [])
                raise ValueError(f"IQ Record validation failed: {errs}")

        # 2. Static Linear Calibration
        if self.config.calibration_enabled:
            cal_opts = copy.deepcopy(self.config.calibration_options)
            out = calibrate_iq(out, **cal_opts)

        # 3. DC Removal
        if self.config.dc_removal_enabled:
            dc_opts = copy.deepcopy(self.config.dc_removal_options)
            out = remove_dc(out, **dc_opts)

        # 4. Streaming FIR Filtering
        if self.config.filtering_enabled:
            filt_opts = copy.deepcopy(self.config.filtering_options)
            state_key = (signal_id, channel_id)
            zi = self._filter_states.get(state_key)

            # Enforce streaming causality
            filt_opts["zero_phase"] = False
            filt_opts["zi"] = zi
            filt_opts["return_info"] = True

            out, filt_info = filter_iq(out, **filt_opts)
            self._filter_states[state_key] = filt_info.get("zf")

        # 5. Resampling / Decimation
        if self.config.resampling_enabled:
            resamp_opts = copy.deepcopy(self.config.resampling_options)
            if "input_sample_rate" not in resamp_opts:
                fs = out.get("sample_rate") if isinstance(out, dict) else self.default_sample_rate
                if fs is not None:
                    resamp_opts["input_sample_rate"] = fs
            if "factor" in resamp_opts:
                factor = resamp_opts.pop("factor")
                out = decimate_iq(out, factor=factor, **resamp_opts)
            else:
                out = resample_iq(out, **resamp_opts)

        # 6. Peak / RMS Normalization (Must default OFF)
        if self.config.normalization_enabled:
            norm_opts = copy.deepcopy(self.config.normalization_options)
            out = normalize_iq(out, **norm_opts)

        return out

    def process_frame(self, frame: IQFrame) -> IQFrame:
        """
        Applies configured acquisition processing to all channels in an IQFrame.

        Parameters
        ----------
        frame : IQFrame
            Assembled multi-channel or single-channel frame.

        Returns
        -------
        IQFrame
            IQFrame with processed channel records.
        """
        processed_channels: dict[int, dict] = {}
        for ch_id, ch_rec in frame.channels.items():
            processed_rec = self.process_record(
                ch_rec,
                signal_id=frame.signal_id,
                channel_id=ch_id,
            )
            processed_channels[ch_id] = processed_rec

        out_frame = copy.deepcopy(frame)
        out_frame.channels = processed_channels
        return out_frame

    def process(
        self,
        data: bytes | bytearray | memoryview | UDPIQPacket | dict[str, Any] | IQFrame,
        signal_id: int = 0,
        channel_id: int = 1,
        sample_rate: float | None = None,
    ) -> dict[str, Any] | IQFrame:
        """
        Unified polymorphic processing entry point.

        Accepts:
        - Raw UDP datagram bytes -> decodes, validates, processes, returns canonical record.
        - UDPIQPacket -> converts to canonical record, processes, returns canonical record.
        - dict (canonical IQ record) -> processes, returns canonical record.
        - IQFrame -> processes all channels in frame, returns processed IQFrame.
        """
        fs = sample_rate if sample_rate is not None else self.default_sample_rate

        if isinstance(data, (bytes, bytearray, memoryview)):
            packet = parse_udp_iq_packet(data, sample_rate=fs)
            rec = udp_packet_to_canonical_iq(packet, sample_rate=fs)
            return self.process_record(rec, signal_id=packet.signal_id, channel_id=packet.channel_id)

        if isinstance(data, UDPIQPacket):
            rec = udp_packet_to_canonical_iq(data, sample_rate=fs)
            return self.process_record(rec, signal_id=data.signal_id, channel_id=data.channel_id)

        if isinstance(data, IQFrame):
            return self.process_frame(data)

        if isinstance(data, dict):
            return self.process_record(data, signal_id=signal_id, channel_id=channel_id)

        raise TypeError(f"Unsupported input type for AcquisitionPipeline: {type(data).__name__}")


__all__ = [
    "AcquisitionProcessingConfig",
    "AcquisitionPipeline",
]


def main() -> None:
    print("=" * 60)
    print("  AcquisitionPipeline: Demonstration")
    print("=" * 60)
    # Configure pipeline with DC removal and streaming FIR filter
    config = AcquisitionProcessingConfig(
        validate=True,
        dc_removal_enabled=True,
        filtering_enabled=True,
        filtering_options={"cutoff": 200e3, "filter_type": "lowpass"},
        normalization_enabled=False,  # Defaults OFF
    )

    pipeline = AcquisitionPipeline(config=config, sample_rate=1.0e6)

    # Simulate block streaming for 2 channels
    fs = 1.0e6
    t = np.arange(256) / fs
    for block_idx in range(3):
        for ch in [1, 2]:
            raw_iq = np.exp(1j * 2 * np.pi * 50e3 * t) + 0.2  # Signal with DC bias
            i_f16 = raw_iq.real.astype(np.float16)
            q_f16 = raw_iq.imag.astype(np.float16)
            payload = np.column_stack([i_f16, q_f16]).tobytes()
            header = struct.pack(UDP_IQ_HEADER_FMT, 100, ch, 256)
            datagram = header + payload

            out_rec = pipeline.process(datagram)
            print(f"Block {block_idx}, Ch {ch} processed -> samples={out_rec['sample_count']}, "
                  f"stages={len(out_rec['metadata']['processing'])}")

    print("=" * 60)


if __name__ == "__main__":
    main()
