"""
Acquisition and Processing Package
Provides source-agnostic IQ acquisition, PDW/IQDW parsing, timestamp alignment,
sample rate handling, calibration, DC removal, normalization, filtering,
resampling, and validation, plus UDP Ingress & Pipeline.
"""

import importlib

_acq = importlib.import_module(".Acquisition", package=__name__)
_pdw = importlib.import_module(".PDW_Parsing", package=__name__)
_iqdw = importlib.import_module(".IQDW_Parsing", package=__name__)
_sr = importlib.import_module(".Sample_Rate", package=__name__)
_cal = importlib.import_module(".Calibration", package=__name__)
_ts_align = importlib.import_module(".Timestamp Alignment", package=__name__)
_dc_rem = importlib.import_module(".DC Removal", package=__name__)
_norm = importlib.import_module(".Peak Normalization", package=__name__)
_filt = importlib.import_module(".Filtering", package=__name__)
_resamp = importlib.import_module(".Resampling", package=__name__)
_val = importlib.import_module(".Validation", package=__name__)
_udp_acq = importlib.import_module(".UDP_Acquisition", package=__name__)
_udp_recv = importlib.import_module(".UDP_Receiver", package=__name__)
_udp_pipe = importlib.import_module(".UDP_Pipeline", package=__name__)

parse_iq = _acq.parse_iq
parse_pdw = _pdw.parse_pdw
parse_iqdw = _iqdw.parse_iqdw
handle_sample_rate = _sr.handle_sample_rate
calibrate_iq = _cal.calibrate_iq
align_timestamps = _ts_align.align_timestamps
remove_dc = _dc_rem.remove_dc
normalize_iq = _norm.normalize_iq
filter_iq = _filt.filter_iq
resample_iq = _resamp.resample_iq
decimate_iq = _resamp.decimate_iq
validate_iq = _val.validate_iq
validate_iq_record = _val.validate_iq_record

parse_udp_iq_packet = _udp_acq.parse_udp_iq_packet
udp_packet_to_canonical_iq = _udp_acq.udp_packet_to_canonical_iq
UDPIQPacket = _udp_acq.UDPIQPacket
UDP_IQ_HEADER_FMT = _udp_acq.UDP_IQ_HEADER_FMT
UDP_IQ_HEADER_SIZE = _udp_acq.UDP_IQ_HEADER_SIZE
UDP_IQ_HEADER_FMT_LEGACY = _udp_acq.UDP_IQ_HEADER_FMT_LEGACY
UDP_IQ_HEADER_SIZE_LEGACY = _udp_acq.UDP_IQ_HEADER_SIZE_LEGACY
UDP_IQ_HEADER_FMT_TIMESTAMPED = _udp_acq.UDP_IQ_HEADER_FMT_TIMESTAMPED
UDP_IQ_HEADER_SIZE_TIMESTAMPED = _udp_acq.UDP_IQ_HEADER_SIZE_TIMESTAMPED
UDP_MAX_DATAGRAM_BYTES = _udp_acq.UDP_MAX_DATAGRAM_BYTES
MAX_IQ_COUNT = _udp_acq.MAX_IQ_COUNT
make_udp_iq_packet = _udp_acq.make_udp_iq_packet
send_iq = _udp_acq.send_iq

UDPReceiver = _udp_recv.UDPReceiver
UDPReceiverConfig = _udp_recv.UDPReceiverConfig
UDPFrameAssembler = _udp_recv.UDPFrameAssembler
IQFrame = _udp_recv.IQFrame

AcquisitionPipeline = _udp_pipe.AcquisitionPipeline
AcquisitionProcessingConfig = _udp_pipe.AcquisitionProcessingConfig

__all__ = [
    "parse_iq",
    "parse_pdw",
    "parse_iqdw",
    "align_timestamps",
    "handle_sample_rate",
    "remove_dc",
    "normalize_iq",
    "calibrate_iq",
    "filter_iq",
    "resample_iq",
    "decimate_iq",
    "validate_iq",
    "validate_iq_record",
    "parse_udp_iq_packet",
    "udp_packet_to_canonical_iq",
    "UDPIQPacket",
    "UDP_IQ_HEADER_FMT",
    "UDP_IQ_HEADER_SIZE",
    "UDP_IQ_HEADER_FMT_LEGACY",
    "UDP_IQ_HEADER_SIZE_LEGACY",
    "UDP_IQ_HEADER_FMT_TIMESTAMPED",
    "UDP_IQ_HEADER_SIZE_TIMESTAMPED",
    "UDP_MAX_DATAGRAM_BYTES",
    "MAX_IQ_COUNT",
    "make_udp_iq_packet",
    "send_iq",
    "UDPReceiver",
    "UDPReceiverConfig",
    "UDPFrameAssembler",
    "IQFrame",
    "AcquisitionPipeline",
    "AcquisitionProcessingConfig",
]
