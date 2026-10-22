"""Immutable classic LTE MIB payload and attributed CRC-valid PBCH outcome."""
from dataclasses import asdict, dataclass
from fractions import Fraction
import math

from signal_processing.sources.streaming import SourceMetadata


@dataclass(frozen=True)
class LTEMIB:
    """24 broadcast bits; trailing release-specific bits are preserved, not decoded."""
    payload_bits: tuple[int, ...]

    def __post_init__(self):
        bits = tuple(self.payload_bits)
        if len(bits) != 24 or any(type(v) is not int or v not in (0, 1) for v in bits):
            raise ValueError("MIB requires exactly 24 integer bits")
        object.__setattr__(self, "payload_bits", bits)
        if self._integer(0, 3) > 5:
            raise ValueError("Reserved LTE downlink-bandwidth code")

    def _integer(self, start, stop):
        return int("".join(map(str, self.payload_bits[start:stop])), 2)

    @property
    def downlink_resource_blocks(self):
        return (6, 15, 25, 50, 75, 100)[self._integer(0, 3)]

    @property
    def downlink_bandwidth_hz(self):
        return (1_400_000, 3_000_000, 5_000_000, 10_000_000, 15_000_000, 20_000_000)[self._integer(0, 3)]

    @property
    def phich_duration(self):
        return "extended" if self.payload_bits[3] else "normal"

    @property
    def phich_resource(self):
        return ("1/6", "1/2", "1", "2")[self._integer(4, 6)]

    @property
    def system_frame_number_msb(self):
        return self._integer(6, 14)

    def to_dict(self):
        return {"payload_bits": list(self.payload_bits),
                "downlink_resource_blocks": self.downlink_resource_blocks,
                "downlink_bandwidth_hz": self.downlink_bandwidth_hz,
                "phich_duration": self.phich_duration, "phich_resource": self.phich_resource,
                "system_frame_number_msb": self.system_frame_number_msb}


@dataclass(frozen=True)
class LTEPBCHObservation:
    mib: LTEMIB
    pci: int
    antenna_ports: int
    frame_quarter: int
    combined_frames: int
    epoch: int
    stream_id: str
    profile_id: str
    processing_frame_start: int
    input_frame_start: Fraction
    frequency_offset_hz: float
    prepared_input_end: int
    source_metadata: SourceMetadata | None = None
    source_configuration_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.mib, LTEMIB):
            raise TypeError("Expected LTEMIB")
        for name, lower, upper in (("pci", 0, 503), ("frame_quarter", 0, 3),
                                   ("combined_frames", 1, 4)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"Invalid {name}")
        if type(self.antenna_ports) is not int or self.antenna_ports not in (1, 2, 4):
            raise ValueError("Invalid antenna-port count")
        if self.combined_frames > self.frame_quarter+1:
            raise ValueError("Cannot combine across a BCH block boundary")
        for name in ("epoch", "processing_frame_start", "prepared_input_end"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Invalid {name}")
        if not isinstance(self.input_frame_start, Fraction):
            raise TypeError("Exact rational input coordinate required")
        for name in ("stream_id", "profile_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"Invalid {name}")
        if isinstance(self.frequency_offset_hz, bool) or not isinstance(self.frequency_offset_hz, (float, int)) or not math.isfinite(self.frequency_offset_hz):
            raise ValueError("Invalid frequency offset")
        if self.source_metadata is not None and not isinstance(self.source_metadata, SourceMetadata):
            raise TypeError("Invalid source metadata")
        if (self.source_metadata is None) != (self.source_configuration_id is None):
            raise ValueError("Incomplete source provenance")
        if self.source_configuration_id is not None and (not isinstance(self.source_configuration_id, str) or not self.source_configuration_id):
            raise ValueError("Invalid source configuration")

    @property
    def system_frame_number(self):
        return 4*self.mib.system_frame_number_msb+self.frame_quarter

    def to_dict(self):
        data = asdict(self)
        data["mib"] = self.mib.to_dict()
        data["input_frame_start"] = {"numerator": self.input_frame_start.numerator,
                                     "denominator": self.input_frame_start.denominator}
        data["system_frame_number"] = self.system_frame_number
        data["crc_valid"] = True
        return data

    @classmethod
    def from_dict(cls, record):
        data = dict(record)
        mib_record = data.pop("mib")
        mib = LTEMIB(tuple(mib_record["payload_bits"]))
        if mib.to_dict() != mib_record or data.pop("crc_valid") is not True:
            raise ValueError("Contradictory MIB serialization")
        sfn = data.pop("system_frame_number")
        coordinate = data.pop("input_frame_start")
        if set(coordinate) != {"numerator", "denominator"} or any(type(v) is not int for v in coordinate.values()):
            raise ValueError("Invalid rational coordinate")
        if data.get("source_metadata") is not None:
            data["source_metadata"] = SourceMetadata(**data["source_metadata"])
        result = cls(mib=mib, input_frame_start=Fraction(**coordinate), **data)
        if type(sfn) is not int or sfn != result.system_frame_number:
            raise ValueError("Contradictory system frame number")
        return result


@dataclass(frozen=True)
class LTEBroadcastBatch:
    synchronization: tuple
    mibs: tuple[LTEPBCHObservation, ...]

    def __post_init__(self):
        from .contracts import LTESyncObservation
        for name, kind in (("synchronization", LTESyncObservation), ("mibs", LTEPBCHObservation)):
            value = tuple(getattr(self, name))
            if any(not isinstance(v, kind) for v in value):
                raise TypeError(f"Invalid {name}")
            object.__setattr__(self, name, value)
