"""Cell synchronization observations, distinct from energy/features/MIB decode."""
from dataclasses import dataclass
from fractions import Fraction
import math
from signal_processing.sources.streaming import SourceMetadata


@dataclass(frozen=True)
class LTESyncObservation:
    pci: int
    state: str
    epoch: int
    profile_id: str
    stream_id: str
    processing_pss_sample: int
    processing_frame_start: int
    input_pss_sample: Fraction
    input_frame_start: Fraction
    frame_start_received: bool
    frequency_offset_hz: float
    pss_metric: float
    sss_metric: float
    sss_peak_ratio: float
    confirmations: int
    prepared_input_end: int
    source_metadata: SourceMetadata | None = None
    source_configuration_id: str | None = None

    def __post_init__(self):
        if type(self.pci) is not int or not 0 <= self.pci < 504:
            raise ValueError("Invalid PCI")
        if self.state not in ("CANDIDATE", "SYNCHRONIZED"):
            raise ValueError("Invalid synchronization state")
        for name in ("epoch", "processing_pss_sample", "confirmations", "prepared_input_end"):
            v = getattr(self, name)
            if type(v) is not int or v < 0:
                raise ValueError(f"{name} must be nonnegative integer")
        if type(self.processing_frame_start) is not int or self.confirmations < 1:
            raise ValueError("Invalid frame index/confirmation count")
        if self.state == "SYNCHRONIZED" and self.confirmations < 2:
            raise ValueError("Synchronization requires repeated observations")
        for name in ("input_pss_sample", "input_frame_start"):
            if not isinstance(getattr(self, name), Fraction):
                raise TypeError("Input timing must retain its exact rational representation")
        for name in ("profile_id", "stream_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"Invalid {name}")
        for name in ("frequency_offset_hz", "pss_metric", "sss_metric", "sss_peak_ratio"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v):
                raise ValueError(f"Invalid {name}")
        if not 0 <= self.pss_metric <= 1+1e-10 or not 0 <= self.sss_metric <= 1+1e-10 or self.sss_peak_ratio < 1:
            raise ValueError("Invalid normalized metrics")
        if type(self.frame_start_received) is not bool:
            raise ValueError("frame_start_received must be bool")
        if self.source_metadata is not None and not isinstance(self.source_metadata, SourceMetadata):
            raise TypeError("Invalid source provenance")
        if (self.source_metadata is None) != (self.source_configuration_id is None):
            raise ValueError("Source configuration and provenance must be present together")
        if self.source_configuration_id is not None and (not isinstance(self.source_configuration_id, str) or not self.source_configuration_id):
            raise ValueError("Invalid source configuration identity")

    def to_dict(self):
        from dataclasses import asdict
        result = asdict(self)
        for name in ("input_pss_sample", "input_frame_start"):
            value = getattr(self, name)
            result[name] = {"numerator": value.numerator, "denominator": value.denominator}
        return result

    @classmethod
    def from_dict(cls, record):
        data = dict(record)
        if data.get("source_metadata") is not None:
            data["source_metadata"] = SourceMetadata(**data["source_metadata"])
        for name in ("input_pss_sample", "input_frame_start"):
            value = data[name]
            if set(value) != {"numerator", "denominator"} or any(type(v) is not int for v in value.values()):
                raise ValueError("Invalid rational coordinate")
            data[name] = Fraction(value["numerator"], value["denominator"])
        return cls(**data)
