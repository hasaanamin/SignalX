"""Immutable identity for the bounded LTE synchronization search."""
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import hashlib
import json
import math


@dataclass(frozen=True)
class LTESyncConfig:
    input_sample_rate_hz: float = 2_000_000.0
    max_frequency_offset_hz: float = 30_000.0
    frequency_step_hz: float = 7_500.0
    pss_threshold: float = .22
    sss_threshold: float = .35
    sss_peak_ratio: float = 1.3
    confirmations: int = 2
    timing_tolerance_samples: int = 4
    loss_timeout_half_frames: int = 4
    max_candidates: int = 8
    max_cells: int = 4
    search_chunk_samples: int = 4096
    max_input_samples: int = 262144
    tracking: bool = True
    integration_half_frames: int = field(default=1, kw_only=True)
    version: str = "lte_fdd_normal_sync_v1"

    def __post_init__(self):
        for name in ("input_sample_rate_hz", "max_frequency_offset_hz", "frequency_step_hz",
                     "pss_threshold", "sss_threshold", "sss_peak_ratio"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(v))
        if self.input_sample_rate_hz not in (1_920_000., 2_000_000.):
            raise ValueError("LTE sync supports 1.92 or 2 MS/s input")
        if not 0 <= self.max_frequency_offset_hz <= 100_000 or not 0 < self.frequency_step_hz <= 7500:
            raise ValueError("Invalid bounded CFO search")
        # Avoid overflow in the ratio for very small positive user steps.
        if self.max_frequency_offset_hz > 20*self.frequency_step_hz:
            raise ValueError("CFO search exceeds 41 branches")
        if not 0 < self.pss_threshold < 1 or not 0 < self.sss_threshold < 1 or self.sss_peak_ratio <= 1:
            raise ValueError("Invalid normalized correlation thresholds")
        for name, lower, upper in (("confirmations", 2, 8), ("timing_tolerance_samples", 0, 16),
                ("loss_timeout_half_frames", 2, 20), ("max_candidates", 1, 32),
                ("max_cells", 1, 16), ("search_chunk_samples", 512, 16384),
                ("max_input_samples", 16384, 1_048_576)):
            v = getattr(self, name)
            if type(v) is not int or not lower <= v <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        if self.version != "lte_fdd_normal_sync_v1":
            raise ValueError("Unsupported synchronization implementation version")
        if type(self.tracking) is not bool:
            raise ValueError("tracking must be bool")
        if type(self.integration_half_frames) is not int or self.integration_half_frames not in (1, 2, 4, 8):
            raise ValueError("integration_half_frames must be 1, 2, 4 or 8")

    @classmethod
    def low_snr(cls, input_sample_rate_hz=2_000_000.):
        """Repeated-sync profile; longer dwell, not a certified false-lock rate."""
        return cls(input_sample_rate_hz=input_sample_rate_hz, integration_half_frames=4,
                   pss_threshold=.10, sss_threshold=.20, sss_peak_ratio=1.5,
                   loss_timeout_half_frames=8)

    @classmethod
    def native(cls, *, low_snr=False):
        """Acquire genuine 1.92 MS/s IQ without a sample-rate converter.

        The source must actually provide this rate; never relabel 2 MS/s IQ.
        Existing 2 MS/s constructors retain their conversion/compatibility path.
        """
        if type(low_snr) is not bool:
            raise ValueError('low_snr must be bool')
        return cls.low_snr(1_920_000.) if low_snr else cls(input_sample_rate_hz=1_920_000.)

    @property
    def profile_id(self):
        return _profile_id(self)


@lru_cache(maxsize=32)
def _profile_id(config):
    record = {**asdict(config), "processing_rate_hz": 1920000,
              "resampling": ("native_1920000_identity_v1" if config.input_sample_rate_hz == 1_920_000.
                             else "causal_24_25_kaiser1201_cutoff700k_v1"),
              "statistic": "normalized_pss_smoothed_channel_sss_v1"}
    if config.integration_half_frames == 1:
        record.pop("integration_half_frames")  # unchanged 2 MS/s compatibility identity
    else:
        record["statistic"] = "repeated_noncoherent_pss_coherent_sss_alias_aware_v1"
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return "lte_sync_" + hashlib.sha256(encoded.encode()).hexdigest()[:20]
