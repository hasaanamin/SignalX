"""Integer/rational sample-time conversion, without accumulated packet rounding."""
from fractions import Fraction


def sample_offset_ns(samples: int, sample_rate_hz: float) -> int:
    """Round one absolute index difference to nearest ns (ties to even)."""
    return round(Fraction(int(samples) * 1_000_000_000, 1) / Fraction(float(sample_rate_hz)))
