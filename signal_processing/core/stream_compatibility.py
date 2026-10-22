"""Exact immutable stream identity shared by positioned admission boundaries.

No numerical tolerance reinterprets sampling clocks or tuner settings. Equal
reconstructed records are equivalent. Only pipeline construction may leave the
first center frequency unspecified; binding is committed after full preflight.
"""
from dataclasses import replace, fields
from .contracts import StreamInfo
from .exceptions import StreamConsistencyError


def require_stream(actual: StreamInfo, expected: StreamInfo, *, bind_center=False) -> StreamInfo:
    if actual is expected:
        return expected
    candidate = expected
    if bind_center and expected.center_frequency_hz is None:
        candidate = replace(expected, center_frequency_hz=actual.center_frequency_hz)
    if actual != candidate:
        changed = ', '.join(f.name.replace('_', ' ') for f in fields(expected)
                            if getattr(actual, f.name) != getattr(candidate, f.name))
        raise StreamConsistencyError(f'Stream metadata differs ({changed}) from the exact admitted definition; correct input or reset/reconfigure')
    return candidate


def same_sample_rate(actual: float, expected: float) -> bool:
    return actual == expected
