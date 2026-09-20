"""Stateless validation helpers for stream integrity and continuity."""

import math

from signal_processing.core.contracts import IQBlock
from signal_processing.core.exceptions import StreamConsistencyError


def validate_consecutive_blocks(previous: IQBlock, current: IQBlock) -> None:
    """Validate stream integrity and continuity between two consecutive IQ blocks.

    Checks:
        1. Both arguments are IQBlock instances.
        2. Stream identifiers match exactly.
        3. Logical source, channel, and dtype metadata match exactly.
        4. Sampling rates match within tolerance.
        5. Center frequencies are compatible (both None or both equal).
        6. Sequence numbers are strictly consecutive (current == previous + 1).
        7. Sample indexing is continuous:
           current.start_sample == previous.end_sample + current.dropped_samples_before.
           If samples were lost, they must be explicitly declared via
           `dropped_samples_before`; unexplained gaps or overlaps are rejected.

    Args:
        previous: The earlier IQBlock.
        current: The subsequent IQBlock.

    Raises:
        StreamConsistencyError: If any consistency or continuity check fails.
    """
    if not isinstance(previous, IQBlock) or not isinstance(current, IQBlock):
        raise StreamConsistencyError(
            f"Both blocks must be IQBlock instances, got {type(previous)} and {type(current)}"
        )

    # 1. Stream ID consistency
    if previous.stream.stream_id != current.stream.stream_id:
        raise StreamConsistencyError(
            f"Stream ID mismatch: previous block has '{previous.stream.stream_id}', "
            f"current block has '{current.stream.stream_id}'"
        )

    # 2. Source ID consistency
    if previous.stream.source_id != current.stream.source_id:
        raise StreamConsistencyError(
            f"Source ID mismatch: previous block has '{previous.stream.source_id}', "
            f"current block has '{current.stream.source_id}'"
        )

    # 3. Channel ID consistency
    if previous.stream.channel_id != current.stream.channel_id:
        raise StreamConsistencyError(
            f"Channel ID mismatch: previous block has '{previous.stream.channel_id}', "
            f"current block has '{current.stream.channel_id}'"
        )

    # 4. dtype consistency
    if previous.stream.dtype != current.stream.dtype:
        raise StreamConsistencyError(
            f"dtype mismatch: previous block has '{previous.stream.dtype}', "
            f"current block has '{current.stream.dtype}'"
        )

    # 5. Sample rate consistency
    if not math.isclose(
        previous.stream.sample_rate_hz, current.stream.sample_rate_hz, rel_tol=1e-9
    ):
        raise StreamConsistencyError(
            f"Sample rate mismatch: previous block is {previous.stream.sample_rate_hz} Hz, "
            f"current block is {current.stream.sample_rate_hz} Hz"
        )

    # 6. Center frequency compatibility
    prev_cf = previous.stream.center_frequency_hz
    curr_cf = current.stream.center_frequency_hz
    if (prev_cf is None) != (curr_cf is None):
        raise StreamConsistencyError(
            f"Incompatible center frequency configuration: previous={prev_cf}, current={curr_cf}"
        )
    if (
        prev_cf is not None
        and curr_cf is not None
        and not math.isclose(prev_cf, curr_cf, rel_tol=1e-9)
    ):
        raise StreamConsistencyError(
            f"Center frequency mismatch: previous block is {prev_cf} Hz, "
            f"current block is {curr_cf} Hz"
        )

    # 7. Sequence number ordering
    if current.sequence_number <= previous.sequence_number:
        raise StreamConsistencyError(
            f"Duplicate or out-of-order block sequence number: previous={previous.sequence_number}, "
            f"current={current.sequence_number}"
        )

    if current.sequence_number != previous.sequence_number + 1:
        raise StreamConsistencyError(
            f"Block sequence number gap: expected {previous.sequence_number + 1}, "
            f"got {current.sequence_number}"
        )

    # 8. Sample continuity & declared drop validation
    if current.start_sample < previous.end_sample:
        raise StreamConsistencyError(
            f"Sample index overlap detected: previous block ends at sample {previous.end_sample}, "
            f"but current block starts backwards at sample {current.start_sample}"
        )

    expected_start = previous.end_sample + current.dropped_samples_before
    if current.start_sample != expected_start:
        gap = current.start_sample - previous.end_sample
        raise StreamConsistencyError(
            f"Unexplained sample discontinuity: previous block ended at {previous.end_sample}, "
            f"current block starts at {current.start_sample} (gap of {gap} samples), "
            f"but declared dropped_samples_before is {current.dropped_samples_before}"
        )
