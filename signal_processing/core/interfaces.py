"""Structural interface definitions (Protocols) for signal processing subsystems.

These protocols establish minimal structural contracts across subsystems without
introducing concrete implementation dependencies or deep inheritance hierarchies.
"""

from typing import Protocol, runtime_checkable

from signal_processing.core.contracts import (
    DetectionMeasurement,
    IQBlock,
    NoiseEstimate,
    SegmentationDecision,
    StreamInfo,
)


@runtime_checkable
class Resettable(Protocol):
    """Protocol for stateful components that support resetting their internal state."""

    def reset(self) -> None:
        """Reset internal state to initial baseline conditions."""
        ...


@runtime_checkable
class IQSource(Resettable, Protocol):
    """Protocol representing any streaming IQ source (synthetic, recorded file, or SDR).

    A source emits discrete IQBlock chunks. Returning None indicates that a
    finite stream has completed.
    """

    @property
    def stream_info(self) -> StreamInfo:
        """Stream metadata describing this source."""
        ...

    def read_block(self) -> IQBlock | None:
        """Read and return the next sequential IQBlock.

        Returns:
            The next IQBlock, or None if the stream has reached EOF / stopped.
        """
        ...

    def reset(self) -> None:
        """Reset the source stream to the beginning or initial sample."""
        ...


@runtime_checkable
class NoiseEstimatorProtocol(Resettable, Protocol):
    """Protocol for stateful noise power estimators."""

    def update(self, block: IQBlock) -> NoiseEstimate:
        """Update noise estimation with a new incoming IQ block.

        Args:
            block: The newly arrived streaming IQBlock.

        Returns:
            Current NoiseEstimate.
        """
        ...

    def reset(self) -> None:
        """Reset estimator history and state."""
        ...


@runtime_checkable
class DetectorProtocol(Resettable, Protocol):
    """Protocol for causal signal detectors."""

    def process(
        self,
        block: IQBlock,
        noise_estimate: NoiseEstimate,
    ) -> DetectionMeasurement:
        """Process an IQ block against a noise reference to measure signal presence.

        Args:
            block: Incoming streaming IQBlock.
            noise_estimate: Reference noise level.

        Returns:
            DetectionMeasurement containing measured powers and threshold decisions.
        """
        ...

    def reset(self) -> None:
        """Reset detector state (e.g. hysteresis or rolling filters)."""
        ...


@runtime_checkable
class SegmenterProtocol(Resettable, Protocol):
    """Protocol for signal segmentation state machines."""

    def update(
        self,
        measurement: DetectionMeasurement,
    ) -> SegmentationDecision:
        """Update segmentation state machine with a new detection measurement.

        Args:
            measurement: The latest detection measurement.

        Returns:
            SegmentationDecision with current state and emitted transitions.
        """
        ...

    def reset(self) -> None:
        """Reset segmentation state machine to IDLE."""
        ...
