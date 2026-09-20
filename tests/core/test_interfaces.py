"""Unit tests for interfaces and protocols."""


from signal_processing.core import (
    DetectionMeasurement,
    DetectorProtocol,
    IQBlock,
    IQSource,
    NoiseEstimate,
    NoiseEstimatorProtocol,
    Resettable,
    SegmentationDecision,
    SegmenterProtocol,
    SegmentState,
    SegmentTransition,
    StreamInfo,
)


class DummyResettable:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


class DummySource:
    def __init__(self, stream_info: StreamInfo) -> None:
        self._stream_info = stream_info
        self.blocks_read = 0

    @property
    def stream_info(self) -> StreamInfo:
        return self._stream_info

    def read_block(self) -> IQBlock | None:
        self.blocks_read += 1
        return None

    def reset(self) -> None:
        self.blocks_read = 0


class DummyNoiseEstimator:
    def update(self, block: IQBlock) -> NoiseEstimate:
        return NoiseEstimate(
            stream_id=block.stream.stream_id,
            block_sequence_number=block.sequence_number,
            noise_power_dbfs=-70.0,
        )

    def reset(self) -> None:
        pass


class DummyDetector:
    def process(self, block: IQBlock, noise_estimate: NoiseEstimate) -> DetectionMeasurement:
        return DetectionMeasurement(
            stream_id=block.stream.stream_id,
            block_sequence_number=block.sequence_number,
            start_sample=block.start_sample,
            end_sample=block.end_sample,
            signal_power_dbfs=-30.0,
            noise_power_dbfs=noise_estimate.noise_power_dbfs,
            snr_estimate_db=40.0,
            on_threshold_dbfs=-50.0,
            off_threshold_dbfs=-55.0,
            above_on_threshold=True,
            below_off_threshold=False,
        )

    def reset(self) -> None:
        pass


class DummySegmenter:
    def update(self, measurement: DetectionMeasurement) -> SegmentationDecision:
        return SegmentationDecision(
            state=SegmentState.ACTIVE,
            transition=SegmentTransition.START,
            block_sequence_number=measurement.block_sequence_number,
            boundary_sample=measurement.start_sample,
        )

    def reset(self) -> None:
        pass


def test_protocol_conformance() -> None:
    """Verify runtime conformance of implementations against protocols."""
    stream = StreamInfo("stream-01", "synthetic", 1e6)

    resettable = DummyResettable()
    assert isinstance(resettable, Resettable)

    source = DummySource(stream)
    assert isinstance(source, IQSource)
    assert isinstance(source, Resettable)

    estimator = DummyNoiseEstimator()
    assert isinstance(estimator, NoiseEstimatorProtocol)
    assert isinstance(estimator, Resettable)

    detector = DummyDetector()
    assert isinstance(detector, DetectorProtocol)
    assert isinstance(detector, Resettable)

    segmenter = DummySegmenter()
    assert isinstance(segmenter, SegmenterProtocol)
    assert isinstance(segmenter, Resettable)
