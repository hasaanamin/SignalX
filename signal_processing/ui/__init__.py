"""Optional user interfaces and dataset-driven pipeline harnesses."""

from .hdf5_runner import (
    DatasetSummary,
    EventMatch,
    FrameTruth,
    HarnessConfig,
    PipelineRunResult,
    RunCancelled,
    inspect_hdf5_dataset,
    run_hdf5_pipeline,
)

__all__ = [
    "DatasetSummary",
    "EventMatch",
    "FrameTruth",
    "HarnessConfig",
    "PipelineRunResult",
    "RunCancelled",
    "inspect_hdf5_dataset",
    "run_hdf5_pipeline",
    "LTEStepInspector",
    "LTEStudioApp",
]


def __getattr__(name: str):
    if name == "LTEStepInspector":
        from .lte_pipeline_inspector import LTEStepInspector
        return LTEStepInspector
    if name == "LTEStudioApp":
        from .lte_app import LTEStudioApp
        return LTEStudioApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

