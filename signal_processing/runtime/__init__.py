"""Streaming runtime API, lazily importing only the requested execution stage.

An acquisition-only spawn must not load the detector, SciPy and all feature
families just to import a source queue or scheduling primitive. Public classes
keep their defining modules/identities; requesting the DSP API loads real DSP.
"""
from importlib import import_module

__all__ = [
    "StreamingCapturePipeline",
    "FeatureAnalysisWorker",
    "WorkerState",
    "FeatureRequestGate", "FeatureRequestSnapshot",
    "SourceRunner", "SourceDelivery", "SourceRunnerSnapshot", "SourceRunnerFailed",
    "ProcessSourceRunner", "ReplayFileFactory",
]

_EXPORTS = {
    'StreamingCapturePipeline':'pipeline',
    'FeatureAnalysisWorker':'worker','WorkerState':'worker',
    'FeatureRequestGate':'feature_requests','FeatureRequestSnapshot':'feature_requests',
    'SourceRunner':'source_runner','SourceDelivery':'source_runner',
    'SourceRunnerSnapshot':'source_runner','SourceRunnerFailed':'source_runner',
    'ProcessSourceRunner':'process_source_runner','ReplayFileFactory':'process_source_runner',
}


def __getattr__(name):
    module=_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value=getattr(import_module(f'{__name__}.{module}'),name)
    globals()[name]=value
    return value


def __dir__():
    return sorted(set(globals())|set(__all__))
