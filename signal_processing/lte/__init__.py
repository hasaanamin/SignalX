"""LTE API; load DSP only when requested, not in acquisition spawn bootstrap."""
from importlib import import_module

__all__ = ["LTESyncConfig", "LTESyncObservation", "LTESynchronizer",
           "LTEBroadcastReceiver", "LTEMIB", "LTEPBCHObservation", "LTEBroadcastBatch", "PBCHDecoder",
           "PCFICHDecoder", "PCFICHObservation", "PDCCHDecoder", "DCI1AAllocation",
           "TurboDecoder", "SIB1Receiver", "LTESIB1Observation"]

_EXPORTS = dict(
    LTESyncConfig='config', LTESyncObservation='contracts', LTESynchronizer='receiver',
    LTEBroadcastReceiver='broadcast', LTEMIB='mib', LTEPBCHObservation='mib',
    LTEBroadcastBatch='mib', PBCHDecoder='pbch',
    PCFICHDecoder='pcfich', PCFICHObservation='pcfich',
    PDCCHDecoder='pdcch', DCI1AAllocation='pdcch',
    TurboDecoder='turbo', SIB1Receiver='sib', LTESIB1Observation='sib',
)



def __getattr__(name):
    module=_EXPORTS.get(name)
    if module is None: raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value=getattr(import_module(f'{__name__}.{module}'),name)
    globals()[name]=value
    return value


def __dir__():
    return sorted(set(globals())|set(__all__))
