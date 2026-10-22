"""Acquisition-only processes do not import/allocate the entire DSP stack."""
import subprocess
import sys
from pathlib import Path

import signal_processing


def test_source_only_public_api_does_not_load_dsp_in_fresh_interpreter(tmp_path):
    root=Path(signal_processing.__file__).resolve().parent.parent
    code=f'''import sys
sys.path.insert(0,{str(root)!r})
import signal_processing.runtime as runtime
from signal_processing.runtime import SourceRunner,ProcessSourceRunner,ReplayFileFactory
from signal_processing.runtime.source_runner import SourceRunner as Original
import signal_processing.lte
from signal_processing.lte.cli import main
assert SourceRunner is Original
assert 'scipy' not in sys.modules
assert 'signal_processing.features' not in sys.modules
assert 'signal_processing.runtime.pipeline' not in sys.modules
assert all(name in dir(runtime) for name in runtime.__all__)
from signal_processing.runtime import StreamingCapturePipeline,FeatureAnalysisWorker
from signal_processing.runtime.pipeline import StreamingCapturePipeline as Pipeline
assert Pipeline is StreamingCapturePipeline
assert FeatureAnalysisWorker.__module__=='signal_processing.runtime.worker'
'''
    result=subprocess.run([sys.executable,'-I','-c',code],cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr


def test_unknown_runtime_export_remains_an_attribute_error():
    import signal_processing.runtime as runtime
    try: getattr(runtime,'no_such_runtime_export')
    except AttributeError: pass
    else: raise AssertionError('Unknown export silently accepted')
