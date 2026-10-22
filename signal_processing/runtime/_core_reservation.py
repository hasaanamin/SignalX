"""Scoped SMT sibling reservation for a dedicated Windows DSP core.

An owned native PAUSE loop keeps ordinary competing work off the sibling while
leaving almost all execution resources for DSP. This is not kernel isolation:
interrupts and higher-priority system work retain their scheduling rights.
"""
from contextlib import contextmanager
from functools import lru_cache
from itertools import count
import sys
import threading

_compile_lock = threading.Lock()
_generations = count()


class Reservation(dict):
    """Serializable diagnostics with an idempotent, bounded early close."""
    _closer = None

    def close(self):
        if self._closer is not None:
            self._closer()


@lru_cache(maxsize=1)
def _native_idle():
    import ctypes
    from llvmlite import binding as llvm
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    machine = llvm.Target.from_default_triple().create_target_machine()
    generation = next(_generations)
    idle_name,stop_name = f'rf_idle_{generation}',f'rf_stop_{generation}'
    module = llvm.parse_assembly('''
        define void @RF_IDLE(ptr %flag) nounwind {
        entry: br label %loop
        loop:
            %value = load atomic i32, ptr %flag acquire, align 4
            %end = icmp ne i32 %value, 0
            br i1 %end, label %exit, label %pause
        pause:
            call void @llvm.x86.sse2.pause()
            br label %loop
        exit: ret void
        }
        define void @RF_STOP(ptr %flag) nounwind {
            store atomic i32 1, ptr %flag seq_cst, align 4
            ret void
        }
        declare void @llvm.x86.sse2.pause()
    '''.replace('RF_IDLE',idle_name).replace('RF_STOP',stop_name))
    module.verify()
    engine = llvm.create_mcjit_compiler(module,machine)
    engine.finalize_object()
    signature = ctypes.CFUNCTYPE(None,ctypes.POINTER(ctypes.c_int32))
    # Keep the engine with its function pointers, including after cache_clear.
    # No executable addresses or machine code are written to a disk cache.
    return engine,signature(engine.get_function_address(idle_name)),signature(engine.get_function_address(stop_name))


def _set_affinity(mask):
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentThread.restype = wintypes.HANDLE
    kernel.SetThreadAffinityMask.argtypes = [wintypes.HANDLE,ctypes.c_size_t]
    kernel.SetThreadAffinityMask.restype = ctypes.c_size_t
    old = kernel.SetThreadAffinityMask(kernel.GetCurrentThread(),mask)
    if not old:
        raise ctypes.WinError(ctypes.get_last_error())
    return old


@contextmanager
def core_reservation(core_cpus,*,enabled=True):
    """Reserve other hardware threads of the selected physical core until exit.

    The first hardware thread is the DSP caller selected by performance_scope.
    Unsupported platforms, disabled policy and cores without SMT create no
    thread. Stop uses an atomic store; joining and restoration are bounded.
    """
    if type(enabled) is not bool:
        raise ValueError('Core reservation selection must be bool')
    if core_cpus is not None and (not isinstance(core_cpus,(tuple,list)) or not core_cpus
            or any(type(cpu) is not int or not 0 <= cpu < 64 for cpu in core_cpus)
            or list(core_cpus) != sorted(set(core_cpus))):
        raise ValueError('Core CPUs must be distinct sorted group-zero CPU indices')
    result = Reservation(requested=enabled,applied=False,cpus=[],reason=None,stopped=True,
                  affinity_restored=True,error=None,worker_policy=None)
    reason = ('disabled' if not enabled else 'unsupported platform' if sys.platform != 'win32'
              else 'no automatic SMT placement' if not core_cpus or len(core_cpus)<2
              else 'unsupported SMT width' if len(core_cpus) != 2 else None)
    if reason is not None:
        result['reason'] = reason
        yield result
        return
    import ctypes
    import platform
    if ctypes.sizeof(ctypes.c_void_p) != 8 or platform.machine().lower() not in ('amd64','x86_64'):
        result['reason'] = 'requires Windows x64'
        yield result
        return
    try:
        with _compile_lock:
            operations = _native_idle()
    except ImportError:
        result['reason'] = 'native acceleration unavailable'
        yield result
        return
    flag = ctypes.c_int32(0)
    ready = threading.Event()
    mask = sum(1 << cpu for cpu in core_cpus[1:])
    result['cpus'] = list(core_cpus[1:])
    def worker():
        old = None
        try:
            old = _set_affinity(mask)
            result['affinity_restored'] = False
            from ._performance import performance_scope
            with performance_scope(prefer_fast_cpus=False) as policy:
                result['worker_policy'] = policy
                ready.set()
                operations[1](ctypes.byref(flag))  # ctypes releases the GIL.
                if flag.value == 0:
                    raise RuntimeError('DSP core reservation stopped before requested')
        except BaseException as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
            ready.set()
        finally:
            if old is not None:
                try:
                    _set_affinity(old)
                    result['affinity_restored'] = True
                except BaseException as exc:
                    result['error'] = f'affinity cleanup: {type(exc).__name__}: {exc}'
    thread = threading.Thread(target=worker,name='rf-core-reservation',daemon=True)
    thread.start()
    result['stopped'] = False
    close_lock = threading.Lock()
    def close_worker():
        with close_lock:
            operations[2](ctypes.byref(flag))
            thread.join(5)
            result['stopped'] = not thread.is_alive()
            if not result['stopped']:
                raise TimeoutError('DSP core reservation did not stop')
            result._closer = None
            if result['error']:
                raise RuntimeError(result['error'])
    result._closer = close_worker
    try:
        if not ready.wait(5):
            raise TimeoutError('DSP core reservation did not become ready')
        if result['error']:
            raise RuntimeError(result['error'])
        result['applied'] = True
        yield result
    finally:
        result.close()
