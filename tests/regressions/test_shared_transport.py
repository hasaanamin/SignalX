import multiprocessing
import queue

import numpy as np
import pytest

from signal_processing.runtime.shared_transport import SharedIQTransport


def _producer(transport):
    for i in range(100):
        transport.put((i*.001,i,10**12+i*16,np.full(16,i,np.complex64).tobytes(),0),timeout=3)
    transport.put(None,timeout=3)


def test_spawn_wrap_exact_bytes_bounded_owned_copies():
    ctx=multiprocessing.get_context('spawn')
    transport=SharedIQTransport(ctx,2,16)
    child=ctx.Process(target=_producer,args=(transport,))
    child.start()
    held=[]
    try:
        for i in range(100):
            record=transport.get(timeout=5)
            assert record[1:3] == (i,10**12+i*16)
            assert record[3] == np.full(16,i,np.complex64).tobytes()
            if i < 2: held.append(record[3])
            assert transport.qsize() <= 2
            transport.task_done()
        assert transport.get(timeout=5) is None
        transport.task_done()
        child.join(5)
        assert child.exitcode == 0 and transport.unfinished_tasks == 0
        assert held[0] == np.zeros(16,np.complex64).tobytes()
        assert held[1] == np.ones(16,np.complex64).tobytes()
    finally:
        if child.is_alive():child.terminate();child.join(2)
        transport.close()


def test_full_empty_invalid_rejection_and_acknowledgement():
    t=SharedIQTransport(multiprocessing.get_context('spawn'),1,16)
    record=(0.,0,0,np.ones(16,np.complex64).tobytes(),0)
    try:
        with pytest.raises(ValueError):t.put((0.,True,0,record[3],0))
        with pytest.raises(queue.Empty):t.get(False)
        t.put_nowait(record)
        with pytest.raises(queue.Full):t.put_nowait(record)
        assert t.get(False)==record
        t.task_done()
        with pytest.raises(ValueError):t.task_done()
    finally:t.close()
