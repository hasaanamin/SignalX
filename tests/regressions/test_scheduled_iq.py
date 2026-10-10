import math
import numpy as np
from benchmarks.process_source import _ScheduledIQ


def test_preallocated_source_is_byte_identical_to_original_generator():
    original=np.random.default_rng(730)
    prepared=_ScheduledIQ(1024)
    retained=[]
    for i,n in enumerate([1024]*20+[1,17,255,832]):
        # Noncontiguous absolute inputs also model already-expired opportunities:
        # a skipped opportunity consumes no RNG draws in either implementation.
        start=i*3072
        expected=(original.normal(size=n)+1j*original.normal(size=n))*math.sqrt(1e-4/2)
        expected[((start+np.arange(n))%4096)<256]+=.1
        actual=prepared.generate(start,n).tobytes()
        assert actual==expected.astype(np.complex64).tobytes()
        retained.append(actual)
    assert retained[0]!=retained[1]
