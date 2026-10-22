# Acquisition-loss diagnosis and implementation plan

Baseline: active `signal_processing/`, 2.4.0rc1, Windows 11 / i5-1235U.
Evidence: `output/pipeline_audit_20261019/` and
`PIPELINE_END_TO_END_AUDIT_RESULTS.md`. Baseline source: 1,431 tests pass;
acceptance: 165 pass; installed: 180 pass, 116 module hashes match. These overlap.

## Established failures and specific causes to address

1. **Large input disables optimized chronological processing.** In
   `pipeline.py`, full-band batching is restricted to whole blocks <=4,096
   samples. A 16,384-sample transport block falls all the way back to 512 tiny
   hop calls at N=64/H=32. Measured p50 service is 48.57 ms against 8.192 ms
   arrivals; 50,562,816/60,000,000 samples overflow the 16-block queue. Repair:
   split numerical preparation into <=4,096-sample, hop-aligned chunks, preserving
   per-hop causal adaptation, ring commits, faults and completed-output ownership.
   Do not increase queue capacity or precommit the transport suffix.
2. **LTE decoding/search exceeds its service budget.** Low-SNR LTE consumed
   14.2814% at 2 MS/s; source-only replay consumed 99.9727% without queue loss.
   Profile real received IQ before changing algorithms. The exact 64-start-state
   tail-biting decoder repeatedly allocates NumPy metrics/temporary arrays in
   Python for each port/quarter hypothesis; native compilation is a candidate,
   not permission to substitute approximate Viterbi or fewer hypotheses.
3. **Feature throughput and dispatch.** One feature process causes 18.24 s of
   backpressure in the 1 MS/s adaptive test; four reduce that to 0.167 s, but
   acquisition still loses 14.49%. At 2 MS/s, child arithmetic CPU was only
   4.95 s versus 105.52 s summed overlapping RPC residence. Inspect dispatcher
   wake/ordering and measure before attributing residence to computation.
4. **Scheduling jitter still exists independently.** Source-only replay expires
   one 16,384-sample block after 10.98 ms lateness. Processing fixes cannot prove
   Windows hard-real-time behavior. Keep expiry and queue loss separate.

## Numba and threading finding

Numba 0.67.0 / llvmlite 0.49.0 are installed in Python 3.14.7, but the production
tree has **no Numba kernels, prange, thread masking or dynamic Numba scheduling**.
The existing concurrency is bounded Python threads plus persistent spawn
processes. It is not Numba multithreading.

Native kernels must retain float64/complex128 computation, exact tie behavior,
finite validation and a reference path. No fast-math or cumulative-subtraction
window powers. Independent hypotheses can be parallelized; temporal adaptation,
segmentation and ownership cannot. Benchmark serial versus bounded parallel
Numba rather than assuming more threads helps these small jobs. Avoid nested
12-thread teams in every feature process and do not change the application's
global multiprocessing start method or BLAS configuration.

Relevant primary references:

- [Numba parallel loops and race constraints](https://numba.readthedocs.io/en/stable/user/parallel.html).
- [Numba threading layers/masking](https://numba.readthedocs.io/en/stable/user/threading-layer.html).
- [Installed Numba release compatibility](https://numba.readthedocs.io/en/stable/release/0.67.0-notes.html).

`parallel=True`/`prange` supplies parallel numerical work, not automatic
queue-aware thread-count scaling. TBB dynamic chunk scheduling is a distinct
facility and must not be claimed merely because Numba is installed.

## Implementation and verification order

1. Save pre-edit production snapshots and measured CPU profiles, separately from
   unperturbed scheduled timings. Profile direct generic processing and LTE on
   the actual encoded file, with no lost-input shortcut.
2. Repair bounded large-block dispatch, with ordinary/extreme-power, adaptive,
   origins, gaps/EOF, exact pre-trigger and mid-block typed-fault comparisons.
3. Implement native acceleration only in measured pure kernels, preserving the
   exact reference and supported optional dependency behavior. Initialize JIT
   before measured production; report readiness cost separately. Test cached
   code, reference equivalence, thread bounds and installed execution.
4. Repair measured dispatcher defects if demonstrated; retain required external
   callbacks outside locks, bounded admission/result reservations, generation
   semantics and timeout/retry cleanup.
5. Repeat the same 30-s scheduled workloads sequentially, retaining all failure
   logs. Check expiry, queue loss, overload, unique terminal identities, lag and
   recovery separately. A faster unpaced loop is not a pass.
6. Run the combined source/acceptance/numerical and installed gates after edits.
   No frozen/lossless claim unless actual required gates pass. Unresolved host
   scheduling, throughput or LTE/feature integration limits remain explicit.

## Implemented changes

- Large-block bounded dispatch is implemented in `runtime/pipeline.py`.
  Profiling 32 actual 16,384-sample blocks reduced `_process_detection_chunk`
  calls from 16,384 to 647; profiled pipeline time fell from 1.708 to 0.151 s.
  This is attribution under profiling, not a scheduled acceptance result.
- Exact BCH is optionally Numba-compiled in `lte/_native.py`, warmed in the
  broadcast receiver constructor. A destination-major one-thread kernel avoids
  strided trellis work. The 500-call median was 188.05 microseconds versus
  1,365.95 for the reference. Multi-thread options remain available, but were
  slower for this small kernel; no automatic queue-aware NumPy/Numba scaling is
  claimed. Bit decisions match the original decoder, including quantized ties.
- Blind search transforms only actual requested repetition regions. The old
  whole-history FFT remains selectable, and live search calls compare against
  it and direct dot/energy oracles at original tolerances in regression tests.
- Default feature-result policy reconstruction uses a 32-entry full-content
  immutable configuration cache. Edited records are still rejected. No signal
  history/vector/result cache was introduced.
- Windows owned-pipe completion uses a bounded single-writer completion token
  only when architecture/alignment and pipe-capacity prerequisites hold. It
  removes repeated readiness syscalls, not protocol/context validation. Other
  platforms retain the polling fallback; death, timeout, reset and required-sink
  accounting remain intact.
- The existing isolated scheduling benchmark now accepts explicit Fs/block
  parameters without changing the 1-MS/s compatibility defaults. This measures
  the existing production capture-process API at 2 MS/s; it is not a replacement
  source or a new easier workload. Completion-order delivery is an explicit
  pre-existing option for independently identified feature jobs.

## Measured remaining bottlenecks and next implementation work

The fixes improve numerical preparation and BCH/search cost but do not resolve
all loss. The event-dense 2-MS/s workload generates about 488 capture opportunities
per second. Feature child arithmetic, parent-side contract reconstruction and
dispatch, and acquisition/capture contend for CPU/GIL/scheduling capacity.
Four fixed processes do not make parent Python object work parallel. The direct
path currently sheds feature jobs; the loss-accounted isolated path waits for
delivery capacity and eventually overflows its bounded upstream queue instead.
Neither is a successful complete pipeline.

The next substantial repair must reduce **whole feature-service cost and parent
coordination** together: profile the updated path, then batch bounded immutable
transactions/configuration transport and/or move coordination with analysis
across a persistent process boundary, with a token/lease and terminal-outcome
protocol. A compiled feature-core is another measured candidate. These are not
implemented or certified by the BCH kernel. Any such change must keep existing
custom-callback compatibility, all 56 validity-bearing values, bounded input and
output reservations, G06 provenance, reset/gap ownership and shutdown recovery.
Do not simply increase queues or add Python threads around the same GIL work.

LTE still repeatedly reacquires after queue-loss gaps; native decoding and
smaller FFT regions reduce but do not eliminate service cost. The new long-run
results below are input-loss measurements, not a new low-SNR Pd qualification.
Further work needs actual per-stage steady/search timing under the updated
kernel, bounded batched hypothesis processing and efficient correlation/state
reuse proven against the retained reference. Do not suppress valid hypotheses,
weaken CRC/ambiguity checks or change thresholds to manufacture throughput.

Windows deadline expiry remains separately visible. No hard-real-time guarantee
or hardware-acquisition claim follows from these software changes.

## Open-source comparison

GNU Radio's native thread-per-block executor runs work iterations and waits on
input/output progress; it is not a separate Python transaction for every small
sample hop. Its buffers maintain bounded storage, write offsets and reader
ownership. The applicable lesson is to amortize numerical work while preserving
our causal decision horizons—not to copy its scheduler or assume threads alone
remove Python costs. [Executor source](https://github.com/gnuradio/gnuradio/blob/main/gnuradio-runtime/lib/tpb_thread_body.cc),
[buffer source](https://github.com/gnuradio/gnuradio/blob/main/gnuradio-runtime/lib/buffer.cc).

UHD's receive example uses a reusable sample buffer, and its manual explicitly
requires regular receive service to avoid device overrun. This supports keeping
reception independently scheduled and bounded; disabling the synthetic expiry
counter is not evidence hardware would be lossless. UHD is an architectural
reference here, **not** a Pluto driver implementation.
[UHD streaming contract](https://files.ettus.com/manual/page_stream.html).

These references informed bounded block work, ownership and native-computation
choices. They do not certify this package's measured throughput. Final actual
results: [repair report](ACQUISITION_LOSS_REPAIR_RESULTS_20261019.md).
