# Adaptive and LTE processing at 1.92 MS/s

This change targets preprocessing of every received IQ sample, exact adaptive
event capture, and real LTE synchronization/PBCH/MIB decoding. Full generic
feature extraction is excluded from the acquisition path. The tested LTE scope
is FDD, normal CP and the central six resource blocks; LTE data-channel decoding
is separate work. A receive-only Pluto adapter is now implemented for separate
adaptive/LTE modes; see [Pluto acquisition](PLUTO_ACQUISITION.md). No device was
available, and the user requested that its actual hardware testing happen later.

## Implementation

### Current scope and remaining qualification, 21 October

The user confirmed PlutoSDR and **separate operating modes**, then requested an
adapter for later hardware testing because no device is available now. The
receive-only adapter, independent source-process integration, live CLI and setup
instructions are in [Pluto acquisition](PLUTO_ACQUISITION.md). No hardware-lossless
claim is made. It verifies actual rate readback, monitors the sticky RX DMA
overflow flag, normalizes signed ADC IQ directly into complex64 and fails on
unknown transport loss. It does not transmit or enable loopback. Its bounded
buffer and cleanup tests are separate from actual hardware qualification.

The software optimizations below retain the original mathematical thresholds,
sample coordinates and real decoding. Earlier 600-second adaptive/LTE/no-cell
observations passed, but subsequent repetitions exposed failures. **The current
candidate is not a universal lossless real-time certificate.** The existing
spin/SMT controls remain experimental; the new live adapter uses blocking waits
with scoped process/thread priority and no busy SMT helper. MMCSS is optional.

A later 30-second reserved-core transition run retained all 57,600,000 samples
and all 400 expected BCH codewords, but failed latency: 79.566-ms tail p99 and
125.727-ms maximum (`reserved_smoke/transition_30.json`). A no-cell smoke passed
with 18.718-ms tail p99 and 26.448-ms maximum. These do not supersede sustained
acceptance. EOF now includes actual acquisition/helper cleanup before reporting.

Per-block traces disproved a proposed explanation that the full 27-filter bank
persisted in quiet gaps. Screening was already active throughout those gaps;
the speculative rearm change was removed. The stronger LTE/noise/LTE parity
check against an unscreened receiver was retained. Frequency counters in an
instrumented reserved-core run showed approximately 0.95–1.32 GHz during its
slow initial stretch and approximately 2.0–2.08 GHz later. Screened noise DSP
time changed from roughly 7.39 to 4.17 ms/block. This is diagnostic correlation,
not proof of the sole cause of every stall.

Temporary documented Windows AC performance-mode requests were accepted as a
user preference but the effective mode stayed Best Power Efficiency. All original
AC/DC, actual and effective mode values were verified restored. No power-plan
change is part of production code. Alternative FFT layouts and a split bank
were benchmarked; the latter improved its isolated bank but slowed the complete
pipeline, so neither was selected.

The most recent 30-second diagnostic uses the existing MMCSS option with blocking
waits and shared SMT: all 57,600,000 samples, zero source/queue loss, 400 decoded
BCH codewords, 11.693-ms publication tail p99 and 30.288-ms maximum
(`mmcss_transition_counter_trace.log`). It includes instrumentation and is not a
sustained acceptance substitute. The fresh 600-second no-cell run with the same
MMCSS/blocking/shared-SMT choices **failed**: zero source expiry, 1,081,344 host
queue-dropped samples, 12.211-ms retained-tail p99, 217.408-ms whole-run maximum
and 81.473-ms EOF cleanup (`mmcss_block_no_cell_600.cli.log`). It processed
1,150,918,656 of 1,152,000,000 input samples. The low tail p99 does not erase the
earlier backlog or acquisition losses. The test ran without concurrent builds,
tests, profilers or CPU performance samplers. MMCSS is therefore not selected
as the live adapter's default. Consistent sustained lossless LTE blind-search
throughput remains unresolved on this host; the original throughput objective
has not been fully achieved. Final source/installed-package checks are separate.

The final source/check suite passed **1,910 tests**, zero failures/errors/skips,
in 263.86 seconds (`pluto_final_source_tests.xml`). Its single warning is the
intentional complex64-overflow demonstration in the arithmetic safety test.
The 36 Pluto tests separately passed in 12.19 seconds (`pluto_source_tests.xml`).
These are functional, lifecycle, numerical and integration checks, not a real-time
or hardware pass. The rebuilt RC wheel contains **129 production Python modules**,
all byte-identical between source, wheel and isolated installation. Its SHA-256
is `a99873d39a167df8b6986aec5036db815a6df0b36e005861a2f6eb738a93a4cf`;
the manifest and packaging ledger are under `pluto_final_candidate/`.
The isolated installed-wheel suite passed **744 tests**, zero failures/errors/
skips, in 231.32 seconds (`pluto_final_installed_tests.xml`), including the Pluto
adapter tests and actual spawn boundaries. Production module origins and byte
parity were verified against the isolated installation. This RC is functionally
verified, with the sustained real-time failure above still unresolved.

### DSP and runtime changes

- Adaptive batching now commits the confirmed quiet prefix after an event ends,
  including multiple causal noise updates, and stops before the next ON window.
  Each update retains its own pre-update threshold. It does not train on future
  or active IQ. Reference processing remains available with
  `fixed_quiet_batch=False`; unsupported configurations retain that path.
- `POSITIVE_SNR_IID_1920` has a distinct native-rate identity and correctly uses
  1,920,000 Hz for adaptation time constants. The old 1 MS/s identity is unchanged.
  The synthetic benchmark now passes its actual rate to the estimator too.
- Already validated received sub-blocks avoid repeated scalar construction;
  ring reads freeze detached bytes without scanning previously validated IQ
  again. SignalEvent still validates/freezes the final capture once. Public
  input validation, byte immutability and chronological ring bounds remain.
- Native LTE processing batches each received PSS/SSS FFT pair and fuses channel
  smoothing, alternating half-frame association, all 336 SSS hypotheses and
  normalization in a pre-warmed serial Numba kernel. All CFO/PSS hypotheses,
  confirmation rules and real PBCH/BCH/CRC processing remain. No fast math or
  predicted MIBs are used. `sync_backend='auto'` now selects the native backend
  when installed, with the explicit NumPy/reference implementation retained.
- The new `rf-adaptive` / `signal_processing.runtime.adaptive_cli` entry point
  owns a bounded process source and a single DSP/capture owner. It does not
  submit feature jobs or serialize captures into another DSP interpreter.
  Optional explicit CPU placement is restored at shutdown. A capture sink's
  actual runtime is included in latency.
- Acquisition child import-graph collection finishes before readiness. Automatic
  garbage collection stays enabled. No receiver sample clock advances during
  compilation, initialization or this collection.
- A capped adaptive signal's above-ON rearm run is committed in one bounded
  prefix. Every window still contributes its own power; the prefix stops before
  an OFF/training decision or reference-expiry boundary. The ordinary estimator
  freeze, received-age and final rearm state are preserved exactly.
- Continuous LTE blind search uses receiver-owned double-precision overlap-save
  FFT plans and bounded power history. Positive dyadic energy sums avoid the
  cancellation of running subtraction. Tiny windows after a loud region are
  repaired with direct correlations, preserving zero and large dynamic ranges.
- A conservative seven-direction bound per PSS family screens redundant CFO
  correlations; every potentially qualifying hypothesis is evaluated with the
  original full template. Dense input switches to a prepared full 27-filter bank,
  with a bounded rebuild from retained received IQ. Fresh lock or a proved
  subthreshold region restores the cheaper screen bank. Both
  banks and their scratch are allocated before acquisition. Portable FFTW/SciPy
  paths use 512-point tiles. The supported Windows x64 FFTW wheel uses a warmed
  native double-precision row loop with 1024-point tiles: one filter's inverse
  scratch stays active at a time, and ring wrapping is split outside the sample
  loop. C plan addresses are runtime arguments and machine code is never cached
  with addresses from another process. Plan creation/destruction share the
  reentrant planner lock, and each receiver owns its aligned arrays and plans.
  A subprocess regression forces collection of an unreachable plan owner while
  holding that lock: finalization must finish without deadlocking. Native-rate
  channel preparation also writes admitted complex64 IQ directly into owned
  double-precision history, avoiding a redundant transport-sized conversion;
  the public channel API retains its double-precision output. Neither
  screening nor fallback changes thresholds, hypotheses or confirmation rules.
- A tracking miss rebuilds only the missing received correlation history, then
  searches every configured chunk until fresh confirmation. There is no search
  backoff or intentional reacquisition pause.
- Acquisition helpers request scoped Windows HighQoS and, when acquisition
  enforcement is enabled, High process priority for their own DSP and source
  processes. Parent priority/affinity and its original power-throttling flags
  are restored on exit, including failure. No host power plan, timer resolution
  or global garbage-collection policy is changed. Plain receiver construction
  does not change process policy. `--cpu-qos system` and `--priority-class normal`
  provide explicit opt-outs; timing qualification identifies its scheduling.
- Windows hybrid machines place the DSP caller on one hardware thread of the
  last allowed fast physical core, using CPU-set EfficiencyClass and CoreIndex.
  On the measured host this is logical CPU 2; source reads/copies now use logical
  CPU 0 on the other performance core. The process affinity is unchanged; caller thread
  affinity is restored independently for every scope, including concurrent
  owners. Explicit thread HighQoS prevents automatic background inference during
  paced idle intervals; both thread and process power flags are restored.
  Explicit CPU settings disable this automatic selection. Homogeneous and
  multi-group systems keep existing placement. On this i5-1235U, the same native
  blind-search workload measured 2.82 MS/s on a performance core versus 1.50 MS/s
  on an efficiency core; unconstrained migration can therefore violate the target
  despite adequate mean throughput. These CPU probes are not acquisition passes.
- Streaming source callers own scoped Windows MMCSS registrations with relative
  critical priority. DSP uses its process priority by default; explicit
  `--dsp-scheduling mmcss` requests an MMCSS registration with relative high
  priority. A native trace observed quota-driven priority demotion, so DSP is not
  placed under that CPU quota by default. Registration can replace
  caller affinity with the task profile: the helper now preserves the configured
  group/mask across registration. Native checks cover this interaction and
  nested success/failure cleanup. Source policy is returned through the real
  process readiness/completion protocol, rather than inferred from DSP policy.
  On supported hybrid CPUs, the source caller selects one hardware thread on
  another fast physical core, falling back to one core of the other class if
  necessary. This avoids DSP SMT siblings and source migration. Only caller thread
  masks change; process masks are preserved. Earlier one-core placement trials
  and their failures remain in the evidence directory.
- Conservative candidate selection clears only the preceding selected score
  entries and visits positions only in branches that survive their maximum
  bound. A shared conservative bound first screens positions once per PSS
  family; each surviving branch then applies its original bound at those
  positions. Independent checks require exactly the original eligible indices
  and order. A sparse maximum is recorded during exact full-template evaluation,
  avoiding a second dense score scan for subthreshold input. This changes no
  threshold, integration, filter, score or candidate ordering. Changing query
  widths and dense-to-empty transitions have independent selection checks.
- Six cached nonnegative basis-power groups replace rereading and summing the
  same 21 components for every integration repeat. The addition order is
  unchanged. Tests compare grouped and original projections bit for bit through
  wraparound, 1/2/4/8 repeats, zero and large dynamic ranges.
- A conservative tonal-input bound uses the exact identity
  `x[k+1] = z*x[k] + e[k]` and Cauchy-Schwarz on each actual filter's remainder.
  FFT work is skipped only when the bound proves EVERY received window's PSS
  score is below threshold, which also bounds their nonnegative incoherent
  integration. Otherwise normal search runs. Independent full-window sums,
  all 27 injected qualifying PSS branches, and a tone-matched filter test verify
  that this is not a rule discarding tones regardless of their filter response.
- LTE JSON publication uses one ordered writer with capacity for 64 outstanding
  batches, including its active batch. It cannot silently drop outputs. Failure
  or saturation fails the run; EOF drains actual output before acceptance.
  Publication latency includes queue wait, serialization, stdout write and
  flush. Separate DSP latency is reported too. Correct decoding and source
  accounting alone cannot pass a failed publication/latency gate.
- Acquisition replay now defaults to bounded sequential buffered reads, with
  the Windows sequential-access hint. This avoids the long-run paging cost of
  mapping large recordings. `--file-backend mmap` retains the bounded mapping
  path; direct `RecordedFileIQSource` construction retains its legacy mmap
  default. Both paths cover byte order, exact positions, immutable output,
  rollover, reset and cleanup. EOF acceptance waits for real source cleanup.
  Explicit source CPU masks are installed in the child before its factory and
  scheduling scopes, so their restoration remains valid too.
  Scheduled-file construction also warms one read and rewinds before readiness.
  A virtual-clock regression makes the first read cost 100 ms and requires all
  bytes, sample zero, the original schedule and zero expiry afterward. Failed
  warmup closes the file. No acquisition clock advances during this setup.

The new LTE reduction uses [Numba's documented native/nogil execution](https://numba.readthedocs.io/en/stable/user/jit.html#nogil)
and [SciPy's axis-based FFT interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html).
These mechanisms do not themselves certify a real-time deadline.
FFTW plans follow the [pyFFTW planning/execution API](https://pyfftw.readthedocs.io/en/latest/source/pyfftw/pyfftw.html);
the scoped Windows policy uses the documented
[SetProcessInformation interface](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation).
Automatic placement uses [Windows CPU-set topology](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_cpu_set_information),
which identifies faster classes without assuming Intel CPU numbers.
Thread policy uses [SetThreadInformation](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadinformation).
Streaming registration follows the [Windows MMCSS API](https://learn.microsoft.com/en-us/windows/win32/procthread/multimedia-class-scheduler-service).
Native FFT execution and lifecycle follow the documented [FFTW 1-D API](https://www.fftw.org/fftw3_doc/Complex-One_002dDimensional-DFTs.html)
and [planner thread-safety rules](https://www.fftw.org/fftw3_doc/Thread-safety.html).

## Transport and acceptance

The transport block is 16,384 complex64 samples (8.533 ms at the actual rate).
Adaptive detector windows remain N=256/H=64, the ring remains 4,096 samples,
and the adaptation statistic remains M=1,024. Transport batching introduces
availability latency; it does not skip samples or detector windows.

The adaptive acquisition command defaults to 128 slots: 17,826,816 bytes
including envelope space, or 1.0923 seconds of IQ coverage. This is an explicit
increase from the historical 16-slot/136.5-ms experiment. A finite buffer absorbs
scheduling pauses only when sustained processing can recover. Peak occupancy,
latency, actual elapsed time and EOF drain are recorded. This is not a sub-10-ms
deadline claim. `--queue-capacity 16` reproduces the smaller pool.
The adaptive reference's existing staleness limit remains enforced: continuous
activity without an accepted quiet update eventually requires recalibration.
Performance batching does not turn a stale reference into a valid one. The
long-active acquisition fixture alternates 262,144 active samples with 65,536
quiet samples, exercising cap/rearm processing without violating that contract.

The source's original **10-ms expiry budget remains unchanged**. Passing requires
all scheduled samples, zero source expiry, zero acquisition queue loss, balanced
accounting, zero failed/cancelled/unresolved input and closed cleanup. No
source-expiry exclusion is used for this report's strict acquisition results.

LTE retains its 16-slot input pool and 50-ms p99 / 100-ms maximum latency gate.
The explicit `bch_block` mode requires a genuine verified MIB in every 40-ms
codeword. `every_frame` is a separate stronger protocol gate, requiring each
redundant 10-ms frame's CRC result; its failures remain visible.

`benchmarks.qualify_realtime_1920` independently compares every adaptive capture
byte with positioned source bytes through a bounded 1-MiB buffered byte window
(or one larger capture). This independent verifier remains synchronous and its
read/comparison cost remains inside measured capture latency. It checks
ordered triggers, feature absence and burst coverage. Extra legitimate noise
captures are reported separately from acquisition loss. Expected burst matching
uses detected-interval overlap, so a noise candidate preceding a real burst does
not falsely make a processed burst disappear from the evaluator.

## Reproduction

Install the existing acceleration extra for the native LTE backend. Run timing
commands serially without a test suite, profiler, trace or editing workload.
Choose new output paths: qualification refuses to overwrite observations.

```powershell
$env:OPENBLAS_NUM_THREADS='1'
py -3.14 -m benchmarks.qualify_realtime_1920 prepare_adaptive --seconds 30 --input output/adaptive_native.cf32 --output output/adaptive_fixture.json
py -3.14 -m benchmarks.qualify_realtime_1920 adaptive --input output/adaptive_native.cf32 --output output/adaptive_result.json
py -3.14 -m benchmarks.audit_pipeline_runtime prepare --seconds 30 --sample-rate 1920000 --input output/lte_native.cf32 --output output/lte_fixture.json
py -3.14 -m benchmarks.qualify_realtime_1920 lte --input output/lte_native.cf32 --output output/lte_result.json
py -3.14 -m pytest -q tests checks
```

Use `--decode-cadence every_frame` for the additional LTE gate. Independent
sequence/coding fixtures are supplied only to the evaluator, never the receiver.
Recorded tests cannot establish USB/device/driver losslessness on an unconnected
SDR. Check a real device's rate and loss counters when its adapter is integrated.

## Evidence

Fresh observations, including unsuccessful experiments, are retained under
`output/realtime_1920_20261020/`. Earlier project evidence is unchanged. Final
isolated timing and regression results are recorded here after execution.

### Buffered-input sustained observations (source on efficiency cores)

All three 600-second runs below process 1,152,000,000 complex IQ samples at the
actual 1,920,000 Hz rate, with zero source expiry, zero queue loss and balanced
completed accounting. Artifacts are in `output/realtime_1920_20261020/buffered_long`.

| Mode / result file | Tail p99, ms | Whole-run maximum, ms | EOF tail, ms | Result |
|---|---:|---:|---:|---|
| Adaptive / `adaptive_600_buffered_oracle.json` | 8.645 | 38.287 | 4.058 | PASS |
| No cell / `no_cell_600.json` | 13.258 | 37.031 | 7.144 | PASS |
| LTE / `lte_600.json` | 9.960 | 37.590 | 16.649 | PASS |

The percentile is explicitly a retained tail: at most 65,536 adaptive blocks
and 4,096 LTE/no-cell blocks. The maximum covers the entire run. These are
block-availability-to-completion delays, including the adaptive capture sink
or LTE publication; RF integration/confirmation dwell is separate. Acceptance
remains tail p99 <=50 ms, global maximum <100 ms and EOF <100 ms. This is not a
sub-10-ms claim or a guarantee against arbitrary OS/device stalls.

Adaptive checks all 281,724 delivered captures byte for byte and captures all
281,250 expected bursts, with 474 additional noise captures, no validation
errors and no feature jobs. LTE verifies all 15,000 expected 40-ms BCH codewords
on genuine -5-dB, 12,345-Hz-CFO, PCI-42, two-port IQ. No-cell input produces no
false cell observations or MIBs. This LTE gate uses `bch_block`: the separately
reported every-10-ms-frame gate is false, with 45,000 redundant-frame MIBs not
produced. No result is substituted for an actual CRC-verified decode.

The first buffered-input adaptive run (`adaptive_600.json`) acquired every sample
and checked every capture correctly, but its 117.859-ms maximum failed timing.
The retry above changed only the independent verifier's I/O from bounded mmap
to bounded buffered bytes; every byte comparison still runs inside the timed
sink, with identical deadlines, pool size, fixture and correctness gates.
The failed observation remains retained. Earlier LTE/no-cell long runs precede
the final explicit-CPU-mask lifecycle repair; that repair affects the opt-in
explicit-placement path, whereas these runs use automatic thread placement.

### Buffered-input source and distribution checks

The source run before the physical-source-core change passes **1,845 tests**, with zero failures, errors or skips
(`final_buffered_verification/source_tests.xml`). Its single warning is the
intentional complex64-overflow demonstration in the arithmetic safety test.
The fresh wheel contains all 126 production Python modules, byte-identical to
both the source tree and isolated installed tree, with a valid ZIP CRC.
Artifacts and the per-module SHA-256 manifest are in
`final_buffered_verification/candidate`.

Wheel: `signal_processing-2.4.0rc1-py3-none-any.whl`  
SHA-256: `4ba2bcf536d09d5eb8e3713668c81da40ba05c25e3a07bd63f33237e4e8e17fd`.

The installed tree passes 679 tests (`installed_tests_verified.xml`). The first
installed run had 678 passes and one harness failure: pytest derived the spawn
helper's module name under `output.*`, which its child could not import. The
verification runner now fixes its root to the copied support tree and refuses
to overwrite reports. The unchanged installed wheel then passes all 679 tests.

### Repeated qualification exposed remaining scheduling failures

`final_buffered_timing` retains all 18 serial 30-second runs: 14 pass and four
fail. `transition_1` and `transition_2` each expire one 16,384-sample source block,
despite verifying all 400 expected BCH blocks. `lte_2` expires 49,152 samples
with a 10.529-ms maximum source lateness, despite complete BCH coverage.
`no_cell_3` is lossless but its 69.046-ms tail p99 fails the 50-ms limit
(92.125-ms whole-run maximum). These failures prevent a final qualification of
that scheduling policy; the earlier 600-second passes do not erase them.

A phase trace records file reads, timed waits and DSP work separately. The
efficiency-core-pool transition diagnostic has 6.771-ms wakeup overshoot,
88.024-ms tail p99 and 104.889-ms maximum, with no garbage collections or slow
output writes. Pinning the source to the other physical performance core in
the otherwise identical diagnostic yields 14.565-ms tail p99, 26.493-ms maximum,
no expiry and no wakeup overshoot above 2 ms. Corresponding no-cell and LTE
diagnostics also acquire every sample: maxima 21.141 and 35.545 ms respectively,
with 750 actual verified MIBs in the LTE case. These instrumented probes justify
the new placement; they are not substituted for subsequent timing qualification.
The production change retains the same pool sizes, 10-ms source expiry,
thresholds, search cadence and all correctness/latency gates. Focused source,
placement, restoration and CLI checks pass 106 cases after the change.

The subsequent `physical_source_long/no_cell_600.json` still fails: zero source
expiry, 32,768 queue-dropped samples, 101.780-ms tail p99 and 149.594-ms maximum.
Source placement alone is insufficient. A fresh CPU profile attributes about
60% of no-cell time to the FFT bank and 20% to screening. Interleaved FFT layouts
are slower; a split bank helps isolated FFT timing but slows the complete
pipeline, so neither experiment is selected. Continuous CPU probes reach
3.8–4.1 MS/s while paced blocking runs can still accumulate backlog.

### Explicit DSP wait strategy

`ProcessSourceRunner.get(spin=True)` keeps the reserved DSP caller active between
blocks. It uses the same ready semaphore and one ownership-transfer path,
checks the same timeout/EOF/failure state, and releases the lifecycle lock
between polls so another thread can shut it down. No data or source-clock
policy changes. The public default remains blocking. LTE enforced process
acquisition with performance QoS, and adaptive performance mode, select spin;
both commands expose `--dsp-wait block` to reduce idle CPU use. Spin can occupy
one core continuously; this is an explicit CPU/latency tradeoff, with no host
power-plan or timer-resolution change. It does not prove a specific underlying
frequency-governor mechanism caused earlier stalls.

A full-broadcast diagnostic (`no_cell_spin_broadcast_trace`) processes every
57,600,000 samples, with zero expiry/drop, 8.224-ms p99 and 10.317-ms maximum,
and no garbage collections. The earlier `no_cell_spin_trace` exercised the
synchronizer alone; its empty DSP trace is not used as full-pipeline evidence.
Focused checks pass 126 cases after adding explicit spin/block modes, including
real process death, zero/finite timeout, paused readiness, concurrent shutdown,
byte delivery, failed sinks and unchanged impossible-deadline enforcement.
Fresh sustained and repeated timing runs remain required for this mode.

The first sustained spin run (`spin_long/no_cell_600.json`) processes all
1,152,000,000 samples with zero expiry and zero queue loss. Its 8.562-ms tail
p99 passes, but its 122.941-ms maximum fails the unchanged 100-ms gate; adaptive
and LTE were therefore not started by that sequential runner. This is a
lossless acquisition result, not a full timing pass.

DSP performance scope now also requests Windows `THREAD_PRIORITY_HIGHEST`
relative priority and restores its exact original value independently per
caller. Under High process priority this is base priority 15, within the
ordinary priority range; no RealTime process class is selected. See
[Microsoft's priority table](https://learn.microsoft.com/en-us/windows/win32/procthread/scheduling-priorities).
Nested, concurrent and failed-scope tests observe the real native priority and
restoration. Source MMCSS scheduling remains separate. The LTE summary now
retains the maximum DSP-latency block and at most 32 slow-block records with
processing wall time, process CPU time, time before DSP and source lateness.
These records distinguish backlog/compute from waiting if another peak occurs;
they do not exclude any delay from acceptance. Focused checks pass 85 cases
after the priority/telemetry change.

The next long run (`highest_spin_long/no_cell_600.json`) still fails: zero
source expiry, 81,920 queue-dropped samples, 8.057-ms tail p99 and 173.227-ms
maximum publication delay. Its maximum DSP block waits 163.363 ms before
processing and takes 9.837 ms in DSP. The priority change alone does not solve
the remaining backlog; this failure is retained.

### Physical-core reservation

A controlled, owned low-priority vector workload on the DSP core's other SMT
thread reduces continuous LTE CPU throughput from 3.569–4.242 MS/s to
2.124–2.539 MS/s. An owned PAUSE loop on that sibling restores 3.561–3.631 MS/s
in the same randomized experiment (`bench_smt_contention.log`). This reproduces
a plausible loss of headroom, but does not identify the cause of every earlier
host stall. No other application's affinity or priority is changed.

Performance spin mode now defaults to `--dsp-smt reserve` on an automatically
selected Windows x64 SMT pair. DSP uses the first hardware thread; an owned
native PAUSE loop occupies its sibling at the same non-RealTime relative
priority. On this host they are CPUs 2 and 3; source I/O remains on CPU 0.
This reserves one physical DSP core and can report two logical CPUs busy.
`--dsp-smt shared`, blocking wait, system policy or explicit CPU placement
disables this automatic reservation. Kernel interrupts and higher/equal-priority
system activity retain their scheduling rights; this is not hard OS isolation.

The helper compiles before acquisition, using process-local
[llvmlite native execution](https://llvmlite.readthedocs.io/en/latest/user-guide/binding/examples.html)
and an [atomic stop flag](https://llvm.org/docs/LangRef.html#load-instruction).
It releases the GIL and never touches IQ. Every compiled engine has distinct
symbols, and an active context holds its engine/function pointers even after
the process-local cache is cleared. No executable addresses are disk-cached.
Stopping and joining are bounded and precede EOF acceptance. Adaptive and LTE
qualification now measure EOF after owned resource cleanup, including this
thread. Earlier EOF observations used their recorded, earlier cleanup scope.

Native start/stop, cache-clear lifetime, failed setup, owner exceptions,
idempotent early close and restoration checks pass. The focused integration
run passes 100 cases, followed by 73 cleanup/CLI cases after completing EOF
measurement. Fresh timing qualification remains required; earlier short probes
and passing source/wheel counts are not final acceptance of this policy.

### Earlier observations and unsuccessful experiments

The 18 serial 30-second runs in `qualified_timing` passed before the final
affinity/sparse-selection changes, covering adaptive bursts, LTE decoding,
continuous noise, cell on/off transitions, long adaptive activity and a tone.
These are intermediate results, not final sustained acceptance.
The subsequent 600-second noise runs failed: `qualified_long/no_cell_600.json`
records 344,064 expired and 688,128 queue-dropped samples, while
`affinity_fixed_long/no_cell_600.json` records zero expiry but 1,212,416
queue-dropped samples and 226.234-ms maximum latency. Neither passes. A
600-second diagnostic recorded no garbage collections in the source or DSP;
GC freeze/disable is therefore not justified or applied. Native stage probes
identified the MMCSS affinity reset, and live queries confirmed its correction;
that fix alone was insufficient for sustained acceptance. All failures remain
available alongside later runs.
The subsequent sparse-selection run, `sparse_long/no_cell_600.json`, also failed:
zero source expiry, 229,376 queue-dropped samples, 9.386-ms tail p99 and
197.984-ms maximum latency. The complete regression suite before source-core
separation passed 1,827 checks (`sparse_verification/source_tests.xml`).
The one-fast-core separation trial `separate_long/no_cell_600.json` failed with
114,688 expired and 1,015,808 queue-dropped samples (212.833-ms maximum latency).
The following static DSP priority trial `priority_long/no_cell_600.json` had no
source expiry but dropped 557,056 queued samples (226.204-ms maximum latency).
Neither passes sustained acceptance. Serial native FFT geometry was selected
only after randomized comparisons against direct double-precision convolution:
the chosen row layout measured 0.507-ms median / 0.855-ms p99 bank time, versus
0.755 / 1.727 ms for the former Python-dispatched layout in that paired probe.
A full CPU probe then measured 3.982 MS/s; this is not an acquisition pass.
The first actual 30-second native-row no-cell qualification passed all
57,600,000 samples, zero expiry/drop, 25.520-ms tail p99 and 44.937-ms maximum
latency (`native_row_smoke/no_cell_30.json`). Numerical/backend/policy checks
passed 210 cases; a separate native plan-lifecycle check passed. Final sustained
and installed-artifact checks remain separate gates.

The 600-second native-row run (`native_row_long/no_cell_600.json`) still failed:
zero expiry, 32,768 queue-dropped samples and 147.004-ms maximum. A later native
history diagnostic (`native_direct_trace_600_v2.json`) recorded zero expiry,
278,528 queue-dropped samples and 171.647-ms maximum. It observed no source or
DSP garbage collections; CPU-intensive slow stretches accumulated backlog.
The family-screen run (`family_long/no_cell_600.json`) then failed with 409,600
expired samples, 49,152 queue-dropped samples and 160.327-ms maximum. Buffered
input was selected afterward. These failed runs are not excluded from the
history or reclassified as passes.

The first source run after the initial changes reported 85 passes and a spawned
capture startup timeout. That exact failing case passed in the subsequent
17-case run. The updated 22-case native-rate suite and 44-case buffer/immutability
run passed. These are overlapping intermediate checks, not a final suite count.

A subsequent combined source/check run was interrupted after a confirmed hang
in the pre-existing process-death test's cleanup. Its captured stack is
`source_stall_stack.log`: multiprocessing.Event.set was waiting for an
acknowledgement from the deliberately killed waiter. Cleanup now signals that
gate only while its child remains alive; the actual death/publication/accounting
assertions are unchanged. The complete suite is rerun after this harness repair.

Initial larger-block trials exposed genuine queue losses and source expiry.
The original whole-recording mapping also caused avoidable measurement memory
pressure. Those failed observations are retained and are not timing evidence
for the final bounded-oracle configuration. A successful small run is not a
universal lossless or frozen-release claim.
