# PlutoSDR acquisition

Target: **1,920,000 complex samples/second**, adaptive and LTE as **separate modes**.
Every acquired IQ sample enters preprocessing; generic feature analysis is
disabled. LTE implements FDD normal-CP central-six-RB synchronization and
CRC-verified PBCH/BCH/MIB decoding. PDSCH traffic decoding is outside this scope.

## Installation and use

Install ADI's native libiio 0.x driver first. The Python extra does not install
the native DLL or USB driver. Then:

```powershell
py -3.14 -m pip install -e ".[acceleration,pluto]"
$env:OPENBLAS_NUM_THREADS='1'
py -3.14 -m signal_processing.runtime.pluto_cli lte --uri ip:192.168.2.1 --center-frequency 2140000000 --gain-db 30 --seconds 600 > pluto-lte.jsonl
py -3.14 -m signal_processing.runtime.pluto_cli adaptive --uri ip:192.168.2.1 --center-frequency 2140000000 --gain-db 30 --initial-noise-power 0.0001 --seconds 600 > pluto-adaptive.jsonl
```

Replace the example URI, frequency and quiet power. A supported USB IIO URI can
also be used. The installed entry point is `rf-pluto`. Standard Pluto frequency
limits are enforced; firmware is not modified to extend them.

Adaptive quiet power must be independently measured with **the same manual gain,
bandwidth and normalized IQ scale**. ADC values are divided by 2048, so quiet
power measured in raw ADC counts must be divided by 2048². Manual gain in both
modes prevents AGC changes invalidating adaptive calibration. Its readback is
recorded in source metadata.

The CLI emits LTE observations/MIBs or adaptive capture metadata, then a summary.
Use LTE's `--require-mib` when testing an expected cell; it returns a nonzero
exit status if no CRC-valid MIB is decoded, even if sample acquisition succeeds.
Python `run(..., result_sink=callback)` receives each complete adaptive capture;
`capture.event.iq` holds the exact normalized input slice. Raw IQ arrays are not
serialized into JSON. The sink's work and flush costs are included in latency.

## Receive path

`PlutoSourceFactory` configures the device inside the independent spawn source
process. pyadi-iio configures AD936x FIR decimation for the low native rate, and
the adapter rejects any sample-rate readback other than 1,920,000. DSP FFT/JIT
initialization and device configuration precede `runner.begin()`. No receive
buffer exists, and no startup IQ is discarded, before acquisition begins.

One libiio buffer is reused continuously with one I/Q pair enabled. Defaults:
16,384 complex samples/block, four kernel buffers (~34 ms), sixteen bounded host
slots (~137 ms). These bounds do not relax the 100 ms maximum host latency gate
or 50 ms tail p99 gate. Acquisition proceeds independently of DSP; host queue
overflow is counted and fails acceptance.

Declared little-endian signed 12-in-16 samples, including channel bit shifts,
convert directly to normalized complex64. Immutable positioned input goes to
the existing adaptive or LTE pipeline. Host sample coordinates count returned
samples; they are not presented as device sequence counters.

Stale DMA overflow status is cleared immediately before buffer creation. The
sticky receive overflow flag is checked after every refill. Overflow, short
read, or transport error (including timeout) ends the run and reports unknown
loss. An uncertain partial buffer is never retried or treated as EOF. Cancellation
interrupts libiio refill; cleanup waits for the reader before releasing its buffer.

Only receive hardware is configured. No transmission, loopback, firmware change
or host power-plan change is performed. Live DSP uses scoped Windows performance
scheduling and blocking waits, without a busy SMT helper. Optional
`--dsp-scheduling mmcss` retains the multimedia scheduling experiment; it has not
passed sustained qualification and is not the live default.

## Validation limits and later hardware test

**No physical PlutoSDR was available. Hardware testing is pending, as requested.**
Simulated-driver tests cover conversion, buffer reuse, paused startup, cancellation,
readback rejection, overflow/timeout failure, and actual spawned acquisition
feeding adaptive capture and genuine LTE decoding. They cannot measure USB or
prove RF continuity.

The initial adapter suite passed 36 tests; the complete source/check suite passed
1,910 tests, and 744 isolated installed-wheel tests passed. These checks do not
establish real-time acceptance. The latest 600-second
LTE no-cell replay still failed with 1,081,344 host queue-dropped samples and
217.408 ms maximum latency against a 100 ms gate. Consistent lossless 1.92 MS/s
operation therefore remains unachieved on this host. See the full
[optimization report](REALTIME_1920_OPTIMIZATION_20261020.md) for retained results.

`acquisition_passed` requires the requested finite interval to be fully processed
with balanced closed accounting, zero host queue drops, zero reported DMA
overflow/transport failure and no incomplete adaptive captures. `passed` also
requires host latency acceptance. Duration rounds up to a whole block; that
entire final block is processed.

Standard Pluto IQ has no per-sample hardware sequence counter or RF timestamp.
The adapter always reports `hardware_qualified: false`: success means no
**observed** acquisition loss. Latency starts at successful libiio refill and
includes DSP, actual sink publication/flush and cleanup. Device/kernel buffering
and RF integration dwell are outside this host latency measurement. Each summary
states these limits explicitly.

Later, test a sustained independent known receive sequence that exposes missing
or repeated samples across buffer boundaries. Use the intended Pluto, firmware,
native driver, transport, laptop and result sink. Exercise quiet/no-cell search,
LTE acquisition/lock loss and adaptive quiet/active transitions. Verify separately
that an intentionally induced receive overrun actually trips the DMA flag.
File replay and source tests cannot substitute for that hardware evidence.

## Primary references

- [ADI driver installation](https://analogdevicesinc.github.io/documentation/solutions/platforms/pluto/get-started/drivers.html)
- [AD936x native-rate/FIR configuration](https://github.com/analogdevicesinc/pyadi-iio/blob/main/adi/ad936x.py)
- [ADI DMA overflow verification](https://github.com/analogdevicesinc/pyadi-iio/blob/main/test/dma_tests.py)
- [libiio buffer/cancellation API](https://github.com/analogdevicesinc/libiio/blob/v0.26/bindings/python/iio.py)
