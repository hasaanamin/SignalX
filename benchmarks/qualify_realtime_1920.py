"""Strict 1.92 MS/s qualification; no producer-expiry exclusions or saved answers."""

import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path


class _ByteWindow:
    def __init__(self,data):
        self.data,self.closed = data,False

    def __len__(self):
        return len(self.data)

    def __getitem__(self,key):
        return self.data[key]

    def close(self):
        self.data,self.closed = b'',True


class CaptureByteOracle:
    """A <=1 MiB received-byte view; validation never pins a whole recording."""

    def __init__(self, path):
        self.file = open(path,'rb',buffering=1024*1024,
            opener=lambda name,flags: os.open(name,flags | getattr(os,'O_SEQUENTIAL',0)))
        self.size = Path(path).stat().st_size
        if self.size < 8 or self.size % 8:
            self.file.close()
            raise ValueError("Recording must contain whole complex64 samples")
        self.mapping = None
        self.offset = self.end = 0

    def matches(self, event):
        begin, end = event.event_start_sample * 8, event.event_end_sample * 8
        if not 0 <= begin < end <= self.size:
            return False
        if self.mapping is None or begin < self.offset or end > self.end:
            if self.mapping is not None:
                self.mapping.close()
            self.offset = begin // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
            length = min(self.size - self.offset, max(1024 * 1024, end - self.offset))
            self.file.seek(self.offset)
            payload = self.file.read(length)
            if len(payload) != length:
                raise ValueError('Recording changed during independent byte verification')
            self.mapping = _ByteWindow(payload)
            self.end = self.offset + length
        return event.iq.tobytes() == self.mapping[begin - self.offset : end - self.offset]

    def close(self):
        if self.mapping is not None:
            self.mapping.close()
        self.file.close()


def prepare(path, seconds):
    from benchmarks.process_source import _ScheduledIQ

    total = round(seconds * 1_920_000)
    if total < 1:
        raise ValueError("Fixture must contain samples")
    generator = _ScheduledIQ(16384)
    digest = hashlib.sha256()
    with path.open("xb") as output:
        for start in range(0, total, 16384):
            payload = generator.generate(start, min(16384, total - start)).tobytes()
            output.write(payload)
            digest.update(payload)
    return dict(
        samples=total,
        sample_rate_hz=1_920_000.0,
        sha256=digest.hexdigest(),
        scope="Offline seed-730 noise with a 256-sample burst every 4096 samples; no timing claim",
    )


def adaptive(path, block_size=16384, pin_primary_host=False, queue_capacity=128,priority_class="high",active=False):
    import numpy as np

    from signal_processing.runtime.adaptive_cli import run

    oracle = CaptureByteOracle(path)
    period,on_samples = (327680,262144) if active else (4096,256)
    seen = np.zeros((oracle.size // 8 + period-1) // period, bool)
    failures = []
    raw_checked = 0
    extra_captures = 0
    previous_trigger = -1
    next_progress = 60 * 1_920_000 if oracle.size >= 120 * 1_920_000 * 8 else 0

    def observe(capture):
        nonlocal raw_checked, extra_captures, previous_trigger, next_progress
        event = capture.event
        if not oracle.matches(event):
            if len(failures) < 16:
                failures.append("Capture IQ differs from positioned source bytes")
        if capture.features is not None or capture.analysis_result is not None:
            if len(failures) < 16:
                failures.append("Unexpected feature work on capture path")
        if event.trigger_sample <= previous_trigger:
            if len(failures) < 16:
                failures.append("Capture triggers are duplicated or out of order")
        previous_trigger = event.trigger_sample
        # The adaptive detector can legitimately trigger on random noise.
        # Report those extra events; they are not acquisition sample losses.
        first = max(0, (event.detected_start_sample - on_samples) // period + 1)
        last = min(len(seen) - 1, (event.detected_end_sample - 1) // period)
        newly_matched = False
        for burst in range(first, last + 1):
            newly_matched |= not seen[burst]
            seen[burst] = True
        if not newly_matched:
            extra_captures += 1
        raw_checked += 1
        if next_progress and event.event_end_sample >= next_progress:
            print(
                json.dumps(
                    dict(
                        adaptive_received_horizon=event.event_end_sample,
                        captures_checked=raw_checked,
                        byte_validation_failures=len(failures),
                    )
                ),
                flush=True,
            )
            next_progress += 60 * 1_920_000

    try:
        result = run(
            path,
            initial_noise_power=1e-4,
            block_size=block_size,
            capture_sink=observe,
            processing_cpus=[0, 1] if pin_primary_host else None,
            acquisition_cpus=[2, 3] if pin_primary_host else None,
            queue_capacity=queue_capacity,
            priority_class=priority_class,
        )
    finally:
        oracle.close()
    result.update(
        raw_captures_checked=raw_checked,
        expected_bursts=len(seen),
        captured_bursts=int(seen.sum()),
        extra_noise_captures=extra_captures,
        capture_validation_failures=failures,
        fixture_period_samples=period,
        fixture_on_samples=on_samples,
        scope="Scheduled file acquisition + adaptive preprocessing + exact raw capture validation; features disabled",
    )
    result['latency_passed'] = bool(result['latency_ms_percentiles']
        and result['latency_ms_percentiles'][2] <= 50 and result['maximum_latency_ms'] < 100
        and result['eof_tail_latency_ms'] is not None and result['eof_tail_latency_ms'] < 100)
    result["passed"] = bool(
        result["scheduled_lossless_passed"]
        and result['latency_passed']
        and np.all(seen)
        and not failures
        and not result["captures_with_input_loss"]
    )
    return result


def lte(path, cadence,priority_class="high"):
    from benchmarks.audit_pipeline_runtime import lte as audit

    from signal_processing.runtime._performance import performance_scope,realtime_scope
    from signal_processing.runtime._core_reservation import core_reservation
    with performance_scope() as scheduling, core_reservation(scheduling['preferred_core_cpus']) as reservation, realtime_scope(False) as realtime:
        result = audit(
            path,
            "low_snr",
            sample_rate_hz=1_920_000.0,
            sync_backend="numba",
            source_backend="process",
            decode_cadence=cadence,
            priority_class=priority_class,
            dsp_wait='spin',
            core_owner=reservation,
            progress_seconds=60 if path.stat().st_size >= 120 * 1_920_000 * 8 else 0,
        )
    result['performance_qos'] = scheduling
    result['realtime_scheduling'] = realtime
    result['smt_reservation'] = reservation
    # The qualifying gate includes *all* scheduled samples, including source
    # expiry. Block cadence still requires genuine verified decoding per BCH
    # block; every_frame additionally requires each frame's actual CRC result.
    result["strict_1920_passed"] = bool(
        result["acquisition_passed"]
        and result["realtime_decoding_passed"]
        and result.get('eof_tail_latency_ms') is not None
        and result['eof_tail_latency_ms'] < 100
        and (cadence != "every_frame" or result["protocol_all_frames_passed"])
    )
    result["legacy_every_frame_passed"] = result["passed"]
    result["passed"] = result["strict_1920_passed"]
    return result


def prepare_no_cell(path,seconds,tone=False,active=False):
    import numpy as np
    total = round(seconds*1_920_000)
    if not 0 < total <= 600*1_920_000:
        raise ValueError('Noise fixture duration must be positive and at most 600 seconds')
    rng = np.random.default_rng(23483)
    digest = hashlib.sha256()
    with path.open('xb') as output:
        for start in range(0,total,16384):
            count = min(16384,total-start)
            if tone:
                iq = np.exp(2j*np.pi*100000*np.arange(start,start+count)/1_920_000.)
            else:
                iq = rng.normal(size=count)+1j*rng.normal(size=count)
                if active:
                    iq *= np.sqrt(5e-5)
                    iq[np.arange(start,start+count)%327680 < 262144] += .2+.1j
            payload = iq.astype('<c8').tobytes()
            output.write(payload)
            digest.update(payload)
    return dict(path=str(path),samples=total,bytes=total*8,sample_rate_hz=1_920_000,
                seed=23483,sha256=digest.hexdigest(),tone_hz=100000 if tone else None,
                active_period_samples=327680 if active else None,
                active_on_samples=262144 if active else None,
                scope='Offline received-IQ fixture; no receiver state or supplied answers')


def no_cell(path,log,priority_class="high"):
    import subprocess
    import sys
    import time
    command = [sys.executable,'-m','signal_processing.lte.cli','--input',str(path.resolve()),
        '--sample-rate','1920000','--profile','low_snr','--scheduled','--enforce-acquisition',
        '--decode-mib','--decode-cadence','bch_block','--max-latency-ms','100',
        '--priority-class',priority_class]
    started = time.monotonic()
    with log.open('x',encoding='utf8') as output:
        process = subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT)
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(f'No-cell qualification active: {time.monotonic()-started:.0f} wall seconds including initialization',flush=True)
    result = {}
    with log.open(encoding='utf8') as output:
        for line in output:
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if item.get('type') == 'summary':
                result = item
    result.update(command=command,exit_code=code,log=str(log))
    result['passed'] = bool(code == 0 and result.get('acquisition_passed')
        and result.get('latency_passed') and result.get('observations') == 0
        and result.get('decoded_mibs') == 0
        and result.get('latency_ms_p99') is not None and result['latency_ms_p99'] <= 50)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare_adaptive", "prepare_lte", "prepare_no_cell", "prepare_tone", "prepare_adaptive_active", "adaptive", "adaptive_active", "lte", "no_cell", "tone"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--block-size", type=int, default=16384)
    parser.add_argument("--queue-capacity", type=int, default=128)
    parser.add_argument(
        "--decode-cadence", choices=("every_frame", "bch_block"), default="bch_block"
    )
    parser.add_argument(
        "--pin-primary-host",
        action="store_true",
        help="Explicit i5-1235U placement: DSP 0/1, source 2/3",
    )
    parser.add_argument("--priority-class",choices=("normal","above-normal","high"),default="high")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == "prepare_adaptive":
        result = prepare(args.input, args.seconds)
    elif args.mode in ('prepare_no_cell','prepare_tone','prepare_adaptive_active'):
        result = prepare_no_cell(args.input,args.seconds,tone=args.mode == 'prepare_tone',active=args.mode == 'prepare_adaptive_active')
    elif args.mode == "prepare_lte":
        from benchmarks.audit_pipeline_runtime import prepare as prepare_lte

        result = prepare_lte(
            args.input, args.seconds, sample_rate_hz=1_920_000.0, max_seconds=600.0
        )
    elif args.mode in ("adaptive","adaptive_active"):
        result = adaptive(args.input, args.block_size, args.pin_primary_host, args.queue_capacity,args.priority_class,active=args.mode == 'adaptive_active')
    elif args.mode in ('no_cell','tone'):
        result = no_cell(args.input,args.output.with_suffix('.cli.log'),args.priority_class)
    else:
        result = lte(args.input, args.decode_cadence,args.priority_class)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "passed",
                    "strict_1920_passed",
                    "processed_samples",
                    "consumed_samples",
                    "acquisition_passed",
                    "maximum_latency_ms",
                    "max_latency_ms",
                )
                if k in result
            }
        )
    )
    return (
        0
        if args.mode.startswith('prepare_')
        else int(not result.get("strict_1920_passed", result.get("passed", False)))
    )


if __name__ == "__main__":
    raise SystemExit(main())
