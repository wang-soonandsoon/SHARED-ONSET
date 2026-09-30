"""Isolated serial solver replay of saved actual decoder application calls.

Each trace/backend/repeat has a fresh Python process and the original tape RNG
states. The fixed order is randomized within each trace/repeat block. A failed
row remains a result: resume skips it and never retries it silently.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

import numpy as np

from tri.evaluation.query_workload import json_value, replay_trace, _process_peak_rss_mib
from tri.runtime import atomic_json, exclusive_run, read_jsonl


BACKENDS = ('product_reuse', 'product_prefix', 'paired_reuse', 'paired_sparse')
STAGES = ('short', 'bar8', 'bar16')
METHODS = ('one_shot_joint', 'tri_direct', 'smc_4')


def _emit(event):
    print(json.dumps(json_value(event), ensure_ascii=False, allow_nan=False), flush=True)


def replay_worker(payload, emit=_emit):
    """Single process entry: emit candidate memory before any validation work."""
    from threadpoolctl import threadpool_limits
    from tri.errors import BudgetExceeded

    cpu = int(payload['cpu'])
    os.sched_setaffinity(0, {cpu})
    phase = 'input_loading'
    partial = {}

    def progress(event):
        nonlocal phase
        phase = event['phase']
        if phase == 'candidate_completed':
            partial.update(event['candidate'])
        emit({'event': 'phase', **event})

    emit({'event': 'ready', 'pid': os.getpid(), 'cpu': cpu,
          'baseline_process_peak_rss_mib': _process_peak_rss_mib()})
    try:
        with threadpool_limits(limits=1):
            result = replay_trace(payload['trace'], payload['backend'], validate=True, phase_callback=progress)
    except Exception as error:
        status = ('internal_budget' if isinstance(error, BudgetExceeded) else
                  'correctness_failure' if isinstance(error, AssertionError) else
                  'allocation_failure' if isinstance(error, MemoryError) else 'unexpected_error')
        result = {**partial, 'status': status, 'failed_phase': phase,
                  'error': {'type': type(error).__name__, 'message': str(error)},
                  'worker_peak_rss_after_failure_mib': _process_peak_rss_mib()}
    result.update(worker_pid=os.getpid(), worker_cpu=cpu, blas_threads=1)
    emit({'event': 'done', 'result': result})
    return result


def isolated_replay(payload, *, timeout_seconds=120., rss_limit_mib=512.):
    """Limit the whole worker (startup, candidate and checker) without retries.

    Process RSS is monitored throughout. Candidate VmHWM is the child-reported
    reading before validation; the parent lifetime peak is a separate field.
    Thus a checker limit failure does not erase already completed candidate
    measurements or reclassify checker memory as candidate solver memory.
    """
    import psutil

    environment = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                   'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    started = time.monotonic()
    process = subprocess.Popen([sys.executable, '-m', 'tri.evaluation.query_replay', '--worker',
                                json.dumps(payload)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=environment, bufsize=0)
    os.sched_setaffinity(process.pid, {int(payload['cpu'])})
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    selector.register(process.stderr, selectors.EVENT_READ)
    buffer = b''
    stderr_tail = b''
    current = 'startup'
    candidate = {}
    result = None
    monitor_peak = 0.
    child = psutil.Process(process.pid)
    try:
        while True:
            try:
                monitor_peak = max(monitor_peak, child.memory_info().rss / 2**20)
            except psutil.NoSuchProcess:
                pass
            # Drain phase events before classifying a limit to keep checker
            # failures separate when the candidate and checker run quickly.
            for key, _ in selector.select(timeout=.01):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                if key.fileobj is process.stderr:
                    stderr_tail = (stderr_tail + chunk)[-8000:]
                    continue
                buffer += chunk
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    event = json.loads(line)
                    if event['event'] == 'ready':
                        current = 'input_loading'
                    elif event['event'] == 'phase':
                        current = event['phase']
                        if current == 'candidate_completed':
                            candidate = event['candidate']
                    elif event['event'] == 'done':
                        result = event['result']
            reported_peak = max(candidate.get('candidate_peak_rss_mib', 0.),
                                (result or {}).get('worker_peak_rss_after_validation_mib', 0.),
                                (result or {}).get('worker_peak_rss_after_failure_mib', 0.))
            monitor_peak = max(monitor_peak, reported_peak)
            if monitor_peak > rss_limit_mib:
                exceeded_phase = ('candidate' if candidate.get('candidate_peak_rss_mib', 0.) > rss_limit_mib
                                  else current)
                result = {**candidate, 'status': 'rss_limit', 'failed_phase': exceeded_phase,
                          'rss_limit_mib': rss_limit_mib}
                break
            if time.monotonic() - started > timeout_seconds:
                result = {**candidate, 'status': 'timeout', 'failed_phase': current,
                          'timeout_seconds': timeout_seconds}
                break
            if process.poll() is not None and not selector.get_map():
                if result is None:
                    result = {**candidate, 'status': 'worker_exit', 'failed_phase': current,
                              'exit_code': process.returncode}
                break
        if process.poll() is None:
            process.terminate()
        try:
            _, tail = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            _, tail = process.communicate()
        if stderr_tail or tail:
            result['stderr'] = (stderr_tail + tail)[-8000:].decode('utf-8', errors='replace')
        result.update(elapsed_worker_seconds=time.monotonic() - started,
                      worker_lifetime_peak_rss_mib=monitor_peak,
                      worker_limit_scope='120-second default wall budget and 512 MiB default RSS apply to entire fresh worker, including independent validation; candidate pre-validation peak is separately recorded.')
        return json_value(result)
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def replay_schedule(trace_root, *, repeats=3, backends=BACKENDS, seed=91423):
    root = Path(trace_root).resolve()
    if repeats < 1 or len(set(backends)) != len(backends) or not set(backends) <= set(BACKENDS):
        raise ValueError('Require positive repeats and distinct registered replay backends')
    if not backends:
        raise ValueError('Replay needs at least one backend')
    jobs = []
    rng = np.random.default_rng(seed)
    for stage in STAGES:
        for method in METHODS:
            trace = root / stage / method / 'trace.json'
            if not trace.exists() or not trace.with_name('q_snapshots.npz').exists():
                raise ValueError(f'Missing saved actual decoder trace: {trace}')
            for repeat in range(repeats):
                for backend in rng.permutation(backends):
                    jobs.append({'job_id': f'{stage}/{method}/{backend}/repeat_{repeat}',
                                 'stage': stage, 'method': method, 'trace': str(trace),
                                 'backend': str(backend), 'repeat': repeat})
    return jobs


def run_replay_suite(trace_root, output, *, repeats=3, backends=BACKENDS, cpu=2,
                     timeout_seconds=120., rss_limit_mib=512., seed=91423, resume=False):
    if cpu not in os.sched_getaffinity(0):
        raise ValueError('Requested replay CPU is unavailable')
    if timeout_seconds <= 0 or rss_limit_mib <= 0:
        raise ValueError('Worker time and RSS limits must be positive')
    jobs = replay_schedule(trace_root, repeats=repeats, backends=backends, seed=seed)
    config = {'trace_root': str(Path(trace_root).resolve()), 'backends': list(backends),
              'repeats': repeats, 'cpu': cpu, 'blas_threads': 1,
              'worker_timeout_seconds': timeout_seconds, 'worker_rss_limit_mib': rss_limit_mib,
              'backend_order_seed': seed, 'jobs': jobs,
              'independent_validation': True, 'budget_source': 'Exact per-engine Budget saved in each trace',
              'repeat_rng_policy': 'Restore each recorded application-call RNG state on every repeat; repeats measure runtime of a fixed workload, not new sample quality.',
              'order_policy': 'Stage/method/repeat blocks; independently randomized backend permutation per block; one fresh serial process per job.',
              'failure_policy': 'Persist every terminal result, continue remaining jobs, never retry recorded failures; matching explicit resume skips all existing rows.',
              'measurement_scope': 'Frozen actual decoder application trajectory, not candidate end-to-end decoding or real user editing logs.'}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(output / 'replay.lock'):
        if (output / 'config.json').exists():
            if not resume or json.loads((output / 'config.json').read_text()) != config:
                raise ValueError('Existing replay suite requires matching explicit --resume')
        elif (output / 'results.jsonl').exists():
            raise ValueError('Refuse results without their replay config')
        atomic_json(output / 'config.json', config)
        rows = read_jsonl(output / 'results.jsonl', recover_tail=resume)
        completed = {row['job_id'] for row in rows}
        if len(completed) != len(rows) or not completed <= {job['job_id'] for job in jobs}:
            raise ValueError('Unexpected or duplicate existing replay jobs')
        start = time.time()
        for job in jobs:
            if job['job_id'] in completed:
                continue
            atomic_json(output / 'status.json', {'status': 'running', 'pid': os.getpid(),
                'started_at': start, 'updated_at': time.time(), 'completed': len(rows),
                'planned': len(jobs), 'active_job': job})
            result = isolated_replay({**job, 'cpu': cpu}, timeout_seconds=timeout_seconds,
                                     rss_limit_mib=rss_limit_mib)
            row = {**job, **result}
            with (output / 'results.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            rows.append(row)
            completed.add(job['job_id'])
            print(f'{len(rows)}/{len(jobs)} {job["job_id"]}: {row["status"]}', flush=True)
        statuses = dict(Counter(row['status'] for row in rows))
        report = {'status': 'completed' if statuses.get('completed', 0) == len(jobs) else 'completed_with_failures',
                  'planned': len(jobs), 'completed': len(rows), 'status_counts': statuses,
                  'updated_at': time.time(), 'elapsed_this_invocation_seconds': time.time() - start,
                  'result_path': str((output / 'results.jsonl').resolve())}
        atomic_json(output / 'status.json', report)
        return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--trace', help='Single-process replay of this saved trace; bypasses isolated suite limits')
    parser.add_argument('--backend', choices=BACKENDS, default='product_prefix')
    parser.add_argument('--trace-root', default='runs/query_workload/actual_decode_v1')
    parser.add_argument('--out', default='runs/decision_study/actual_replay')
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout-seconds', type=float, default=120.)
    parser.add_argument('--rss-limit-mib', type=float, default=512.)
    parser.add_argument('--seed', type=int, default=91423)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.worker:
        replay_worker(json.loads(args.worker))
    elif args.trace:
        destination = Path(args.out)
        if destination.exists():
            raise ValueError('Single replay output must be a new JSON file')
        result = replay_worker({'trace': args.trace, 'backend': args.backend, 'cpu': args.cpu})
        atomic_json(destination, result)
        if result['status'] != 'completed':
            raise SystemExit(1)
    else:
        run_replay_suite(args.trace_root, args.out, repeats=args.repeats, cpu=args.cpu,
                         timeout_seconds=args.timeout_seconds, rss_limit_mib=args.rss_limit_mib,
                         seed=args.seed, resume=args.resume)


if __name__ == '__main__':
    main()
