"""Isolated, single-thread solver comparison on frozen full-vocabulary targets.

Run prepare separately (it may load torch/GPU). Each measured worker imports
only the discrete solver stack; wall time begins at each announced operation.
Timeouts and RSS limits preserve every already-completed measurement.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import time
import traceback

import numpy as np

from tri.domain.music import verify_music
from tri.inference.exact import Budget
from tri.runtime import atomic_json, exclusive_run, read_jsonl


BASELINES = ('paired', 'template', 've', 've_aligned', 've_minfill', 'product_chain')


def failure_status(error, backend):
    # The original dense VE keeps singleton axes, which can exceed NumPy's
    # supported ndarray rank even when the table has few entries. This is an
    # implementation capacity limit, neither a timeout nor zero probability.
    if backend == 've' and isinstance(error, ValueError) and re.fullmatch(
            r'maximum supported dimension for an ndarray is \d+, found \d+', str(error)):
        return 'implementation_limit'
    return {'BudgetExceeded': 'internal_budget', 'UnsupportedSpec': 'unsupported',
            'MemoryError': 'allocation_failure', 'ZeroMass': 'zero_mass'}.get(
                type(error).__name__, 'unexpected_error')


def make_solver(spec, logq, backend, budget):
    if backend in ('ve_aligned', 've_minfill'):
        from tri.inference.ordered_ve import OrderedMusicVE
        return OrderedMusicVE(spec, logq, budget, order='aligned' if backend == 've_aligned' else 'minfill')
    if backend == 'product_chain':
        from tri.inference.product_chain import ProductChainMusicInference
        return ProductChainMusicInference(spec, logq, budget)
    if backend == 'product_reuse':
        from tri.inference.product_cached import CachedProductChainMusicInference
        return CachedProductChainMusicInference(spec, logq, budget)
    if backend in ('paired_reuse', 'paired_sparse'):
        from tri.inference.paired_optimized import ReusedPairedInference
        return ReusedPairedInference(spec, logq, budget, sparse=(backend == 'paired_sparse'))
    from tri.inference.music_backends import make_music_engine
    return make_music_engine(spec, logq, backend, budget)


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            raise ValueError('NaN cannot be serialized as a solver result')
        return float(value) if math.isfinite(value) else ('-inf' if value < 0 else 'inf')
    if isinstance(value, np.integer):
        return int(value)
    return value


def emit(event):
    print(json.dumps(json_value(event), ensure_ascii=False, allow_nan=False), flush=True)


def _rss_mib():
    # Linux ru_maxrss can retain a pre-exec parent's high-water mark. VmHWM
    # belongs to this exec's address space, so torch imported by a test/launcher
    # cannot falsely mark a small isolated solver as over its memory budget.
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1]) / 1024
    raise RuntimeError('Linux VmHWM is required for isolated memory accounting')


def _check_draw(draw, spec, q, logz):
    tokens = tuple(draw.assignment[f'y{i}'] for i in range(spec.length))
    checked = verify_music(tokens, spec)
    if not checked.valid:
        raise AssertionError(f'Invalid joint sample: {checked.violations}')
    log_weight = checked.soft_score + sum(q[i, tokens[i]] for i in range(spec.length) if i not in spec.observed)
    error = abs(float(draw.log_probability) - (float(log_weight) - logz))
    clamped_error = abs(float(draw.log_clamped_partition) - float(log_weight))
    if not math.isfinite(error) or error > 1e-8:
        raise AssertionError(f'Full-sample log probability differs from original token weight by {error}')
    if not math.isfinite(clamped_error) or clamped_error > 1e-8:
        raise AssertionError(f'Full-sample clamped partition differs from original token weight by {clamped_error}')
    return {'valid': True, 'log_probability_error': error,
            'log_clamped_partition_error': clamped_error, 'tokens': list(tokens)}


def worker(payload):
    from tri.evaluation.solver_cases import load_case
    case = load_case(Path(payload['path']))
    budget = Budget(**payload['budget'])
    spec, q = case.spec, case.logq
    backend = payload['backend']
    rng = np.random.default_rng(payload['sample_seed'])
    variables = [f'y{i}' for i in range(spec.length)]
    emit({'event': 'ready', 'baseline_rss_mib': _rss_mib(), 'pid': os.getpid()})
    current = None

    def operation(name, function):
        nonlocal current
        gc.collect()
        current = name
        emit({'event': 'start', 'operation': name})
        t = time.perf_counter()
        result = function()
        seconds = time.perf_counter() - t
        emit({'event': 'measurement', 'operation': name, 'seconds': seconds,
              'peak_rss_mib': _rss_mib(), **result})
        return result

    try:
        def partition():
            t = time.perf_counter()
            engine = make_solver(spec, q, backend, budget)
            construct = time.perf_counter() - t
            t = time.perf_counter()
            z = engine.log_partition()
            return {'log_z': z, 'construct_seconds': construct,
                    'query_seconds': time.perf_counter() - t, 'stats': engine.last_stats}

        base = operation('cold_partition', partition)
        z = float(base['log_z'])
        if np.isneginf(z):
            emit({'event': 'done', 'status': 'zero_mass', 'peak_rss_mib': _rss_mib()})
            return
        if not math.isfinite(z):
            raise AssertionError('Invalid partition')
        engine = None

        def cold_sample():
            nonlocal engine
            t = time.perf_counter()
            engine = make_solver(spec, q, backend, budget)
            construct = time.perf_counter() - t
            t = time.perf_counter()
            draw = engine.sample_batch(variables, rng)
            query_seconds = time.perf_counter() - t
            checked = _check_draw(draw, spec, q, z)
            return {'construct_seconds': construct, 'query_seconds': query_seconds,
                    'draw': checked, 'stats': engine.last_stats}

        operation('cold_full_sample', cold_sample)

        def repeats():
            durations, errors = [], []
            for _ in range(payload['warm_samples']):
                t = time.perf_counter()
                draw = engine.sample_batch(variables, rng)
                durations.append(time.perf_counter() - t)
                errors.append(_check_draw(draw, spec, q, z)['log_probability_error'])
            return {'samples': len(durations), 'sample_seconds': durations,
                    'maximum_log_probability_error': max(errors), 'stats': engine.last_stats}

        operation('warm_full_samples', repeats)
        # Keep clamp timing independent of which random full sequences a
        # backend happened to cache while sampling. Prime only E={}.
        engine = None
        gc.collect()

        def query_setup():
            nonlocal engine
            engine = make_solver(spec, q, backend, budget)
            value = engine.log_partition()
            if abs(value-z) > 1e-8:
                raise AssertionError('Fresh query engine changed the frozen target')
            return {'log_z': value, 'stats': engine.last_stats}

        operation('query_setup', query_setup)
        # Fixed probes depend only on the frozen case, never on a solver draw.
        probes = case.metadata.get('query_evidence', [])
        if isinstance(probes, dict):
            joint = probes
            probes = [{name: value} for name, value in sorted(joint.items())]
            if len(joint) > 1:
                probes.append(joint)
        if not probes:
            free = [i for i in range(spec.length) if i not in spec.observed]
            probes = [{f'y{i}': 0} for i in free[:3]]

        def clamps():
            values, durations, stats = [], [], []
            for evidence in probes:
                t = time.perf_counter()
                value = engine.log_partition(evidence)
                if math.isnan(value) or value == math.inf:
                    raise AssertionError('Invalid clamped partition')
                values.append(value)
                durations.append(time.perf_counter() - t)
                stats.append(dict(engine.last_stats))
            return {'evidence': probes, 'log_z': values, 'query_seconds': durations,
                    'probe_stats': stats, 'stats': engine.last_stats}

        operation('clamp_queries', clamps)
        # A repeated identical query is a scalar-cache check, reported separately.
        operation('repeated_identical_clamps', clamps)
        free = [name for name in variables if int(name[1:]) not in spec.observed]
        if free:
            def marginal():
                variable = free[0]
                values = engine.marginal_log_probs(variable)
                from scipy.special import logsumexp
                norm = float(logsumexp(values))
                if not math.isfinite(norm) or abs(norm) > 1e-8:
                    raise AssertionError('Marginal is not normalized')
                return {'variable': variable, 'domain': list(engine.graph.domains[variable]),
                        'log_probs': values, 'stats': engine.last_stats}
            operation('token_marginal', marginal)
        emit({'event': 'done', 'status': 'completed', 'peak_rss_mib': _rss_mib()})
    except Exception as error:
        name = type(error).__name__
        status = failure_status(error, backend)
        emit({'event': 'done', 'status': status, 'failed_operation': current,
              'error_type': name, 'error': str(error), 'peak_rss_mib': _rss_mib(),
              'traceback': traceback.format_exc() if status == 'unexpected_error' else None})


def isolated(payload, *, operation_seconds, rss_limit_mib, cpu):
    import psutil
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    process = subprocess.Popen([sys.executable, '-m', __name__ if __name__ != '__main__' else
                                'tri.evaluation.solver_study', '--worker', json.dumps(payload)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, env=env)
    if cpu is not None:
        os.sched_setaffinity(process.pid, {cpu})
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    selector.register(process.stderr, selectors.EVENT_READ)
    started = time.monotonic()
    deadline = started + 60  # Import startup is recorded, outside query budget.
    current = 'startup'
    rss_failure_operation = None
    buffer = b''
    stderr_tail = b''
    row = {'backend': payload['backend'], 'status': 'running', 'measurements': {},
           'sample_seed': payload['sample_seed'], 'peak_rss_mib': 0.0}
    try:
        while True:
            try:
                rss = psutil.Process(process.pid).memory_info().rss / 2**20
                row['peak_rss_mib'] = max(row['peak_rss_mib'], rss)
            except psutil.NoSuchProcess:
                pass
            if row['peak_rss_mib'] > rss_limit_mib:
                row.update(status='rss_limit', failed_operation=rss_failure_operation or current)
                break
            if time.monotonic() > deadline:
                row.update(status='timeout', failed_operation=current,
                           timeout_seconds=60 if current == 'startup' else operation_seconds)
                break
            for key, _ in selector.select(timeout=.01):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                if key.fileobj is process.stderr:
                    stderr_tail = (stderr_tail + chunk)[-4000:]
                    continue
                buffer += chunk
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    event = json.loads(line)
                    kind = event.pop('event')
                    if 'peak_rss_mib' in event:
                        row['peak_rss_mib'] = max(row['peak_rss_mib'], event['peak_rss_mib'])
                        if row['peak_rss_mib'] > rss_limit_mib and rss_failure_operation is None:
                            # One pipe drain may also contain the next start
                            # and done events. Retain the first over-budget
                            # operation before those events advance current.
                            rss_failure_operation = (event.get('operation') or
                                                     event.get('failed_operation') or current)
                    if kind == 'ready':
                        row.update(baseline_rss_mib=event['baseline_rss_mib'], startup_seconds=time.monotonic()-started)
                    elif kind == 'start':
                        current = event['operation']
                        deadline = time.monotonic() + operation_seconds
                    elif kind == 'measurement':
                        row['measurements'][event.pop('operation')] = event
                    elif kind == 'done':
                        row.update(event)
            if row['peak_rss_mib'] > rss_limit_mib:
                row.update(status='rss_limit', failed_operation=rss_failure_operation or current)
            if row['status'] != 'running':
                break
            if process.poll() is not None and not selector.get_map():
                row.update(status='worker_exit', failed_operation=current)
                break
        if process.poll() is None:
            process.terminate()
        _, stderr = process.communicate(timeout=5)
        if stderr_tail or stderr:
            row['stderr'] = (stderr_tail + stderr)[-4000:].decode('utf-8', errors='replace')
        row['elapsed_worker_seconds'] = time.monotonic()-started
        row['incremental_peak_rss_mib'] = max(0, row['peak_rss_mib']-row.get('baseline_rss_mib', 0))
        return row
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def run(manifest_path, output, config, *, resume=False, backend_override=None, case_limit=None):
    manifest = json.loads(Path(manifest_path).read_text())
    entries = manifest['cases']
    if case_limit is not None:
        entries = entries[:case_limit]
    backends = list(backend_override or config['backends'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = {'manifest': str(Path(manifest_path).resolve()), 'cases': [e['case_id'] for e in entries],
                'config': config, 'backends': backends}
    with exclusive_run(output/'study.lock'):
        if (output/'config.json').exists():
            if not resume or json.loads((output/'config.json').read_text()) != snapshot:
                raise ValueError('Existing study requires matching --resume')
        atomic_json(output/'config.json', snapshot)
        rows = read_jsonl(output/'results.jsonl', recover_tail=resume)
        completed = {(r['case_id'], r['backend']) for r in rows}
        expected = {(e['case_id'], b) for e in entries for b in backends}
        if not completed <= expected or len(completed) != len(rows):
            raise ValueError('Unexpected or duplicate previous rows')
        start = time.time()
        cpu = config.get('cpu')
        if cpu is not None and cpu not in os.sched_getaffinity(0):
            raise ValueError('Configured benchmark CPU is unavailable')
        for index, entry in enumerate(entries):
            order = np.random.default_rng(config['seed']+index).permutation(backends)
            for backend in order:
                if (entry['case_id'], backend) in completed:
                    continue
                atomic_json(output/'status.json', {'status': 'running', 'pid': os.getpid(),
                    'started_at': start, 'updated_at': time.time(), 'completed': len(rows),
                    'planned': len(expected), 'case_id': entry['case_id'], 'backend': backend})
                payload = {'path': entry['path'], 'backend': str(backend), 'budget': config['budget'],
                           'sample_seed': config['seed']+index, 'warm_samples': config['warm_samples']}
                row = isolated(payload, operation_seconds=config['operation_seconds'],
                               rss_limit_mib=config['rss_limit_mib'], cpu=cpu)
                row.update(case_id=entry['case_id'], metadata=entry['metadata'])
                with (output/'results.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    stream.flush()
                rows.append(row)
                completed.add((entry['case_id'], backend))
                print(f"{len(rows)}/{len(expected)} {entry['case_id']} {backend}: {row['status']}", flush=True)
                if row['status'] in ('unexpected_error', 'worker_exit', 'allocation_failure'):
                    atomic_json(output/'status.json', {'status': 'failed', 'row': row, 'completed': len(rows), 'planned': len(expected)})
                    raise RuntimeError(f'Unexpected solver failure: {row}')
        errors = compare_rows(rows)
        report = {'status': 'failed' if errors else 'completed', 'rows': len(rows),
                  'cases': len(entries), 'backends': backends, 'comparison_errors': errors,
                  'completed_at': time.time(), 'config': snapshot}
        atomic_json(output/'report.json', report)
        atomic_json(output/'status.json', report)
        if errors:
            raise AssertionError(str(errors))
        return report


def compare_rows(rows, tolerance=1e-8):
    by_case = {}
    for row in rows:
        by_case.setdefault(row['case_id'], []).append(row)
    errors = []
    for case_id, group in by_case.items():
        targets = {}
        signatures = {}
        for row in group:
            for operation in ('cold_partition', 'clamp_queries', 'token_marginal'):
                result = row['measurements'].get(operation)
                if result is None:
                    continue
                values = result['log_probs'] if operation == 'token_marginal' else result['log_z']
                if not isinstance(values, list):
                    values = [values]
                signature = json.dumps({'evidence': result.get('evidence'), 'variable': result.get('variable'),
                                        'domain': result.get('domain'), 'length': len(values)}, sort_keys=True)
                if operation in signatures and signatures[operation] != signature:
                    errors.append({'case_id': case_id, 'query': operation, 'error': 'query identity mismatch'})
                signatures[operation] = signature
                for index, value in enumerate(values):
                    if math.isnan(float(value)) or float(value) == math.inf:
                        errors.append({'case_id': case_id, 'query': operation, 'error': 'invalid numerical result'})
                    targets.setdefault((operation, index), []).append((row['backend'], float(value)))
        for query, values in targets.items():
            reference = values[0][1]
            for backend, value in values[1:]:
                equal = value == reference or (math.isfinite(value) and math.isfinite(reference)
                                               and abs(value-reference) <= tolerance)
                if not equal:
                    errors.append({'case_id': case_id, 'query': query, 'values': values})
                    break
        expectation = group[0]['metadata'].get('expected_support')
        for backend, value in targets.get(('cold_partition', 0), []):
            if expectation in ('feasible', True) and not math.isfinite(value):
                errors.append({'case_id': case_id, 'backend': backend, 'error': 'feasible case returned zero'})
            if expectation in ('infeasible', False) and value != -math.inf:
                errors.append({'case_id': case_id, 'backend': backend, 'error': 'infeasible case returned mass'})
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--config', type=Path, default=Path('configs/solver_study.yaml'))
    parser.add_argument('--out', type=Path, default=Path('runs/solver_study/baselines'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--backends', nargs='+')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(argv)
    if args.worker:
        worker(json.loads(args.worker))
        return
    import yaml
    config = yaml.safe_load(args.config.read_text())
    report = run(args.manifest, args.out, config, resume=args.resume,
                 backend_override=args.backends, case_limit=args.limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
