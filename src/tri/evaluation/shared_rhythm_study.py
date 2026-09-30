"""Serial, isolated exact-inference and rejection-sampling structural pilot.

Every process has a whole-worker deadline and physical RSS limit, including
startup and checks. Completed operations survive a later resource failure.
Sampling-only methods never publish the proposal normalizer as target Z.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
import traceback

import numpy as np

from tri.domain.music import verify_music
from tri.evaluation.solver_cases import load_case
from tri.evaluation.solver_study import _rss_mib, json_value
from tri.inference.exact import Budget
from tri.runtime import atomic_json, exclusive_run, read_jsonl


BACKENDS = ('template_stream', 've_aligned', 'product_prefix', 'onset_reset',
            'onset_rejection_drop', 'onset_rejection_boundary')
SAMPLERS = frozenset(('onset_rejection_drop', 'onset_rejection_boundary'))


def support_reason(metadata, backend):
    if backend not in BACKENDS:
        raise ValueError('Unknown shared-rhythm backend')
    if backend == 'product_prefix' and metadata['R'] != 2:
        return 'Existing standard product_prefix implementation supports exactly two aligned spans; R>2 is unsupported, not a timeout.'
    if backend == 'onset_reset' and metadata['beta'] != 0:
        return 'Base onset-reset exact target requires motion_cost=0; it cannot substitute Z0 or a base sample for the beta>0 target.'
    return None


def make_solver(spec, q, backend, budget, max_proposals, progress_callback=None):
    if backend == 'onset_reset':
        from tri.inference.onset_reset import OnsetResetMusicInference
        return OnsetResetMusicInference(spec, q, budget)
    if backend in SAMPLERS:
        from tri.sampling.onset_rejection import OnsetRejectionSampler
        proposal = 'drop' if backend.endswith('_drop') else 'boundary'
        return OnsetRejectionSampler(spec, q, budget, proposal=proposal, max_proposals=max_proposals,
                                     progress_callback=progress_callback)
    if backend == 've_aligned':
        from tri.inference.ordered_ve import OrderedMusicVE
        return OrderedMusicVE(spec, q, budget, order='aligned')
    if backend == 'template_stream':
        from tri.inference.template_stream import StreamingTemplateMusicInference
        return StreamingTemplateMusicInference(spec, q, budget)
    if backend == 'product_prefix':
        from tri.inference.product_prefix import PrefixCachedProductChainMusicInference
        return PrefixCachedProductChainMusicInference(spec, q, budget)
    raise ValueError('Unknown shared-rhythm backend')


def _failure_status(error):
    return {'BudgetExceeded': 'internal_budget', 'UnsupportedSpec': 'unsupported',
            'MemoryError': 'allocation_failure', 'ZeroMass': 'zero_mass',
            'AssertionError': 'correctness_failure', 'VerificationError': 'correctness_failure'}.get(
                type(error).__name__, 'unexpected_error')


def _sample_complete(engine, backend, rng):
    """Use actual full-path draws without redundant full-assignment queries.

    The legacy template public batch interface recomputes a clamped partition
    even for a full path. Its existing internal full sampler has the same
    enumeration and budget checks, with no unnecessary probability query.
    Product/ordered-VE already directly weight complete public batch draws.
    """
    if backend in SAMPLERS:
        draw = engine.sample_full(rng)
        return tuple(draw.tokens), draw
    if backend == 'onset_reset':
        return tuple(engine.sample_full(rng)), None
    if backend == 'template_stream':
        choices, allowed = engine._choices({})
        stats = engine._new_stats()
        tokens = engine._sample_template(choices, allowed, rng, stats)
        engine.last_stats = {**engine.last_stats, 'sampling': stats,
                             'sampling_adapter': 'existing_full_template_path_no_clamped_partition'}
        return tuple(tokens), None
    variables = tuple(f'y{i}' for i in range(engine.spec.length))
    draw = engine.sample_batch(variables, rng)
    return tuple(draw.assignment[name] for name in variables), draw


def _emit(event):
    print(json.dumps(json_value(event), ensure_ascii=False, allow_nan=False), flush=True)


def _validate_sample(tokens, spec, q, *, draw=None, logz=None):
    checked = verify_music(tokens, spec)
    if not checked.valid:
        raise AssertionError(f'Original token verification failed: {checked.violations}')
    weight = checked.soft_score + sum(q[i, token] for i, token in enumerate(tokens) if i not in spec.observed)
    if not math.isfinite(weight):
        raise AssertionError('Sampler returned a zero-mass complete sequence')
    result = {'valid': True, 'tokens': list(tokens), 'target_log_weight': float(weight),
              'target_soft_score': checked.soft_score}
    if draw is not None and hasattr(draw, 'log_probability'):
        if logz is None:
            raise AssertionError('Probability-bearing exact draw lacks a target normalizer')
        probability_error = abs(float(draw.log_probability) - (weight - logz))
        weight_error = abs(float(draw.log_clamped_partition) - weight)
        if not math.isfinite(probability_error + weight_error) or max(probability_error, weight_error) > 1e-8:
            raise AssertionError('Sample probability differs from original-token target weight')
        result.update(log_probability_error=probability_error, log_clamped_partition_error=weight_error)
    return result


def worker(payload, emit=_emit):
    from threadpoolctl import threadpool_limits
    os.sched_setaffinity(0, {int(payload['cpu'])})
    current = 'load_input'
    engine = None
    measured_seconds = 0.
    validation_seconds = 0.
    try:
        case = load_case(payload['path'])
        backend = payload['backend']
        reason = support_reason(case.metadata, backend)
        if reason:
            emit({'event': 'done', 'status': 'unsupported', 'reason': reason, 'failed_operation': 'support_check'})
            return
        rng = np.random.default_rng(payload['sample_seed'])
        budget = Budget(**payload['budget'])
        emit({'event': 'ready', 'pid': os.getpid(), 'cpu': int(payload['cpu']),
              'baseline_rss_mib': _rss_mib(), 'target_beta': case.spec.motion_cost,
              'target_normalizer_available': backend not in SAMPLERS,
              'sample_adapter': ('existing_private_template_full_path' if backend == 'template_stream'
                else 'public_sample_full' if backend in SAMPLERS or backend == 'onset_reset'
                else 'public_full_batch_with_direct_weight'),
              'sampling_policy': 'Complete path only; no extra clamped partition for a fully assigned sample.'})

        def operation(name, function):
            nonlocal current, measured_seconds
            current = name
            emit({'event': 'start', 'operation': name})
            started = time.perf_counter()
            value = function()
            elapsed = time.perf_counter() - started
            measured_seconds += elapsed
            return value, {'operation': name, 'seconds': elapsed, 'peak_rss_mib': _rss_mib(),
                           'stats': json_value(getattr(engine, 'last_stats', {}))}

        with threadpool_limits(limits=1):
            # Resolve lazy imports before the measured constructor, equally
            # for every backend. Import startup remains inside worker limits.
            if backend == 'product_prefix':
                import tri.inference.product_prefix  # noqa: F401
            elif backend == 'template_stream':
                import tri.inference.template_stream  # noqa: F401
            elif backend == 've_aligned':
                import tri.inference.ordered_ve  # noqa: F401
            elif backend == 'onset_reset':
                import tri.inference.onset_reset  # noqa: F401
            else:
                import tri.inference.onset_reset  # noqa: F401
                import tri.sampling.onset_rejection  # noqa: F401
            def progress(event):
                emit({'event': 'rejection_progress', 'operation': current, 'progress': event})
            engine, measure = operation('compile', lambda: make_solver(case.spec, case.logq, backend, budget, payload['max_proposals'],
                                                                       progress_callback=progress if backend in SAMPLERS else None))
            measure['stats'] = json_value(engine.last_stats)
            emit({'event': 'measurement', **measure})
            logz = None
            if backend not in SAMPLERS:
                logz, measure = operation('target_partition', engine.log_partition)
                if not math.isfinite(logz):
                    raise AssertionError('Positive witness fixture did not yield finite target Z')
                emit({'event': 'measurement', **measure, 'log_z': float(logz), 'target_beta': case.spec.motion_cost})
            else:
                emit({'event': 'normalization', 'target_normalizer_available': False,
                      'reason': 'Exact accepted draws only; proposal_log_z in diagnostics is not target Z_beta.'})
            for index in range(1 + payload['warm_samples']):
                name = 'first_full_sample' if index == 0 else f'repeated_full_sample_{index}'
                (tokens, draw), measure = operation(name, lambda: _sample_complete(engine, backend, rng))
                if backend in SAMPLERS:
                    measure['sampling_diagnostics'] = json_value(draw.diagnostics)
                # Emit timing before the checker. A checker failure is still
                # saved as failure and cannot count as a validated sample.
                measure['sample_index'] = index
                measure['validated'] = False
                emit({'event': 'measurement', **measure})
                current = name + '_verification'
                emit({'event': 'start', 'operation': current})
                started = time.perf_counter()
                checked = _validate_sample(tokens, case.spec, case.logq, draw=draw, logz=logz)
                elapsed = time.perf_counter() - started
                validation_seconds += elapsed
                emit({'event': 'validation', 'operation': name, 'seconds': elapsed, **checked})
            emit({'event': 'done', 'status': 'completed', 'solver_seconds': measured_seconds,
                  'validation_seconds': validation_seconds, 'verified_samples': 1 + payload['warm_samples'],
                  'peak_rss_mib': _rss_mib(), 'stats': json_value(engine.last_stats)})
    except Exception as error:
        emit({'event': 'done', 'status': _failure_status(error), 'failed_operation': current,
              'error': {'type': type(error).__name__, 'message': str(error)},
              'stats': json_value(getattr(error, 'diagnostics', None) or getattr(engine, 'last_stats', {})),
              'completed_solver_seconds': measured_seconds, 'validation_seconds': validation_seconds,
              'peak_rss_mib': _rss_mib()})
        if _failure_status(error) == 'unexpected_error':
            traceback.print_exc(file=sys.stderr)


def isolated(payload, *, worker_seconds, rss_limit_mib, log_prefix):
    import psutil
    log_prefix = Path(log_prefix)
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = log_prefix.with_suffix('.stdout.jsonl')
    stderr_path = log_prefix.with_suffix('.stderr.log')
    environment = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                   'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    started = time.monotonic()
    process = subprocess.Popen([sys.executable, '-m', 'tri.evaluation.shared_rhythm_study', '--worker',
                                json.dumps(payload)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=environment, bufsize=0)
    os.sched_setaffinity(process.pid, {int(payload['cpu'])})
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    selector.register(process.stderr, selectors.EVENT_READ)
    child = psutil.Process(process.pid)
    row = {'status': 'running', 'measurements': {}, 'validations': {}, 'peak_rss_mib': 0.,
           'target_normalizer_available': payload['backend'] not in SAMPLERS,
           'stdout_log': str(stdout_path.resolve()), 'stderr_log': str(stderr_path.resolve())}
    current = 'startup'
    buffer = b''
    done = False
    try:
        with stdout_path.open('wb') as stdout_log, stderr_path.open('wb') as stderr_log:
            while True:
                try:
                    row['peak_rss_mib'] = max(row['peak_rss_mib'], child.memory_info().rss / 2**20)
                except psutil.NoSuchProcess:
                    pass
                for key, _ in selector.select(timeout=.01):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if key.fileobj is process.stderr:
                        stderr_log.write(chunk)
                        continue
                    stdout_log.write(chunk)
                    buffer += chunk
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        event = json.loads(line)
                        kind = event.pop('event')
                        if 'peak_rss_mib' in event:
                            row['peak_rss_mib'] = max(row['peak_rss_mib'], event['peak_rss_mib'])
                        if kind == 'ready':
                            row.update(event)
                            current = 'imports'
                        elif kind == 'start':
                            current = event['operation']
                        elif kind == 'measurement':
                            row['measurements'][event['operation']] = event
                        elif kind == 'validation':
                            row['validations'][event['operation']] = event
                            row['measurements'][event['operation']]['validated'] = True
                        elif kind == 'normalization':
                            row.update(event)
                        elif kind == 'rejection_progress':
                            row['last_rejection_progress'] = event
                        elif kind == 'done':
                            peak = row['peak_rss_mib']
                            row.update(event)
                            row['peak_rss_mib'] = max(peak, row.get('peak_rss_mib', 0.))
                            done = True
                if row['peak_rss_mib'] > rss_limit_mib:
                    row.update(status='rss_limit', failed_operation=current, rss_limit_mib=rss_limit_mib)
                    break
                if time.monotonic() - started > worker_seconds:
                    row.update(status='timeout', failed_operation=current, worker_seconds=worker_seconds)
                    break
                if process.poll() is not None and not selector.get_map():
                    if not done or process.returncode != 0:
                        row.update(status='worker_exit', failed_operation=current, exit_code=process.returncode)
                    break
            if process.poll() is None:
                process.terminate()
            try:
                tail, errors = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                tail, errors = process.communicate()
            stdout_log.write(tail or b'')
            stderr_log.write(errors or b'')
        row['elapsed_worker_seconds'] = time.monotonic() - started
        row['incremental_peak_rss_mib'] = max(0., row['peak_rss_mib'] - row.get('baseline_rss_mib', 0.))
        measurements = row['measurements']
        row['completed_solver_seconds'] = sum(m['seconds'] for m in measurements.values())
        if 'first_full_sample' in row['validations']:
            row['seconds_to_first_verified_sample_excluding_checks'] = sum(
                m['seconds'] for name, m in measurements.items() if name in ('compile', 'target_partition', 'first_full_sample'))
        row['verified_samples'] = len(row['validations'])
        return json_value(row)
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def check_partitions(rows, tolerance=1e-8):
    grouped = defaultdict(list)
    for row in rows:
        partition = row.get('measurements', {}).get('target_partition')
        if partition is not None:
            if not row['target_normalizer_available'] or row['backend'] in SAMPLERS:
                raise AssertionError('Sampling-only backend published a target partition')
            grouped[row['case_id']].append((row['job_id'], float(partition['log_z'])))
    errors = []
    maximum = 0.
    for case_id, values in grouped.items():
        finite = [z for _, z in values]
        difference = max(finite) - min(finite)
        maximum = max(maximum, difference)
        if not math.isfinite(difference) or difference > tolerance:
            errors.append({'case_id': case_id, 'spread': difference, 'records': values})
    return {'cases_with_target_z': len(grouped), 'target_z_records': sum(map(len, grouped.values())),
            'maximum_target_log_z_spread': maximum, 'errors': errors}


def run(config, output, *, resume=False):
    from tri.evaluation.shared_rhythm_cases import prepare_shared_rhythm_cases
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    available_cpus = os.sched_getaffinity(0)
    if config['cpu'] not in available_cpus:
        raise ValueError('Requested CPU is unavailable')
    driver_cpus = sorted(cpu for cpu in available_cpus if cpu >= 4 and cpu != config['cpu'])
    driver_cpu = driver_cpus[0] if driver_cpus else min(available_cpus)
    os.sched_setaffinity(0, {driver_cpu})
    if set(config['backends']) - set(BACKENDS) or len(set(config['backends'])) != len(config['backends']):
        raise ValueError('Backends must be distinct supported names')
    if config['timing_repeats'] < 1 or config['warm_samples'] < 0 or config['worker_seconds'] <= 0 or config['rss_limit_mib'] <= 0:
        raise ValueError('Invalid study repetitions or resource limits')
    with exclusive_run(output / 'study.lock'):
        if (output / 'config.json').exists():
            if not resume or json.loads((output / 'config.json').read_text()) != config:
                raise ValueError('Existing shared-rhythm study requires matching explicit --resume')
        elif (output / 'results.jsonl').exists():
            raise ValueError('Refuse results without a matching study config')
        atomic_json(output / 'config.json', config)
        manifest = prepare_shared_rhythm_cases(config['grid'], output / 'inputs')
        jobs = []
        for case_index, entry in enumerate(manifest['cases']):
            for repeat in range(config['timing_repeats']):
                order = np.random.default_rng(config['order_seed'] + case_index * config['timing_repeats'] + repeat).permutation(config['backends'])
                for backend in order:
                    jobs.append({'job_id': f'{entry["case_id"]}__{backend}__repeat{repeat}',
                                 'case_id': entry['case_id'], 'backend': str(backend), 'repeat': repeat,
                                 'metadata': entry['metadata'], 'path': entry['path'],
                                 # Timing repeats keep the complete sample RNG workload fixed.
                                 'sample_seed': config['sample_seed'] + case_index})
        planned = {job['job_id']: job for job in jobs}
        atomic_json(output / 'plan.json', {'planned_rows': len(jobs), 'jobs': jobs,
            'driver_cpu': driver_cpu, 'worker_cpu': config['cpu'],
            'predeclared_unsupported_rows': sum(support_reason(j['metadata'], j['backend']) is not None for j in jobs),
            'timing_policy': 'Fresh serial process per supported case/backend/repeat; randomized backend order inside each case/repeat. Whole-worker deadline includes imports, compile, optional target Z, first+repeated samples, checks and logging.',
            'sample_workload': f'Compile + optional target partition + first complete sample + {config["warm_samples"]} repeated complete samples on the same engine.'})
        rows = read_jsonl(output / 'results.jsonl', recover_tail=resume)
        completed = set()
        for row in rows:
            key = row['job_id']
            if key in completed or key not in planned or any(row.get(k) != value for k, value in planned[key].items()):
                raise ValueError('Existing result does not match the declared job')
            completed.add(key)
            record_path = output / 'records' / (key + '.json')
            if not record_path.exists():
                atomic_json(record_path, row)
        started = time.time()
        for job in jobs:
            if job['job_id'] in completed:
                continue
            atomic_json(output / 'status.json', {'status': 'running', 'pid': os.getpid(), 'started_at': started,
                'updated_at': time.time(), 'completed': len(rows), 'planned': len(jobs), 'active_job': job['job_id']})
            reason = support_reason(job['metadata'], job['backend'])
            if reason:
                result = {'status': 'unsupported', 'reason': reason, 'support_check': 'predeclared',
                          'target_normalizer_available': False, 'measurements': {}, 'validations': {}}
            else:
                payload = {**job, 'budget': config['budget'], 'max_proposals': config['max_proposals'],
                           'warm_samples': config['warm_samples'], 'cpu': config['cpu']}
                result = isolated(payload, worker_seconds=config['worker_seconds'], rss_limit_mib=config['rss_limit_mib'],
                                  log_prefix=output / 'logs' / job['job_id'])
            row = {**job, **result}
            # Cross-check any newly available target Z before going on to
            # another job. Sampling-only proposal Z never enters this check.
            if row.get('measurements', {}).get('target_partition') is not None:
                consistency = check_partitions([r for r in rows if r['case_id'] == row['case_id']] + [row])
                if consistency['errors']:
                    row['original_worker_status'] = row['status']
                    row.update(status='correctness_failure', failed_operation='cross_backend_target_partition',
                               partition_consistency=consistency)
            with (output / 'results.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            atomic_json(output / 'records' / (job['job_id'] + '.json'), row)
            rows.append(row)
            completed.add(job['job_id'])
            print(f'{len(rows)}/{len(jobs)} {job["job_id"]}: {row["status"]}', flush=True)
            if row['status'] in ('unexpected_error', 'worker_exit', 'correctness_failure'):
                atomic_json(output / 'status.json', {'status': 'failed', 'completed': len(rows),
                    'planned': len(jobs), 'updated_at': time.time(), 'failed_job': job['job_id'],
                    'failure_status': row['status'], 'row_path': str(output / 'records' / (job['job_id'] + '.json')),
                    'note': 'Unexpected/correctness failure was retained. Investigate before explicit resume; no automatic retry.'})
                from tri.evaluation.shared_rhythm_report import write_report
                write_report(output)
                raise RuntimeError(f'Shared-rhythm worker failure retained: {job["job_id"]}: {row["status"]}')
        consistency = check_partitions(rows)
        statuses = dict(Counter(row['status'] for row in rows))
        unexpected = sum(statuses.get(name, 0) for name in ('unexpected_error', 'worker_exit', 'correctness_failure'))
        report = {'status': 'failed' if consistency['errors'] or unexpected else 'completed',
                  'planned': len(jobs), 'recorded': len(rows), 'status_counts': statuses,
                  'partition_consistency': consistency, 'updated_at': time.time(),
                  'elapsed_this_invocation_seconds': time.time() - started,
                  'scope': 'Synthetic controlled multi-span inference; budget and unsupported outcomes are retained. Completed means all planned rows recorded, not every backend solved every target.'}
        atomic_json(output / 'report.json', report)
        atomic_json(output / 'status.json', report)
        from tri.evaluation.shared_rhythm_report import write_report
        write_report(output)
        return report


def main():
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--config', default='configs/shared_rhythm_pilot.yaml')
    parser.add_argument('--out', default='runs/shared_rhythm_pilot')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--status', action='store_true', help='Read current status without starting or resuming work')
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker))
    elif args.status:
        status_path = Path(args.out) / 'status.json'
        print(status_path.read_text() if status_path.exists() else json.dumps({'status': 'not_started', 'out': args.out}))
    else:
        config = yaml.safe_load(Path(args.config).read_text())
        if args.prepare_only:
            from tri.evaluation.shared_rhythm_cases import prepare_shared_rhythm_cases
            prepare_shared_rhythm_cases(config['grid'], Path(args.out) / 'inputs')
        else:
            run(config, args.out, resume=args.resume)


if __name__ == '__main__':
    main()
