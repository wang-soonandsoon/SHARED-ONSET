"""Versioned extension of the frozen shared-rhythm pilot, without editing it.

The synthetic stage reruns a reduced matched baseline set on the original 60
inputs. Real manifests use one fresh engine and individually seeded full draws:
first latency and reused draw latency are separate views of those same draws.
"""
from __future__ import annotations

from collections import Counter
import importlib
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

from tri.evaluation.shared_rhythm_study import (
    SAMPLERS as PILOT_SAMPLERS, _failure_status, _sample_complete as _pilot_sample_complete, _validate_sample,
    check_partitions, make_solver as make_pilot_solver, support_reason as pilot_support_reason,
)
from tri.evaluation.solver_cases import load_case
from tri.evaluation.solver_study import _rss_mib, json_value
from tri.inference.exact import Budget
from tri.runtime import atomic_json, exclusive_run, read_jsonl


NEW_SAMPLERS = frozenset(('onset_rejection_visible', 'onset_rejection_boundary_early',
                          'onset_rejection_visible_early'))
SAMPLERS = PILOT_SAMPLERS | NEW_SAMPLERS
BACKENDS = ('product_multi', 'product_prefix', 've_aligned', 'template_stream', 'onset_reset',
            'onset_rejection_drop', 'onset_rejection_boundary',
            'onset_rejection_visible', 'onset_rejection_boundary_early', 'onset_rejection_visible_early')
PROTOCOL = 'shared_rhythm_extension_v1_independent_draw_seeds'


def support_reason(metadata, backend):
    if backend == 'product_multi' or backend in NEW_SAMPLERS:
        return None  # Cartesian growth is a real budget limit, not unsupported R.
    return pilot_support_reason(metadata, backend)


def make_solver(spec, q, backend, budget, max_proposals, progress_callback=None):
    if backend in NEW_SAMPLERS:
        from tri.sampling.onset_rejection_adaptive import AdaptiveOnsetRejectionSampler
        return AdaptiveOnsetRejectionSampler(spec, q, budget,
            proposal=backend.removeprefix('onset_rejection_'), max_proposals=max_proposals,
            progress_callback=progress_callback)
    if backend == 'product_multi':
        from tri.inference.product_multi import MultiSpanProductChainMusicInference
        return MultiSpanProductChainMusicInference(spec, q, budget)
    return make_pilot_solver(spec, q, backend, budget, max_proposals, progress_callback)


def _sample_complete(engine, backend, rng):
    if backend in NEW_SAMPLERS:
        draw = engine.sample_full(rng)
        return tuple(draw.tokens), draw
    return _pilot_sample_complete(engine, backend, rng)


def _preload(backend):
    if backend in NEW_SAMPLERS:
        importlib.import_module('tri.inference.rank_one_onset')
        importlib.import_module('tri.sampling.onset_rejection_adaptive')
        return
    module = {'product_multi': 'product_multi', 'product_prefix': 'product_prefix',
              've_aligned': 'ordered_ve', 'template_stream': 'template_stream',
              'onset_reset': 'onset_reset'}.get(backend)
    if module:
        importlib.import_module('tri.inference.' + module)
    else:
        importlib.import_module('tri.inference.onset_reset')
        importlib.import_module('tri.sampling.onset_rejection')


def _emit(event):
    print(json.dumps(json_value(event), ensure_ascii=False, allow_nan=False), flush=True)


def worker(payload, emit=_emit):
    from threadpoolctl import threadpool_limits
    os.sched_setaffinity(0, {int(payload['cpu'])})
    current, engine = 'input_loading', None
    solver_seconds, check_seconds, verified = 0., 0., 0
    try:
        case = load_case(payload['path'])
        backend = payload['backend']
        reason = support_reason(case.metadata, backend)
        if reason:
            emit({'event': 'done', 'status': 'unsupported', 'reason': reason})
            return
        if not payload['draw_seeds'] or len(payload['draw_seeds']) != len(set(payload['draw_seeds'])):
            raise ValueError('Every complete draw requires a distinct declared seed')
        _preload(backend)
        emit({'event': 'ready', 'pid': os.getpid(), 'cpu': int(payload['cpu']),
              'baseline_rss_mib': _rss_mib(), 'target_beta': case.spec.motion_cost,
              'target_normalizer_available': backend not in SAMPLERS,
              'sample_adapter': 'public_sample_full' if backend == 'product_multi' or backend == 'onset_reset' or backend in SAMPLERS
                  else 'existing_private_template_full_path' if backend == 'template_stream' else 'public_full_batch_direct_weight',
              'sampling_policy': 'Fresh independent RNG seed per complete draw; one engine/target, no between-draw conditioning. No full-assignment partition recomputation.'})

        def operation(name, function):
            nonlocal current, solver_seconds
            current = name
            emit({'event': 'start', 'operation': name})
            started = time.perf_counter()
            value = function()
            elapsed = time.perf_counter() - started
            solver_seconds += elapsed
            return value, {'operation': name, 'seconds': elapsed, 'peak_rss_mib': _rss_mib(),
                           'stats': json_value(getattr(engine, 'last_stats', {}))}

        def progress(event):
            emit({'event': 'rejection_progress', 'operation': current, 'progress': event})

        with threadpool_limits(limits=1):
            engine, measurement = operation('compile', lambda: make_solver(case.spec, case.logq, backend,
                Budget(**payload['budget']), payload['max_proposals'], progress if backend in SAMPLERS else None))
            measurement['stats'] = json_value(engine.last_stats)
            emit({'event': 'measurement', **measurement})
            logz = None
            if backend not in SAMPLERS:
                logz, measurement = operation('target_partition', engine.log_partition)
                if not math.isfinite(logz):
                    raise AssertionError('Feasible frozen request did not yield finite target Z')
                emit({'event': 'measurement', **measurement, 'log_z': float(logz), 'target_beta': case.spec.motion_cost})
            else:
                emit({'event': 'normalization', 'target_normalizer_available': False,
                      'reason': 'Proposal normalizers support rejection only; no target Z_beta is provided.'})
            for index, seed in enumerate(payload['draw_seeds']):
                name = 'first_full_sample' if index == 0 else f'repeated_full_sample_{index}'
                # RNG setup is excluded consistently; every actual sampler call
                # includes all rejection/diagnostic/progress work.
                rng = np.random.default_rng(seed)
                def sample():
                    return (tuple(engine.sample_full(rng)), None) if backend == 'product_multi' else _sample_complete(engine, backend, rng)
                (tokens, draw), measurement = operation(name, sample)
                measurement.update(sample_index=index, draw_seed=seed, validated=False)
                if backend in SAMPLERS:
                    measurement['sampling_diagnostics'] = json_value(draw.diagnostics)
                emit({'event': 'measurement', **measurement})
                current = name + '_verification'
                emit({'event': 'start', 'operation': current})
                started = time.perf_counter()
                checked = _validate_sample(tokens, case.spec, case.logq, draw=draw, logz=logz)
                elapsed = time.perf_counter() - started
                check_seconds += elapsed
                verified += 1
                emit({'event': 'validation', 'operation': name, 'sample_index': index, 'draw_seed': seed,
                      'seconds': elapsed, **checked})
            emit({'event': 'done', 'status': 'completed', 'solver_seconds': solver_seconds,
                  'validation_seconds': check_seconds, 'verified_samples': verified,
                  'peak_rss_mib': _rss_mib(), 'stats': json_value(engine.last_stats)})
    except Exception as error:
        # Extension inputs have an independently checked feasible witness and
        # finite full130 q; zero mass here is a correctness error, not workload failure.
        status = 'correctness_failure' if type(error).__name__ == 'ZeroMass' else _failure_status(error)
        emit({'event': 'done', 'status': status, 'failed_operation': current,
              'error': {'type': type(error).__name__, 'message': str(error)},
              'stats': json_value(getattr(error, 'diagnostics', None) or getattr(engine, 'last_stats', {})),
              'completed_solver_seconds': solver_seconds, 'validation_seconds': check_seconds,
              'verified_samples': verified, 'peak_rss_mib': _rss_mib()})
        if _failure_status(error) == 'unexpected_error':
            traceback.print_exc(file=sys.stderr)


def isolated(payload, *, worker_seconds, rss_limit_mib, log_prefix):
    """Physical process cap; persist events and last in-flight proposal counts."""
    import psutil
    prefix = Path(log_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = prefix.with_suffix('.stdout.jsonl'), prefix.with_suffix('.stderr.log')
    environment = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                   'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    started = time.monotonic()
    process = subprocess.Popen([sys.executable, '-m', 'tri.evaluation.shared_rhythm_extension', '--worker',
        json.dumps(payload)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, env=environment)
    os.sched_setaffinity(process.pid, {int(payload['cpu'])})
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    selector.register(process.stderr, selectors.EVENT_READ)
    child = psutil.Process(process.pid)
    row = {'status': 'running', 'measurements': {}, 'validations': {}, 'peak_rss_mib': 0.,
           'target_normalizer_available': payload['backend'] not in SAMPLERS,
           'stdout_log': str(stdout_path.resolve()), 'stderr_log': str(stderr_path.resolve())}
    current, buffer, done = 'startup', b'', False
    first_rss_failure = None
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
                            if event['peak_rss_mib'] > rss_limit_mib and first_rss_failure is None:
                                first_rss_failure = event.get('operation', current)
                        if kind == 'ready':
                            row.update(event)
                            current = 'compile_setup'
                        elif kind == 'start':
                            current = event['operation']
                        elif kind == 'measurement':
                            row['measurements'][event['operation']] = event
                            reference = payload.get('reference_target_log_z')
                            if event['operation'] == 'target_partition' and reference is not None:
                                difference = abs(float(event['log_z']) - reference)
                                if not math.isfinite(difference) or difference > 1e-8:
                                    row.update(status='correctness_failure', failed_operation='cross_backend_target_partition',
                                               error={'message': 'Target Z differs from an earlier same-input exact record', 'log_error': difference})
                        elif kind == 'validation':
                            row['validations'][event['operation']] = event
                            row['measurements'][event['operation']]['validated'] = True
                        elif kind == 'normalization':
                            row.update(event)
                        elif kind == 'rejection_progress':
                            row['last_rejection_progress'] = event
                        elif kind == 'done':
                            previous_status, peak = row['status'], row['peak_rss_mib']
                            row.update(event)
                            row['peak_rss_mib'] = max(peak, row.get('peak_rss_mib', 0.))
                            if previous_status == 'correctness_failure':
                                row['status'] = previous_status
                            done = True
                if row['peak_rss_mib'] > rss_limit_mib:
                    row.update(status='rss_limit', failed_operation=first_rss_failure or current, rss_limit_mib=rss_limit_mib)
                    break
                if row['status'] == 'correctness_failure':
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
        row['completed_solver_seconds'] = sum(m['seconds'] for m in row['measurements'].values())
        row['verified_samples'] = len(row['validations'])
        row['requested_samples'] = len(payload['draw_seeds'])
        row['draws_truncated'] = row['verified_samples'] < row['requested_samples']
        return json_value(row)
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def prepare_plan(configuration, stage):
    if configuration.get('protocol') != PROTOCOL or stage not in configuration['stages']:
        raise ValueError('Unknown extension protocol or stage')
    settings = {**configuration['common'], **configuration['stages'][stage]}
    if not settings['backends'] or len(set(settings['backends'])) != len(settings['backends']) or set(settings['backends']) - set(BACKENDS):
        raise ValueError('Declare distinct supported backends')
    if settings['timing_repeats'] < 1 or not 1 <= settings['samples_per_worker'] < 100000:
        raise ValueError('Invalid repeat or sample counts')
    if stage != 'synthetic' and settings['timing_repeats'] != 1:
        raise ValueError('Real sampling uses independent draws, not duplicate timing histories')
    if settings['worker_seconds'] <= 0 or settings['rss_limit_mib'] <= 0:
        raise ValueError('Worker limits must be positive')
    Budget(**settings['budget'])
    manifest_path = Path(settings['manifest']).resolve()
    manifest = json.loads(manifest_path.read_text())
    entries = manifest['cases']
    if len(entries) != settings['expected_cases'] or len({entry['case_id'] for entry in entries}) != len(entries):
        raise ValueError('Frozen manifest does not match the declared unique request count')
    jobs = []
    for case_index, entry in enumerate(entries):
        case_path = Path(entry['path'])
        if not case_path.is_absolute():
            case_path = manifest_path.parent / case_path
        case = load_case(case_path)
        if case.case_id != entry['case_id'] or case.metadata != entry['metadata']:
            raise ValueError('Manifest and frozen case disagree')
        if case.metadata['family'] != settings['family']:
            raise ValueError('Manifest provenance differs from declared stage family')
        for field in ('R', 'beta', 'visibility'):
            if field not in case.metadata:
                raise ValueError('Case lacks required shared-rhythm metadata')
        if case.metadata['beta'] != case.spec.motion_cost:
            raise ValueError('Case metadata beta differs from original MusicSpec')
        seeds = [int(settings['sample_seed'] + case_index * 100000 + index) for index in range(settings['samples_per_worker'])]
        for repeat in range(settings['timing_repeats']):
            order = np.random.default_rng(settings['order_seed'] + case_index * settings['timing_repeats'] + repeat).permutation(settings['backends'])
            for backend in order:
                jobs.append({'job_id': f'{case.case_id}__{backend}__repeat{repeat}', 'case_id': case.case_id,
                    'backend': str(backend), 'repeat': repeat, 'path': str(case_path.resolve()),
                    'metadata': case.metadata, 'draw_seeds': seeds})
    stat = manifest_path.stat()
    snapshot = {'version': 1, 'protocol': PROTOCOL, 'stage': stage, 'settings': settings,
                'manifest_identity': {'path': str(manifest_path), 'bytes': stat.st_size, 'modified_ns': stat.st_mtime_ns},
                'case_ids': [entry['case_id'] for entry in entries]}
    unsupported = sum(support_reason(job['metadata'], job['backend']) is not None for job in jobs)
    plan = {'version': 1, 'protocol': PROTOCOL, 'stage': stage, 'jobs': jobs, 'planned_rows': len(jobs),
            'predeclared_unsupported_rows': unsupported, 'planned_workers': len(jobs) - unsupported,
            'maximum_worker_budget_seconds': (len(jobs) - unsupported) * settings['worker_seconds'],
            'sampling_policy': 'Each draw uses its own declared numpy seed, identical across methods. Synthetic timing repeats replay the same seeds; real uses no timing repeat. One cold engine followed by reused independent draws.',
            'comparison_policy': 'Only this version/run supplies matched timing ratios. Older template/VE/pilot results are historical context, never mixed into new speedups.'}
    return snapshot, plan


def run(configuration, stage, output, *, resume=False, plan_only=False):
    from tri.evaluation.shared_rhythm_extension_report import write_report
    available = os.sched_getaffinity(0)
    target_cpu = configuration['common']['cpu']
    if target_cpu not in available:
        raise ValueError('Requested worker CPU is outside launch affinity')
    control = sorted(cpu for cpu in available if cpu >= 4 and cpu != target_cpu)
    os.sched_setaffinity(0, {control[0] if control else min(available)})
    snapshot, plan = prepare_plan(configuration, stage)
    settings = snapshot['settings']
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(output / 'study.lock'):
        config_path = output / 'config.json'
        if config_path.exists():
            if not resume or json.loads(config_path.read_text()) != snapshot:
                raise ValueError('Existing extension run requires matching explicit --resume')
        elif (output / 'results.jsonl').exists():
            raise ValueError('Results lack their frozen extension config')
        atomic_json(config_path, snapshot)
        if (output / 'plan.json').exists() and json.loads((output / 'plan.json').read_text()) != plan:
            raise ValueError('Frozen extension plan changed')
        atomic_json(output / 'plan.json', plan)
        rows = read_jsonl(output / 'results.jsonl', recover_tail=resume)
        planned = {job['job_id']: job for job in plan['jobs']}
        completed = set()
        for row in rows:
            key = row['job_id']
            if key in completed or key not in planned or any(row.get(k) != v for k, v in planned[key].items()):
                raise ValueError('Existing record does not uniquely match its declared job')
            completed.add(key)
            record_path = output / 'records' / (key + '.json')
            if not record_path.exists():
                atomic_json(record_path, row)
        if plan_only:
            atomic_json(output / 'status.json', {'status': 'planned', 'recorded': len(rows),
                        'planned': len(plan['jobs']), 'planned_workers': plan['planned_workers']})
            return plan
        print(json.dumps({'status': 'started', 'stage': stage, 'planned_rows': len(plan['jobs']),
                          'already_recorded': len(rows), 'workers': plan['planned_workers']}), flush=True)
        for index, job in enumerate(plan['jobs']):
            if job['job_id'] in completed:
                continue
            atomic_json(output / 'status.json', {'status': 'running', 'pid': os.getpid(),
                'recorded': len(rows), 'planned': len(plan['jobs']), 'active_job': job['job_id'], 'updated_at': time.time()})
            reason = support_reason(job['metadata'], job['backend'])
            if reason:
                result = {'status': 'unsupported', 'reason': reason, 'support_check': 'predeclared',
                          'target_normalizer_available': False, 'measurements': {}, 'validations': {}}
            else:
                reference = next((float(row['measurements']['target_partition']['log_z']) for row in rows
                    if row['case_id'] == job['case_id'] and 'target_partition' in row.get('measurements', {})), None)
                payload = {**job, 'cpu': settings['cpu'], 'budget': settings['budget'],
                           'max_proposals': settings['max_proposals'], 'reference_target_log_z': reference}
                result = isolated(payload, worker_seconds=settings['worker_seconds'], rss_limit_mib=settings['rss_limit_mib'],
                                  log_prefix=output / 'logs' / job['job_id'])
            row = {**job, **result, 'protocol': PROTOCOL, 'stage': stage}
            consistency = check_partitions([r for r in rows if r['case_id'] == job['case_id']] + [row])
            if consistency['errors']:
                row.update(original_worker_status=row['status'], status='correctness_failure',
                           failed_operation='cross_backend_target_partition', partition_consistency=consistency)
            with (output / 'results.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            atomic_json(output / 'records' / (job['job_id'] + '.json'), row)
            rows.append(row)
            completed.add(job['job_id'])
            if row['status'] in ('unexpected_error', 'worker_exit', 'correctness_failure'):
                atomic_json(output / 'status.json', {'status': 'failed', 'recorded': len(rows),
                    'planned': len(plan['jobs']), 'failed_job': job['job_id'], 'failure_status': row['status']})
                write_report(output)
                raise RuntimeError('Extension failure retained; investigate before resuming: ' + job['job_id'])
            if index + 1 == len(plan['jobs']) or plan['jobs'][index + 1]['case_id'] != job['case_id']:
                write_report(output)
        consistency = check_partitions(rows)
        report = {'status': 'failed' if consistency['errors'] else 'completed', 'planned': len(plan['jobs']),
                  'recorded': len(rows), 'status_counts': dict(Counter(row['status'] for row in rows)),
                  'partition_consistency': consistency, 'updated_at': time.time()}
        atomic_json(output / 'status.json', report)
        write_report(output)
        print(json.dumps(report), flush=True)
        return report


def main():
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--config', default='configs/shared_rhythm_extension.yaml')
    parser.add_argument('--stage', default='synthetic')
    parser.add_argument('--out', default='runs/shared_rhythm_extension')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker))
    elif args.status:
        path = Path(args.out) / 'status.json'
        print(path.read_text() if path.exists() else json.dumps({'status': 'not_started'}))
    else:
        run(yaml.safe_load(Path(args.config).read_text()), args.stage, args.out, resume=args.resume, plan_only=args.plan_only)


if __name__ == '__main__':
    main()
