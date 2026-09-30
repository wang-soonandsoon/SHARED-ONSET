"""Small, actual neural-decoder timing control for standard product caches.

Three fixed requests x three methods x three timing repeats x two backends.
The original replicate-0 seed is reused in every timing repeat. These are 54
actual decodes, but only three requests from two works, not 54 music examples.
The provider wrapper only synchronizes and counts/time calls; it records no q
arrays, factory events, or internal solver calls. Independent output checks and
JSON writing occur after the synchronized research_decode wall timer stops.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from statistics import median
import time

import numpy as np

from tri.domain.music import verify_music
from tri.inference.exact import Budget
from tri.runtime import atomic_json, exclusive_run, read_jsonl

STAGES = ('short', 'bar8', 'bar16')
METHODS = ('one_shot_joint', 'tri_direct', 'smc_4')
BACKENDS = ('product_reuse', 'product_prefix')
TIMING_REPEATS = 3
ORDER_SEED = 91917


def load_source(revision_root, stage):
    """Read the same frozen first-unknown request and replicate-0 source as the probe.

    File identities are the repository's ordinary size/mtime identities. Clean
    hidden melody is never used to choose a seed, a generated token, or a winner.
    """
    from tri.evaluation.cohorts import file_identity

    evaluation_dir = Path(revision_root) / stage / 'evaluation'
    evaluation = json.loads((evaluation_dir / 'config.json').read_text())
    cohort_path = Path(evaluation['cohort']['path'])
    if file_identity(cohort_path) != evaluation['cohort']:
        raise ValueError('Frozen cohort identity changed')
    cohort = json.loads(cohort_path.read_text())
    case_index, case = next((i, case) for i, case in enumerate(cohort['requests'])
                            if case['kind'] == 'unknown')
    identities = {'checkpoint': evaluation['checkpoint'], 'cohort': evaluation['cohort'],
                  'dataset': cohort['options']['dataset'], 'chords': cohort['options']['chords']}
    for name, identity in identities.items():
        if file_identity(identity['path']) != identity:
            raise ValueError(f'Frozen {name} identity changed')
    original = read_jsonl(evaluation_dir / 'results.jsonl')
    originals = {}
    for method in METHODS:
        selected = [row for row in original if row['cohort_case_id'] == case['case_id']
                    and row['method'] == method and row['replicate'] == 0]
        if len(selected) != 1 or selected[0].get('raw_tokens') is None:
            raise ValueError('Expected one retained original replicate-0 output per method')
        row = selected[0]
        originals[method] = {'request_id': row['request_id'], 'raw_tokens': row['raw_tokens'],
                             'model_calls': row['model_calls'],
                             'elapsed_decode_seconds': row['elapsed_decode_seconds']}
    return {'stage': stage, 'case_id': case['case_id'], 'case_index': case_index,
            'kind': case['kind'], 'work_id': str(case['work_id']),
            'source_index': case['source_index'], 'source_start_cell': case['source_start_cell'],
            'spec': case['spec'], 'seed': evaluation['seed'] + case_index * 10000,
            'steps': evaluation['steps'], 'original_backend': evaluation['backend'],
            'original_replicate': 0, 'identities': identities,
            'evaluation_config': file_identity(evaluation_dir / 'config.json'),
            'original_results': file_identity(evaluation_dir / 'results.jsonl'),
            'originals': originals}


def make_plan(sources, *, order_seed=ORDER_SEED):
    """Adjacent backend pairs; alternate their order over the three timing repeats.

    This independent RNG only chooses execution order. research_decode always
    receives the original source seed, with no timing-repeat or backend offset.
    """
    rng = np.random.default_rng(order_seed)
    plan = []
    for source in sources:
        first_orders = {method: list(rng.permutation(BACKENDS)) for method in METHODS}
        for repeat in range(TIMING_REPEATS):
            for method in rng.permutation(METHODS):
                order = first_orders[method][::(-1 if repeat % 2 else 1)]
                for backend in order:
                    plan.append({'job_id': f"{source['stage']}:{method}:timing{repeat}:{backend}",
                                 'order_index': len(plan), 'stage': source['stage'],
                                 'case_id': source['case_id'], 'method': str(method),
                                 'backend': str(backend), 'timing_repeat': repeat,
                                 'seed': source['seed'], 'steps': source['steps']})
    return plan


class TimedProvider:
    """Transparent, lightweight timing; the returned q object is unmodified.

    Each provider measurement includes input tensors, neural forward, full130
    float64 logsoftmax, device-to-host copy and the finishing synchronization.
    The initial synchronization and wrapper accounting remain in decode wall
    time. No nested solver timings are added to this sum.
    """
    def __init__(self, provider, synchronize, clock=time.perf_counter):
        self.provider = provider
        self.synchronize = synchronize
        self.clock = clock
        self.model_calls = 0
        self.provider_seconds = 0.0

    def __call__(self, state, noise):
        self.synchronize()
        started = self.clock()
        self.model_calls += 1
        try:
            return self.provider(state, noise)
        finally:
            self.synchronize()
            self.provider_seconds += self.clock() - started


def measure_decode(source, job, spec, provider, *, synchronize, decoder=None,
                   checker=verify_music, clock=time.perf_counter):
    """One actual decode, preserving failures and checking only after timing.

    Internal checker calls already made by research_decode remain inside wall
    time; the extra independent checker below is excluded and timed separately.
    """
    if decoder is None:
        from tri.sampling.research_methods import research_decode
        decoder = research_decode
    timed = TimedProvider(provider, synchronize, clock)
    row = {**job, 'work_id': source['work_id'], 'source_index': source['source_index'],
           'original_replicate': 0, 'status': 'running'}
    result = None
    error = None
    synchronize()
    started = clock()
    try:
        result = decoder(job['method'], spec, timed, steps=job['steps'], seed=job['seed'],
                         backend=job['backend'], budget=Budget())
    except Exception as caught:
        error = caught
    finally:
        synchronize()
        elapsed = clock() - started
    row.update(decode_wall_seconds=elapsed, provider_seconds=timed.provider_seconds,
               non_provider_wall_seconds=elapsed - timed.provider_seconds,
               measured_model_calls=timed.model_calls)
    if error is not None:
        row.update(status='decode_error', failed_phase='research_decode',
                   error={'type': type(error).__name__, 'message': str(error)})
        return row
    row.update(raw_tokens=[int(token) for token in result.tokens], model_calls=int(result.model_calls),
               model_call_accounting_valid=int(result.model_calls) == timed.model_calls)
    original = source['originals'][job['method']]
    row.update(original_tokens_exact_match=row['raw_tokens'] == original['raw_tokens'],
               original_model_calls_exact_match=result.model_calls == original['model_calls'])
    started = clock()
    try:
        checked = checker(result.tokens, spec)
        row['verification'] = {'valid': bool(checked.valid), 'violations': list(checked.violations),
                               'soft_score': float(checked.soft_score)}
        row['status'] = ('completed' if checked.valid and row['model_call_accounting_valid']
                         else 'correctness_failure')
    except Exception as caught:
        row.update(status='correctness_failure', failed_phase='independent_checker',
                   error={'type': type(caught).__name__, 'message': str(caught)})
    row['external_verification_seconds'] = clock() - started
    return row


def summarize(config, rows):
    """Use the planned nine stage/method comparisons, including missing/failed rows.

    Ratios use per-backend medians of all three successful timing repeats. A
    changed token trajectory is retained and labeled, never repaired or dropped.
    Wall minus provider is decoder/solver/bookkeeping residual, not pure solver
    time. The provider wrapper/synchronization overhead is included in wall.
    """
    indexed = {row['job_id']: row for row in rows}
    comparisons = []
    ratios = []
    for source in config['sources']:
        for method in METHODS:
            groups = {}
            token_pairs = []
            for backend in BACKENDS:
                planned = [job for job in config['plan'] if job['stage'] == source['stage']
                           and job['method'] == method and job['backend'] == backend]
                observed = [indexed[job['job_id']] for job in planned if job['job_id'] in indexed]
                completed = [row for row in observed if row['status'] == 'completed']
                all_complete = len(completed) == TIMING_REPEATS
                group = {'planned_repetitions': TIMING_REPEATS, 'recorded_repetitions': len(observed),
                         'completed_repetitions': len(completed),
                         'status_counts': dict(Counter(row['status'] for row in observed)),
                         'model_calls': [row.get('model_calls') for row in observed],
                         'tokens_identical_across_timing_repeats':
                             all(row['raw_tokens'] == completed[0]['raw_tokens'] for row in completed)
                             if all_complete else None}
                for metric in ('decode_wall_seconds', 'provider_seconds', 'non_provider_wall_seconds',
                               'external_verification_seconds'):
                    group['median_' + metric] = median(row[metric] for row in completed) if all_complete else None
                groups[backend] = group
            for repeat in range(TIMING_REPEATS):
                pair = [indexed.get(f"{source['stage']}:{method}:timing{repeat}:{backend}") for backend in BACKENDS]
                comparable = all(row is not None and 'raw_tokens' in row for row in pair)
                details = {'timing_repeat': repeat, 'outputs_available': comparable}
                if comparable:
                    left, right = (row['raw_tokens'] for row in pair)
                    same = left == right
                    details.update(tokens_exact_match=same, trajectory='same_output' if same else 'different_output',
                                   differing_positions=[i for i in range(max(len(left), len(right)))
                                       if i >= len(left) or i >= len(right) or left[i] != right[i]],
                                   model_calls_exact_match=pair[0].get('model_calls') == pair[1].get('model_calls'))
                token_pairs.append(details)
            left, right = (groups[backend]['median_decode_wall_seconds'] for backend in BACKENDS)
            ratio = left / right if left is not None and right is not None and right > 0 else None
            if ratio is not None:
                ratios.append(ratio)
            comparisons.append({'stage': source['stage'], 'method': method, 'case_id': source['case_id'],
                                'work_id': source['work_id'], 'seed': source['seed'], 'backends': groups,
                                'token_pairs': token_pairs,
                                'decode_speedup_product_reuse_over_product_prefix': ratio})
    failed = sum(row['status'] != 'completed' for row in rows)
    completed_plan = len(indexed) == len(config['plan'])
    return {'status': ('completed_with_failures' if failed else 'completed') if completed_plan else 'partial',
            'planned_decodes': len(config['plan']), 'recorded_decodes': len(rows),
            'successful_decodes': len(rows) - failed, 'failed_decodes': failed,
            'unique_frozen_requests': len({(s['stage'], s['case_id']) for s in config['sources']}),
            'unique_works': len({s['work_id'] for s in config['sources']}),
            'planned_stage_method_comparisons': len(config['sources']) * len(METHODS),
            'complete_stage_method_comparisons': len(ratios),
            'geomean_decode_speedup_on_complete_comparisons':
                math.exp(sum(math.log(value) for value in ratios) / len(ratios)) if ratios else None,
            'comparisons': comparisons,
            'scope': 'Three latency repeats reuse each original replicate-0 seed. Three frozen requests / two works in the full plan; no additional independent musical samples or quality conclusions.',
            'timing_scope': 'Synchronized actual research_decode wall includes lightweight provider timing, its synchronization, and decoder-internal checks. Model load, one per-stage neural warmup, the extra independent checker, and output I/O are excluded. non_provider_wall_seconds includes decoder/solver/bookkeeping overhead and is not pure solver time.',
            'trajectory_scope': 'Different returned tokens are retained. An equal final sequence is not proof of identical internal SMC particles or every model input; this lightweight run does not record full trajectories. Speed ratios with different outputs describe actual runs, not isolated fixed-trajectory cache cost.'}


def _load_stage(source, device):
    from tri.data.chords import load_chord_sidecar
    from tri.evaluation.batch import load_evaluation_windows
    from tri.evaluation.cohorts import deserialize_spec, file_identity
    from tri.models.grid import ModelProbabilityProvider
    from tri.models.train import load_checkpoint

    for identity in (*source['identities'].values(), source['evaluation_config'], source['original_results']):
        if file_identity(identity['path']) != identity:
            raise ValueError('Frozen source changed after plan creation')
    dataset = source['identities']['dataset']['path']
    chords = source['identities']['chords']['path']
    window = next(window for window in load_evaluation_windows(dataset, limit=2**31-1, split='validation')
                  if window.source_index == source['source_index'])
    if str(window.work_id) != str(source['work_id']) or window.start_cell != source['source_start_cell']:
        raise ValueError('Frozen request source location changed')
    condition = load_chord_sidecar(dataset, chords)['chord_features'][window.source_index]
    spec = deserialize_spec(source['spec'])
    model = load_checkpoint(source['identities']['checkpoint']['path'], device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    provider = ModelProbabilityProvider(model, tuple(i for i in range(spec.length) if i not in spec.observed),
                                        condition if model.config.condition_dim else None)
    return spec, provider


def _append_row(path, row):
    # Serialize before append: a nonfinite output must not leave a partial row.
    encoded = json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _execute(config, output, *, stage_loader, synchronize, resume=False,
             decoder=None, checker=verify_music, clock=time.perf_counter):
    """Durable serial driver, separated from GPU setup for meaningful CPU tests."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(output / 'control.lock'):
        config_path = output / 'config.json'
        if config_path.exists():
            if not resume:
                raise ValueError('Output already exists; use --resume to retain existing rows')
            if json.loads(config_path.read_text()) != config:
                raise ValueError('Resume configuration or frozen source identity changed')
        else:
            if (output / 'results.jsonl').exists():
                raise ValueError('Results exist without their frozen configuration')
            atomic_json(config_path, config)
        rows = read_jsonl(output / 'results.jsonl', recover_tail=resume)
        planned = {job['job_id']: job for job in config['plan']}
        seen = set()
        for row in rows:
            key = row['job_id']
            if key in seen or key not in planned or any(row.get(k) != v for k, v in planned[key].items()):
                raise ValueError('Existing result does not uniquely match its planned job')
            seen.add(key)
        run_started = clock()
        current = None

        def publish(state):
            report = summarize(config, rows)
            atomic_json(output / 'summary.json', report)
            atomic_json(output / 'status.json', {'status': state, 'planned': len(planned),
                        'recorded': len(rows), 'completed': report['successful_decodes'],
                        'failed': report['failed_decodes'], 'current_job': current,
                        'elapsed_invocation_seconds': clock() - run_started})
            return report

        publish('running')
        try:
            for source in config['sources']:
                jobs = [job for job in config['plan'] if job['stage'] == source['stage'] and job['job_id'] not in seen]
                if not jobs:
                    continue
                current = {'stage': source['stage'], 'phase': 'model_load_and_warmup'}
                publish('running')
                spec, provider = stage_loader(source, config['device'])
                initial = tuple(spec.observed.get(i) for i in range(spec.length))
                synchronize()
                warmup_started = clock()
                provider(initial, 1.0)
                synchronize()
                warmup_seconds = clock() - warmup_started
                for job in jobs:
                    current = job
                    publish('running')
                    row = measure_decode(source, job, spec, provider, synchronize=synchronize,
                                         decoder=decoder, checker=checker, clock=clock)
                    row['warmup_seconds_excluded'] = warmup_seconds
                    _append_row(output / 'results.jsonl', row)
                    rows.append(row)
                    seen.add(job['job_id'])
                    publish('running')
                    print(json.dumps({'job_id': job['job_id'], 'status': row['status'],
                                      'decode_wall_seconds': row['decode_wall_seconds'],
                                      'provider_seconds': row['provider_seconds'],
                                      'model_calls': row['measured_model_calls']}), flush=True)
                    if row['status'] != 'completed':
                        raise RuntimeError(f"Recorded {row['status']} for {job['job_id']}; investigate before resuming")
                del provider
        except BaseException as error:
            publish('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')
            raise
        current = None
        report = summarize(config, rows)
        publish(report['status'])
        return report


def run_control(revision_root, output, *, device='cuda:0', cpu=2, resume=False):
    """Run only after the separate CPU timing studies have finished.

    CPU 2 or 4, GPU 0, one compute thread. No CPU fallback or new training.
    Resume skips all durably recorded rows, including failures, and rewarms the
    stage model before any remaining jobs; failed measurements are never retried.
    """
    if cpu not in (2, 4) or device != 'cuda:0':
        raise ValueError('Formal control requires CPU 2 or 4 and device cuda:0')
    os.sched_setaffinity(0, {cpu})
    import importlib
    import torch
    from threadpoolctl import threadpool_limits

    # Lazy backend imports must not charge only their first timed invocation.
    # This initializes no engine, messages or neural outputs.
    for module in ('tri.inference.product_cached', 'tri.inference.product_prefix',
                   'tri.sampling.research_methods'):
        importlib.import_module(module)
    torch.set_num_threads(1)
    revision_root, output = Path(revision_root).resolve(), Path(output).resolve()
    if output == revision_root or revision_root in output.parents:
        raise ValueError('Control output must not modify the frozen revision tree')
    sources = [load_source(revision_root, stage) for stage in STAGES]
    config = {'version': 1, 'revision_root': str(revision_root), 'device': device, 'cpu': cpu,
              'backends': list(BACKENDS), 'methods': list(METHODS), 'stages': list(STAGES),
              'timing_repeats': TIMING_REPEATS, 'order_seed': ORDER_SEED,
              'budget': asdict(Budget()), 'sources': sources, 'plan': make_plan(sources),
              'neural_warmup_calls_per_loaded_stage': 1,
              'backend_modules_imported_before_measurement': True,
              'selection': 'First frozen unknown request per stage; original replicate0 seed reused across timing repetitions and backends.',
              'order': 'Adjacent backend pairs; fixed independent RNG shuffles methods and first backend order, reversed at timing repeat1.',
              'instrumentation': 'Provider synchronization/counting/timing only; no q snapshots, factory patches or solver tape.',
              'no_training': True}
    with threadpool_limits(limits=1):
        return _execute(config, output, stage_loader=_load_stage,
                        synchronize=lambda: torch.cuda.synchronize(device), resume=resume)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision-root', default='runs/revision')
    parser.add_argument('--out', default='runs/decision_study/end_to_end')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu', type=int, choices=(2, 4), default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run_control(args.revision_root, args.out, device=args.device, cpu=args.cpu, resume=args.resume)


if __name__ == '__main__':
    main()
