from collections import Counter
import json
import os

import numpy as np
import pytest

from tri.evaluation.shared_rhythm_cases import prepare_shared_rhythm_cases, make_shared_rhythm_case
from tri.evaluation.shared_rhythm_study import support_reason, check_partitions, worker, isolated, _validate_sample


def _small_config():
    return {'version': 1, 'grid': {'R': [2, 3], 'L': 2, 'D': 3, 'K': 1, 'betas': [0., .02],
                'visibility': ['unknown'], 'q_seeds': [3]},
            'backends': ['template_stream', 've_aligned', 'product_prefix', 'onset_reset',
                         'onset_rejection_drop', 'onset_rejection_boundary'],
            'cpu': min(os.sched_getaffinity(0)), 'timing_repeats': 3, 'warm_samples': 1,
            'sample_seed': 35, 'order_seed': 45, 'worker_seconds': 15, 'rss_limit_mib': 512,
            'max_proposals': 100,
            'budget': {'max_factor_entries': 67108864, 'max_workspace_bytes': 536870912,
                       'max_oracle_assignments': 9223372036854775807}}


def _payload(tmp_path, backend='template_stream', R=2):
    config = _small_config()
    grid = {**config['grid'], 'R': [R], 'betas': [0.]}
    case = prepare_shared_rhythm_cases(grid, tmp_path / 'inputs')['cases'][0]
    return {**case, 'backend': backend, **{k: config[k] for k in ('cpu', 'budget', 'max_proposals', 'warm_samples', 'sample_seed')}}


@pytest.mark.parametrize('R,beta,backend,unsupported', [
    (2, .02, 'product_prefix', False), (3, 0., 'product_prefix', True),
    (8, 0., 'onset_reset', False), (2, .02, 'onset_reset', True),
    (8, .02, 've_aligned', False), (8, .02, 'onset_rejection_boundary', False)])
def test_support_matrix_is_explicit_and_does_not_call_budget_a_layout_failure(R, beta, backend, unsupported):
    assert (support_reason({'R': R, 'beta': beta}, backend) is not None) == unsupported


@pytest.mark.parametrize('R,backend', [(2, 'template_stream'), (3, 'template_stream'), (4, 'template_stream'),
                                     (2, 've_aligned'), (3, 've_aligned'), (4, 've_aligned'), (2, 'product_prefix')])
def test_generic_baselines_really_accept_multi_span_small_cases(tmp_path, R, backend):
    events = []
    worker(_payload(tmp_path, backend, R), emit=events.append)
    final = events[-1]
    assert final['event'] == 'done'
    assert final['status'] == 'completed', events
    partitions = [e for e in events if e['event'] == 'measurement' and e['operation'] == 'target_partition']
    assert len(partitions) == 1 and np.isfinite(partitions[0]['log_z'])
    assert len([e for e in events if e['event'] == 'validation']) == 2


def test_original_token_checker_catches_invalid_or_zero_weight_draw():
    case = make_shared_rhythm_case(L=2, D=3, K=1)
    changed = list(case.witness)
    changed[0] = 0
    with pytest.raises(AssertionError, match='token verification'):
        _validate_sample(changed, case.spec, case.logq)
    q = case.logq.copy()
    pos = next(i for i in range(case.spec.length) if i not in case.spec.observed)
    q[pos, case.witness[pos]] = -np.inf
    with pytest.raises(AssertionError, match='zero-mass'):
        _validate_sample(case.witness, case.spec, q)


@pytest.mark.parametrize('backend', ['onset_reset', 'onset_rejection_drop', 'onset_rejection_boundary'])
@pytest.mark.parametrize('R', [2, 3])
def test_new_backend_smoke_reuses_plan_and_counts_every_proposal(tmp_path, backend, R):
    events = []
    worker(_payload(tmp_path, backend, R), emit=events.append)
    assert events[-1]['status'] == 'completed', events
    samples = [e for e in events if e['event'] == 'measurement' and 'full_sample' in e['operation']]
    assert len(samples) == 2
    if backend == 'onset_reset':
        assert all(s['stats']['cache_hit'] and s['stats']['prepare_seconds_this_call'] == 0 for s in samples)
    else:
        assert not any(e.get('operation') == 'target_partition' for e in events)
        assert samples[-1]['stats']['cumulative_accepted_samples'] == 2
        assert samples[-1]['stats']['cumulative_proposals'] == 2  # beta=0 accepts every draw.
        assert all(s['stats']['proposal_stats']['cache_hit'] for s in samples)


@pytest.mark.parametrize('backend', ['template_stream', 've_aligned', 'product_prefix', 'onset_reset'])
def test_complete_sampling_adapter_never_performs_a_nonempty_clamp(backend, monkeypatch):
    from tri.evaluation.shared_rhythm_study import make_solver, _sample_complete
    from tri.inference.exact import Budget
    case = make_shared_rhythm_case(L=2, D=3, K=1)
    engine = make_solver(case.spec, case.logq, backend, Budget(), 100)
    engine.log_partition()
    original = engine.log_partition
    def partition(evidence=None):
        assert not evidence, 'A full draw must not require a clamped partition'
        return original(evidence)
    monkeypatch.setattr(engine, 'log_partition', partition)
    for _ in range(2):
        tokens, draw = _sample_complete(engine, backend, np.random.default_rng(75))
        assert _validate_sample(tokens, case.spec, case.logq, draw=draw, logz=original())['valid']


@pytest.mark.parametrize('R', [3, 4])
def test_generic_multi_span_template_and_ve_partitions_agree(R):
    from itertools import product
    from scipy.special import logsumexp
    from tri.domain.music import verify_music
    from tri.evaluation.shared_rhythm_study import make_solver
    from tri.inference.exact import Budget
    case = make_shared_rhythm_case(R=R, L=2, D=3, K=1, beta=.02)
    a = make_solver(case.spec, case.logq, 'template_stream', Budget(), 100).log_partition()
    b = make_solver(case.spec, case.logq, 've_aligned', Budget(), 100).log_partition()
    assert a == pytest.approx(b, abs=1e-10)
    missing = [p for p in range(case.spec.length) if p not in case.spec.observed]
    vocabulary = (0, 1, *(p + 2 for p in case.spec.pitches))
    weights = []
    for values in product(vocabulary, repeat=len(missing)):
        tokens = [case.spec.observed.get(p, 0) for p in range(case.spec.length)]
        for p, token in zip(missing, values):
            tokens[p] = token
        checked = verify_music(tokens, case.spec)
        if checked.valid:
            weights.append(checked.soft_score + sum(case.logq[p, tokens[p]] for p in missing))
    assert a == pytest.approx(float(logsumexp(weights)), abs=1e-10)


@pytest.mark.parametrize('backend', ['product_prefix', 'onset_reset', 'onset_rejection_drop', 'onset_rejection_boundary'])
def test_independent_worker_saves_logs_and_original_checks(tmp_path, backend):
    payload = _payload(tmp_path, backend)
    result = isolated(payload, worker_seconds=15, rss_limit_mib=512, log_prefix=tmp_path / 'logs' / 'one')
    assert result['status'] == 'completed', result
    assert result['pid'] != os.getpid()
    assert result['verified_samples'] == 2
    assert result['seconds_to_first_verified_sample_excluding_checks'] > 0
    assert result['peak_rss_mib'] < 512
    assert result['measurements']['first_full_sample']['validated']
    assert (tmp_path / 'logs' / 'one.stdout.jsonl').read_text()
    assert (tmp_path / 'logs' / 'one.stderr.log').exists()
    if backend.startswith('onset_rejection_'):
        assert result['last_rejection_progress']['progress']['cumulative_accepted_samples'] == 2
        assert not result['last_rejection_progress']['progress']['in_flight']
        assert not result['target_normalizer_available']


@pytest.mark.parametrize('resource,status', [('time', 'timeout'), ('memory', 'rss_limit')])
def test_isolated_limits_are_real_terminal_records(tmp_path, resource, status):
    result = isolated(_payload(tmp_path), worker_seconds=.001 if resource == 'time' else 15,
                      rss_limit_mib=.001 if resource == 'memory' else 512,
                      log_prefix=tmp_path / 'logs' / resource)
    assert result['status'] == status
    assert result['failed_operation']


def test_external_kill_keeps_latest_completed_proposals_and_inflight(tmp_path, monkeypatch):
    import sys
    import tri.evaluation.shared_rhythm_study as module
    original = module.subprocess.Popen
    progress = {'event': 'rejection_progress', 'operation': 'first_full_sample',
                'progress': {'event': 'proposal_started', 'in_flight': True,
                             'cumulative_proposals': 7, 'cumulative_proposal_calls': 8,
                             'cumulative_accepted_samples': 0, 'cumulative_rejections': 7}}
    code = 'import json,time; print(' + repr(json.dumps(progress)) + ', flush=True); time.sleep(2)'
    def launch(*args, **kwargs):
        return original([sys.executable, '-u', '-c', code], **kwargs)
    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    result = module.isolated(_payload(tmp_path, 'onset_rejection_drop'), worker_seconds=.3,
                             rss_limit_mib=512, log_prefix=tmp_path / 'logs' / 'progress')
    assert result['status'] == 'timeout'
    saved = result['last_rejection_progress']['progress']
    assert saved['cumulative_proposals'] == 7
    assert saved['cumulative_proposal_calls'] == 8
    assert saved['in_flight']


def test_partition_comparison_never_compares_proposal_z_to_target_z():
    rows = [{'job_id': 'a', 'case_id': 'one', 'backend': 've_aligned', 'target_normalizer_available': True,
             'measurements': {'target_partition': {'log_z': -3.}}},
            {'job_id': 'b', 'case_id': 'one', 'backend': 'onset_rejection_drop', 'target_normalizer_available': False,
             'measurements': {'compile': {'stats': {'proposal_log_z': 900.}}}}]
    result = check_partitions(rows)
    assert result['target_z_records'] == 1
    assert not result['errors']
    rows[1]['measurements']['target_partition'] = {'log_z': 900.}
    with pytest.raises(AssertionError, match='Sampling-only'):
        check_partitions(rows)


def test_plan_resume_retains_all_failures_and_predeclared_unsupported(tmp_path, monkeypatch):
    import tri.evaluation.shared_rhythm_study as module
    calls = []
    def fake_isolated(payload, **kwargs):
        calls.append(payload)
        return {'status': 'timeout', 'measurements': {}, 'validations': {}, 'target_normalizer_available': False}
    monkeypatch.setattr(module, 'isolated', fake_isolated)
    config = _small_config()
    result = module.run(config, tmp_path)
    assert result['recorded'] == 4 * 6 * 3 == 72
    # product R3: 2 beta*3 =6; base reset beta>0: 2 R*3 =6.
    assert result['status_counts'] == {'unsupported': 12, 'timeout': 60}
    assert len(calls) == 60
    repeated = module.run(config, tmp_path, resume=True)
    assert repeated['status_counts'] == result['status_counts']
    assert len(calls) == 60
    with pytest.raises(ValueError, match='matching explicit'):
        module.run(config, tmp_path)
    with pytest.raises(ValueError, match='matching explicit'):
        module.run({**config, 'warm_samples': 3}, tmp_path, resume=True)


@pytest.mark.parametrize('failure', ['unexpected_error', 'different_z'])
def test_driver_stops_immediately_after_saving_unexpected_or_target_mismatch(tmp_path, monkeypatch, failure):
    import tri.evaluation.shared_rhythm_study as module
    config = _small_config()
    config.update(timing_repeats=1, backends=['template_stream', 've_aligned'])
    config['grid'] = {**config['grid'], 'R': [2], 'betas': [0.]}
    calls = []
    def fake_isolated(payload, **kwargs):
        calls.append(payload)
        return {'status': 'unexpected_error' if failure == 'unexpected_error' else 'completed',
                'measurements': {'target_partition': {'seconds': .01, 'log_z': -float(len(calls))}},
                'validations': {}, 'target_normalizer_available': True}
    monkeypatch.setattr(module, 'isolated', fake_isolated)
    with pytest.raises(RuntimeError, match='failure retained'):
        module.run(config, tmp_path)
    expected = 1 if failure == 'unexpected_error' else 2
    assert len(calls) == expected
    rows = [json.loads(line) for line in (tmp_path / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == expected
    assert rows[-1]['status'] == ('unexpected_error' if failure == 'unexpected_error' else 'correctness_failure')
    assert json.loads((tmp_path / 'status.json').read_text())['status'] == 'failed'
    assert (tmp_path / 'summary.json').exists()
