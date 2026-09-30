import json
import os
from pathlib import Path
import sys

import pytest

from tri.evaluation.shared_rhythm_cases import prepare_shared_rhythm_cases
import tri.evaluation.shared_rhythm_extension as module


def configuration(tmp_path):
    grid = {'R': [2, 3], 'L': 2, 'D': 3, 'K': 1, 'betas': [0., .02],
            'visibility': ['unknown'], 'q_seeds': [7]}
    prepare_shared_rhythm_cases(grid, tmp_path / 'inputs')
    return {'version': 1, 'protocol': module.PROTOCOL,
        'common': {'cpu': min(os.sched_getaffinity(0)), 'rss_limit_mib': 512, 'max_proposals': 1000,
                   'budget': {'max_factor_entries': 67108864, 'max_workspace_bytes': 536870912,
                              'max_oracle_assignments': 9223372036854775807}},
        'stages': {'synthetic': {'manifest': str(tmp_path / 'inputs' / 'cases.json'),
            'family': 'synthetic_control', 'expected_cases': 4,
            'backends': ['product_multi', 'product_prefix', 'onset_reset', 'onset_rejection_boundary'],
            'timing_repeats': 2, 'samples_per_worker': 3, 'worker_seconds': 10,
            'sample_seed': 811, 'order_seed': 91}}}


def payload(tmp_path, backend='product_multi', samples=3, beta=0.):
    config = configuration(tmp_path)
    snapshot, plan = module.prepare_plan(config, 'synthetic')
    job = next(j for j in plan['jobs'] if j['backend'] == backend and j['metadata']['beta'] == beta
               and (backend != 'product_prefix' or j['metadata']['R'] == 2))
    return {**job, **config['common'], 'draw_seeds': list(range(711, 711 + samples))}


def test_plan_preserves_frozen_inputs_and_explicit_seed_repeat_semantics(tmp_path):
    config = configuration(tmp_path)
    before = (tmp_path / 'inputs' / 'cases.json').read_bytes()
    snapshot, plan = module.prepare_plan(config, 'synthetic')
    assert plan['planned_rows'] == 32
    assert plan['predeclared_unsupported_rows'] == 8
    assert plan['planned_workers'] == 24
    assert plan['maximum_worker_budget_seconds'] == 240
    assert snapshot['settings']['expected_cases'] == 4
    assert before == (tmp_path / 'inputs' / 'cases.json').read_bytes()
    grouped = {}
    for job in plan['jobs']:
        assert len(set(job['draw_seeds'])) == 3
        previous = grouped.setdefault(job['case_id'], job['draw_seeds'])
        assert previous == job['draw_seeds']
    assert len({seed for seeds in grouped.values() for seed in seeds}) == 12


@pytest.mark.parametrize('backend', ['product_multi', 'product_prefix', 'onset_reset', 'onset_rejection_boundary'])
def test_worker_new_baseline_and_existing_adapters_verify_all_samples(tmp_path, backend):
    events = []
    module.worker(payload(tmp_path, backend), emit=events.append)
    assert events[-1]['status'] == 'completed', events
    checked = [event for event in events if event['event'] == 'validation']
    assert len(checked) == 3
    assert len({event['draw_seed'] for event in checked}) == 3
    if backend.startswith('onset_rejection_'):
        assert not any(e.get('operation') == 'target_partition' for e in events)
    else:
        assert sum(e['event'] == 'measurement' and e['operation'] == 'target_partition' for e in events) == 1


def test_32_draw_worker_constructs_once_and_keeps_individual_rng_seeds(tmp_path, monkeypatch):
    original = module.make_solver
    engines = []
    def factory(*args, **kwargs):
        engine = original(*args, **kwargs)
        engines.append(engine)
        return engine
    monkeypatch.setattr(module, 'make_solver', factory)
    events = []
    module.worker(payload(tmp_path, 'onset_reset', samples=32), emit=events.append)
    assert len(engines) == 1
    checked = [event for event in events if event['event'] == 'validation']
    assert len(checked) == 32
    assert [e['draw_seed'] for e in checked] == list(range(711, 743))
    samples = [e for e in events if e['event'] == 'measurement' and 'full_sample' in e['operation']]
    assert all(sample['stats']['cache_hit'] for sample in samples)


@pytest.mark.parametrize('backend', ['product_multi', 'onset_rejection_boundary'])
def test_isolated_child_and_log_protocol(tmp_path, backend):
    row = module.isolated(payload(tmp_path, backend, beta=.02), worker_seconds=10, rss_limit_mib=512,
                          log_prefix=tmp_path / 'logs' / backend)
    assert row['status'] == 'completed', row
    assert row['pid'] != os.getpid()
    assert row['verified_samples'] == row['requested_samples'] == 3
    assert not row['draws_truncated']
    assert Path(row['stdout_log']).exists()
    if backend.startswith('onset_rejection_'):
        assert row['last_rejection_progress']['progress']['cumulative_accepted_samples'] == 3


def test_parent_preserves_inflight_counts_on_external_deadline(tmp_path, monkeypatch):
    original = module.subprocess.Popen
    event = {'event': 'rejection_progress', 'operation': 'repeated_full_sample_4',
             'progress': {'in_flight': True, 'cumulative_proposals': 20, 'cumulative_proposal_calls': 21,
                          'cumulative_accepted_samples': 4}}
    code = 'import time; print(' + repr(json.dumps(event)) + ', flush=True); time.sleep(2)'
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **kw: original([sys.executable, '-u', '-c', code], **kw))
    row = module.isolated(payload(tmp_path, 'onset_rejection_boundary', samples=32),
        worker_seconds=.3, rss_limit_mib=512, log_prefix=tmp_path / 'logs' / 'deadline')
    assert row['status'] == 'timeout'
    assert row['draws_truncated']
    assert row['last_rejection_progress']['progress']['cumulative_proposals'] == 20
    assert row['last_rejection_progress']['progress']['in_flight']


def test_parent_rejects_bad_z_as_soon_as_measurement_arrives(tmp_path, monkeypatch):
    original = module.subprocess.Popen
    event = {'event': 'measurement', 'operation': 'target_partition', 'log_z': -2., 'seconds': .1}
    code = 'import time; print(' + repr(json.dumps(event)) + ', flush=True); time.sleep(2)'
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **kw: original([sys.executable, '-u', '-c', code], **kw))
    row = module.isolated({**payload(tmp_path), 'reference_target_log_z': -3.}, worker_seconds=10,
                          rss_limit_mib=512, log_prefix=tmp_path / 'logs' / 'bad_z')
    assert row['status'] == 'correctness_failure'
    assert row['failed_operation'] == 'cross_backend_target_partition'
    assert row['elapsed_worker_seconds'] < 2


def test_run_resume_preserves_terminal_failures_and_is_quiet(tmp_path, monkeypatch, capsys):
    config = configuration(tmp_path)
    calls = []
    def fake_worker(*args, **kwargs):
        calls.append(args[0])
        return {'status': 'timeout', 'measurements': {}, 'validations': {}}
    monkeypatch.setattr(module, 'isolated', fake_worker)
    out = tmp_path / 'run'
    first = module.run(config, 'synthetic', out)
    assert first['recorded'] == 32
    assert first['status_counts'] == {'timeout': 24, 'unsupported': 8}
    assert len(capsys.readouterr().out.splitlines()) == 2
    module.run(config, 'synthetic', out, resume=True)
    assert len(calls) == 24
    assert (out / 'summary.json').exists()
    with pytest.raises(ValueError, match='matching explicit'):
        module.run(config, 'synthetic', out)


def test_plan_only_does_not_start_any_worker(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    def unexpected(*args, **kwargs):
        raise AssertionError('Plan-only must never dispatch a worker')
    monkeypatch.setattr(module, 'isolated', unexpected)
    plan = module.run(config, 'synthetic', tmp_path / 'run', plan_only=True)
    assert plan['planned_rows'] == 32
    assert not (tmp_path / 'run' / 'results.jsonl').exists()


def test_wrong_family_or_request_count_is_rejected(tmp_path):
    config = configuration(tmp_path)
    config['stages']['synthetic']['family'] = 'real_validation_shared_rhythm'
    with pytest.raises(ValueError, match='provenance'):
        module.prepare_plan(config, 'synthetic')
    config['stages']['synthetic']['expected_cases'] = 24
    with pytest.raises(ValueError, match='unique request count'):
        module.prepare_plan(config, 'synthetic')
