"""Probability and isolation checks for the same-target solver experiment."""
from copy import deepcopy
from dataclasses import replace
from itertools import product
import json
import math
import os
import sys

import numpy as np
import pytest
from scipy.special import logsumexp

from tri.domain.music import verify_music
from tri.evaluation import solver_study as study
from tri.evaluation.solver_cases import build_tiny_cases, save_cases
from tri.inference.exact import BatchSample


BUDGET = {'max_factor_entries': 2_000_000, 'max_workspace_bytes': 128 * 2**20,
          'max_oracle_assignments': 1_000_000}


def original_token_weights(case, evidence=None):
    spec = case.spec
    free = [i for i in range(spec.length) if i not in spec.observed]
    weights = {}
    for assignment in product((0, 1, *(p + 2 for p in spec.pitches)), repeat=len(free)):
        tokens = [spec.observed.get(i) for i in range(spec.length)]
        for index, token in zip(free, assignment):
            tokens[index] = token
        if any(tokens[int(name[1:])] != value for name, value in (evidence or {}).items()):
            continue
        checked = verify_music(tokens, spec)
        if checked.valid:
            weights[tuple(tokens)] = checked.soft_score + sum(case.logq[i, tokens[i]] for i in free)
    return weights


def payload_for(case, directory, backend='product_chain'):
    manifest = save_cases([case], directory)
    return {'path': manifest['cases'][0]['path'], 'backend': backend,
            'budget': dict(BUDGET), 'sample_seed': 129, 'warm_samples': 2}


@pytest.mark.parametrize('backend', (*study.BASELINES, 'paired_reuse', 'paired_sparse', 'product_reuse', 'template_stream'))
def test_worker_tiny_probabilities_clamps_and_marginal_against_original_tokens(tmp_path, capsys, backend):
    case = build_tiny_cases()[0]
    study.worker(payload_for(case, tmp_path, backend))
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]['status'] == 'completed', events[-1]
    measurements = {e['operation']: e for e in events if e['event'] == 'measurement'}
    starts = [e['operation'] for e in events if e['event'] == 'start']
    assert starts[-1] == 'token_marginal'  # costly marginals must not censor other APIs
    assert set(measurements) == {'cold_partition', 'cold_full_sample', 'warm_full_samples',
                                 'query_setup', 'clamp_queries', 'repeated_identical_clamps', 'token_marginal'}
    weights = original_token_weights(case)
    expected_z = float(logsumexp(list(weights.values())))
    assert measurements['cold_partition']['log_z'] == pytest.approx(expected_z, abs=1e-10)
    sample = measurements['cold_full_sample']['draw']
    assert tuple(sample['tokens']) in weights
    assert sample['log_probability_error'] < 1e-10
    assert measurements['warm_full_samples']['samples'] == 2
    for operation in ('clamp_queries', 'repeated_identical_clamps'):
        for evidence, actual in zip(measurements[operation]['evidence'], measurements[operation]['log_z']):
            expected = logsumexp(list(original_token_weights(case, evidence).values()))
            assert float(actual) == pytest.approx(expected, abs=1e-10)
    assert measurements['query_setup']['log_z'] == pytest.approx(expected_z, abs=1e-10)
    assert measurements['clamp_queries']['evidence'] == [{'y1': 62}, {'y4': 62}, {'y1': 62, 'y4': 62}]
    assert all(not probe['cache_hit'] for probe in measurements['clamp_queries']['probe_stats'])
    # Compact VE keeps one evidence graph; the cyclic three-probe sequence
    # legitimately evicts it. Other baselines retain several scalar partitions.
    expected_hit = backend not in ('ve_aligned', 've_minfill')
    assert all(probe['cache_hit'] == expected_hit
               for probe in measurements['repeated_identical_clamps']['probe_stats'])
    marginal = measurements['token_marginal']
    pos = int(marginal['variable'][1:])
    expected = [logsumexp([mass for seq, mass in weights.items() if seq[pos] == token]) - expected_z
                for token in marginal['domain']]
    np.testing.assert_allclose(np.array(marginal['log_probs'], dtype=float), expected, atol=1e-10, rtol=0)
    assert all(e['seconds'] >= 0 for e in measurements.values())


def test_worker_zero_mass_and_internal_budget_are_distinct(tmp_path, capsys):
    case = build_tiny_cases()[4]
    study.worker(payload_for(case, tmp_path / 'zero'))
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]['status'] == 'zero_mass'
    completed = [e for e in events if e['event'] == 'measurement']
    assert len(completed) == 1
    assert completed[0]['operation'] == 'cold_partition'
    assert float(completed[0]['log_z']) == -math.inf
    payload = payload_for(build_tiny_cases()[0], tmp_path / 'budget')
    payload['budget']['max_factor_entries'] = 1
    study.worker(payload)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]['status'] == 'internal_budget'
    assert events[-1]['failed_operation'] == 'cold_partition'


def test_only_verified_dense_ve_rank_limit_is_classified_as_capacity():
    error = ValueError('maximum supported dimension for an ndarray is 32, found 33')
    assert study.failure_status(error, 've') == 'implementation_limit'
    assert study.failure_status(error, 'product_chain') == 'unexpected_error'
    assert study.failure_status(ValueError('different array bug'), 've') == 'unexpected_error'
    assert study.failure_status(AssertionError(str(error)), 've') == 'unexpected_error'


@pytest.mark.parametrize('backend', ('ve_aligned', 've_minfill', 'product_chain',
                                    'product_reuse', 'paired_reuse', 'paired_sparse', 'template_stream'))
def test_public_factory_exposes_exact_study_backends(backend):
    from tri.inference.music_backends import make_music_engine
    from tri.inference.exact import Budget
    case = build_tiny_cases()[0]
    engine = make_music_engine(case.spec, case.logq, backend, Budget(**BUDGET))
    expected = float(logsumexp(list(original_token_weights(case).values())))
    assert engine.log_partition() == pytest.approx(expected, abs=1e-10)


def test_full_sample_checker_rejects_incorrect_probability_and_broken_relation():
    case = build_tiny_cases()[0]
    weights = original_token_weights(case)
    z = float(logsumexp(list(weights.values())))
    tokens = next(iter(weights))
    assignment = {f'y{i}': token for i, token in enumerate(tokens)}
    valid = BatchSample(assignment, weights[tokens] - z, weights[tokens])
    assert study._check_draw(valid, case.spec, case.logq, z)['valid']
    with pytest.raises(AssertionError, match='log probability'):
        study._check_draw(replace(valid, log_probability=valid.log_probability + .5), case.spec, case.logq, z)
    altered = dict(assignment)
    left, right = case.spec.equal_onsets[0]
    altered[f'y{left}'] = 62 if altered[f'y{right}'] < 2 else 0
    with pytest.raises(AssertionError, match='Invalid joint sample'):
        study._check_draw(replace(valid, assignment=altered), case.spec, case.logq, z)


def stub_worker(monkeypatch, script):
    original_popen = study.subprocess.Popen

    def launch(command, **kwargs):
        return original_popen([sys.executable, '-c', script], **kwargs)

    monkeypatch.setattr(study.subprocess, 'Popen', launch)


STUB_HEADER = '''import json,os,sys,time

def emit(event):
    print(json.dumps(event),flush=True)
emit({'event':'ready','baseline_rss_mib':10.,'pid':os.getpid()})
'''
STUB_PAYLOAD = {'backend': 'probe', 'sample_seed': 9}


def test_isolated_timeout_retains_finished_measurements_and_thread_settings(monkeypatch):
    script = STUB_HEADER + '''emit({'event':'start','operation':'cold_partition'})
emit({'event':'measurement','operation':'cold_partition','seconds':.001,'log_z':-7.,
      'threads':[os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']],
      'peak_rss_mib':12.})
emit({'event':'start','operation':'cold_full_sample'})
time.sleep(2)
'''
    stub_worker(monkeypatch, script)
    row = study.isolated(STUB_PAYLOAD, operation_seconds=.2, rss_limit_mib=128, cpu=None)
    assert row['status'] == 'timeout'
    assert row['failed_operation'] == 'cold_full_sample'
    assert row['measurements']['cold_partition']['log_z'] == -7.
    assert row['measurements']['cold_partition']['threads'] == ['1', '1', '1']
    assert row['elapsed_worker_seconds'] < 1.5


def test_isolated_enforces_reported_peak_rss_even_if_worker_finishes_immediately(monkeypatch):
    script = STUB_HEADER + '''emit({'event':'start','operation':'cold_partition'})
emit({'event':'measurement','operation':'cold_partition','seconds':.001,'log_z':-7.,'peak_rss_mib':600.})
emit({'event':'done','status':'completed','peak_rss_mib':600.})
'''
    stub_worker(monkeypatch, script)
    row = study.isolated(STUB_PAYLOAD, operation_seconds=1, rss_limit_mib=128, cpu=None)
    assert row['status'] == 'rss_limit'
    assert row['peak_rss_mib'] >= 600.
    assert row['measurements']['cold_partition']['log_z'] == -7.


def test_isolated_rss_attribution_survives_next_start_in_same_pipe_buffer(monkeypatch):
    script = STUB_HEADER + '''events = [
    {'event':'start','operation':'cold_partition'},
    {'event':'measurement','operation':'cold_partition','seconds':.001,
     'log_z':-7.,'peak_rss_mib':600.},
    {'event':'start','operation':'cold_full_sample'},
    {'event':'done','status':'completed','peak_rss_mib':600.},
]
packet = ('\\n'.join(json.dumps(event) for event in events) + '\\n').encode()
assert len(packet) <= os.fpathconf(1, 'PC_PIPE_BUF')
os.write(1, packet)
'''
    stub_worker(monkeypatch, script)
    row = study.isolated(STUB_PAYLOAD, operation_seconds=1, rss_limit_mib=128, cpu=None)
    assert row['status'] == 'rss_limit'
    assert row['failed_operation'] == 'cold_partition'
    assert row['peak_rss_mib'] >= 600.
    assert row['measurements']['cold_partition']['log_z'] == -7.


def test_isolated_stderr_cannot_block_otherwise_successful_worker(monkeypatch):
    script = STUB_HEADER + '''emit({'event':'start','operation':'cold_partition'})
os.write(2,b'x'*262144)
emit({'event':'measurement','operation':'cold_partition','seconds':.001,'log_z':-7.,'peak_rss_mib':12.})
emit({'event':'done','status':'completed','peak_rss_mib':12.})
'''
    stub_worker(monkeypatch, script)
    row = study.isolated(STUB_PAYLOAD, operation_seconds=1, rss_limit_mib=128, cpu=None)
    assert row['status'] == 'completed'
    assert row['measurements']['cold_partition']['log_z'] == -7.
    assert len(row['stderr']) <= 4000


def comparison_row(backend='paired', logz=-10., clamp=-12., marginal=(math.log(.4), math.log(.6))):
    return {'case_id': 'test', 'backend': backend, 'metadata': {'expected_support': 'feasible'},
            'measurements': {'cold_partition': {'log_z': logz},
                'clamp_queries': {'evidence': [{'y1': 62}], 'log_z': [clamp]},
                'token_marginal': {'variable': 'y2', 'domain': [0, 62], 'log_probs': list(marginal)}}}


def test_comparison_detects_wrong_base_clamp_marginal_and_changed_query_identity():
    first = comparison_row()
    same = comparison_row('product_chain')
    assert study.compare_rows([first, same]) == []
    mutations = []
    altered = deepcopy(same)
    altered['measurements']['cold_partition']['log_z'] -= .1
    mutations.append(altered)
    altered = deepcopy(same)
    altered['measurements']['clamp_queries']['log_z'][0] -= .1
    mutations.append(altered)
    altered = deepcopy(same)
    altered['measurements']['token_marginal']['log_probs'] = [math.log(.5), math.log(.5)]
    mutations.append(altered)
    altered = deepcopy(same)
    altered['measurements']['clamp_queries']['evidence'] = [{'y1': 0}]
    mutations.append(altered)
    altered = deepcopy(same)
    altered['measurements']['token_marginal']['variable'] = 'y3'
    mutations.append(altered)
    altered = deepcopy(same)
    altered['measurements']['token_marginal']['domain'].reverse()
    mutations.append(altered)
    for altered in mutations:
        assert study.compare_rows([first, altered]), altered


def test_real_isolated_tiny_case_and_resume_do_not_repeat_rows(tmp_path, monkeypatch):
    case = build_tiny_cases()[0]
    manifest = save_cases([case], tmp_path / 'cases')
    config = {'seed': 129, 'backends': ['product_chain'], 'cpu': min(os.sched_getaffinity(0)),
              'operation_seconds': 10, 'rss_limit_mib': 512, 'warm_samples': 2, 'budget': BUDGET}
    report = study.run(tmp_path / 'cases' / 'cases.json', tmp_path / 'results', config)
    assert report['status'] == 'completed'
    rows_file = tmp_path / 'results' / 'results.jsonl'
    rows = [json.loads(line) for line in rows_file.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['status'] == 'completed'
    assert rows[0]['peak_rss_mib'] >= rows[0]['baseline_rss_mib'] > 0
    assert rows[0]['incremental_peak_rss_mib'] >= 0
    assert rows[0]['measurements']['cold_full_sample']['draw']['valid']
    original = rows_file.read_bytes()

    def should_not_run(*args, **kwargs):
        raise AssertionError('matching resume must reuse completed measurements')

    monkeypatch.setattr(study, 'isolated', should_not_run)
    assert study.run(tmp_path / 'cases' / 'cases.json', tmp_path / 'results', config, resume=True)['status'] == 'completed'
    assert rows_file.read_bytes() == original
    changed = {**config, 'warm_samples': 3}
    with pytest.raises(ValueError, match='matching'):
        study.run(tmp_path / 'cases' / 'cases.json', tmp_path / 'results', changed, resume=True)


@pytest.mark.parametrize('fault', ['clamp', 'marginal'])
def test_worker_rejects_nan_probability_queries_instead_of_serializing_as_infinity(tmp_path, capsys, monkeypatch, fault):
    original_factory = study.make_solver

    class NonFiniteQuery:
        def __init__(self, engine):
            self.engine = engine

        def __getattr__(self, name):
            return getattr(self.engine, name)

        def log_partition(self, evidence=None):
            if fault == 'clamp' and evidence:
                return float('nan')
            return self.engine.log_partition(evidence)

        def marginal_log_probs(self, variable, evidence=None):
            if fault == 'marginal':
                return np.full(len(self.graph.domains[variable]), np.nan)
            return self.engine.marginal_log_probs(variable, evidence)

    monkeypatch.setattr(study, 'make_solver', lambda *a, **k: NonFiniteQuery(original_factory(*a, **k)))
    study.worker(payload_for(build_tiny_cases()[0], tmp_path))
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]['status'] == 'unexpected_error'
    assert events[-1]['failed_operation'] == ('clamp_queries' if fault == 'clamp' else 'token_marginal')


def test_full_sample_checker_validates_clamped_partition_field_too():
    case = build_tiny_cases()[0]
    weights = original_token_weights(case)
    z = float(logsumexp(list(weights.values())))
    tokens = next(iter(weights))
    assignment = {f'y{i}': token for i, token in enumerate(tokens)}
    wrong = BatchSample(assignment, weights[tokens] - z, weights[tokens] + .5)
    with pytest.raises(AssertionError):
        study._check_draw(wrong, case.spec, case.logq, z)
