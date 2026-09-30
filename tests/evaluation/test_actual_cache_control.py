from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

from tri.domain.music import CountRule, MusicSpec, verify_music
from tri.evaluation import actual_cache_control as control
from tri.evaluation.cohorts import file_identity, serialize_spec
from tri.inference.exact import Budget
from tri.runtime import read_jsonl
from tri.sampling.research_methods import ResearchResult, research_decode


def tiny_case():
    spec = MusicSpec(length=7, pitches=(60, 62), observed={0: 62, 3: 62, 6: 62},
                     fixed_soundings={0: 60, 3: 60, 6: 60}, equal_onsets=((1, 4), (2, 5)),
                     onset_counts=(CountRule((1, 2), 1), CountRule((4, 5), 1)), motion_cost=.1)
    def provider(state, noise):
        logits = np.full((7, 130), -3.)
        logits[:, 0] = sum(v is not None for v in state) * .1
        logits[:, 1] = noise * .3
        logits[:, 62] = .3
        logits[:, 64] = -.2 * noise
        return logits - logsumexp(logits, axis=1, keepdims=True)
    return spec, provider


def source_record(stage='short'):
    spec, _ = tiny_case()
    return {'stage': stage, 'case_id': stage + '-unknown', 'case_index': 1, 'kind': 'unknown',
            'work_id': '258' if stage == 'short' else '461', 'source_index': 4,
            'source_start_cell': 16, 'spec': serialize_spec(spec), 'seed': 12341,
            'steps': 4, 'original_replicate': 0, 'original_backend': 'paired',
            'originals': {method: {'request_id': stage + ':unknown', 'model_calls': 2,
                          'elapsed_decode_seconds': 999., 'raw_tokens': [62, 62, 1, 62, 62, 1, 62]}
                          for method in control.METHODS}}


def config_record(sources=None):
    sources = sources or [source_record()]
    return {'version': 1, 'device': 'fake', 'cpu': 6, 'sources': sources,
            'plan': control.make_plan(sources), 'budget': asdict(Budget())}


class FakeClock:
    def __init__(self):
        self.value = 0.
    def __call__(self):
        return self.value
    def advance(self, value):
        self.value += value


def test_fixed_plan_has_54_actual_jobs_with_unchanged_source_seeds_and_alternating_order():
    sources = [source_record(stage) for stage in control.STAGES]
    plan = control.make_plan(sources)
    assert plan == control.make_plan(sources)
    assert len(plan) == len({job['job_id'] for job in plan}) == 54
    assert Counter(job['backend'] for job in plan) == dict.fromkeys(control.BACKENDS, 27)
    for i in range(0, len(plan), 2):
        pair = plan[i:i+2]
        assert {job['backend'] for job in pair} == set(control.BACKENDS)
        assert len({(job['stage'], job['method'], job['timing_repeat']) for job in pair}) == 1
    for source in sources:
        for method in control.METHODS:
            jobs = [job for job in plan if job['stage'] == source['stage'] and job['method'] == method]
            assert {job['seed'] for job in jobs} == {source['seed']}
            orders = [[job['backend'] for job in jobs if job['timing_repeat'] == repeat] for repeat in range(3)]
            assert orders[0] == orders[1][::-1] == orders[2]


def test_provider_preserves_q_identity_and_inputs_and_times_one_call_once():
    clock = FakeClock()
    q = np.arange(130)
    observed = []
    state = (62, None)
    def provider(tokens, noise):
        observed.append((tokens, noise))
        clock.advance(.25)
        return q
    def sync():
        clock.advance(.1)
    wrapped = control.TimedProvider(provider, sync, clock)
    assert wrapped(state, .75) is q
    assert observed == [(state, .75)]
    assert observed[0][0] is state
    assert wrapped.model_calls == 1
    assert wrapped.provider_seconds == pytest.approx(.35)
    assert clock() == pytest.approx(.45)
    assert not hasattr(wrapped, 'arrays')


def test_measurement_excludes_external_checker_and_preserves_provider_subset_of_wall():
    source = source_record()
    job = control.make_plan([source])[0]
    spec, _ = tiny_case()
    clock = FakeClock()
    def sync():
        clock.advance(.1)
    def provider(state, noise):
        clock.advance(.25)
        return 'unaltered q'
    def decoder(method, actual_spec, actual_provider, *, steps, seed, backend, budget):
        assert actual_spec is spec
        assert (method, steps, seed, backend) == (job['method'], job['steps'], source['seed'], job['backend'])
        assert budget == Budget()
        clock.advance(.5)
        assert actual_provider((None,), 1.) == 'unaltered q'
        clock.advance(.5)
        actual_provider((62,), .5)
        clock.advance(.5)
        return ResearchResult((62, 62, 1, 62, 62, 1, 62), 2)
    def checker(tokens, actual_spec):
        clock.advance(100.)
        return verify_music(tokens, actual_spec)
    row = control.measure_decode(source, job, spec, provider, synchronize=sync,
                                 decoder=decoder, checker=checker, clock=clock)
    assert row['status'] == 'completed'
    assert row['decode_wall_seconds'] == pytest.approx(2.5)
    assert row['provider_seconds'] == pytest.approx(.7)
    assert row['non_provider_wall_seconds'] == pytest.approx(1.8)
    assert row['external_verification_seconds'] == pytest.approx(100.)
    assert row['model_calls'] == row['measured_model_calls'] == 2
    assert row['original_tokens_exact_match']


@pytest.mark.parametrize('method', control.METHODS)
@pytest.mark.parametrize('backend', control.BACKENDS)
def test_lightweight_control_matches_unwrapped_actual_small_cpu_decode(method, backend):
    spec, provider = tiny_case()
    source = source_record()
    job = next(job for job in control.make_plan([source]) if job['method'] == method and job['backend'] == backend)
    expected = research_decode(method, spec, provider, steps=job['steps'], seed=job['seed'], backend=backend, budget=Budget())
    row = control.measure_decode(source, job, spec, provider, synchronize=lambda: None)
    assert row['status'] == 'completed'
    assert row['raw_tokens'] == list(expected.tokens)
    assert row['model_calls'] == row['measured_model_calls'] == expected.model_calls
    assert row['decode_wall_seconds'] >= row['provider_seconds'] > 0
    assert row['verification']['valid']


def fake_runtime(clock, *, different_backends=False, fail_at=None, wrong_accounting=False):
    calls = []
    loaded = []
    spec, _ = tiny_case()
    def provider(state, noise):
        clock.advance(.125)
        return None
    def loader(source, device):
        loaded.append(source['stage'])
        clock.advance(1000.)
        return spec, provider
    def decoder(method, spec, provider, *, steps, seed, backend, budget):
        calls.append((method, seed, backend))
        clock.advance(2. if backend == 'product_reuse' else 1.)
        provider((None,) * 7, 1.)
        if fail_at is not None and len(calls) == fail_at:
            raise RuntimeError('deliberate failure')
        pitch = 64 if different_backends and backend == 'product_prefix' else 62
        return ResearchResult((62, pitch, 1, 62, pitch, 1, 62), 55 if wrong_accounting else 1)
    return loader, decoder, calls, loaded


def test_driver_loads_and_warms_each_stage_once_and_keeps_different_outputs(tmp_path):
    config = config_record([source_record(stage) for stage in control.STAGES])
    clock = FakeClock()
    loader, decoder, calls, loaded = fake_runtime(clock, different_backends=True)
    report = control._execute(config, tmp_path, stage_loader=loader, synchronize=lambda: None,
                              decoder=decoder, clock=clock)
    assert report['status'] == 'completed'
    assert len(calls) == 54
    assert loaded == list(control.STAGES)
    assert report['unique_frozen_requests'] == 3 and report['unique_works'] == 2
    assert report['complete_stage_method_comparisons'] == report['planned_stage_method_comparisons'] == 9
    rows = read_jsonl(tmp_path / 'results.jsonl')
    assert all(row['warmup_seconds_excluded'] == .125 for row in rows)
    assert all(row['decode_wall_seconds'] < 3 for row in rows)  # excludes model load/warmup
    for comparison in report['comparisons']:
        assert comparison['decode_speedup_product_reuse_over_product_prefix'] == pytest.approx(2.125 / 1.125)
        assert all(not pair['tokens_exact_match'] and pair['differing_positions'] == [1, 4]
                   for pair in comparison['token_pairs'])
        assert all(group['tokens_identical_across_timing_repeats']
                   for group in comparison['backends'].values())
    before = (tmp_path / 'results.jsonl').read_bytes()
    report_again = control._execute(config, tmp_path, stage_loader=loader, synchronize=lambda: None,
                                    decoder=decoder, clock=clock, resume=True)
    assert report_again == report
    assert len(calls) == 54 and len(loaded) == 3
    assert (tmp_path / 'results.jsonl').read_bytes() == before
    with pytest.raises(ValueError, match='already exists'):
        control._execute(config, tmp_path, stage_loader=loader, synchronize=lambda: None)
    altered = deepcopy(config)
    altered['plan'][0]['seed'] += 1
    with pytest.raises(ValueError, match='configuration'):
        control._execute(altered, tmp_path, stage_loader=loader, synchronize=lambda: None, resume=True)


def test_failed_row_is_durable_and_resume_skips_it_without_retry(tmp_path):
    config = config_record()
    clock = FakeClock()
    loader, decoder, calls, loaded = fake_runtime(clock, fail_at=2)
    with pytest.raises(RuntimeError, match='Recorded decode_error'):
        control._execute(config, tmp_path, stage_loader=loader, synchronize=lambda: None,
                         decoder=decoder, clock=clock)
    first_bytes = (tmp_path / 'results.jsonl').read_bytes()
    first_rows = read_jsonl(tmp_path / 'results.jsonl')
    assert len(first_rows) == 2 and first_rows[-1]['status'] == 'decode_error'
    assert first_rows[-1]['error']['message'] == 'deliberate failure'
    assert first_rows[-1]['measured_model_calls'] == 1
    assert json.loads((tmp_path / 'status.json').read_text())['status'] == 'failed'
    # Simulated interrupted final append is recoverable; completed failure rows survive.
    with (tmp_path / 'results.jsonl').open('ab') as stream:
        stream.write(b'{"interrupted_tail":')
    report = control._execute(config, tmp_path, stage_loader=loader, synchronize=lambda: None,
                              decoder=decoder, clock=clock, resume=True)
    assert len(calls) == 18  # failed row not rerun
    assert loaded == ['short', 'short']  # resumed stage rewarms
    assert (tmp_path / 'results.jsonl').read_bytes().startswith(first_bytes)
    assert report['status'] == 'completed_with_failures'
    assert report['failed_decodes'] == 1 and report['successful_decodes'] == 17
    assert report['complete_stage_method_comparisons'] == 2
    affected = next(c for c in report['comparisons'] if c['method'] == first_rows[-1]['method'])
    assert affected['decode_speedup_product_reuse_over_product_prefix'] is None


@pytest.mark.parametrize('problem', ['calls', 'checker', 'invalid'])
def test_checker_and_accounting_failures_are_preserved_after_decode_timing(problem):
    source = source_record()
    spec, _ = tiny_case()
    job = control.make_plan([source])[0]
    clock = FakeClock()
    _, decoder, _, _ = fake_runtime(clock, wrong_accounting=problem == 'calls')
    def checker(tokens, spec):
        clock.advance(200.)
        if problem == 'checker':
            raise ValueError('checker crashed')
        return verify_music((1,) * spec.length if problem == 'invalid' else tokens, spec)
    row = control.measure_decode(source, job, spec, lambda *args: None, synchronize=lambda: None,
                                 decoder=decoder, checker=checker, clock=clock)
    assert row['status'] == 'correctness_failure'
    assert row['decode_wall_seconds'] < 3
    assert row['external_verification_seconds'] == 200.
    assert 'raw_tokens' in row


def test_summary_uses_three_repeat_medians_and_keeps_planned_missing_denominators():
    config = config_record([source_record(stage) for stage in control.STAGES])
    rows = []
    for job in config['plan']:
        if job['stage'] != 'short' or job['method'] != 'one_shot_joint':
            continue
        duration = ([1., 100., 3.] if job['backend'] == 'product_reuse' else [2., 4., 20.])[job['timing_repeat']]
        rows.append({**job, 'status': 'completed', 'raw_tokens': [62], 'model_calls': 1,
                     'decode_wall_seconds': duration, 'provider_seconds': .1,
                     'non_provider_wall_seconds': duration - .1, 'external_verification_seconds': .01})
    report = control.summarize(config, rows)
    assert report['planned_decodes'] == 54 and report['recorded_decodes'] == 6
    assert report['planned_stage_method_comparisons'] == 9
    assert report['complete_stage_method_comparisons'] == 1
    comparison = next(c for c in report['comparisons'] if c['stage'] == 'short' and c['method'] == 'one_shot_joint')
    assert comparison['decode_speedup_product_reuse_over_product_prefix'] == .75
    assert report['geomean_decode_speedup_on_complete_comparisons'] == pytest.approx(.75)
    rows.pop()
    report = control.summarize(config, rows)
    assert report['complete_stage_method_comparisons'] == 0
    assert report['geomean_decode_speedup_on_complete_comparisons'] is None


def test_source_selects_first_unknown_and_original_replicate_zero(tmp_path):
    evaluation = tmp_path / 'short' / 'evaluation'
    evaluation.mkdir(parents=True)
    paths = {}
    for name in ('dataset', 'chords', 'checkpoint'):
        path = tmp_path / name
        path.write_text('frozen')
        paths[name] = file_identity(path)
    source = source_record()
    case = {key: source[key] for key in ('case_id', 'kind', 'work_id', 'source_index', 'source_start_cell', 'spec')}
    cohort_path = tmp_path / 'cohort.json'
    cohort_path.write_text(json.dumps({'options': {'dataset': paths['dataset'], 'chords': paths['chords']},
                                     'requests': [{**case, 'kind': 'known'}, case, {**case, 'case_id': 'later-unknown'}]}))
    config = {'cohort': file_identity(cohort_path), 'checkpoint': paths['checkpoint'],
              'seed': 342, 'steps': 8, 'backend': 'paired'}
    (evaluation / 'config.json').write_text(json.dumps(config))
    originals = [{**source['originals'][method], 'cohort_case_id': source['case_id'],
                  'method': method, 'replicate': repeat} for method in control.METHODS for repeat in (0, 1)]
    (evaluation / 'results.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in originals))
    actual = control.load_source(tmp_path, 'short')
    assert actual['case_id'] == source['case_id'] and actual['case_index'] == 1
    assert actual['seed'] == 342 + 10000 and actual['original_replicate'] == 0
    assert actual['originals'] == source['originals']
    Path(paths['checkpoint']['path']).write_text('changed checkpoint')
    with pytest.raises(ValueError, match='checkpoint'):
        control.load_source(tmp_path, 'short')


def test_formal_entry_refuses_cpu_sibling_or_fallback_without_initializing_gpu():
    for cpu, device in ((3, 'cuda:0'), (6, 'cuda:0'), (2, 'cpu'), (4, 'cuda:1')):
        with pytest.raises(ValueError, match='requires CPU 2 or 4'):
            control.run_control('unused', 'unused', cpu=cpu, device=device)
