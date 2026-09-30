import json

import pytest

from tri.evaluation.shared_rhythm_extension_delivery import (
    build_delivery, read_snapshot, request_detail, summarize_stage,
)
from tri.runtime import atomic_json


def job(backend='product_multi', *, repeat=0, case='case_a', draws=3, wide=False):
    return {'job_id': f'{case}_{backend}_{repeat}', 'case_id': case, 'backend': backend,
        'repeat': repeat, 'draw_seeds': list(range(draws)),
        'metadata': {'R': 3, 'L': 16, 'D': 3, 'K': 1, 'beta': .02, 'visibility': 'unknown',
            'work_id': 'work1', 'q_pairing_key': 'same_source_q',
            **({'wide_pitch_bounds': [36, 96]} if wide else {})}}


def record(job, *, warm=True):
    measurements = {'compile': {'seconds': .002},
                    'target_partition': {'seconds': .003, 'log_z': -12.},
                    'first_full_sample': {'seconds': .010}}
    validations = {'first_full_sample': {'valid': True}}
    if warm:
        for i in range(1, len(job['draw_seeds'])):
            measurements[f'repeated_full_sample_{i}'] = {'seconds': .010 + i * .010}
            validations[f'repeated_full_sample_{i}'] = {'valid': True}
    if job['backend'].startswith('onset_rejection_'):
        measurements.pop('target_partition')
    return {**job, 'status': 'completed', 'measurements': measurements,
            'validations': validations, 'target_normalizer_available': not job['backend'].startswith('onset_rejection_')}


def settings(repeats=1):
    return {'timing_repeats': repeats, 'worker_seconds': 30}


def test_timeout_retains_partial_draw_statistics_without_inventing_tail_times():
    j = job(draws=32)
    row = record(j, warm=False)
    row['measurements']['repeated_full_sample_1'] = {'seconds': .02}
    row['validations']['repeated_full_sample_1'] = {'valid': True}
    row.update(status='timeout', failed_operation='repeated_full_sample_2', elapsed_worker_seconds=30.01)
    detail = request_detail(j, row, settings())
    assert detail['cold_solver_seconds'] == pytest.approx(.015)
    assert detail['verified_draws'] == 2 and detail['not_verified_draws'] == 30
    assert detail['observed_warm_draws']['mean'] == detail['observed_warm_draws']['p95'] == .02
    assert detail['observed_warm_draws']['available'] == 1
    assert detail['full_warm_mean_seconds'] is None and detail['full_workload_solver_seconds'] is None
    assert detail['worker_completion_right_censored']
    assert detail['draws'][2]['seconds'] is None


def test_completed_measurement_over_rss_budget_is_not_counted_as_available():
    j = job()
    row = record(j)
    row.update(status='rss_limit', failed_operation='first_full_sample')
    detail = request_detail(j, row, settings())
    assert detail['cold_solver_seconds'] is None
    assert detail['first_draw_seconds'] is None
    assert not detail['worker_completion_right_censored']
    assert detail['resource_budget_failure']
    assert detail['verified_draws'] == 0
    assert detail['raw_verified_draws'] == 3
    assert detail['full_warm_mean_seconds'] is None


@pytest.mark.parametrize('failed_operation,verified', [('compile', 0), ('target_partition', 0),
                                                      ('repeated_full_sample_2', 2)])
def test_budget_failure_suppresses_later_buffered_results_but_keeps_raw_counts(failed_operation, verified):
    j = job(draws=4)
    row = record(j)
    row.update(status='rss_limit', failed_operation=failed_operation)
    detail = request_detail(j, row, settings())
    assert detail['verified_draws'] == verified
    assert detail['raw_returned_draws'] == detail['raw_verified_draws'] == 4
    assert detail['budget_excluded_raw_verified_draws'] == 4 - verified
    assert detail['full_warm_mean_seconds'] is None
    assert detail['full_workload_solver_seconds'] is None
    if verified:
        assert detail['cold_solver_seconds'] == pytest.approx(.015)
        assert detail['observed_warm_draws']['available'] == 1
    else:
        assert detail['cold_solver_seconds'] is None
        assert detail['observed_warm_draws']['available'] == 0


def test_matched_comparisons_require_every_repeat_and_preserve_failure_denominators():
    jobs = [job(backend, repeat=i) for backend in ('product_multi', 'onset_reset') for i in (0, 1)]
    rows = [record(j) for j in jobs]
    rows[-1].update(status='internal_budget', failed_operation='first_full_sample')
    report = summarize_stage({'stage': 'synthetic', 'settings': settings(2)}, {'jobs': jobs}, rows)
    pairs = [p for p in report['matched_within_stage'] if p['metric'] == 'cold_solver_seconds']
    assert len(pairs) == 1 and pairs[0]['ratio_reference_over_candidate'] is None
    assert pairs[0]['reference_seconds'] is None  # alphabetically onset_reset
    assert pairs[0]['candidate_seconds'] == pytest.approx(.015)
    group = next(g for g in report['groups'] if g['backend'] == 'onset_reset')
    assert group['cold_solver_seconds']['available'] == 0
    assert group['cold_solver_seconds']['planned'] == 1
    assert group['status_counts'] == {'completed': 1, 'internal_budget': 1}


def test_early_rejection_progress_reports_rhythm_attempts_not_full_paths():
    j = job('onset_rejection_visible_early', draws=32)
    row = record(j, warm=False)
    row.update(status='timeout', failed_operation='repeated_full_sample_1',
        stats={'cumulative_proposals': 2, 'cumulative_accepted_samples': 1, 'proposal_log_z': -11.},
        last_rejection_progress={'operation': 'repeated_full_sample_1', 'progress': {
            'proposal_unit': 'shared_rhythm_attempt_with_completed_accept_or_reject_decision',
            'in_flight': True, 'cumulative_proposals': 6, 'cumulative_proposal_calls': 7,
            'cumulative_accepted_samples': 2, 'cumulative_full_candidates': 2,
            'cumulative_partial_candidates': 4, 'cumulative_spans_sampled': 15}})
    detail = request_detail(j, row, settings())
    assert detail['completed_attempts'] == 6 and detail['proposal_calls'] == 7
    assert detail['full_candidates'] == 2 and detail['partial_candidates'] == 4
    assert detail['spans_sampled'] == 15
    assert detail['accepted_per_completed_attempt'] == pytest.approx(2 / 6)
    assert detail['accepted_decisions'] == 2 and detail['verified_draws'] == 1
    assert detail['proposal_in_flight']
    assert detail['proposal_unit'].startswith('shared_rhythm_attempt')


def test_wide_and_narrow_are_distinct_targets_but_share_one_q_input():
    jobs = [job(case='narrow'), job(case='narrow_wide', wide=True)]
    report = summarize_stage({'stage': 'long', 'settings': settings()}, {'jobs': jobs}, [record(j) for j in jobs])
    assert report['target_cases'] == 2 and report['q_input_count'] == report['work_count'] == 1
    assert {g['target_variant'] for g in report['groups']} == {'wide', 'narrow'}


def test_reading_active_jsonl_tail_does_not_repair_or_modify_source(tmp_path):
    path = tmp_path / 'results.jsonl'
    content = b'{"a": 1}\n{"a":'
    path.write_bytes(content)
    assert read_snapshot(path) == ([{'a': 1}], True)
    assert path.read_bytes() == content
    path.write_bytes(b'corrupt\n')
    with pytest.raises(ValueError, match='Corrupt complete'):
        read_snapshot(path)


def test_delivery_refreshes_from_raw_rows_even_when_cached_summary_lags(tmp_path):
    stage = tmp_path / 'study' / 'synthetic'
    stage.mkdir(parents=True)
    j = job()
    atomic_json(stage / 'config.json', {'stage': 'synthetic', 'settings': settings()})
    atomic_json(stage / 'plan.json', {'jobs': [j]})
    atomic_json(stage / 'summary.json', {'recorded_rows': 0})
    original = json.dumps(record(j)) + '\n'
    (stage / 'results.jsonl').write_text(original)
    output = tmp_path / 'report.md'
    report = build_delivery(tmp_path / 'study', output, plots=False)
    assert report['stages']['synthetic']['recorded_rows'] == 1
    assert report['stages']['synthetic']['cached_summary_recorded_rows'] == 0
    assert report['stages']['real_proposal']['recording_status'] == 'not_started'
    assert output.exists() and output.with_suffix('.csv').exists()
    assert (stage / 'results.jsonl').read_text() == original
    assert json.loads((stage / 'summary.json').read_text()) == {'recorded_rows': 0}


def test_delivery_reads_each_stage_budget_and_describes_full_api_not_kernel_time(tmp_path):
    stage = tmp_path / 'study' / 'long_sampling'
    stage.mkdir(parents=True)
    j = job(draws=8)
    atomic_json(stage / 'config.json', {'stage': 'long', 'settings': {
        **settings(), 'worker_seconds': 75, 'rss_limit_mib': 768}})
    atomic_json(stage / 'plan.json', {'jobs': [j]})
    (stage / 'results.jsonl').write_text(json.dumps(record(j)) + '\n')
    output = tmp_path / 'report.md'
    report = build_delivery(tmp_path / 'study', output, plots=False)
    body = output.read_text()
    assert '75 秒，RSS 上限为 768 MiB' in body
    assert '共计划 8 次采样' in body and '加 7 次独立 seed' in body
    assert '30秒' not in body and '30s' not in report['censoring_scope']
    assert 'proposal_seconds' in body and '不能称为纯数值 kernel 加速' in body
    detail = report['stages']['long_sampling']['per_request_backend_repeat'][0]
    assert detail['worker_limit_seconds'] == 75 and detail['rss_limit_mib'] == 768
