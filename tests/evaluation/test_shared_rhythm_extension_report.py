import math

import pytest

from tri.evaluation.shared_rhythm_extension_report import summarize


def data():
    settings = {'timing_repeats': 1, 'backends': ['product_multi', 'onset_rejection_boundary']}
    snapshot = {'settings': settings, 'stage': 'real', 'protocol': 'test'}
    jobs, rows = [], []
    for backend in settings['backends']:
        sampler = backend.startswith('onset_rejection_')
        job = {'job_id': backend, 'case_id': 'request0', 'backend': backend, 'repeat': 0,
               'metadata': {'R': 3, 'beta': .02, 'visibility': 'unknown', 'work_id': 'work1',
                            'probability_provider_seconds': .03, 'model_calls': 1},
               'draw_seeds': list(range(32))}
        jobs.append(job)
        measurements = {'compile': {'seconds': 1. if sampler else .5}}
        if not sampler:
            measurements['target_partition'] = {'seconds': .5, 'log_z': -2.}
        validations = {}
        for index in range(32):
            name = 'first_full_sample' if index == 0 else f'repeated_full_sample_{index}'
            measurements[name] = {'seconds': float(index + 1), 'sample_index': index, 'validated': True}
            validations[name] = {'valid': True}
        rows.append({**job, 'status': 'completed', 'target_normalizer_available': not sampler,
                     'measurements': measurements, 'validations': validations, 'peak_rss_mib': 64.,
                     'stats': {'proposal_log_z': -1., 'cumulative_proposals': 64,
                               'cumulative_accepted_samples': 32} if sampler else {}})
    return snapshot, {'jobs': jobs}, rows


def test_cold_and_reuse_are_views_of_the_same_32_independent_draws():
    snapshot, plan, rows = data()
    report = summarize(snapshot, plan, rows)
    for item in report['per_request_backend_repeat']:
        assert item['requested_samples'] == item['verified_samples'] == 32
        assert item['solver_seconds_to_n_verified_samples'] == {'1': 2., '8': 37., '32': 529.}
        assert item['per_draw_seconds_including_all_rejections']['mean'] == 16.5
        assert item['per_draw_seconds_including_all_rejections']['p50'] == 16.5
        assert item['per_draw_seconds_including_all_rejections']['p95'] == pytest.approx(30.45)
        assert item['reused_draw_seconds_including_all_rejections']['available'] == 31
        assert item['frozen_model_once_seconds'] == .03
        assert len({draw['draw_seed'] for draw in item['draws']}) == 32
    accepted = report['acceptance_by_request'][0]
    assert accepted['observed_accepted_per_completed_proposal'] == .5
    assert accepted['acceptance_from_partition_ratio'] == pytest.approx(math.exp(-1))
    assert all(r['ratio_of_median_solver_seconds'] == 1. for r in report['matched_same_run_cost_ratios'])


def test_failure_keeps_partial_draw_distribution_and_missing_milestones_null():
    snapshot, plan, rows = data()
    row = rows[1]
    row['status'] = 'timeout'
    for index in range(5, 32):
        name = f'repeated_full_sample_{index}'
        del row['measurements'][name]
        del row['validations'][name]
    row['last_rejection_progress'] = {'operation': 'repeated_full_sample_5', 'progress': {
        'in_flight': True, 'cumulative_proposals': 12, 'cumulative_proposal_calls': 13,
        'cumulative_accepted_samples': 5, 'proposals': 2, 'proposal_calls': 3, 'accepted': 0}}
    report = summarize(snapshot, plan, rows)
    detail = report['per_request_backend_repeat'][1]
    assert detail['verified_samples'] == 5
    assert detail['truncated']
    assert detail['solver_seconds_to_n_verified_samples']['1'] == 2.
    assert detail['solver_seconds_to_n_verified_samples']['8'] is None
    assert detail['solver_seconds_to_n_verified_samples']['32'] is None
    assert detail['per_draw_seconds_including_all_rejections']['available'] == 5
    assert detail['per_draw_seconds_including_all_rejections']['planned'] == 32
    assert detail['draws'][5]['completed_proposals'] == 2
    assert detail['draws'][5]['proposal_in_flight']
    accepted = report['acceptance_by_request'][0]
    assert accepted['cumulative_proposals'] == 12
    assert accepted['cumulative_accepted_decisions'] == 5
    assert report['matched_same_run_cost_ratios'][-1]['ratio_of_median_solver_seconds'] is None


def test_missing_rows_and_original_model_time_are_never_zero_filled():
    snapshot, plan, rows = data()
    del plan['jobs'][1]['metadata']['probability_provider_seconds']
    report = summarize(snapshot, plan, rows[:1])
    detail = report['per_request_backend_repeat'][1]
    assert detail['record_status'] == 'not_recorded'
    assert detail['frozen_model_once_seconds'] is None
    assert detail['compile_seconds'] is None
    assert detail['per_draw_seconds_including_all_rejections']['p50'] is None
    assert report['recording_status'] == 'partial'


def test_proposal_z_never_enters_exact_target_comparison():
    snapshot, plan, rows = data()
    rows[1]['measurements']['target_partition'] = {'log_z': -1., 'seconds': .1}
    with pytest.raises(AssertionError, match='Sampling-only'):
        summarize(snapshot, plan, rows)
