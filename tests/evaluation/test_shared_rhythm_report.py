import math

import pytest

from tri.evaluation.shared_rhythm_report import summarize


def records():
    config = {'timing_repeats': 3, 'warm_samples': 2, 'grid': {'q_seeds': [1, 2]}}
    jobs, rows = [], []
    for seed in (1, 2):
        for backend in ('template_stream', 'onset_rejection_drop'):
            for repeat in range(3):
                job = {'job_id': f'{seed}/{backend}/{repeat}', 'case_id': f'case{seed}', 'backend': backend,
                       'repeat': repeat, 'metadata': {'R': 2, 'beta': .02, 'visibility': 'unknown', 'seed': seed}}
                jobs.append(job)
                samples = {'first_full_sample': {'seconds': .2}, 'repeated_full_sample_1': {'seconds': .1},
                           'repeated_full_sample_2': {'seconds': .1}}
                stats = {'proposal_log_z': 0., 'cumulative_proposals': 6 if repeat == 0 else 999,
                         'cumulative_accepted_samples': 3}
                measurements = {'compile': {'seconds': .1}, **samples}
                if backend == 'template_stream':
                    measurements['target_partition'] = {'seconds': .1, 'log_z': math.log(.5)}
                row = {**job, 'status': 'completed', 'target_normalizer_available': backend == 'template_stream',
                       'measurements': measurements, 'validations': {name: {'valid': True} for name in samples},
                       'solver_seconds': .5, 'seconds_to_first_verified_sample_excluding_checks': .3,
                       'peak_rss_mib': 64., 'stats': stats if backend != 'template_stream' else {}}
                rows.append(row)
    return config, {'jobs': jobs}, rows


def test_report_preserves_group_denominators_and_uses_only_repeat_zero_for_acceptance():
    config, plan, rows = records()
    report = summarize(config, plan, rows)
    assert report['recorded_rows'] == report['planned_rows'] == 12
    assert report['status'] == 'complete_recording'
    assert len(report['groups']) == 2
    for group in report['groups']:
        assert group['planned_rows'] == group['recorded_rows'] == 6
        assert group['completed_workload_solver_seconds']['median'] == .5
        assert group['repeated_samples_total_seconds']['median'] == .2
    for acceptance in report['acceptance_by_case']:
        assert acceptance['cumulative_proposals'] == 6
        assert acceptance['cumulative_accepted_samples'] == 3
        assert acceptance['observed_accepted_per_proposal'] == .5
        assert acceptance['exact_acceptance_probability_from_partitions'] == pytest.approx(.5)
        assert acceptance['available_target_z_records'] == 3


def test_partial_report_keeps_absent_cost_and_counts_null_not_zero():
    config, plan, rows = records()
    report = summarize(config, plan, rows[:1])
    assert report['status'] == 'partial_recording'
    sampler = next(g for g in report['groups'] if g['backend'] == 'onset_rejection_drop')
    assert sampler['recorded_rows'] == 0
    assert sampler['compile_seconds']['median'] is None
    assert sampler['compile_seconds']['available_timing_records'] == 0
    assert all(a['cumulative_proposals'] is None and a['exact_acceptance_probability_from_partitions'] is None
               for a in report['acceptance_by_case'])


def test_inconsistent_target_z_is_reported_without_creating_an_acceptance_probability():
    config, plan, rows = records()
    rows[0]['measurements']['target_partition']['log_z'] -= 1
    report = summarize(config, plan, rows)
    case = report['acceptance_by_case'][0]
    assert case['exact_acceptance_probability_from_partitions'] is None
    assert case['exact_acceptance_unavailable_reason'] == 'inconsistent_exact_target_normalizers'


def test_report_rejects_duplicate_rows_and_proposal_target_confusion():
    config, plan, rows = records()
    with pytest.raises(ValueError, match='Duplicate'):
        summarize(config, plan, rows + rows[:1])
    rows[3]['measurements']['target_partition'] = {'log_z': 0.}
    with pytest.raises(AssertionError, match='Proposal normalizer'):
        summarize(config, plan, rows)


def test_report_uses_live_completed_proposal_counters_after_worker_timeout():
    config, plan, rows = records()
    row = rows[3]
    row.update(status='timeout', last_rejection_progress={'operation': 'repeated_full_sample_1',
        'progress': {'in_flight': True, 'cumulative_proposals': 9, 'cumulative_proposal_calls': 10,
                     'cumulative_rejections': 8, 'cumulative_accepted_samples': 1}})
    report = summarize(config, plan, rows)
    case = report['acceptance_by_case'][0]
    assert case['cumulative_proposals'] == 9
    assert case['cumulative_proposal_calls'] == 10
    assert case['cumulative_accepted_samples'] == 1
    assert case['proposal_in_flight_when_last_reported']
    assert case['observed_accepted_per_proposal'] == pytest.approx(1 / 9)
