"""Matched extension results and independent per-request draw diagnostics."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import median

import numpy as np

from tri.evaluation.shared_rhythm_report import _sampling_stats
from tri.evaluation.shared_rhythm_study import check_partitions
from tri.runtime import atomic_json, read_jsonl


def distribution(values, planned):
    values = [float(value) for value in values if value is not None]
    if any(not math.isfinite(value) for value in values):
        raise ValueError('Nonfinite measured distribution value')
    return {'available': len(values), 'planned': planned,
            'mean': float(np.mean(values)) if values else None,
            'p50': float(np.quantile(values, .5)) if values else None,
            'p95': float(np.quantile(values, .95)) if values else None,
            'minimum': min(values) if values else None, 'maximum': max(values) if values else None}


def draw_records(job, row):
    row = row or {}
    measurements, validations = row.get('measurements', {}), row.get('validations', {})
    progress_record = row.get('last_rejection_progress', {})
    output = []
    for index, seed in enumerate(job['draw_seeds']):
        name = 'first_full_sample' if index == 0 else f'repeated_full_sample_{index}'
        measured, verified = measurements.get(name), validations.get(name)
        diagnostic = measured.get('sampling_diagnostics') if measured else None
        progress = progress_record.get('progress') if progress_record.get('operation') == name else None
        counts = diagnostic or progress or {}
        output.append({'sample_index': index, 'draw_seed': seed,
            'state': 'verified' if verified else 'returned_unverified' if measured else 'not_returned',
            'seconds': measured['seconds'] if measured else None,
            'proposal_calls': counts.get('proposal_calls'), 'completed_proposals': counts.get('proposals'),
            'rejections': counts.get('rejections'), 'accepted_decision': counts.get('accepted'),
            'proposal_unit': counts.get('proposal_unit', 'complete_original_token_path'),
            'full_candidates': counts.get('full_candidates'),
            'partial_candidates': counts.get('partial_candidates'),
            'spans_sampled': counts.get('spans_sampled'),
            'proposal_in_flight': progress.get('in_flight') if progress else False if diagnostic else None,
            'diagnostic_scope': 'Returned draw diagnostics or latest progress for this exact draw; an unfinished proposal is not completed. Missing is null.'})
    return output


def summarize(snapshot, plan, rows):
    settings = snapshot['settings']
    planned = {job['job_id']: job for job in plan['jobs']}
    indexed = {row['job_id']: row for row in rows}
    if len(indexed) != len(rows) or not indexed.keys() <= planned.keys():
        raise ValueError('Unexpected or duplicate extension result')
    partition_consistency = check_partitions(rows)
    z_values = defaultdict(list)
    for row in rows:
        z = row.get('measurements', {}).get('target_partition', {}).get('log_z')
        if z is not None:
            z_values[row['case_id']].append(float(z))
    jobs_by_case_backend = defaultdict(list)
    group_jobs = defaultdict(list)
    details = []
    detail_index = {}
    for job in plan['jobs']:
        row = indexed.get(job['job_id'])
        metadata = job['metadata']
        jobs_by_case_backend[job['case_id'], job['backend']].append(job)
        group_jobs[metadata['R'], metadata['beta'], metadata['visibility'], job['backend']].append(job)
        draws = draw_records(job, row)
        measurements = (row or {}).get('measurements', {})
        setup_available = 'compile' in measurements and (job['backend'].startswith('onset_rejection_') or 'target_partition' in measurements)
        setup = sum(measurements[name]['seconds'] for name in ('compile', 'target_partition') if name in measurements)
        milestones = {}
        for count in (1, 8, 32):
            milestones[str(count)] = (setup + sum(draw['seconds'] for draw in draws[:count])
                if setup_available and len(draws) >= count and all(draw['state'] == 'verified' for draw in draws[:count]) else None)
        completed = [draw for draw in draws if draw['state'] == 'verified']
        detail = {'job_id': job['job_id'], 'case_id': job['case_id'], 'backend': job['backend'], 'repeat': job['repeat'],
            'R': metadata['R'], 'beta': metadata['beta'], 'visibility': metadata['visibility'],
            'work_id': metadata.get('work_id'), 'record_status': row['status'] if row else 'not_recorded',
            'requested_samples': len(draws), 'verified_samples': len(completed),
            'truncated': len(completed) < len(draws),
            'solver_seconds_to_n_verified_samples': milestones,
            'compile_seconds': measurements.get('compile', {}).get('seconds'),
            'target_partition_seconds': measurements.get('target_partition', {}).get('seconds'),
            'per_draw_seconds_including_all_rejections': distribution([d['seconds'] for d in completed], len(draws)),
            'reused_draw_seconds_including_all_rejections': distribution([d['seconds'] for d in completed if d['sample_index'] > 0], max(0, len(draws) - 1)),
            'peak_rss_mib': (row or {}).get('peak_rss_mib'),
            'worker_seconds': (row or {}).get('elapsed_worker_seconds'),
            'frozen_model_once_seconds': metadata.get('probability_provider_seconds'),
            'frozen_model_calls': metadata.get('model_calls'), 'draws': draws}
        details.append(detail)
        detail_index[job['job_id']] = detail
    groups = []
    for (R, beta, visibility, backend), jobs in group_jobs.items():
        selected = [detail_index[job['job_id']] for job in jobs]
        groups.append({'R': R, 'beta': beta, 'visibility': visibility, 'backend': backend,
            'planned_rows': len(jobs), 'recorded_rows': sum(job['job_id'] in indexed for job in jobs),
            'independent_requests_or_q_fixtures': len({job['case_id'] for job in jobs}),
            'timing_repeats_per_input': settings['timing_repeats'],
            'status_counts': dict(Counter(detail['record_status'] for detail in selected)),
            'compile_seconds': distribution([detail['compile_seconds'] for detail in selected], len(jobs)),
            'solver_seconds_to_n_samples': {str(n): distribution([d['solver_seconds_to_n_verified_samples'][str(n)] for d in selected], len(jobs)) for n in (1, 8, 32)},
            'observed_rss_mib_all_statuses': distribution([detail['peak_rss_mib'] for detail in selected], len(jobs))})
    acceptance = []
    for (case_id, backend), jobs in jobs_by_case_backend.items():
        if not backend.startswith('onset_rejection_'):
            continue
        job = next(job for job in jobs if job['repeat'] == 0)
        row = indexed.get(job['job_id'])
        stats = _sampling_stats(row) if row else None
        proposal_z = stats.get('proposal_log_z') if stats else None
        values = z_values.get(case_id, [])
        exact = None
        log_alpha = None
        if proposal_z is not None and values and max(values) - min(values) <= 1e-8:
            log_alpha = median(values) - float(proposal_z)
            if log_alpha <= 1e-8:
                exact = math.exp(min(0., log_alpha))
        proposals = stats.get('cumulative_proposals') if stats else None
        accepted = stats.get('cumulative_accepted_samples') if stats else None
        detail = detail_index[job['job_id']]
        acceptance.append({'case_id': case_id, 'backend': backend, 'R': job['metadata']['R'],
            'beta': job['metadata']['beta'], 'visibility': job['metadata']['visibility'],
            'record_status': detail['record_status'], 'sampling_history_repeat': 0,
            'requested_draws': len(job['draw_seeds']), 'verified_returned_draws': detail['verified_samples'],
            'cumulative_proposals': proposals, 'cumulative_accepted_decisions': accepted,
            'proposal_unit': stats.get('proposal_unit', 'complete_original_token_path') if stats else None,
            'cumulative_full_candidates': stats.get('cumulative_full_candidates') if stats else None,
            'cumulative_partial_candidates': stats.get('cumulative_partial_candidates') if stats else None,
            'cumulative_spans_sampled': stats.get('cumulative_spans_sampled') if stats else None,
            'cumulative_proposal_calls': stats.get('cumulative_proposal_calls') if stats else None,
            'proposal_in_flight': stats.get('proposal_in_flight') if stats else None,
            'observed_accepted_per_completed_proposal': accepted / proposals if proposals else None,
            'proposal_log_z': proposal_z, 'target_log_z': median(values) if values else None,
            'available_target_z_records': len(values),
            'log_acceptance_from_partition_ratio': log_alpha,
            'acceptance_from_partition_ratio': exact})
    matched = []
    case_ids = list(dict.fromkeys(job['case_id'] for job in plan['jobs']))
    for case_id in case_ids:
        for numerator, denominator in combinations(settings['backends'], 2):
            left = jobs_by_case_backend[case_id, numerator]
            right = jobs_by_case_backend[case_id, denominator]
            for n in (1, 8, 32):
                a = [detail_index[j['job_id']]['solver_seconds_to_n_verified_samples'][str(n)] for j in left]
                b = [detail_index[j['job_id']]['solver_seconds_to_n_verified_samples'][str(n)] for j in right]
                available = len(a) == len(b) == settings['timing_repeats'] and all(x is not None for x in a + b)
                matched.append({'case_id': case_id, 'numerator': numerator, 'denominator': denominator,
                    'cumulative_samples': n, 'planned_repeats_per_backend': settings['timing_repeats'],
                    'available_numerator_repeats': sum(x is not None for x in a),
                    'available_denominator_repeats': sum(x is not None for x in b),
                    'ratio_of_median_solver_seconds': median(a) / median(b) if available else None})
    return {'version': 1, 'protocol': snapshot['protocol'], 'stage': snapshot['stage'],
        'recording_status': 'complete' if len(rows) == len(planned) else 'partial',
        'planned_rows': len(planned), 'recorded_rows': len(rows),
        'status_counts': dict(Counter(row['status'] for row in rows)),
        'unique_requests_or_q_fixtures': len(case_ids),
        'unique_works': len({job['metadata']['work_id'] for job in plan['jobs'] if 'work_id' in job['metadata']}),
        'partition_consistency': partition_consistency, 'groups': groups,
        'per_request_backend_repeat': details, 'acceptance_by_request': acceptance,
        'matched_same_run_cost_ratios': matched,
        'sampling_scope': 'Real: 32 distinct seeded full draws per request/backend, one fresh engine then31 reused draws, no timing repetitions. Synthetic:3 distinct draw seeds replayed in2 timing repetitions; repetitions are not independent acceptance histories.',
        'timing_scope': 'Cumulative1/8/32 includes measured compile, optional target Z and every relevant full sampler call. Sampler calls include all rejections, diagnostics and in-loop progress logging. RNG setup, external checks and between-operation logging are outside solver timing but inside worker limits.',
        'quantile_scope': 'Mean/p50/p95 use verified returned draws only; requested/verified/truncated counts accompany them. Failed in-flight draw duration is censored by the worker wall limit, never replaced by zero.',
        'acceptance_scope': 'One predeclared repeat0 history per input/backend. Early rejection counts completed shared-rhythm accept/reject attempts, not necessarily full paths; explicit full/partial/spans counters accompany the proposal unit. Live progress can include an accepted decision not yet returned/verified. Proposal Z is never target Z.',
        'model_scope': 'Original model probability-provider cost is read from frozen input metadata and archived separately. It is not charged again or mixed into measured discrete solver timings.',
        'comparison_scope': 'Cost ratios require every declared repeat for both methods on the same input and sample milestone. Older pilot VE/template timings are never joined. Missing is null; partial results alone do not justify scaling claims.'}


def write_report(output):
    output = Path(output)
    snapshot = json.loads((output / 'config.json').read_text())
    plan = json.loads((output / 'plan.json').read_text())
    rows = read_jsonl(output / 'results.jsonl')
    report = summarize(snapshot, plan, rows)
    atomic_json(output / 'summary.json', report)
    lines = ['# Shared-rhythm extension', '',
        f'Stage: {report["stage"]}; recorded {report["recorded_rows"]}/{report["planned_rows"]} ({report["recording_status"]}).', '',
        '| Status | Rows |', '| --- | ---: |']
    lines += [f'| {status} | {count} |' for status, count in report['status_counts'].items()]
    lines += ['', 'Detailed JSON keeps per-draw timings/seeds, truncated outcomes, cumulative1/8/32 costs, and proposal counters.',
        'Matched ratios require the complete same-input denominator. Historical pilot times are not combined.',
        'Frozen model-once cost is separate. A missing statistic remains null; incomplete sampling is not a zero-cost success.', '']
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='runs/shared_rhythm_extension')
    args = parser.parse_args()
    report = write_report(args.out)
    print(json.dumps({'stage': report['stage'], 'recorded_rows': report['recorded_rows'],
                      'planned_rows': report['planned_rows'], 'status_counts': report['status_counts']}))


if __name__ == '__main__':
    main()
