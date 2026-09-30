"""Read-only aggregation of shared-rhythm pilot records; no solver reruns."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import median

from tri.runtime import atomic_json, read_jsonl


def _metric(values, planned):
    values = [float(value) for value in values if value is not None]
    if any(not math.isfinite(value) for value in values):
        raise ValueError('Nonfinite timing/statistic in shared-rhythm report')
    return {'available_timing_records': len(values), 'planned_timing_records': planned,
            'median': median(values) if values else None,
            'minimum': min(values) if values else None, 'maximum': max(values) if values else None}


def _sampling_stats(row):
    candidates = [row.get('stats', {})]
    candidates += [measurement.get('sampling_diagnostics', measurement.get('stats', {}))
                   for measurement in row.get('measurements', {}).values()]
    candidates = [stats for stats in candidates if 'cumulative_proposals' in stats]
    result = dict(max(candidates, key=lambda stats: stats['cumulative_proposals'])) if candidates else {}
    progress = row.get('last_rejection_progress', {}).get('progress')
    if progress is not None:
        result.update({key: value for key, value in progress.items() if key.startswith('cumulative_')})
        result['proposal_in_flight'] = progress['in_flight']
        result['external_progress_available'] = True
    return result or None


def summarize(config, plan, rows):
    planned = {job['job_id']: job for job in plan['jobs']}
    indexed = {row['job_id']: row for row in rows}
    if len(indexed) != len(rows) or not indexed.keys() <= planned.keys():
        raise ValueError('Duplicate or unexpected study records')
    groups = defaultdict(list)
    case_jobs = defaultdict(list)
    target_z = defaultdict(list)
    for job in plan['jobs']:
        m = job['metadata']
        groups[m['R'], m['beta'], m['visibility'], job['backend']].append(job)
        case_jobs[job['case_id'], job['backend']].append(job)
    for row in rows:
        measurement = row.get('measurements', {}).get('target_partition')
        if measurement is not None:
            if not row['target_normalizer_available'] or row['backend'].startswith('onset_rejection_'):
                raise AssertionError('Proposal normalizer was mislabeled as target Z')
            target_z[row['case_id']].append(float(measurement['log_z']))
    output = []
    for (R, beta, visibility, backend), jobs in groups.items():
        records = [indexed[job['job_id']] for job in jobs if job['job_id'] in indexed]
        first, warm, compile_times, solver_times, rss = [], [], [], [], []
        for row in records:
            measurements = row.get('measurements', {})
            first.append(row.get('seconds_to_first_verified_sample_excluding_checks'))
            names = [f'repeated_full_sample_{i}' for i in range(1, config['warm_samples'] + 1)]
            warm.append(sum(measurements[name]['seconds'] for name in names)
                        if names and all(name in row.get('validations', {}) for name in names) else None)
            compile_times.append(measurements.get('compile', {}).get('seconds'))
            solver_times.append(row.get('solver_seconds') if row['status'] == 'completed' else None)
            rss.append(row.get('peak_rss_mib'))
        output.append({'R': R, 'beta': beta, 'visibility': visibility, 'backend': backend,
            'planned_rows': len(jobs), 'recorded_rows': len(records),
            'independent_q_seeds': len({job['metadata']['seed'] for job in jobs}),
            'timing_repeats_per_q': config['timing_repeats'],
            'status_counts': dict(Counter(row['status'] for row in records)),
            'compile_seconds': _metric(compile_times, len(jobs)),
            'seconds_to_first_verified_sample_excluding_checks': _metric(first, len(jobs)),
            'repeated_samples_total_seconds': _metric(warm, len(jobs)),
            'completed_workload_solver_seconds': _metric(solver_times, len(jobs)),
            'observed_peak_rss_mib_including_resource_failures': _metric(rss, len(jobs))})
    acceptance = []
    for (case_id, backend), jobs in case_jobs.items():
        if not backend.startswith('onset_rejection_'):
            continue
        # The same RNG workload is repeated for latency. Keep one predeclared
        # repeat (zero), never count three identical histories as independent.
        first_job = next(job for job in jobs if job['repeat'] == 0)
        row = indexed.get(first_job['job_id'])
        stats = _sampling_stats(row) if row is not None else None
        proposal_z = stats.get('proposal_log_z') if stats is not None else None
        values = target_z.get(case_id, [])
        exact = None
        unavailable_reason = None
        spread = max(values) - min(values) if values else None
        if proposal_z is not None and values:
            if spread > 1e-8:
                unavailable_reason = 'inconsistent_exact_target_normalizers'
            else:
                log_acceptance = median(values) - float(proposal_z)
                if log_acceptance > 1e-8:
                    unavailable_reason = 'target_partition_exceeds_proposal_partition'
                else:
                    exact = math.exp(min(0., log_acceptance))
        else:
            unavailable_reason = 'missing_target_or_proposal_normalizer'
        proposals = stats.get('cumulative_proposals') if stats is not None else None
        accepted = stats.get('cumulative_accepted_samples') if stats is not None else None
        acceptance.append({'case_id': case_id, 'backend': backend,
            'R': first_job['metadata']['R'], 'beta': first_job['metadata']['beta'],
            'visibility': first_job['metadata']['visibility'], 'q_seed': first_job['metadata']['seed'],
            'predeclared_sampling_history_repeat': 0,
            'record_status': row['status'] if row is not None else 'not_recorded',
            'cumulative_proposals': proposals, 'cumulative_accepted_samples': accepted,
            'externally_verified_returned_samples': row.get('verified_samples') if row is not None else None,
            'returned_sample_events': sum('full_sample' in name for name in row.get('measurements', {})) if row is not None else None,
            'cumulative_proposal_calls': stats.get('cumulative_proposal_calls') if stats is not None else None,
            'proposal_in_flight_when_last_reported': stats.get('proposal_in_flight') if stats is not None else None,
            'external_proposal_progress_available': stats.get('external_progress_available', False) if stats is not None else False,
            'observed_accepted_per_proposal': accepted / proposals if proposals else None,
            'counter_scope': 'Repeat0 only. Live callback counts preserve completed proposals when an external limit interrupts a draw; a started but unfinished proposal remains in_flight and is not counted completed. Accepted paths may not yet have returned or passed the external checker; verified_samples remains separate.',
            'proposal_log_z': proposal_z, 'target_log_z': median(values) if values else None,
            'available_target_z_records': len(values), 'target_log_z_spread': spread,
            'exact_acceptance_probability_from_partitions': exact,
            'exact_acceptance_unavailable_reason': unavailable_reason})
    statuses = dict(Counter(row['status'] for row in rows))
    complete = len(rows) == len(planned)
    return {'status': 'complete_recording' if complete else 'partial_recording',
        'planned_rows': len(planned), 'recorded_rows': len(rows), 'status_counts': statuses,
        'planned_cases': len({job['case_id'] for job in plan['jobs']}),
        'planned_independent_q_seeds': len(config['grid']['q_seeds']),
        'timing_repeats': config['timing_repeats'], 'groups': output, 'acceptance_by_case': acceptance,
        'interpretation': 'No ranking is inferred from unmatched successful timings. Read availability and status denominators alongside every metric; partial results do not establish a scaling claim.',
        'sampling_scope': 'One fixed RNG sample workload per q/backend, repeated only for timing. Acceptance histories use predeclared repeat0, not three independent trials. Two q seeds are a small structural pilot.',
        'time_scope': 'First latency includes measured constructor, optional target Z and first full sample. Repeated cost includes every rejected proposal, diagnostic construction and in-loop progress callback/logging. Only the external checker and between-operation event logging are excluded from solver seconds; all are inside whole-worker limits.',
        'memory_scope': 'Fresh worker physical RSS includes interpreter/input and checks; resource-failure peaks are observed limits, not successful workspace requirements.',
        'acceptance_scope': 'Exact target/proposal partition ratios only for the same case and beta. Proposal Z is never substituted for target Z. Missing normalization or interrupted counters remain null.'}


def write_report(output):
    output = Path(output)
    config = json.loads((output / 'config.json').read_text())
    plan = json.loads((output / 'plan.json').read_text())
    rows = read_jsonl(output / 'results.jsonl')
    report = summarize(config, plan, rows)
    atomic_json(output / 'summary.json', report)
    lines = ['# 共享节奏多片段先导实验', '',
             f'已记录 {report["recorded_rows"]}/{report["planned_rows"]} 行；状态：{report["status"]}。', '',
             '此目录是 synthetic_control。两个 q 种子及其计时重复不能当作大量真实音乐请求。', '',
             '| 结果状态 | 行数 |', '| --- | ---: |']
    lines += [f'| {status} | {count} |' for status, count in report['status_counts'].items()]
    lines += ['', '按 R/beta/visibility/backend 的耗时、内存与完整分母见 `summary.json` 的 `groups`。',
              '逐输入提议/接受计数和同目标 Z 比得到的准确接受率见 `acceptance_by_case`。', '',
              '首样本成本包含编译、可用的目标 Z 和完整采样；重复样本成本包含全部拒绝提议。',
              '采样循环内部诊断与progress日志计入solver时间；外部checker和操作间事件记录仅计入worker总墙钟。',
              '同一随机轨迹的三个时延重复不算独立接受率试验；只保留预先指定 repeat 0 的历史。',
              '缺失统计保留 null；预算失败和不支持不能记作 0 秒或 0 次提议。',
              '结果尚不完整时不作规模结论；不同成功分母不能直接用于算法排名。', '']
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='runs/shared_rhythm_pilot')
    args = parser.parse_args()
    report = write_report(args.out)
    print(json.dumps({'status': report['status'], 'recorded_rows': report['recorded_rows'],
                      'planned_rows': report['planned_rows'], 'status_counts': report['status_counts']}))


if __name__ == '__main__':
    main()
