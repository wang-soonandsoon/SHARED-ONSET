"""Regenerate the extension delivery from immutable plans and raw snapshots.

No source report, runner, solver or raw record is modified. In-progress JSONL
tails are ignored only when incomplete, never repaired. A cached summary may
lag the raw records; its row count is recorded, while metrics use this snapshot.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

import numpy as np

from tri.evaluation.shared_rhythm_report import _sampling_stats
from tri.runtime import atomic_json


DEFAULT_SOURCES = ('synthetic', 'real_sampling', 'real_proposal', 'long_sampling')
BUDGET_STATUSES = {'timeout', 'rss_limit', 'internal_budget'}
LABELS = {'product_multi': 'Standard multi-product', 'product_prefix': 'Standard pair-prefix',
    've_aligned': 'Aligned VE', 'template_stream': 'Exact template', 'onset_reset': 'Onset reset',
    'onset_rejection_drop': 'Drop rejection', 'onset_rejection_boundary': 'Boundary rejection',
    'onset_rejection_visible': 'Visible rejection',
    'onset_rejection_boundary_early': 'Boundary + early rejection',
    'onset_rejection_visible_early': 'Visible + early rejection'}
METRICS = ('cold_solver_seconds', 'full_warm_mean_seconds', 'full_warm_p95_seconds',
           'full_workload_solver_seconds')


def read_snapshot(path):
    """Read complete JSONL records without touching an active writer's file."""
    path = Path(path)
    if not path.exists():
        return [], False
    lines = path.read_bytes().splitlines(keepends=True)
    rows, ignored = [], False
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b'\n'):
                raise ValueError(f'Corrupt complete JSONL record in {path}, line {index + 1}')
            ignored = True
    return rows, ignored


def stats(values, planned):
    values = [float(v) for v in values if v is not None]
    if any(not math.isfinite(v) for v in values):
        raise ValueError('Nonfinite reported measurement')
    return {'available': len(values), 'planned': planned,
            'mean': float(np.mean(values)) if values else None,
            'median': median(values) if values else None,
            'p95': float(np.quantile(values, .95)) if values else None,
            'minimum': min(values) if values else None, 'maximum': max(values) if values else None}


def _operation_order(name):
    fixed = {'support_check': -3, 'compile': -2, 'target_partition': -1, 'first_full_sample': 0}
    if name in fixed:
        return fixed[name]
    prefix = 'repeated_full_sample_'
    if isinstance(name, str) and name.startswith(prefix) and name[len(prefix):].isdigit():
        return int(name[len(prefix):])
    return None


def _measurement(row, name):
    if row.get('status') in BUDGET_STATUSES:
        failed, current = _operation_order(row.get('failed_operation')), _operation_order(name)
        # A single pipe drain may contain later completed operations after the
        # first budget violation. Their raw outputs still exist, but none is a
        # within-budget result. Unknown ordering is conservatively unavailable.
        if failed is None or current is None or current >= failed:
            return None
    return row.get('measurements', {}).get(name)


def _acceptance(row):
    value = _sampling_stats(row) or {}
    progress = row.get('last_rejection_progress', {}).get('progress', {})
    unit = progress.get('proposal_unit', value.get('proposal_unit', 'completed_full_path_accept_or_reject_decision'))
    completed, accepted = value.get('cumulative_proposals'), value.get('cumulative_accepted_samples')
    return {'proposal_unit': unit, 'completed_attempts': completed, 'accepted_decisions': accepted,
            'accepted_per_completed_attempt': accepted / completed if completed else None,
            'proposal_calls': value.get('cumulative_proposal_calls'),
            'proposal_in_flight': value.get('proposal_in_flight'),
            'proposal_log_z': value.get('proposal_log_z'),
            'full_candidates': value.get('cumulative_full_candidates'),
            'partial_candidates': value.get('cumulative_partial_candidates'),
            'spans_sampled': value.get('cumulative_spans_sampled'),
            'patterns_sampled': value.get('cumulative_patterns_sampled')}


def request_detail(job, row, settings):
    row = row or {}
    metadata = job['metadata']
    requested = len(job['draw_seeds'])
    draws = []
    for index, seed in enumerate(job['draw_seeds']):
        name = 'first_full_sample' if index == 0 else f'repeated_full_sample_{index}'
        measurement = _measurement(row, name)
        raw_measurement = row.get('measurements', {}).get(name)
        validation = row.get('validations', {}).get(name, {})
        verified = measurement is not None and validation.get('valid') is True
        draws.append({'index': index, 'seed': seed, 'verified_within_budget': verified,
                      'raw_returned': raw_measurement is not None,
                      'raw_verified': raw_measurement is not None and validation.get('valid') is True,
                      'seconds': measurement.get('seconds') if verified else None})
    compile_measure = _measurement(row, 'compile')
    partition = _measurement(row, 'target_partition')
    rejection = job['backend'].startswith('onset_rejection_')
    setup = (compile_measure['seconds'] + (partition['seconds'] if partition else 0.)
             if compile_measure and (partition or rejection) else None)
    first = draws[0]['seconds'] if draws else None
    warm = [d['seconds'] for d in draws[1:] if d['verified_within_budget']]
    complete = all(d['verified_within_budget'] for d in draws)
    warm_complete = bool(draws[1:]) and len(warm) == requested - 1
    detail = {'job_id': job['job_id'], 'case_id': job['case_id'], 'backend': job['backend'],
        'repeat': job['repeat'], 'R': metadata['R'], 'L': metadata['L'], 'D': metadata['D'],
        'K': metadata.get('K'), 'beta': metadata['beta'], 'visibility': metadata['visibility'],
        'target_variant': 'wide' if metadata.get('wide_pitch_bounds') else 'narrow',
        'q_pairing_key': metadata.get('q_pairing_key', job['case_id']),
        'work_id': metadata.get('work_id'), 'record_status': row.get('status', 'not_recorded'),
        'requested_draws': requested, 'verified_draws': sum(d['verified_within_budget'] for d in draws),
        'raw_returned_draws': sum(d['raw_returned'] for d in draws),
        'raw_verified_draws': sum(d['raw_verified'] for d in draws),
        'budget_excluded_raw_verified_draws': sum(d['raw_verified'] and not d['verified_within_budget'] for d in draws),
        'not_verified_draws': sum(not d['verified_within_budget'] for d in draws),
        'cold_solver_seconds': setup + first if setup is not None and first is not None else None,
        'first_draw_seconds': first, 'compile_seconds': compile_measure.get('seconds') if compile_measure else None,
        'partition_seconds': partition.get('seconds') if partition else None,
        'observed_warm_draws': stats(warm, max(0, requested - 1)),
        'full_warm_mean_seconds': float(np.mean(warm)) if warm_complete else None,
        'full_warm_p95_seconds': float(np.quantile(warm, .95)) if warm_complete else None,
        'full_workload_solver_seconds': setup + sum(d['seconds'] for d in draws) if setup is not None and complete else None,
        'failed_operation': row.get('failed_operation'), 'worker_limit_seconds': settings['worker_seconds'],
        'rss_limit_mib': settings.get('rss_limit_mib'),
        'elapsed_worker_seconds': row.get('elapsed_worker_seconds'),
        'worker_completion_right_censored': row.get('status') == 'timeout',
        'resource_budget_failure': row.get('status') in BUDGET_STATUSES,
        'model_source_once_seconds': metadata.get('probability_provider_seconds'),
        'model_calls_for_this_variant': metadata.get('model_calls'), 'draws': draws,
        **_acceptance(row)}
    detail['cold_plus_source_model_seconds'] = (detail['cold_solver_seconds'] + metadata['probability_provider_seconds']
        if detail['cold_solver_seconds'] is not None and metadata.get('probability_provider_seconds') is not None else None)
    return detail


def summarize_stage(snapshot, plan, rows):
    settings = snapshot['settings']
    planned = {job['job_id']: job for job in plan['jobs']}
    indexed = {row['job_id']: row for row in rows}
    if len(planned) != len(plan['jobs']) or len(indexed) != len(rows) or not indexed.keys() <= planned.keys():
        raise ValueError('Duplicate/unplanned job in extension snapshot')
    details = [request_detail(job, indexed.get(job['job_id']), settings) for job in plan['jobs']]
    cases, groups = defaultdict(list), defaultdict(list)
    for row in details:
        cases[row['case_id'], row['backend']].append(row)
    per_case = []
    for (case_id, backend), repeats in cases.items():
        item = {key: repeats[0][key] for key in ('case_id', 'backend', 'R', 'L', 'D', 'beta', 'visibility', 'target_variant', 'q_pairing_key', 'work_id')}
        item['planned_repeats'] = len(repeats)
        item['status_counts'] = dict(Counter(r['record_status'] for r in repeats))
        for metric in METRICS:
            values = [r[metric] for r in repeats]
            item[metric] = median(values) if len(values) == settings['timing_repeats'] and all(v is not None for v in values) else None
        per_case.append(item)
        groups[item['R'], item['L'], item['beta'], item['visibility'], item['target_variant'], backend].append(item)
    grouped = []
    for key, selected in sorted(groups.items()):
        R, L, beta, visibility, variant, backend = key
        chosen_details = [r for r in details if (r['R'], r['L'], r['beta'], r['visibility'], r['target_variant'], r['backend']) == key]
        acceptance = [r for r in chosen_details if r['repeat'] == 0]
        grouped.append({'R': R, 'L': L, 'D_values': sorted({r['D'] for r in selected}), 'beta': beta,
            'visibility': visibility, 'target_variant': variant, 'backend': backend,
            'planned_cases': len(selected), 'planned_rows': len(chosen_details),
            'status_counts': dict(Counter(r['record_status'] for r in chosen_details)),
            'worker_timeouts': sum(r['worker_completion_right_censored'] for r in chosen_details),
            'verified_draws': sum(r['verified_draws'] for r in chosen_details),
            'requested_draws': sum(r['requested_draws'] for r in chosen_details),
            'acceptance_per_request': stats([r['accepted_per_completed_attempt'] for r in acceptance], len(acceptance)),
            **{metric: stats([r[metric] for r in selected], len(selected)) for metric in METRICS}})
    pairings = []
    for metric in METRICS:
        for reference, candidate in combinations(sorted({r['backend'] for r in per_case}), 2):
            a = {r['case_id']: r for r in per_case if r['backend'] == reference}
            b = {r['case_id']: r for r in per_case if r['backend'] == candidate}
            for case_id in sorted(a.keys() & b.keys()):
                x, y = a[case_id][metric], b[case_id][metric]
                pairings.append({'case_id': case_id, 'R': a[case_id]['R'], 'L': a[case_id]['L'],
                    'beta': a[case_id]['beta'], 'visibility': a[case_id]['visibility'],
                    'target_variant': a[case_id]['target_variant'], 'reference': reference, 'candidate': candidate,
                    'metric': metric, 'reference_seconds': x, 'candidate_seconds': y,
                    'ratio_reference_over_candidate': x / y if x is not None and y is not None and y > 0 else None})
    target_z, z_backends, errors = defaultdict(list), defaultdict(set), []
    for row in rows:
        m = row.get('measurements', {}).get('target_partition')
        if m is not None:
            if row.get('target_normalizer_available') is not True or row['backend'].startswith('onset_rejection_'):
                errors.append({'job_id': row['job_id'], 'reason': 'proposal_mislabeled_as_target_partition'})
                continue
            target_z[row['case_id']].append(float(m['log_z']))
            z_backends[row['case_id']].add(row['backend'])
    for case_id, values in target_z.items():
        if not all(math.isfinite(v) for v in values) or max(values) - min(values) > 1e-8:
            errors.append({'case_id': case_id, 'reason': 'target_partition_disagreement'})
    for detail in details:
        values, proposal_z = target_z.get(detail['case_id'], []), detail['proposal_log_z']
        detail['exact_acceptance_from_z_ratio'] = None
        if values and proposal_z is not None and max(values) - min(values) <= 1e-8:
            residual = median(values) - proposal_z
            if residual <= 1e-8:
                detail['exact_acceptance_from_z_ratio'] = math.exp(min(0., residual))
    return {'stage': snapshot['stage'], 'planned_rows': len(planned), 'recorded_rows': len(rows),
        'recording_status': 'complete' if len(rows) == len(planned) else 'partial',
        'status_counts': dict(Counter(r['status'] for r in rows)),
        'backends': sorted({j['backend'] for j in plan['jobs']}), 'settings': settings,
        'target_cases': len({r['case_id'] for r in details}),
        'q_input_count': len({r['q_pairing_key'] for r in details}),
        'work_count': len({r['work_id'] for r in details if r['work_id'] is not None}),
        'partition_audit': {'target_z_records': sum(map(len, target_z.values())),
            'cases_with_target_z': len(target_z),
            'cases_with_multiple_target_backends': sum(len(v) >= 2 for v in z_backends.values()),
            'maximum_log_z_spread': max((max(v) - min(v) for v in target_z.values()), default=None), 'errors': errors},
        'verified_draws_including_timing_repeats': sum(r['verified_draws'] for r in details),
        'groups': grouped, 'per_request_backend_repeat': details, 'per_case_backend': per_case,
        'matched_within_stage': pairings,
        'failures': [{k: r[k] for k in ('job_id', 'case_id', 'backend', 'record_status', 'failed_operation', 'verified_draws', 'requested_draws', 'worker_completion_right_censored')}
                     for r in details if r['record_status'] not in ('completed', 'not_recorded')]}


def _load_stage(path):
    if not (path / 'plan.json').exists() or not (path / 'config.json').exists():
        return {'recording_status': 'not_started', 'source': str(path)}
    snapshot = json.loads((path / 'config.json').read_text())
    plan = json.loads((path / 'plan.json').read_text())
    rows, ignored_tail = read_snapshot(path / 'results.jsonl')
    result = summarize_stage(snapshot, plan, rows)
    cached = json.loads((path / 'summary.json').read_text()) if (path / 'summary.json').exists() else {}
    result.update(source=str(path), incomplete_last_record_ignored=ignored_tail,
                  cached_summary_recorded_rows=cached.get('recorded_rows'))
    return result


def _model_inputs(path):
    if not path.exists():
        return {'status': 'not_frozen'}
    manifest = json.loads(path.read_text())
    times = [r['metadata']['probability_provider_seconds'] for r in manifest['cases']]
    return {'status': manifest['status'], 'path': str(path), 'requests': manifest['count'],
        'works': manifest.get('selected_work_count'), 'model_calls': manifest.get('model_calls'),
        'first_provider_seconds': times[0] if times else None,
        'later_provider_seconds': stats(times[1:], max(0, len(times) - 1)),
        'first_provider_preprocessing_seconds': manifest['cases'][0]['metadata'].get('provider_preprocessing_seconds') if times else None,
        'model_loading_seconds': [s['model_load_seconds'] for s in manifest.get('timing', {}).get('sessions', [])],
        'sessions': manifest.get('timing', {}).get('sessions', []), 'policy': manifest.get('policy', {})}


def _format(value, *, scale=1., digits=3):
    return '—' if value is None else f'{value * scale:.{digits}f}'


def _status_text(counts):
    return ', '.join(f'{key}={value}' for key, value in sorted(counts.items())) or '—'


def _table_groups(lines, groups, *, synthetic=False):
    if synthetic:
        lines += ['| R | 后端 | 冷启动到首样本 ms | 复用单样本均值 ms | 可用输入 冷/暖 | 原始行状态 |',
                  '|---:|---|---:|---:|---|---|']
        for g in groups:
            a, b = g['cold_solver_seconds'], g['full_warm_mean_seconds']
            lines.append(f'| {g["R"]} | `{g["backend"]}` | {_format(a["median"], scale=1000)} | {_format(b["median"], scale=1000)} | {a["available"]}/{a["planned"]}; {b["available"]}/{b["planned"]} | {_status_text(g["status_counts"])} |')
    else:
        lines += ['| R / 可见性 / 音域 | 后端 | 冷 mean / p95 ms | 暖 mean / p95 ms¹ | 完整暖请求 | 已校验/计划 draw | timeout | 接受率中位数² |',
                  '|---|---|---:|---:|---|---|---:|---:|']
        for g in groups:
            a, b, c = (g[k] for k in ('cold_solver_seconds', 'full_warm_mean_seconds', 'full_warm_p95_seconds'))
            lines.append(f'| {g["R"]} / {g["visibility"]} / {g["target_variant"]} | `{g["backend"]}` | {_format(a["mean"], scale=1000)} / {_format(a["p95"], scale=1000)} ({a["available"]}/{a["planned"]}) | {_format(b["median"], scale=1000)} / {_format(c["median"], scale=1000)} | {b["available"]}/{b["planned"]} | {g["verified_draws"]}/{g["requested_draws"]} | {g["worker_timeouts"]} | {_format(g["acceptance_per_request"]["median"], scale=100, digits=2)}% |')


def _narrative_pairs(stage, visibility, beta, reference, candidate):
    rows = [r for r in stage['matched_within_stage'] if r['visibility'] == visibility and r['beta'] == beta
            and {r['reference'], r['candidate']} == {reference, candidate} and r['metric'] == 'cold_solver_seconds']
    values = [(r['ratio_reference_over_candidate'] if r['reference'] == reference else 1 / r['ratio_reference_over_candidate'])
              for r in rows if r['ratio_reference_over_candidate'] is not None]
    return stats(values, len(rows)), sum(v > 1.05 for v in values), sum(v < 1 / 1.05 for v in values)


def _write_csv(path, report):
    keys = ['source', 'case_id', 'q_pairing_key', 'work_id', 'R', 'L', 'D', 'K', 'beta', 'visibility',
        'target_variant', 'backend', 'repeat', 'record_status', 'requested_draws', 'verified_draws',
        'raw_returned_draws', 'raw_verified_draws', 'budget_excluded_raw_verified_draws',
        'cold_solver_seconds', 'cold_plus_source_model_seconds', 'first_draw_seconds',
        'observed_warm_count', 'observed_warm_mean', 'observed_warm_p95',
        'full_warm_mean_seconds', 'full_warm_p95_seconds', 'full_workload_solver_seconds',
        'failed_operation', 'worker_completion_right_censored', 'worker_limit_seconds', 'rss_limit_mib', 'elapsed_worker_seconds',
        'proposal_unit', 'proposal_calls', 'completed_attempts', 'accepted_decisions',
        'accepted_per_completed_attempt', 'proposal_in_flight', 'full_candidates', 'partial_candidates',
        'spans_sampled', 'exact_acceptance_from_z_ratio']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for source, stage in report['stages'].items():
            for detail in stage.get('per_request_backend_repeat', []):
                row = {key: detail.get(key) for key in keys}
                row.update(source=source, observed_warm_count=detail['observed_warm_draws']['available'],
                    observed_warm_mean=detail['observed_warm_draws']['mean'], observed_warm_p95=detail['observed_warm_draws']['p95'])
                writer.writerow(row)


def _figures(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = []
    for source, stage in report['stages'].items():
        if not stage.get('recorded_rows'):
            continue
        synthetic = source == 'synthetic'
        groups = stage['groups']
        selectors = [('unknown', 0.), ('known', .02)] if synthetic else [(v, .02) for v in sorted({g['visibility'] for g in groups})]
        fig, axes = plt.subplots(len(selectors), 2, figsize=(12, 4 * len(selectors)), squeeze=False)
        methods = stage['backends']
        cmap = plt.get_cmap('tab10')
        for i, (visibility, beta) in enumerate(selectors):
            for j, metric in enumerate(('cold_solver_seconds', 'full_warm_mean_seconds')):
                ax = axes[i, j]
                for method_index, method in enumerate(methods):
                    for variant in sorted({g['target_variant'] for g in groups}):
                        selected = sorted([g for g in groups if g['visibility'] == visibility and g['beta'] == beta
                            and g['backend'] == method and g['target_variant'] == variant], key=lambda g: g['R'])
                        if not selected:
                            continue
                        values = [g[metric]['median'] * 1000 if g[metric]['median'] is not None else np.nan for g in selected]
                        label = LABELS.get(method, method) + (' / wide' if variant == 'wide' else '')
                        ax.plot([g['R'] for g in selected], values, marker='o', linestyle='--' if variant == 'wide' else '-',
                                color=cmap(method_index % 10), label=label)
                        for g, value in zip(selected, values):
                            if np.isfinite(value) and g[metric]['available'] != g[metric]['planned']:
                                ax.annotate(f'{g[metric]["available"]}/{g[metric]["planned"]}', (g['R'], value), fontsize=7)
                ax.set_yscale('log')
                ax.set_xlabel('Number of synchronized spans R')
                ax.set_ylabel('Milliseconds (log scale)')
                ax.set_title(f'{visibility}, beta={beta:g}: ' + ('cold to first verified draw' if j == 0 else 'mean reused draw / request'))
                ax.grid(True, alpha=.25)
                ax.set_xticks(sorted({g['R'] for g in groups}))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
        fig.suptitle(f'{source}: {stage["recorded_rows"]}/{stage["planned_rows"]} rows; missing points are unavailable, not zero')
        fig.tight_layout(rect=(0, .11, 1, .95))
        stem = output / f'shared_rhythm_extension_{source}'
        for suffix in ('png', 'pdf'):
            dest = stem.with_suffix('.' + suffix)
            fig.savefig(dest, dpi=200, bbox_inches='tight')
            figures.append(str(dest))
        plt.close(fig)
    return figures


def build_delivery(study_root='runs/shared_rhythm_extension', output='reports/SHARED_RHYTHM_EXTENSION_RESULTS.md', *, sources=DEFAULT_SOURCES, plots=True):
    root, output = Path(study_root).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'version': 1, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'stages': {name: _load_stage(root / name) for name in sources},
        'frozen_inputs': {name: _model_inputs(root / name / 'cases.json') for name in ('real_inputs', 'long_inputs')},
        'scope': 'Snapshot only; no rerun/repair of experiments. Counts include the whole plan. Missing, unsupported and each budget failure remain distinct. Within-stage matched costs require all declared repetitions on both methods; no success-only universal winner. Timing differences below 5% are not ranked.',
        'sampling_scope': 'Synthetic: three distinct draws replayed for two latency repetitions. Real stages: one cold then the declared number of independently seeded reused complete draws per request/backend; draws are not independent requests. Narrow/wide are changed targets sharing one frozen q.',
        'censoring_scope': 'Conditional mean/p95 of observed warm draws are preserved per request. Headline warm metrics require every declared warm draw returned/verified before any budget failure. Each stage settings supplies its worker and RSS limits. A worker timeout includes startup/verification and is not assigned as the duration of an unreturned draw. Raw returned/verified counts remain separate from within-budget counts after a buffered budget violation.',
        'proposal_timing_scope': 'Phase3 compares complete sampling API wall time. proposal_seconds can include different internal checker work across proposals; it is a diagnostic component, not a pure numeric-kernel speedup. Cold costs additionally include constructor/precomputation.',
        'acceptance_scope': 'Use repeat0 only. Old proposals are full-path attempts; adaptive proposals count completed shared-rhythm accept/reject attempts, possibly stopped early. Full/partial candidates and spans sampled remain separate. Accepted decisions may precede delivery; verified output counts remain separate.'}
    csv_path = output.with_suffix('.csv')
    _write_csv(csv_path, report)
    figures_dir = output.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    report['figures'] = _figures(report, figures_dir) if plots else []
    lines = ['# 共享节奏扩展实验结果', '', f'生成时间（UTC）：{report["generated_at_utc"]}。这是原始记录的只读快照；运行中的阶段会随重建更新。', '',
        '| 目录 | 已记录/计划 | 状态 |', '|---|---:|---|']
    for name, stage in report['stages'].items():
        lines.append(f'| `{name}` | {stage.get("recorded_rows", 0)}/{stage.get("planned_rows", "未建计划")} | {_status_text(stage.get("status_counts", {})) if stage.get("planned_rows") else "尚未开始"} |')
    lines += ['', 'unsupported 是适用范围限制；internal_budget、rss_limit、timeout 分别是内部预算、实际内存和整个 worker 时限。未测量项不是零，也不计作另一方法胜出。', '',
        '冷成本包含构造、所需目标配分函数和第一个完整采样调用；拒绝方法构造中的提议预计算已包含。暖成本是同一引擎后续完整采样调用，包含所有拒绝、诊断和循环内进度记录。独立外部 checker、进程启动及调用间日志只计入 worker 时限，不混入求解器 API 时间。', '',
        '第三阶段 real_proposal 比较完整采样 API 的 wall time。各提议路径的 proposal_seconds 可能包含不同的内部 checker 工作，因此只能作为内部耗时分项，不能称为纯数值 kernel 加速。预算在某个操作首次失败后，缓冲区里随后返回的操作也全部排除出预算内成本和样本统计；原始返回/校验数量另行保留。', '']
    for name, model in report['frozen_inputs'].items():
        if model['status'] != 'completed':
            lines += [f'`{name}` 尚未冻结模型概率。', '']
            continue
        later = model['later_provider_seconds']
        lines += [f'`{name}`：{model["requests"]} 个请求、{model["works"]} 首作品，模型调用 {model["model_calls"]} 次。首个 provider 调用 {_format(model["first_provider_seconds"], scale=1000)} ms；后续 {later["available"]} 次为 {_format(later["minimum"], scale=1000)}–{_format(later["maximum"], scale=1000)} ms。模型加载/CUDA 初始化记录为 {", ".join(_format(t) for t in model["model_loading_seconds"])} s；首例 provider 预处理另为 {_format(model["first_provider_preprocessing_seconds"], scale=1000)} ms。没有 warmup，首例含首次 forward 初始化，不能当作稳态模型时延。', '']
    for name, stage in report['stages'].items():
        if not stage.get('planned_rows'):
            continue
        p = stage['partition_audit']
        lines += [f'## {name}', '', f'{stage["target_cases"]} 个目标规格，{stage["q_input_count"]} 个共享 q 输入键，{stage["work_count"]} 首真实作品；记录 {stage["recorded_rows"]}/{stage["planned_rows"]}。', '',
            f'本阶段每个 worker 时限为 {_format(stage["settings"]["worker_seconds"], digits=0)} 秒，RSS 上限为 {_format(stage["settings"].get("rss_limit_mib"), digits=0)} MiB；两者均取自该阶段实际配置。', '',
            f'返回目标 Z 的记录 {p["target_z_records"]} 条，覆盖 {p["cases_with_target_z"]} 个目标，其中 {p["cases_with_multiple_target_backends"]} 个目标有至少两个不同精确后端可核对；最大 logZ 差 {format(p["maximum_log_z_spread"], ".4g") if p["maximum_log_z_spread"] is not None else "未返回"}，错误 {len(p["errors"])}。提议 Z 从未作为原软目标 Z。已独立校验返回 draw {stage["verified_draws_including_timing_repeats"]} 次（合成轮包含计时复测，不能当作独立样本数）。', '']
        if name == 'synthetic':
            lines += ['每个输入用两个相同随机工作负载的计时重复；每重复三个不同 draw seed。下表先对每输入的计时重复取中位数，再对相应两个 q 输入取中位数。暖指标为后两次完整 draw 的均值；缺测保留实际状态。', '', '### unknown / β=0', '']
            _table_groups(lines, [g for g in stage['groups'] if g['visibility'] == 'unknown' and g['beta'] == 0.], synthetic=True)
            result, candidate_wins, reference_wins = _narrative_pairs(stage, 'unknown', 0., 'product_multi', 'onset_reset')
            lines += ['', f'上述同输入冷成本比较：可配对 {result["available"]}/{result["planned"]}；product_multi/onset_reset 的加速比中位数 {_format(result["median"])}。超过5%的 onset_reset 优势有 {candidate_wins} 个，反向优势 {reference_wins} 个；未返回的高R乘积状态结果没有被计为数值速度优势。', '', '### known / β=0.02', '']
            _table_groups(lines, [g for g in stage['groups'] if g['visibility'] == 'known' and g['beta'] == .02], synthetic=True)
            result, candidate_wins, reference_wins = _narrative_pairs(stage, 'known', .02, 'onset_rejection_boundary', 'product_multi')
            lines += ['', f'标准 product_multi 在节奏已知时使用独立链分解。同输入冷成本可配对 {result["available"]}/{result["planned"]}；boundary rejection/product_multi 比值中位数 {_format(result["median"])}，标准独立链超过5%更快的输入 {candidate_wins} 个，反向 {reference_wins} 个。这限制了“重置分解处处更快”的表述；标准乘积链、已知模板分解和通用缓存本身不是新的贡献。', '']
        else:
            draw_counts = sorted({r['requested_draws'] for r in stage['per_request_backend_repeat']})
            declared = '/'.join(str(n) for n in draw_counts)
            reused = '/'.join(str(max(0, n - 1)) for n in draw_counts)
            lines += [f'每个请求/backend 单个引擎共计划 {declared} 次采样：一次冷启动，加 {reused} 次独立 seed 的复用采样。cold mean/p95 跨请求计算；每请求只有一次冷样本，不能声称有请求内冷p95。¹暖栏分别为完整请求内全部计划复用 draw 的 mean/p95，再跨请求取中位数；小样本p95只是描述量。未完成请求的已返回 draw 条件统计保留在CSV，不填补被删失的尾部。²接受率逐请求取中位数，只用repeat0；提前拒绝的分母是完整接受/拒绝决策的节奏尝试，不是完整路径数。', '']
            _table_groups(lines, stage['groups'])
            failures = Counter((f['record_status'], f['failed_operation']) for f in stage['failures'])
            lines += ['', '失败/适用范围及所在操作：' + '；'.join(f'{status} @ {operation}: {n}' for (status, operation), n in sorted(failures.items(), key=lambda pair: str(pair[0]))) + '。', '',
                f'{_format(stage["settings"]["worker_seconds"], digits=0)} 秒是整个worker时限，未返回的单个draw没有被填为该时限；accepted decision 与已交付且校验通过的样本分开记录。窄/宽音域若共享q，它们仍是不同目标；不可要求其配分函数相等，也不可把两份规格计作两次模型调用。', '']
        figure = f'figures/shared_rhythm_extension_{name}.png'
        if str((figures_dir / Path(figure).name).resolve()) in report['figures']:
            lines += [f'![{name} 成本快照]({figure})', '']
    lines += ['## 范围与复现', '',
        '真实输入来自既有人工对齐的 validation 数据，模型由同一validation划分选择；没有新增训练或使用test。公开HOLD的发声音高是明确的边界条件，可能携带缺口内延续音高。新请求仅依据可见上下文筛选，但历史源窗口预处理曾按整窗活动量筛选。L16多段与训练的两段L32掩码不同，且首个缺口从0开始；成功求解不证明音乐质量优于基线或未见分布泛化。', '',
        'L32同时使用K8和首8格partial，保持25%公开但改变绝对公开格数；这是扩展规模实验，不是严格单变量消融。宽音域只扩大spec.pitches，原full130 q、观察和边界不变；新增方法的成本是否下降必须依据本轮匹配记录，不从算法名称推断。未开展真人听评。', '',
        f'[逐请求/backend CSV]({csv_path.name}) · [完整结构化汇总]({output.with_suffix(".json").name})。原始记录和历史 SHARED_RHYTHM_RESULTS.md 保持原样。', '',
        '```bash', 'taskset -c 5 scripts/python.sh -m tri.evaluation.shared_rhythm_extension_delivery', '```', '']
    output.write_text('\n'.join(lines), encoding='utf-8')
    atomic_json(output.with_suffix('.json'), report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', default='runs/shared_rhythm_extension')
    parser.add_argument('--output', default='reports/SHARED_RHYTHM_EXTENSION_RESULTS.md')
    parser.add_argument('--sources', nargs='+', default=DEFAULT_SOURCES)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    report = build_delivery(args.study_root, args.output, sources=args.sources, plots=not args.no_plots)
    print(json.dumps({name: {key: stage.get(key) for key in ('recording_status', 'recorded_rows', 'planned_rows')}
                      for name, stage in report['stages'].items()}))


if __name__ == '__main__':
    main()
