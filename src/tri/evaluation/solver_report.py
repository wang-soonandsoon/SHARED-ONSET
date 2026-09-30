"""Reproducible same-target solver tables and scientific figures.

Missing operations are never replaced by their timeout budget. Comparisons use
matched cases and retain the availability denominator and source run for each
number. This module only reads benchmark outputs; it never starts inference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np

from tri.runtime import atomic_json


OPERATIONS = ('cold_partition', 'cold_full_sample', 'warm_full_samples', 'query_setup',
              'clamp_queries', 'repeated_identical_clamps', 'token_marginal')
SPEED_METRICS = ('cold_partition_seconds', 'cold_full_sample_seconds',
                 'cold_full_sample_query_seconds', 'warm_full_sample_median_seconds',
                 'clamp_total_query_seconds', 'token_marginal_seconds',
                 'peak_rss_mib', 'incremental_peak_rss_mib')
SETTING_KEYS = ('L', 'D', 'K', 'visible_fraction_first_span', 'guard',
                'max_adjacent_interval', 'expected_support')
LIMIT_STATUSES = frozenset(('rss_limit', 'internal_budget', 'timeout'))
SOURCE_SETS = ('baselines', 'optimizations', 'template_refinement')
ABLATION_PAIRS = (('paired', 'paired_reuse'), ('paired_reuse', 'paired_sparse'),
                  ('product_chain', 'product_reuse'), ('product_reuse', 'paired_sparse'),
                  ('template', 'template_stream'))


def _over_budget_operation(row, operation):
    return row.get('failed_operation') == operation and row.get('status') in LIMIT_STATUSES


def _read_rows(path):
    """Read a consistent prefix if the writer is currently appending a line."""
    rows = []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    ignored_tail = False
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b'\n'):
                raise ValueError(f'corrupt benchmark record in {path}, line {index + 1}')
            ignored_tail = True
    keys = [(row['case_id'], row['backend']) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f'duplicate case/backend rows in {path}')
    return rows, ignored_tail


def _summary(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return {'n': 0, 'median': None, 'minimum': None, 'maximum': None, 'mean': None}
    return {'n': len(values), 'median': float(np.median(values)), 'minimum': min(values),
            'maximum': max(values), 'mean': float(np.mean(values))}


def _times(values):
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]
    result = [float(v) for v in values]
    if any(not math.isfinite(v) or v < 0 for v in result):
        raise ValueError('benchmark time must be finite and nonnegative')
    return result


def _metrics(row):
    result = {}
    for name in ('peak_rss_mib', 'incremental_peak_rss_mib', 'baseline_rss_mib'):
        if name in row:
            result[name] = row[name]
    for operation, value in row.get('measurements', {}).items():
        # A worker can emit a numerical result just before the supervisor sees
        # that its peak RSS exceeds the cap. Keep the raw event in per_query,
        # but it is not an in-budget timing for aggregate/paired comparisons.
        if _over_budget_operation(row, operation):
            continue
        if value.get('seconds') is not None:
            result[f'{operation}_seconds'] = _times(value['seconds'])[0]
        for component in ('construct_seconds', 'query_seconds'):
            seconds = value.get(component)
            if seconds is not None and not isinstance(seconds, list):
                result[f'{operation}_{component}'] = _times(seconds)[0]
        if operation == 'cold_full_sample':
            # Report the whole operation too, but do not attribute independent
            # verification time to the inference implementation.
            result['cold_full_sample_operation_seconds'] = value.get('seconds')
            if value.get('construct_seconds') is not None and value.get('query_seconds') is not None:
                result['cold_full_sample_seconds'] = value['construct_seconds'] + value['query_seconds']
        if operation == 'warm_full_samples':
            times = _times(value.get('sample_seconds'))
            if times:
                result['warm_full_sample_median_seconds'] = float(np.median(times))
                result['warm_full_sample_total_seconds'] = sum(times)
        if operation in ('clamp_queries', 'repeated_identical_clamps'):
            times = _times(value.get('query_seconds'))
            if times:
                prefix = 'clamp' if operation == 'clamp_queries' else 'repeated_clamp'
                result[f'{prefix}_total_query_seconds'] = sum(times)
                result[f'{prefix}_median_query_seconds'] = float(np.median(times))
    return result


def _availability(row, operation):
    if row is None:
        return 'pending'
    if _over_budget_operation(row, operation):
        return row['status']
    if operation in row.get('measurements', {}):
        return 'measured'
    partition = row.get('measurements', {}).get('cold_partition', {})
    if row.get('status') == 'zero_mass' and float(partition.get('log_z', 0.)) == -math.inf:
        return 'not_applicable_zero_mass'
    if row.get('failed_operation') == operation:
        return row['status']
    if row.get('failed_operation'):
        return f'not_reached_after_{row["failed_operation"]}'
    return 'not_recorded'


def _csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (list, tuple, dict)) else value for key, value in row.items()})


def _load_sources(study_root, source_sets=SOURCE_SETS):
    sources, rows, planned = [], [], []
    for source_set in source_sets:
        directory = study_root / source_set
        path = directory / 'results.jsonl'
        if not path.exists():
            sources.append({'source_set': source_set, 'present': False, 'recorded_rows': 0})
            continue
        observed, ignored_tail = _read_rows(path)
        config_path = directory / 'config.json'
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        status_path = directory / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        metadata = {row['case_id']: row['metadata'] for row in observed}
        manifest_path = Path(config.get('manifest', study_root / 'cases/cases.json'))
        if manifest_path.exists():
            for entry in json.loads(manifest_path.read_text())['cases']:
                metadata[entry['case_id']] = entry['metadata']
        case_ids = config.get('cases', list(dict.fromkeys(row['case_id'] for row in observed)))
        backends = config.get('backends', list(dict.fromkeys(row['backend'] for row in observed)))
        for row in observed:
            if row['case_id'] not in case_ids or row['backend'] not in backends:
                raise ValueError('observed row is outside the saved case/backend plan')
            if row['metadata'] != metadata[row['case_id']]:
                raise ValueError('observed row metadata differs from the frozen case manifest')
            row = dict(row)
            row['source_set'] = source_set
            row['metrics'] = _metrics(row)
            rows.append(row)
        for case_id in case_ids:
            if case_id not in metadata:
                raise ValueError(f'missing metadata for planned case {case_id}')
            for backend in backends:
                planned.append({'source_set': source_set, 'case_id': case_id,
                                'backend': backend, 'metadata': metadata[case_id]})
        sources.append({'source_set': source_set, 'present': True, 'path': str(path),
                        'recorded_rows': len(observed), 'planned_rows': len(case_ids) * len(backends),
                        'planned_cases': len(case_ids), 'backends': backends,
                        'status': status.get('status', 'unknown'), 'ignored_incomplete_tail': ignored_tail,
                        'config': config})
    if not rows:
        raise FileNotFoundError('No solver study result rows are available')
    return sources, rows, planned


def _correctness(rows):
    from tri.evaluation.solver_study import compare_rows
    compared = []
    for row in rows:
        checked = dict(row)
        checked['backend'] = f'{row["source_set"]}/{row["backend"]}'
        compared.append(checked)
    errors = compare_rows(compared)
    buckets = defaultdict(list)
    sample_errors, full_mass_errors, warm_errors = [], [], []
    for row in rows:
        for operation in ('cold_partition', 'clamp_queries', 'token_marginal'):
            result = row.get('measurements', {}).get(operation)
            if result is None:
                continue
            values = result['log_probs'] if operation == 'token_marginal' else result['log_z']
            values = values if isinstance(values, list) else [values]
            for index, value in enumerate(values):
                buckets[(row['case_id'], operation, index)].append(float(value))
        cold = row.get('measurements', {}).get('cold_full_sample', {}).get('draw', {})
        if cold.get('log_probability_error') is not None:
            sample_errors.append(float(cold['log_probability_error']))
        if cold.get('log_clamped_partition_error') is not None:
            full_mass_errors.append(float(cold['log_clamped_partition_error']))
        warm = row.get('measurements', {}).get('warm_full_samples', {})
        if warm.get('maximum_log_probability_error') is not None:
            warm_errors.append(float(warm['maximum_log_probability_error']))
        first = row.get('measurements', {}).get('clamp_queries')
        repeated = row.get('measurements', {}).get('repeated_identical_clamps')
        if first and repeated:
            if first['evidence'] != repeated['evidence']:
                errors.append({'case_id': row['case_id'], 'backend': row['backend'], 'error': 'repeated clamp identity mismatch'})
            elif len(first['log_z']) != len(repeated['log_z']) or any(
                    float(a) != float(b) and not (math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= 1e-8)
                    for a, b in zip(first['log_z'], repeated['log_z'])):
                errors.append({'case_id': row['case_id'], 'backend': row['backend'], 'error': 'repeated clamp value mismatch'})
    coverage = {}
    for operation in ('cold_partition', 'clamp_queries', 'token_marginal'):
        queries = {key: values for key, values in buckets.items() if key[1] == operation}
        multiple = {key: values for key, values in queries.items() if len(values) >= 2}
        deltas = [max(values) - min(values) for values in multiple.values() if all(math.isfinite(v) for v in values)]
        coverage[operation] = {'observed_cases': len({key[0] for key in queries}),
                               'cases_with_two_or_more_solvers': len({key[0] for key in multiple}),
                               'scalar_queries_with_two_or_more_solvers': len(multiple),
                               'maximum_finite_difference': max(deltas) if deltas else None,
                               'all_zero_mass_scalar_queries': sum(all(v == -math.inf for v in values) for values in multiple.values())}
    return {'passed_on_available_comparisons': not errors, 'errors': errors, 'coverage': coverage,
            'independent_cold_sample_checks': len(sample_errors),
            'maximum_cold_log_probability_error': max(sample_errors) if sample_errors else None,
            'maximum_cold_clamped_partition_error': max(full_mass_errors) if full_mass_errors else None,
            'maximum_warm_log_probability_error': max(warm_errors) if warm_errors else None,
            'absolute_tolerance': 1e-8,
            'scope': 'Available results only; absent/budget-stopped operations are not correctness passes.'}


def _query_table(rows, planned):
    lookup = {(r['source_set'], r['case_id'], r['backend']): r for r in rows}
    output = []
    for plan in planned:
        row = lookup.get((plan['source_set'], plan['case_id'], plan['backend']))
        meta = plan['metadata']
        for operation in OPERATIONS:
            value = (row or {}).get('measurements', {}).get(operation, {})
            query_times = value.get('query_seconds')
            item = {'source_set': plan['source_set'], 'case_id': plan['case_id'], 'backend': plan['backend'],
                    'family': meta.get('family'), 'stage': meta.get('stage'), 'kind': meta.get('kind'),
                    **{key: meta.get(key) for key in SETTING_KEYS}, 'seed': meta.get('seed'),
                    'operation': operation, 'row_status': (row or {}).get('status', 'pending'),
                    'availability': _availability(row, operation), 'failed_operation': (row or {}).get('failed_operation'),
                    'seconds': value.get('seconds'), 'construct_seconds': value.get('construct_seconds'),
                    'query_seconds': query_times if not isinstance(query_times, list) else None,
                    'query_seconds_list': query_times if isinstance(query_times, list) else None,
                    'query_seconds_total': sum(query_times) if isinstance(query_times, list) else query_times,
                    'sample_seconds_list': value.get('sample_seconds'),
                    'sample_seconds_median': float(np.median(value['sample_seconds'])) if value.get('sample_seconds') else None,
                    'solver_construct_plus_query_seconds': (value.get('construct_seconds', 0.) + query_times)
                       if value.get('construct_seconds') is not None and isinstance(query_times, (int, float)) else None,
                    'query_evidence': value.get('evidence'), 'log_z': value.get('log_z'),
                    'marginal_variable': value.get('variable'), 'marginal_domain': value.get('domain'),
                    'marginal_log_probs': value.get('log_probs'),
                    'operation_cumulative_peak_rss_mib': value.get('peak_rss_mib'),
                    'worker_peak_rss_mib': (row or {}).get('peak_rss_mib'),
                    'worker_incremental_peak_rss_mib': (row or {}).get('incremental_peak_rss_mib'),
                    'baseline_rss_mib': (row or {}).get('baseline_rss_mib'),
                    'error': (row or {}).get('error')}
            output.append(item)
    return output


def _availability_table(query_rows):
    groups = defaultdict(list)
    for row in query_rows:
        groups[(row['source_set'], row['family'], row['backend'], row['operation'])].append(row)
    output = []
    for (source, family, backend, operation), group in groups.items():
        counts = Counter(row['availability'] for row in group)
        applicable = len(group) - counts['not_applicable_zero_mass']
        output.append({'source_set': source, 'family': family, 'backend': backend, 'operation': operation,
                       'planned_cases': len(group), 'applicable_cases': applicable,
                       'measured_cases': counts['measured'], 'fraction_measured': counts['measured'] / applicable if applicable else None,
                       'statuses': dict(counts)})
    return output


def _sweep_groups(rows, planned):
    lookup = {(r['source_set'], r['case_id'], r['backend']): r for r in rows}
    groups = defaultdict(list)
    for plan in planned:
        meta = plan['metadata']
        if meta['family'] != 'controlled':
            continue
        for sweep in meta['sweeps']:
            key = (plan['source_set'], plan['backend'], sweep, *(meta.get(k) for k in SETTING_KEYS))
            groups[key].append((plan, lookup.get((plan['source_set'], plan['case_id'], plan['backend']))))
    result = []
    for key, group in groups.items():
        source, backend, sweep, *setting = key
        observed = [row for _, row in group if row is not None]
        item = {'source_set': source, 'backend': backend, 'sweep': sweep,
                **dict(zip(SETTING_KEYS, setting)), 'planned_seeds': len(group),
                'observed_seeds': len(observed), 'seeds': [plan['metadata']['seed'] for plan, _ in group],
                'case_ids': [plan['case_id'] for plan, _ in group],
                'row_status_counts': dict(Counter(row['status'] if row else 'pending' for _, row in group)),
                'operations': {}, 'metrics': {}}
        for operation in OPERATIONS:
            item['operations'][operation] = dict(Counter(_availability(row, operation) for _, row in group))
        for metric in SPEED_METRICS:
            item['metrics'][metric] = _summary([row['metrics'].get(metric) for row in observed])
        result.append(item)
    return result


def _matched_speedups(rows, planned, *, pairs=None):
    """Shared pair logic for original paired comparisons and direct ablations.

    References need not be paired. Both candidate and reference must plan a
    case before it enters the availability denominator; timing intersections
    then use only their recorded, in-budget values for that same operation.
    """
    observed = {(row['source_set'], row['case_id'], row['backend']): row for row in rows}
    source_backends = sorted({(row['source_set'], row['backend']) for row in planned})
    source_order = list(dict.fromkeys(row['source_set'] for row in planned))
    groups = defaultdict(list)
    for plan in planned:
        if plan['metadata'].get('expected_support') not in ('feasible', True):
            continue
        family = plan['metadata']['family']
        groups[(plan['source_set'], plan['backend'], family, None)].append(plan)
        if family == 'learned':
            groups[(plan['source_set'], plan['backend'], family, plan['metadata'].get('stage'))].append(plan)
    result = []
    for (source, backend, family, stage), plans in groups.items():
        references = ([reference for reference, candidate in pairs if candidate == backend]
                      if pairs is not None else ([] if backend == 'paired' else ['paired']))
        for reference in references:
            reference_source = next((location for location in (source, *source_order)
                                     if (location, reference) in source_backends), None)
            if reference_source is None:
                continue
            reference_plans = groups.get((reference_source, reference, family, stage), [])
            reference_ids = {plan['case_id'] for plan in reference_plans}
            common_plans = [plan for plan in plans if plan['case_id'] in reference_ids]
            for metric in SPEED_METRICS:
                values = []
                reference_available = candidate_available = 0
                for plan in common_plans:
                    ref = observed.get((reference_source, plan['case_id'], reference))
                    candidate = observed.get((source, plan['case_id'], backend))
                    a = ref['metrics'].get(metric) if ref else None
                    b = candidate['metrics'].get(metric) if candidate else None
                    # Whole-worker memory ratios are only comparable for
                    # complete workflows, never an early failure vs seven ops.
                    if metric.endswith('rss_mib'):
                        if not ref or ref['status'] != 'completed':
                            a = None
                        if not candidate or candidate['status'] != 'completed':
                            b = None
                    valid_a = a is not None and math.isfinite(float(a)) and a > 0
                    valid_b = b is not None and math.isfinite(float(b)) and b > 0
                    reference_available += valid_a
                    candidate_available += valid_b
                    if valid_a and valid_b:
                        values.append({'case_id': plan['case_id'], 'reference': a, 'candidate': b, 'ratio': a / b})
                ratios = [row['ratio'] for row in values]
                result.append({'source_set': source, 'reference_source_set': reference_source,
                               'reference': reference, 'candidate': backend, 'family': family, 'stage': stage,
                               'metric': metric, 'planned_feasible_cases': len(common_plans),
                               'reference_planned_feasible_cases': len(reference_plans),
                               'candidate_planned_feasible_cases': len(plans),
                               'reference_available': reference_available, 'candidate_available': candidate_available,
                               'matched_cases': len(values), 'median_ratio': float(np.median(ratios)) if ratios else None,
                               'geometric_mean_ratio': float(np.exp(np.mean(np.log(ratios)))) if ratios else None,
                               'minimum_ratio': min(ratios) if ratios else None, 'maximum_ratio': max(ratios) if ratios else None,
                               'matched_reference_median': float(np.median([v['reference'] for v in values])) if values else None,
                               'matched_candidate_median': float(np.median([v['candidate'] for v in values])) if values else None,
                               'matched_values': values,
                               'interpretation': 'reference/candidate; >1 favors candidate on this matched intersection only; availability denominators use jointly planned feasible cases'})
    return result


def _plots(output, sweep_groups, availability):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 180})
    identities = list(dict.fromkeys((g['source_set'], g['backend']) for g in sweep_groups))
    known_backends = Counter(backend for _, backend in identities)
    labels = {(source, backend): backend if known_backends[backend] == 1 else f'{backend} ({source})'
              for source, backend in identities}
    colors = {identity: plt.get_cmap('tab10')(i % 10) for i, identity in enumerate(identities)}
    artifacts = []
    metric_titles = [('cold_partition_seconds', 'Cold partition (construction + log Z)'),
                     ('cold_full_sample_seconds', 'Cold full sample (construction + query)'),
                     ('warm_full_sample_median_seconds', 'Warm full sample (per-draw median)'),
                     ('token_marginal_seconds', 'Token marginal after clamp workload')]
    for sweep, xkey, xlabel in [('length', 'L', 'Cells per span L (K=L/4)'),
                                ('pitch_states', 'D', 'Sounding states D (including silence)'),
                                ('visibility', 'visible_fraction_first_span', 'Observed fraction in span A (B unknown)')]:
        subset = [g for g in sweep_groups if g['sweep'] == sweep]
        if not subset:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True)
        for ax, (metric, title) in zip(axes.flat, metric_titles):
            for identity in identities:
                group = sorted([g for g in subset if (g['source_set'], g['backend']) == identity], key=lambda g: g[xkey])
                if not group:
                    continue
                x = [g[xkey] for g in group]
                summaries = [g['metrics'][metric] for g in group]
                y = [s['median'] if s['median'] is not None and s['median'] > 0 else np.nan for s in summaries]
                ax.plot(x, y, marker='o', markersize=3, linewidth=1.1, label=labels[identity], color=colors[identity])
                for point, summary, setting in zip(x, summaries, group):
                    if summary['n'] and summary['median'] > 0:
                        ax.vlines(point, summary['minimum'], summary['maximum'], color=colors[identity], linewidth=.7)
                        if summary['n'] < setting['planned_seeds']:
                            ax.annotate(f'n={summary["n"]}', (point, summary['median']), fontsize=6, xytext=(2, 3), textcoords='offset points', color=colors[identity])
                    else:
                        transform = blended_transform_factory(ax.transData, ax.transAxes)
                        ax.plot(point, .035 + .018 * identities.index(identity), 'x', transform=transform,
                                markersize=4, color=colors[identity], clip_on=False)
            ax.set_yscale('log')
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Seconds (log scale)')
            ax.grid(True, which='major', alpha=.22)
            ax.set_xticks(sorted({g[xkey] for g in subset}))
        handles, names = axes.flat[0].get_legend_handles_labels()
        legend_columns = min(5 if len(names) >= 9 else 4, len(names))
        fig.legend(handles, names, loc='lower center', ncol=legend_columns, frameon=False, bbox_to_anchor=(.5, .015))
        fig.suptitle(f'Exact solver {sweep} sweep: frozen q and identical target', y=.98)
        legend_rows = math.ceil(len(names) / legend_columns)
        caption_y, footer = (.12, .18) if legend_rows > 2 else (.085, .14)
        fig.text(.5, caption_y, 'Median over available probability seeds; whiskers=min/max. Bottom x=no completed measurement (not a time).\nPartial n is labelled. See availability.csv for budget/timeout/not-reached counts; these curves are not unmatched winner rankings.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, footer, 1, .95))
        for extension in ('png', 'pdf'):
            dest = output / f'scaling_{sweep}.{extension}'
            fig.savefig(dest)
            artifacts.append(str(dest))
        plt.close(fig)

    families = [f for f in ('tiny', 'controlled', 'learned') if any(r['family'] == f for r in availability)]
    if families:
        identities = list(dict.fromkeys((r['source_set'], r['backend']) for r in availability))
        fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), max(4, .45 * len(identities) + 2)), squeeze=False)
        for ax, family in zip(axes[0], families):
            matrix = np.full((len(identities), len(OPERATIONS)), np.nan)
            text_values = {}
            for row in availability:
                if row['family'] == family:
                    i = identities.index((row['source_set'], row['backend']))
                    j = OPERATIONS.index(row['operation'])
                    matrix[i, j] = row['fraction_measured'] if row['fraction_measured'] is not None else np.nan
                    text_values[(i, j)] = f'{row["measured_cases"]}/{row["applicable_cases"]}'
            ax.imshow(matrix, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
            for (i, j), value in text_values.items():
                ax.text(j, i, value, ha='center', va='center', fontsize=7, color='white' if matrix[i, j] > .65 else 'black')
            ax.set_xticks(range(len(OPERATIONS)), ['log Z', 'cold sample', 'warm samples', 'query setup', 'clamps', 'repeat clamps', 'marginal'], rotation=45, ha='right')
            ax.set_yticks(range(len(identities)), [labels.get(identity, f'{identity[1]} ({identity[0]})') for identity in identities])
            ax.set_title(f'{family}: measured / applicable cases')
        fig.suptitle('Operation availability: previous measurements survive later failures')
        fig.text(.5, .015, 'Post-zero-mass operations are not applicable. Missing rows and early failures remain in applicable denominators.\nPeak memory is cumulative within a worker; operation figures are not isolated allocation peaks.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .09, 1, .94))
        for extension in ('png', 'pdf'):
            dest = output / f'availability.{extension}'
            fig.savefig(dest)
            artifacts.append(str(dest))
        plt.close(fig)
    return artifacts


def build_solver_report(study_root='runs/solver_study', output='runs/solver_study/analysis', *, plots=True, sources=None):
    study_root, output = Path(study_root).resolve(), Path(output).resolve()
    source_names = SOURCE_SETS if sources is None else ((sources,) if isinstance(sources, str) else tuple(sources))
    if (not source_names or len(set(source_names)) != len(source_names) or
            any(not isinstance(name, str) or not name or name in ('.', '..') or Path(name).name != name for name in source_names)):
        raise ValueError('sources must be distinct immediate study subdirectory names')
    source_records, rows, planned = _load_sources(study_root, source_names)
    output.mkdir(parents=True, exist_ok=True)
    query_rows = _query_table(rows, planned)
    availability = _availability_table(query_rows)
    sweeps = _sweep_groups(rows, planned)
    speedups = _matched_speedups(rows, planned)
    ablations = _matched_speedups(rows, planned, pairs=ABLATION_PAIRS)
    correctness = _correctness(rows)
    backend_summaries = []
    for source, backend, family in sorted({(r['source_set'], r['backend'], r['metadata']['family']) for r in rows}):
        group = [r for r in rows if (r['source_set'], r['backend'], r['metadata']['family']) == (source, backend, family)]
        expected = [r for r in planned if (r['source_set'], r['backend'], r['metadata']['family']) == (source, backend, family)]
        backend_summaries.append({'source_set': source, 'backend': backend, 'family': family,
                                  'recorded_cases': len(group), 'planned_cases': len(expected),
                                  'status_counts': dict(Counter(r['status'] for r in group)),
                                  'failed_operation_counts': dict(Counter(r.get('failed_operation') for r in group if r.get('failed_operation'))),
                                  'metrics': {metric: _summary([r['metrics'].get(metric) for r in group]) for metric in SPEED_METRICS},
                                  'interpretation': 'Descriptive available-row summaries; compare methods using matched_speedups.'})
    _csv(output / 'per_query.csv', query_rows)
    _csv(output / 'availability.csv', availability)
    _csv(output / 'grouped_sweeps.csv', [{**{k: v for k, v in row.items() if k not in ('metrics', 'operations')},
        'operation_availability': row['operations'], **{f'{metric}_{stat}': value for metric, stats in row['metrics'].items()
                                                       for stat, value in stats.items()}} for row in sweeps])
    _csv(output / 'matched_speedups.csv', [{k: v for k, v in row.items() if k != 'matched_values'} for row in speedups])
    _csv(output / 'matched_pairs.csv', [{**{k: row[k] for k in ('source_set', 'reference_source_set', 'reference', 'candidate', 'family', 'stage', 'metric')}, **pair}
                                       for row in speedups for pair in row['matched_values']])
    _csv(output / 'ablation_speedups.csv', [{k: v for k, v in row.items() if k != 'matched_values'} for row in ablations])
    _csv(output / 'ablation_pairs.csv', [{**{k: row[k] for k in ('source_set', 'reference_source_set', 'reference', 'candidate', 'family', 'stage', 'metric')}, **pair}
                                        for row in ablations for pair in row['matched_values']])
    figures = _plots(output, sweeps, availability) if plots else []
    present = [source for source in source_records if source['present']]
    finished = all(source['status'] == 'completed' and source['recorded_rows'] == source['planned_rows']
                   and not source['ignored_incomplete_tail'] for source in present)
    actual_backends = sorted({row['backend'] for row in rows})
    report = {'status': 'completed' if finished else 'partial', 'study_root': str(study_root),
              'selected_sources': list(source_names),
              'optimization_source_present': any(s['present'] and s['source_set'] == 'optimizations' for s in source_records),
              'template_refinement_source_present': any(s['present'] and s['source_set'] == 'template_refinement' for s in source_records),
              'actual_backends': actual_backends,
              'has_optimization_backends': bool(set(actual_backends) & {'paired_reuse', 'paired_sparse', 'product_reuse'}),
              'has_template_stream_backend': 'template_stream' in actual_backends,
              'recorded_rows': len(rows), 'planned_rows_in_present_sources': len(planned),
              'sources': source_records, 'correctness': correctness, 'backend_summaries': backend_summaries,
              'availability': availability, 'controlled_sweeps': sweeps, 'matched_speedups': speedups,
              'ablation_speedups': ablations,
              'ablation_definitions': [{'reference': reference, 'candidate': candidate} for reference, candidate in ABLATION_PAIRS],
              'artifacts': {'summary': str(output / 'summary.json'), 'figures': figures,
                            'per_query': str(output / 'per_query.csv'), 'availability': str(output / 'availability.csv'),
                            'sweeps': str(output / 'grouped_sweeps.csv'), 'speedups': str(output / 'matched_speedups.csv'),
                            'matched_pairs': str(output / 'matched_pairs.csv'),
                            'ablation_speedups': str(output / 'ablation_speedups.csv'),
                            'ablation_pairs': str(output / 'ablation_pairs.csv')},
              'definitions': {'cold_full_sample_seconds': 'construct_seconds + query_seconds; independent token verifier excluded; operation total separately retained.',
                              'warm_full_sample_median_seconds': 'Within-case median of recorded subsequent full-sample queries; preprocessing is in cold measurement.',
                              'clamp_total_query_seconds': 'Sum over the predeclared distinct singleton/joint evidence queries; query_setup cost is separately retained.',
                              'repeated_identical_clamps': 'Repeated same evidence sequence; whether this is a cache hit depends on the backend.',
                              'token_marginal_seconds': 'Measured after the defined clamp workload, including whichever earlier query caches this backend actually retains.',
                              'memory': 'Actual worker peak RSS and baseline-subtracted peak. Operation RSS values are cumulative peaks, not per-operation isolated allocation.',
                              'ratio': 'paired / candidate on the exact shared set of measured feasible cases. Memory ratios require both full workflows completed.',
                              'ablation_ratio': 'Named reference / named candidate, using the same matching, availability and memory rules; reference need not be paired. These five comparisons separate reuse, sparse contraction and template budget correction.',
                              'source_selection': 'Only selected_sources are read. References prefer the candidate run when available, then another selected run; no measurements are borrowed from unselected pilot runs.',
                              'backend_presence': 'actual_backends is the sorted set with at least one recorded row in selected sources, regardless of outcome; presence does not imply completion. has_optimization_backends means any paired_reuse, paired_sparse or product_reuse row exists. Dedicated *_source_present flags describe selected subdirectories, not backend availability.',
                              'sweep_statistics': 'Median/min/max over available fixed probability seeds. Incomplete availability is explicit, never imputed with a time limit.',
                              'limits': 'Status names distinguish timeout, internal table/workspace budget, RSS limit, unsupported, zero mass, and implementation errors.'},
              'limitations': ['Measurements are from frozen validation/control inputs, not independent music quality or test-set evidence.',
                              'A backend stopping before a later operation has no measurement for that operation; correctness only covers available comparisons.',
                              'A recorded operation later flagged over budget retains its raw values, but is excluded from headline timing summaries and speedups; numerical agreement can still be checked.',
                              'No universal winner follows from successful-only or unmatched aggregate timing.',
                              'Three probability seeds measure variation across frozen targets, not repeated timing trials or confidence intervals.',
                              'Case visibility=1 means span A known while B remains unknown; the known-template crossover is shown by the visibility sweep.']}
    if not correctness['passed_on_available_comparisons']:
        report['status'] = 'correctness_failed'
    atomic_json(output / 'summary.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', default='runs/solver_study')
    parser.add_argument('--output', default='runs/solver_study/analysis')
    parser.add_argument('--sources', nargs='+', help='Only read these study subdirectories; default: baselines optimizations template_refinement')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    report = build_solver_report(args.study_root, args.output, plots=not args.no_plots, sources=args.sources)
    print(json.dumps({'status': report['status'], 'rows': report['recorded_rows'],
                      'correctness': report['correctness']['passed_on_available_comparisons'],
                      'artifacts': report['artifacts']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
