"""Aggregate fresh-process repeats before comparing equal-prefix solvers."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import median

from tri.evaluation.solver_report import _availability
from tri.evaluation.solver_study import compare_rows
from tri.runtime import atomic_json


PAIRS = (('product_reuse', 'product_prefix'), ('paired_reuse', 'product_prefix'),
         ('paired_sparse', 'product_prefix'))


def summarize(rows, repetitions=3, plans=None):
    seen = set()
    query_rows = []
    planned = defaultdict(set)
    plans = rows if plans is None else plans
    target_metadata = {row['metadata']['source_case_id']: row['metadata'] for row in plans}
    for plan in plans:
        planned[plan['backend']].add(plan['metadata']['source_case_id'])
    expected_repeats = set(range(repetitions))
    for row in rows:
        meta = row['metadata']
        key = (meta['source_case_id'], row['backend'], meta['timing_repetition'])
        if key in seen or key[-1] not in expected_repeats:
            raise ValueError('Duplicate or unexpected timing repetition')
        seen.add(key)
        measurement = row['measurements'].get('clamp_queries')
        if _availability(row, 'clamp_queries') != 'measured':
            continue
        if not (len(measurement['query_seconds']) == len(meta['query_labels']) == len(measurement['log_z'])):
            raise ValueError('Probe counts differ from fixed labels')
        for j, label in enumerate(meta['query_labels']):
            stats = measurement.get('probe_stats', [{}] * len(meta['query_labels']))[j]
            query_rows.append({'case_id': row['case_id'], 'source_case_id': key[0],
                               'backend': key[1], 'repetition': key[2], 'family': meta['family'],
                               'stage': meta.get('stage'), 'kind': meta.get('kind'),
                               'L': meta['L'], 'D': meta['D'], 'K': meta['K'],
                               'label': label, 'seconds': measurement['query_seconds'][j],
                               'zero_mass': float(measurement['log_z'][j]) == -math.inf,
                               'prefix_layers_reused': stats.get('prefix_layers_reused'),
                               'transfers_rebuilt': stats.get('transfers_rebuilt'),
                               'scalar_cache_hit': stats.get('scalar_cache_hit', stats.get('cache_hit', False)),
                               'evidence': measurement['evidence'][j]})
    groups = defaultdict(list)
    for row in query_rows:
        groups[(row['source_case_id'], row['backend'], row['label'])].append(row)
    aggregated = []
    for key, group in groups.items():
        complete = {row['repetition'] for row in group} == expected_repeats
        first = group[0]
        if any(row['evidence'] != first['evidence'] or row['zero_mass'] != first['zero_mass'] for row in group):
            raise ValueError('Repeated target has changed evidence or support')
        aggregated.append({k: first[k] for k in ('source_case_id', 'backend', 'label', 'family',
                                                'stage', 'kind', 'L', 'D', 'K', 'zero_mass')} |
                          {'completed_repetitions': len(group), 'all_repetitions_completed': complete,
                           'median_seconds': median(row['seconds'] for row in group) if complete else None,
                           'minimum_seconds': min(row['seconds'] for row in group),
                           'maximum_seconds': max(row['seconds'] for row in group),
                           'prefix_layers_reused': sorted({row['prefix_layers_reused'] for row in group
                                                         if row['prefix_layers_reused'] is not None})})
    lookup = {(r['source_case_id'], r['backend'], r['label']): r for r in aggregated}
    labels = sorted({label for row in plans for label in row['metadata']['query_labels']})
    comparisons = []
    for reference, candidate in PAIRS:
        common = sorted(planned[reference] & planned[candidate])
        for family in ('controlled', 'learned'):
            for label in labels:
                matched = []
                opportunities = []
                for target in common:
                    a, b = lookup.get((target, reference, label)), lookup.get((target, candidate, label))
                    meta = target_metadata[target]
                    if meta['family'] != family:
                        continue
                    opportunities.append(target)
                    if not a or not b or a['median_seconds'] is None or b['median_seconds'] is None:
                        continue
                    if a['median_seconds'] <= 0 or b['median_seconds'] <= 0:
                        raise ValueError('Cannot compare a nonpositive measured duration')
                    matched.append({'source_case_id': target,
                                    'reference_seconds': a['median_seconds'],
                                    'candidate_seconds': b['median_seconds'],
                                    'ratio': a['median_seconds'] / b['median_seconds'],
                                    'zero_mass': a['zero_mass']})
                comparisons.append({'reference': reference, 'candidate': candidate, 'family': family,
                                    'label': label, 'planned_targets': len(opportunities),
                                    'matched_targets': len(matched),
                                    'median_ratio': median(r['ratio'] for r in matched) if matched else None,
                                    'zero_mass_matched_targets': sum(r['zero_mass'] for r in matched),
                                    'matched_values': matched})
    errors = compare_rows(rows)
    return {'recorded_rows': len(rows), 'planned_rows': len(plans), 'unique_targets': len(target_metadata),
            'repetitions_per_target': repetitions, 'comparison_errors': errors,
            'probabilities_agree_on_returned_results': not errors,
            'status_counts': dict(Counter(row['status'] for row in rows)),
            'query_rows': query_rows, 'per_target': aggregated, 'comparisons': comparisons,
            'interpretation': 'For each unchanged target and backend take the median of three new-process timings, then form paired ratios. Missing repeats exclude that target from paired timing. A ratio above one favors the candidate. Process repeats are not extra targets or works.'}


def _csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)


def plot_queries(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = ('product_chain', 'product_reuse', 'product_prefix', 'paired_reuse', 'paired_sparse')
    labels = ('early_singleton', 'middle_singleton', 'late_singleton')
    panels = [('Control L32/D16/K8 (3 fixed targets)',
               lambda r: r['family'] == 'controlled' and r['L'] == 32 and r['D'] == 16)]
    panels += [(f'{stage}: unknown template (1 fixed target)',
                lambda r, stage=stage: r['family'] == 'learned' and r['stage'] == stage and r['kind'] == 'unknown')
               for stage in ('short', 'bar8', 'bar16')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (title, choose) in zip(axes.flat, panels):
        for backend in names:
            values = []
            for label in labels:
                times = [r['median_seconds'] * 1000 for r in rows if choose(r) and r['backend'] == backend
                         and r['label'] == label and r['median_seconds'] is not None]
                values.append(median(times) if times else math.nan)
            ax.plot(range(3), values, marker='o', label=backend)
        ax.set_title(title)
        ax.set_xticks(range(3), ('early', 'middle', 'late'))
        ax.set_ylabel('New conditional query / ms (log scale)')
        ax.set_yscale('log')
        ax.grid(alpha=.2)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=5, bbox_to_anchor=(.5, .04))
    fig.suptitle('Equal-prefix control: fixed q, same evidence, new conditional queries')
    fig.text(.5, .018, 'Three fresh-process repeats per target; medians, not independent music examples. Missing operations are not time limits.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .95))
    for suffix in ('png', 'pdf'):
        fig.savefig(output/f'prefix_queries.{suffix}', dpi=180)
    plt.close(fig)


def build(source='runs/decision_study/equal_prefix', output='runs/decision_study/query_analysis'):
    source, output = Path(source), Path(output)
    rows = [json.loads(line) for line in (source/'results.jsonl').read_text().splitlines()]
    configuration = json.loads((source/'config.json').read_text())
    manifest = json.loads(Path(configuration['manifest']).read_text())
    entries = {entry['case_id']: entry for entry in manifest['cases']}
    plans = [{'case_id': name, 'backend': backend, 'metadata': entries[name]['metadata']}
             for name in configuration['cases'] for backend in configuration['backends']]
    allowed = {(plan['case_id'], plan['backend']) for plan in plans}
    for row in rows:
        if (row['case_id'], row['backend']) not in allowed or row['metadata'] != entries[row['case_id']]['metadata']:
            raise ValueError('Returned row differs from the frozen plan')
    result = summarize(rows, plans=plans)
    status = json.loads((source/'status.json').read_text())
    result['status'] = status['status']
    if result['status'] == 'completed' and len(rows) != len(plans):
        raise ValueError('Completed study is missing planned rows')
    result['configuration'] = configuration
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'summary.json', result)
    _csv(output/'queries.csv', result['query_rows'])
    _csv(output/'targets.csv', result['per_target'])
    _csv(output/'comparisons.csv', result['comparisons'])
    plot_queries(output, result['per_target'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='runs/decision_study/equal_prefix')
    parser.add_argument('--out', default='runs/decision_study/query_analysis')
    args = parser.parse_args()
    report = build(args.source, args.out)
    print(json.dumps({key: report[key] for key in ('status', 'recorded_rows', 'unique_targets', 'comparison_errors')}))
