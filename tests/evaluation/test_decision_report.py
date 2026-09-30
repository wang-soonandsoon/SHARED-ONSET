from copy import deepcopy

import pytest

from tri.evaluation.decision_report import summarize


def row(backend, rep, seconds):
    return {'case_id': f'base_r{rep}', 'backend': backend, 'status': 'completed',
            'metadata': {'source_case_id': 'base', 'timing_repetition': rep,
                         'family': 'controlled', 'L': 8, 'D': 4, 'K': 2,
                         'query_labels': ['late_singleton'], 'expected_support': 'feasible'},
            'measurements': {'cold_partition': {'log_z': -10},
                             'clamp_queries': {'log_z': [-12], 'query_seconds': [seconds],
                                               'evidence': [{'y8': 0}]}}}


def test_repeat_aggregation_uses_target_medians_and_excludes_incomplete_repeats():
    rows = [row('product_reuse', j, t) for j, t in enumerate((1, 9, 5))]
    rows += [row('product_prefix', j, t) for j, t in enumerate((1, 3, 2))]
    result = summarize(rows)
    comparison = next(r for r in result['comparisons'] if r['reference'] == 'product_reuse' and r['family'] == 'controlled')
    assert comparison['matched_targets'] == 1
    assert comparison['median_ratio'] == 2.5
    partial = summarize(rows[:-1])
    comparison = next(r for r in partial['comparisons'] if r['reference'] == 'product_reuse' and r['family'] == 'controlled')
    assert comparison['matched_targets'] == 0 and comparison['median_ratio'] is None


def test_over_budget_returned_measurement_is_checked_but_not_timed():
    rows = [row(b, r, 1) for b in ('product_reuse', 'product_prefix') for r in range(3)]
    rows[-1].update(status='rss_limit', failed_operation='clamp_queries')
    result = summarize(rows)
    assert result['probabilities_agree_on_returned_results']
    assert len(result['query_rows']) == 5
    changed = deepcopy(rows)
    changed[-1]['measurements']['clamp_queries']['log_z'] = [-20]
    assert not summarize(changed)['probabilities_agree_on_returned_results']
    with pytest.raises(ValueError, match='Duplicate'):
        summarize(rows + [rows[0]])


def test_entire_missing_backend_keeps_frozen_opportunity_denominator():
    plans = [row(b, r, 1) for b in ('product_reuse', 'product_prefix') for r in range(3)]
    result = summarize(plans[:3], plans=plans)
    comparison = next(r for r in result['comparisons'] if r['reference'] == 'product_reuse' and r['family'] == 'controlled')
    assert result['planned_rows'] == 6 and result['recorded_rows'] == 3
    assert comparison['planned_targets'] == 1 and comparison['matched_targets'] == 0
