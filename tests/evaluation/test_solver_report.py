import copy
import csv
import json
from pathlib import Path

import pytest

from tri.evaluation.solver_report import build_solver_report


def measured(case, backend, seconds=1.):
    metadata = {'family': 'controlled', 'sweeps': ['length'], 'L': 8, 'D': 3, 'K': 2,
                'visible_fraction_first_span': 0., 'guard': 'NOTE', 'max_adjacent_interval': None,
                'expected_support': 'feasible', 'seed': int(case[-1])}
    partition = {'seconds': seconds, 'construct_seconds': seconds * .1,
                 'query_seconds': seconds * .9, 'log_z': -7., 'peak_rss_mib': 90.}
    sample = {'seconds': seconds * 2. + .3, 'construct_seconds': seconds * .1,
              'query_seconds': seconds * 1.9, 'peak_rss_mib': 92.,
              'draw': {'valid': True, 'log_probability_error': 0., 'log_clamped_partition_error': 0.}}
    clamps = {'seconds': seconds * .5, 'query_seconds': [seconds * .1, seconds * .2],
              'evidence': [{'y1': 0}, {'y2': 1}], 'log_z': [-9., '-inf'], 'peak_rss_mib': 93.}
    marginal = {'seconds': seconds, 'variable': 'y1', 'domain': [0, 1],
                'log_probs': [0., '-inf'], 'peak_rss_mib': 94.}
    return {'case_id': case, 'backend': backend, 'metadata': metadata, 'status': 'completed',
            'peak_rss_mib': 94., 'incremental_peak_rss_mib': 14., 'baseline_rss_mib': 80.,
            'measurements': {'cold_partition': partition, 'cold_full_sample': sample,
                             'warm_full_samples': {'seconds': seconds * 2., 'sample_seconds': [seconds, seconds * .5],
                                                   'maximum_log_probability_error': 0., 'peak_rss_mib': 92.},
                             'query_setup': copy.deepcopy(partition), 'clamp_queries': clamps,
                             'repeated_identical_clamps': copy.deepcopy(clamps), 'token_marginal': marginal}}


def source(root, name, rows, backends, cases=('case1', 'case2', 'case3'), status='completed'):
    out = root / name
    out.mkdir(parents=True)
    manifest = root / 'manifest.json'
    if not manifest.exists():
        manifest.write_text(json.dumps({'cases': [{'case_id': case, 'metadata': measured(case, 'paired')['metadata']}
                                                for case in cases]}))
    (out / 'config.json').write_text(json.dumps({'manifest': str(manifest), 'cases': list(cases), 'backends': backends}))
    (out / 'status.json').write_text(json.dumps({'status': status}))
    (out / 'results.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))


@pytest.fixture
def study(tmp_path):
    rows = [measured(f'case{i}', 'paired') for i in range(1, 4)]
    fast = measured('case1', 'product_chain', .2)
    timeout = measured('case2', 'product_chain', .1)
    timeout['measurements'] = {'cold_partition': timeout['measurements']['cold_partition']}
    timeout.update(status='timeout', failed_operation='cold_full_sample')
    limited = measured('case3', 'product_chain', .01)
    limited['measurements'] = {}
    limited.update(status='internal_budget', failed_operation='cold_partition')
    source(tmp_path, 'baselines', rows + [fast, timeout, limited], ['paired', 'product_chain'])
    return tmp_path


def test_report_preserves_earlier_measurements_and_uses_exact_matched_sets(study):
    result = build_solver_report(study, study / 'analysis', plots=False)
    assert result['status'] == 'completed'
    assert result['recorded_rows'] == 6 and result['optimization_source_present'] is False
    assert result['actual_backends'] == ['paired', 'product_chain']
    assert not result['has_optimization_backends'] and not result['has_template_stream_backend']
    assert result['correctness']['passed_on_available_comparisons']
    part = next(row for row in result['availability'] if row['backend'] == 'product_chain' and row['operation'] == 'cold_partition')
    assert part['measured_cases'] == 2 and part['applicable_cases'] == 3
    warm = next(row for row in result['availability'] if row['backend'] == 'product_chain' and row['operation'] == 'warm_full_samples')
    assert warm['measured_cases'] == 1 and warm['applicable_cases'] == 3
    speed = next(row for row in result['matched_speedups'] if row['candidate'] == 'product_chain' and row['metric'] == 'cold_full_sample_seconds')
    assert speed['matched_cases'] == 1 and speed['planned_feasible_cases'] == 3
    assert speed['median_ratio'] == pytest.approx(5.)
    assert speed['matched_values'][0]['case_id'] == 'case1'
    partition = next(row for row in result['matched_speedups'] if row['candidate'] == 'product_chain' and row['metric'] == 'cold_partition_seconds')
    assert partition['matched_cases'] == 2
    with Path(result['artifacts']['per_query']).open() as stream:
        table = list(csv.DictReader(stream))
    missing = next(row for row in table if row['backend'] == 'product_chain' and row['case_id'] == 'case2' and row['operation'] == 'cold_full_sample')
    assert missing['seconds'] == '' and missing['availability'] == 'timeout'
    group = next(row for row in result['controlled_sweeps'] if row['backend'] == 'product_chain')
    assert group['planned_seeds'] == 3 and group['metrics']['cold_full_sample_seconds']['n'] == 1


def test_partial_optimization_cross_compares_against_original_paired(study):
    source(study, 'optimizations', [measured('case1', 'paired_reuse', .1)], ['paired_reuse'], status='running')
    result = build_solver_report(study, study / 'analysis', plots=False)
    assert result['status'] == 'partial'
    assert result['recorded_rows'] == 7 and result['planned_rows_in_present_sources'] == 9
    assert result['optimization_source_present'] and result['has_optimization_backends']
    speed = next(row for row in result['matched_speedups'] if row['candidate'] == 'paired_reuse' and row['metric'] == 'cold_full_sample_seconds')
    assert speed['reference_source_set'] == 'baselines'
    assert speed['matched_cases'] == 1 and speed['median_ratio'] == pytest.approx(10.)
    coverage = next(row for row in result['availability'] if row['backend'] == 'paired_reuse' and row['operation'] == 'cold_partition')
    assert coverage['statuses']['pending'] == 2


def test_cross_run_numerical_disagreement_is_explicit(study):
    bad = measured('case1', 'paired_reuse', .1)
    bad['measurements']['cold_partition']['log_z'] += .01
    source(study, 'optimizations', [bad], ['paired_reuse'], status='running')
    result = build_solver_report(study, study / 'analysis', plots=False)
    assert result['status'] == 'correctness_failed'
    assert result['correctness']['errors']
    assert result['correctness']['coverage']['cold_partition']['maximum_finite_difference'] == pytest.approx(.01)


def test_negative_infinity_agrees_and_post_zero_operations_not_applicable(tmp_path):
    rows = []
    for backend in ('paired', 'product_chain'):
        row = measured('case1', backend)
        row['metadata']['expected_support'] = 'infeasible'
        row['status'] = 'zero_mass'
        row['measurements'] = {'cold_partition': {**row['measurements']['cold_partition'], 'log_z': '-inf'}}
        rows.append(row)
    source(tmp_path, 'baselines', rows, ['paired', 'product_chain'], cases=('case1',))
    # Source row metadata and manifest must describe the same frozen input.
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    manifest['cases'][0]['metadata']['expected_support'] = 'infeasible'
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    result = build_solver_report(tmp_path, tmp_path / 'analysis', plots=False)
    assert result['correctness']['passed_on_available_comparisons']
    assert result['correctness']['coverage']['cold_partition']['all_zero_mass_scalar_queries'] == 1
    warm = next(row for row in result['availability'] if row['operation'] == 'warm_full_samples')
    assert warm['applicable_cases'] == 0 and warm['fraction_measured'] is None
    assert result['matched_speedups'] == []


def test_live_incomplete_tail_is_ignored_without_modifying_raw_results(study):
    path = study / 'baselines/results.jsonl'
    with path.open('a') as stream:
        stream.write('{"in_progress":')
    before = path.read_bytes()
    result = build_solver_report(study, study / 'analysis', plots=False)
    assert result['sources'][0]['ignored_incomplete_tail']
    assert path.read_bytes() == before


def test_same_run_paired_reference_preferred_when_optimization_repeats_it(study):
    source(study, 'optimizations', [measured('case1', 'paired', 2.), measured('case1', 'paired_reuse', .1)],
           ['paired', 'paired_reuse'], status='running')
    result = build_solver_report(study, study / 'analysis', plots=False)
    speed = next(row for row in result['matched_speedups'] if row['candidate'] == 'paired_reuse' and row['metric'] == 'cold_full_sample_seconds')
    assert speed['reference_source_set'] == 'optimizations'
    assert speed['median_ratio'] == pytest.approx(20.)


@pytest.mark.parametrize('status', ['rss_limit', 'internal_budget', 'timeout'])
def test_recorded_but_over_budget_operation_keeps_raw_values_not_headline_time(study, status):
    path = study / 'baselines/results.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(row for row in rows if row['case_id'] == 'case1' and row['backend'] == 'product_chain')
    row.update(status=status, failed_operation='cold_full_sample', peak_rss_mib=513.)
    row['measurements'] = {key: value for key, value in row['measurements'].items()
                           if key in ('cold_partition', 'cold_full_sample')}
    raw_seconds = row['measurements']['cold_full_sample']['seconds']
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    result = build_solver_report(study, study / 'analysis', plots=False)
    availability = next(row for row in result['availability'] if row['backend'] == 'product_chain'
                        and row['operation'] == 'cold_full_sample')
    assert availability['measured_cases'] == 0
    assert availability['statuses'][status] >= 1
    speed = next(row for row in result['matched_speedups'] if row['candidate'] == 'product_chain'
                 and row['metric'] == 'cold_full_sample_seconds')
    assert speed['matched_cases'] == 0 and speed['median_ratio'] is None
    group = next(row for row in result['controlled_sweeps'] if row['backend'] == 'product_chain')
    assert group['metrics']['cold_full_sample_seconds']['n'] == 0
    assert group['metrics']['cold_partition_seconds']['n'] == 2  # Earlier operations survive.
    with Path(result['artifacts']['per_query']).open() as stream:
        raw = next(row for row in csv.DictReader(stream) if row['backend'] == 'product_chain'
                   and row['case_id'] == 'case1' and row['operation'] == 'cold_full_sample')
    assert float(raw['seconds']) == raw_seconds
    assert raw['availability'] == status


def test_third_source_and_direct_ablations_use_named_references(study):
    base = study / 'baselines'
    rows = [json.loads(line) for line in (base / 'results.jsonl').read_text().splitlines()]
    rows += [measured(f'case{i}', 'template', 4.) for i in range(1, 4)]
    (base / 'results.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    config = json.loads((base / 'config.json').read_text())
    config['backends'].append('template')
    (base / 'config.json').write_text(json.dumps(config))
    optimizations = [measured(f'case{i}', backend, seconds)
                     for backend, seconds in [('paired_reuse', .5), ('paired_sparse', .25), ('product_reuse', .125)]
                     for i in range(1, 4)]
    source(study, 'optimizations', optimizations, ['paired_reuse', 'paired_sparse', 'product_reuse'])
    source(study, 'template_refinement', [measured(f'case{i}', 'template_stream', 1.) for i in range(1, 4)],
           ['template_stream'])
    result = build_solver_report(study, study / 'analysis', plots=False)
    assert result['status'] == 'completed' and result['template_refinement_source_present']
    assert result['recorded_rows'] == 21
    assert result['correctness']['passed_on_available_comparisons']
    expected = {('paired', 'paired_reuse'): (2., 3), ('paired_reuse', 'paired_sparse'): (2., 3),
                ('product_chain', 'product_reuse'): (1.6, 1), ('product_reuse', 'paired_sparse'): (.5, 3),
                ('template', 'template_stream'): (4., 3)}
    for pair, (ratio, count) in expected.items():
        row = next(row for row in result['ablation_speedups'] if (row['reference'], row['candidate']) == pair
                   and row['metric'] == 'cold_full_sample_seconds')
        assert row['median_ratio'] == pytest.approx(ratio)
        assert row['matched_cases'] == count
        assert row['planned_feasible_cases'] == 3
    classical = next(row for row in result['ablation_speedups'] if row['reference'] == 'product_reuse'
                     and row['candidate'] == 'paired_sparse' and row['metric'] == 'cold_partition_seconds')
    assert classical['reference_source_set'] == 'optimizations'
    template = next(row for row in result['ablation_speedups'] if row['candidate'] == 'template_stream'
                    and row['metric'] == 'cold_partition_seconds')
    assert template['source_set'] == 'template_refinement'
    assert template['reference_source_set'] == 'baselines'
    assert len([row for row in result['backend_summaries'] if row['backend'] in ('template', 'template_stream')]) == 2
    for artifact in ('ablation_speedups', 'ablation_pairs'):
        with Path(result['artifacts'][artifact]).open() as stream:
            assert len(list(csv.DictReader(stream))) > 0


def test_direct_ablation_intersects_plans_and_requires_complete_memory_workflow(study):
    complete = measured('case1', 'product_reuse', .1)
    later_failure = measured('case2', 'product_reuse', .1)
    later_failure.update(status='timeout', failed_operation='token_marginal')
    del later_failure['measurements']['token_marginal']
    source(study, 'optimizations', [complete, later_failure], ['product_reuse'], cases=('case1', 'case2'))
    result = build_solver_report(study, study / 'analysis', plots=False)
    timing = next(row for row in result['ablation_speedups'] if row['reference'] == 'product_chain'
                  and row['candidate'] == 'product_reuse' and row['metric'] == 'cold_partition_seconds')
    assert timing['reference_planned_feasible_cases'] == 3
    assert timing['candidate_planned_feasible_cases'] == 2
    assert timing['planned_feasible_cases'] == 2 and timing['matched_cases'] == 2
    memory = next(row for row in result['ablation_speedups'] if row['reference'] == 'product_chain'
                  and row['candidate'] == 'product_reuse' and row['metric'] == 'peak_rss_mib')
    assert memory['matched_cases'] == 1
    assert memory['matched_values'][0]['case_id'] == 'case1'


def test_partial_template_refinement_does_not_replace_old_template_failures(study):
    base = study / 'baselines'
    config = json.loads((base / 'config.json').read_text())
    config['backends'].append('template')
    (base / 'config.json').write_text(json.dumps(config))
    template = measured('case1', 'template', 4.)
    template.update(status='internal_budget', failed_operation='cold_partition')
    template['measurements'] = {}
    with (base / 'results.jsonl').open('a') as stream:
        stream.write(json.dumps(template) + '\n')
    source(study, 'template_refinement', [measured('case1', 'template_stream', .2)],
           ['template_stream'], status='running')
    result = build_solver_report(study, study / 'analysis', plots=False)
    assert result['status'] == 'partial'
    pair = next(row for row in result['ablation_speedups'] if row['candidate'] == 'template_stream'
                and row['metric'] == 'cold_partition_seconds')
    assert pair['candidate_available'] == 1 and pair['reference_available'] == 0
    assert pair['matched_cases'] == 0 and pair['median_ratio'] is None
    old = next(row for row in result['availability'] if row['backend'] == 'template' and row['operation'] == 'cold_partition')
    assert old['statuses']['internal_budget'] == 1 and old['statuses']['pending'] == 2


def test_single_unified_source_excludes_pilots_and_uses_same_run_for_all_pairs(study):
    seconds = {'paired': 2., 'template': 8., 've': 6., 've_aligned': 3., 've_minfill': 5.,
               'product_chain': 4., 'paired_reuse': 1., 'paired_sparse': .5,
               'product_reuse': .25, 'template_stream': 1.}
    unified = [measured(f'case{i}', backend, value) for backend, value in seconds.items() for i in range(1, 4)]
    source(study, 'final_comparison', unified, list(seconds))
    result = build_solver_report(study, study / 'final_analysis', plots=False, sources=['final_comparison'])
    assert result['status'] == 'completed'
    assert result['selected_sources'] == ['final_comparison']
    assert result['actual_backends'] == sorted(seconds)
    assert result['has_optimization_backends'] and result['has_template_stream_backend']
    assert not result['optimization_source_present'] and not result['template_refinement_source_present']
    assert 'optimization_present' not in result and 'template_refinement_present' not in result
    assert result['recorded_rows'] == 30 and len(result['sources']) == 1
    assert result['correctness']['passed_on_available_comparisons']
    expected = {('paired', 'paired_reuse'): 2., ('paired_reuse', 'paired_sparse'): 2.,
                ('product_chain', 'product_reuse'): 16., ('product_reuse', 'paired_sparse'): .5,
                ('template', 'template_stream'): 8.}
    for pair, ratio in expected.items():
        row = next(row for row in result['ablation_speedups'] if (row['reference'], row['candidate']) == pair
                   and row['metric'] == 'cold_full_sample_seconds')
        assert row['source_set'] == row['reference_source_set'] == 'final_comparison'
        assert row['median_ratio'] == pytest.approx(ratio)
        assert row['matched_cases'] == row['planned_feasible_cases'] == 3
    main = next(row for row in result['matched_speedups'] if row['candidate'] == 'product_reuse'
                and row['metric'] == 'cold_partition_seconds')
    assert main['median_ratio'] == pytest.approx(8.)  # Pilot paired=1 must not supply the numerator.
    with Path(result['artifacts']['per_query']).open() as stream:
        assert {row['source_set'] for row in csv.DictReader(stream)} == {'final_comparison'}
    default = build_solver_report(study, study / 'pilot_analysis', plots=False)
    assert default['recorded_rows'] == 6  # Existing default behavior remains unchanged.


def test_cli_forwards_explicit_sources_without_default_pilot_merge(monkeypatch, capsys):
    import tri.evaluation.solver_report as module
    calls = []

    def fake(study_root, output, **kwargs):
        calls.append(kwargs)
        return {'status': 'completed', 'recorded_rows': 750,
                'correctness': {'passed_on_available_comparisons': True}, 'artifacts': {}}

    monkeypatch.setattr(module, 'build_solver_report', fake)
    module.main(['--sources', 'final_comparison', '--no-plots'])
    assert calls == [{'plots': False, 'sources': ['final_comparison']}]
    assert json.loads(capsys.readouterr().out)['rows'] == 750
