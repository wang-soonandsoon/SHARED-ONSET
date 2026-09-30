"""Predeclared L32 validation inputs and same-q wider-pitch target variants.

This module prepares requests on CPU. It never loads or calls a neural model.
Wide variants are generated only by the explicit pair-wide command after q is frozen.
L32 changes K to eight and the visible prefix to eight cells (still 25% of the
first span); comparisons against L16 are scaling studies, not one-variable
causal ablations. Wider pitches change the target, so equality of narrow/wide
partition functions or output distributions is not expected.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from tri.domain.music import CountRule, MusicSpec, verify_music
from tri.errors import BudgetExceeded, ZeroMass
from tri.evaluation import shared_rhythm_real_cases as real
from tri.evaluation.solver_cases import SolverCase, _serialize_spec, _soundings, _validate_case, load_case, save_cases
from tri.runtime import atomic_json, exclusive_run


DEFAULT_OPTIONS = {
    'context_cells': 256, 'L': 32, 'K': 8, 'beta': .02,
    'quotas': {'3': 4, '4': 4}, 'partial_prefix_cells': 8,
    'min_visible_onsets': 8, 'max_per_work': 2, 'minimum_distinct_works': 8,
    'seed': 20260951, 'pitch_low': 48, 'pitch_high': 84,
}
LONG_POLICY = {**real.POLICY,
    'scope': 'Eight distinct anchored validation requests, R3/R4, L32 and K8 in 256 cells. The partial prefix grows to eight cells, preserving 25% visibility in the first span. Length, count, absolute visible prefix and gap positions change together; this is not a strict single-variable length ablation. The existing checkpoint was trained with two L32 gaps; R3/R4 still changes the training mask layout. No human-listening or quality-superiority claim.',
    'selection': 'Fixed two-bar spans and K8; visible token activity, visible onset flags and model-independent uniform-q support. Prefer unused works, cap two requests per work, select each source window once. No hidden-melody, q, quality or timing selection. The candidate audit counts screened candidates only.',
    'cross_cohort': 'This is a separately seeded eight-request cohort. A source work/window may also occur in the L16 cohort; cohorts are not paired by melody unless an explicit source-identity intersection is reported.',
}


def _options(options=None):
    result = {**DEFAULT_OPTIONS, **(options or {})}
    if set(result) != set(DEFAULT_OPTIONS):
        raise ValueError('Unknown long-request option')
    for key in set(DEFAULT_OPTIONS) - {'quotas', 'beta'}:
        if isinstance(result[key], bool) or not isinstance(result[key], int):
            raise ValueError('Long-request dimensions/counts must be integers')
    quotas = {str(k): v for k, v in result['quotas'].items()}
    if (result['context_cells'] != 256 or result['L'] != 32 or result['K'] != 8
            or result['partial_prefix_cells'] != 8 or quotas != {'3': 4, '4': 4}
            or not 0 <= result['pitch_low'] <= result['pitch_high'] <= 127
            or min(result['min_visible_onsets'], result['max_per_work'], result['minimum_distinct_works']) < 1
            or isinstance(result['beta'], bool) or not math.isfinite(result['beta']) or result['beta'] < 0):
        raise ValueError('Long cohort is fixed to eight R3/R4 requests, L32/K8 and eight-cell partial prefix')
    result['quotas'] = {'3': 4, '4': 4}
    return result


def long_spans(R):
    if isinstance(R, bool) or R not in (3, 4):
        raise ValueError('Long requests support R3/R4 only')
    starts = [16 * (j * 12 // (R - 1)) for j in range(R)]
    return tuple(tuple(range(start, start + 32)) for start in starts)


def project_long_visible(tokens, initial_pitch, R, visibility, *, options=None):
    options = _options(options)
    if len(tokens) != 256 or visibility not in ('unknown', 'partial'):
        raise ValueError('Invalid long context or visibility')
    spans = long_spans(R)
    hidden = {p for span in spans for p in span}
    if visibility == 'partial':
        hidden.difference_update(spans[0][:options['partial_prefix_cells']])
    observed = {i: int(tokens[i]) for i in range(256) if i not in hidden}
    soundings = _soundings(tokens, initial_pitch)
    return {'observed': observed, 'fixed_soundings': {i: soundings[i] for i in observed},
            'initial_pitch': initial_pitch}


def long_request_from_visible(visible, R, visibility, *, options=None):
    options = _options(options)
    if visibility not in ('unknown', 'partial'):
        raise ValueError('Invalid long visibility')
    spans = long_spans(R)
    inside = {p for span in spans for p in span}
    expected = set(range(256)) - inside
    if visibility == 'partial':
        expected.update(spans[0][:8])
    observed = {int(k): int(v) for k, v in visible['observed'].items()}
    anchors = {int(k): v for k, v in visible['fixed_soundings'].items()}
    if set(observed) != expected or set(anchors) != expected:
        raise ValueError('Long request requires exactly its predeclared visible tokens/anchors')
    pitches = set(range(options['pitch_low'], options['pitch_high'] + 1))
    pitches.update(t - 2 for t in observed.values() if t >= 2)
    pitches.update(p for p in (visible['initial_pitch'], *anchors.values()) if p is not None)
    spec = MusicSpec(length=256, pitches=tuple(sorted(pitches)), observed=observed,
        initial_pitch=visible['initial_pitch'], fixed_soundings=anchors,
        equal_onsets=tuple((spans[0][j], spans[r][j]) for r in range(1, R) for j in range(32)),
        onset_counts=tuple(CountRule(span, 8) for span in spans), motion_cost=options['beta'])
    known = [observed[p] >= 2 for p in spans[0] if p in observed]
    remaining, unknown = 8 - sum(known), 32 - len(known)
    templates = math.comb(unknown, remaining) if 0 <= remaining <= unknown else 0
    metadata = {'family': real.FAMILY, 'source_type': 'learned', 'stage': 'bar16',
        'scale_variant': 'long_L32', 'R': R, 'L': 32, 'D': len(pitches) + 1,
        'K': 8, 'beta': options['beta'], 'length': 256, 'visibility': visibility,
        'spans': [list(span) for span in spans],
        'visible_onsets': sum(t >= 2 for t in observed.values()),
        'outside_visible_onsets': sum(t >= 2 for p, t in observed.items() if p not in inside),
        'observed_positions_in_first_span': len(known),
        'visible_fraction_first_span': len(known) / 32,
        'visible_onsets_in_first_span': sum(known), 'remaining_onsets_in_first_span': remaining,
        'onset_templates_consistent_with_observations_and_count': templates,
        'unknown_positions': 256 - len(observed), 'emittable_pitch_count': len(pitches),
        'sounding_state_count_including_silence': len(pitches) + 1,
        'D_definition': f'Silence plus NOTE pitches in {options["pitch_low"]}..{options["pitch_high"]} and supplied public/boundary pitches.',
        'guard': 'published_aligned_visible_tokens_and_soundings',
        'hard_harmony': False, 'provider_noise': 1., 'scope': LONG_POLICY['scope']}
    if templates == 0:
        reason = 'infeasible_visible_onset_flags'
    elif visibility == 'partial' and (remaining == 0 or templates == 1):
        reason = 'partial_degenerated_to_known_template'
    elif metadata['visible_onsets'] < options['min_visible_onsets']:
        reason = 'insufficient_visible_activity'
    else:
        reason = None
    return spec, metadata, reason


def select_long_requests(windows, *, options=None):
    options = _options(options)
    rng = np.random.default_rng(options['seed'])
    by_work = defaultdict(list)
    for row in windows:
        by_work[row['work_id']].append(row)
    works = sorted(by_work)
    rng.shuffle(works)
    for work in works:
        by_work[work].sort(key=lambda row: (row['source_start_cell'], row['source_index']))
        rng.shuffle(by_work[work])
    candidates = [by_work[work][n] for n in range(max(map(len, by_work.values()), default=0))
                  for work in works if n < len(by_work[work])]
    rank = {row['source_index']: i for i, row in enumerate(candidates)}
    selected, audit, screened, used = [], [], set(), set()
    work_counts, counts = Counter(), Counter()
    slots = [(R, visibility) for _ in range(2) for R in (3, 4) for visibility in ('unknown', 'partial')]
    for R, visibility in slots:
        ordered = sorted(candidates, key=lambda row: (work_counts[row['work_id']], rank[row['source_index']]))
        for window in ordered:
            key = (R, visibility, window['source_index'])
            if key in screened or window['source_index'] in used or work_counts[window['work_id']] >= options['max_per_work']:
                continue
            visible = project_long_visible(window['tokens'], window['initial_pitch'], R, visibility, options=options)
            spec, metadata, reason = long_request_from_visible(visible, R, visibility, options=options)
            case_id = f'real_bar16_w{window["work_id"]}_c{window["source_start_cell"]}_R{R}_L32_{visibility}'
            row = {'case_id': case_id, 'source_index': window['source_index'], 'work_id': window['work_id'],
                'source_start_cell': window['source_start_cell'], 'R': R, 'visibility': visibility,
                'candidate_rank': rank[window['source_index']], 'visible_onsets': metadata['visible_onsets'],
                'visible_onsets_in_first_span': metadata['visible_onsets_in_first_span'],
                'remaining_onsets_in_first_span': metadata['remaining_onsets_in_first_span'],
                'onset_templates_consistent_with_observations_and_count': metadata['onset_templates_consistent_with_observations_and_count'],
                'status': reason, 'selected': False}
            if reason is None:
                try:
                    witness, z = real._uniform_witness(spec, options['seed'] + len(audit))
                    row.update(status='feasible', uniform_log_partition=z, witness_verified=True, selected=True)
                    metadata.update(expected_support='feasible', witness_verified=True,
                        work_id=window['work_id'], source_index=window['source_index'],
                        source_start_cell=window['source_start_cell'], split='validation',
                        query_evidence={f'y{span[-1]}': int(witness[span[-1]]) for span in long_spans(R)})
                    selected.append({**row, 'metadata': metadata, 'spec': _serialize_spec(spec), 'witness': list(witness)})
                    used.add(window['source_index'])
                    work_counts[window['work_id']] += 1
                    counts[(R, visibility)] += 1
                except ZeroMass as exc:
                    row.update(status='infeasible_boundary_or_visible_context', reason=str(exc))
                except BudgetExceeded as exc:
                    row.update(status='support_budget_unknown', reason=str(exc))
            audit.append(row)
            screened.add(key)
            if row['selected']:
                break
    complete = len(selected) == 8 and len(work_counts) >= options['minimum_distinct_works']
    return {'status': 'completed' if complete else 'insufficient_eligible_requests',
        'selected_count': len(selected), 'selected_work_count': len(work_counts),
        'selected_by_work': dict(work_counts),
        'selected_by_R_visibility': {f'R{R}_{v}': counts[(R, v)] for R, v in slots[:4]},
        'candidate_window_pool_count': len(candidates), 'candidate_work_pool_count': len(works),
        'possible_R_visibility_candidates': len(candidates) * 4,
        'screened_candidate_count': len(audit),
        'screened_status_counts': dict(Counter(row['status'] for row in audit)),
        'screening_denominator': 'Only actually screened candidates; skipped/unvisited combinations have no support label.',
        'requests': selected, 'candidate_audit': audit}


def prepare_long_requests(output='runs/shared_rhythm_extension/long_inputs', *, revision_root='runs/revision', options=None, provenance=None):
    options = _options(options)
    output = Path(output).resolve()
    with exclusive_run(output / 'prepare.lock'):
        if provenance is None:
            provenance = real._provenance(revision_root)
        else:
            provenance = dict(provenance)
            real._check_identities(provenance)
        path = output / 'cohort.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if previous['options'] != options or previous['provenance'] != provenance:
                raise ValueError('Existing long cohort differs; use a new output directory')
            if previous['status'] != 'completed':
                raise ValueError('Existing long cohort is incomplete; inspect its retained audit')
            return previous
        start = perf_counter()
        result = select_long_requests(real._validation_windows(provenance), options=options)
        result.update(version=1, options=options, provenance=provenance, policy=LONG_POLICY,
                      preparation_seconds=perf_counter() - start, model_calls=0)
        atomic_json(path, result)
        if result['status'] != 'completed':
            raise ValueError(f'Insufficient long requests; see {path}')
        return result


def make_wide_pitch_case(case: SolverCase, *, low=36, high=96):
    """Derive a changed pitch-domain target with identical original full130 q.

    No provider call, new observations, q slicing or probability normalization
    occurs. The source's extraction seconds remain as shared input provenance;
    additional model calls/time are zero. Do not sum that shared extraction cost
    across narrow/wide variants. Compare solvers on each target independently.
    """
    if (isinstance(low, bool) or isinstance(high, bool) or not isinstance(low, int)
            or not isinstance(high, int) or not 0 <= low <= high <= 127):
        raise ValueError('Wide pitch range requires integer MIDI bounds')
    if case.metadata.get('source_type') != 'learned':
        raise ValueError('Wide real variants require a frozen learned-probability source')
    _validate_case(case)
    pitches = tuple(sorted(set(case.spec.pitches) | set(range(low, high + 1))))
    if pitches == case.spec.pitches:
        raise ValueError('Requested pitch range does not expand the source domain')
    spec = replace(case.spec, pitches=pitches)
    sounding = set(pitches) | {p for p in (spec.initial_pitch, spec.end_pitch, *spec.fixed_soundings.values()) if p is not None}
    metadata = deepcopy(case.metadata)
    metadata.update(scale_variant='wide_pitch', source_case_id=case.case_id,
        q_pairing_key=case.metadata.get('q_pairing_key', case.case_id),
        wide_pitch_bounds=[low, high], source_D=case.metadata.get('D'), D=len(sounding) + 1,
        emittable_pitch_count=len(pitches), sounding_state_count_including_silence=len(sounding) + 1,
        D_definition='Silence plus expanded NOTE pitches and retained supplied sounding states.',
        probability_source='unchanged_full130_snapshot_from_source_case',
        source_model_calls=case.metadata.get('source_model_calls', case.metadata.get('model_calls')),
        model_calls=0, additional_model_calls=0, additional_probability_provider_seconds=0.,
        provider_preprocessing_seconds=0.,
        probability_provider_seconds_role='Inherited one-time source extraction cost; no new model call and do not sum across paired variants.',
        scale_comparison='Only spec.pitches is widened; all original token/sounding observations, relations, counts, soft costs and original full130 q are unchanged. The target support/distribution changes, so narrow and wide Z need not match.')
    result = SolverCase(f'{case.case_id}_wide{low}_{high}', spec, case.logq.copy(), metadata, case.witness)
    _validate_case(result)
    return result


def prepare_long_scale_cases(source='runs/shared_rhythm_extension/long_inputs/cases.json',
                             output='runs/shared_rhythm_extension/long_scale_inputs'):
    """Freeze all eight original targets and their predeclared wide variants."""
    manifest = json.loads(Path(source).read_text())
    if manifest.get('status') != 'completed' or manifest.get('count') != 8:
        raise ValueError('Expected the completed eight-request L32 model manifest')
    cases = []
    for entry in manifest['cases']:
        case = load_case(entry['path'])
        if case.metadata['L'] != 32 or case.metadata.get('model_calls') != 1:
            raise ValueError('Expected one original learned call for each L32 request')
        metadata = {**case.metadata, 'scale_variant': 'narrow_pitch',
                    'q_pairing_key': case.case_id, 'source_case_id': case.case_id}
        narrow = replace(case, metadata=metadata)
        cases.extend((narrow, make_wide_pitch_case(narrow)))
    with exclusive_run(Path(output) / 'pair.lock'):
        return save_cases(cases, output, policy={
            'source_manifest': str(Path(source).resolve()), 'unique_model_calls': 8,
            'request_count': 8, 'target_variants': 16,
            'rule': 'All eight predeclared visible-selected L32 requests, each with its original domain and MIDI36..96 union. Full130 q, observations, anchors and soft costs unchanged. Wide changes target support, never creates a new model call; compare methods within each target. No result-dependent selection.',
            'source_policy': manifest.get('policy')})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare-long', 'pair-wide'))
    parser.add_argument('--config', default='configs/shared_rhythm_long.yaml')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    if args.command == 'pair-wide':
        result = prepare_long_scale_cases(output=args.output or 'runs/shared_rhythm_extension/long_scale_inputs')
        print(json.dumps({'status': result['status'], 'targets': result['count']}))
        return
    import yaml
    config = yaml.safe_load(Path(args.config).read_text())
    result = prepare_long_requests(args.output or config['output'], revision_root=config['revision_root'], options=config['requests'])
    print(json.dumps({key: result[key] for key in ('status', 'selected_count', 'selected_work_count', 'model_calls')}))


if __name__ == '__main__':
    main()
