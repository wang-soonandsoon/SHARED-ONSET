"""Validation requests and one-call learned q for the shared-rhythm extension.

Preparation is CPU/model-independent; freezing is a separate explicit command.
The published aligned melody supplies visible token/sounding conditions only.
Neither a hidden reference melody nor neural scores select requests. Historical
source-window activity filtering is retained and disclosed, not reclassified as
visible-only selection. All 24 requests use the same bar16 checkpoint; changing
the number of gaps is an out-of-training-mask-layout evaluation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import argparse
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from tri.domain.music import CountRule, MusicSpec, verify_music
from tri.errors import BudgetExceeded, ZeroMass
from tri.evaluation.solver_cases import (
    SolverCase, _deserialize_spec, _identity, _serialize_spec, _soundings,
    _validate_case, load_case,
)
from tri.runtime import atomic_json, exclusive_run


DEFAULT_OPTIONS = {
    'L': 16, 'K': 4, 'beta': .02, 'context_cells': 256,
    'quotas': {'3': 8, '4': 8, '6': 4, '8': 4},
    'partial_prefix_cells': 4, 'min_visible_onsets': 8,
    'max_per_work': 2, 'minimum_distinct_works': 8,
    'seed': 20260941, 'pitch_low': 48, 'pitch_high': 84,
}
FAMILY = 'real_validation_shared_rhythm'
POLICY = {
    'selection': 'Fixed bar positions and K, visible token activity and flags, uniform-q exact hard support; never hidden melody, model q, output quality or solver timing. Distinct source windows; prefer unused works, cap two requests per work.',
    'boundaries': 'The published aligned source supplies initial pitch and sounding pitch at every visible token, including HOLD immediately after a gap. These are explicit conditioning facts; they may encode a pitch sustained from inside the hidden span. This is anchored completion, not unconditioned boundary reconstruction.',
    'source_preprocessing': 'Existing aligned_bars16 preprocessing retained active whole windows and capped windows per work before this study. That historical full-window activity filter is not a claim of visible-only source-dataset selection. New request eligibility reads only the visible projection and supplied anchors.',
    'model': 'Frozen validation-selected bar16 checkpoint, one genuine full130 call per request at noise=1. External 38-dimensional aligned chord features are retained; no hard harmony rule. No new training or test split.',
    'q': 'Original full130 log probabilities are saved without pitch cropping or renormalization. Observed tokens become delta in inference; q is unchanged for every backend/draw.',
    'scope': '24 distinct anchored validation completion requests, not 24 independent works unless the recorded work count says so. The checkpoint was trained with two gaps; R=3/4/6/8 and L16 use a different mask layout. No human-listening or quality-superiority claim.',
    'timing': 'Per-request provider seconds include input-tensor construction, forward pass, full-vocabulary normalization and CPU transfer, bracketed by CUDA synchronization. Provider preparation, checkpoint/CUDA initialization, validation/storage and whole freeze wall time are separate. No warmup or unrecorded request calls.',
}


def _options(options=None):
    value = {**DEFAULT_OPTIONS, **(options or {})}
    if set(value) != set(DEFAULT_OPTIONS):
        raise ValueError('Unknown real-request option')
    integer_keys = ('L', 'K', 'context_cells', 'partial_prefix_cells', 'min_visible_onsets',
                    'max_per_work', 'minimum_distinct_works', 'seed', 'pitch_low', 'pitch_high')
    if any(isinstance(value[k], bool) or not isinstance(value[k], int) for k in integer_keys):
        raise ValueError('Request dimensions, counts and seeds must be integers')
    if (value['context_cells'] != 256 or value['L'] != 16 or not 0 < value['K'] <= value['L']
            or not 0 < value['partial_prefix_cells'] < value['L']
            or min(value['min_visible_onsets'], value['max_per_work'], value['minimum_distinct_works']) < 1
            or not 0 <= value['pitch_low'] <= value['pitch_high'] <= 127
            or isinstance(value['beta'], bool) or not math.isfinite(value['beta']) or value['beta'] < 0):
        raise ValueError('Unsupported real-request options; this cohort is bar16/L16')
    quotas = {str(k): n for k, n in value['quotas'].items()}
    if (not quotas or any(k not in ('3', '4', '6', '8') or isinstance(n, bool)
            or not isinstance(n, int) or n < 2 or n % 2 for k, n in quotas.items())):
        raise ValueError('Each R quota must be positive, even, and R in 3/4/6/8')
    value['quotas'] = dict(sorted(quotas.items(), key=lambda p: int(p[0])))
    return value


def fixed_spans(R, *, length=256, L=16):
    """Evenly spread whole-bar gaps, leaving at least one visible bar between."""
    if R not in (3, 4, 6, 8) or length != 256 or L != 16:
        raise ValueError('This real cohort requires bar16/L16 and R=3/4/6/8')
    starts = [16 * (j * 14 // (R - 1)) for j in range(R)]
    return tuple(tuple(range(start, start + L)) for start in starts)


def project_visible(tokens, initial_pitch, R, visibility, *, options=None):
    """The ONLY raw melody access: export declared visible tokens and anchors.

    Downstream request building receives no hidden token array. The sounding
    scan is needed to supply visible HOLD pitch conditions; anchors are an
    explicit input, not a ground-truth target used for choosing notes/counts.
    """
    options = _options(options)
    if len(tokens) != options['context_cells'] or visibility not in ('unknown', 'partial'):
        raise ValueError('Invalid real context or visibility')
    spans = fixed_spans(R)
    hidden = {i for span in spans for i in span}
    if visibility == 'partial':
        hidden.difference_update(spans[0][:options['partial_prefix_cells']])
    observed = {i: int(tokens[i]) for i in range(len(tokens)) if i not in hidden}
    sounding = _soundings(tokens, initial_pitch)
    return {'observed': observed, 'fixed_soundings': {i: sounding[i] for i in observed},
            'initial_pitch': initial_pitch}


def request_from_visible(visible, R, visibility, *, options=None):
    """Build target/eligibility statistics without access to missing notes."""
    options = _options(options)
    if visibility not in ('unknown', 'partial'):
        raise ValueError('Visibility must be unknown or partial')
    spans = fixed_spans(R)
    inside = {i for span in spans for i in span}
    expected = set(range(256)) - inside
    if visibility == 'partial':
        expected.update(spans[0][:options['partial_prefix_cells']])
    observed = {int(k): int(v) for k, v in visible['observed'].items()}
    anchors = {int(k): v for k, v in visible['fixed_soundings'].items()}
    if set(observed) != expected or set(anchors) != expected:
        raise ValueError('Visible input must contain exactly the predeclared token/anchor positions')
    pitches = set(range(options['pitch_low'], options['pitch_high'] + 1))
    pitches.update(t - 2 for t in observed.values() if t >= 2)
    pitches.update(p for p in (visible['initial_pitch'], *anchors.values()) if p is not None)
    spec = MusicSpec(length=256, pitches=tuple(sorted(pitches)), observed=observed,
        initial_pitch=visible['initial_pitch'], fixed_soundings=anchors,
        equal_onsets=tuple((spans[0][j], spans[r][j]) for r in range(1, R) for j in range(16)),
        onset_counts=tuple(CountRule(span, options['K']) for span in spans), motion_cost=options['beta'])
    known = [observed[p] >= 2 for p in spans[0] if p in observed]
    remaining = options['K'] - sum(known)
    unknown = 16 - len(known)
    templates = math.comb(unknown, remaining) if 0 <= remaining <= unknown else 0
    metadata = {'family': FAMILY, 'source_type': 'learned', 'stage': 'bar16',
        'R': R, 'L': 16, 'D': len(pitches) + 1, 'K': options['K'], 'beta': options['beta'],
        'visibility': visibility, 'length': 256, 'spans': [list(span) for span in spans],
        'visible_onsets': sum(t >= 2 for t in observed.values()),
        'outside_visible_onsets': sum(t >= 2 for p, t in observed.items() if p not in inside),
        'observed_positions_in_first_span': len(known), 'visible_onsets_in_first_span': sum(known),
        'remaining_onsets_in_first_span': remaining,
        'onset_templates_consistent_with_observations_and_count': templates,
        'unknown_positions': 256 - len(observed), 'emittable_pitch_count': len(pitches),
        'sounding_state_count_including_silence': len(pitches) + 1,
        'D_definition': 'Silence plus all allowed NOTE pitches, including supplied visible/boundary pitches outside the fixed 48..84 base range.',
        'guard': 'published_aligned_visible_tokens_and_soundings',
        'hard_harmony': False, 'provider_noise': 1.0}
    if templates == 0:
        reason = 'infeasible_visible_onset_flags'
    elif visibility == 'partial' and (remaining == 0 or templates == 1):
        reason = 'partial_degenerated_to_known_template'
    elif metadata['visible_onsets'] < options['min_visible_onsets']:
        reason = 'insufficient_visible_activity'
    else:
        reason = None
    return spec, metadata, reason


def _provenance(revision_root):
    stage = Path(revision_root).resolve() / 'bar16'
    plan = json.loads((stage / 'cohort.json').read_text())
    config = json.loads((stage / 'evaluation/config.json').read_text())
    if plan['status'] != 'completed' or plan['options']['split'] != 'validation':
        raise ValueError('Expected the completed validation bar16 source cohort')
    identities = {key: plan['options'][key] for key in ('dataset', 'chords')}
    identities['checkpoint'] = config['checkpoint']
    _check_identities(identities)
    return {**identities, 'checkpoint_seed': config.get('checkpoint_seed'),
            'historical_cohort': _identity(stage / 'cohort.json'),
            'historical_evaluation_config': _identity(stage / 'evaluation/config.json')}


def _check_identities(provenance):
    for key in ('dataset', 'chords', 'checkpoint'):
        if _identity(provenance[key]['path']) != provenance[key]:
            raise ValueError(f'Frozen {key} identity changed')


def _validation_windows(provenance):
    """Refuse a test-containing materialization before touching its token data."""
    with np.load(provenance['dataset']['path'], allow_pickle=False) as data:
        splits = data['splits'].astype(str)
        if any(s not in ('train', 'validation', 'val') for s in splits):
            raise ValueError('Real preparation requires train/validation-only materialized data; no test reads')
        indices = np.flatnonzero(np.isin(splits, ('validation', 'val')))
        tokens = data['tokens']
        if tokens.ndim != 2 or tokens.shape != (len(splits), 256):
            raise ValueError('Expected aligned 256-cell windows')
        works, starts, initial = data['work_ids'].astype(str), data['start_cells'], data['initial_pitches']
        rows = [{'source_index': int(i), 'work_id': str(works[i]), 'source_start_cell': int(starts[i]),
                 'tokens': tuple(int(t) for t in tokens[i]),
                 'initial_pitch': None if int(initial[i]) == -1 else int(initial[i])} for i in indices]
    if not rows or len({(r['work_id'], r['source_start_cell']) for r in rows}) != len(rows):
        raise ValueError('Missing validation windows or duplicate source identities')
    if any(r['source_start_cell'] % 16 for r in rows):
        raise ValueError('Source windows must begin on the aligned bar grid')
    return rows


def _uniform_witness(spec, seed):
    # Selection uses a positive model-independent q, not checkpoint outputs.
    from tri.inference.onset_reset import OnsetResetMusicInference
    engine = OnsetResetMusicInference(replace(spec, motion_cost=0.), np.full((spec.length, 130), -math.log(130)))
    z = engine.log_partition()
    if not math.isfinite(z):
        raise ZeroMass('No hard-compatible completion under positive uniform q')
    witness = tuple(engine.sample_full(np.random.default_rng(seed)))
    if not verify_music(witness, spec).valid:
        raise RuntimeError('Independent full-original-token checker rejected the support witness')
    return witness, z


def select_requests(windows, *, options=None):
    """Deterministic quota selection with an audit of every screened candidate.

    Candidate order is shuffled work round-robin, then prefers works with fewer
    selected requests. Stop when quotas are full. Unscreened candidates are not
    counted as feasible/infeasible; the pool and screened denominators are both
    reported. Each source window is selected at most once.
    """
    options = _options(options)
    rng = np.random.default_rng(options['seed'])
    by_work = defaultdict(list)
    for row in windows:
        by_work[row['work_id']].append(row)
    works = sorted(by_work)
    rng.shuffle(works)
    for work in works:
        by_work[work].sort(key=lambda w: (w['source_start_cell'], w['source_index']))
        rng.shuffle(by_work[work])
    ordered = [by_work[work][n] for n in range(max(map(len, by_work.values()), default=0))
               for work in works if n < len(by_work[work])]
    rank = {row['source_index']: i for i, row in enumerate(ordered)}
    quotas = {(int(R), vis): n // 2 for R, n in options['quotas'].items() for vis in ('unknown', 'partial')}
    counts, work_counts, selected_indices = Counter(), Counter(), set()
    audit, screened, selected = [], {}, []
    for round_index in range(max(quotas.values())):
        for (R, visibility), count in quotas.items():
            if round_index >= count:
                continue
            candidates = sorted(ordered, key=lambda r: (work_counts[r['work_id']], rank[r['source_index']]))
            for window in candidates:
                if window['source_index'] in selected_indices or work_counts[window['work_id']] >= options['max_per_work']:
                    continue
                key = (R, visibility, window['source_index'])
                if key in screened:
                    continue  # previously ineligible; feasible rows are selected immediately
                spec, metadata, reason = request_from_visible(project_visible(window['tokens'], window['initial_pitch'], R, visibility, options=options), R, visibility, options=options)
                case_id = f'real_bar16_w{window["work_id"]}_c{window["source_start_cell"]}_R{R}_L16_{visibility}'
                row = {'case_id': case_id, 'source_index': window['source_index'],
                       'work_id': window['work_id'], 'source_start_cell': window['source_start_cell'],
                       'candidate_rank': rank[window['source_index']], 'R': R, 'visibility': visibility,
                       'visible_onsets': metadata['visible_onsets'],
                       'visible_onsets_in_first_span': metadata['visible_onsets_in_first_span'],
                       'remaining_onsets_in_first_span': metadata['remaining_onsets_in_first_span'],
                       'onset_templates_consistent_with_observations_and_count': metadata['onset_templates_consistent_with_observations_and_count'],
                       'status': reason, 'selected': False}
                if reason is None:
                    try:
                        witness, z = _uniform_witness(spec, options['seed'] + len(audit))
                        row.update(status='feasible', uniform_log_partition=z, witness_verified=True, selected=True)
                        metadata.update(expected_support='feasible', witness_verified=True,
                            work_id=window['work_id'], source_index=window['source_index'],
                            source_start_cell=window['source_start_cell'], split='validation',
                            query_evidence={f'y{span[-1]}': int(witness[span[-1]]) for span in fixed_spans(R)})
                        selected.append({**row, 'spec': _serialize_spec(spec), 'metadata': metadata,
                                         'witness': list(witness)})
                        counts[(R, visibility)] += 1
                        work_counts[window['work_id']] += 1
                        selected_indices.add(window['source_index'])
                    except ZeroMass as exc:
                        row.update(status='infeasible_boundary_or_visible_context', reason=str(exc))
                    except BudgetExceeded as exc:
                        row.update(status='support_budget_unknown', reason=str(exc))
                audit.append(row)
                screened[key] = row
                if row['selected']:
                    break
    complete = all(counts[key] == n for key, n in quotas.items()) and len(work_counts) >= options['minimum_distinct_works']
    return {'status': 'completed' if complete else 'insufficient_eligible_requests',
        'selected_count': len(selected), 'selected_work_count': len(work_counts),
        'selected_by_work': dict(work_counts),
        'selected_by_R_visibility': {f'R{R}_{vis}': counts[(R, vis)] for R, vis in quotas},
        'candidate_window_pool_count': len(ordered), 'candidate_work_pool_count': len(works),
        'possible_R_visibility_candidates': len(ordered) * len(quotas),
        'screened_candidate_count': len(audit),
        'screened_status_counts': dict(Counter(row['status'] for row in audit)),
        'screening_denominator': 'Only actually screened candidates. Unvisited or quota/work-cap-skipped combinations have no support label.',
        'requests': selected, 'candidate_audit': audit}


def prepare_real_requests(output='runs/shared_rhythm_extension/real_inputs', *, revision_root='runs/revision', options=None, provenance=None):
    """Freeze CPU-selected requests before any model runtime is initialized."""
    output = Path(output).resolve()
    options = _options(options)
    with exclusive_run(output / 'prepare.lock'):
        if provenance is None:
            provenance = _provenance(revision_root)
        else:
            provenance = dict(provenance)
            _check_identities(provenance)
        path = output / 'cohort.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if previous['options'] != options or previous['provenance'] != provenance:
                raise ValueError('Existing real cohort differs; use a new output directory')
            if previous['status'] != 'completed':
                raise ValueError('Existing real cohort has insufficient eligible requests; inspect its retained audit')
            return previous
        start = perf_counter()
        result = select_requests(_validation_windows(provenance), options=options)
        result.update(version=1, options=options, provenance=provenance, policy=POLICY,
                      preparation_seconds=perf_counter() - start, model_calls=0)
        atomic_json(path, result)
        if result['status'] != 'completed':
            raise ValueError(f'Insufficient real requests; see {path}')
        return result


@dataclass
class ProbabilityRuntime:
    make_provider: Callable
    synchronize: Callable
    model_config: dict
    description: dict


def _cuda_runtime(checkpoint, device):
    import torch
    from dataclasses import asdict
    from tri.models.grid import ModelProbabilityProvider
    from tri.models.train import load_checkpoint
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Real learned freeze requires the requested CUDA runtime; no CPU fallback')
    torch.set_num_threads(1)
    model = load_checkpoint(checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.config.length != 256 or model.config.condition_dim != 38:
        raise ValueError('Expected the frozen bar16 checkpoint with 38 aligned chord features')
    def make_provider(spec, condition):
        return ModelProbabilityProvider(model, tuple(i for i in range(spec.length) if i not in spec.observed), condition)
    return ProbabilityRuntime(make_provider, lambda: torch.cuda.synchronize(device), asdict(model.config),
                              {'kind': 'frozen_neural_checkpoint', 'device': str(device),
                               'torch_version': torch.__version__, 'cuda_device': torch.cuda.get_device_name(device)})


def _conditions(provenance, requests):
    """Load external validation features; never reload the hidden melody."""
    with np.load(provenance['dataset']['path'], allow_pickle=False) as data, np.load(provenance['chords']['path'], allow_pickle=False) as chords:
        for key in ('work_ids', 'start_cells', 'splits'):
            if not np.array_equal(data[key], chords[key]):
                raise ValueError(f'Chord sidecar {key} alignment mismatch')
        splits, works, starts = data['splits'].astype(str), data['work_ids'].astype(str), data['start_cells']
        features = chords['chord_features']
        selected = {}
        for row in requests:
            i = row['source_index']
            if splits[i] not in ('validation', 'val') or (works[i], int(starts[i])) != (row['work_id'], row['source_start_cell']):
                raise ValueError('Frozen request identity or validation split changed')
            condition = features[i].copy()
            if condition.shape != (256, 38) or not np.isfinite(condition).all():
                raise ValueError('Expected finite aligned 256 x 38 chord condition')
            selected[row['case_id']] = condition
    return selected


def freeze_real_probabilities(output='runs/shared_rhythm_extension/real_inputs', *, device='cuda:0', runtime_factory=None):
    """Save exactly one provider result per fixed request, retaining failures.

    The injected runtime exists for CPU correctness tests. Production defaults
    to CUDA with no fallback. A started/failed provider call is never silently
    repeated after interruption: its uncertainty is retained for inspection.
    Completed cases are resumed without loading the model or recomputing q.
    """
    output = Path(output).resolve()
    with exclusive_run(output / 'freeze.lock'):
        cohort_path = output / 'cohort.json'
        plan = json.loads(cohort_path.read_text())
        if plan['status'] != 'completed' or plan['model_calls'] != 0:
            raise ValueError('Freeze requires a completed model-independent cohort')
        policy = {**POLICY, **plan.get('policy', {})}
        _check_identities(plan['provenance'])
        cohort_identity = _identity(cohort_path)
        manifest_path, status_path = output / 'cases.json', output / 'freeze_status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {
            'version': 1, 'status': 'pending', 'cohort_identity': cohort_identity,
            'planned_requests': len(plan['requests']), 'attempts': {}, 'sessions': []}
        if status['cohort_identity'] != cohort_identity:
            raise ValueError('Frozen request cohort changed after probability extraction started')
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest['cohort_identity'] != cohort_identity or manifest['count'] != len(plan['requests']):
                raise ValueError('Probability manifest disagrees with its frozen cohort')
            for entry, request in zip(manifest['cases'], plan['requests']):
                case = load_case(entry['path'])
                if case.case_id != request['case_id'] or _serialize_spec(case.spec) != request['spec']:
                    raise ValueError('Frozen case differs from selected request')
            return manifest
        uncertain = [key for key, attempt in status['attempts'].items() if attempt['status'] != 'completed']
        if uncertain:
            raise ValueError(f'Prior provider calls are failed or in flight and cannot be repeated implicitly: {uncertain}')
        start = perf_counter()
        conditions = _conditions(plan['provenance'], plan['requests'])
        feature_load_seconds = perf_counter() - start
        todo = [row for row in plan['requests'] if row['case_id'] not in status['attempts']]
        runtime, model_load_seconds = None, 0.
        if todo:
            load_start = perf_counter()
            runtime = (runtime_factory or _cuda_runtime)(plan['provenance']['checkpoint']['path'], device)
            runtime.synchronize()
            model_load_seconds = perf_counter() - load_start
            if runtime.model_config.get('length') != 256 or runtime.model_config.get('condition_dim') != 38:
                raise ValueError('Runtime must use bar16/38-feature checkpoint configuration')
        session = {'session_index': len(status['sessions']), 'device': str(device),
                   'feature_load_seconds': feature_load_seconds, 'model_load_seconds': model_load_seconds,
                   'status': 'running', 'model_calls_started': 0, 'model_calls_completed': 0}
        status['sessions'].append(session)
        status['status'] = 'running'
        atomic_json(status_path, status)
        for row in todo:
            key = row['case_id']
            runtime.synchronize()
            preprocessing_start = perf_counter()
            spec = _deserialize_spec(row['spec'])
            state = tuple(spec.observed.get(i) for i in range(spec.length))
            provider = runtime.make_provider(spec, conditions[key])
            runtime.synchronize()
            preprocessing_seconds = perf_counter() - preprocessing_start
            attempt = {'status': 'provider_call_started', 'model_calls_started': 1,
                       'model_calls_completed': 0, 'session_index': session['session_index']}
            status['attempts'][key] = attempt
            session['model_calls_started'] += 1
            atomic_json(status_path, status)
            try:
                runtime.synchronize()
                call_start = perf_counter()
                q = np.asarray(provider(state, 1.))
                runtime.synchronize()
                provider_seconds = perf_counter() - call_start
                attempt['model_calls_completed'] = 1
                session['model_calls_completed'] += 1
                validation_start = perf_counter()
                metadata = {**row['metadata'], 'source': plan['provenance'],
                    'checkpoint': plan['provenance']['checkpoint'], 'model_config': runtime.model_config,
                    'probability_source': 'one_frozen_full130_model_call', 'probability_runtime': runtime.description,
                    'probability_provider_seconds': provider_seconds,
                    'provider_preprocessing_seconds': preprocessing_seconds, 'model_calls': 1,
                    'probability_freeze_session': session['session_index'],
                    'provider_state': list(state), 'condition_shape': [256, 38],
                    'q_policy': policy['q'], 'scope': policy['scope']}
                case = SolverCase(key, spec, q, metadata, tuple(row['witness']))
                _validate_case(case)
                path, qpath = output / f'{key}.json', output / f'{key}.npz'
                if path.exists() or qpath.exists():
                    raise ValueError('Untracked probability output already exists; preserve and inspect it')
                temporary = qpath.with_suffix('.npz.tmp')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, logq=q)
                os.replace(temporary, qpath)
                atomic_json(path, {'version': 1, 'case_id': key, 'spec': row['spec'],
                    'metadata': metadata, 'witness': row['witness'], 'probabilities': qpath.name})
                attempt.update(status='completed', path=str(path),
                    probability_provider_seconds=provider_seconds,
                    preprocessing_seconds=preprocessing_seconds,
                    validation_and_storage_seconds=perf_counter() - validation_start)
                atomic_json(status_path, status)
            except Exception as exc:
                attempt.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
                status['status'] = session['status'] = 'failed'
                session['wall_seconds'] = perf_counter() - start
                atomic_json(status_path, status)
                raise
        entries = []
        for row in plan['requests']:
            attempt = status['attempts'][row['case_id']]
            case = load_case(attempt['path'])
            if case.case_id != row['case_id'] or _serialize_spec(case.spec) != row['spec']:
                raise ValueError('Completed probability snapshot differs from frozen request')
            entries.append({'case_id': case.case_id, 'path': attempt['path'], 'metadata': case.metadata})
        session.update(status='completed', wall_seconds=perf_counter() - start)
        status['status'] = 'completed'
        status['completed_requests'] = len(entries)
        atomic_json(status_path, status)
        manifest = {'version': 1, 'status': 'completed', 'count': len(entries), 'cases': entries,
            'families': dict(Counter(e['metadata']['family'] for e in entries)), 'cohort_identity': cohort_identity,
            'model_calls': sum(a['model_calls_started'] for a in status['attempts'].values()),
            'selected_work_count': plan['selected_work_count'], 'policy': policy,
            'timing': {'sessions': status['sessions'],
                'probability_provider_seconds_sum': sum(e['metadata']['probability_provider_seconds'] for e in entries),
                'provider_preprocessing_seconds_sum': sum(e['metadata']['provider_preprocessing_seconds'] for e in entries)}}
        atomic_json(manifest_path, manifest)
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'freeze'))
    parser.add_argument('--config', default='configs/shared_rhythm_real.yaml')
    parser.add_argument('--output')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    import yaml
    config = yaml.safe_load(Path(args.config).read_text())
    output = args.output or config['output']
    if args.command == 'prepare':
        result = prepare_real_requests(output, revision_root=config['revision_root'], options=config['requests'])
    else:
        result = freeze_real_probabilities(output, device=args.device)
    print(json.dumps({k: result[k] for k in ('status', 'selected_count', 'selected_work_count', 'count', 'model_calls') if k in result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
