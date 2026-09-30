"""Frozen, full-vocabulary inputs for solver comparisons, without new training.

The cheap load path deliberately imports neither torch nor the model/data
pipelines: isolated solver workers must not inherit their memory overhead.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re

import numpy as np

from tri.domain.music import CountRule, MusicSpec, verify_music
from tri.runtime import atomic_json

CASE_VERSION = 1
DEFAULT_SEEDS = (20260921, 20260922, 20260923)
KINDS = ('unknown', 'partial', 'known', 'harmony')


@dataclass(frozen=True)
class SolverCase:
    case_id: str
    spec: MusicSpec
    logq: np.ndarray
    metadata: dict
    witness: tuple[int, ...] | None = None


def _serialize_spec(spec):
    # The same on-disk schema as cohorts.serialize_spec, kept here so loading
    # a solver input does not import the MIDI/model evaluation stack.
    return {'length': spec.length, 'pitches': list(spec.pitches),
            'observed': {str(k): v for k, v in spec.observed.items()},
            'initial_pitch': spec.initial_pitch, 'enforce_end': spec.enforce_end,
            'end_pitch': spec.end_pitch,
            'fixed_soundings': {str(k): v for k, v in spec.fixed_soundings.items()},
            'equal_onsets': [list(p) for p in spec.equal_onsets],
            'onset_counts': [{'positions': list(c.positions), 'count': c.count} for c in spec.onset_counts],
            'pitch_ranges': {str(k): list(v) for k, v in spec.pitch_ranges.items()},
            'pitch_classes': {str(k): list(v) for k, v in spec.pitch_classes.items()},
            'max_adjacent_interval': spec.max_adjacent_interval, 'motion_cost': spec.motion_cost}


def _deserialize_spec(data):
    data = dict(data)
    for key in ('observed', 'fixed_soundings', 'pitch_classes', 'pitch_ranges'):
        data[key] = {int(k): v for k, v in data.get(key, {}).items()}
    data['onset_counts'] = tuple(CountRule(tuple(c['positions']), c['count']) for c in data['onset_counts'])
    return MusicSpec(**data)


def _identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'modified_ns': stat.st_mtime_ns}


def _probabilities(length, seed):
    """Positive probabilities on ALL 130 outputs; independent of pitch crop."""
    logits = np.random.default_rng(seed).normal(0., 1., size=(length, 130))
    logits[:, :2] += 2.
    logits -= logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def _soundings(tokens, initial=None):
    pitch = initial
    result = []
    for token in tokens:
        if token == 0:
            pitch = None
        elif token >= 2:
            pitch = token - 2
        result.append(pitch)
    return result


def _dimensions(spec):
    pitches = set(spec.pitches)
    pitches.update(p for p in (spec.initial_pitch, spec.end_pitch, *spec.fixed_soundings.values()) if p is not None)
    spans = tuple(tuple(pair[j] for pair in sorted(spec.equal_onsets)) for j in (0, 1))
    known = sum(i in spec.observed for i in spans[0])
    missing = [i for i in range(spec.length) if i not in spec.observed]
    probes = [next((f'y{i}' for i in span if i not in spec.observed), None) for span in spans]
    return {'L': len(spans[0]), 'D': len(pitches) + 1,
            'K': spec.onset_counts[0].count if spec.onset_counts else None,
            'length': spec.length, 'emittable_pitch_count': len(spec.pitches),
            'visible_fraction_first_span': known / len(spans[0]),
            'unknown_fraction_both_spans': len(missing) / (2 * len(spans[0])),
            'unknown_positions': len(missing), 'probe_variables': [p for p in probes if p is not None]}


def make_controlled_case(*, L=16, D=16, K=4, observed_fraction=0., guard='NOTE',
                         seed=DEFAULT_SEEDS[0], infeasible=False, max_interval=None,
                         family='controlled', sweeps=()):
    """Construct an anchored pair and an independently checked feasible witness.

    D counts silence plus pitches. Visibility is the fraction of the FIRST
    span observed; 1.0 therefore means a known template and an unknown B span.
    """
    if not (L >= 1 and 3 <= D <= 128 and 0 <= K <= L and 0 <= observed_fraction <= 1):
        raise ValueError('invalid controlled case dimensions')
    if guard not in ('REST', 'NOTE', 'HOLD'):
        raise ValueError('unknown guard')
    if infeasible and (K != 0 or guard != 'HOLD' or observed_fraction != 0):
        raise ValueError('infeasible fixture requires K=0, HOLD guards and unknown spans')
    low = max(0, min(60 - (D - 1) // 2, 129 - D))
    pitches = tuple(range(low, low + D - 1))
    anchor_pitch = 60
    # For D=3 use two distinct C/D pitches, useful for exhaustive fixtures.
    if D == 3:
        pitches = (60, 62)
    length = 2 * L + 3
    spans = (tuple(range(1, L + 1)), tuple(range(L + 2, 2 * L + 2)))
    guard_token = {'REST': 0, 'NOTE': anchor_pitch + 2, 'HOLD': 1}[guard]
    tokens = [0] * length
    tokens[0] = anchor_pitch + 2
    tokens[L + 1] = tokens[-1] = guard_token
    onset_indices = {j * L // K for j in range(K)} if K else set()
    for span in spans:
        sounding = anchor_pitch if span is spans[0] or guard != 'REST' else None
        for j, pos in enumerate(span):
            tokens[pos] = anchor_pitch + 2 if j in onset_indices else (1 if sounding is not None else 0)
            if j in onset_indices:
                sounding = anchor_pitch
    hidden = set(spans[0] + spans[1])
    observed = {i: value for i, value in enumerate(tokens) if i not in hidden}
    visible_count = round(observed_fraction * L)
    visible_indices = {j * L // visible_count for j in range(visible_count)} if visible_count else set()
    observed.update({spans[0][j]: tokens[spans[0][j]] for j in visible_indices})
    soundings = _soundings(tokens)
    anchors = {i: soundings[i] for i in observed}
    if infeasible:
        # No NOTE in B can change its inherited C into the final required D.
        anchors[length - 1] = 62
    spec = MusicSpec(length=length, pitches=pitches, observed=observed,
                     fixed_soundings=anchors, equal_onsets=tuple(zip(*spans)),
                     onset_counts=tuple(CountRule(s, K) for s in spans),
                     motion_cost=.03, max_adjacent_interval=max_interval)
    witness = None if infeasible else tuple(tokens)
    if witness is not None and not verify_music(witness, spec).valid:
        raise RuntimeError('controlled witness does not satisfy its specification')
    name = (f'{family}_L{L}_D{D}_K{K}_v{visible_count}of{L}_{guard.lower()}'
            f'_interval{max_interval if max_interval is not None else "any"}'
            f'{"_infeasible" if infeasible else ""}_s{seed}')
    metadata = {'family': family, 'sweeps': list(sweeps), **_dimensions(spec), 'seed': seed,
                'guard': guard, 'motion_cost': .03, 'max_adjacent_interval': max_interval,
                'expected_support': 'infeasible' if infeasible else 'feasible',
                'witness_verified': witness is not None,
                'probability_source': 'full130_normal_logits_rest_hold_bias2',
                'query_source': 'constructed witness' if witness else 'deterministic REST; zero allowed',
                'support_reason': 'K=0 cannot change inherited pitch 60 into fixed final HOLD 62' if infeasible else 'independent semantic witness'}
    metadata['query_evidence'] = {p: int(witness[int(p[1:])]) if witness else 0 for p in metadata['probe_variables']}
    return SolverCase(name, spec, _probabilities(length, seed), metadata, witness)


def build_controlled_cases(seeds=DEFAULT_SEEDS):
    """One-factor sweeps, deduplicating their common reference setting."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('provide distinct probability seeds')
    settings = {}

    def add(sweep, **options):
        config = dict(L=16, D=16, K=4, observed_fraction=0., guard='NOTE', infeasible=False, max_interval=None)
        config.update(options)
        key = tuple(config.items())
        settings.setdefault(key, [config, []])[1].append(sweep)

    for length in (8, 16, 32, 64):
        add('length', L=length, K=length // 4)
    for size in (8, 16, 32, 64):
        add('pitch_states', D=size)
    for count in (0, 1, 4, 8, 16):
        add('onset_count', K=count)
    for visible in (0., .25, .5, 1.):
        add('visibility', observed_fraction=visible)
    for guard in ('REST', 'NOTE', 'HOLD'):
        add('boundary', guard=guard)
    add('infeasible', K=0, guard='HOLD', infeasible=True)
    add('zero_onset_hold', K=0, guard='HOLD')
    add('sparse_interval', max_interval=2)
    return [make_controlled_case(**config, seed=int(seed), sweeps=sweeps)
            for config, sweeps in settings.values() for seed in seeds]


def build_tiny_cases():
    """Tiny complete-token oracles, including carried pitches and zero mass."""
    cases = [make_controlled_case(L=length, D=3, K=count, guard=guard,
                observed_fraction=visible, seed=seed, family='tiny', sweeps=('correctness',), infeasible=infeasible)
             for length, count, guard, visible, seed, infeasible in
             [(2, 1, 'NOTE', 0., 31, False), (2, 1, 'HOLD', 0., 32, False),
              (3, 1, 'REST', 1/3, 33, False), (2, 0, 'HOLD', 0., 34, False),
              (2, 0, 'HOLD', 0., 35, True)]]
    spec = MusicSpec(length=7, pitches=(60, 62), initial_pitch=55,
                     observed={0: 1, 3: 1, 6: 1}, fixed_soundings={0: 55, 3: 55, 6: 55},
                     equal_onsets=((1, 4), (2, 5)),
                     onset_counts=(CountRule((1, 2), 0), CountRule((4, 5), 0)), motion_cost=.07)
    witness = (1,) * 7
    assert verify_music(witness, spec).valid
    metadata = {'family': 'tiny', 'sweeps': ['correctness'], **_dimensions(spec), 'seed': 36,
                'guard': 'HOLD', 'expected_support': 'feasible', 'witness_verified': True,
                'probability_source': 'full130_normal_logits_rest_hold_bias2',
                'query_source': 'constructed witness', 'query_evidence': {'y1': 1, 'y4': 1},
                'support_reason': 'carried pitch 55 need not be an emittable NOTE'}
    cases.append(SolverCase('tiny_carried_pitch_outside_vocabulary_s36', spec,
                            _probabilities(7, 36), metadata, witness))
    return cases


def build_learned_cases(revision_root=Path('runs/revision'), device='cuda:0'):
    """Freeze one original model call per selected validation request.

    Selection is first-per-kind in the already frozen request order, independent
    of model outputs. No clean token array, training run or test case is used.
    """
    import torch
    from dataclasses import asdict
    from tri.models.grid import ModelProbabilityProvider
    from tri.models.train import load_checkpoint

    torch.set_num_threads(2)
    revision_root = Path(revision_root).resolve()
    cases = []
    for stage in ('short', 'bar8', 'bar16'):
        cohort_path = revision_root / stage / 'cohort.json'
        plan = json.loads(cohort_path.read_text())
        evaluation_config = json.loads((revision_root / stage / 'evaluation/config.json').read_text())
        if plan['status'] != 'completed' or plan['options']['split'] != 'validation':
            raise ValueError('learned solver cases require frozen completed validation cohorts')
        for key in ('dataset', 'chords'):
            if _identity(plan['options'][key]['path']) != plan['options'][key]:
                raise ValueError(f'frozen {stage} {key} changed')
        checkpoint = Path(evaluation_config['checkpoint']['path'])
        if _identity(checkpoint) != evaluation_config['checkpoint']:
            raise ValueError('frozen checkpoint changed')
        selected = [next(row for row in plan['requests'] if row['kind'] == kind) for kind in KINDS]
        model = load_checkpoint(checkpoint, device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        with np.load(plan['options']['dataset']['path'], allow_pickle=False) as data, np.load(plan['options']['chords']['path'], allow_pickle=False) as chords:
            works = data['work_ids'].astype(str)
            starts = data['start_cells']
            splits = data['splits'].astype(str)
            chord_features = chords['chord_features']
            for row in selected:
                index = row['source_index']
                if (works[index], int(starts[index])) != (row['work_id'], row['source_start_cell']) or splits[index] not in ('validation', 'val'):
                    raise ValueError('selected request identity or split changed')
                spec = _deserialize_spec(row['spec'])
                if spec.length != model.config.length:
                    raise ValueError('checkpoint length does not match frozen request')
                editable = tuple(i for i in range(spec.length) if i not in spec.observed)
                state = tuple(spec.observed.get(i) for i in range(spec.length))
                condition = chord_features[index] if model.config.condition_dim else None
                provider = ModelProbabilityProvider(model, editable, condition)
                q = provider(state, 1.)
                metadata = {'family': 'learned', 'sweeps': ['learned'], **_dimensions(spec),
                            'stage': stage, 'kind': row['kind'], 'source_case_id': row['case_id'],
                            'work_id': row['work_id'], 'split': 'validation', 'seed': None,
                            'expected_support': 'feasible', 'witness_verified': bool(row.get('witness_verified')),
                            'probability_source': 'frozen_original_checkpoint_full130',
                            'model_calls': 1, 'model_noise': 1., 'device': device,
                            'checkpoint': evaluation_config['checkpoint'], 'checkpoint_seed': evaluation_config['checkpoint_seed'],
                            'model_config': asdict(model.config), 'cohort': _identity(cohort_path),
                            'dataset': plan['options']['dataset'], 'chords': plan['options']['chords'],
                            'source_request': row, 'provider_state': list(state),
                            'query_source': 'deterministic REST clamp; zero conditional mass allowed',
                            'selection': 'first frozen request per kind; no output-based selection',
                            'guard': 'original_context'}
                metadata['query_evidence'] = {p: 0 for p in metadata['probe_variables']}
                cases.append(SolverCase(f'learned_{stage}_{row["kind"]}', spec, q, metadata))
        del model
    return cases


def _validate_case(case):
    q = np.asarray(case.logq)
    if q.shape != (case.spec.length, 130) or not np.isfinite(q).all():
        raise ValueError('frozen study cases require finite N by 130 log probabilities')
    if not np.allclose(np.exp(q).sum(axis=1), 1., atol=1e-12, rtol=1e-12):
        raise ValueError('probabilities must be normalized on the full vocabulary')
    if case.witness is not None and not verify_music(case.witness, case.spec).valid:
        raise ValueError('saved witness is not valid')


def load_case(path):
    path = Path(path).resolve()
    record = json.loads(path.read_text())
    if record['version'] != CASE_VERSION:
        raise ValueError('unsupported solver case version')
    with np.load(path.parent / record['probabilities'], allow_pickle=False) as data:
        q = data['logq'].copy()
    case = SolverCase(record['case_id'], _deserialize_spec(record['spec']), q, record['metadata'],
                      tuple(record['witness']) if record['witness'] is not None else None)
    _validate_case(case)
    return case


def save_cases(cases, output, *, policy=None):
    """Save immutable JSON+NPZ cases, preserving an identical previous snapshot."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = list(cases)
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError('duplicate solver case IDs')
    entries = []
    for case in cases:
        _validate_case(case)
        if re.fullmatch(r'[A-Za-z0-9_.-]+', case.case_id) is None:
            raise ValueError('case IDs must be safe filenames')
        path = output / f'{case.case_id}.json'
        qpath = path.with_suffix('.npz')
        record = {'version': CASE_VERSION, 'case_id': case.case_id, 'spec': _serialize_spec(case.spec),
                  'metadata': case.metadata, 'witness': list(case.witness) if case.witness is not None else None,
                  'probabilities': qpath.name}
        if path.exists():
            saved = load_case(path)
            if json.loads(path.read_text()) != record or not np.array_equal(saved.logq, case.logq):
                raise ValueError(f'case already exists with different content: {path}')
        else:
            if qpath.exists():
                with np.load(qpath, allow_pickle=False) as data:
                    if not np.array_equal(data['logq'], case.logq):
                        raise ValueError(f'unmatched probability snapshot already exists: {qpath}')
            else:
                np.savez_compressed(qpath, logq=case.logq)
            atomic_json(path, record)
        entries.append({'case_id': case.case_id, 'path': str(path), 'metadata': case.metadata})
    manifest = {'version': CASE_VERSION, 'status': 'completed', 'count': len(entries),
                'families': dict(Counter(c.metadata['family'] for c in cases)), 'cases': entries,
                'policy': policy if policy is not None else 'Same frozen full130 q/spec per solver; constructed controlled support; first-per-kind validation weights; no new training or hidden-token input.'}
    dest = output / 'cases.json'
    if dest.exists():
        if json.loads(dest.read_text()) != manifest:
            raise ValueError('existing manifest has different cases; choose another output')
    else:
        atomic_json(dest, manifest)
    return manifest


def prepare_solver_cases(output, *, include_learned=True, seeds=DEFAULT_SEEDS,
                         revision_root=Path('runs/revision'), device='cuda:0'):
    output = Path(output).resolve()
    options = {'version': CASE_VERSION, 'include_learned': include_learned,
               'seeds': list(seeds), 'revision_root': str(Path(revision_root).resolve()), 'device': device}
    options_path = output / 'preparation.json'
    manifest_path = output / 'cases.json'
    if options_path.exists() and json.loads(options_path.read_text()) != options:
        raise ValueError('case preparation options changed; choose another output')
    if manifest_path.exists():
        if not options_path.exists():
            raise ValueError('case manifest lacks matching preparation options')
        manifest = json.loads(manifest_path.read_text())
        for row in manifest['cases']:
            load_case(row['path'])
        return manifest
    atomic_json(options_path, options)
    cases = build_tiny_cases() + build_controlled_cases(seeds)
    if include_learned:
        cases += build_learned_cases(revision_root, device)
    return save_cases(cases, output)
