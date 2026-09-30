"""Synthetic full-vocabulary fixtures for R shared-rhythm melody spans.

These are controlled probability tables, not neural predictions or real music
requests. Visibility and beta change the target while retaining exactly the
same q for each R/seed. Every fixture has an independently checked witness.
"""
from __future__ import annotations

from itertools import product
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from tri.domain.music import CountRule, MusicSpec, verify_music
from tri.evaluation.solver_cases import SolverCase, _serialize_spec, load_case
from tri.runtime import atomic_json, exclusive_run


VISIBILITY = ('unknown', 'partial', 'known')


def make_shared_rhythm_case(*, R=2, L=16, D=16, K=4, beta=0., visibility='unknown',
                            seed=20260931, logit_scale=.7, hold_bias=2., rest_bias=-1.):
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in (R, L, D, K, seed)):
        raise ValueError('R/L/D/K/seed must be integers')
    if not (R >= 2 and L >= 1 and 2 <= D <= 128 and 0 <= K <= L):
        raise ValueError('Invalid shared-rhythm dimensions')
    if visibility not in VISIBILITY or not math.isfinite(beta) or beta < 0:
        raise ValueError('Invalid visibility or beta')
    if not all(math.isfinite(x) for x in (logit_scale, hold_bias, rest_bias)) or logit_scale <= 0:
        raise ValueError('Invalid synthetic probability parameters')
    low = max(0, min(60 - (D - 1) // 2, 129 - D))
    pitches = tuple(range(low, low + D - 1))
    anchor = 60
    assert anchor in pitches
    length = R * (L + 1) + 1
    spans = tuple(tuple(range(1 + r * (L + 1), 1 + r * (L + 1) + L)) for r in range(R))
    onset_indices = {j * L // K for j in range(K)} if K else set()
    tokens = [anchor + 2] * length
    for span in spans:
        for j, position in enumerate(span):
            tokens[position] = anchor + 2 if j in onset_indices else 1
    hidden = {position for span in spans for position in span}
    observed = {i: token for i, token in enumerate(tokens) if i not in hidden}
    # A prefix contains one onset at the reference L16/K4, leaving three
    # genuinely unknown onsets. Equally spaced positions 0/4/8/12 would expose
    # all K onsets and accidentally make "partial" a known-template fixture.
    visible_count = 0 if visibility == 'unknown' else L if visibility == 'known' else max(1, L // 4)
    observed.update({position: tokens[position] for position in spans[0][:visible_count]})
    relations = tuple((spans[0][j], spans[r][j]) for r in range(1, R) for j in range(L))
    spec = MusicSpec(length=length, pitches=pitches, observed=observed,
        fixed_soundings={i: anchor for i in observed}, equal_onsets=relations,
        onset_counts=tuple(CountRule(span, K) for span in spans), motion_cost=float(beta))
    witness = tuple(tokens)
    if not verify_music(witness, spec).valid:
        raise RuntimeError('Shared-rhythm fixture has an invalid original-token witness')
    logits = np.random.default_rng(seed).normal(0., logit_scale, size=(length, 130))
    logits[:, 0] += rest_bias
    logits[:, 1] += hold_bias
    logq = logits - logsumexp(logits, axis=1, keepdims=True)
    beta_label = format(beta, '.12g').replace('.', 'p')
    known_onsets = sum(tokens[pos] >= 2 for pos in spans[0][:visible_count])
    remaining = L - visible_count
    remaining_onsets = K - known_onsets
    template_count = math.comb(remaining, remaining_onsets)
    metadata = {'family': 'synthetic_control', 'R': R, 'L': L, 'D': D, 'K': K, 'beta': float(beta),
        'visibility': visibility, 'seed': int(seed), 'length': length, 'spans': [list(span) for span in spans],
        'emittable_pitch_count': len(pitches), 'sounding_state_count_including_silence': D,
        'observed_positions_in_first_span': visible_count, 'unknown_positions': length - len(observed),
        'onset_templates_consistent_with_observations_and_count': template_count,
        'expected_support': 'feasible', 'witness_verified': True, 'guard': 'NOTE_60',
        'q_source': 'synthetic_full130_normal_logits', 'q_pairing_key': f'R{R}_L{L}_s{seed}',
        'q_parameters': {'scale': logit_scale, 'hold_bias': hold_bias, 'rest_bias': rest_bias},
        'q_policy': 'Exactly identical original full130 q across beta and visibility for the same R/L/seed; observations replace q by delta, no working-domain renormalization.',
        'scope': 'Constructed multi-span control, not stitched learned q or an actual music request.'}
    return SolverCase(f'shared_R{R}_L{L}_D{D}_K{K}_{visibility}_beta{beta_label}_s{seed}',
                      spec, logq, metadata, witness)


def build_shared_rhythm_cases(grid):
    required = ('R', 'L', 'D', 'K', 'betas', 'visibility', 'q_seeds')
    if any(key not in grid for key in required):
        raise ValueError('Shared-rhythm grid is missing required dimensions')
    for key in ('R', 'betas', 'visibility', 'q_seeds'):
        if not grid[key] or len(set(grid[key])) != len(grid[key]):
            raise ValueError(f'Grid {key} requires distinct values')
    probability = grid.get('probabilities', {})
    cases = [make_shared_rhythm_case(R=R, L=grid['L'], D=grid['D'], K=grid['K'], beta=beta,
              visibility=visibility, seed=seed, **probability)
             for visibility, R, beta, seed in product(grid['visibility'], grid['R'], grid['betas'], grid['q_seeds'])]
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError('Ambiguous case IDs')
    return cases


def prepare_shared_rhythm_cases(grid, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(output / 'prepare.lock'):
        options = {'version': 1, 'family': 'synthetic_control', 'grid': grid}
        config_path = output / 'preparation.json'
        if config_path.exists() and json.loads(config_path.read_text()) != options:
            raise ValueError('Existing shared-rhythm cases have different preparation options')
        manifest_path = output / 'cases.json'
        if manifest_path.exists():
            if not config_path.exists():
                raise ValueError('Existing shared-rhythm manifest lacks preparation config')
            manifest = json.loads(manifest_path.read_text())
            for entry in manifest['cases']:
                load_case(entry['path'])
            return manifest
        atomic_json(config_path, options)
        cases = build_shared_rhythm_cases(grid)
        entries = []
        for case in cases:
            path = output / f'{case.case_id}.json'
            qpath = path.with_suffix('.npz')
            record = {'version': 1, 'case_id': case.case_id, 'spec': _serialize_spec(case.spec),
                      'metadata': case.metadata, 'witness': list(case.witness), 'probabilities': qpath.name}
            if path.exists():
                previous = load_case(path)
                if json.loads(path.read_text()) != record or not np.array_equal(previous.logq, case.logq):
                    raise ValueError('Existing fixture differs from its declared grid')
            else:
                if qpath.exists():
                    with np.load(qpath, allow_pickle=False) as data:
                        if not np.array_equal(data['logq'], case.logq):
                            raise ValueError('Existing probability snapshot differs from grid')
                else:
                    np.savez_compressed(qpath, logq=case.logq)
                atomic_json(path, record)
            entries.append({'case_id': case.case_id, 'path': str(path), 'metadata': case.metadata})
        manifest = {'version': 1, 'count': len(entries), 'families': {'synthetic_control': len(entries)},
                    'grid': grid, 'cases': entries,
                    'policy': 'Synthetic full130 probabilities with independently verified witnesses; original observations are delta, shared q across beta/visibility, no neural or real-music claim.'}
        atomic_json(manifest_path, manifest)
        return manifest
