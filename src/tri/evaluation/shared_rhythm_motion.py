"""Localize discarded motion scores on frozen-q proposal paths.

This is a cost-factor diagnostic, not a timing study or a new sampler. Paths
are sampled independently from the existing boundary proposal. The categories
use initial visible anchors/tokens only; generated values never make a factor
eligible for retention. No hidden original melody is loaded.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from tri.domain.music import verify_music
from tri.errors import InvalidSpecification
from tri.evaluation.solver_cases import load_case
from tri.inference.exact import Budget
from tri.inference.onset_reset import OnsetResetMusicInference
from tri.runtime import atomic_json


CATEGORIES = ('outside', 'fixed_predecessor', 'observed_note', 'unknown_endpoints')


def motion_parts(tokens, spec, spans):
    """Partition the original soft score without inference-compiler helpers."""
    checked = verify_music(tokens, spec)
    if not checked.valid:
        raise InvalidSpecification('Motion diagnosis needs a valid original-token path')
    inside = {position for span in spans for position in span}
    previous = spec.initial_pitch
    jumps = {name: 0 for name in CATEGORIES}
    transitions = {name: 0 for name in CATEGORIES}
    for position, token in enumerate(tokens):
        if token == 0:
            previous = None
        elif token >= 2:
            pitch = token - 2
            if position not in inside:
                name = 'outside'
            elif position == 0 or position - 1 in spec.fixed_soundings:
                name = 'fixed_predecessor'
            elif position in spec.observed:
                name = 'observed_note'
            else:
                name = 'unknown_endpoints'
            jumps[name] += 0 if previous is None else abs(pitch - previous)
            transitions[name] += 1
            previous = pitch
    scores = {name: -spec.motion_cost * value for name, value in jumps.items()}
    if not np.isclose(sum(scores.values()), checked.soft_score, atol=1e-10, rtol=1e-12):
        raise AssertionError('Motion decomposition does not preserve original soft score')
    return {'jumps': jumps, 'transitions': transitions, 'scores': scores,
            'original_soft_score': checked.soft_score,
            'boundary_residual_score': checked.soft_score - scores['outside']}


def diagnose(manifest_path, output, *, samples=32, seed=20260940):
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise ValueError('samples must be positive')
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    records = []
    for index, entry in enumerate(manifest['cases']):
        case = load_case(entry['path'])
        engine = OnsetResetMusicInference(replace(case.spec, motion_cost=0.), case.logq,
            Budget(), boundary_motion_cost=case.spec.motion_cost)
        parts = []
        for replicate in range(samples):
            rng = np.random.default_rng(np.random.SeedSequence([seed, index, replicate]))
            tokens = engine.sample_full(rng)
            parts.append(motion_parts(tokens, case.spec, engine.spans))
        records.append({'case_id': case.case_id, 'metadata': case.metadata,
            'independent_proposal_samples': samples,
            'mean_jumps': {name: float(np.mean([p['jumps'][name] for p in parts])) for name in CATEGORIES},
            'mean_scores': {name: float(np.mean([p['scores'][name] for p in parts])) for name in CATEGORIES},
            'mean_current_residual_score': float(np.mean([p['boundary_residual_score'] for p in parts])),
            'mean_additional_visible_score': float(np.mean([p['scores']['fixed_predecessor'] + p['scores']['observed_note'] for p in parts])),
            'parts': parts})
    report = {'manifest': str(manifest_path), 'seed': seed, 'samples_per_case': samples,
        'cases': records, 'case_count': len(records),
        'scope': 'Independent draws from the existing boundary proposal, not accepted target paths. Factor-score diagnostics only; no latency or true acceptance-rate claim.',
        'eligible_information': 'Initial fixed predecessor (including initial state) or initially observed NOTE destination; never newly generated token values.',
        'next_use': 'A candidate retention must separately preserve exact rank-one onset structure and pass original-token target tests before comparison.'}
    atomic_json(Path(output), report)
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20260940)
    args = parser.parse_args()
    report = diagnose(args.manifest, args.out, samples=args.samples, seed=args.seed)
    print(json.dumps({'cases': report['case_count'], 'samples_per_case': args.samples}))


if __name__ == '__main__':
    main()
