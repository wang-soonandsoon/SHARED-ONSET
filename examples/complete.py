"""Generate motion-weighted joint completions and export MIDI alternatives."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from shared_onset import AdaptiveOnsetRejectionSampler, verify_music
from tri.data.midi import write_grid_midi
from tri.evaluation.solver_cases import load_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, help='Extracted frozen JSON case; otherwise use synthetic q')
    parser.add_argument('--out', type=Path, default=Path('runs/example'))
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    if args.case:
        case = load_case(args.case)
        spec, q = case.spec, case.logq
    else:
        from minimal import make_request
        spec, q, _ = make_request()
        spec = replace(spec, motion_cost=0.02)
    sampler = AdaptiveOnsetRejectionSampler(spec, q, proposal='visible_early')
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(args.samples):
        draw = sampler.sample_full(np.random.default_rng(args.seed + index))
        checked = verify_music(draw.tokens, spec)
        assert checked.valid, checked.violations
        write_grid_midi(list(draw.tokens), args.out / f'completion_{index + 1:02d}.mid',
                        initial_pitch=spec.initial_pitch)
        records.append({'seed': args.seed + index, 'tokens': draw.tokens, 'diagnostics': draw.diagnostics})
    (args.out / 'samples.json').write_text(json.dumps(records, indent=2) + '\n')
    print(f'{len(records)} verified full-target completions saved to {args.out}')


if __name__ == '__main__':
    main()
