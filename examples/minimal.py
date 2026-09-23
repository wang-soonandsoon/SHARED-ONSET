"""Sample aligned spans using synthetic probabilities; no model or data needed."""
import argparse

import numpy as np
from scipy.special import logsumexp

from shared_onset import CountRule, MusicSpec, OnsetResetMusicInference, note_token, verify_music


def make_request(spans=3, length=8, onsets=2, seed=7):
    """Observed C4 guards separate the spans and anchor their sounding state."""
    pitches = (60, 62, 64, 67)
    positions = tuple(tuple(range(1 + r * (length + 1), 1 + r * (length + 1) + length))
                      for r in range(spans))
    guards = (0, *(span[-1] + 1 for span in positions))
    spec = MusicSpec(
        length=spans * (length + 1) + 1,
        pitches=pitches,
        observed={i: note_token(60) for i in guards},
        fixed_soundings={i: 60 for i in guards},
        equal_onsets=tuple((positions[0][j], positions[r][j])
                           for r in range(1, spans) for j in range(length)),
        onset_counts=tuple(CountRule(span, onsets) for span in positions),
    )
    logits = np.random.default_rng(seed).normal(size=(spec.length, 130))
    logits[:, :2] += 1.0
    logits[:, [note_token(p) for p in pitches]] += 2.0
    log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
    return spec, log_probs, positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spans', type=int, default=3)
    parser.add_argument('--length', type=int, default=8)
    parser.add_argument('--onsets', type=int, default=2)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if args.spans < 2 or args.length < 1 or not 0 <= args.onsets <= args.length:
        parser.error('Require spans >= 2, length >= 1, and 0 <= onsets <= length.')
    spec, q, positions = make_request(args.spans, args.length, args.onsets, args.seed)
    engine = OnsetResetMusicInference(spec, q)
    log_z = engine.log_partition()
    tokens = engine.sample_full(np.random.default_rng(args.seed + 1))
    checked = verify_music(tokens, spec)
    assert checked.valid, checked.violations
    print(f'Base target: R={args.spans}, L={args.length}, K={args.onsets}, motion_cost=0')
    print(f'log Z = {log_z:.9f}')
    for r, span in enumerate(positions, 1):
        rhythm = ''.join('x' if tokens[i] >= 2 else '.' for i in span)
        names = ['REST' if tokens[i] == 0 else 'HOLD' if tokens[i] == 1
                 else f'N{tokens[i] - 2}' for i in span]
        print(f'Span {r}: {rhythm} | {" ".join(names)}')
    print('Joint sample verified. x = NOTE onset; . = HOLD or REST.')


if __name__ == '__main__':
    main()
