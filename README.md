# SHARED-ONSE

Core implementation for **Exact Joint Sampling for Shared-Onset Music Infilling**.

**Zhenduo Wang, Xu Liu, Dan Zhang, Lingling Li, Licheng Jiao, Fang Liu, Wenping Ma**

Xidian University

> **Core preview available. Full code coming soon.**
>
> This release contains the onset-reset decomposition, a runnable synthetic
> example, and independent correctness tests. The full experimental pipeline
> and motion-weighted rejection samplers will be released separately.

## What the core does

Several music spans share their NOTE-onset positions while retaining separate
pitch and HOLD/REST paths. With zero internal motion cost, a new NOTE resets the
dependence on the preceding pitch. The sampler uses that structure to:

1. Sum each span's compatible pitch and HOLD/REST paths into interval weights.
2. Multiply the per-span weights and run dynamic programming over onset times.
3. Draw a shared onset pattern, then draw each span's pitch path conditioned on it.

The result is a complete **joint sample**, rather than a collection of independent
token marginals. The implementation uses log-space arithmetic and avoids both
onset-template enumeration and a Cartesian product of all spans' pitch states.
For R spans of length L and a pitch-state size D, interval preparation costs
O(R L² D), and the onset-time DP with K onsets costs O(L² K).

## Quick start

Python 3.11 or newer is required. The core runs on CPU and only needs NumPy and
SciPy; tests additionally use pytest. No dataset, checkpoint, or GPU is needed.

```bash
git clone https://github.com/wang-soonandsoon/SHARED-ONSE.git
cd SHARED-ONSE
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python examples/minimal.py
python examples/minimal.py --spans 8 --length 16 --onsets 4
python -m pytest -q
```

The example prints the shared onset pattern and the individual token sequences.
Its probabilities are synthetic; it demonstrates the inference mechanism, not
music quality or the paper's experimental results.

## Inputs and supported target

`MusicSpec` describes token observations, pitch restrictions, onset equalities,
and sounding-state anchors. `OnsetResetMusicInference(spec, log_probs)` accepts
a fixed array of shape `[spec.length, 130]` containing normalized log
probabilities over the complete output vocabulary:

- `0`: REST; `1`: HOLD; `pitch + 2`: NOTE with MIDI pitch 0–127.
- HOLD requires an active preceding pitch. NOTE starts an onset, including when
  its pitch repeats the preceding pitch.
- At least two contiguous, disjoint, equal-length spans are related at every
  corresponding onset position. Observed, sounding-anchored separators keep
  the spans distinct.
- Every position outside the spans must be observed and sounding-anchored.
  Observed tokens inside spans, pitch ranges, and pitch-class constraints are
  supported as well.
- Onset counts may cover a whole span or a single position. Zero onsets are
  supported when compatible with HOLD/REST and the boundaries.
- The base sampler requires `motion_cost=0` and no `max_adjacent_interval`.

For the default base target, the unnormalized weight is the product of the
supplied probabilities at unobserved positions, multiplied by the indicator
of all specified constraints. Observed tokens have unit weight. Restricting
the working pitch vocabulary does **not** renormalize the supplied probabilities.
Temporary evidence conditions this same target without changing its weights.

```python
import numpy as np
from shared_onset import OnsetResetMusicInference

# spec and log_probs: construct as in examples/minimal.py
sampler = OnsetResetMusicInference(spec, log_probs)
log_z = sampler.log_partition()
tokens = sampler.sample_full(np.random.default_rng(7))
```

Conditional queries accept token assignments such as `{"y1": 62}` (NOTE C4 at
position 1). The same evidence can be passed to `log_partition` and `sample_full`.
An infeasible or zero-probability condition has log partition `-inf`; sampling
it raises `ZeroMass`.

The optional `boundary_motion_cost` retains motion scores only at outside
positions, including right guards. With this option, returned probabilities
and partitions describe that **proposal distribution**, not the paper's full
motion-weighted target. The rejection correction for the full target is not
part of this preview.

## Correctness checks

The tests compare partition functions and token weights against independent
original-token enumeration for small R = 2–4 cases. They also cover conditional
probabilities, empirical joint sampling frequencies, zero onsets, visible
tokens, HOLD and right boundaries, optional outside motion scores, immutable
probabilities, and invalid or unsupported inputs.

## Release scope

| Included now | Planned for the full release |
| --- | --- |
| Onset-reset interval decomposition and DP | Motion-weighted exact rejection sampling and proposal variants |
| Joint token sampling and conditional queries | Model inference and checkpoint integration |
| Music specification and semantic checker | Dataset preparation and real-request construction |
| Synthetic example and correctness tests | Baselines, evaluation scripts, and figure reproduction |

This preview is not a complete reproduction package for the paper. Datasets,
model weights, manuscript sources, and internal experiment records are not
bundled here. No third-party dataset or checkpoint is redistributed.

## Source map

- `src/shared_onset/onset_reset.py`: interval weights, onset-time DP, and sampling.
- `src/shared_onset/music.py`: token semantics, request specification, and checker.
- `src/shared_onset/_music_base.py`: token transition and probability plumbing.
- `src/shared_onset/contracts.py`: budgets and conditional-query interfaces.
- `examples/minimal.py`: self-contained synthetic request.
- `tests/test_onset_reset.py`: independent enumeration and sampling checks.
