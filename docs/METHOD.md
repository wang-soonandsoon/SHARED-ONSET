# Method and code map

We complete R disjoint, equally long, aligned monophonic passages with one unknown shared onset pattern. Pitches may differ across passages. Token 0 is REST, 1 is HOLD, and MIDI pitch p is NOTE token p+2. A NOTE starts a new note even when its pitch equals the preceding pitch. HOLD requires an active sounding pitch.

For fixed model probabilities q, the target is proportional to `q(Y) C(Y) exp(S(Y))`: the product of original probabilities at unknown positions, hard musical/context constraints, and nonpositive motion scores. Initially observed tokens are delta constraints. Probabilities are not renormalized after restricting pitches or conditioning an unknown token. Visible sounding-state anchors, including right-boundary HOLDs, remain protected.

Without motion dependence, a new NOTE forgets its preceding pitch. Each passage supplies scalar interval masses between consecutive onsets. Multiplying corresponding masses couples the passages; an onset-time dynamic program samples the common rhythm, then each passage samples a conditional pitch path. This avoids enumerating onset templates or Cartesian products of all passages' pitches. Interval preparation is O(R L² D); the onset DP is O(L² K).

With motion factors, the sampler retains compatible factors in its proposal and applies exact residual rejection. Drop removes all motion factors; Boundary retains compatible external and boundary terms; Visible also retains factors compatible with the rank-one transition. Early rejection can stop a losing proposal before all passages are sampled. Any rejection restarts the entire joint proposal, including the shared rhythm. Accepted samples follow the original target; precomputation and acceptance cost are reported separately. Acceptance depends on the target, so a total-runtime guarantee linear in R is not claimed.

| Component | Code |
|---|---|
| Input semantics and independent checker | `src/tri/domain/music.py` |
| Onset decomposition and base sampler | `src/tri/inference/onset_reset.py` |
| Retained rank-one factors | `src/tri/inference/rank_one_onset.py` |
| Drop and Boundary rejection | `src/tri/sampling/onset_rejection.py` |
| Visible and Early rejection | `src/tri/sampling/onset_rejection_adaptive.py` |
| Standard factorized multi-passage product chain | `src/tri/inference/product_multi.py` |
| Pair-prefix baseline, ordered VE, streamed templates | `src/tri/inference/{product_prefix,ordered_ve,template_stream}.py` |
| Neural model and training | `src/tri/models/` |
| Fixed-input experiment runner | `src/tri/evaluation/shared_rhythm_extension.py` |
| Raw-record aggregation | `src/tri/evaluation/shared_rhythm_extension_delivery.py` |

## Public API

```python
import numpy as np
from shared_onset import AdaptiveOnsetRejectionSampler, verify_music
from tri.evaluation.solver_cases import load_case

case = load_case("path/to/case.json")
engine = AdaptiveOnsetRejectionSampler(
    case.spec, case.logq, proposal="visible_early", max_proposals=10000
)
draw = engine.sample_full(np.random.default_rng(7))
assert verify_music(draw.tokens, case.spec).valid
print(draw.diagnostics)
```

Reuse the engine for independent draws of the same fixed target. `OnsetResetMusicInference` supplies base-target `log_partition()` and full-path sampling. Rejection samplers supply complete samples and diagnostics; their proposal normalizer is not the motion-weighted target normalizer. They do not expose normalized conditional queries for that target. Unsupported reset structures and hard adjacent-interval constraints raise explicit errors; resource exhaustion never returns a repaired or approximate path as an exact sample.

The complete research implementation also retains general finite-factor inference, older two-passage methods, TRI/SMC sampling, listening-pack tools, and their tests. Those modules support the development history and comparison workflows; the paper's main path is the one listed above. Standard solvers are implemented in this repository, not downloaded wrappers masquerading as equivalent timing baselines.

Correctness tests include independent original-token enumeration, partition comparisons, full-sample distribution checks, zero onsets, visible prefixes, HOLD/boundary cases, infeasibility, proposal-factor identities, and early-rejection behavior. Equivalent exact solvers target the same distribution; speedups alone do not establish superior musical quality.
