"""Four decoding controls sharing one supplied neural probability provider.

The raw reference deliberately returns invalid music for independent scoring.
The other methods enforce only the constraints stated in their descriptions.
No method here implements particle resampling or establishes music quality.
"""

from dataclasses import dataclass, replace
import math
from types import MappingProxyType

import numpy as np
from scipy.special import logsumexp

from tri.domain.compiler import compile_music
from tri.domain.music import MusicSpec, VOCAB_SIZE, verify_music
from tri.errors import InvalidSpecification, VerificationError
from tri.inference.exact import Budget, ExactInference
from tri.sampling.direct import ProbabilityProvider, direct_decode
from tri.sampling.schedules import RevealSchedule


METHODS = ("raw_reference", "local_constraints", "one_shot_joint", "tri_direct")
METHOD_DESCRIPTIONS = MappingProxyType({
    "raw_reference": "Full-vocabulary independent neural reveals under the reference schedule; invalid music is returned unchanged.",
    "local_constraints": "Multi-round joint local constraints, including HOLD/anchors/ranges/motion; onset equality and onset count rules are removed.",
    "one_shot_joint": "One joint sample from the initial q_0 C exp(S); this is not the multi-round reference terminal conditional.",
    "tri_direct": "Multi-round exact joint reveals with all request constraints; no terminal-exact or music-quality claim.",
})


@dataclass(frozen=True)
class MethodResult:
    tokens: tuple[int, ...]
    model_calls: int
    trace: tuple[dict, ...] = ()


def _model_snapshot(provider: ProbabilityProvider, state, noise) -> np.ndarray:
    """Validate all 130 categories before sampling; observed rows are ignored."""
    raw = provider(state, noise)
    if np.iscomplexobj(raw):
        raise InvalidSpecification("Model log probabilities must be real")
    try:
        log_q = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise InvalidSpecification("Model log probabilities must be numeric") from exc
    if log_q.shape != (len(state), VOCAB_SIZE):
        raise InvalidSpecification(f"Model log probabilities must have shape ({len(state)}, {VOCAB_SIZE})")
    for position, token in enumerate(state):
        if token is not None:
            continue
        row = log_q[position]
        if np.isnan(row).any() or np.isposinf(row).any():
            raise InvalidSpecification(f"Unknown slot {position} has invalid log probabilities")
        total = float(logsumexp(row))
        if not math.isfinite(total) or abs(total) > 2e-6:
            raise InvalidSpecification(f"Unknown slot {position} needs normalized full-vocabulary probabilities")
    return log_q


def _checked(tokens, spec):
    checked = verify_music(tokens, spec)
    if not checked.valid:
        raise VerificationError(str(checked.violations))


def _raw_reference(spec, provider, schedule, seed):
    rng = np.random.default_rng(seed)
    state = tuple(spec.observed.get(i) for i in range(spec.length))
    trace = []
    model_calls = 0
    for step in range(schedule.steps):
        missing = tuple(i for i, token in enumerate(state) if token is None)
        if not missing:
            break
        noise = 1.0 - step / schedule.steps
        log_q = _model_snapshot(provider, state, noise)
        model_calls += 1
        reveal = schedule.sample(step, missing, rng)
        values, log_probability = [], 0.0
        for position in reveal.positions:
            # Every output category participates. This is only normalization
            # at floating-point precision, never a pitch crop or music repair.
            normalized = log_q[position] - logsumexp(log_q[position])
            probabilities = np.exp(normalized)
            probabilities /= probabilities.sum()
            token = int(rng.choice(VOCAB_SIZE, p=probabilities))
            values.append(token)
            log_probability += float(math.log(probabilities[token]))
        updated = list(state)
        for position, token in zip(reveal.positions, values):
            updated[position] = token
        state = tuple(updated)
        trace.append({
            "step": step, "noise": noise, "positions": reveal.positions,
            "values": tuple(values), "log_batch_probability": log_probability,
            "log_rho_reference": reveal.log_reference,
            "log_rho_proposal": reveal.log_proposal,
        })
    if any(token is None for token in state):
        raise VerificationError("Reference schedule failed to finish the request")
    # Do not verify or repair here: invalid outputs are baseline observations.
    return MethodResult(tuple(int(token) for token in state), model_calls, tuple(trace))


def _one_shot_joint(spec, provider, seed, budget):
    state = tuple(spec.observed.get(i) for i in range(spec.length))
    missing = tuple(i for i, token in enumerate(state) if token is None)
    if not missing:
        tokens = tuple(int(token) for token in state)
        _checked(tokens, spec)
        return MethodResult(tokens, 0)
    log_q = _model_snapshot(provider, state, 1.0)
    engine = ExactInference(compile_music(spec, log_q, budget=budget), budget=budget)
    log_z = engine.log_partition()
    sample = engine.sample_batch(tuple(f"y{i}" for i in missing), np.random.default_rng(seed))
    values = tuple(int(sample.assignment[f"y{i}"]) for i in missing)
    updated = list(state)
    for position, value in zip(missing, values):
        updated[position] = value
    tokens = tuple(int(token) for token in updated)
    _checked(tokens, spec)
    trace = ({
        "step": 0, "noise": 1.0, "positions": missing, "values": values,
        "log_z": log_z, "log_z_clamped": sample.log_clamped_partition,
        "log_batch_probability": sample.log_probability,
        "log_q_batch": float(sum(log_q[i, value] for i, value in zip(missing, values))),
    },)
    return MethodResult(tokens, 1, trace)


def decode_method(method: str, spec: MusicSpec, provider: ProbabilityProvider, *,
                  steps: int = 4, seed: int = 20260912,
                  budget: Budget | None = None) -> MethodResult:
    """Decode with the supplied provider; never train/load a separate model.

    The provider must retain the original editable-role mask across calls.
    Exceptions, including zero model mass and budget failure, propagate to the
    independent evaluator. One-shot uses one query regardless of ``steps``.
    """
    if method not in METHODS:
        raise InvalidSpecification(f"Unknown decoding method {method!r}; choose one of {METHODS}")
    if not isinstance(spec, MusicSpec) or not callable(provider):
        raise InvalidSpecification("Expected MusicSpec and a callable probability provider")
    schedule = RevealSchedule(steps, epsilon=1.0)
    if method == "raw_reference":
        return _raw_reference(spec, provider, schedule, seed)
    if method == "one_shot_joint":
        return _one_shot_joint(spec, provider, seed, budget)
    active_spec = replace(spec, equal_onsets=(), onset_counts=()) if method == "local_constraints" else spec
    result = direct_decode(active_spec, provider, steps=steps, seed=seed,
                           epsilon=1.0, track_path_weights=False, budget=budget)
    return MethodResult(result.tokens, result.model_calls, result.trace)
