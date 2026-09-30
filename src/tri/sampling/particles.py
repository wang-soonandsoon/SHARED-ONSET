"""Finite-particle IS and SMC for a fixed reference reveal process.

The incremental potential uses OLD-q clamps and NEW-q future partitions.
Weights include the initial partition. Finite particle outputs are weighted
approximations, not exact independent samples from the reference conditional.
"""
from dataclasses import dataclass, replace
import math
from numbers import Integral

import numpy as np
from scipy.special import logsumexp

from tri.domain.music import MusicSpec, verify_music
from tri.domain.compiler import compile_music
from tri.errors import InvalidSpecification, VerificationError, ZeroMass
from tri.inference.exact import Budget, ExactInference
from tri.sampling.direct import direct_decode
from tri.sampling.schedules import RevealSchedule
from tri.sampling.weights import log_path_increment


@dataclass(frozen=True)
class ParticleResult:
    tokens: tuple[int, ...]
    model_calls: int
    trace: tuple[dict, ...]
    particles: tuple[tuple[int, ...] | None, ...]
    normalized_weights: tuple[float, ...]
    log_normalizer_estimate: float
    ess: float
    unique_ancestors: int


def _validate_count(particles):
    if isinstance(particles, bool) or not isinstance(particles, Integral) or particles < 1:
        raise InvalidSpecification("particles must be a positive integer")


def _normalize(log_weights):
    normalizer = float(logsumexp(log_weights))
    if not math.isfinite(normalizer):
        raise ZeroMass("all particles have zero weight; no restart or uniform fallback")
    normalized = np.asarray(log_weights, dtype=np.float64) - normalizer
    return normalized, normalizer


def systematic_resample(weights, rng):
    """Unbiased offspring counts for normalized nonnegative weights."""
    probabilities = np.asarray(weights, dtype=np.float64)
    if probabilities.ndim != 1 or not len(probabilities) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
        raise InvalidSpecification("resampling needs normalized finite weights")
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    points = (rng.random() + np.arange(len(probabilities))) / len(probabilities)
    return np.searchsorted(cumulative, points, side="right")


def path_is(spec: MusicSpec, provider, *, particles=4, steps=4, seed=0,
            epsilon=1.0, backend="ve", budget: Budget | None = None):
    """Independent complete proposals, then a weighted terminal draw."""
    _validate_count(particles)
    rng = np.random.default_rng(seed)
    outputs, weights, trace = [], [], []
    calls = 0

    def counted(state, noise):
        nonlocal calls
        calls += 1
        return provider(state, noise)

    for index in range(particles):
        child_seed = int(rng.integers(0, 2**63 - 1))
        try:
            result = direct_decode(spec, counted, steps=steps, seed=child_seed,
                                   epsilon=epsilon, backend=backend, budget=budget,
                                   track_path_weights=True)
            outputs.append(result.tokens)
            weights.append(result.log_path_weight)
            trace.append({"particle": index, "seed": child_seed, "log_weight": result.log_path_weight,
                          "path": result.trace})
        except ZeroMass as error:
            outputs.append(None)
            weights.append(-math.inf)
            trace.append({"particle": index, "seed": child_seed, "zero_mass": str(error)})
    log_weights, log_total = _normalize(weights)
    probabilities = np.exp(log_weights)
    probabilities /= probabilities.sum()
    chosen = int(rng.choice(particles, p=probabilities))
    ess = float(1 / np.square(probabilities).sum())
    return ParticleResult(outputs[chosen], calls, tuple(trace), tuple(outputs), tuple(probabilities),
                          log_total - math.log(particles), ess, sum(v is not None for v in outputs))


def smc_decode(spec: MusicSpec, provider, *, particles=4, steps=4, seed=0,
               epsilon=1.0, ess_threshold=0.5, backend="ve", budget: Budget | None = None):
    """Sequential importance/resampling with exact inner music inference.

    The model provider must be a deterministic function of partial state/time
    and frozen external context. Identical initial queries are shared. The
    reference schedule is fixed; epsilon changes only the proposal schedule.
    """
    _validate_count(particles)
    if not math.isfinite(ess_threshold) or not 0 <= ess_threshold <= 1:
        raise InvalidSpecification("ESS threshold must lie in [0,1]")
    schedule = RevealSchedule(steps, epsilon)
    rng = np.random.default_rng(seed)
    initial = tuple(spec.observed.get(i) for i in range(spec.length))
    calls = 0

    def prepare(state, step):
        nonlocal calls
        calls += 1
        log_q = np.array(provider(state, 1 - step / steps), dtype=np.float64, copy=True)
        conditioned = replace(spec, observed={i: v for i, v in enumerate(state) if v is not None})
        if backend == "ve":
            engine = ExactInference(compile_music(conditioned, log_q, budget), budget=budget)
        else:
            from tri.inference.music_backends import make_music_engine
            engine = make_music_engine(conditioned, log_q, backend=backend, budget=budget)
        return log_q, engine, engine.log_partition()

    if all(v is not None for v in initial):
        checked = verify_music(initial, spec)
        if not checked.valid:
            raise VerificationError(str(checked.violations))
        return ParticleResult(initial, 0, (), (initial,) * particles, (1 / particles,) * particles,
                              checked.soft_score, float(particles), particles)
    first = prepare(initial, 0)
    if not math.isfinite(first[2]):
        raise ZeroMass("initial proposal has zero mass")
    states = [initial] * particles
    prepared = [first] * particles
    log_weights = np.full(particles, -math.log(particles))
    log_normalizer = first[2]
    ancestors = np.arange(particles)
    trace = []
    for step in range(steps):
        increments = np.zeros(particles)
        edges = []
        for index, state in enumerate(states):
            if not np.isfinite(log_weights[index]):
                increments[index] = -math.inf
                edges.append({"particle": index, "dead": True})
                continue
            missing = tuple(i for i, value in enumerate(state) if value is None)
            if not missing:
                edges.append({"particle": index, "absorbing": True, "log_g": 0.0})
                continue
            log_q, engine, log_z = prepared[index]
            reveal = schedule.sample(step, missing, rng)
            batch = engine.sample_batch(tuple(f"y{i}" for i in reveal.positions), rng)
            values = tuple(int(batch.assignment[f"y{i}"]) for i in reveal.positions)
            updated = list(state)
            for i, value in zip(reveal.positions, values):
                updated[i] = value
            updated = tuple(updated)
            log_q_batch = float(sum(log_q[i, value] for i, value in zip(reveal.positions, values)))
            if all(v is not None for v in updated):
                checked = verify_music(updated, spec)
                if not checked.valid:
                    raise VerificationError(str(checked.violations))
                log_next = checked.soft_score
                prepared[index] = None
            else:
                prepared[index] = prepare(updated, step + 1)
                log_next = prepared[index][2]
            log_g = log_path_increment(log_rho_reference=reveal.log_reference,
                                       log_rho_proposal=reveal.log_proposal,
                                       log_q_batch=log_q_batch, log_z_next=log_next,
                                       log_z_clamped=batch.log_clamped_partition)
            increments[index] = log_g
            states[index] = updated
            edges.append({"particle": index, "positions": reveal.positions, "values": values,
                          "log_z": log_z, "log_z_clamped": batch.log_clamped_partition,
                          "log_q_batch": log_q_batch, "log_z_next": log_next,
                          "log_rho_reference": reveal.log_reference,
                          "log_rho_proposal": reveal.log_proposal, "log_g": log_g})
        log_weights, log_increment = _normalize(log_weights + increments)
        log_normalizer += log_increment
        probabilities = np.exp(log_weights)
        ess = float(1 / np.square(probabilities).sum())
        resampled = step < steps - 1 and ess < ess_threshold * particles
        selected = None
        if resampled:
            selected = systematic_resample(probabilities, rng)
            states = [states[i] for i in selected]
            prepared = [prepared[i] for i in selected]
            ancestors = ancestors[selected]
            log_weights = np.full(particles, -math.log(particles))
        trace.append({"step": step, "ess_before_resample": ess, "resampled": resampled,
                      "parent_indices": None if selected is None else selected.tolist(),
                      "unique_ancestors": len(set(ancestors.tolist())),
                      "log_normalizer_estimate": log_normalizer, "edges": edges})
        if all(all(v is not None for v in state) for state, weight in zip(states, log_weights) if np.isfinite(weight)):
            break
    outputs = []
    for state, weight in zip(states, log_weights):
        if np.isfinite(weight):
            if any(v is None for v in state) or not verify_music(state, spec).valid:
                raise VerificationError("positive-weight particle is incomplete or invalid")
            outputs.append(tuple(int(v) for v in state))
        else:
            outputs.append(None)
    probabilities = np.exp(log_weights)
    probabilities /= probabilities.sum()
    chosen = int(rng.choice(particles, p=probabilities))
    return ParticleResult(outputs[chosen], calls, tuple(trace), tuple(outputs), tuple(probabilities),
                          float(log_normalizer), float(1 / np.square(probabilities).sum()),
                          len(set(ancestors[np.isfinite(log_weights)].tolist())))
