"""Music decoding using exact joint reveal proposals.

Path weights are optional diagnostics. This module does not resample particles
and does not claim that one decoded trajectory is a terminal-exact sample.
"""
from collections.abc import Callable
from dataclasses import dataclass, replace
import math
import time

import numpy as np

from tri.domain.music import MusicSpec, verify_music
from tri.domain.compiler import compile_music
from tri.errors import VerificationError, ZeroMass
from tri.inference.exact import Budget, ExactInference
from tri.sampling.schedules import RevealSchedule
from tri.sampling.weights import log_path_increment

ProbabilityProvider = Callable[[tuple[int | None, ...], float], np.ndarray]


@dataclass(frozen=True)
class DecodeResult:
    tokens: tuple[int, ...]
    trace: tuple[dict, ...]
    log_path_weight: float | None
    log_relative_path_weight: float | None
    initial_log_partition: float
    soft_score: float
    elapsed_seconds: float
    model_calls: int


def direct_decode(spec: MusicSpec, probability_provider: ProbabilityProvider, *,
                  steps: int = 4, seed: int = 0, epsilon: float = 1.0,
                  track_path_weights: bool = False, budget: Budget | None = None,
                  backend: str = "ve") -> DecodeResult:
    schedule = RevealSchedule(steps, epsilon)
    rng = np.random.default_rng(seed)
    state = tuple(spec.observed.get(i) for i in range(spec.length))
    initial_spec = spec
    trace: list[dict] = []
    started = time.perf_counter()
    model_calls = 0
    cumulative_weight = 0.0
    initial_log_partition = None

    def prepare(tokens: tuple[int | None, ...], step: int):
        nonlocal model_calls
        # Calling on empty reveals is intentional: noise/time still changes.
        log_q = np.array(probability_provider(tokens, 1.0 - step / steps), dtype=np.float64, copy=True)
        model_calls += 1
        conditioned = replace(initial_spec, observed={i: v for i, v in enumerate(tokens) if v is not None})
        if backend == "ve":
            graph = compile_music(conditioned, log_q, budget=budget)
            engine = ExactInference(graph, budget=budget)
        else:
            from tri.inference.music_backends import make_music_engine
            engine = make_music_engine(conditioned, log_q, backend=backend, budget=budget)
        log_z = engine.log_partition()
        if not math.isfinite(log_z):
            raise ZeroMass("current music proposal has zero model mass; no UNSAT claim is made")
        return log_q, engine, log_z

    prepared = None
    for step in range(steps):
        missing = tuple(i for i, token in enumerate(state) if token is None)
        if not missing:
            # Remaining reference/proposal transitions are absorbing, G=1.
            break
        log_q, engine, log_z = prepared if prepared is not None else prepare(state, step)
        if initial_log_partition is None:
            initial_log_partition = log_z
        prepared = None
        # Deterministic first-missing guide is based only on current state.
        # epsilon=1 is the default and uses the reference schedule directly.
        reveal = schedule.sample(step, missing, rng)
        variables = tuple(f"y{i}" for i in reveal.positions)
        batch = engine.sample_batch(variables, rng)
        new_state = list(state)
        values = tuple(int(batch.assignment[f"y{i}"]) for i in reveal.positions)
        for i, value in zip(reveal.positions, values):
            new_state[i] = value
        state = tuple(new_state)
        log_q_batch = float(sum(log_q[i, v] for i, v in zip(reveal.positions, values)))
        row = {
            "step": step, "noise": 1.0 - step / steps,
            "positions": reveal.positions, "values": values,
            "log_z": log_z, "log_z_clamped": batch.log_clamped_partition,
            "log_batch_probability": batch.log_probability,
            "log_q_batch": log_q_batch,
            "log_rho_reference": reveal.log_reference,
            "log_rho_proposal": reveal.log_proposal,
        }
        if not math.isclose(batch.log_probability, batch.log_clamped_partition - log_z, abs_tol=1e-9, rel_tol=1e-9):
            raise VerificationError("joint batch probability disagrees with clamped partition")
        if track_path_weights:
            if all(v is not None for v in state):
                checked = verify_music(state, initial_spec)
                if not checked.valid:
                    raise VerificationError(str(checked.violations))
                log_z_next = checked.soft_score
            else:
                prepared = prepare(state, step + 1)
                log_z_next = prepared[2]
            log_g = log_path_increment(
                log_rho_reference=reveal.log_reference, log_rho_proposal=reveal.log_proposal,
                log_q_batch=log_q_batch, log_z_next=log_z_next,
                log_z_clamped=batch.log_clamped_partition,
            )
            cumulative_weight += log_g
            row.update(log_z_next=log_z_next, log_g=log_g)
        trace.append(row)
    if any(v is None for v in state):
        raise VerificationError("final reveal failed to complete the request")
    final = tuple(int(v) for v in state)
    checked = verify_music(final, initial_spec)
    if not checked.valid:
        raise VerificationError(str(checked.violations))
    if initial_log_partition is None:
        initial_log_partition = checked.soft_score
    return DecodeResult(final, tuple(trace),
                        initial_log_partition + cumulative_weight if track_path_weights else None,
                        cumulative_weight if track_path_weights else None, initial_log_partition,
                        checked.soft_score, time.perf_counter() - started, model_calls)
