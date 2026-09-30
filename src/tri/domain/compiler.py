"""Compile the bounded music request to probability-preserving log factors.

Every legal original sequence determines exactly one sounding-state path and
one path for each counter. Summing auxiliaries therefore yields q C exp(S),
without a multiplicity factor. Original observations are delta factors;
temporary inference evidence must instead retain these original q factors.
"""

from __future__ import annotations

from math import prod

import numpy as np
from scipy.special import logsumexp

from tri.errors import InvalidSpecification
from tri.inference.exact import Budget
from tri.inference.factors import FactorGraph, LogFactor

from .music import HOLD, REST, SILENCE, VOCAB_SIZE, MusicSpec


def compile_music(spec: MusicSpec, log_probs: np.ndarray, budget: Budget | None = None) -> FactorGraph:
    """Compile normalized full-vocabulary probabilities, retaining cropped mass.

    Only unknown rows must be normalized. Observed rows are ignored entirely,
    including their numeric values. The generated vocabulary is REST, HOLD and
    NOTE(p) for p in spec.pitches. Boundary pitches can be outside this vocabulary:
    they can persist through HOLD but cannot be newly generated as NOTE.
    fixed_soundings anchors the original audio at observed slots: fixing a HOLD
    token alone cannot preserve its pitch when the preceding gap is regenerated.

    A contradictory request compiles to zero mass. Invalid specifications and
    budget overruns raise explicit errors. Zero model support is not labelled
    logical infeasibility by this compiler.
    """
    if not isinstance(spec, MusicSpec):
        raise InvalidSpecification("compile_music requires a MusicSpec")
    budget = budget or Budget()
    if np.iscomplexobj(log_probs):
        raise InvalidSpecification("log_probs must contain real log probabilities")
    try:
        probabilities = np.asarray(log_probs, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidSpecification("log_probs must be a numeric array") from exc
    if probabilities.shape != (spec.length, VOCAB_SIZE):
        raise InvalidSpecification(f"log_probs shape must be ({spec.length}, {VOCAB_SIZE})")
    for pos in range(spec.length):
        if pos in spec.observed:
            continue
        row = probabilities[pos]
        if np.isnan(row).any() or np.isposinf(row).any():
            raise InvalidSpecification(f"slot {pos}: only finite or -inf log probabilities are accepted")
        total = float(logsumexp(row))
        if not np.isfinite(total) or abs(total) > 2e-6:
            raise InvalidSpecification(f"slot {pos}: full 130-category probabilities must sum to one")

    domains: dict[str, tuple[int, ...]] = {}
    factors: list[LogFactor] = []
    stored_bytes = 0

    def allocate(scope: tuple[str, ...], context: str) -> np.ndarray:
        nonlocal stored_bytes
        shape = tuple(len(domains[name]) for name in scope)
        entries = prod(shape)
        budget.check_factor_entries(entries, context=context)
        # Reserve room for the new tensor and one defensive copy by LogFactor.
        budget.check_workspace_bytes(stored_bytes + entries * 16, context=context)
        stored_bytes += entries * 8
        return np.full(shape, -np.inf, dtype=np.float64)

    vocabulary = (REST, HOLD) + tuple(p + 2 for p in spec.pitches)
    previous_states = (SILENCE if spec.initial_pitch is None else spec.initial_pitch,)
    for pos in range(spec.length):
        y_name, h_name = f"y{pos}", f"h{pos}"
        tokens = (spec.observed[pos],) if pos in spec.observed else vocabulary
        domains[y_name] = tokens
        unary = allocate((y_name,), f"token weights at {pos}")
        unary[:] = 0.0 if pos in spec.observed else probabilities[pos, list(tokens)]
        factors.append(LogFactor((y_name,), unary))

        # Forward reachability reduces all-observed stretches to singletons.
        reachable: set[int] = set()
        for token in tokens:
            if token == REST:
                reachable.add(SILENCE)
            elif token == HOLD:
                reachable.update(p for p in previous_states if p != SILENCE)
            else:
                reachable.add(token - 2)
        if pos in spec.fixed_soundings:
            anchor = spec.fixed_soundings[pos]
            reachable.add(SILENCE if anchor is None else anchor)
        # An impossible HOLD still needs a nonempty domain and a zero factor.
        current_states = tuple(sorted(reachable)) or (SILENCE,)
        domains[h_name] = current_states
        state_index = {pitch: index for index, pitch in enumerate(current_states)}
        scope = (y_name, h_name) if pos == 0 else (f"h{pos - 1}", y_name, h_name)
        transition = allocate(scope, f"sounding-state transition at {pos}")
        for previous_index, previous in enumerate(previous_states):
            for token_index, token in enumerate(tokens):
                if token == REST:
                    current, score = SILENCE, 0.0
                elif token == HOLD:
                    if previous == SILENCE:
                        continue
                    current, score = previous, 0.0
                else:
                    current = token - 2
                    jump = 0 if previous == SILENCE else abs(current - previous)
                    if spec.max_adjacent_interval is not None and jump > spec.max_adjacent_interval:
                        continue
                    score = -spec.motion_cost * jump
                current_index = state_index[current]
                index = (token_index, current_index) if pos == 0 else (previous_index, token_index, current_index)
                transition[index] = score
        factors.append(LogFactor(scope, transition))
        if pos in spec.pitch_ranges or pos in spec.pitch_classes or pos in spec.fixed_soundings or (spec.enforce_end and pos == spec.length - 1):
            restrictions = allocate((h_name,), f"sounding-state restrictions at {pos}")
            for index, pitch in enumerate(current_states):
                allowed = True
                if pitch != SILENCE:
                    if pos in spec.pitch_ranges:
                        low, high = spec.pitch_ranges[pos]
                        allowed &= low <= pitch <= high
                    if pos in spec.pitch_classes:
                        allowed &= pitch % 12 in spec.pitch_classes[pos]
                if spec.enforce_end and pos == spec.length - 1:
                    end = SILENCE if spec.end_pitch is None else spec.end_pitch
                    allowed &= pitch == end
                if pos in spec.fixed_soundings:
                    anchor = spec.fixed_soundings[pos]
                    allowed &= pitch == (SILENCE if anchor is None else anchor)
                restrictions[index] = 0.0 if allowed else -np.inf
            factors.append(LogFactor((h_name,), restrictions))
        previous_states = current_states

    for left, right in sorted(set(tuple(sorted(pair)) for pair in spec.equal_onsets)):
        if left == right:
            continue
        scope = (f"y{left}", f"y{right}")
        relation = allocate(scope, f"onset equality ({left}, {right})")
        for i, left_token in enumerate(domains[scope[0]]):
            for j, right_token in enumerate(domains[scope[1]]):
                if (left_token >= 2) == (right_token >= 2):
                    relation[i, j] = 0.0
        factors.append(LogFactor(scope, relation))

    for rule_index, rule in enumerate(spec.onset_counts):
        previous_counts = (0,)
        length = len(rule.positions)
        for step, pos in enumerate(rule.positions):
            y_name, c_name = f"y{pos}", f"c{rule_index}_{step}"
            # Prefix counts must leave enough positions to reach the target.
            low = max(0, rule.count - (length - step - 1))
            high = min(step + 1, rule.count)
            counts = tuple(range(low, high + 1))
            domains[c_name] = counts
            count_index = {value: index for index, value in enumerate(counts)}
            scope = (y_name, c_name) if step == 0 else (f"c{rule_index}_{step - 1}", y_name, c_name)
            transition = allocate(scope, f"onset counter {rule_index} at {step}")
            for previous_index, previous in enumerate(previous_counts):
                for token_index, token in enumerate(domains[y_name]):
                    current = previous + (token >= 2)
                    if current in count_index:
                        index = (token_index, count_index[current]) if step == 0 else (previous_index, token_index, count_index[current])
                        transition[index] = 0.0
            factors.append(LogFactor(scope, transition))
            previous_counts = counts

    return FactorGraph(domains=domains, factors=tuple(factors))
