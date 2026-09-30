from collections import Counter
from itertools import product

import numpy as np
import pytest

from tri.errors import BudgetExceeded, InvalidSpecification, ZeroMass
from tri.inference.exact import Budget, ExactInference
from tri.inference.factors import FactorGraph, LogFactor
from tri.inference.rhythm import SharedOnsetCountInference


def generic_graph(log_q, flags, count, observed=None, allowed=None):
    """Same factorization for optimized generic VE, with narrow count states."""
    observed = observed or {}
    r_count, length, vocabulary = log_q.shape
    domains, factors = {}, []
    for j in range(length):
        domains[f"z{j}"] = (0, 1)
        for r in range(r_count):
            name = f"y{r}_{j}"
            domains[name] = tuple(range(vocabulary))
            weights = log_q[r, j].copy()
            if (r, j) in observed:
                weights[:] = -np.inf
                weights[observed[r, j]] = 0.0
            if allowed is not None:
                weights[~allowed[r, j]] = -np.inf
            factors.append(LogFactor((name,), weights))
            link = np.full((vocabulary, 2), -np.inf)
            for token, flag in enumerate(flags):
                link[token, int(flag)] = 0.0
            factors.append(LogFactor((name, f"z{j}"), link))
        domains[f"c{j}"] = tuple(range(max(0, count - (length - j - 1)), min(j + 1, count) + 1))
        previous_counts = (0,) if j == 0 else domains[f"c{j - 1}"]
        current_counts = domains[f"c{j}"]
        scope = (f"z{j}", f"c{j}") if j == 0 else (f"c{j - 1}", f"z{j}", f"c{j}")
        shape = tuple(len(domains[name]) for name in scope)
        values = np.full(shape, -np.inf)
        for i, previous in enumerate(previous_counts):
            for flag in (0, 1):
                for k, current in enumerate(current_counts):
                    if previous + flag == current:
                        values[(flag, k) if j == 0 else (i, flag, k)] = 0.0
        factors.append(LogFactor(scope, values))
    return FactorGraph(domains, tuple(factors))


def original_weights(log_q, flags, count, observed=None, allowed=None):
    observed = observed or {}
    replicas, length, vocabulary = log_q.shape
    weights = {}
    for values in product(range(vocabulary), repeat=replicas * length):
        if any(flags[values[r * length + j]] != flags[values[j]] for r in range(1, replicas) for j in range(length)):
            continue
        if sum(flags[values[j]] for j in range(length)) != count:
            continue
        score = 0.0
        for r in range(replicas):
            for j in range(length):
                token = values[r * length + j]
                if allowed is not None and not allowed[r, j, token]:
                    score = -np.inf
                if (r, j) in observed:
                    if token != observed[r, j]:
                        score = -np.inf
                else:
                    score += log_q[r, j, token]
        if np.isfinite(score):
            weights[values] = np.exp(score)
    return weights


def test_partition_clamp_mass_and_observed_delta():
    q = np.array([[[0.1, 0.6, 0.3]], [[0.2, 0.3, 0.5]]])
    flags = [False, True, True]
    solver = SharedOnsetCountInference(np.log(q), flags, 1)
    assert np.exp(solver.log_partition()) == pytest.approx(0.72)
    assert np.exp(solver.log_clamped_partition({(0, 0): 1, (1, 0): 2})) == pytest.approx(0.3)
    assert np.exp(solver.log_probability({(0, 0): 1})) == pytest.approx(2 / 3)
    allowed = np.ones_like(q, dtype=bool)
    allowed[0, 0, 1] = False
    cropped = SharedOnsetCountInference(np.log(q), flags, 1, allowed=allowed)
    assert np.exp(cropped.log_partition()) == pytest.approx(0.24)
    log_q = np.log(q)
    log_q[0, 0] = np.nan
    fixed = SharedOnsetCountInference(log_q, flags, 1, observed={(0, 0): 2})
    assert np.exp(fixed.log_partition()) == pytest.approx(0.8)
    assert fixed.log_clamped_partition({(0, 0): 1}) == -np.inf
    assert fixed.log_probability({(0, 0): 2}) == 0.0


def test_partitions_and_clamps_match_ve_and_original_enumeration():
    rng = np.random.default_rng(72)
    flags = np.array([False, True, True])
    for count in range(4):
        for fixed in (False, True):
            log_q = np.log(rng.dirichlet(np.ones(3), size=(2, 3)))
            allowed = rng.random((2, 3, 3)) > 0.2
            observed = {(0, 1): int(rng.integers(3))} if fixed else {}
            dp = SharedOnsetCountInference(log_q, flags, count, observed=observed, allowed=allowed)
            ve = ExactInference(generic_graph(log_q, flags, count, observed, allowed))
            weights = original_weights(log_q, flags, count, observed, allowed)
            expected = sum(weights.values())
            assert np.exp(dp.log_partition()) == pytest.approx(expected, abs=1e-15, rel=1e-11)
            assert np.exp(ve.log_partition()) == pytest.approx(expected, abs=1e-15, rel=1e-11)
            for first, second in product(range(3), repeat=2):
                assignment = {(0, 0): first, (1, 2): second}
                target = sum(w for y, w in weights.items() if y[0] == first and y[5] == second)
                assert np.exp(dp.log_clamped_partition(assignment)) == pytest.approx(target, abs=1e-15)
                assert np.exp(ve.log_clamped_partition({"y0_0": first, "y1_2": second})) == pytest.approx(target, abs=1e-15)
                if expected:
                    assert np.exp(dp.log_probability(assignment)) == pytest.approx(target / expected, abs=1e-12)


def test_joint_sampling_and_partial_batch_probability_not_latent_probability():
    log_q = np.full((2, 2, 3), -np.log(3))
    flags = [False, True, True]
    solver = SharedOnsetCountInference(log_q, flags, 1)
    rng = np.random.default_rng(491)
    frequencies = Counter()
    for _ in range(1600):
        sample = solver.sample_batch([(0, 0), (1, 0)], rng)
        pair = (sample.assignment[0, 0], sample.assignment[1, 0])
        frequencies[pair] += 1
        assert flags[pair[0]] == flags[pair[1]]
        expected = 0.5 if pair == (0, 0) else 0.125
        assert np.exp(sample.log_probability) == pytest.approx(expected)
        assert sample.log_clamped_partition == pytest.approx(solver.log_clamped_partition(sample.assignment))
    assert frequencies[0, 0] / 1600 == pytest.approx(0.5, abs=0.04)
    assert len(frequencies) == 5
    for _ in range(30):
        sample = solver.sample_batch([(0, 0), (0, 1), (1, 0), (1, 1)], rng)
        assert sum(flags[sample.assignment[0, j]] for j in range(2)) == 1
        assert all(flags[sample.assignment[0, j]] == flags[sample.assignment[1, j]] for j in range(2))
        assert np.exp(sample.log_probability) == pytest.approx(0.125)
    empty = solver.sample_batch([], rng)
    assert empty.assignment == {}
    assert empty.log_probability == 0.0


@pytest.mark.parametrize("flags,count,expected", [
    ([False, False], 0, 1.0), ([False, False], 1, 0.0),
    ([True, True], 3, 1.0), ([True, True], 0, 0.0),
    ([False, True], 0, 0.5**6), ([False, True], 3, 0.5**6),
])
def test_zero_all_onset_counts_and_single_feature_vocab(flags, count, expected):
    solver = SharedOnsetCountInference(np.full((2, 3, 2), -np.log(2)), flags, count)
    assert np.exp(solver.log_partition()) == pytest.approx(expected)
    if expected:
        sample = solver.sample_batch([(r, j) for r in range(2) for j in range(3)], np.random.default_rng(4))
        assert sum(flags[sample.assignment[0, j]] for j in range(3)) == count
    else:
        with pytest.raises(ZeroMass):
            solver.sample_batch([], np.random.default_rng(4))
        with pytest.raises(ZeroMass):
            solver.log_probability({})


def test_conflicting_observations_allowed_masks_and_model_zero_support():
    q = np.full((2, 1, 2), -np.log(2))
    conflict = SharedOnsetCountInference(q, [False, True], 1, observed={(0, 0): 0, (1, 0): 1})
    assert conflict.log_partition() == -np.inf
    masked = SharedOnsetCountInference(q, [False, True], 1, observed={(0, 0): 1}, allowed=np.zeros((2, 1, 2), dtype=bool))
    assert masked.log_partition() == -np.inf
    zero = SharedOnsetCountInference(np.array([[[0.0, -np.inf]]]), [False, True], 1)
    assert zero.log_partition() == -np.inf
    with pytest.raises(ZeroMass):
        zero.sample_batch([(0, 0)], np.random.default_rng(5))


def test_input_arrays_and_observations_are_snapshots():
    log_q = np.full((2, 2, 2), -np.log(2))
    flags = np.array([False, True])
    allowed = np.ones((2, 2, 2), dtype=bool)
    observed = {(0, 0): 1}
    solver = SharedOnsetCountInference(log_q, flags, 1, observed=observed, allowed=allowed)
    before = solver.log_partition()
    log_q[:] = np.nan
    flags[:] = False
    allowed[:] = False
    observed[0, 0] = 0
    assert solver.log_partition() == before
    assert solver.observed[0, 0] == 1
    assert tuple(solver.onset_flags) == (False, True)
    with pytest.raises(ValueError):
        solver.onset_flags.setflags(write=True)


@pytest.mark.parametrize("kwargs", [
    {"count": True}, {"count": -1}, {"count": 3},
    {"onset_flags": [0, 1]}, {"onset_flags": [False]},
    {"observed": {(0, 2): 0}}, {"observed": {(0, 1): 2}},
    {"observed": {(False, 0): 1}}, {"observed": {(0, 0): True}},
    {"allowed": np.ones((2, 2, 2), dtype=int)},
    {"log_probs": np.zeros((2, 2, 2))},
    {"log_probs": np.full((2, 2, 2), np.nan)},
    {"log_probs": np.zeros((2, 2, 2), dtype=complex)},
    {"log_probs": np.zeros((0, 2, 2))},
])
def test_invalid_inputs_are_explicit(kwargs):
    base = {"log_probs": np.full((2, 2, 2), -np.log(2)), "onset_flags": [False, True], "count": 1}
    with pytest.raises(InvalidSpecification):
        SharedOnsetCountInference(**{**base, **kwargs})


def test_invalid_queries_and_budgets():
    solver = SharedOnsetCountInference(np.full((2, 2, 2), -np.log(2)), [False, True], 1)
    for assignment in ({(2, 0): 0}, {(0, 0): 2}, {(0, 0): True}, {0: 1}, []):
        with pytest.raises(InvalidSpecification):
            solver.log_clamped_partition(assignment)
    with pytest.raises(InvalidSpecification):
        solver.sample_batch([(0, 0), (0, 0)], np.random.default_rng(5))
    with pytest.raises(InvalidSpecification):
        solver.sample_batch([(0, 0)], 5)
    with pytest.raises(BudgetExceeded):
        SharedOnsetCountInference(np.full((2, 10, 2), -np.log(2)), [False, True], 5, budget=Budget(max_factor_entries=10))
    with pytest.raises(BudgetExceeded):
        SharedOnsetCountInference(np.full((2, 10, 2), -np.log(2)), [False, True], 5, budget=Budget(max_workspace_bytes=100))
