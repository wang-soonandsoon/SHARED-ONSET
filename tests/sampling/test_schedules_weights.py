from fractions import Fraction
from itertools import combinations, product
import math

import numpy as np
import pytest

from examples import reference_oracle as old
from tri.errors import InvalidSpecification
from tri.sampling.schedules import RevealSchedule
from tri.sampling.weights import log_path_increment


def subsets(items):
    return [a for n in range(len(items) + 1) for a in combinations(items, n)]


def test_full_mixture_normalizes_and_has_reference_support():
    s = RevealSchedule(3, epsilon=0.25)
    for step in range(3):
        for missing in [(), (1,), (0, 2, 4)]:
            rows = [s.log_probs(step, missing, a) for a in subsets(missing)]
            assert sum(math.exp(r) for r, _ in rows) == pytest.approx(1)
            assert sum(math.exp(p) for _, p in rows) == pytest.approx(1)
            assert all(p > -math.inf for r, p in rows if r > -math.inf)
    r, p = s.log_probs(0, (0, 1, 2), (0,))
    assert math.exp(p) == pytest.approx(0.75 + 0.25 * math.exp(r))


def test_end_absorption_empty_batches_and_no_deterministic_strict_guide():
    s = RevealSchedule(3)
    assert s.log_probs(0, (), ()) == (0, 0)
    assert s.log_probs(2, (1, 4), ()) == (-math.inf, -math.inf)
    assert s.sample(2, (1, 4), np.random.default_rng(1)).positions == (1, 4)
    assert math.isfinite(s.log_probs(0, (1, 4), ())[0])
    with pytest.raises(InvalidSpecification):
        RevealSchedule(3, epsilon=0)
    with pytest.raises(InvalidSpecification):
        s.sample(0, (1, 4), np.random.default_rng(1), guide=(7,))


def test_float_path_weights_against_independent_fraction_oracle():
    # All reachable partial assignments, times and reveal batches, including
    # early completion/empty reveals. Independent existing Fraction formulas.
    for x in product((None, 0, 1), repeat=3):
        if old.partition(0, x) == 0:
            continue
        for t in range(3):
            for a_positions in old.subsets(old.missing(x)):
                for values in product((0, 1), repeat=len(a_positions)):
                    nxt, _, r, g = old.transition(t, x, a_positions, values)
                    if not r:
                        continue
                    q_batch = old.multiply(old.q1(t, x, i) if v else 1-old.q1(t, x, i) for i, v in zip(a_positions, values))
                    result = log_path_increment(
                        log_rho_reference=math.log(old.rho_ref(t, x, a_positions)),
                        log_rho_proposal=math.log(old.rho_prop(t, x, a_positions)),
                        log_q_batch=math.log(q_batch),
                        log_z_next=math.log(old.partition(t+1, nxt)),
                        log_z_clamped=math.log(old.clamped_partition(t, x, a_positions, values)),
                    )
                    assert result == pytest.approx(math.log(g), abs=1e-12)


def test_zero_future_mass_is_zero_weight_and_not_uniform_restart():
    assert log_path_increment(log_rho_reference=0, log_rho_proposal=0,
                              log_q_batch=0, log_z_next=-math.inf,
                              log_z_clamped=0) == -math.inf
    with pytest.raises(InvalidSpecification):
        log_path_increment(log_rho_reference=0, log_rho_proposal=0,
                           log_q_batch=0, log_z_next=0, log_z_clamped=-math.inf)
