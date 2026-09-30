"""Original-Y exhaustive checks for independent exact music strategies."""

from dataclasses import replace
from itertools import islice, product
import math
from unittest.mock import patch

import numpy as np
import pytest
from scipy.special import logsumexp

from tri.domain.music import HOLD, REST, CountRule, MusicSpec, verify_music
from tri.errors import BudgetExceeded, InvalidSpecification, ZeroMass
from tri.inference.exact import Budget
from tri.inference.music_backends import MusicExactInference, make_music_engine


SPECIALIZED = ("template", "automaton", "auto")


def uniform(length):
    return np.full((length, 130), -np.log(130))


def original_log_masses(spec, q, evidence=None):
    evidence = evidence or {}
    vocabulary = (0, 1) + tuple(p + 2 for p in spec.pitches)
    choices = [(spec.observed[i],) if i in spec.observed else vocabulary for i in range(spec.length)]
    masses = {}
    for y in product(*choices):
        if any(y[int(name[1:])] != value for name, value in evidence.items()):
            continue
        checked = verify_music(y, spec)
        if checked.valid:
            masses[y] = checked.soft_score + sum(q[i, token] for i, token in enumerate(y) if i not in spec.observed)
    return masses


def brute_log_z(spec, q, evidence=None):
    return float(logsumexp(list(original_log_masses(spec, q, evidence).values())))


@pytest.mark.parametrize("seed", range(20))
def test_random_feasible_witnesses_keep_nonzero_support(seed):
    rng = np.random.default_rng(7300 + seed)
    witness, soundings, active = [], [], 60
    for _ in range(5):
        token = int(rng.choice((0, 1, 62, 66)))
        if token == HOLD and active is None:
            token = REST
        active = None if token == REST else (active if token == HOLD else token - 2)
        witness.append(token)
        soundings.append(active)
    pairs = tuple((i, j) for i in range(5) for j in range(i + 1, 5) if (witness[i] >= 2) == (witness[j] >= 2) and rng.random() < .4)
    observed = {i: token for i, token in enumerate(witness) if rng.random() < .3}
    spec = MusicSpec(5, (60, 64), initial_pitch=60, observed=observed,
                     fixed_soundings={i: soundings[i] for i in observed}, equal_onsets=pairs,
                     onset_counts=tuple(CountRule(positions, sum(witness[i] >= 2 for i in positions)) for positions in ((0, 2, 4), (1, 2, 3))),
                     pitch_classes={i: (pitch % 12,) for i, pitch in enumerate(soundings) if pitch is not None and rng.random() < .3},
                     pitch_ranges={2: (60, 64)}, max_adjacent_interval=4, motion_cost=.19,
                     enforce_end=True, end_pitch=soundings[-1])
    q = np.log(rng.dirichlet(np.ones(130), size=5))
    masses = original_log_masses(spec, q)
    expected = float(logsumexp(list(masses.values())))
    assert tuple(witness) in masses and np.isfinite(expected)
    evidence = {"y1": witness[1]}
    event = float(logsumexp([value for y, value in masses.items() if y[1] == witness[1]]))
    for backend in ("ve",) + SPECIALIZED:
        solver = make_music_engine(spec, q, backend)
        assert solver.log_partition() == pytest.approx(expected, abs=1e-11)
        assert solver.log_clamped_partition(evidence) == pytest.approx(event, abs=1e-11)
        batch = solver.sample_batch(("y0", "y2", "y4"), rng, evidence)
        joint = float(logsumexp([value for y, value in masses.items() if y[1] == witness[1] and all(y[int(k[1:])] == v for k, v in batch.assignment.items())]))
        assert batch.log_probability == pytest.approx(joint - event, abs=1e-11)


@pytest.mark.parametrize("seed", range(50))
def test_random_coupled_music_matches_original_y_and_ve(seed):
    rng = np.random.default_rng(6200 + seed)
    length = 4
    vocabulary = (REST, HOLD, 62, 66)
    observed = {i: int(rng.choice(vocabulary)) for i in range(length) if rng.random() < .23}
    anchors = {i: rng.choice([None, 60, 64]) for i in observed if rng.random() < .5}
    spec = MusicSpec(length, (60, 64), observed=observed,
                     initial_pitch=rng.choice([None, 60, 72]),
                     fixed_soundings=anchors,
                     equal_onsets=((0, 2), (1, 3)) if seed % 3 else ((0, 2), (2, 3), (3, 0)),
                     onset_counts=(CountRule((0, 1, 2), int(rng.integers(4))), CountRule((1, 3), int(rng.integers(3)))),
                     pitch_ranges={int(rng.integers(4)): (60, 64)},
                     pitch_classes={int(rng.integers(4)): (0,)},
                     max_adjacent_interval=int(rng.choice([0, 4, 12])), motion_cost=.17,
                     enforce_end=bool(seed % 2), end_pitch=None if not seed % 2 else rng.choice([None, 60, 64, 72]))
    q = np.log(rng.dirichlet(np.ones(130), size=length))
    expected = brute_log_z(spec, q)
    evidence = {"y0": observed.get(0, 62)}
    clamped = brute_log_z(spec, q, evidence)
    for backend in ("ve",) + SPECIALIZED:
        solver = make_music_engine(spec, q, backend)
        assert solver.log_partition() == pytest.approx(expected, abs=1e-11)
        assert solver.log_clamped_partition(evidence) == pytest.approx(clamped, abs=1e-11)
        if np.isfinite(clamped):
            marginal = solver.marginal_log_probs("y3", evidence)
            for value, log_p in zip(solver.graph.domains["y3"], marginal):
                assert log_p == pytest.approx(brute_log_z(spec, q, {**evidence, "y3": value}) - clamped, abs=1e-11)
            sampled = solver.sample_batch(("y1", "y3"), rng, evidence)
            assert sampled.log_probability == pytest.approx(brute_log_z(spec, q, {**evidence, **sampled.assignment}) - clamped, abs=1e-11)
        else:
            with pytest.raises(ZeroMass):
                solver.sample_batch((), rng, evidence)


@pytest.mark.parametrize("backend", SPECIALIZED)
@pytest.mark.parametrize("spec", [
    MusicSpec(3, (60, 64), initial_pitch=60, observed={1: HOLD}, fixed_soundings={1: 60},
              equal_onsets=((0, 2),), motion_cost=.25),
    MusicSpec(2, (), initial_pitch=72, observed={1: HOLD}, fixed_soundings={1: 72},
              onset_counts=(CountRule((0, 1), 0),)),
    MusicSpec(4, (60, 64), equal_onsets=((0, 2), (1, 3)),
              onset_counts=(CountRule((0, 1), 1), CountRule((2, 3), 1)),
              pitch_classes={0: (0,), 1: (0,), 2: (4,), 3: (4,)}, motion_cost=.5),
    MusicSpec(4, (60, 64), initial_pitch=60, observed={1: HOLD}, fixed_soundings={1: 60},
              equal_onsets=((0, 2), (2, 3)), onset_counts=(CountRule((0, 2, 3), 3),), motion_cost=.3),
    MusicSpec(3, (60,), initial_pitch=60, equal_onsets=((0, 0),),
              onset_counts=(CountRule((), 0),), enforce_end=True, end_pitch=None),
    MusicSpec(4, (60, 64), observed={0: 62, 1: HOLD, 2: REST, 3: 66},
              fixed_soundings={0: 60, 1: 60, 2: None, 3: 64}),
])
def test_full_semantics_and_joint_batch_probabilities(backend, spec):
    rng = np.random.default_rng(413)
    q = np.log(rng.dirichlet(np.ones(130), size=spec.length))
    for i in spec.observed:
        q[i] = np.nan
    masses = original_log_masses(spec, q)
    base = float(logsumexp(list(masses.values())))
    assert np.isfinite(base)
    solver = make_music_engine(spec, q, backend)
    assert solver.log_partition() == pytest.approx(base, abs=1e-11)
    for _ in range(12):
        names = tuple(f"y{i}" for i in range(spec.length))
        sample = solver.sample_batch(names, rng)
        tokens = tuple(sample.assignment[name] for name in names)
        assert verify_music(tokens, spec).valid
        assert sample.log_probability == pytest.approx(masses[tokens] - base, abs=1e-11)
        partial = solver.sample_batch(names[::2], rng)
        assert partial.log_probability == pytest.approx(brute_log_z(spec, q, partial.assignment) - base, abs=1e-11)


@pytest.mark.parametrize("backend", SPECIALIZED)
def test_clamp_keeps_old_q_and_missing_pitch_mass(backend):
    spec = MusicSpec(2, (60, 64), equal_onsets=((0, 1),), motion_cost=.4)
    q = uniform(2)
    solver = make_music_engine(spec, q, backend)
    evidence = {"y0": 62}
    expected = brute_log_z(spec, q, evidence)
    assert solver.log_clamped_partition(evidence) == pytest.approx(expected, abs=1e-12)
    reobserved = make_music_engine(replace(spec, observed={0: 62}), q, backend)
    assert reobserved.log_partition() == pytest.approx(expected + np.log(130), abs=1e-12)
    single = make_music_engine(MusicSpec(1, (60,)), uniform(1), backend)
    assert single.log_partition() == pytest.approx(np.log(2 / 130))


@pytest.mark.parametrize("backend", SPECIALIZED)
def test_no_equality_or_count_still_sums_pitch_and_hold_chain(backend):
    spec = MusicSpec(4, (60, 64), initial_pitch=60, max_adjacent_interval=4,
                     motion_cost=.2, observed={3: HOLD}, fixed_soundings={3: 64})
    solver = make_music_engine(spec, uniform(4), backend)
    assert solver.log_partition() == pytest.approx(brute_log_z(spec, uniform(4)), abs=1e-12)
    if solver.backend_name == "template":
        assert solver.last_stats["onset_templates"] == 1


@pytest.mark.parametrize("backend", SPECIALIZED)
def test_sampler_frequencies_follow_joint_law(backend):
    spec = MusicSpec(2, (60,), initial_pitch=60, equal_onsets=((0, 1),))
    q = np.full((2, 130), -np.inf)
    q[:, [0, 1, 62]] = np.log([[.15, .25, .6], [.25, .25, .5]])
    masses = original_log_masses(spec, q)
    z = float(logsumexp(list(masses.values())))
    expected = {y: np.exp(weight - z) for y, weight in masses.items()}
    solver = make_music_engine(spec, q, backend)
    rng = np.random.default_rng(514)
    counts = {y: 0 for y in masses}
    for _ in range(800):
        sampled = solver.sample_batch(("y0", "y1"), rng)
        y = sampled.assignment["y0"], sampled.assignment["y1"]
        counts[y] += 1
        assert sampled.log_probability == pytest.approx(math.log(expected[y]), abs=1e-12)
    for y in counts:
        assert abs(counts[y] / 800 - expected[y]) < .07


@pytest.mark.parametrize("backend", SPECIALIZED)
def test_immutability_empty_batch_and_invalid_queries(backend):
    q = uniform(2)
    solver = make_music_engine(MusicSpec(2, (60,)), q, backend)
    expected = solver.log_partition()
    q[:] = 0.0
    assert solver.log_partition() == expected
    with pytest.raises(ValueError):
        solver.log_probs.flags.writeable = True
    rng = np.random.default_rng(42)
    before = rng.bit_generator.state
    empty = solver.sample_batch((), rng)
    assert empty.assignment == {} and empty.log_probability == 0.
    assert before == rng.bit_generator.state
    with pytest.raises(InvalidSpecification):
        solver.log_partition({"h0": 60})
    with pytest.raises(InvalidSpecification):
        solver.log_partition({"y0": 129})
    with pytest.raises(InvalidSpecification):
        solver.sample_batch(("y0", "y0"), rng)
    assert solver.log_probability({"y0": 62}, {"y0": 0}) == -np.inf


@pytest.mark.parametrize("backend", SPECIALIZED)
def test_zero_mass_is_not_repaired(backend):
    q = np.full((1, 130), -np.inf)
    q[0, 129] = 0.
    solver = make_music_engine(MusicSpec(1, (60,)), q, backend)
    assert solver.log_partition() == -np.inf
    with pytest.raises(ZeroMass):
        solver.sample_batch((), np.random.default_rng(42))
    contradictory = MusicSpec(2, (60,), equal_onsets=((0, 1),), observed={0: 62, 1: REST})
    assert make_music_engine(contradictory, uniform(2), backend).log_partition() == -np.inf


def test_template_count_guard_runs_before_any_pitch_chain():
    spec = MusicSpec(14, (60,), equal_onsets=tuple((i, i + 7) for i in range(7)))
    solver = make_music_engine(spec, uniform(14), "template", Budget(max_oracle_assignments=10))
    with patch.object(MusicExactInference, "_chain", side_effect=AssertionError("enumerated too soon")):
        with pytest.raises(BudgetExceeded, match="template count"):
            solver.log_partition()


def test_automaton_guard_runs_before_transition_expansion():
    solver = make_music_engine(MusicSpec(2, (60,)), uniform(2), "automaton", Budget(max_factor_entries=2))
    with patch.object(MusicExactInference, "_automaton_step", side_effect=AssertionError("expanded too soon")):
        with pytest.raises(BudgetExceeded, match="candidate transitions"):
            solver.log_partition()


def test_template_transition_work_and_workspace_guards():
    spec = MusicSpec(8, (60,), equal_onsets=tuple((i, i + 4) for i in range(4)),
                     onset_counts=(CountRule((0, 1, 2, 3), 1),))
    solver = make_music_engine(spec, uniform(8), "template", Budget(max_factor_entries=12))
    with pytest.raises(BudgetExceeded, match="transition slab"):
        solver.log_partition()
    with pytest.raises(BudgetExceeded):
        make_music_engine(spec, uniform(8), "automaton", Budget(max_workspace_bytes=1024))


def test_auto_uses_computed_bounds_and_handles_template_explosion():
    spec = MusicSpec(12, (60,), equal_onsets=tuple((2 * i, 2 * i + 1) for i in range(6)))
    engine = make_music_engine(spec, uniform(12), "auto", Budget(max_oracle_assignments=10))
    assert engine.backend_name == "automaton"
    assert "template_budget_reason" in engine.planning_stats
    assert engine.planning_stats["automaton"]["max_open_equalities"] == 1
    assert engine.log_partition() == pytest.approx(make_music_engine(spec, uniform(12), "ve").log_partition(), abs=1e-11)


def test_scalar_partition_cache_stays_bounded():
    solver = make_music_engine(MusicSpec(4, (60, 64)), uniform(4), "automaton")
    for y in islice(product((0, 1, 62, 66), repeat=4), 130):
        solver.log_partition({f"y{i}": token for i, token in enumerate(y)})
    assert len(solver._cache) == 128


def test_invalid_backend_and_model_probabilities():
    spec = MusicSpec(1, (60,))
    with pytest.raises(InvalidSpecification):
        make_music_engine(spec, uniform(1), "approximate")
    for q in (np.zeros((1, 130)), np.zeros((1, 129)), np.full((1, 130), np.nan), uniform(1).astype(complex) + 1j):
        with pytest.raises(InvalidSpecification):
            make_music_engine(spec, q, "automaton")
