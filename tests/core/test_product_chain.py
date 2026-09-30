"""Independent original-token checks for the standard product-chain baseline."""
from collections import Counter
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from scipy.special import logsumexp

from tri.domain.music import CountRule, MusicSpec, verify_music
from tri.errors import BudgetExceeded, InvalidSpecification, UnsupportedSpec, ZeroMass
from tri.inference.exact import Budget
from tri.inference.music_backends import MusicExactInference
from tri.inference.paired import PairedMusicInference
from tri.inference.product_chain import ProductChainMusicInference


def enumerate_original_tokens(spec, q, evidence=None):
    """No inference local transitions or auxiliary states enter this oracle."""
    evidence = evidence or {}
    missing = tuple(i for i in range(spec.length) if i not in spec.observed)
    vocabulary = (0, 1, *(p + 2 for p in spec.pitches))
    masses = {}
    for values in product(vocabulary, repeat=len(missing)):
        sequence = [spec.observed.get(i) for i in range(spec.length)]
        for pos, value in zip(missing, values):
            sequence[pos] = value
        if any(sequence[int(name[1:])] != value for name, value in evidence.items()):
            continue
        checked = verify_music(sequence, spec)
        if checked.valid:
            mass = sum(q[i, sequence[i]] for i in missing) + checked.soft_score
            if np.isfinite(mass):
                masses[tuple(sequence)] = mass
    return masses


def toy(seed=0, count=True):
    rng = np.random.default_rng(seed)
    observed = {0: 62, 3: 64, 4: 62, 7: 1 if seed % 3 == 0 else 64}
    anchors = {0: 60, 3: 62, 4: 60, 7: 62}
    if seed % 4 == 0:
        observed[1] = 0
        anchors[1] = None
    rules = (CountRule((1, 2), seed % 3), CountRule((5, 6), seed % 3)) if count else ()
    spec = MusicSpec(length=8, pitches=(60, 62), observed=observed, fixed_soundings=anchors,
                     equal_onsets=((1, 5), (2, 6)), onset_counts=rules,
                     pitch_classes={2: (0,)} if seed % 2 else {},
                     pitch_ranges={5: (60, 61)} if seed % 7 == 0 else {},
                     max_adjacent_interval=1 if seed % 5 == 0 else None,
                     motion_cost=.17, enforce_end=True, end_pitch=62)
    q = rng.dirichlet(np.ones(130), size=8)
    q[:, 129] += 5  # most full-vocabulary mass is deliberately outside working pitches
    q /= q.sum(axis=1, keepdims=True)
    return spec, np.log(q)


@pytest.mark.parametrize('seed', range(32))
@pytest.mark.parametrize('count', [False, True])
def test_product_chain_matches_brute_paired_template_marginals_and_samples(seed, count):
    spec, q = toy(seed, count)
    masses = enumerate_original_tokens(spec, q)
    expected = float(logsumexp(list(masses.values())))
    engine = ProductChainMusicInference(spec, q)
    assert engine.log_partition() == pytest.approx(expected, abs=1e-11)
    assert PairedMusicInference(spec, q).log_partition() == pytest.approx(expected, abs=1e-11)
    assert MusicExactInference(spec, q, 'template').log_partition() == pytest.approx(expected, abs=1e-11)
    if not masses:
        with pytest.raises(ZeroMass):
            engine.sample_batch(['y2'], np.random.default_rng(seed))
        with pytest.raises(ZeroMass):
            engine.marginal_log_probs('y5')
        return
    chosen = next(iter(masses))
    evidence = {'y2': chosen[2]}
    cmasses = enumerate_original_tokens(spec, q, evidence)
    cz = float(logsumexp(list(cmasses.values())))
    assert engine.log_partition(evidence) == pytest.approx(cz, abs=1e-11)
    for variable in ('y0', 'y1', 'y2', 'y5', 'y6'):
        actual = engine.marginal_log_probs(variable, evidence)
        pos = int(variable[1:])
        expected_marginal = np.array([
            logsumexp([weight for sequence, weight in cmasses.items() if sequence[pos] == token]) - cz
            for token in engine.graph.domains[variable]])
        np.testing.assert_allclose(actual, expected_marginal, atol=1e-11, rtol=0)
        assert logsumexp(actual) == pytest.approx(0, abs=1e-11)
    rng = np.random.default_rng(seed)
    for variables in ([], ['y2', 'y5'], [f'y{i}' for i in range(spec.length)]):
        drawn = engine.sample_batch(variables, rng, evidence)
        constrained = enumerate_original_tokens(spec, q, {**evidence, **drawn.assignment})
        dz = float(logsumexp(list(constrained.values())))
        assert drawn.log_probability == pytest.approx(dz - cz, abs=1e-11)
        assert drawn.log_clamped_partition == pytest.approx(dz, abs=1e-11)


def test_extreme_dynamic_range_preserves_sole_feasible_rare_paths_and_full_q():
    spec = MusicSpec(length=7, pitches=(60, 62), initial_pitch=55,
        observed={0: 1, 3: 1, 6: 1}, fixed_soundings={0: 55, 3: 55, 6: 55},
        equal_onsets=((1, 4), (2, 5)), onset_counts=())
    q = np.full((7, 130), -1000.)
    q[:, 62] = 0.  # high model mass cannot pass HOLD-only right guards
    q[[0, 3, 6]] = np.nan  # observed q must be ignored
    engine = ProductChainMusicInference(spec, q)
    expected = -4000.
    assert engine.log_partition() == pytest.approx(expected)
    assert engine.last_stats['log_semiring_products'] > 0
    assert engine.log_partition({'y1': 1}) == pytest.approx(expected)
    draw = engine.sample_batch([f'y{i}' for i in range(7)], np.random.default_rng(0))
    assert list(draw.assignment.values()) == [1] * 7
    assert draw.log_probability == pytest.approx(0)
    np.testing.assert_allclose(engine.marginal_log_probs('y4'), [-np.inf, 0., -np.inf, -np.inf])


def test_guarded_matrix_product_never_erases_supported_rare_intersection():
    stats = {'blas_products': 0, 'log_semiring_products': 0}
    a = np.array([[0., -1000.], [-np.inf, -2000.]])
    b = np.array([[-np.inf, -900.], [-1000., 0.]])
    actual = ProductChainMusicInference._log_matrix_product(a, b, stats)
    expected = logsumexp(a[:, :, None] + b[None, :, :], axis=1)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    assert np.isfinite(actual).all()
    assert stats['log_semiring_products'] == 1


def test_sampling_empirical_full_joint_instead_of_independent_marginals():
    spec = MusicSpec(length=5, pitches=(60,), observed={0: 62, 2: 62, 4: 62},
                     fixed_soundings={0: 60, 2: 60, 4: 60}, equal_onsets=((1, 3),))
    q = np.full((5, 130), 1 / 130)
    q[1] = .05 / 127
    q[1, :2] = .2
    q[1, 62] = .55
    q[3] = .15 / 127
    q[3, :2] = .1
    q[3, 62] = .65
    q = np.log(q)
    masses = enumerate_original_tokens(spec, q)
    z = float(logsumexp(list(masses.values())))
    expected = {sequence: np.exp(weight - z) for sequence, weight in masses.items()}
    engine = ProductChainMusicInference(spec, q)
    rng = np.random.default_rng(47)
    frequencies = Counter()
    for _ in range(2500):
        draw = engine.sample_batch([f'y{i}' for i in range(5)], rng)
        sequence = tuple(draw.assignment[f'y{i}'] for i in range(5))
        assert sequence in expected
        assert (sequence[1] >= 2) == (sequence[3] >= 2)
        assert draw.log_probability == pytest.approx(masses[sequence] - z)
        frequencies[sequence] += 1
    assert len(frequencies) == len(expected) == 5
    for sequence, probability in expected.items():
        assert frequencies[sequence] / 2500 == pytest.approx(probability, abs=.025)


def test_no_counter_reduction_singleton_conflicts_and_unsupported():
    spec, q = toy(2, count=False)
    engine = ProductChainMusicInference(spec, q)
    assert engine.count_size == 1
    assert engine.planning_stats['state_entries'] == 9
    flags = replace(spec, onset_counts=(CountRule((1,), 0), CountRule((5,), 1)))
    assert ProductChainMusicInference(flags, q).log_partition() == -np.inf
    inconsistent = replace(spec, onset_counts=(CountRule((1, 2), 0), CountRule((5, 6), 1)))
    assert ProductChainMusicInference(inconsistent, q).log_partition() == -np.inf
    repeated = replace(spec, onset_counts=(CountRule((1,), 0), CountRule((1,), 1)))
    assert ProductChainMusicInference(repeated, q).log_partition() == -np.inf
    for bad in (replace(spec, fixed_soundings={}), replace(spec, equal_onsets=()),
                replace(spec, equal_onsets=((1, 5),)),
                replace(spec, onset_counts=(CountRule((1, 5), 1),))):
        with pytest.raises(UnsupportedSpec):
            ProductChainMusicInference(bad, q)
    with pytest.raises(BudgetExceeded):
        ProductChainMusicInference(spec, q, Budget(max_factor_entries=1))
    with pytest.raises(BudgetExceeded):
        ProductChainMusicInference(spec, q, Budget(max_workspace_bytes=1))


def test_q_snapshot_observed_delta_queries_validation_and_cached_scalar():
    spec, q = toy(2)
    engine = ProductChainMusicInference(spec, q)
    before = engine.log_partition()
    assert not engine.last_stats['cache_hit']
    q[:] = -np.log(130)
    assert engine.log_partition() == before
    assert engine.last_stats['cache_hit']
    assert not engine.log_probs.flags.writeable
    assert engine.log_probability({'y2': 0}, {'y2': 62}) == -np.inf
    with pytest.raises(InvalidSpecification):
        engine.marginal_log_probs('y999')
    for names in ('y1', ['y1', 'y1'], ['z0']):
        with pytest.raises(InvalidSpecification):
            engine.sample_batch(names, np.random.default_rng())
    with pytest.raises(InvalidSpecification):
        engine.sample_batch(['y1'], 0)
    with pytest.raises(InvalidSpecification):
        engine.log_partition({'y1': 500})


@pytest.mark.parametrize('total', [None, 0, 1, 2])
def test_spans_touch_sequence_edges_and_terminal_guard(total):
    spec = MusicSpec(length=5, pitches=(60, 62), observed={2: 62},
        fixed_soundings={2: 60}, initial_pitch=55, enforce_end=True, end_pitch=62,
        equal_onsets=((0, 3), (1, 4)), motion_cost=.19,
        onset_counts=() if total is None else (CountRule((0, 1), total),))
    rng = np.random.default_rng(880)
    q = np.log(rng.dirichlet(np.ones(130), size=5))
    masses = enumerate_original_tokens(spec, q)
    z = float(logsumexp(list(masses.values())))
    engine = ProductChainMusicInference(spec, q)
    assert engine.log_partition() == pytest.approx(z, abs=1e-11)
    if masses:
        for var in ('y0', 'y1', 'y3', 'y4'):
            pos = int(var[1:])
            expected = [logsumexp([w for s, w in masses.items() if s[pos] == t]) - z
                        for t in engine.graph.domains[var]]
            np.testing.assert_allclose(engine.marginal_log_probs(var), expected, atol=1e-11)
        draw = engine.sample_batch([f'y{i}' for i in range(5)], rng)
        sequence = tuple(draw.assignment[f'y{i}'] for i in range(5))
        assert sequence in masses
        assert draw.log_probability == pytest.approx(masses[sequence] - z, abs=1e-11)
