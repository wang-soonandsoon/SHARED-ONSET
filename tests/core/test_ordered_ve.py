from dataclasses import replace
from itertools import product
from unittest.mock import patch

import numpy as np
import pytest
from scipy.special import logsumexp

from tri.domain.compiler import compile_music
from tri.domain.music import CountRule, MusicSpec, verify_music
from tri.errors import BudgetExceeded, InvalidSpecification, ZeroMass
from tri.inference.exact import Budget, ExactInference
from tri.inference.music_backends import make_music_engine
from tri.inference.ordered_ve import OrderedMusicVE


def brute(spec, q, evidence=None):
    evidence = evidence or {}
    missing = [i for i in range(spec.length) if i not in spec.observed]
    weighted = []
    for values in product((0, 1) + tuple(p + 2 for p in spec.pitches), repeat=len(missing)):
        tokens = [spec.observed.get(i) for i in range(spec.length)]
        for i, value in zip(missing, values):
            tokens[i] = value
        if any(tokens[int(name[1:])] != value for name, value in evidence.items()):
            continue
        check = verify_music(tokens, spec)
        if check.valid:
            weighted.append((tuple(tokens), sum(q[i, tokens[i]] for i in missing) + check.soft_score))
    return float(logsumexp([weight for _, weight in weighted])), weighted


def small_spec(seed=1):
    observed = {0: 62, 3: 64, 4: 62, 7: 1 if seed % 3 == 0 else 64}
    anchors = {0: 60, 3: 62, 4: 60, 7: 62}
    if seed % 4 == 0:
        observed[1] = 0
        anchors[1] = None
    return MusicSpec(length=8, pitches=(60, 62), observed=observed, fixed_soundings=anchors,
        equal_onsets=((1, 5), (2, 6)), onset_counts=(CountRule((1, 2), seed % 3), CountRule((5, 6), seed % 3)),
        pitch_classes={2: (0,)} if seed % 2 else {}, max_adjacent_interval=1 if seed % 5 == 0 else None,
        motion_cost=.17, enforce_end=True, end_pitch=62)


@pytest.mark.parametrize('seed', range(24))
@pytest.mark.parametrize('order', ['aligned', 'minfill', 'chronological'])
def test_all_orders_match_original_token_enumeration(seed, order):
    rng = np.random.default_rng(seed)
    spec = small_spec(seed)
    q = rng.dirichlet(np.ones(130), size=spec.length)
    q[:, 129] += 5
    q = np.log(q / q.sum(axis=1, keepdims=True))
    expected, rows = brute(spec, q)
    engine = OrderedMusicVE(spec, q, order=order)
    assert engine.log_partition() == pytest.approx(expected, abs=1e-11)
    assert make_music_engine(spec, q, backend='paired').log_partition() == pytest.approx(expected, abs=1e-11)
    if not rows:
        with pytest.raises(ZeroMass):
            engine.sample_batch(['y2'], rng)
        with pytest.raises(ZeroMass):
            engine.marginal_log_probs('y2')
        return
    evidence = {'y2': rows[0][0][2]}
    clamped, _ = brute(spec, q, evidence)
    assert engine.log_partition(evidence) == pytest.approx(clamped, abs=1e-11)
    for name in ('y0', 'y1', 'y2', 'y5'):
        expected_marginal = np.array([brute(spec, q, {**evidence, name: value})[0]
                                     if name not in evidence or value == evidence[name] else -np.inf
                                     for value in engine.graph.domains[name]]) - clamped
        np.testing.assert_allclose(engine.marginal_log_probs(name, evidence), expected_marginal, atol=1e-11)
    draw = engine.sample_batch(['y1', 'y5'], rng, evidence)
    probability = brute(spec, q, {**evidence, **draw.assignment})[0] - clamped
    assert draw.log_probability == pytest.approx(probability, abs=1e-11)
    draw = engine.sample_batch(list(engine.graph.domains), rng, evidence)
    assert verify_music(tuple(draw.assignment.values()), spec).valid
    assert draw.log_probability == pytest.approx(brute(spec, q, draw.assignment)[0] - clamped, abs=1e-11)


@pytest.mark.parametrize('seed', range(18))
def test_generic_nonpaired_relations_and_overlapping_weighted_counters(seed):
    rng = np.random.default_rng(seed + 500)
    spec = MusicSpec(length=5, pitches=(60, 62), initial_pitch=60,
        equal_onsets=((0, 3), (3, 1)) if seed % 3 else ((0, 4), (1, 3)),
        onset_counts=(CountRule((0, 1, 2, 3, 4), seed % 6), CountRule((1, 3), seed % 3)),
        observed={2: 62} if seed % 2 else {}, motion_cost=.2,
        pitch_ranges={4: (60, 60)} if seed % 2 else {}, enforce_end=seed % 2 == 0)
    q = np.log(rng.dirichlet(np.ones(130), size=5))
    expected, _ = brute(spec, q)
    reference = ExactInference(compile_music(spec, q)).log_partition()
    for order in ('aligned', 'minfill', 'chronological'):
        engine = OrderedMusicVE(spec, q, order=order)
        assert engine.log_partition() == pytest.approx(expected, abs=1e-11)
        assert engine.log_partition() == pytest.approx(reference, abs=1e-11)
        if np.isfinite(expected):
            draw = engine.sample_batch(list(engine.graph.domains), rng)
            assert verify_music(tuple(draw.assignment.values()), spec).valid


def test_carried_pitch_zero_onsets_observed_q_ignored_and_duplicate_constraints():
    spec = MusicSpec(length=7, pitches=(60,), initial_pitch=55,
        observed={0: 1, 3: 1, 6: 1}, fixed_soundings={0: 55, 3: 55, 6: 55},
        equal_onsets=((1, 4), (2, 5)),
        onset_counts=(CountRule((1, 2), 0), CountRule((4, 5), 0), CountRule((1, 2), 0)))
    q = np.full((7, 130), -np.log(130))
    q[[0, 3, 6]] = np.nan
    engine = OrderedMusicVE(spec, q)
    assert engine.log_partition() == pytest.approx(-4 * np.log(130))
    draw = engine.sample_batch(list(engine.graph.domains), np.random.default_rng(0))
    assert list(draw.assignment.values()) == [1] * 7
    assert draw.log_probability == pytest.approx(0)
    conflicting = replace(spec, onset_counts=(CountRule((1, 2), 0), CountRule((4, 5), 1)))
    assert OrderedMusicVE(conflicting, q).log_partition() == -np.inf


def test_clamps_retain_tiny_q_snapshot_and_normalized_full_vocabulary():
    spec = MusicSpec(length=3, pitches=(60,), observed={0: 62}, fixed_soundings={0: 60})
    q = np.full((3, 130), -1000.)
    q[:, 129] = 0
    engine = OrderedMusicVE(spec, q)
    expected, _ = brute(spec, q, {'y1': 62})
    assert engine.log_partition({'y1': 62}) == pytest.approx(expected)
    q[:] = 900
    assert engine.log_partition({'y1': 62}) == pytest.approx(expected)
    with pytest.raises(ValueError):
        engine.log_probs.flags.writeable = True
    with pytest.raises(InvalidSpecification):
        OrderedMusicVE(spec, np.zeros((3, 130)))
    with pytest.raises(InvalidSpecification):
        OrderedMusicVE(spec, np.full((3, 130), np.nan))
    with pytest.raises(InvalidSpecification):
        OrderedMusicVE(spec, np.ones((3, 4)))
    with pytest.raises(InvalidSpecification):
        OrderedMusicVE(spec, np.ones((3, 130), dtype=complex))


def test_saved_backward_joint_draw_distribution_and_no_tokenwise_reinference():
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
    z, rows = brute(spec, q)
    expected = sum(np.exp(weight - z) for sequence, weight in rows if sequence[1] >= 2)
    engine = OrderedMusicVE(spec, q)
    engine.log_partition()
    rng = np.random.default_rng(501)
    bits = []
    with patch.object(engine._buckets, 'eliminate', side_effect=AssertionError('Repeated VE for full draw')):
        for _ in range(1600):
            draw = engine.sample_batch(['y1', 'y3'], rng)
            assert (draw.assignment['y1'] >= 2) == (draw.assignment['y3'] >= 2)
            assert not engine.last_stats['projected_partition_query']
            bits.append(draw.assignment['y1'] >= 2)
    assert abs(np.mean(bits) - expected) < .035


def test_compile_simplification_and_factor_budget_before_dense_allocation():
    spec = small_spec(1)
    q = np.full((8, 130), -np.log(130))
    engine = OrderedMusicVE(spec, q)
    assert engine.planning_stats['count_chains'] == 1
    graph = compile_music(spec, q)
    assert engine.planning_stats['input_factor_bytes'] < sum(f.values.nbytes for f in graph.factors)
    with pytest.raises(BudgetExceeded):
        OrderedMusicVE(spec, q, Budget(max_factor_entries=1))
    with pytest.raises(BudgetExceeded):
        OrderedMusicVE(spec, q, Budget(max_workspace_bytes=10))
    # The compact graph fits but its generic aligned intermediate does not.
    engine = OrderedMusicVE(spec, q, Budget(max_factor_entries=30))
    with pytest.raises(BudgetExceeded):
        engine.log_partition()


def test_empty_batch_query_validation_and_partial_probability():
    spec = small_spec(1)
    q = np.full((8, 130), -np.log(130))
    engine = OrderedMusicVE(spec, q)
    rng = np.random.default_rng(3)
    assert engine.sample_batch([], rng).log_probability == 0
    for evidence in ({'h1': 60}, {'y1': True}, {'y1': 99}, {'y1': 62.0}):
        with pytest.raises(InvalidSpecification):
            engine.log_partition(evidence)
    for variables in ('y1', ['y1', 'y1'], ['h1']):
        with pytest.raises(InvalidSpecification):
            engine.sample_batch(variables, rng)
    with pytest.raises(InvalidSpecification):
        engine.marginal_log_probs('h1')
    with pytest.raises(InvalidSpecification):
        OrderedMusicVE(spec, q, order='paired')
    partial = engine.sample_batch(['y1'], rng)
    assert engine.last_stats['projected_partition_query']
    assert partial.log_probability == pytest.approx(engine.log_probability(partial.assignment))
