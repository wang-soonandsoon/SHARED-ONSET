"""Independent semantic and query-regression checks for measured DP optimizations."""
from dataclasses import replace
from itertools import product
from unittest.mock import patch

import numpy as np
import pytest
from scipy.special import logsumexp

from tri.domain.music import CountRule, MusicSpec, verify_music
from tri.errors import BudgetExceeded, InvalidSpecification, ZeroMass
from tri.inference.exact import Budget
from tri.inference.paired import PairedMusicInference
from tri.inference.paired_optimized import ReusedPairedInference


def brute(spec, q, evidence=None):
    evidence = evidence or {}
    missing = [i for i in range(spec.length) if i not in spec.observed]
    rows = []
    for values in product((0, 1) + tuple(p + 2 for p in spec.pitches), repeat=len(missing)):
        tokens = [spec.observed.get(i) for i in range(spec.length)]
        for position, value in zip(missing, values):
            tokens[position] = value
        if any(tokens[int(name[1:])] != value for name, value in evidence.items()):
            continue
        check = verify_music(tokens, spec)
        if check.valid:
            rows.append((tuple(tokens), sum(q[i, tokens[i]] for i in missing) + check.soft_score))
    return float(logsumexp([weight for _, weight in rows])), rows


def paired_spec(width=2, pitches=(60, 62), count=None, motion=.13):
    left = tuple(range(1, width + 1))
    right = tuple(range(width + 2, 2 * width + 2))
    observed = {0: 62, width + 1: 62, 2 * width + 2: 62}
    return MusicSpec(length=2 * width + 3, pitches=pitches, observed=observed,
        fixed_soundings={i: 60 for i in observed}, equal_onsets=tuple(zip(left, right)),
        onset_counts=() if count is None else (CountRule(left, count), CountRule(right, count)),
        motion_cost=motion)


def randomized_spec(seed):
    spec = paired_spec(count=seed % 3)
    if seed % 4 == 0:
        spec = replace(spec, observed={**spec.observed, 1: 0}, fixed_soundings={**spec.fixed_soundings, 1: None})
    if seed % 3 == 0:
        spec = replace(spec, observed={**spec.observed, 6: 1})
    return replace(spec, pitch_classes={2: (0,)} if seed % 2 else {},
                   max_adjacent_interval=1 if seed % 5 == 0 else None,
                   enforce_end=True, end_pitch=60)


@pytest.mark.parametrize('seed', range(18))
@pytest.mark.parametrize('sparse', [False, True])
def test_matches_original_token_enumeration_and_old_paired(seed, sparse):
    rng = np.random.default_rng(seed + 803)
    spec = randomized_spec(seed)
    q = rng.dirichlet(np.ones(130), size=spec.length)
    q[:, 129] += 5
    q = np.log(q / q.sum(axis=1, keepdims=True))
    engine = ReusedPairedInference(spec, q, sparse=sparse)
    expected, rows = brute(spec, q)
    assert engine.log_partition() == pytest.approx(expected, abs=1e-11)
    assert PairedMusicInference(spec, q).log_partition() == pytest.approx(expected, abs=1e-11)
    if not rows:
        with pytest.raises(ZeroMass):
            engine.sample_batch(['y1'], rng)
        with pytest.raises(ZeroMass):
            engine.marginal_log_probs('y1')
        return
    evidence = {'y2': rows[0][0][2]}
    clamped, _ = brute(spec, q, evidence)
    assert engine.log_partition(evidence) == pytest.approx(clamped, abs=1e-11)
    for name in ('y1', 'y2', 'y4'):
        marginal = np.array([brute(spec, q, {**evidence, name: value})[0]
                            if name not in evidence or value == evidence[name] else -np.inf
                            for value in engine.graph.domains[name]]) - clamped
        np.testing.assert_allclose(engine.marginal_log_probs(name, evidence), marginal, atol=1e-11)
    draw = engine.sample_batch(list(engine.graph.domains), rng, evidence)
    assert verify_music(tuple(draw.assignment.values()), spec).valid
    assert draw.log_probability == pytest.approx(brute(spec, q, draw.assignment)[0] - clamped, abs=1e-11)
    partial = engine.sample_batch(['y1'], rng, evidence)
    expected_partial = brute(spec, q, {**evidence, **partial.assignment})[0]
    assert partial.log_clamped_partition == pytest.approx(expected_partial, abs=1e-11)
    assert partial.log_probability == pytest.approx(expected_partial - clamped, abs=1e-11)


@pytest.mark.parametrize('sparse', [False, True])
def test_nonempty_cold_query_then_unconditional_then_nonnested_evidence(sparse):
    spec = paired_spec(width=4, count=2)
    q = np.log(np.random.default_rng(87).dirichlet(np.ones(130), size=spec.length))
    engine = ReusedPairedInference(spec, q, sparse=sparse)
    old = PairedMusicInference(spec, q)
    early = {'y1': 62}
    assert engine.log_partition(early) == pytest.approx(old.log_partition(early), abs=1e-11)
    assert engine._base_trace is None
    assert engine.log_partition() == pytest.approx(old.log_partition(), abs=1e-11)
    base = engine._base_trace
    layers_before = [layer.copy() for layer in base[1]]
    transfers_before = [[matrix.copy() for matrix in step] for step in base[2]]
    queries = [
        {'y4': 62},         # Last column, propagate NOTE to its paired partner.
        {'y1': 0},          # Earlier column and nonnested replacement.
        {'y4': 1},          # HOLD and REST have equal onset but different state semantics.
        {'y4': 0},
        {'y6': 64},         # Clamp other span, propagating to first span.
        {'y2': 62, 'y7': 0}, # Conflicting shared onset, must not poison base.
        {},
        {'y3': 1, 'y8': 1},
    ]
    for evidence in queries:
        assert engine.log_partition(evidence) == pytest.approx(old.log_partition(evidence), abs=1e-11)
        assert engine._base_trace is base
        if evidence == {'y4': 62}:
            assert engine.last_stats['prefix_layers_reused'] == 3
            assert engine.last_stats['transfers_rebuilt'] == 2
    assert engine.log_partition({'y4': 62}) == pytest.approx(old.log_partition({'y4': 62}), abs=1e-11)
    assert engine.last_stats['prefix_layers_reused'] == 3
    assert engine.last_stats['transfers_rebuilt'] == 0  # Scalar cache hit does no work.
    for before, after in zip(layers_before, base[1]):
        np.testing.assert_array_equal(before, after)
        assert not after.flags.writeable
    for before_step, after_step in zip(transfers_before, base[2]):
        for before, after in zip(before_step, after_step):
            np.testing.assert_array_equal(before, after)
            assert not after.flags.writeable
    assert not base[3].flags.writeable


@pytest.mark.parametrize('sparse', [False, True])
def test_pitch_only_clamp_rebuilds_one_position_after_onset_is_already_forced(sparse):
    spec = paired_spec(width=3, count=1)
    spec = replace(spec, onset_counts=(*spec.onset_counts, CountRule((2,), 1)))
    q = np.full((spec.length, 130), -np.log(130))
    engine = ReusedPairedInference(spec, q, sparse=sparse)
    engine.log_partition()
    evidence = {'y2': 62}
    assert engine.log_partition(evidence) == pytest.approx(PairedMusicInference(spec, q).log_partition(evidence))
    # forced_bits is applied at transfer compilation, not _choices, so both
    # tuple choices may legitimately change even if one final transfer is the same.
    assert engine.last_stats['prefix_layers_reused'] == 1
    assert engine.last_stats['transfers_rebuilt'] in (1, 2)


@pytest.mark.parametrize('sparse', [False, True])
def test_no_count_zero_onsets_carried_pitch_and_full_observation(sparse):
    q = np.full((7, 130), -np.log(130))
    spec = MusicSpec(length=7, pitches=(60,), initial_pitch=55,
        observed={0: 1, 3: 1, 6: 1}, fixed_soundings={0: 55, 3: 55, 6: 55},
        equal_onsets=((1, 4), (2, 5)),
        onset_counts=(CountRule((1, 2), 0), CountRule((4, 5), 0)))
    q[[0, 3, 6]] = np.nan
    engine = ReusedPairedInference(spec, q, sparse=sparse)
    assert engine.log_partition() == pytest.approx(-4 * np.log(130))
    assert list(engine.sample_batch(list(engine.graph.domains), np.random.default_rng(5)).assignment.values()) == [1] * 7
    for no_count in (replace(spec, onset_counts=()), paired_spec(count=None)):
        current_q = np.full((no_count.length, 130), -np.log(130))
        expected, _ = brute(no_count, current_q)
        assert ReusedPairedInference(no_count, current_q, sparse=sparse).log_partition() == pytest.approx(expected)
    all_known = replace(spec, observed={i: 1 for i in range(7)}, fixed_soundings={i: 55 for i in range(7)})
    engine = ReusedPairedInference(all_known, np.full((7, 130), np.nan), sparse=sparse)
    assert engine.log_partition() == pytest.approx(0)
    assert engine.sample_batch(['y1'], np.random.default_rng(8)).log_probability == pytest.approx(0)


@pytest.mark.parametrize('sparse', [False, True])
def test_tiny_finite_q_preserved_model_zero_and_input_validation(sparse):
    spec = paired_spec(count=1)
    q = np.full((spec.length, 130), -1000.)
    q[:, 129] = 0
    engine = ReusedPairedInference(spec, q, sparse=sparse)
    evidence = {'y1': 62}
    expected, _ = brute(spec, q, evidence)
    assert np.isfinite(expected)
    assert engine.log_partition(evidence) == pytest.approx(expected, abs=1e-11)
    q[:] = 900
    assert engine.log_partition(evidence) == pytest.approx(expected, abs=1e-11)
    with pytest.raises(ValueError):
        engine.log_probs.flags.writeable = True
    zero_q = np.full((spec.length, 130), -np.inf)
    zero_q[:, 129] = 0
    zero_engine = ReusedPairedInference(spec, zero_q, sparse=sparse)
    assert zero_engine.log_partition({'y1': 62}) == -np.inf
    assert zero_engine._base_trace is None
    assert zero_engine.log_partition() == -np.inf
    with pytest.raises(ZeroMass):
        zero_engine.sample_batch([], np.random.default_rng(9))
    with pytest.raises(InvalidSpecification):
        ReusedPairedInference(spec, np.zeros((spec.length, 130)), sparse=sparse)
    with pytest.raises(InvalidSpecification):
        ReusedPairedInference(spec, np.full((spec.length, 130), np.nan), sparse=sparse)


@pytest.mark.parametrize('size', [8, 17])
@pytest.mark.parametrize('kind', ['random_sparse', 'empty_columns', 'all_empty', 'dense', 'very_negative'])
def test_sparse_contract_equals_independent_dense_log_semiring(size, kind):
    rng = np.random.default_rng(size + 921)
    spec = paired_spec(pitches=tuple(range(60, 60 + size - 1)))
    engine = ReusedPairedInference(spec, np.full((spec.length, 130), -np.log(130)), sparse=True)
    assert engine.size == size
    alpha = rng.normal(size=(size, size)) - 10
    first, second = rng.normal(size=(2, size, size))
    if kind != 'dense':
        first[rng.random(first.shape) > .15] = -np.inf
        second[rng.random(second.shape) > .15] = -np.inf
    if kind == 'empty_columns':
        first[:, :size // 2] = -np.inf
        second[:, 1] = -np.inf
    elif kind == 'all_empty':
        first[:] = -np.inf
    elif kind == 'very_negative':
        alpha -= 1000
        first -= 1000
        second -= 1000
    expected = logsumexp(alpha[:, :, None, None] + first[:, None, :, None] + second[None, :, None, :], axis=(0, 1))
    actual = engine._contract(alpha, first, second)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=1e-13)
    # Repeated calls exercise pointer/shape/stride plan caching.
    np.testing.assert_allclose(engine._contract(alpha, first, second), expected, atol=2e-12, rtol=1e-13)
    if kind != 'dense':
        assert engine._kernel_counts['sparse_products'] > 0
    else:
        assert engine._kernel_counts['dense_products'] > 0


@pytest.mark.parametrize('sparse', [False, True])
def test_joint_sampling_frequencies_and_no_full_draw_requery(sparse):
    spec = paired_spec(width=1, count=None)
    rng = np.random.default_rng(729)
    q = np.log(rng.dirichlet(np.ones(130), size=spec.length))
    normalizer, rows = brute(spec, q)
    expected = {tuple(seq[i] for i in (1, 3)): np.exp(weight - normalizer) for seq, weight in rows}
    frequencies = dict.fromkeys(expected, 0)
    engine = ReusedPairedInference(spec, q, sparse=sparse)
    engine.log_partition()
    with patch.object(engine, 'log_partition', side_effect=AssertionError('Unnecessary full-draw clamped Z')):
        for _ in range(1800):
            draw = engine.sample_batch(['y1', 'y3'], rng)
            key = (draw.assignment['y1'], draw.assignment['y3'])
            frequencies[key] += 1
            assert draw.log_probability == pytest.approx(np.log(expected[key]), abs=1e-11)
            assert engine.last_stats['sample_probability'] == 'complete_sequence_weight'
    variation = .5 * sum(abs(frequencies[key] / 1800 - probability) for key, probability in expected.items())
    assert variation < .065


@pytest.mark.parametrize('sparse', [False, True])
def test_cache_memory_fallback_preserves_probabilities(sparse):
    spec = paired_spec(width=4, pitches=tuple(range(60, 67)), count=1)
    q = np.full((spec.length, 130), -np.log(130))
    original = PairedMusicInference(spec, q)
    # Exactly enough for the old backend but not retained base + another suffix.
    budget = Budget(max_workspace_bytes=original.storage_bytes)
    engine = ReusedPairedInference(spec, q, budget, sparse=sparse)
    assert not engine._cache_enabled
    if sparse:
        assert not engine._sparse_enabled
    assert engine.log_partition() == pytest.approx(original.log_partition(), abs=1e-11)
    assert engine._base_trace is None
    assert engine.last_stats['peak_workspace_bytes'] <= budget.max_workspace_bytes
    evidence = {'y2': 62}
    assert engine.log_partition(evidence) == pytest.approx(original.log_partition(evidence), abs=1e-11)
    draw = engine.sample_batch(['y4'], np.random.default_rng(402), evidence)
    assert draw.log_probability == pytest.approx(original.log_probability(draw.assignment, evidence), abs=1e-11)
    with pytest.raises(BudgetExceeded):
        ReusedPairedInference(spec, q, Budget(max_workspace_bytes=original.storage_bytes - 1), sparse=sparse)


@pytest.mark.parametrize('sparse', [False, True])
def test_projected_sample_with_nonempty_evidence_has_true_conditional_mass(sparse):
    spec = paired_spec(width=3, count=1)
    q = np.log(np.random.default_rng(482).dirichlet(np.ones(130), size=spec.length))
    engine = ReusedPairedInference(spec, q, sparse=sparse)
    old = PairedMusicInference(spec, q)
    engine.log_partition()
    evidence = {'y2': 1}
    draw = engine.sample_batch(['y1', 'y7'], np.random.default_rng(712), evidence)
    assert engine.last_stats['sample_probability'] == 'projected_partition'
    assert draw.log_probability == pytest.approx(old.log_probability(draw.assignment, evidence), abs=1e-11)
    assert draw.log_clamped_partition == pytest.approx(old.log_partition({**evidence, **draw.assignment}), abs=1e-11)


def test_query_validation_and_scalar_cache_bound():
    spec = paired_spec(width=3, count=None)
    q = np.full((spec.length, 130), -np.log(130))
    engine = ReusedPairedInference(spec, q)
    rng = np.random.default_rng(513)
    for evidence in ({'h1': 60}, {'y1': True}, {'y1': 62.0}, {'y1': 99}):
        with pytest.raises(InvalidSpecification):
            engine.log_partition(evidence)
    for variables in ('y1', ['y1', 'y1'], ['h1']):
        with pytest.raises(InvalidSpecification):
            engine.sample_batch(variables, rng)
    assert engine.sample_batch([], rng).log_probability == 0
    for values in list(product((0, 1, 62, 64), repeat=4))[:140]:
        engine.log_partition(dict(zip(('y1', 'y2', 'y3', 'y5'), values)))
    assert len(engine._cache) == 128
    assert all(isinstance(value, float) and isinstance(stats, dict) for value, stats in engine._cache.values())


@pytest.mark.parametrize('sparse', [False, True])
def test_conditional_projection_releases_first_suffix_before_second_query(sparse):
    import gc
    import weakref

    spec = paired_spec(width=4, pitches=tuple(range(60, 67)), count=1)
    q = np.full((spec.length, 130), -np.log(130))
    engine = ReusedPairedInference(spec, q, sparse=sparse)
    engine.log_partition()
    base = engine._base_trace
    base_array_ids = {id(array) for array in (*base[1], base[3], *(matrix for step in base[2] for matrix in step))}
    references = []
    original_trace, original_partition = engine._trace, engine.log_partition

    def tracked_trace(evidence):
        result = original_trace(evidence)
        arrays = (*result[1], result[3], *(matrix for step in result[2] for matrix in step))
        references.extend(weakref.ref(array) for array in arrays if id(array) not in base_array_ids)
        return result

    def checked_partition(evidence):
        gc.collect()
        assert references, 'A conditional suffix was expected'
        assert all(reference() is None for reference in references), 'The sampled suffix still coexists with projected-query allocation'
        return original_partition(evidence)

    with patch.object(engine, '_trace', side_effect=tracked_trace), patch.object(engine, 'log_partition', side_effect=checked_partition):
        draw = engine.sample_batch(['y4'], np.random.default_rng(392), {'y1': 1})
    assert np.isfinite(draw.log_probability)


def test_music_sparse_branch_runs_with_pitch_restrictions_and_boundary_hold():
    spec = paired_spec(width=2, pitches=tuple(range(60, 67)), count=1)
    spec = replace(spec, observed={**spec.observed, 6: 1}, pitch_classes={2: (0, 2), 5: (0, 2)}, max_adjacent_interval=2)
    q = np.log(np.random.default_rng(932).dirichlet(np.ones(130), size=spec.length))
    expected, _ = brute(spec, q)
    engine = ReusedPairedInference(spec, q, sparse=True)
    assert engine.log_partition() == pytest.approx(expected, abs=1e-11)
    assert engine.last_stats['sparse_products'] > 0
    for evidence in ({'y2': 1}, {'y4': 62}, {'y1': 0}):
        assert engine.log_partition(evidence) == pytest.approx(brute(spec, q, evidence)[0], abs=1e-11)
