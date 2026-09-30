"""Conditional-prefix reuse against independent original-token probabilities."""
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from scipy.special import logsumexp

from tri.domain.music import CountRule, MusicSpec, verify_music
from tri.errors import InvalidSpecification, ZeroMass
from tri.inference.exact import Budget
from tri.inference.paired_optimized import ReusedPairedInference
from tri.inference.product_chain import ProductChainMusicInference
from tri.inference.product_cached import CachedProductChainMusicInference
from tri.inference.product_prefix import PrefixCachedProductChainMusicInference


def fixture(seed=0, *, width=3, count=True, partial=False, edge_spans=False):
    if edge_spans:
        first = tuple(range(width))
        second = tuple(range(width + 1, 2 * width + 1))
        length = 2 * width + 1
        observed = {width: 64}
        anchors = {width: 62}
    else:
        first = tuple(range(1, width + 1))
        second = tuple(range(width + 2, 2 * width + 2))
        length = 2 * width + 3
        observed = {0: 62, width + 1: 64, length - 1: 1 if seed % 2 else 64}
        anchors = {0: 60, width + 1: 62, length - 1: 62}
    if partial:
        observed[first[0]] = 62
        anchors[first[0]] = 60
    spec = MusicSpec(length=length, pitches=(60, 62), initial_pitch=55 if edge_spans else None,
        observed=observed, fixed_soundings=anchors, equal_onsets=tuple(zip(first, second)),
        onset_counts=(CountRule(first, seed % (width + 1)), CountRule(second, seed % (width + 1))) if count else (),
        pitch_classes={first[-1]: (0, 2)} if seed % 3 == 0 else {},
        motion_cost=.13, enforce_end=True, end_pitch=62)
    q = np.log(np.random.default_rng(seed + 23).dirichlet(np.ones(130), size=length))
    return spec, q


def brute(spec, q, evidence=None):
    free = [i for i in range(spec.length) if i not in spec.observed]
    masses = {}
    for values in product((0, 1, *(p + 2 for p in spec.pitches)), repeat=len(free)):
        sequence = [spec.observed.get(i) for i in range(spec.length)]
        for i, token in zip(free, values):
            sequence[i] = token
        if any(sequence[int(name[1:])] != value for name, value in (evidence or {}).items()):
            continue
        checked = verify_music(sequence, spec)
        if checked.valid:
            weight = checked.soft_score + sum(q[i, sequence[i]] for i in free)
            if np.isfinite(weight):
                masses[tuple(sequence)] = weight
    return masses


def z_of(masses):
    return float(logsumexp(list(masses.values())))


def conditional(masses, evidence):
    return {seq: value for seq, value in masses.items()
            if all(seq[int(name[1:])] == token for name, token in evidence.items())}


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('count', [True, False])
def test_early_middle_late_multisite_sampling_and_marginals_match_brute(seed, count):
    spec, q = fixture(seed, count=count, partial=seed % 2 == 0, edge_spans=seed % 3 == 0)
    all_masses = brute(spec, q)
    engine = PrefixCachedProductChainMusicInference(spec, q)
    plain = ProductChainMusicInference(spec, q)
    reused = CachedProductChainMusicInference(spec, q)
    paired = ReusedPairedInference(spec, q)
    for other in (engine, plain, reused, paired):
        assert other.log_partition() == pytest.approx(z_of(all_masses), abs=1e-11)
    if not all_masses:
        with pytest.raises(ZeroMass):
            engine.sample_batch(['y0'], np.random.default_rng(seed))
        return
    witness = next(iter(all_masses))
    first, second = engine.spans
    queries = [{}, {f'y{first[0]}': witness[first[0]]},
               {f'y{first[1]}': witness[first[1]]}, {f'y{second[-1]}': witness[second[-1]]},
               {f'y{first[1]}': witness[first[1]], f'y{second[-1]}': witness[second[-1]]}]
    rng = np.random.default_rng(seed + 99)
    for evidence in queries:
        masses = conditional(all_masses, evidence)
        z = z_of(masses)
        for other in (engine, plain, reused, paired):
            assert other.log_partition(evidence) == pytest.approx(z, abs=1e-11)
        for pos in (first[0], first[-1], second[1]):
            variable = f'y{pos}'
            expected = [logsumexp([w for seq, w in masses.items() if seq[pos] == token]) - z
                        for token in engine.graph.domains[variable]]
            np.testing.assert_allclose(engine.marginal_log_probs(variable, evidence), expected, atol=1e-11, rtol=0)
        for variables in ([], [f'y{first[-1]}', f'y{second[0]}'], [f'y{i}' for i in range(spec.length)]):
            sample = engine.sample_batch(variables, rng, evidence)
            constrained = conditional(masses, sample.assignment)
            assert sample.log_clamped_partition == pytest.approx(z_of(constrained), abs=1e-11)
            assert sample.log_probability == pytest.approx(z_of(constrained) - z, abs=1e-11)
    assert engine.log_partition() == pytest.approx(z_of(all_masses), abs=1e-11)


def test_actual_prefix_and_unchanged_transfers_are_shared_but_base_never_mutated():
    spec, q = fixture(1, width=5, count=False)
    engine = PrefixCachedProductChainMusicInference(spec, q)
    engine.log_partition()
    base = engine._base_trace
    base_messages = [a.copy() for a in base[1]]
    first, second = engine.spans
    for index in (0, 2, 4, 1, 3):
        evidence = {f'y{first[index]}': 62}
        z, layers, final, query, stats = engine._forward_chain(evidence, save=True)
        expected = ProductChainMusicInference(spec, q).log_partition(evidence)
        assert z == pytest.approx(expected, abs=1e-11)
        assert stats['prefix_layers_reused'] == index
        assert stats['suffix_layers_recomputed'] == engine.width - index
        # NOTE evidence also removes REST/HOLD from its paired position.
        assert stats['transfers_rebuilt'] == 2
        assert stats['transfers_reused'] == 2 * engine.width - 2
        for j in range(index + 1):
            assert layers[j] is base[1][j]
        for j in range(engine.width):
            for side in (0, 1):
                assert (query[4][j][side] is base[3][4][j][side]) == (j != index)
        assert engine._base_trace is base
        for actual, original in zip(base[1], base_messages):
            np.testing.assert_array_equal(actual, original)
    assert all(not a.flags.writeable for a in (*base[1], base[2], *base[3][2], *(m for pair in base[3][4] for m in pair)))


def test_multisite_earliest_change_uses_base_instead_of_previous_conditional():
    spec, q = fixture(1, width=5, count=False)
    engine = PrefixCachedProductChainMusicInference(spec, q)
    engine.log_partition()
    first, second = engine.spans
    queries = [{f'y{first[4]}': 62}, {f'y{second[1]}': 0, f'y{first[3]}': 62},
               {f'y{first[2]}': 1}, {f'y{second[4]}': 64}]
    for evidence, earliest in zip(queries, (4, 1, 2, 4)):
        assert engine.log_partition(evidence) == pytest.approx(ProductChainMusicInference(spec, q).log_partition(evidence), abs=1e-11)
        assert engine.last_stats['prefix_layers_reused'] == earliest
    engine.log_partition(queries[0])
    assert engine.last_stats['scalar_cache_hit']
    assert engine.last_stats['transfers_rebuilt'] == engine.last_stats['suffix_layers_recomputed'] == 0
    assert engine.last_stats['blas_products'] == engine.last_stats['log_semiring_products'] == 0


@pytest.mark.parametrize('operation', ['partition', 'sample', 'marginal'])
def test_first_nonempty_query_does_not_build_an_unconditional_trace(monkeypatch, operation):
    spec, q = fixture(1, count=False)
    engine = PrefixCachedProductChainMusicInference(spec, q)
    original = engine._compile_query
    conditions = []

    def record(evidence):
        conditions.append(dict(evidence))
        return original(evidence)

    monkeypatch.setattr(engine, '_compile_query', record)
    evidence = {'y2': 62}
    if operation == 'partition':
        engine.log_partition(evidence)
    elif operation == 'sample':
        engine.sample_batch(['y1', 'y7'], np.random.default_rng(3), evidence)
    else:
        engine.marginal_log_probs('y1', evidence)
    assert conditions and all(conditions)
    assert engine._base_trace is None
    assert engine.last_stats['prefix_layers_reused'] == 0


def test_budget_fallback_uses_uncached_standard_queries():
    spec, q = fixture(1, width=3)
    plain = ProductChainMusicInference(spec, q)
    required = max(plain.storage_bytes, 2 * plain._input_bytes + plain._metadata_bytes)
    engine = PrefixCachedProductChainMusicInference(spec, q, Budget(max_workspace_bytes=required))
    assert not engine._cache_enabled
    for evidence in ({}, {'y2': 62}, {'y3': 1}):
        assert engine.log_partition(evidence) == pytest.approx(plain.log_partition(evidence), abs=1e-11)
        assert engine._base_trace is None
        assert engine.last_stats['prefix_layers_reused'] == 0
        assert engine.last_stats['peak_workspace_bytes'] <= required
    draw = engine.sample_batch(['y2'], np.random.default_rng(4), {'y3': 1})
    assert draw.log_probability == pytest.approx(plain.log_probability(draw.assignment, {'y3': 1}), abs=1e-11)


@pytest.mark.parametrize('count', [False, True])
def test_extreme_q_zero_conditionals_and_carried_hold_preserve_frozen_weights(count):
    spec = MusicSpec(length=9, pitches=(60, 62), initial_pitch=55,
        observed={0: 1, 4: 1, 8: 1}, fixed_soundings={0: 55, 4: 55, 8: 55},
        equal_onsets=((1, 5), (2, 6), (3, 7)),
        onset_counts=(CountRule((1, 2, 3), 0), CountRule((5, 6, 7), 0)) if count else ())
    q = np.full((9, 130), -1000.)
    q[:, 62] = 0.
    q[[0, 4, 8]] = np.nan
    engine = PrefixCachedProductChainMusicInference(spec, q)
    assert engine.log_partition() == pytest.approx(-6000.)
    if not count:
        assert engine.last_stats['log_semiring_products'] > 0
    for evidence in ({'y7': 1}, {'y2': 1, 'y7': 1}, {'y3': 62}):
        expected = -np.inf if evidence.get('y3') == 62 else -6000.
        assert engine.log_partition(evidence) == pytest.approx(expected)
        if not np.isfinite(expected):
            with pytest.raises(ZeroMass):
                engine.marginal_log_probs('y1', evidence)
            with pytest.raises(ZeroMass):
                engine.sample_batch(['y1'], np.random.default_rng(0), evidence)
        else:
            sample = engine.sample_batch(['y1', 'y7'], np.random.default_rng(0), evidence)
            assert sample.assignment == {'y1': 1, 'y7': 1}
            assert sample.log_probability == pytest.approx(0.)
            assert sample.log_clamped_partition == pytest.approx(-6000.)
    assert engine.log_partition() == pytest.approx(-6000.)


def test_zero_mass_base_and_impossible_fixed_boundary_can_be_reused_safely():
    spec, q = fixture(0, count=False)
    # The first observed NOTE emits C, so anchoring it to D makes the base zero.
    bad = replace(spec, fixed_soundings={**spec.fixed_soundings, 0: 62})
    engine = PrefixCachedProductChainMusicInference(bad, q)
    assert engine.log_partition() == -np.inf
    assert engine._base_trace is not None
    assert engine.log_partition({'y3': 62}) == -np.inf
    assert engine.last_stats['zero_mass_base_shortcut']
    with pytest.raises(ZeroMass):
        engine.sample_batch(['y1'], np.random.default_rng(0), {'y3': 62})
    with pytest.raises(ZeroMass):
        engine.marginal_log_probs('y1', {'y3': 62})


def test_observed_evidence_needs_no_changed_transfer_and_new_q_needs_new_engine():
    spec, q = fixture(1, count=False)
    original_q = q.copy()
    engine = PrefixCachedProductChainMusicInference(spec, q)
    z = engine.log_partition()
    assert engine.log_partition({'y0': 62}) == pytest.approx(z)
    assert engine.last_stats['prefix_layers_reused'] == engine.width
    assert engine.last_stats['transfers_rebuilt'] == 0
    assert engine.last_stats['suffix_layers_recomputed'] == 0
    q[:] = -np.log(130)
    assert engine.log_partition({'y3': 62}) == pytest.approx(ProductChainMusicInference(spec, original_q).log_partition({'y3': 62}), abs=1e-11)
    other = PrefixCachedProductChainMusicInference(spec, q)
    assert other._base_trace is None
    assert other.log_partition() == pytest.approx(ProductChainMusicInference(spec, q).log_partition(), abs=1e-11)
    with pytest.raises(InvalidSpecification):
        engine.log_partition({'y0': 64})
    with pytest.raises(InvalidSpecification):
        engine.sample_batch(['y1'], 3)


def test_restrictive_pitch_and_interval_rules_and_explicit_model_zeros():
    spec, q = fixture(1, count=False)
    spec = replace(spec, pitch_classes={2: (0,), 7: (2,)},
                   pitch_ranges={3: (60, 60)}, max_adjacent_interval=0)
    masses = brute(spec, q)
    assert masses
    engine = PrefixCachedProductChainMusicInference(spec, q)
    assert engine.log_partition() == pytest.approx(z_of(masses), abs=1e-11)
    for evidence in ({'y3': 62}, {'y2': 1, 'y7': 64}, {'y1': 0}):
        expected = z_of(conditional(masses, evidence))
        assert engine.log_partition(evidence) == pytest.approx(expected, abs=1e-11)
    spec, q = fixture(1, count=False)
    q[3] = -np.inf
    q[3, 62] = 0.
    engine = PrefixCachedProductChainMusicInference(spec, q)
    assert np.isfinite(engine.log_partition())
    assert engine.log_partition({'y3': 1}) == -np.inf
    assert engine.log_partition({'y3': 62}) == pytest.approx(z_of(brute(spec, q)), abs=1e-11)


def test_factory_registration_and_public_query_contract():
    from tri.inference.music_backends import make_music_engine
    spec, q = fixture(1, width=2)
    engine = make_music_engine(spec, q, backend='product_prefix')
    assert isinstance(engine, PrefixCachedProductChainMusicInference)
    assert engine.backend_name == engine.requested_backend == 'product_prefix'
    expected = ProductChainMusicInference(spec, q)
    assert engine.log_partition() == pytest.approx(expected.log_partition(), abs=1e-11)
    assert engine.log_clamped_partition({'y2': 62}) == pytest.approx(expected.log_clamped_partition({'y2': 62}), abs=1e-11)
    assert engine.log_probability({'y2': 62}) == pytest.approx(expected.log_probability({'y2': 62}), abs=1e-11)
