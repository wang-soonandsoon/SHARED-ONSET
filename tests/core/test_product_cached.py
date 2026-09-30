"""Equal-cache HMM control: original-token probabilities and memory lifetime."""
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from scipy.special import logsumexp

from tri.domain.music import MusicSpec, CountRule, verify_music
from tri.errors import InvalidSpecification, ZeroMass
from tri.inference.exact import Budget
from tri.inference.paired import PairedMusicInference
from tri.inference.product_chain import ProductChainMusicInference
from tri.inference.product_cached import CachedProductChainMusicInference


def fixture(seed=0, count=True):
    spec = MusicSpec(length=7, pitches=(60, 62), observed={0: 62, 3: 64, 6: 1},
        fixed_soundings={0: 60, 3: 62, 6: 62}, equal_onsets=((1, 4), (2, 5)),
        onset_counts=(CountRule((1, 2), seed % 3),) if count else (), motion_cost=.19,
        pitch_classes={2: (0, 2)} if seed % 2 else {}, enforce_end=True, end_pitch=62)
    q = np.log(np.random.default_rng(seed).dirichlet(np.ones(130), size=7))
    return spec, q


def brute(spec, q, evidence=None):
    free = [i for i in range(spec.length) if i not in spec.observed]
    weights = {}
    for values in product((0, 1, *(p + 2 for p in spec.pitches)), repeat=len(free)):
        sequence = [spec.observed.get(i) for i in range(spec.length)]
        for pos, value in zip(free, values):
            sequence[pos] = value
        if any(sequence[int(name[1:])] != value for name, value in (evidence or {}).items()):
            continue
        checked = verify_music(sequence, spec)
        if checked.valid:
            weights[tuple(sequence)] = checked.soft_score + sum(q[i, sequence[i]] for i in free)
    return weights


@pytest.mark.parametrize('seed', range(9))
@pytest.mark.parametrize('count', [True, False])
def test_cached_full_marginal_projected_and_evidence_queries_match_original_target(seed, count):
    spec, q = fixture(seed, count)
    engine = CachedProductChainMusicInference(spec, q)
    weights = brute(spec, q)
    z = float(logsumexp(list(weights.values())))
    assert engine.log_partition() == pytest.approx(z, abs=1e-11)
    assert ProductChainMusicInference(spec, q).log_partition() == pytest.approx(z, abs=1e-11)
    assert PairedMusicInference(spec, q).log_partition() == pytest.approx(z, abs=1e-11)
    if not weights:
        with pytest.raises(ZeroMass):
            engine.sample_batch(['y1'], np.random.default_rng(seed))
        return
    chosen = next(iter(weights))
    for evidence in ({}, {'y1': chosen[1]}, {'y5': chosen[5]}):
        conditional = brute(spec, q, evidence)
        cz = float(logsumexp(list(conditional.values())))
        assert engine.log_partition(evidence) == pytest.approx(cz, abs=1e-11)
        for variable in ('y1', 'y2', 'y4', 'y5'):
            pos = int(variable[1:])
            expected = [logsumexp([w for seq, w in conditional.items() if seq[pos] == token]) - cz
                        for token in engine.graph.domains[variable]]
            np.testing.assert_allclose(engine.marginal_log_probs(variable, evidence), expected, rtol=0, atol=1e-11)
        for variables in ([], ['y2', 'y4'], [f'y{i}' for i in range(7)]):
            sample = engine.sample_batch(variables, np.random.default_rng(seed), evidence)
            projected = brute(spec, q, {**evidence, **sample.assignment})
            expected = float(logsumexp(list(projected.values())))
            assert sample.log_clamped_partition == pytest.approx(expected, abs=1e-11)
            assert sample.log_probability == pytest.approx(expected - cz, abs=1e-11)
    assert engine.log_partition() == pytest.approx(z, abs=1e-11)


def test_unconditional_trace_is_shared_readonly_and_queries_do_not_mutate_it(monkeypatch):
    spec, q = fixture(1)
    engine = CachedProductChainMusicInference(spec, q)
    original_compile = engine._compile_query
    calls = []

    def counted(evidence):
        calls.append(dict(evidence))
        return original_compile(evidence)

    monkeypatch.setattr(engine, '_compile_query', counted)
    z = engine.log_partition()
    base = engine._base_trace
    original_final = base[2].copy()
    for _ in range(3):
        sample = engine.sample_batch([f'y{i}' for i in range(7)], np.random.default_rng(3))
        assert engine.last_stats['message_cache_hit']
        assert engine.last_stats['blas_products'] == engine.last_stats['log_semiring_products'] == 0
    engine.marginal_log_probs('y4')
    assert calls == [{}]
    assert engine._base_trace is base
    engine.log_partition({'y1': sample.assignment['y1']})
    assert len(calls) == 2
    assert engine._base_trace is base
    np.testing.assert_array_equal(base[2], original_final)
    arrays = (*base[1], base[2], *base[3][2], *(m for pair in base[3][4] for m in pair))
    assert all(not array.flags.writeable for array in arrays)
    with pytest.raises(ValueError):
        base[1][0][0, 0, 0] = 123
    q[:] = -np.log(130)
    assert engine.log_partition() == z


def test_first_conditional_query_does_not_build_unconditional_trace(monkeypatch):
    spec, q = fixture(1)
    engine = CachedProductChainMusicInference(spec, q)
    evidence = {'y1': 62}
    original_compile = engine._compile_query
    calls = []

    def counted(condition):
        calls.append(dict(condition))
        return original_compile(condition)

    monkeypatch.setattr(engine, '_compile_query', counted)
    expected = float(logsumexp(list(brute(spec, q, evidence).values())))
    assert engine.log_partition(evidence) == pytest.approx(expected)
    assert engine._base_trace is None
    assert calls == [evidence]
    engine.sample_batch(['y2'], np.random.default_rng(91), evidence)
    assert engine._base_trace is None
    assert all(condition for condition in calls)


def test_insufficient_extra_cache_budget_falls_back_to_original_product(monkeypatch):
    spec, q = fixture(1)
    plain = ProductChainMusicInference(spec, q)
    baseline_required = max(plain.storage_bytes, 2 * plain._input_bytes + plain._metadata_bytes)
    budget = Budget(max_workspace_bytes=baseline_required)
    engine = CachedProductChainMusicInference(spec, q, budget)
    assert not engine._cache_enabled
    assert engine._workspace_bound == plain.storage_bytes
    assert engine.log_partition() == pytest.approx(plain.log_partition(), abs=1e-11)
    assert engine._base_trace is None
    sample = engine.sample_batch(['y1', 'y4'], np.random.default_rng(23))
    expected = float(logsumexp(list(brute(spec, q, sample.assignment).values())))
    assert sample.log_clamped_partition == pytest.approx(expected, abs=1e-11)
    assert engine._base_trace is None
    assert not engine.last_stats['message_cache_enabled']
    assert engine.last_stats['peak_workspace_bytes'] <= budget.max_workspace_bytes


def test_cache_budget_reserves_base_and_transient_trace_and_retained_bytes():
    spec, q = fixture(2, count=False)
    plain = ProductChainMusicInference(spec, q)
    engine = CachedProductChainMusicInference(spec, q)
    engine.log_partition()
    assert engine._cache_enabled
    assert engine._workspace_bound >= plain.storage_bytes + engine._retained_array_bytes
    assert engine.last_stats['retained_base_array_bytes'] > 0
    engine.marginal_log_probs('y5', {'y1': 62})
    assert engine.last_stats['peak_workspace_bytes'] == engine._workspace_bound
    assert engine._base_trace is not None


def test_extreme_q_and_carried_hold_keep_old_mass_under_cached_clamps():
    spec = MusicSpec(length=7, pitches=(60, 62), initial_pitch=55,
        observed={0: 1, 3: 1, 6: 1}, fixed_soundings={0: 55, 3: 55, 6: 55},
        equal_onsets=((1, 4), (2, 5)))
    q = np.full((7, 130), -1000.)
    q[:, 62] = 0.
    q[[0, 3, 6]] = np.nan
    engine = CachedProductChainMusicInference(spec, q)
    assert engine.log_partition() == pytest.approx(-4000.)
    assert engine.last_stats['log_semiring_products'] > 0
    for _ in range(3):
        full = engine.sample_batch([f'y{i}' for i in range(7)], np.random.default_rng(0))
        assert list(full.assignment.values()) == [1] * 7
        assert full.log_probability == pytest.approx(0.)
        assert full.log_clamped_partition == pytest.approx(-4000.)
    assert engine.log_partition({'y1': 1}) == pytest.approx(-4000.)
    assert engine.log_partition({'y1': 62}) == -np.inf
    assert engine.log_partition() == pytest.approx(-4000.)
    np.testing.assert_allclose(engine.marginal_log_probs('y4'), [-np.inf, 0., -np.inf, -np.inf])
    with pytest.raises(InvalidSpecification):
        engine.sample_batch(['y1'], 0)
    with pytest.raises(InvalidSpecification):
        engine.sample_batch('y1', np.random.default_rng(0))
