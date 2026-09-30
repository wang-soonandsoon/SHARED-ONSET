from dataclasses import replace
from itertools import product
from unittest.mock import patch

import numpy as np
import pytest
from scipy.special import logsumexp

from tri.domain.music import CountRule, MusicSpec, verify_music
from tri.errors import BudgetExceeded, InvalidSpecification, ZeroMass
from tri.inference.exact import Budget
from tri.inference.music_backends import MusicExactInference
from tri.inference.paired import PairedMusicInference
from tri.inference.template_stream import StreamingTemplateMusicInference


def case(width=3, count=1):
    left = tuple(range(1, width + 1))
    right = tuple(range(width + 2, 2 * width + 2))
    observed = {0: 62, width + 1: 62, 2 * width + 2: 62}
    spec = MusicSpec(length=2 * width + 3, pitches=(60, 62), observed=observed,
        fixed_soundings={i: 60 for i in observed}, equal_onsets=tuple(zip(left, right)),
        onset_counts=(CountRule(left, count), CountRule(right, count)), motion_cost=.13)
    q = np.log(np.random.default_rng(width + count).dirichlet(np.ones(130), size=spec.length))
    return spec, q


def brute(spec, q, evidence=None):
    evidence = evidence or {}
    missing = [i for i in range(spec.length) if i not in spec.observed]
    weights = []
    for values in product((0, 1) + tuple(p + 2 for p in spec.pitches), repeat=len(missing)):
        tokens = [spec.observed.get(i) for i in range(spec.length)]
        for position, value in zip(missing, values):
            tokens[position] = value
        if any(tokens[int(name[1:])] != value for name, value in evidence.items()):
            continue
        check = verify_music(tokens, spec)
        if check.valid:
            weights.append(sum(q[i, tokens[i]] for i in missing) + check.soft_score)
    return float(logsumexp(weights))


def test_old_fictitious_slab_rejects_while_serial_budget_computes_exact_z():
    spec, q = case()
    budget = Budget(max_factor_entries=20, max_workspace_bytes=100_000)
    old = MusicExactInference(spec, q, 'template', budget)
    with pytest.raises(BudgetExceeded, match='templates x states x tokens'):
        old.log_partition()
    engine = StreamingTemplateMusicInference(spec, q, budget)
    with patch.object(engine, '_automaton_forward', side_effect=AssertionError('Wrong dispatch')):
        assert engine.log_partition() == pytest.approx(brute(spec, q), abs=1e-11)
    assert engine.last_stats['onset_templates'] == 3
    assert engine.last_stats['transition_slab_bound'] <= 20
    assert engine.last_stats['transition_budget_kind'] == 'per_template_live_chain'
    assert engine.last_stats['backend'] == 'template_stream'
    assert engine.requested_backend == 'template_stream'


@pytest.mark.parametrize('seed', range(8))
def test_original_numerics_samples_and_clamps_are_unchanged(seed):
    spec, q = case(width=2, count=seed % 3)
    if seed % 2:
        spec = replace(spec, pitch_classes={2: (0,)}, observed={**spec.observed, 6: 1})
    old = MusicExactInference(spec, q, 'template')
    engine = StreamingTemplateMusicInference(spec, q)
    expected = brute(spec, q)
    assert engine.log_partition() == pytest.approx(expected, abs=1e-11)
    assert engine.log_partition() == old.log_partition()
    if np.isneginf(expected):
        with pytest.raises(ZeroMass):
            engine.sample_batch(['y1'], np.random.default_rng(seed))
        return
    evidence = {'y1': 1 if seed % 3 == 0 else 62}
    assert engine.log_partition(evidence) == pytest.approx(brute(spec, q, evidence), abs=1e-11)
    # Same seeds give identical samples because no enumeration/sampling order changed.
    for variables in (list(engine.graph.domains), ['y1', 'y4']):
        with patch.object(engine, '_sample_automaton', side_effect=AssertionError('Wrong sampling dispatch')):
            sampled = engine.sample_batch(variables, np.random.default_rng(seed + 500))
        reference = old.sample_batch(variables, np.random.default_rng(seed + 500))
        assert sampled.assignment == reference.assignment
        assert sampled.log_probability == reference.log_probability
        assert sampled.log_clamped_partition == reference.log_clamped_partition
        assert sampled.log_clamped_partition == pytest.approx(brute(spec, q, sampled.assignment), abs=1e-11)
        assert engine.last_stats['sampling']['backend'] == 'template_stream'
    np.testing.assert_allclose(engine.marginal_log_probs('y2'), old.marginal_log_probs('y2'), atol=1e-11)


def test_actual_stored_sampling_weights_still_exceed_small_workspace():
    spec, q = case(width=10, count=5)
    budget = Budget(max_workspace_bytes=100_000)
    engine = StreamingTemplateMusicInference(spec, q, budget)
    assert engine.log_partition() == pytest.approx(PairedMusicInference(spec, q).log_partition(), abs=1e-11)
    assert engine.last_stats['onset_templates'] == 252
    with pytest.raises(BudgetExceeded, match='Template plan and sampling weights'):
        engine.sample_batch(list(engine.graph.domains), np.random.default_rng(9))


def test_live_plan_transitions_and_enumeration_budget_are_not_disabled():
    spec, q = case()
    with pytest.raises(BudgetExceeded):
        StreamingTemplateMusicInference(spec, q, Budget(max_factor_entries=2)).log_partition()
    with pytest.raises(BudgetExceeded, match='max_oracle_assignments'):
        StreamingTemplateMusicInference(spec, q, Budget(max_oracle_assignments=2)).log_partition()
    with pytest.raises(BudgetExceeded):
        StreamingTemplateMusicInference(spec, q, Budget(max_workspace_bytes=10))


def test_zero_mass_observed_delta_tiny_q_and_validation():
    spec, _ = case(width=2, count=1)
    q = np.full((spec.length, 130), -1000.)
    q[:, 129] = 0
    q[list(spec.observed)] = np.nan
    engine = StreamingTemplateMusicInference(spec, q)
    expected = brute(spec, q, {'y1': 62})
    assert engine.log_partition({'y1': 62}) == pytest.approx(expected, abs=1e-11)
    q[:] = 900
    assert engine.log_partition({'y1': 62}) == pytest.approx(expected, abs=1e-11)
    contradictory = replace(spec, onset_counts=(CountRule((1, 2), 1), CountRule((4, 5), 0)))
    zero_engine = StreamingTemplateMusicInference(contradictory, np.full((spec.length, 130), -np.log(130)))
    assert zero_engine.log_partition() == -np.inf
    with pytest.raises(ZeroMass):
        zero_engine.sample_batch(['y1'], np.random.default_rng(8))
    with pytest.raises(InvalidSpecification):
        StreamingTemplateMusicInference(spec, np.zeros((spec.length, 130)))
