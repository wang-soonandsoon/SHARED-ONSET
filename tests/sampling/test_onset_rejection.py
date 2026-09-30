"""Original-token correctness checks for complete soft-target rejection draws."""
from collections import Counter
from dataclasses import replace
from itertools import product
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from tri.domain.music import CountRule, MusicSpec, verify_music
from tri.errors import BudgetExceeded, VerificationError, ZeroMass
from tri.sampling.onset_rejection import OnsetRejectionSampler


def rejection_case(beta=.3, *, fixed_external_cost=False):
    if fixed_external_cost:
        spans = ((2,), (5,))
        observed = {0: 62, 1: 64, 3: 62, 4: 64, 6: 62, 7: 64}
        length, initial = 8, 58
    else:
        spans = ((1,), (3,))
        observed = {0: 62, 2: 62, 4: 62}
        length, initial = 5, None
    spec = MusicSpec(length=length, pitches=(60, 62), observed=observed,
                     fixed_soundings={i: value - 2 for i, value in observed.items()},
                     initial_pitch=initial, equal_onsets=((spans[0][0], spans[1][0]),),
                     onset_counts=tuple(CountRule(span, 1) for span in spans), motion_cost=beta)
    probabilities = np.full((length, 130), .2 / 126)
    probabilities[:, [0, 1, 62, 64]] = [.1, .1, .3, .3]
    return spec, np.log(probabilities), spans


def independent_keep(tokens, spec, spans):
    inside = {i for span in spans for i in span}
    previous, score = spec.initial_pitch, 0.
    for i, token in enumerate(tokens):
        if token == 0:
            previous = None
        elif token >= 2:
            pitch = token - 2
            if previous is not None and i not in inside:
                score -= spec.motion_cost * abs(previous - pitch)
            previous = pitch
    return score


def oracle(spec, q, spans):
    unknown = [i for i in range(spec.length) if i not in spec.observed]
    result = {}
    for assignment in product((0, 1, 62, 64), repeat=len(unknown)):
        values = {**spec.observed, **dict(zip(unknown, assignment))}
        tokens = tuple(values[i] for i in range(spec.length))
        checked = verify_music(tokens, spec)
        if not checked.valid:
            continue
        q_weight = sum(float(q[i, values[i]]) for i in unknown)
        if math.isfinite(q_weight):
            result[tokens] = {'log_q': q_weight, 'soft_score': checked.soft_score,
                              'keep': independent_keep(tokens, spec, spans)}
    return result


@pytest.mark.parametrize('proposal', ['drop', 'boundary'])
@pytest.mark.parametrize('external', [False, True])
def test_proposal_normalizer_and_residual_recover_exact_target_mass(proposal, external):
    spec, q, spans = rejection_case(fixed_external_cost=external)
    support = oracle(spec, q, spans)
    sampler = OnsetRejectionSampler(spec, q, proposal=proposal)
    target_z = float(logsumexp([row['log_q'] + row['soft_score'] for row in support.values()]))
    proposal_z = float(logsumexp([row['log_q'] + (row['keep'] if proposal == 'boundary' else 0.)
                                 for row in support.values()]))
    assert sampler.proposal_log_z == pytest.approx(proposal_z, abs=2e-12)
    accepted_mass = 0.
    for tokens, row in support.items():
        expected_keep = row['keep'] if proposal == 'boundary' else 0.
        retained = sampler.proposal_engine.retained_soft_score(tokens)
        assert retained == pytest.approx(expected_keep, abs=2e-12)
        residual = row['soft_score'] - retained
        assert residual <= 0.
        accepted_mass += math.exp(row['log_q'] + retained - proposal_z) * math.exp(residual)
    assert accepted_mass == pytest.approx(math.exp(target_z - proposal_z), abs=2e-12)
    assert not sampler.last_stats['target_normalizer_available']
    for name in ('log_partition', 'log_probability', 'sample_batch'):
        assert not hasattr(sampler, name)
    assert 'target_log_z' not in sampler.last_stats and 'log_probability' not in sampler.last_stats


@pytest.mark.parametrize('proposal', ['drop', 'boundary'])
def test_accepted_complete_joint_distribution_matches_independent_original_target(proposal):
    spec, q, spans = rejection_case()
    support = oracle(spec, q, spans)
    assert len(support) == 4
    target_z = float(logsumexp([v['log_q'] + v['soft_score'] for v in support.values()]))
    sampler = OnsetRejectionSampler(spec, q, proposal=proposal)
    rng = np.random.default_rng(2904)
    observed = Counter()
    for _ in range(1800):
        result = sampler.sample_full(rng)
        assert result.tokens in support
        assert result.diagnostics['status'] == 'success'
        assert result.diagnostics['accepted'] == 1
        assert result.diagnostics['proposals'] == result.diagnostics['rejections'] + 1
        observed[result.tokens] += 1
    assert set(observed) == set(support)
    for tokens, weight in support.items():
        probability = math.exp(weight['log_q'] + weight['soft_score'] - target_z)
        assert abs(observed[tokens] / 1800 - probability) < .05
    stats = sampler.last_stats
    assert stats['cumulative_accepted_samples'] == 1800
    expected_acceptance = math.exp(target_z - sampler.proposal_log_z)
    assert abs(1800 / stats['cumulative_proposals'] - expected_acceptance) < .04
    assert stats['cumulative_rejections'] == stats['cumulative_proposals'] - 1800


@pytest.mark.parametrize('proposal', ['drop', 'boundary'])
def test_acceptance_gate_uses_residual_once_and_preserves_failed_attempt_accounting(monkeypatch, proposal):
    spec, q, spans = rejection_case()
    sampler = OnsetRejectionSampler(spec, q, proposal=proposal, max_proposals=2)
    bad = (62, 64, 62, 64, 62)
    good = (62,) * 5
    candidates = iter((bad, good))
    monkeypatch.setattr(sampler.proposal_engine, 'sample_full', lambda rng: next(candidates))
    # This proposal stub consumes no randomness: U0=.6369 rejects the costly
    # first path for either proposal; U1 accepts the zero-residual path.
    result = sampler.sample_full(np.random.default_rng(0))
    assert result.tokens == good
    diagnostics = result.diagnostics
    assert (diagnostics['proposal_calls'], diagnostics['proposals'], diagnostics['rejections'], diagnostics['accepted']) == (2, 2, 1, 1)
    for tokens, attempt in zip((bad, good), diagnostics['attempts']):
        total = verify_music(tokens, spec).soft_score
        retained = independent_keep(tokens, spec, spans) if proposal == 'boundary' else 0.
        assert attempt['original_soft_score'] == pytest.approx(total)
        assert attempt['retained_soft_score'] == pytest.approx(retained)
        assert attempt['log_acceptance'] == pytest.approx(total - retained)
        assert 'tokens' not in attempt
    assert diagnostics['attempts'][0]['accepted'] is False
    assert diagnostics['attempts'][1]['accepted'] is True
    assert diagnostics['total_seconds'] >= diagnostics['precompute_seconds']
    assert diagnostics['sampling_seconds'] >= diagnostics['proposal_seconds'] + diagnostics['verification_seconds'] + diagnostics['score_acceptance_seconds']


def test_proposal_limit_returns_no_unaccepted_path_and_keeps_attempts_and_cumulative_cost(monkeypatch):
    spec, q, spans = rejection_case(beta=10.)
    sampler = OnsetRejectionSampler(spec, q, proposal='boundary', max_proposals=3)
    monkeypatch.setattr(sampler.proposal_engine, 'sample_full', lambda rng: (62, 64, 62, 64, 62))
    with pytest.raises(BudgetExceeded) as caught:
        sampler.sample_full(np.random.default_rng(0))
    failed = caught.value.diagnostics
    assert failed == sampler.last_stats
    assert failed['status'] == 'budget_exceeded'
    assert (failed['proposals'], failed['rejections'], failed['accepted']) == (3, 3, 0)
    assert failed['cumulative_proposals'] == 3
    assert not hasattr(caught.value, 'tokens')
    assert all('tokens' not in attempt for attempt in failed['attempts'])
    monkeypatch.setattr(sampler.proposal_engine, 'sample_full', lambda rng: (62,) * 5)
    result = sampler.sample_full(np.random.default_rng(1))
    assert result.tokens == (62,) * 5
    assert result.diagnostics['proposals'] == 1  # max_proposals is per call
    assert result.diagnostics['cumulative_proposals'] == 4
    assert result.diagnostics['cumulative_rejections'] == 3
    assert result.diagnostics['cumulative_accepted_samples'] == 1
    assert failed['cumulative_proposals'] == 3  # durable snapshot, not mutated


def test_invalid_proposal_path_and_positive_residual_are_errors_not_retryable_samples(monkeypatch):
    spec, q, spans = rejection_case()
    sampler = OnsetRejectionSampler(spec, q)
    monkeypatch.setattr(sampler.proposal_engine, 'sample_full', lambda rng: (1,) * 5)
    with pytest.raises(VerificationError) as invalid:
        sampler.sample_full(np.random.default_rng(0))
    assert invalid.value.diagnostics['proposal_calls'] == 1
    assert invalid.value.diagnostics['accepted'] == 0
    monkeypatch.setattr(sampler.proposal_engine, 'sample_full', lambda rng: (62,) * 5)
    monkeypatch.setattr(sampler.proposal_engine, 'retained_soft_score', lambda tokens: -1.)
    with pytest.raises(VerificationError) as residual:
        sampler.sample_full(np.random.default_rng(0))
    assert residual.value.diagnostics['failed_phase'] == 'score_and_accept'
    assert residual.value.diagnostics['accepted'] == 0


def test_proposal_sampling_budget_failure_is_not_counted_as_a_complete_candidate(monkeypatch):
    spec, q, _ = rejection_case()
    sampler = OnsetRejectionSampler(spec, q)
    def fail(rng):
        raise BudgetExceeded('proposal chain workspace exhausted')
    monkeypatch.setattr(sampler.proposal_engine, 'sample_full', fail)
    with pytest.raises(BudgetExceeded) as caught:
        sampler.sample_full(np.random.default_rng(0))
    diagnostics = caught.value.diagnostics
    assert diagnostics['proposal_calls'] == 1
    assert diagnostics['proposals'] == diagnostics['rejections'] == diagnostics['accepted'] == 0
    assert diagnostics['failed_phase'] == 'proposal_sampling'


def test_exact_zero_mass_is_reported_at_construction_and_is_not_a_rejection_timeout():
    spec, q, _ = rejection_case()
    q[:] = -math.inf
    q[:, 129] = 0.
    with pytest.raises(ZeroMass) as caught:
        OnsetRejectionSampler(spec, q)
    assert caught.value.diagnostics['status'] == 'zero_mass'
    assert caught.value.diagnostics['proposals'] == 0
    assert caught.value.diagnostics['proposal_log_z'] == -math.inf


def test_zero_beta_accepts_every_proposal_and_does_not_invent_a_soft_target_z():
    spec, q, _ = rejection_case(beta=0.)
    for proposal in ('drop', 'boundary'):
        sampler = OnsetRejectionSampler(spec, q, proposal=proposal, max_proposals=1)
        rng = np.random.default_rng(9)
        for _ in range(10):
            draw = sampler.sample_full(rng)
            assert draw.diagnostics['proposals'] == draw.diagnostics['accepted'] == 1
            assert draw.diagnostics['attempts'][0]['log_acceptance'] == 0.
            assert 'log_probability' not in draw.diagnostics
