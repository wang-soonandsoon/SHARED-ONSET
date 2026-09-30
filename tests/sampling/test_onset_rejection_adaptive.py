"""Independent full-token target and lifecycle checks for early rejection."""
from collections import Counter
from itertools import product
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from tri.domain.music import CountRule, MusicSpec, verify_music
from tri.errors import BudgetExceeded, InvalidSpecification, VerificationError, ZeroMass
from tri.inference.exact import Budget
from tri.sampling.onset_rejection_adaptive import AdaptiveOnsetRejectionSampler


MODES = ('visible', 'boundary_early', 'visible_early')


def case(beta=.15, R=2, K=1):
    spans = tuple(tuple(range(3*r + 1, 3*r + 3)) for r in range(R))
    length = R * 3 + 1
    observed = {i: 62 for i in range(0, length, 3)}
    spec = MusicSpec(length=length, pitches=(60, 62), observed=observed,
        fixed_soundings={i: 60 for i in observed}, initial_pitch=60,
        equal_onsets=tuple((spans[0][j], s[j]) for s in spans[1:] for j in range(2)),
        onset_counts=tuple(CountRule(s, K) for s in spans), motion_cost=beta)
    probabilities = np.full((length, 130), .1 / 126)
    probabilities[:, [0, 1, 62, 64]] = [.15, .2, .25, .3]
    probabilities[2, [62, 64]] = [.35, .2]
    return spec, np.log(probabilities), spans


def oracle(spec, q):
    unknown = [i for i in range(spec.length) if i not in spec.observed]
    support = {}
    for values in product((0, 1, 62, 64), repeat=len(unknown)):
        complete = {**spec.observed, **dict(zip(unknown, values))}
        tokens = tuple(complete[i] for i in range(spec.length))
        check = verify_music(tokens, spec)
        if check.valid:
            support[tokens] = sum(q[i, complete[i]] for i in unknown) + check.soft_score
    return support


@pytest.mark.parametrize('mode', MODES)
def test_complete_joint_distribution_including_rhythm_matches_original_token_oracle(mode):
    spec, q, spans = case()
    weights = oracle(spec, q)
    z = float(logsumexp(list(weights.values())))
    sampler = AdaptiveOnsetRejectionSampler(spec, q, proposal=mode)
    rng = np.random.default_rng(3701)
    n = 1400
    counts = Counter(sampler.sample_full(rng).tokens for _ in range(n))
    assert set(counts) <= set(weights)
    # Total variation tests the whole joint, including shared onset pattern,
    # rather than checking separate note marginals alone.
    tv = .5 * sum(abs(counts[t] / n - math.exp(w-z)) for t, w in weights.items())
    assert tv < .09
    stats = sampler.last_stats
    assert stats['cumulative_accepted_samples'] == n
    assert stats['cumulative_proposals'] == stats['cumulative_rejections'] + n
    assert stats['cumulative_proposals'] == stats['cumulative_full_candidates'] + stats['cumulative_partial_candidates']
    assert stats['cumulative_patterns_sampled'] == stats['cumulative_proposal_calls']
    expected_acceptance = math.exp(z - sampler.proposal_log_z)
    assert abs(n / stats['cumulative_proposals'] - expected_acceptance) < .05
    assert not stats['target_normalizer_available']
    assert 'shared_rhythm_attempt' in stats['proposal_unit']
    for name in ('log_partition', 'sample_batch', 'log_probability'):
        assert not hasattr(sampler, name)


@pytest.mark.parametrize('mode', ('boundary_early', 'visible_early'))
def test_early_rejection_redraws_rhythm_and_discards_all_previous_spans(monkeypatch, mode):
    spec, q, spans = case()
    events, calls = [], []
    sampler = AdaptiveOnsetRejectionSampler(spec, q, proposal=mode, max_proposals=2,
                                           progress_callback=events.append)
    patterns = iter(((1, 0), (0, 1)))
    def pattern(rng):
        bits = next(patterns)
        calls.append(('pattern', bits))
        return bits
    def span(r, bits, rng):
        calls.append(('span', r, bits))
        return (64, 1) if bits == (1, 0) else (0, 62)
    monkeypatch.setattr(sampler.proposal_engine, 'sample_pattern', pattern)
    monkeypatch.setattr(sampler.proposal_engine, 'sample_span', span)
    monkeypatch.setattr(sampler.proposal_engine, 'span_residual_score',
                        lambda r, tokens: -1000. if tokens == (64, 1) else 0.)
    draw = sampler.sample_full(np.random.default_rng(0))
    assert calls == [('pattern', (1, 0)), ('span', 0, (1, 0)),
                     ('pattern', (0, 1)), ('span', 0, (0, 1)), ('span', 1, (0, 1))]
    assert verify_music(draw.tokens, spec).valid
    stats = draw.diagnostics
    assert (stats['proposals'], stats['partial_candidates'], stats['full_candidates'], stats['spans_sampled']) == (2, 1, 1, 3)
    assert all(e['in_flight'] == (e['event'] != 'proposal_completed') for e in events)
    assert events[-1]['cumulative_accepted_samples'] == 1 and not events[-1]['in_flight']
    assert all('tokens' not in e['attempt'] and 'span_decisions' not in e['attempt'] for e in events)


def test_limit_exhaustion_and_later_call_keep_partial_costs_without_returning_candidate(monkeypatch):
    spec, q, _ = case()
    sampler = AdaptiveOnsetRejectionSampler(spec, q, proposal='boundary_early', max_proposals=2)
    monkeypatch.setattr(sampler.proposal_engine, 'span_residual_score', lambda r, tokens: -1000.)
    with pytest.raises(BudgetExceeded) as failure:
        sampler.sample_full(np.random.default_rng(0))
    stats = failure.value.diagnostics
    assert stats['accepted'] == stats['full_candidates'] == 0
    assert stats['proposals'] == stats['partial_candidates'] == stats['rejections'] == 2
    assert stats['spans_sampled'] == 2 and stats['failed_phase'] == 'proposal_limit'
    assert not hasattr(failure.value, 'tokens')


def test_hard_stop_inside_later_span_preserves_partial_inflight_progress():
    spec, q, _ = case(beta=0.)
    events = []
    class HardStop(BaseException):
        pass
    def callback(event):
        events.append(event)
        if event['event'] == 'span_started' and event['attempt']['span_index'] == 1:
            raise HardStop()
    sampler = AdaptiveOnsetRejectionSampler(spec, q, proposal='visible_early', progress_callback=callback)
    with pytest.raises(HardStop):
        sampler.sample_full(np.random.default_rng(10))
    event = events[-1]
    assert event['in_flight'] and event['patterns_sampled'] == event['spans_sampled'] == 1
    assert event['proposals'] == event['full_candidates'] == event['partial_candidates'] == 0
    assert event['cumulative_accepted_samples'] == 0


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('K', [0, 2])
def test_zero_beta_degenerate_counts_and_three_spans_are_exact(mode, K):
    spec, q, _ = case(beta=0., R=3, K=K)
    sampler = AdaptiveOnsetRejectionSampler(spec, q, proposal=mode)
    for seed in range(4):
        draw = sampler.sample_full(np.random.default_rng(seed))
        assert verify_music(draw.tokens, spec).valid
        assert draw.diagnostics['proposals'] == 1
        assert draw.diagnostics['spans_sampled'] == 3
        assert draw.diagnostics['attempts'][0]['log_acceptance'] == 0.


@pytest.mark.parametrize('mode', MODES)
def test_callback_observations_do_not_change_samples_or_rng(mode):
    spec, q, _ = case()
    events = []
    a = AdaptiveOnsetRejectionSampler(spec, q, proposal=mode)
    b = AdaptiveOnsetRejectionSampler(spec, q, proposal=mode, progress_callback=events.append)
    ar, br = np.random.default_rng(49), np.random.default_rng(49)
    assert a.sample_full(ar).tokens == b.sample_full(br).tokens
    assert ar.random() == br.random()
    assert events[-1]['event'] == 'proposal_completed'


def test_full_acceptance_cross_checks_sum_of_span_residuals(monkeypatch):
    spec, q, _ = case(beta=1.)
    sampler = AdaptiveOnsetRejectionSampler(spec, q, proposal='boundary_early')
    monkeypatch.setattr(sampler.proposal_engine, 'sample_pattern', lambda rng: (1, 0))
    monkeypatch.setattr(sampler.proposal_engine, 'sample_span', lambda r, bits, rng: (64, 1))
    monkeypatch.setattr(sampler.proposal_engine, 'span_residual_score', lambda r, tokens: 0.)
    with pytest.raises(VerificationError, match='sum of span') as caught:
        sampler.sample_full(np.random.default_rng(9))
    assert caught.value.diagnostics['accepted'] == caught.value.diagnostics['proposals'] == 0
    assert caught.value.diagnostics['full_candidates'] == 1


def test_positive_span_residual_fails_without_accepting(monkeypatch):
    spec, q, _ = case()
    sampler = AdaptiveOnsetRejectionSampler(spec, q, proposal='visible_early')
    monkeypatch.setattr(sampler.proposal_engine, 'span_residual_score', lambda r, tokens: 1.)
    with pytest.raises(VerificationError, match='nonpositive') as caught:
        sampler.sample_full(np.random.default_rng(9))
    assert caught.value.diagnostics['proposals'] == caught.value.diagnostics['accepted'] == 0


def test_diagnostic_budget_fails_before_drawing_another_rhythm():
    spec, q, _ = case()
    sampler = AdaptiveOnsetRejectionSampler(spec, q, proposal='visible_early')
    sampler.budget = Budget(max_workspace_bytes=sampler._proposal_workspace_bytes + sampler._diagnostic_bytes(0))
    with pytest.raises(BudgetExceeded) as caught:
        sampler.sample_full(np.random.default_rng(0))
    assert caught.value.diagnostics['failed_phase'] == 'diagnostic_workspace'
    assert caught.value.diagnostics['proposal_calls'] == 0


@pytest.mark.parametrize('mode', MODES)
def test_zero_mass_preserved_and_not_misreported_as_sampling_limit(mode):
    spec, q, _ = case()
    q[:] = -math.inf
    q[:, 129] = 0.
    with pytest.raises(ZeroMass) as caught:
        AdaptiveOnsetRejectionSampler(spec, q, proposal=mode)
    assert caught.value.diagnostics['status'] == 'zero_mass'


@pytest.mark.parametrize('mode', [None, 'drop', 'boundary'])
def test_invalid_mode_raises_original_validation_error(mode):
    spec, q, _ = case()
    with pytest.raises(InvalidSpecification):
        AdaptiveOnsetRejectionSampler(spec, q, proposal=mode)
