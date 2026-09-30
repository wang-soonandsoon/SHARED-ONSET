"""One-shot integration: one frozen model call, no invented target Z."""
from dataclasses import replace

import numpy as np
import pytest

from tri.domain.music import CountRule, MusicSpec, verify_music
from tri.errors import BudgetExceeded, InvalidSpecification, UnsupportedSpec, ZeroMass
from tri.inference.exact import Budget
from tri.sampling.onset_rejection import OnsetRejectionSampler
from tri.sampling.research_methods import RESEARCH_METHODS, research_decode


METHODS = ('onset_rejection_drop', 'onset_rejection_boundary')
ADAPTIVE_METHODS = ('onset_rejection_visible', 'onset_rejection_boundary_early',
                    'onset_rejection_visible_early')
ALL_METHODS = METHODS + ADAPTIVE_METHODS


def tiny_spec(beta=.2):
    return MusicSpec(5, (60, 62), observed={0: 62, 2: 62, 4: 62},
                     fixed_soundings={0: 60, 2: 60, 4: 60},
                     equal_onsets=((1, 3),),
                     onset_counts=(CountRule((1,), 1), CountRule((3,), 1)),
                     motion_cost=beta)


class Provider:
    def __init__(self, spec, *, zero_support=False):
        self.calls = []
        self.q = np.full((spec.length, 130), -np.inf)
        self.q[:, 129] = 0. if zero_support else np.log(.6)
        if not zero_support:
            self.q[:, 62] = self.q[:, 64] = np.log(.2)
        for i in spec.observed:
            self.q[i] = np.nan  # Initial observed rows must remain deltas.

    def __call__(self, state, noise):
        self.calls.append((state, noise))
        return self.q


@pytest.mark.parametrize('method', ALL_METHODS)
def test_one_shot_rejection_freezes_one_full_q_and_never_uses_joint_api(method, monkeypatch):
    import tri.sampling.research_methods as methods
    spec = tiny_spec(beta=0)
    provider = Provider(spec)
    before = provider.q.copy()
    def forbidden(*args, **kwargs):
        raise AssertionError('rejection must not call the normalized joint or factory API')
    monkeypatch.setattr(methods, '_joint', forbidden)
    import tri.inference.music_backends as factory
    monkeypatch.setattr(factory, 'make_music_engine', forbidden)
    result = research_decode(method, spec, provider, steps=123, seed=4, max_proposals=100)
    assert result.model_calls == len(provider.calls) == 1
    assert provider.calls == [((62, None, 62, None, 62), 1.0)]
    assert np.array_equal(provider.q, before, equal_nan=True)
    assert verify_music(result.tokens, spec).valid
    assert result.diagnostics['proposals'] == result.diagnostics['accepted'] == 1
    # 60% outside the working vocabulary is not renormalized away.
    assert result.diagnostics['proposal_log_z'] == pytest.approx(2 * np.log(.4), abs=1e-12)
    assert result.diagnostics['target_normalizer_available'] is False
    for name in ('log_z', 'log_probability', 'log_sample_probability', 'log_normalizer_estimate'):
        assert name not in result.diagnostics
    assert result.trace == ()


@pytest.mark.parametrize('method', ALL_METHODS)
def test_complete_context_skips_model_and_proposals_but_checks_original_rules(method):
    spec = tiny_spec()
    complete = replace(spec, observed={0: 62, 1: 64, 2: 62, 3: 64, 4: 62},
                       fixed_soundings={0: 60, 1: 62, 2: 60, 3: 62, 4: 60})
    def forbidden(*args):
        raise AssertionError('fully observed input needs no model call')
    result = research_decode(method, complete, forbidden)
    assert result.tokens == (62, 64, 62, 64, 62)
    assert result.model_calls == 0
    assert result.diagnostics['complete_observed']
    assert result.diagnostics['proposals'] == 0
    bad = replace(complete, onset_counts=(CountRule((1,), 0), CountRule((3,), 0)))
    with pytest.raises(ZeroMass):
        research_decode(method, bad, forbidden)


@pytest.mark.parametrize('method', ALL_METHODS)
def test_hard_interval_is_unsupported_before_model_call(method):
    spec = replace(tiny_spec(), max_adjacent_interval=2)
    provider = Provider(spec)
    with pytest.raises(UnsupportedSpec):
        research_decode(method, spec, provider)
    assert provider.calls == []
    with pytest.raises(UnsupportedSpec) as raised:
        OnsetRejectionSampler(spec, provider.q)
    assert raised.value.diagnostics['status'] == 'unsupported'


@pytest.mark.parametrize('limit', (0, -1, True, 2.5, '2'))
def test_invalid_proposal_limit_does_not_call_model_or_start_precomputation(limit):
    spec = tiny_spec()
    provider = Provider(spec)
    with pytest.raises(InvalidSpecification):
        research_decode(METHODS[0], spec, provider, max_proposals=limit)
    assert provider.calls == []
    with pytest.raises(InvalidSpecification) as raised:
        OnsetRejectionSampler(spec, provider.q, max_proposals=limit)
    assert raised.value.diagnostics['failed_phase'] == 'validate_specification'


@pytest.mark.parametrize('method', ALL_METHODS)
def test_zero_support_is_not_reported_as_a_failed_random_acceptance(method):
    spec = tiny_spec()
    provider = Provider(spec, zero_support=True)
    with pytest.raises(ZeroMass) as raised:
        research_decode(method, spec, provider)
    assert len(provider.calls) == 1
    diag = raised.value.diagnostics
    assert diag['status'] == 'zero_mass'
    assert diag['proposals'] == diag['proposal_calls'] == 0
    assert diag['target_normalizer_available'] is False


def test_proposal_precomputation_budget_failure_retains_diagnostics():
    spec = tiny_spec()
    with pytest.raises(BudgetExceeded) as raised:
        OnsetRejectionSampler(spec, Provider(spec).q, Budget(max_factor_entries=1))
    diag = raised.value.diagnostics
    assert diag['status'] == 'budget_exceeded'
    assert diag['proposals'] == 0
    assert diag['precompute_seconds'] >= 0
    assert 'tokens' not in diag


def test_diagnostic_workspace_stops_before_allocating_another_attempt(monkeypatch):
    spec = tiny_spec(beta=10)
    sampler = OnsetRejectionSampler(spec, Provider(spec).q, max_proposals=5)
    # Leave room for two per-proposal records and their snapshot copies, while
    # the numerical engine itself remains comfortably within its old budget.
    allowed = (sampler._proposal_workspace_bytes + sampler._DIAGNOSTIC_BASE_BYTES
               + 2 * sampler._DIAGNOSTIC_BYTES_PER_ATTEMPT)
    sampler.budget = Budget(max_workspace_bytes=allowed)
    monkeypatch.setattr(sampler.proposal_engine, 'sample_full', lambda rng: (62, 64, 62, 64, 62))
    with pytest.raises(BudgetExceeded) as raised:
        sampler.sample_full(np.random.default_rng(0))
    diag = raised.value.diagnostics
    assert diag['failed_phase'] == 'diagnostic_workspace'
    assert diag['proposals'] == diag['proposal_calls'] == diag['rejections'] == 2
    assert len(diag['attempts']) == 2
    assert diag['cumulative_proposals'] == 2
    assert diag['accepted'] == 0 and 'tokens' not in diag


def test_progress_retains_completed_rejection_and_marks_interrupted_proposal_in_flight(monkeypatch):
    class ExternalStop(BaseException):
        pass
    progress = []
    spec = tiny_spec(beta=10)
    sampler = OnsetRejectionSampler(spec, Provider(spec).q, max_proposals=5,
                                   progress_callback=progress.append)
    calls = 0
    def proposal(rng):
        nonlocal calls
        calls += 1
        if calls == 2:
            # A hard stop does not return through the sampler's Exception
            # handler, so its previous final draw stats cannot recover work.
            raise ExternalStop()
        return (62, 64, 62, 64, 62)
    monkeypatch.setattr(sampler.proposal_engine, 'sample_full', proposal)
    with pytest.raises(ExternalStop):
        sampler.sample_full(np.random.default_rng(0))
    assert [row['event'] for row in progress] == [
        'proposal_started', 'proposal_completed', 'proposal_started']
    completed, latest = progress[1:]
    assert completed['in_flight'] is False
    assert completed['proposals'] == completed['rejections'] == 1
    assert completed['attempt']['accepted'] is False
    assert latest['in_flight'] is True and latest['proposal_index'] == 2
    assert latest['cumulative_proposal_calls'] == 2
    assert latest['cumulative_proposals'] == latest['cumulative_rejections'] == 1
    assert latest['cumulative_accepted_samples'] == 0
    assert sampler.last_stats['status'] == 'ready'  # Only progress survived.
    assert all('tokens' not in row and 'tokens' not in row['attempt'] for row in progress)
    assert 'proposal_seconds' not in progress[0]['attempt']  # durable start snapshot


def test_progress_updates_acceptance_before_callback_and_does_not_change_rng():
    spec = tiny_spec(beta=0)
    progress = []
    observed = OnsetRejectionSampler(spec, Provider(spec).q, progress_callback=progress.append)
    plain = OnsetRejectionSampler(spec, Provider(spec).q)
    left_rng, right_rng = np.random.default_rng(33), np.random.default_rng(33)
    assert observed.sample_full(left_rng).tokens == plain.sample_full(right_rng).tokens
    assert left_rng.random() == right_rng.random()
    assert progress[-1]['event'] == 'proposal_completed' and not progress[-1]['in_flight']
    assert progress[-1]['accepted'] == progress[-1]['cumulative_accepted_samples'] == 1
    assert progress[-1]['proposals'] == 1 and progress[-1]['rejections'] == 0


def test_progress_callback_must_be_callable():
    spec = tiny_spec()
    with pytest.raises(InvalidSpecification, match='progress_callback'):
        OnsetRejectionSampler(spec, Provider(spec).q, progress_callback=42)


def test_new_methods_are_explicit_and_do_not_replace_existing_research_names():
    assert all(method in RESEARCH_METHODS for method in ALL_METHODS)
    assert {'one_shot_joint', 'tri_direct', 'smc_4', 'pooled_template'} <= set(RESEARCH_METHODS)


@pytest.mark.parametrize('method', ALL_METHODS)
def test_rejection_failure_keeps_single_model_call_and_does_not_fall_back(method, monkeypatch):
    # Isolate the integration boundary; independent sampler tests exercise the
    # actual finite rejection loop and its target distribution.
    import tri.sampling.onset_rejection as module
    import tri.sampling.onset_rejection_adaptive as adaptive_module
    seen = {}
    class ExhaustedSampler:
        def __init__(self, spec, q, budget, *, proposal, max_proposals):
            seen.update(q=q, proposal=proposal, limit=max_proposals)
        def sample_full(self, rng):
            error = BudgetExceeded('no accepted sample')
            error.diagnostics = {'status': 'budget_exceeded', 'proposals': 3, 'accepted': 0}
            raise error
    if method in ADAPTIVE_METHODS:
        monkeypatch.setattr(adaptive_module, 'AdaptiveOnsetRejectionSampler', ExhaustedSampler)
    else:
        monkeypatch.setattr(module, 'OnsetRejectionSampler', ExhaustedSampler)
    spec = tiny_spec()
    provider = Provider(spec)
    with pytest.raises(BudgetExceeded) as raised:
        research_decode(method, spec, provider, max_proposals=3)
    assert len(provider.calls) == 1
    assert seen == {'q': provider.q, 'proposal': method.removeprefix('onset_rejection_'), 'limit': 3}
    assert raised.value.diagnostics['accepted'] == 0


@pytest.mark.parametrize('method', ADAPTIVE_METHODS)
def test_adaptive_entry_matches_direct_sampler_and_keeps_shared_rhythm_units(method):
    from tri.sampling.onset_rejection_adaptive import AdaptiveOnsetRejectionSampler
    spec, seed = tiny_spec(beta=.4), 197
    provider = Provider(spec)
    proposal = method.removeprefix('onset_rejection_')
    expected = AdaptiveOnsetRejectionSampler(spec, provider.q, proposal=proposal).sample_full(
        np.random.default_rng(seed))
    actual = research_decode(method, spec, provider, seed=seed, steps=999, backend='unused')
    assert actual.tokens == expected.tokens
    assert actual.model_calls == len(provider.calls) == 1
    assert actual.diagnostics['proposal_unit'] == expected.diagnostics['proposal_unit']
    assert actual.diagnostics['proposals'] == expected.diagnostics['proposals']
    assert actual.diagnostics['spans_sampled'] == expected.diagnostics['spans_sampled']


@pytest.mark.parametrize('method', ADAPTIVE_METHODS)
def test_adaptive_invalid_limit_fails_before_single_provider_call(method):
    provider = Provider(tiny_spec())
    with pytest.raises(InvalidSpecification, match='max_proposals'):
        research_decode(method, tiny_spec(), provider, max_proposals=0)
    assert provider.calls == []
