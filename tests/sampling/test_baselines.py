"""Independent baseline semantics and shared-provider comparison checks."""

from dataclasses import fields, replace
from itertools import product
import math

import numpy as np
import pytest

from tri.domain.music import HOLD, REST, CountRule, MusicSpec, verify_music
from tri.errors import BudgetExceeded, InvalidSpecification, VerificationError, ZeroMass
from tri.inference.exact import Budget
from tri.sampling.baselines import METHODS, METHOD_DESCRIPTIONS, decode_method


class CountingProvider:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)
        self.calls = []

    def __call__(self, state, noise):
        self.calls.append((state, noise))
        return self.probabilities


def uniform(length):
    return np.full((length, 130), -np.log(130))


def masses_from_original_tokens(spec, log_q):
    vocabulary = (0, 1) + tuple(p + 2 for p in spec.pitches)
    choices = [(spec.observed[i],) if i in spec.observed else vocabulary for i in range(spec.length)]
    result = {}
    for tokens in product(*choices):
        verified = verify_music(tokens, spec)
        if verified.valid:
            log_weight = verified.soft_score + sum(log_q[i, token] for i, token in enumerate(tokens) if i not in spec.observed)
            result[tokens] = math.exp(log_weight)
    return result


def test_public_methods_and_descriptions_are_precise():
    assert METHODS == ("raw_reference", "local_constraints", "one_shot_joint", "tri_direct")
    assert set(METHOD_DESCRIPTIONS) == set(METHODS)
    assert "onset count" in METHOD_DESCRIPTIONS["local_constraints"]
    assert "initial" in METHOD_DESCRIPTIONS["one_shot_joint"]
    with pytest.raises(InvalidSpecification):
        decode_method("unknown", MusicSpec(1, (60,)), CountingProvider(uniform(1)))


def test_raw_reference_keeps_full_vocab_and_returns_invalid_music():
    spec = MusicSpec(2, (60,), observed={0: REST}, fixed_soundings={0: None},
                     onset_counts=(CountRule((1,), 0),))
    q = np.full((2, 130), -np.inf)
    q[0] = np.nan  # Observed-row neural predictions are irrelevant.
    q[1, 129] = 0.0
    provider = CountingProvider(q)
    result = decode_method("raw_reference", spec, provider, steps=1)
    assert result.tokens == (REST, 129)
    assert result.model_calls == len(provider.calls) == 1
    assert not verify_music(result.tokens, spec).valid
    assert result.trace[0]["log_batch_probability"] == 0.0


@pytest.mark.parametrize("bad", ["shape", "unnormalized", "nan", "inf", "zero", "complex"])
@pytest.mark.parametrize("method", ["raw_reference", "one_shot_joint"])
def test_full_unknown_probability_rows_are_validated(method, bad):
    q = uniform(1)
    if bad == "shape":
        q = q[:, :129]
    elif bad == "unnormalized":
        q += 0.1
    elif bad == "nan":
        q[0, 4] = np.nan
    elif bad == "inf":
        q[0, 4] = np.inf
    elif bad == "zero":
        q[:] = -np.inf
    elif bad == "complex":
        q = q.astype(complex) + 1j
    with pytest.raises(InvalidSpecification):
        decode_method(method, MusicSpec(1, (60,)), CountingProvider(q))


def test_one_shot_target_is_original_q0_times_constraints_and_soft_score():
    spec = MusicSpec(3, (60, 64), initial_pitch=60, observed={1: HOLD},
                     fixed_soundings={1: 60}, equal_onsets=((0, 2),), motion_cost=.21)
    rng = np.random.default_rng(779)
    log_q = np.log(rng.dirichlet(np.ones(130), size=3))
    log_q[1] = np.nan
    masses = masses_from_original_tokens(spec, log_q)
    total = sum(masses.values())
    for seed in range(20):
        provider = CountingProvider(log_q)
        result = decode_method("one_shot_joint", spec, provider, steps=7, seed=seed)
        assert result.model_calls == len(provider.calls) == 1
        assert provider.calls == [((None, HOLD, None), 1.0)]
        assert result.tokens in masses
        trace = result.trace[0]
        assert trace["positions"] == (0, 2)
        assert trace["log_z"] == pytest.approx(math.log(total), abs=1e-12)
        assert trace["log_z_clamped"] == pytest.approx(math.log(masses[result.tokens]), abs=1e-12)
        assert trace["log_batch_probability"] == pytest.approx(math.log(masses[result.tokens] / total), abs=1e-12)
        assert trace["log_q_batch"] == pytest.approx(sum(log_q[i, result.tokens[i]] for i in (0, 2)), abs=1e-12)


def test_local_removes_only_relations_and_counts_preserving_fixed_hold_audio():
    spec = MusicSpec(2, (60, 64), observed={1: HOLD}, fixed_soundings={1: 60},
                     equal_onsets=((0, 1),), onset_counts=(CountRule((0,), 0),),
                     pitch_ranges={0: (60, 64)}, pitch_classes={1: (0,)},
                     max_adjacent_interval=4, motion_cost=.37)
    expected = replace(spec, equal_onsets=(), onset_counts=())
    # Expected spec differs in exactly these two public fields.
    for field in fields(spec):
        if field.name not in ("equal_onsets", "onset_counts"):
            assert getattr(expected, field.name) == getattr(spec, field.name)
    result = decode_method("local_constraints", spec, CountingProvider(uniform(2)), steps=1)
    assert result.tokens == (62, HOLD)
    assert verify_music(result.tokens, expected).valid
    assert not verify_music(result.tokens, spec).valid
    for method in ("tri_direct", "one_shot_joint"):
        with pytest.raises(ZeroMass):
            decode_method(method, spec, CountingProvider(uniform(2)), steps=1)


def test_local_soft_motion_and_range_mass_matches_original_enumeration():
    spec = MusicSpec(3, (60, 64), initial_pitch=60, observed={2: REST},
                     fixed_soundings={2: None}, equal_onsets=((0, 1),),
                     onset_counts=(CountRule((0, 1), 2),), pitch_ranges={0: (60, 60)},
                     pitch_classes={1: (0, 4)}, max_adjacent_interval=4, motion_cost=.4)
    local = replace(spec, equal_onsets=(), onset_counts=())
    q = uniform(3)
    masses = masses_from_original_tokens(local, q)
    result = decode_method("local_constraints", spec, CountingProvider(q), steps=1, seed=62)
    assert result.trace[0]["log_z"] == pytest.approx(math.log(sum(masses.values())), abs=1e-12)
    assert result.trace[0]["log_batch_probability"] == pytest.approx(math.log(masses[result.tokens] / sum(masses.values())), abs=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_complete_input_uses_no_model_calls(method):
    spec = MusicSpec(2, (60,), observed={0: 62, 1: HOLD}, fixed_soundings={0: 60, 1: 60})
    def forbidden_provider(state, noise):
        raise AssertionError("Complete input must not call the model")
    result = decode_method(method, spec, forbidden_provider)
    assert result.tokens == (62, HOLD)
    assert result.model_calls == 0
    assert result.trace == ()


def test_invalid_complete_input_is_returned_only_by_raw_control():
    spec = MusicSpec(1, (60,), observed={0: HOLD})
    provider = CountingProvider(uniform(1))
    assert decode_method("raw_reference", spec, provider).tokens == (HOLD,)
    for method in ("local_constraints", "one_shot_joint", "tri_direct"):
        with pytest.raises(VerificationError):
            decode_method(method, spec, provider)
    assert provider.calls == []


def test_same_seed_gives_reference_schedule_parity_and_correct_model_counts():
    spec = MusicSpec(4, (60, 64), observed={0: REST}, fixed_soundings={0: None},
                     equal_onsets=((1, 3),), onset_counts=(CountRule((1, 2, 3), 2),))
    provider = CountingProvider(uniform(4))
    saw_empty = False
    for seed in range(12):
        paths = []
        for method in ("raw_reference", "local_constraints", "tri_direct"):
            provider.calls.clear()
            result = decode_method(method, spec, provider, steps=4, seed=seed)
            assert result.model_calls == len(provider.calls) == len(result.trace)
            assert result.tokens[0] == REST
            assert provider.calls[0][0] == (REST, None, None, None)
            for step, (state, noise) in enumerate(provider.calls):
                assert noise == 1 - step / 4
                assert state[0] == REST
            paths.append(tuple(row["positions"] for row in result.trace))
            saw_empty |= any(not row["positions"] for row in result.trace)
        assert paths[0] == paths[1] == paths[2]
    assert saw_empty


@pytest.mark.parametrize("method", ["local_constraints", "one_shot_joint", "tri_direct"])
def test_zero_model_support_and_budget_failures_stay_explicit(method):
    spec = MusicSpec(1, (60,))
    q = np.full((1, 130), -np.inf)
    q[0, 129] = 0.0  # Valid model distribution, no supported working token.
    with pytest.raises(ZeroMass):
        decode_method(method, spec, CountingProvider(q), steps=1)
    with pytest.raises(BudgetExceeded):
        decode_method(method, spec, CountingProvider(uniform(1)), steps=1,
                      budget=Budget(max_factor_entries=1))
