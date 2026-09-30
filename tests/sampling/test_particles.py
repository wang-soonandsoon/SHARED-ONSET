import math
from itertools import product

import numpy as np
import pytest
from scipy.special import logsumexp

from tri.domain.music import MusicSpec, CountRule, verify_music
from tri.errors import InvalidSpecification, ZeroMass
from tri.sampling.direct import direct_decode
from tri.sampling.particles import path_is, smc_decode, systematic_resample


def provider(state, noise):
    q = np.zeros((len(state), 130))
    visible = sum(v is not None for v in state)
    for i in range(len(state)):
        q[i, [0, 1, 62, 66]] = [0.1 + noise, 0.1, 0.3 + visible * .4, 0.8 - .5 * noise]
    q /= q.sum(1, keepdims=True)
    with np.errstate(divide='ignore'):
        return np.log(q)


def mass(spec, state, q):
    total = 0.
    for y in product(*[(v,) if v is not None else (0, 1, 62, 66) for v in state]):
        checked = verify_music(y, spec)
        if checked.valid:
            total += math.exp(checked.soft_score + sum(q[i, v] for i, v in enumerate(y) if state[i] is None))
    return total


@pytest.mark.parametrize('seed', range(5))
def test_one_particle_equals_existing_full_path_weight(seed):
    spec = MusicSpec(3, (60, 64), initial_pitch=60, equal_onsets=((0, 2),), motion_cost=.2)
    direct = direct_decode(spec, provider, steps=3, seed=seed, epsilon=.4, track_path_weights=True)
    particle = smc_decode(spec, provider, particles=1, steps=3, seed=seed, epsilon=.4)
    assert particle.tokens == direct.tokens
    assert particle.model_calls == direct.model_calls
    assert particle.log_normalizer_estimate == pytest.approx(direct.log_path_weight, abs=1e-11)
    assert particle.normalized_weights == (1.,)


def test_multi_particle_edges_weights_and_resampling_against_original_tokens():
    spec = MusicSpec(3, (60, 64), initial_pitch=60, equal_onsets=((0, 2),), motion_cost=.7)
    count, steps, seed = 9, 4, 71
    result = smc_decode(spec, provider, particles=count, steps=steps, seed=seed, epsilon=.25, ess_threshold=1.)
    states = [(None,) * 3] * count
    weights = np.full(count, -math.log(count))
    normalizer = math.log(mass(spec, states[0], provider(states[0], 1)))
    saw_resample = False
    for row in result.trace:
        increments = []
        for edge in row['edges']:
            idx = edge['particle']
            if edge.get('absorbing'):
                increments.append(0.)
                continue
            state = states[idx]
            q = provider(state, 1-row['step']/steps)
            old = mass(spec, state, q)
            updated = list(state)
            for i, v in zip(edge['positions'], edge['values']):
                updated[i] = v
            updated = tuple(updated)
            # Independent clamp sums retain q for originally missing variables.
            clamped = 0.
            for y in product(*[(v,) if v is not None else (0, 1, 62, 66) for v in state]):
                checked = verify_music(y, spec)
                if checked.valid and all(y[i] == v for i, v in zip(edge['positions'], edge['values'])):
                    clamped += math.exp(checked.soft_score+sum(q[i, v] for i, v in enumerate(y) if state[i] is None))
            next_mass = math.exp(verify_music(updated, spec).soft_score) if all(v is not None for v in updated) else mass(spec, updated, provider(updated, 1-(row['step']+1)/steps))
            assert edge['log_z'] == pytest.approx(math.log(old))
            assert edge['log_z_clamped'] == pytest.approx(math.log(clamped))
            expected = edge['log_rho_reference']-edge['log_rho_proposal']+edge['log_q_batch']+math.log(next_mass/clamped)
            assert edge['log_g'] == pytest.approx(expected)
            states[idx] = updated
            increments.append(expected)
        delta = logsumexp(weights+increments)
        normalizer += delta
        weights = weights+increments-delta
        assert row['log_normalizer_estimate'] == pytest.approx(normalizer)
        if row['resampled']:
            saw_resample = True
            states = [states[i] for i in row['parent_indices']]
            weights = np.full(count,-math.log(count))
    assert saw_resample
    assert result.log_normalizer_estimate == pytest.approx(normalizer)
    np.testing.assert_allclose(result.normalized_weights, np.exp(weights))
    assert all(verify_music(y,spec).valid for y in result.particles)


def test_resampling_has_unbiased_counts_and_zero_weights_get_no_offspring():
    rng = np.random.default_rng(2)
    frequencies = np.zeros(4)
    for _ in range(4000):
        frequencies += np.bincount(systematic_resample([0., .1, .2, .7], rng),minlength=4)
    np.testing.assert_allclose(frequencies/(4000*4), [0.,.1,.2,.7], atol=.006)
    with pytest.raises(InvalidSpecification):
        systematic_resample([1, 2], rng)


def test_shared_provider_buffer_cannot_change_other_particles_old_q():
    spec=MusicSpec(3,(60,64),initial_pitch=60,equal_onsets=((0,2),),motion_cost=.5)
    buffer=np.empty((3,130))
    def reuse(state,noise):
        np.copyto(buffer,provider(state,noise))
        return buffer
    expected=smc_decode(spec,provider,particles=8,steps=4,seed=73)
    actual=smc_decode(spec,reuse,particles=8,steps=4,seed=73)
    assert actual.particles==expected.particles
    assert actual.log_normalizer_estimate==pytest.approx(expected.log_normalizer_estimate,abs=1e-12)
    np.testing.assert_allclose(actual.normalized_weights,expected.normalized_weights)


@pytest.mark.parametrize('method', [path_is, smc_decode])
def test_constant_reference_has_exact_common_normalizer_and_counted_calls(method):
    spec = MusicSpec(2,(60,64),initial_pitch=60,equal_onsets=((0,1),),motion_cost=.3)
    calls=[]
    def static(state,noise):
        calls.append((state,noise))
        return provider((None,)*2,1.)
    result=method(spec,static,particles=5,steps=3,seed=51)
    z=mass(spec,(None,)*2,provider((None,)*2,1.))
    assert result.log_normalizer_estimate == pytest.approx(math.log(z),abs=1e-11)
    assert result.ess == pytest.approx(5)
    assert result.model_calls == len(calls)
    assert verify_music(result.tokens,spec).valid


@pytest.mark.parametrize('method', [path_is, smc_decode])
def test_all_zero_mass_has_no_uniform_fallback(method):
    spec=MusicSpec(1,(60,),pitch_classes={0:()},onset_counts=(CountRule((0,),1),))
    with pytest.raises(ZeroMass) as caught:
        method(spec,provider,particles=4)
    assert 'zero' in str(caught.value).lower() or 'count' in str(caught.value).lower()
    with pytest.raises(InvalidSpecification):
        method(MusicSpec(1,(60,)),provider,particles=0)
