from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from scipy.special import logsumexp

from tri.domain.music import CountRule, MusicSpec, verify_music
from tri.errors import BudgetExceeded, UnsupportedSpec, ZeroMass
from tri.inference.exact import Budget
from tri.inference.music_backends import make_music_engine


def brute(spec, q, evidence=None):
    evidence = evidence or {}
    missing = [i for i in range(spec.length) if i not in spec.observed]
    weights, sequences = [], []
    for values in product((0, 1) + tuple(p+2 for p in spec.pitches), repeat=len(missing)):
        tokens = [spec.observed.get(i) for i in range(spec.length)]
        for i, value in zip(missing, values):
            tokens[i] = value
        if any(tokens[int(k[1:])] != v for k, v in evidence.items()):
            continue
        check = verify_music(tokens, spec)
        if check.valid:
            weights.append(sum(q[i, tokens[i]] for i in missing) + check.soft_score)
            sequences.append(tuple(tokens))
    return float(logsumexp(weights)), sequences


@pytest.mark.parametrize('seed', range(36))
def test_paired_matches_independent_original_token_enumeration(seed):
    rng = np.random.default_rng(seed)
    observed = {0: 62, 3: 64, 4: 62, 7: 1 if seed % 3 == 0 else 64}
    anchors = {0: 60, 3: 62, 4: 60, 7: 62}
    if seed % 4 == 0:
        observed[1] = 0
        anchors[1] = None
    spec = MusicSpec(length=8, pitches=(60, 62), observed=observed, fixed_soundings=anchors,
                     equal_onsets=((1,5),(2,6)), onset_counts=(CountRule((1,2), seed % 3), CountRule((5,6), seed % 3)),
                     pitch_classes={2:(0,)} if seed % 2 else {},
                     max_adjacent_interval=1 if seed % 5 == 0 else None,
                     motion_cost=.17, enforce_end=True, end_pitch=62)
    q = rng.dirichlet(np.ones(130), size=8)
    q[:, 129] += 5
    q /= q.sum(axis=1, keepdims=True)
    q = np.log(q)
    paired = make_music_engine(spec, q, backend='paired')
    expected, sequences = brute(spec,q)
    assert paired.log_partition() == pytest.approx(expected, abs=1e-11)
    for backend in ['template','automaton']:
        assert make_music_engine(spec,q,backend=backend).log_partition() == pytest.approx(expected, abs=1e-11)
    if not sequences:
        with pytest.raises(ZeroMass):
            paired.sample_batch(['y2'], rng)
        return
    chosen = sequences[0]
    clamp = {'y2': chosen[2]}
    clamped, _ = brute(spec,q,clamp)
    assert paired.log_clamped_partition(clamp) == pytest.approx(clamped, abs=1e-11)
    draw = paired.sample_batch(['y2','y5'], rng, clamp)
    reference, _ = brute(spec,q,{**clamp, **draw.assignment})
    assert draw.log_probability == pytest.approx(reference-clamped, abs=1e-11)
    all_draw = paired.sample_batch([f'y{i}' for i in range(8)], rng)
    assert verify_music(tuple(all_draw.assignment[f'y{i}'] for i in range(8)),spec).valid


def long_spec(width=32):
    length=4*width
    a=tuple(range(width,2*width)); b=tuple(range(3*width-1,4*width-1))
    observed={i:62 for i in range(length) if i not in a+b}
    return MusicSpec(length=length,pitches=(60,62),observed=observed,
                     fixed_soundings={i:60 for i in observed},equal_onsets=tuple(zip(a,b)),
                     onset_counts=(CountRule(a,width//4),CountRule(b,width//4)),motion_cost=.03)


def test_two_bar_gaps_can_sample_without_enumerating_templates():
    spec=long_spec(); q=np.full((spec.length,130),-np.log(130))
    engine=make_music_engine(spec,q,backend='paired')
    assert np.isfinite(engine.log_partition())
    sampled=engine.sample_batch([f'y{i}' for i in range(spec.length)],np.random.default_rng(9))
    assert verify_music(tuple(sampled.assignment[f'y{i}'] for i in range(spec.length)),spec).valid
    assert engine.planning_stats['state_entries'] == 81


def test_paired_keeps_tiny_clamped_q_and_rejects_unsupported_or_over_budget():
    spec=long_spec(2); q=np.full((spec.length,130),-np.log(130))
    spec=replace(spec,onset_counts=())
    q[2] = -1000
    q[2,129] = 0
    engine=make_music_engine(spec,q,backend='paired')
    assert np.isfinite(engine.log_partition())
    assert engine.log_clamped_partition({'y2':62}) < -1000
    with pytest.raises(UnsupportedSpec):
        make_music_engine(replace(spec,fixed_soundings={}),q,backend='paired')
    with pytest.raises(BudgetExceeded):
        make_music_engine(spec,q,backend='paired',budget=Budget(max_factor_entries=1))


def test_singleton_template_flags_and_conflicting_counts():
    spec=long_spec(2); q=np.full((spec.length,130),-np.log(130))
    spec=replace(spec,onset_counts=(CountRule(spec.equal_onsets[0],1),))
    with pytest.raises(UnsupportedSpec):
        make_music_engine(spec,q,backend='paired')
    spec=long_spec(2)
    spec=replace(spec,onset_counts=(CountRule((2,3),1),CountRule((5,6),0)))
    assert make_music_engine(spec,q,backend='paired').log_partition() == -np.inf


def test_carried_pitch_outside_note_vocabulary_and_singleton_flags():
    spec=MusicSpec(length=7,pitches=(60,),initial_pitch=55,
        observed={0:1,3:1,6:1},fixed_soundings={0:55,3:55,6:55},
        equal_onsets=((1,4),(2,5)),onset_counts=(CountRule((1,),0),CountRule((2,),0)))
    q=np.full((7,130),-np.log(130));engine=make_music_engine(spec,q,backend='paired')
    expected,_=brute(spec,q)
    assert engine.log_partition()==pytest.approx(expected)
    draw=engine.sample_batch([f'y{i}' for i in range(7)],np.random.default_rng(0))
    assert list(draw.assignment.values())==[1]*7


def test_paired_empirical_draws_match_full_joint_not_product_of_marginals():
    spec=MusicSpec(length=5,pitches=(60,),observed={0:62,2:62,4:62},
        fixed_soundings={0:60,2:60,4:60},equal_onsets=((1,3),))
    q=np.full((5,130),1/130)
    q[1]=.05/127;q[1,:2]=.2;q[1,62]=.55
    q[3]=.15/127;q[3,:2]=.1;q[3,62]=.65;q=np.log(q)
    z,sequences=brute(spec,q);expected=sum(np.exp(sum(q[i,s[i]] for i in (1,3))-z) for s in sequences if s[1]>=2)
    engine=make_music_engine(spec,q,backend='paired');rng=np.random.default_rng(42);bits=[]
    for _ in range(700):
        values=engine.sample_batch(['y1','y3'],rng).assignment
        assert (values['y1']>=2)==(values['y3']>=2)
        bits.append(values['y1']>=2)
    assert abs(np.mean(bits)-expected)<.035
