"""Independent original-token oracle for the visible rank-one proposal.

The keep policy is frozen from the original spec: all outside NOTE costs,
inside NOTE costs whose predecessor has an explicit sounding anchor (or the
window initial state), and costs of initially observed NOTE destinations.
Temporary evidence never changes this policy or removes its original q.
"""
from collections import Counter
from dataclasses import replace
from itertools import product
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from tri.domain.music import CountRule, MusicSpec, verify_music
from tri.errors import BudgetExceeded, UnsupportedSpec, ZeroMass
from tri.inference.exact import Budget
from tri.inference.onset_reset import OnsetResetMusicInference
from tri.inference.rank_one_onset import RankOneOnsetMusicInference


def case(R=2,L=2,K=1,*,beta=.35,pitches=(60,62),partial=False,guard=None,seed=819):
    spans=tuple(tuple(range(1+r*(L+1),1+r*(L+1)+L)) for r in range(R))
    observed,anchors={0:62},{0:60}
    for r,span in enumerate(spans):
        token,pitch=(guard or {}).get(r,(62,60))
        observed[span[-1]+1]=token; anchors[span[-1]+1]=pitch
    if partial:
        observed[spans[0][1]]=64; anchors[spans[0][1]]=62
    spec=MusicSpec(length=R*(L+1)+1,pitches=pitches,observed=observed,fixed_soundings=anchors,
        equal_onsets=tuple((spans[0][j],spans[r][j]) for r in range(1,R) for j in range(L)),
        onset_counts=() if K is None else tuple(CountRule(span,K) for span in spans),motion_cost=beta)
    logits=np.random.default_rng(seed).normal(scale=.7,size=(spec.length,130))
    return spec,logits-logsumexp(logits,axis=1,keepdims=True),spans


def independent_scores(tokens,spec,spans,retain_visible=True):
    span_at={position:r for r,span in enumerate(spans) for position in span}
    previous=spec.initial_pitch
    retained=0.; residual=[0.]*len(spans)
    for position,token in enumerate(tokens):
        if token==0:
            previous=None
        elif token>=2:
            pitch=token-2
            cost=0. if previous is None else -spec.motion_cost*abs(pitch-previous)
            visible_condition=(position==0 or position-1 in spec.fixed_soundings
                               or spec.observed.get(position,0)>=2)
            if position not in span_at or (retain_visible and visible_condition):
                retained+=cost
            else:
                residual[span_at[position]]+=cost
            previous=pitch
    return retained,tuple(residual)


def oracle(spec,q,spans,*,retain_visible=True,evidence=None):
    free=[i for i in range(spec.length) if i not in spec.observed]
    vocabulary=(0,1,*[p+2 for p in spec.pitches])
    domains=[tuple(t for t in vocabulary if np.isfinite(q[i,t])) for i in free]
    support={}
    for assignment in product(*domains):
        values={**spec.observed,**dict(zip(free,assignment))}
        tokens=tuple(values[i] for i in range(spec.length))
        if any(tokens[int(name[1:])]!=value for name,value in (evidence or {}).items()): continue
        checked=verify_music(tokens,spec)
        if checked.valid:
            retained,residual=independent_scores(tokens,spec,spans,retain_visible)
            assert checked.soft_score==pytest.approx(retained+sum(residual),abs=2e-12)
            support[tokens]={'log_weight':sum(float(q[i,tokens[i]]) for i in free)+retained,
                             'retained':retained,'residual':residual,'soft':checked.soft_score}
    return support


def z(support):
    return float(logsumexp([row['log_weight'] for row in support.values()])) if support else -math.inf


@pytest.mark.parametrize('R,L,K,pitches,partial',[(2,2,1,(60,62),False),(2,3,2,(60,62),True),
    (3,2,1,(60,62),True),(4,2,1,(60,),False),(2,2,0,(60,62),False),
    (3,2,None,(60,62),True),(2,2,2,(60,62),False)])
@pytest.mark.parametrize('retain_visible',[False,True])
def test_rank_one_proposal_partition_weight_and_per_span_residual_match_original_tokens(R,L,K,pitches,partial,retain_visible):
    spec,q,spans=case(R,L,K,pitches=pitches,partial=partial)
    expected=oracle(spec,q,spans,retain_visible=retain_visible)
    engine=RankOneOnsetMusicInference(spec,q,retain_visible=retain_visible)
    assert expected
    assert engine.log_partition()==pytest.approx(z(expected),abs=3e-11)
    for tokens,row in expected.items():
        assert engine.log_weight(tokens)==pytest.approx(row['log_weight'],abs=3e-11)
        assert engine.retained_soft_score(tokens)==pytest.approx(row['retained'],abs=3e-12)
        residual=[engine.span_residual_score(r,tuple(tokens[i] for i in span)) for r,span in enumerate(spans)]
        np.testing.assert_allclose(residual,row['residual'],atol=3e-12,rtol=0)
        assert row['soft']==pytest.approx(engine.retained_soft_score(tokens)+sum(residual),abs=3e-12)
        assert all(value<=0. for value in residual)
    for seed in range(5): assert engine.sample_full(np.random.default_rng(seed)) in expected


@pytest.mark.parametrize('evidence',[{'y2':64},{'y1':62,'y2':64},{'y2':64,'y5':1},{'y1':0,'y4':62}])
def test_temporary_clamps_do_not_promote_newly_known_destinations_or_lose_q(evidence):
    spec,q,spans=case(K=None)
    engine=RankOneOnsetMusicInference(spec,q)
    original=engine.log_partition()
    expected=oracle(spec,q,spans,evidence=evidence)
    assert engine.log_partition(evidence)==pytest.approx(z(expected),abs=3e-11)
    assert engine.log_partition()==pytest.approx(original,abs=3e-11)
    # A previously conditioned cache must not leak into component sampling.
    engine.log_partition(evidence)
    tokens=engine.sample_full(np.random.default_rng(75))
    assert tokens in oracle(spec,q,spans)
    assert engine._prepared.key==()
    if expected:
        with pytest.raises(UnsupportedSpec): engine.sample_full(np.random.default_rng(1),evidence)


@pytest.mark.parametrize('guard',[(1,60),(1,62),(0,None),(64,62)])
def test_right_boundary_cost_and_hold_rules_remain_in_proposal(guard):
    spec,q,spans=case(guard={0:guard},partial=True)
    expected=oracle(spec,q,spans)
    engine=RankOneOnsetMusicInference(spec,q)
    assert engine.log_partition()==pytest.approx(z(expected),abs=3e-11)
    if not expected:
        # Initially observed NOTE62 cannot end in a right HOLD anchored to60.
        with pytest.raises(ZeroMass):
            engine.sample_full(np.random.default_rng(0))
        return
    for seed in range(8):
        tokens=engine.sample_full(np.random.default_rng(seed))
        assert tokens in expected and verify_music(tokens,spec).valid


def test_fixed_initial_predecessor_outside_pitch_vocabulary_and_window_edges():
    spans=((0,1),(3,4))
    spec=MusicSpec(length=5,pitches=(60,62),initial_pitch=55,observed={2:1},fixed_soundings={2:60},
        equal_onsets=((0,3),(1,4)),onset_counts=tuple(CountRule(s,1) for s in spans),
        motion_cost=.21,enforce_end=True,end_pitch=60)
    q=np.full((5,130),-np.log(130.))
    expected=oracle(spec,q,spans)
    engine=RankOneOnsetMusicInference(spec,q)
    assert engine.log_partition()==pytest.approx(z(expected),abs=3e-11)
    assert engine.sample_full(np.random.default_rng(7)) in expected
    for tokens,row in expected.items():
        assert engine.retained_soft_score(tokens)==pytest.approx(row['retained'],abs=2e-12)


def test_phi_incoming_factors_control_reverse_predecessor_sampling_not_just_z():
    spec,q,spans=case(partial=True,beta=.7)
    q[:]=-np.log(130.)
    expected=oracle(spec,q,spans)
    assert len(expected)==8
    engine=RankOneOnsetMusicInference(spec,q)
    rng=np.random.default_rng(9091)
    draws=Counter(engine.sample_full(rng) for _ in range(2200))
    assert set(draws)==set(expected)
    normalizer=z(expected)
    for tokens,row in expected.items():
        assert abs(draws[tokens]/2200-math.exp(row['log_weight']-normalizer))<.04
    # y2 is an initially observed NOTE62; y1 is REST or HOLD60. Incoming
    # motion lowers HOLD probability from 1/2 to sigmoid(-2*beta).
    empirical=sum(count for tokens,count in draws.items() if tokens[1]==1)/2200
    expected_hold=math.exp(-1.4)/(1+math.exp(-1.4))
    assert abs(empirical-expected_hold)<.04


def test_rank_one_joint_frequencies_include_unknown_rhythm_and_retained_destination_weights():
    spec,q,spans=case(L=2,K=1,beta=.7)
    expected=oracle(spec,q,spans)
    engine=RankOneOnsetMusicInference(spec,q)
    rng=np.random.default_rng(2371)
    draws=Counter(engine.sample_full(rng) for _ in range(2600))
    assert set(draws)<=set(expected) and len(draws)>8
    normalizer=z(expected)
    for tokens,row in expected.items():
        assert abs(draws[tokens]/2600-math.exp(row['log_weight']-normalizer))<.04
    for pattern in ((1,0),(0,1)):
        reference=sum(math.exp(row['log_weight']-normalizer) for tokens,row in expected.items()
                      if tuple(int(tokens[i]>=2) for i in spans[0])==pattern)
        observed=sum(count for tokens,count in draws.items()
                     if tuple(int(tokens[i]>=2) for i in spans[0])==pattern)/2600
        assert abs(observed-reference)<.04


def test_fixed_template_component_paths_agree_with_full_proposal_conditional_distribution():
    spec,q,spans=case(K=1,beta=.45)
    support=oracle(spec,q,spans)
    engine=RankOneOnsetMusicInference(spec,q)
    bits=(1,0)
    conditional={tokens:row for tokens,row in support.items()
                 if tuple(int(tokens[i]>=2) for i in spans[0])==bits}
    normalizer=z(conditional)
    expected={values:sum(math.exp(row['log_weight']-normalizer) for tokens,row in conditional.items()
                         if tuple(tokens[i] for i in spans[0])==values)
              for values in {tuple(tokens[i] for i in spans[0]) for tokens in conditional}}
    rng=np.random.default_rng(712)
    draws=Counter(engine.sample_span(0,bits,rng) for _ in range(1800))
    assert set(draws)==set(expected)
    for values,p in expected.items(): assert abs(draws[values]/1800-p)<.04


def test_boundary_only_and_zero_beta_reduce_to_existing_base_targets():
    for beta in (0.,.29):
        spec,q,spans=case(R=3,partial=True,beta=beta)
        boundary=OnsetResetMusicInference(replace(spec,motion_cost=0.),q,boundary_motion_cost=beta)
        rank=RankOneOnsetMusicInference(spec,q,retain_visible=False)
        assert rank.log_partition()==pytest.approx(boundary.log_partition(),abs=3e-11)
        if beta==0.:
            full_rank=RankOneOnsetMusicInference(spec,q)
            assert full_rank.log_partition()==pytest.approx(boundary.log_partition(),abs=3e-11)
            for seed in range(5):
                tokens=full_rank.sample_full(np.random.default_rng(seed))
                assert full_rank.retained_soft_score(tokens)==0.
                assert all(full_rank.span_residual_score(r,tuple(tokens[i] for i in span))==0.
                           for r,span in enumerate(spans))


@pytest.mark.parametrize('K,token',[(0,1),(2,62)])
def test_extreme_finite_q_and_zero_support(K,token):
    spec,q,spans=case(R=3,K=K,pitches=(60,),guard={r:(1,60) for r in range(3)})
    q[:]=-math.inf; q[:,129]=0.; q[:,token]=-1000.
    expected=oracle(spec,q,spans)
    engine=RankOneOnsetMusicInference(spec,q)
    assert engine.log_partition()==pytest.approx(z(expected),abs=3e-10)
    assert engine.sample_full(np.random.default_rng(33)) in expected
    q[:,token]=-math.inf
    empty=RankOneOnsetMusicInference(spec,q)
    assert empty.log_partition()==-math.inf
    with pytest.raises(ZeroMass): empty.sample_full(np.random.default_rng(0))


def test_proposal_normalized_queries_report_proposal_not_original_soft_target():
    spec,q,spans=case(L=3,K=2,beta=.43)
    expected=oracle(spec,q,spans)
    engine=RankOneOnsetMusicInference(spec,q)
    assert engine.log_partition()==pytest.approx(z(expected),abs=3e-11)
    target_z=float(logsumexp([row['log_weight']+sum(row['residual']) for row in expected.values()]))
    assert target_z<engine.log_partition()-1e-3
    batch=engine.sample_batch([f'y{i}' for i in range(spec.length)],np.random.default_rng(10))
    tokens=tuple(batch.assignment[f'y{i}'] for i in range(spec.length))
    assert batch.log_clamped_partition==pytest.approx(expected[tokens]['log_weight'],abs=3e-11)
    assert batch.log_probability==pytest.approx(expected[tokens]['log_weight']-z(expected),abs=3e-11)
    assert engine.last_stats['partition_semantics']=='proposal'


def test_more_retained_nonpositive_factors_reduce_proposal_z_without_changing_target_support():
    spec,q,spans=case(R=3,partial=True,L=3,K=2,beta=.29)
    full=RankOneOnsetMusicInference(spec,q)
    boundary=RankOneOnsetMusicInference(spec,q,retain_visible=False)
    assert full.log_partition()<boundary.log_partition()
    support=oracle(spec,q,spans)
    target_z=float(logsumexp([row['log_weight']+sum(row['residual']) for row in support.values()]))
    assert target_z<=full.log_partition()<=boundary.log_partition()


def test_workspace_incoming_storage_and_hard_interval_scope_are_explicit():
    spec,q,_=case()
    with pytest.raises(UnsupportedSpec): RankOneOnsetMusicInference(replace(spec,max_adjacent_interval=1),q)
    baseline=OnsetResetMusicInference(replace(spec,motion_cost=0.),q,boundary_motion_cost=spec.motion_cost)
    base_bytes=baseline._workspace()
    with pytest.raises(BudgetExceeded):
        RankOneOnsetMusicInference(spec,q,Budget(max_workspace_bytes=base_bytes)).log_partition()
    full=RankOneOnsetMusicInference(spec,q)
    full.log_partition()
    assert full.last_stats['workspace_bytes_estimate']>=base_bytes+8*full.R*full.L*full.D
