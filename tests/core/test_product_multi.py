"""Original-token oracles for the standard multi-axis product HMM."""
from collections import Counter
from dataclasses import replace
from itertools import product
import math
import weakref

import numpy as np
import pytest
from scipy.special import logsumexp

from tri.domain.music import CountRule, MusicSpec, verify_music
from tri.errors import BudgetExceeded, InvalidSpecification, UnsupportedSpec, ZeroMass
from tri.inference.exact import Budget
from tri.inference.product_multi import MultiSpanProductChainMusicInference
from tri.inference.product_prefix import PrefixCachedProductChainMusicInference


def case(R=2, L=2, K=1, *, beta=.17, interval=None, pitches=(60, 62), guards=None, seed=8):
    spans = tuple(tuple(range(1+r*(L+1), 1+r*(L+1)+L)) for r in range(R))
    observed, anchors = {0: 62}, {0: 60}
    for r, span in enumerate(spans):
        token, sounding = (guards or {}).get(r, (62, 60))
        observed[span[-1]+1] = token
        anchors[span[-1]+1] = sounding
    spec = MusicSpec(length=R*(L+1)+1, pitches=pitches, observed=observed, fixed_soundings=anchors,
        equal_onsets=tuple((spans[0][j], spans[r][j]) for r in range(1,R) for j in range(L)),
        onset_counts=() if K is None else tuple(CountRule(span,K) for span in spans),
        motion_cost=beta, max_adjacent_interval=interval)
    logits = np.random.default_rng(seed).normal(scale=.7, size=(spec.length,130))
    logits[:, [0,1,*[p+2 for p in pitches]]] += 1.
    q = logits-logsumexp(logits,axis=1,keepdims=True)
    return spec,q,spans


def oracle(spec,q,evidence=None):
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
            support[tokens]=checked.soft_score+sum(float(q[i,tokens[i]]) for i in free)
    return support


def z(support):
    return float(logsumexp(list(support.values()))) if support else -math.inf


@pytest.mark.parametrize('R,L,K,pitches',[(2,2,0,(60,62)),(2,3,1,(60,62)),(2,2,2,(60,62)),
    (3,2,1,(60,62)),(4,2,1,(60,)),(3,2,None,(60,62))])
@pytest.mark.parametrize('beta,interval',[(0.,None),(.17,None),(.23,1)])
def test_partition_and_original_soft_weight_match_independent_token_oracle(R,L,K,pitches,beta,interval):
    spec,q,spans=case(R,L,K,beta=beta,interval=interval,pitches=pitches)
    expected=oracle(spec,q)
    engine=MultiSpanProductChainMusicInference(spec,q)
    assert expected
    assert engine.log_partition()==pytest.approx(z(expected),abs=3e-11)
    for tokens,weight in expected.items():
        assert engine.log_weight(tokens)==pytest.approx(weight,abs=3e-11)
    for seed in range(4):
        tokens=engine.sample_full(np.random.default_rng(seed))
        assert tokens in expected and verify_music(tokens,spec).valid


@pytest.mark.parametrize('evidence',[{'y1':62},{'y2':1},{'y1':62,'y5':1},{'y1':62,'y4':0}])
def test_clamps_keep_original_q_and_projected_sample_probabilities(evidence):
    spec,q,_=case(R=3)
    engine=MultiSpanProductChainMusicInference(spec,q)
    original=engine.log_partition()
    expected=oracle(spec,q,evidence)
    assert engine.log_partition(evidence)==pytest.approx(z(expected),abs=3e-11)
    if not expected:
        with pytest.raises(ZeroMass): engine.sample_full(np.random.default_rng(4),evidence)
    else:
        batch=engine.sample_batch(['y2','y4'],np.random.default_rng(28),evidence)
        clamp={**evidence,**batch.assignment}
        mass=z(oracle(spec,q,clamp))
        assert batch.log_clamped_partition==pytest.approx(mass,abs=3e-11)
        assert batch.log_probability==pytest.approx(mass-z(expected),abs=3e-11)
        assert engine.log_probability(batch.assignment,evidence)==pytest.approx(mass-z(expected),abs=3e-11)
    assert engine.log_partition()==pytest.approx(original,abs=3e-11)


@pytest.mark.parametrize('R,K',[(2,1),(3,1),(3,None),(4,0)])
def test_forward_backward_token_marginals_match_oracle_at_each_axis(R,K):
    spec,q,spans=case(R=R,K=K,pitches=(60,) if R==4 else (60,62),interval=1)
    engine=MultiSpanProductChainMusicInference(spec,q)
    for evidence in ({},{f'y{spans[0][0]}':0}):
        support=oracle(spec,q,evidence)
        if not support: continue
        for span in spans:
            for pos in span:
                actual=engine.marginal_log_probs(f'y{pos}',evidence)
                expected=[z({t:w for t,w in support.items() if t[pos]==token})-z(support)
                          for token in engine.graph.domains[f'y{pos}']]
                np.testing.assert_allclose(actual,expected,atol=4e-11,rtol=0)
                assert logsumexp(actual)==pytest.approx(0.,abs=4e-11)
    assert engine.marginal_log_probs('y0').tolist()==[0.]


@pytest.mark.parametrize('count',[True,False])
@pytest.mark.parametrize('interval',[None,1])
def test_two_span_matches_existing_optimized_product_prefix(count,interval):
    spec,q,spans=case(K=1 if count else None,interval=interval,guards={0:(64,62),1:(1,62)})
    engines=[MultiSpanProductChainMusicInference(spec,q),PrefixCachedProductChainMusicInference(spec,q)]
    for evidence in ({},{'y1':0},{'y2':1},{'y1':64,'y5':62}):
        values=[engine.log_partition(evidence) for engine in engines]
        assert values[0]==pytest.approx(values[1],abs=3e-11)
        if math.isfinite(values[0]):
            np.testing.assert_allclose(engines[0].marginal_log_probs('y4',evidence),
                                       engines[1].marginal_log_probs('y4',evidence),atol=4e-11,rtol=0)


def test_partial_observations_are_deltas_and_single_count_plus_flags_work():
    spec,q,spans=case(R=3)
    spec=replace(spec,observed={**spec.observed,spans[0][0]:62,spans[0][1]:1},
                 fixed_soundings={**spec.fixed_soundings,spans[0][0]:60,spans[0][1]:60},
                 onset_counts=(CountRule(spans[1],1),CountRule((spans[2][0],),1)))
    expected=z(oracle(spec,q))
    engine=MultiSpanProductChainMusicInference(spec,q)
    assert engine.log_partition()==pytest.approx(expected,abs=3e-11)
    changed=q.copy()
    for i in spec.observed: changed[i]=-np.log(130.)
    assert MultiSpanProductChainMusicInference(spec,changed).log_partition()==pytest.approx(expected,abs=3e-11)


@pytest.mark.parametrize('right',[(1,60),(0,None),(64,62)])
def test_right_guard_and_sounding_pitch_rules(right):
    spec,q,spans=case(R=3,guards={0:right},interval=2)
    spec=replace(spec,pitch_ranges={spans[1][1]:(60,60)},pitch_classes={spans[2][0]:(0,)})
    support=oracle(spec,q)
    engine=MultiSpanProductChainMusicInference(spec,q)
    assert engine.log_partition()==pytest.approx(z(support),abs=3e-11)
    for seed in range(5): assert engine.sample_full(np.random.default_rng(seed)) in support


def test_inherited_outside_vocabulary_hold_and_end_boundary():
    spec,q,spans=case(R=3,K=0,pitches=(60,))
    spec=replace(spec,initial_pitch=55,observed={i:1 for i in spec.observed},
                 fixed_soundings={i:55 for i in spec.observed},enforce_end=True,end_pitch=55)
    engine=MultiSpanProductChainMusicInference(spec,q)
    assert engine.D==3
    assert engine.log_partition()==pytest.approx(z(oracle(spec,q)),abs=3e-11)
    assert engine.sample_full(np.random.default_rng(4))==(1,)*spec.length


def test_window_edge_spans_retain_initial_and_final_motion_constraints():
    spec=MusicSpec(length=5,pitches=(60,62),observed={2:64},fixed_soundings={2:62},
        initial_pitch=60,enforce_end=True,end_pitch=60,motion_cost=.4,max_adjacent_interval=1,
        equal_onsets=((0,3),(1,4)),onset_counts=(CountRule((0,1),1),CountRule((3,4),1)))
    q=np.full((5,130),-np.log(130.))
    support=oracle(spec,q)
    engine=MultiSpanProductChainMusicInference(spec,q)
    assert engine.log_partition()==pytest.approx(z(support),abs=3e-11)
    if support: assert engine.sample_full(np.random.default_rng(3)) in support


@pytest.mark.parametrize('kind',['zero_q','totals','flags','anchors'])
def test_zero_support_is_zero_mass_distinct_from_budget_and_unsupported(kind):
    spec,q,spans=case()
    if kind=='zero_q':
        q[:]=-math.inf; q[:,129]=0.
    elif kind=='totals': spec=replace(spec,onset_counts=(CountRule(spans[0],0),CountRule(spans[1],2)))
    elif kind=='flags': spec=replace(spec,onset_counts=spec.onset_counts+(CountRule((1,),0),CountRule((4,),1)))
    else: spec=replace(spec,fixed_soundings={**spec.fixed_soundings,0:62})
    assert not oracle(spec,q)
    engine=MultiSpanProductChainMusicInference(spec,q)
    assert engine.log_partition()==-math.inf
    with pytest.raises(ZeroMass): engine.sample_full(np.random.default_rng(0))
    with pytest.raises(ZeroMass): engine.marginal_log_probs('y1')


def test_complete_batch_uses_direct_weight_and_empty_batch_does_not_draw(monkeypatch):
    spec,q,_=case(R=3)
    engine=MultiSpanProductChainMusicInference(spec,q)
    original=engine.log_partition()
    prepared=engine._prepared
    full=engine.sample_batch(tuple(engine.graph.domains),np.random.default_rng(313))
    tokens=tuple(full.assignment[f'y{i}'] for i in range(spec.length))
    assert full.log_clamped_partition==pytest.approx(oracle(spec,q)[tokens],abs=3e-11)
    assert full.log_probability==pytest.approx(full.log_clamped_partition-original,abs=3e-11)
    assert engine._prepared is prepared
    assert engine.last_stats['sample_probability']=='complete_sequence_weight'
    def fail(*args,**kwargs): raise AssertionError('empty batch must not draw')
    monkeypatch.setattr(engine,'sample_full',fail)
    assert engine.sample_batch([],np.random.default_rng(1)).log_probability==0.


def test_trace_cache_is_immutable_evidence_keyed_and_accounted_in_workspace():
    spec,q,_=case(R=3)
    engine=MultiSpanProductChainMusicInference(spec,q)
    value=engine.log_partition({'y1':62})  # first conditional has no base prewarm
    plan=engine._prepared
    assert plan.key==( ('y1',62), )
    assert not engine.last_stats['cache_hit']
    arrays=(*plan.history,*plan.final,*plan.tails,*(b for step in plan.matrices for b in step))
    assert all(not a.flags.writeable for a in arrays)
    assert sum(a.nbytes for a in arrays)==engine.last_stats['retained_array_bytes']
    assert engine.last_stats['retained_array_bytes']<=engine.storage_bytes<=engine.budget.max_workspace_bytes
    assert engine.last_stats['saved_message_bytes']==engine.R*(engine.L+1)*engine.D*8
    assert engine.last_stats['dispatch']=='known_template_chains'
    assert engine.log_partition({'y1':62})==value
    assert engine.last_stats['cache_hit'] and engine.last_stats['blas_products']==0
    assert engine._prepared is plan
    old=weakref.ref(plan); del arrays,plan
    engine.log_partition({'y1':0})
    assert old() is None
    assert not engine.last_stats['cache_hit']
    assert engine.last_stats['maximum_kernel_entries']<=engine.last_stats['kernel_entry_limit']
    frozen=engine.log_partition()
    q[:]=-np.log(130.)
    assert engine.log_partition()==frozen
    assert MultiSpanProductChainMusicInference(spec,q).log_partition()!=pytest.approx(frozen)


@pytest.mark.parametrize('K,token',[(0,1),(2,62)])
def test_extreme_legal_weights_remain_finite_under_complete_ffbs(K,token):
    spec,q,spans=case(R=3,K=K,pitches=(60,),guards={r:(1,60) for r in range(3)})
    q[:]=-math.inf; q[:,129]=0.; q[:,token]=-1000.
    engine=MultiSpanProductChainMusicInference(spec,q)
    assert engine.log_partition()==pytest.approx(-6000.,abs=2e-10)
    assert verify_music(engine.sample_full(np.random.default_rng(3)),spec).valid


@pytest.mark.parametrize('kernel_entries',[8,32])
def test_blocked_log_fallback_retains_sole_extreme_branches(kernel_entries):
    spec,q,_=case()
    engine=MultiSpanProductChainMusicInference(spec,q)
    engine._kernel_entries=kernel_entries  # exercise rectangular blocking and strict fallback
    left=np.array([[0.,-1000.,-math.inf],[-999.,0.,-1001.],[-math.inf]*3])
    right=np.array([[-math.inf,-1000.,0.,-math.inf],[0.,-math.inf,-1000.,-math.inf],[-1000.,0.,-math.inf,-math.inf]])
    stats=engine._stats()
    actual=engine._log_matmul(left,right,stats)
    expected=logsumexp(left[:,:,None]+right[None,:,:],axis=1)
    np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=0)
    if kernel_entries==32: assert stats['log_semiring_products']>0
    assert stats['maximum_kernel_entries']<=kernel_entries
    assert actual[0,0]==-1000.


@pytest.mark.parametrize('count',[True,False])
def test_actual_full_joint_distribution_matches_original_tokens(count):
    spec,q,_=case(K=1 if count else None,pitches=(60,62),beta=.7)
    support=oracle(spec,q); normalizer=z(support)
    engine=MultiSpanProductChainMusicInference(spec,q)
    rng=np.random.default_rng(756)
    draws=Counter(engine.sample_full(rng) for _ in range(2200))
    assert set(draws)<=set(support) and len(draws)>3
    for tokens,weight in support.items():
        assert abs(draws[tokens]/2200-math.exp(weight-normalizer))<.04


def test_product_state_and_saved_history_budget_fail_as_budget_not_unsupported():
    spec,q,_=case(R=8,pitches=tuple(range(60,66)))
    with pytest.raises(BudgetExceeded): MultiSpanProductChainMusicInference(spec,q).log_partition()
    spec,q,_=case(R=3)
    spacious=MultiSpanProductChainMusicInference(spec,q)
    with pytest.raises(BudgetExceeded):
        MultiSpanProductChainMusicInference(spec,q,Budget(max_workspace_bytes=spacious._base_bytes()+1000)).log_partition()
    tight=MultiSpanProductChainMusicInference(spec,q,Budget(max_factor_entries=max(spacious.state_entries,2*spacious.joint_states)))
    assert tight.log_partition()==pytest.approx(spacious.log_partition(),abs=3e-11)
    assert tight.last_stats['maximum_kernel_entries']<=tight.budget.max_factor_entries


@pytest.mark.parametrize('change',['no_relations','missing_column','unknown_separator','overlap_count'])
def test_unsupported_layouts_are_explicit(change):
    spec,q,spans=case()
    if change=='no_relations': spec=replace(spec,equal_onsets=())
    elif change=='missing_column': spec=replace(spec,equal_onsets=((1,4),))
    elif change=='overlap_count': spec=replace(spec,onset_counts=(CountRule((1,2,4),1),))
    else:
        spec=replace(spec,observed={i:v for i,v in spec.observed.items() if i!=3},
                     fixed_soundings={i:v for i,v in spec.fixed_soundings.items() if i!=3})
    with pytest.raises(UnsupportedSpec): MultiSpanProductChainMusicInference(spec,q)


@pytest.mark.parametrize('bad',['shape','nan','renormalization','complex'])
def test_full130_input_validation_remains_in_force(bad):
    spec,q,_=case()
    if bad=='shape': q=q[:,:4]
    elif bad=='nan': q[1,129]=math.nan
    elif bad=='renormalization': q[1]+=.1
    else: q=q.astype(complex)
    with pytest.raises(InvalidSpecification): MultiSpanProductChainMusicInference(spec,q)


@pytest.mark.parametrize('determination',['K0','Kfull','flags_and_K','visible_first_span'])
def test_eight_known_rhythm_chains_keep_soft_and_hard_rules_under_modest_budget(determination):
    from tri.inference.music_backends import MusicExactInference
    K=0 if determination=='K0' else 3 if determination=='Kfull' else 1
    spec,q,spans=case(R=8,L=3,K=K,pitches=(60,61,62,63),beta=.31,interval=1)
    if determination=='flags_and_K':
        spec=replace(spec,onset_counts=spec.onset_counts+(CountRule((spans[0][0],),1),))
    elif determination=='visible_first_span':
        visible=dict(zip(spans[0],(62,1,1)))
        spec=replace(spec,observed={**spec.observed,**visible},
                     fixed_soundings={**spec.fixed_soundings,**{i:60 for i in visible}},onset_counts=())
    engine=MultiSpanProductChainMusicInference(spec,q,Budget(max_workspace_bytes=256_000))
    reference=MusicExactInference(spec,q,'template')
    assert engine.log_partition()==pytest.approx(reference.log_partition(),abs=4e-11)
    assert engine.last_stats['dispatch']=='known_template_chains'
    assert not engine.last_stats['product_state_materialized']
    assert engine.last_stats['saved_message_bytes']==engine.R*(engine.L+1)*engine.D*8
    assert engine.storage_bytes<=256_000
    for seed in range(4):
        tokens=engine.sample_full(np.random.default_rng(seed))
        assert verify_music(tokens,spec).valid
    np.testing.assert_allclose(engine.marginal_log_probs(f'y{spans[-1][-1]}'),
                               reference.marginal_log_probs(f'y{spans[-1][-1]}'),atol=4e-11,rtol=0)


def test_first_condition_can_fix_rhythm_before_any_cartesian_budget_check():
    from tri.inference.music_backends import MusicExactInference
    spec,q,spans=case(R=8,L=3,K=1,pitches=(60,61,62,63),beta=.21,interval=2)
    budget=Budget(max_workspace_bytes=256_000)
    engine=MultiSpanProductChainMusicInference(spec,q,budget)
    evidence={f'y{spans[0][0]}':62}  # K=1 now forces all other onset bits to zero
    expected=MusicExactInference(spec,q,'template').log_partition(evidence)
    assert engine.log_partition(evidence)==pytest.approx(expected,abs=4e-11)
    assert engine.last_stats['dispatch']=='known_template_chains'
    assert verify_music(engine.sample_full(np.random.default_rng(5),evidence),spec).valid
    with pytest.raises(BudgetExceeded): engine.log_partition()
    assert engine._prepared is None  # prior known-template trace was released
    assert engine.log_partition(evidence)==pytest.approx(expected,abs=4e-11)


def test_unknown_template_cache_retains_only_real_joint_arrays():
    spec,q,_=case(R=3,L=3,K=1)
    engine=MultiSpanProductChainMusicInference(spec,q)
    initial=engine.log_partition()
    plan=engine._prepared
    assert engine.last_stats['dispatch']=='multi_axis_product'
    assert engine.last_stats['product_state_materialized']
    assert plan.known_bits is None
    assert plan.final.shape==(engine.D,)*engine.R
    assert all(layer.shape==(engine.count_size,*((engine.D,)*engine.R)) for layer in plan.history)
    assert engine.last_stats['saved_message_bytes']==(engine.L+1)*engine.state_entries*8
    assert engine.log_partition()==initial and engine._prepared is plan
    assert engine.last_stats['axis_contractions']==0
    engine.sample_full(np.random.default_rng(45))
    assert engine._prepared is plan
    assert all(not layer.flags.writeable for layer in plan.history)
