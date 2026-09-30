"""Same-checkpoint research controls with explicit finite compute budgets."""
from dataclasses import dataclass, replace
import math
from numbers import Integral

import numpy as np
from scipy.special import logsumexp

from tri.domain.music import CountRule, verify_music
from tri.errors import BudgetExceeded, InvalidSpecification, UnsupportedSpec, ZeroMass
from tri.inference.exact import Budget
from tri.sampling.baselines import decode_method
from tri.sampling.direct import direct_decode
from tri.sampling.particles import path_is, smc_decode

RESEARCH_METHODS = ('raw_reference', 'local_constraints', 'one_shot_joint', 've_joint',
                    'template_joint', 'fa_joint', 'single_template', 'pooled_template',
                    'tri_direct', 'tri_guided', 'path_is_4', 'path_is_16', 'smc_4', 'smc_16',
                    'rejection_16', 'rejection_64', 'candidate_16',
                    'onset_rejection_drop', 'onset_rejection_boundary',
                    'onset_rejection_visible', 'onset_rejection_boundary_early',
                    'onset_rejection_visible_early')


@dataclass(frozen=True)
class ResearchResult:
    tokens: tuple[int, ...]
    model_calls: int
    trace: tuple[dict, ...] = ()
    diagnostics: dict | None = None


def _joint(spec, provider, backend, budget, rng):
    from tri.inference.music_backends import make_music_engine
    state = tuple(spec.observed.get(i) for i in range(spec.length))
    missing = tuple(f'y{i}' for i,v in enumerate(state) if v is None)
    if not missing:
        checked = verify_music(state,spec)
        if not checked.valid:
            raise ZeroMass('complete requested context violates constraints')
        return state, {'backend': backend, 'log_z': checked.soft_score}
    q = provider(state,1.)
    engine = make_music_engine(spec,q,backend=backend,budget=budget)
    z = engine.log_partition()
    draw = engine.sample_batch(missing,rng)
    tokens = tuple(spec.observed[i] if i in spec.observed else int(draw.assignment[f'y{i}']) for i in range(spec.length))
    return tokens, {'backend': engine.backend_name, 'log_z': z, 'log_sample_probability': draw.log_probability,
                    'backend_stats': engine.last_stats}


def _approximate_template(spec, q, pooled, rng, budget):
    """Sample shared onset structure using independent feature evidence only.

    Counts/observed flags are respected, but pitch-chain feasibility is not
    folded into the template weights. A subsequently impossible template fails;
    there is no hidden retry. Full exact template conditioning is a separate
    baseline implemented in music_backends.
    """
    positions = sorted(set(p for pair in spec.equal_onsets for p in pair) |
                       set(p for rule in spec.onset_counts for p in rule.positions))
    parent = {p:p for p in positions}
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for a,b in spec.equal_onsets:
        parent[find(b)] = find(a)
    groups = {}
    for p in positions:
        groups.setdefault(find(p),[]).append(p)
    groups = list(groups.values())
    fixed = {}
    for j, group in enumerate(groups):
        flags = {spec.observed[p]>=2 for p in group if p in spec.observed}
        if len(flags)>1:
            raise ZeroMass('visible onset relation is contradictory')
        if flags:
            fixed[j] = int(next(iter(flags)))
    # Exact count-state sampling of the SAME approximate feature distribution.
    # This avoids enumerating 2**width candidates at 1–2 bar gap lengths.
    rules=[set(rule.positions) for rule in spec.onset_counts]
    coefficients=[tuple(sum(p in rule for p in group) for rule in rules) for group in groups]
    choices=[]; scores=[]
    for j,group in enumerate(groups):
        choices.append((fixed[j],) if j in fixed else (0,1))
        evidence=[p for p in group if p not in spec.observed]
        if not pooled:
            evidence=evidence[:1]
        scores.append(tuple(0. if j in fixed else sum(float(logsumexp(q[p,2:] if bit else q[p,:2])) for p in evidence) for bit in (0,1)))
    memo={}
    def suffix(j,remaining):
        if any(v<0 for v in remaining):
            return -math.inf
        if j==len(groups):
            return 0. if not any(remaining) else -math.inf
        key=(j,remaining)
        if key not in memo:
            budget.check_factor_entries(len(memo)+1,context='Feature-template count states')
            budget.check_workspace_bytes((len(memo)+1)*(256+48*len(rules)),context='Feature-template count states')
            memo[key]=-math.inf  # Reserve before descending; budget covers live recursion too.
            memo[key]=float(logsumexp([scores[j][bit]+suffix(j+1,tuple(v-bit*c for v,c in zip(remaining,coefficients[j]))) for bit in choices[j]]))
        return memo[key]
    remaining=tuple(rule.count for rule in spec.onset_counts)
    if not math.isfinite(suffix(0,remaining)):
        raise ZeroMass('template evidence has zero mass')
    assignment={}
    for j,group in enumerate(groups):
        tails=[tuple(v-bit*c for v,c in zip(remaining,coefficients[j])) for bit in choices[j]]
        weights=np.asarray([scores[j][bit]+suffix(j+1,tail) for bit,tail in zip(choices[j],tails)])
        probabilities=np.exp(weights-logsumexp(weights));probabilities/=probabilities.sum()
        selected=int(rng.choice(len(choices[j]),p=probabilities))
        assignment.update({p:choices[j][selected] for p in group})
        remaining=tails[selected]
    return assignment


def research_decode(method, spec, provider, *, steps=8, seed=0, backend='auto', budget=None,
                    max_proposals=10_000):
    if method not in RESEARCH_METHODS:
        raise ValueError(f'unknown research method {method}')
    budget=budget or Budget()
    rng=np.random.default_rng(seed)
    calls=0
    def counted(state,noise):
        nonlocal calls
        calls+=1
        return provider(state,noise)
    if method in ('onset_rejection_drop', 'onset_rejection_boundary',
                  'onset_rejection_visible', 'onset_rejection_boundary_early',
                  'onset_rejection_visible_early'):
        from tri.sampling.onset_rejection import OnsetRejectionSampler
        if isinstance(max_proposals, (bool, np.bool_)) or not isinstance(max_proposals, Integral) or max_proposals < 1:
            raise InvalidSpecification('max_proposals must be a positive integer')
        if spec.max_adjacent_interval is not None:
            raise UnsupportedSpec('onset rejection does not support max_adjacent_interval')
        proposal = method.removeprefix('onset_rejection_')
        state = tuple(spec.observed.get(i) for i in range(spec.length))
        if all(token is not None for token in state):
            if not verify_music(state, spec).valid:
                raise ZeroMass('complete requested context violates constraints')
            return ResearchResult(state, calls, diagnostics={
                'status': 'success', 'complete_observed': True, 'proposal': proposal,
                'proposals': 0, 'rejections': 0, 'target_normalizer_available': False})
        # One neural call freezes q for the entire rejection loop. In
        # particular this branch must not use _joint or its normalized batch.
        q = counted(state, 1.0)
        if proposal in ('drop', 'boundary'):
            sampler = OnsetRejectionSampler(spec, q, budget, proposal=proposal, max_proposals=max_proposals)
        else:
            from tri.sampling.onset_rejection_adaptive import AdaptiveOnsetRejectionSampler
            sampler = AdaptiveOnsetRejectionSampler(spec, q, budget, proposal=proposal, max_proposals=max_proposals)
        draw = sampler.sample_full(rng)
        return ResearchResult(draw.tokens, calls, diagnostics=draw.diagnostics)
    if method=='raw_reference':
        result=decode_method(method,spec,counted,steps=steps,seed=seed,budget=budget)
        return ResearchResult(result.tokens,calls,result.trace)
    if method in ('local_constraints','tri_direct','tri_guided'):
        active=replace(spec,equal_onsets=(),onset_counts=()) if method=='local_constraints' else spec
        result=direct_decode(active,counted,steps=steps,seed=seed,budget=budget,backend=backend,
                             epsilon=.5 if method=='tri_guided' else 1.)
        return ResearchResult(result.tokens,calls,result.trace,{'backend':backend,'epsilon':.5 if method=='tri_guided' else 1.})
    if method in ('one_shot_joint','ve_joint','template_joint','fa_joint'):
        selected={'ve_joint':'ve','template_joint':'template','fa_joint':'automaton'}.get(method,backend)
        tokens,diag=_joint(spec,counted,selected,budget,rng)
        return ResearchResult(tokens,calls,diagnostics=diag)
    if method in ('single_template','pooled_template'):
        from tri.inference.music_backends import make_music_engine
        state=tuple(spec.observed.get(i) for i in range(spec.length))
        q=counted(state,1.)
        flags=_approximate_template(spec,q,method=='pooled_template',rng,budget)
        active=replace(spec,onset_counts=spec.onset_counts+tuple(CountRule((p,),v) for p,v in flags.items()))
        engine=make_music_engine(active,q,backend=backend,budget=budget)
        missing=tuple(f'y{i}' for i,v in enumerate(state) if v is None)
        batch=engine.sample_batch(missing,rng)
        tokens=tuple(spec.observed[i] if i in spec.observed else int(batch.assignment[f'y{i}']) for i in range(spec.length))
        return ResearchResult(tokens,calls,diagnostics={'template':flags,'backend':engine.backend_name,'template_retry':False})
    if method.startswith(('path_is_','smc_')):
        particles=int(method.rsplit('_',1)[1])
        function=path_is if method.startswith('path_is_') else smc_decode
        result=function(spec,counted,particles=particles,steps=steps,seed=seed,backend=backend,budget=budget)
        return ResearchResult(result.tokens,calls,result.trace,{'particles':particles,'ess':result.ess,
                    'unique_ancestors':result.unique_ancestors,'normalized_weights':result.normalized_weights,
                    'particle_tokens':result.particles,'log_normalizer_estimate':result.log_normalizer_estimate})
    candidate_limit=int(method.rsplit('_',1)[1])
    attempts=[]
    best=None
    for index in range(candidate_limit):
        child_seed=seed if index==0 else int(rng.integers(0,2**63-1))
        result=decode_method('raw_reference',spec,counted,steps=steps,seed=child_seed,budget=budget)
        checked=verify_music(result.tokens,spec)
        attempts.append({'attempt':index+1,'valid':checked.valid,'violations':checked.violations,'soft_score':checked.soft_score})
        if method.startswith('rejection_'):
            # Current MusicSpec has S=-motion_cost*jumps <=0; M=1 is a valid
            # envelope. This accounts for BOTH hard C and soft exp(S).
            if checked.valid and rng.random()<math.exp(checked.soft_score):
                return ResearchResult(result.tokens,calls,tuple(attempts),{'candidate_limit':candidate_limit})
        else:
            rank=(not checked.valid,len(checked.violations),-checked.soft_score)
            if best is None or rank<best[0]:
                best=(rank,result.tokens)
    if method.startswith('rejection_'):
        raise BudgetExceeded(f'rejection exhausted {candidate_limit} full reference proposals; no accepted sample')
    return ResearchResult(best[1],calls,tuple(attempts),{'candidate_limit':candidate_limit,'selection':'fewest violations, then soft score; heuristic'})
