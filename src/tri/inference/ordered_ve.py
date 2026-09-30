"""Compact music compilation followed by ordinary ordered bucket elimination.

This is a deliberately strong *standard-method* comparator, not a paired DP
under another name.  Equality components share binary onset variables.  Local
original-token variables are eliminated analytically: given onset and resulting
sounding state there is exactly one token (NOTE, REST or HOLD).  Equal count
constraints are deduplicated after that substitution.  Dense generic buckets
then use either a relation-aligned order, weighted min-fill, or chronological
order; no paired recurrence or tensor contraction is called.

The compiler supports general MusicSpec relations/counts, although arbitrary
layouts may have prohibitive induced width.  Observations have delta weight;
query clamps restrict choices while preserving the immutable full-130 q.
Saved bucket potentials support an exact backward joint draw.  A complete
original-token draw has a directly computable probability; projecting a draw
uses one additional clamped partition, not one VE per emitted token.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.special import logsumexp

from tri.domain.music import SILENCE, verify_music
from tri.errors import InvalidSpecification, VerificationError, ZeroMass
from tri.inference.exact import BatchSample, ExactInference
from tri.inference.factors import FactorGraph, LogFactor
from tri.inference.music_backends import MusicExactInference


@dataclass
class _Compilation:
    graph: FactorGraph
    onset_at: tuple[str, ...]
    aligned_order: tuple[str, ...]
    chronological_order: tuple[str, ...]
    stats: dict


class _SavedBuckets(ExactInference):
    """Generic dense log-space VE with retained backward-sampling potentials."""

    def __init__(self, graph, budget, order, external_bytes=0):
        super().__init__(graph, budget, order)
        self._external_bytes = external_bytes
        self._fixed = {v: domain[0] for v, domain in graph.domains.items() if len(domain) == 1}
        self._trace = None
        self._result = None
        self._trace_bytes = 0
        self._saved_stats = None

    def _combine_bucket(self, bucket, retained, stats, saved_bytes, reduce_variable=None):
        # Parent implementation accounts for immutable graph tensors, retained
        # factors, inputs and logsumexp temporaries. Add separately owned q and
        # saved bucket tensors before it performs pre-allocation budget checks.
        old = self._input_bytes
        self._input_bytes += self._external_bytes + saved_bytes
        try:
            return self._combine(bucket, retained, stats, reduce_variable)
        finally:
            self._input_bytes = old

    def _minfill_order(self, factors, remaining, keep):
        adjacency = {v: set() for v in remaining}
        for scope, _ in factors:
            for variable in scope:
                adjacency[variable].update(set(scope) - {variable})
        available = set(remaining) - set(keep)
        order = []
        while available:
            def cost(variable):
                neighbors = sorted(adjacency[variable])
                fill = sum(len(self.graph.domains[a]) * len(self.graph.domains[b])
                           for i, a in enumerate(neighbors) for b in neighbors[i + 1:]
                           if b not in adjacency[a])
                entries = len(self.graph.domains[variable]) * math.prod(len(self.graph.domains[n]) for n in neighbors)
                return fill, entries, variable
            chosen = min(available, key=cost)
            neighbors = adjacency.pop(chosen)
            for variable in neighbors:
                adjacency[variable].discard(chosen)
                adjacency[variable].update(neighbors - {variable})
            available.remove(chosen)
            order.append(chosen)
        return order

    def eliminate(self, keep=(), *, save=False):
        keep = tuple(v for v in keep if v not in self._fixed)
        factors = []
        for factor in self.graph.factors:
            index = tuple(0 if v in self._fixed else slice(None) for v in factor.scope)
            scope = tuple(v for v in factor.scope if v not in self._fixed)
            factors.append((scope, np.asarray(factor.values[index])))
        remaining = set(self.graph.domains) - set(self._fixed)
        if self.order is None:
            # Protect all requested variables, not just the scalar-marginal
            # interface's single `keep`. They still contribute to fill scores.
            order = self._minfill_order(factors, remaining, keep)
        else:
            order = [v for v in self.order if v in remaining and v not in keep]
        stats = {
            'elimination_order': order, 'max_factor_entries': max((f.values.size for f in self.graph.factors), default=1),
            'peak_workspace_bytes': self._input_bytes + self._external_bytes + self._trace_bytes,
            'factor_additions': 0, 'cache_hit': False,
        }
        trace = []
        saved_bytes = self._trace_bytes  # A retained partition trace may coexist with a marginal query.
        for variable in order:
            bucket = [f for f in factors if variable in f[0]]
            factors = [f for f in factors if variable not in f[0]]
            if not bucket:
                bucket = [((variable,), np.zeros(len(self.graph.domains[variable]), dtype=np.float64))]
            if save:
                scope, joint = self._combine_bucket(bucket, factors, stats, saved_bytes)
                axis = scope.index(variable)
                # Reserve scipy's reduction temporaries while the saved joint
                # and all existing trace potentials remain live.
                entries = joint.size
                output_entries = entries // len(self.graph.domains[variable])
                workspace = (self._input_bytes + self._external_bytes + saved_bytes +
                             sum(a.nbytes for _, a in factors + bucket) + 8 * (3 * entries + output_entries))
                self.budget.check_workspace_bytes(workspace, context='Saved VE reduction')
                stats['peak_workspace_bytes'] = max(stats['peak_workspace_bytes'], workspace)
                message = np.asarray(logsumexp(joint, axis=axis))
                trace.append((variable, scope, joint))
                saved_bytes += joint.nbytes
                factors.append((tuple(v for v in scope if v != variable), message))
            else:
                factors.append(self._combine_bucket(bucket, factors, stats, saved_bytes, variable))
        for variable in keep:
            if not any(variable in scope for scope, _ in factors):
                factors.append(((variable,), np.zeros(len(self.graph.domains[variable]))))
        scope, values = self._combine_bucket(factors, [], stats, saved_bytes)
        stats['saved_bucket_bytes'] = sum(a.nbytes for _, _, a in trace) if save else self._trace_bytes
        stats['input_factor_bytes'] = self._input_bytes
        self.last_stats = stats
        if save:
            self._trace, self._result = trace, float(values)
            self._trace_bytes = stats['saved_bucket_bytes']
            self._saved_stats = dict(stats)
        return scope, values

    def partition(self):
        if self._result is not None:
            self.last_stats = {**self._saved_stats, 'cache_hit': True}
            return self._result
        return float(self.eliminate(save=True)[1])

    def draw(self, rng):
        if not np.isfinite(self.partition()):
            raise ZeroMass('Cannot sample a zero-mass compiled graph')
        assignment = dict(self._fixed)
        for variable, scope, joint in reversed(self._trace):
            index = tuple(slice(None) if v == variable else self._indices[v][assignment[v]] for v in scope)
            weights = np.asarray(joint[index])
            normalizer = float(logsumexp(weights))
            if not np.isfinite(normalizer):
                raise ZeroMass('No supported backward bucket continuation')
            probabilities = np.exp(weights - normalizer)
            probabilities /= probabilities.sum()
            assignment[variable] = self.graph.domains[variable][int(rng.choice(len(probabilities), p=probabilities))]
        return assignment


class OrderedMusicVE(MusicExactInference):
    """Exact MusicSpec queries via compact factors and configurable VE order.

    ``order`` is ``aligned``, ``minfill`` or ``chronological``. All three share
    precisely the same compact compiler. Only the latest evidence graph and
    one backward-sampling trace are retained, so caches have a byte budget and
    cannot accumulate an unbounded family of message tensors.
    """

    def __init__(self, spec, log_probs, budget=None, order='aligned'):
        if order not in ('aligned', 'minfill', 'chronological'):
            raise InvalidSpecification('VE order must be aligned, minfill, or chronological')
        super().__init__(spec, log_probs, 'template', budget)
        self.order = order
        self.backend_name = self.requested_backend = 'ordered_ve_' + order
        self._compiled_key = None
        self._compiled = None
        self._buckets = None
        self._prepare({})
        self.planning_stats = {
            'selected_backend': self.backend_name, 'planner': order,
            'compiler': 'binary_onset_components_local_token_elimination',
            **self._compiled.stats,
        }

    def _compile(self, evidence):
        spec = self.spec
        choices, _ = self._choices(evidence)
        parents = list(range(spec.length))

        def root(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        for a, b in spec.equal_onsets:
            a, b = root(a), root(b)
            parents[max(a, b)] = min(a, b)
        components = {}
        for pos in range(spec.length):
            components.setdefault(root(pos), []).append(pos)
        onset_at = tuple(f'b{root(i)}' for i in range(spec.length))
        domains = {}
        factors = []
        ranks = {}
        times = {}
        stored_bytes = 0

        def factor(scope, values):
            nonlocal stored_bytes
            self.budget.check_factor_entries(values.size, context=f'Compact music factor {scope}')
            self.budget.check_workspace_bytes(self._base_bytes() + stored_bytes + 2 * values.nbytes,
                                               context='Compact music factor storage')
            factors.append(LogFactor(scope, values))
            stored_bytes += values.nbytes

        def zeros(scope):
            shape = tuple(len(domains[v]) for v in scope)
            entries = math.prod(shape)
            self.budget.check_factor_entries(entries, context=f'Compact music factor {scope}')
            self.budget.check_workspace_bytes(self._base_bytes() + stored_bytes + 2 * entries * 8,
                                               context='Compact music factor allocation')
            return np.full(shape, -np.inf, dtype=np.float64)

        def impossible():
            graph = FactorGraph({}, (LogFactor((), np.asarray(-np.inf)),))
            return _Compilation(graph, onset_at, (), (), {'input_factor_bytes': 8, 'compiled_variables': 0,
                'onset_components': len(components), 'count_chains': 0, 'zero_support': True})

        for representative, positions in components.items():
            bits = set.intersection(*({int(token >= 2) for token in choices[pos]} for pos in positions))
            if not bits:
                return impossible()
            name = f'b{representative}'
            domains[name] = tuple(sorted(bits))
            ranks[name] = (representative, 2, name)
            times[name] = (representative, 2, name)

        previous_states = (SILENCE if spec.initial_pitch is None else spec.initial_pitch,)
        for pos in range(spec.length):
            name, bit_name = f'h{pos}', onset_at[pos]
            transitions = []
            reachable = set()
            for old_index, previous in enumerate(previous_states):
                for token in choices[pos]:
                    bit = int(token >= 2)
                    if bit not in domains[bit_name]:
                        continue
                    edge = self._local(pos, previous, token)
                    if edge is not None:
                        current, soft_score = edge
                        log_weight = self._weight(0.0, self._q(pos, token), soft_score)
                        reachable.add(current)
                        transitions.append((old_index, bit, current, log_weight))
            if not reachable:
                return impossible()
            current_states = tuple(sorted(reachable))
            domains[name] = current_states
            current_indices = {value: index for index, value in enumerate(current_states)}
            bit_indices = {value: index for index, value in enumerate(domains[bit_name])}
            scope = (bit_name, name) if pos == 0 else (f'h{pos - 1}', bit_name, name)
            values = zeros(scope)
            for old_index, bit, current, weight in transitions:
                index = (bit_indices[bit], current_indices[current]) if pos == 0 else (old_index, bit_indices[bit], current_indices[current])
                # The token is unique for (onset, current); logaddexp also makes
                # the analytic local summation explicit without changing q.
                values[index] = np.logaddexp(values[index], weight)
            factor(scope, values)
            next_rank = root(pos + 1) if pos + 1 < spec.length else spec.length
            ranks[name] = (next_rank, 0, name)
            times[name] = (pos + 1, 0, name)
            previous_states = current_states

        canonical = {}
        for rule in spec.onset_counts:
            coefficients = Counter(onset_at[pos] for pos in rule.positions)
            target = rule.count
            for bit_name in list(coefficients):
                if len(domains[bit_name]) == 1:
                    target -= coefficients.pop(bit_name) * domains[bit_name][0]
            signature = tuple(sorted(coefficients.items(), key=lambda item: int(item[0][1:])))
            if signature in canonical and canonical[signature] != target:
                return impossible()
            canonical[signature] = target
        count_chains = 0
        for signature, target in canonical.items():
            total = sum(coefficient for _, coefficient in signature)
            if target < 0 or target > total:
                return impossible()
            if not signature:
                if target:
                    return impossible()
                continue
            previous_counts = (0,)
            consumed = 0
            for step, (bit_name, coefficient) in enumerate(signature):
                consumed += coefficient
                remaining = total - consumed
                counts = tuple(c for c in sorted({old + coefficient * bit for old in previous_counts for bit in domains[bit_name]})
                               if c <= target <= c + remaining)
                if not counts:
                    return impossible()
                name = f'c{count_chains}_{step}'
                domains[name] = counts
                indices = {value: index for index, value in enumerate(counts)}
                scope = (bit_name, name) if step == 0 else (f'c{count_chains}_{step-1}', bit_name, name)
                values = zeros(scope)
                for old_index, old in enumerate(previous_counts):
                    for bit_index, bit in enumerate(domains[bit_name]):
                        current = old + coefficient * bit
                        if current in indices:
                            index = (bit_index, indices[current]) if step == 0 else (old_index, bit_index, indices[current])
                            values[index] = 0.0
                factor(scope, values)
                next_rank = int(signature[step + 1][0][1:]) if step + 1 < len(signature) else spec.length
                ranks[name] = (next_rank, 1, name)
                times[name] = (next_rank, 1, name)
                previous_counts = counts
            count_chains += 1
        graph = FactorGraph(domains, tuple(factors))
        return _Compilation(graph, onset_at,
            tuple(sorted(domains, key=ranks.__getitem__)), tuple(sorted(domains, key=times.__getitem__)),
            {'input_factor_bytes': stored_bytes, 'compiled_variables': len(domains),
             'singleton_variables': sum(len(d) == 1 for d in domains.values()),
             'onset_components': len(components), 'count_chains': count_chains,
             'original_count_rules': len(spec.onset_counts), 'zero_support': False})

    def _prepare(self, evidence):
        key = tuple(sorted(evidence.items()))
        if key == self._compiled_key:
            return
        # Release the old graph/trace before allocating a new evidence graph.
        self._buckets = None
        self._compiled = None
        self._compiled_key = None
        compilation = self._compile(evidence)
        order = {'aligned': compilation.aligned_order, 'chronological': compilation.chronological_order,
                 'minfill': None}[self.order]
        buckets = _SavedBuckets(compilation.graph, self.budget, order, self._base_bytes())
        self._compiled, self._buckets, self._compiled_key = compilation, buckets, key

    def _stats(self, operation):
        self.last_stats = {'backend': self.backend_name, 'planner': self.order,
                           **self._compiled.stats, **self._buckets.last_stats, 'operation': operation}

    def log_partition(self, evidence=None):
        evidence = self._evidence(evidence)
        self._prepare(evidence)
        value = self._buckets.partition()
        self._stats('partition')
        return value

    def marginal_log_probs(self, variable, evidence=None):
        if variable not in self.graph.domains:
            raise InvalidSpecification(f'Unknown original-token variable {variable!r}')
        evidence = self._evidence(evidence)
        self._prepare(evidence)
        position = self._positions[variable]
        domain = self.graph.domains[variable]
        if variable in evidence or position in self.spec.observed:
            if not np.isfinite(self.log_partition(evidence)):
                raise ZeroMass('Conditioning event has zero mass')
            result = np.full(len(domain), -np.inf)
            token = evidence.get(variable, self.spec.observed.get(position))
            result[domain.index(token)] = 0.0
            return result
        if self._compiled.stats['zero_support']:
            raise ZeroMass('Conditioning event has zero mass')
        bit_name, state_name = self._compiled.onset_at[position], f'h{position}'
        scope, values = self._buckets.eliminate((bit_name, state_name))
        normalization = float(logsumexp(values))
        if not np.isfinite(normalization):
            raise ZeroMass('Conditioning event has zero mass')
        result = np.full(len(domain), -np.inf)
        for index in np.ndindex(values.shape):
            assignment = {**self._buckets._fixed,
                          **{name: self._compiled.graph.domains[name][i] for name, i in zip(scope, index)}}
            bit, state = assignment[bit_name], assignment[state_name]
            token = state + 2 if bit else (0 if state == SILENCE else 1)
            if token in domain:
                token_index = domain.index(token)
                result[token_index] = np.logaddexp(result[token_index], values[index] - normalization)
        self._stats('marginal')
        return result

    def sample_batch(self, variables: Sequence[str], rng, evidence=None):
        if isinstance(variables, str):
            raise InvalidSpecification('Batch variables must be a sequence')
        variables = tuple(variables)
        if len(variables) != len(set(variables)) or any(v not in self.graph.domains for v in variables):
            raise InvalidSpecification('Batch variables must be distinct original-token names')
        if not isinstance(rng, np.random.Generator):
            raise InvalidSpecification('Expected numpy Generator')
        evidence = self._evidence(evidence)
        base = self.log_partition(evidence)
        if not np.isfinite(base):
            raise ZeroMass('Cannot sample a zero-mass music query')
        if not variables:
            return BatchSample({}, 0.0, base)
        auxiliary = self._buckets.draw(rng)
        full = {}
        for position in range(self.spec.length):
            bit, state = auxiliary[self._compiled.onset_at[position]], auxiliary[f'h{position}']
            full[f'y{position}'] = state + 2 if bit else (0 if state == SILENCE else 1)
        checked = verify_music(tuple(full[f'y{i}'] for i in range(self.spec.length)), self.spec)
        if not checked.valid:
            raise VerificationError('Compact VE sample failed independent music verification: ' + '; '.join(checked.violations))
        if any(full[v] != value for v, value in evidence.items()):
            raise VerificationError('Compact VE sample failed temporary evidence')
        assignment = {v: full[v] for v in variables}
        sample_stats = {**self._compiled.stats, **self._buckets.last_stats}
        conditioned = {**evidence, **assignment}
        unknown = {f'y{i}' for i in range(self.spec.length) if i not in self.spec.observed}
        if unknown <= conditioned.keys():
            clamped = sum(self._q(i, full[f'y{i}']) for i in range(self.spec.length)) + checked.soft_score
            extra = False
        else:
            clamped = self.log_partition(conditioned)
            extra = True
        self.last_stats = {'backend': self.backend_name, 'planner': self.order, **sample_stats,
                           'operation': 'joint_backward_sample', 'projected_partition_query': extra,
                           'peak_workspace_bytes': max(sample_stats.get('peak_workspace_bytes', 0),
                                                       self.last_stats.get('peak_workspace_bytes', 0))}
        return BatchSample(assignment, clamped - base, clamped)
