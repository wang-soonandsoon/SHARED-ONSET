"""Measured engineering improvements to the original paired log-space DP.

The probability target and recurrence are unchanged. An immutable unconditional
trace supports repeated draws and evidence-prefix reuse. A separate sparse flag
only changes the exact contraction kernel; finite edges are never truncated.
These are caching/sparse-linear-algebra techniques, not a new inference law.
"""
import math

import numpy as np
from scipy.special import logsumexp

from tri.domain.music import verify_music
from tri.errors import InvalidSpecification, VerificationError, ZeroMass
from tri.inference.exact import BatchSample
from tri.inference.paired import PairedMusicInference


class ReusedPairedInference(PairedMusicInference):
    def __init__(self, spec, log_probs, budget=None, *, sparse=False):
        super().__init__(spec, log_probs, budget)
        self.backend_name = self.requested_backend = 'paired_sparse' if sparse else 'paired_reuse'
        self._base_trace = None
        self._base_choices = None
        self._base_plans = {}
        self._plans = {}
        # Reserve BOTH retained base and a full changed suffix, transfer/index
        # arrays, Python plan metadata, and original dense contraction workspace.
        plan_bytes = 64 * self.width * self.size**2 + 4096 * self.width if sparse else 0
        additional_trace = 8 * ((self.width+1)*(self.count_limit+1)*self.size**2 +
                                4*self.width*self.size**2)
        self._cache_bound = self.storage_bytes + additional_trace + plan_bytes
        self._cache_enabled = self._cache_bound <= self.budget.max_workspace_bytes
        self._sparse_enabled = sparse and self.storage_bytes + plan_bytes <= self.budget.max_workspace_bytes
        self._workspace_bound = (self._cache_bound if self._cache_enabled else
                                 self.storage_bytes + (plan_bytes if self._sparse_enabled else 0))
        self._kernel_counts = {'sparse_products': 0, 'dense_products': 0}

    def _incoming_plan(self, matrix):
        # Retain the view in each value: an allocator cannot recycle its address
        # while a cached plan is live. Array ids of temporary views are unsafe.
        key = (matrix.__array_interface__['data'][0], matrix.shape, matrix.strides)
        if key in self._plans:
            return self._plans[key][1]
        mask = np.isfinite(matrix)
        if self.size < 4 or np.count_nonzero(mask) > matrix.size / 3:
            plan = None
        else:
            degree = np.sum(mask, axis=0)
            plan = []
            for n in sorted(set(degree.tolist()) - {0}):
                columns = np.flatnonzero(degree == n)
                parents = np.stack([np.flatnonzero(mask[:, column]) for column in columns])
                weights = matrix[parents, columns[:, None]]
                plan.append((columns, parents, weights))
        self._plans[key] = (matrix, plan)
        return plan

    def _left_product(self, alpha, matrix):
        plan = self._incoming_plan(matrix)
        if plan is None:
            self._kernel_counts['dense_products'] += 1
            return logsumexp(matrix.T[:, :, None] + alpha[None, :, :], axis=1)
        self._kernel_counts['sparse_products'] += 1
        result = np.full_like(alpha, -np.inf)
        # Group by indegree so REST's one dense destination does not pad every
        # HOLD destination to D parents. Work is O(D * nnz(matrix)).
        for columns, parents, weights in plan:
            result[columns] = logsumexp(alpha[parents] + weights[:, :, None], axis=1)
        return result

    def _contract(self, alpha, first, second):
        if not self._sparse_enabled:
            self._kernel_counts['dense_products'] += 2
            return PairedMusicInference._contract(alpha, first, second)
        part = self._left_product(alpha, first)
        return self._left_product(part.T, second).T

    def _one_transfer(self, pos, choices):
        matrix = np.full((2, self.size, self.size), -np.inf)
        for token in choices:
            bit = int(token >= 2)
            if pos in self.forced_bits and bit != self.forced_bits[pos]:
                continue
            for old, previous in enumerate(self.pitches):
                edge = self._local(pos, previous, token)
                if edge is not None:
                    new = self.pitch_index[edge[0]]
                    matrix[bit, old, new] = np.logaddexp(matrix[bit, old, new], self._q(pos, token)+edge[1])
        return matrix

    def _stats(self, **extra):
        return {**self.planning_stats, 'selected_backend': self.backend_name, 'backend': self.backend_name,
                'cache_hit': False, 'message_cache_enabled': self._cache_enabled,
                'sparse_enabled': self._sparse_enabled, 'peak_workspace_bytes': self._workspace_bound,
                **self._kernel_counts, **extra}

    def _trace(self, evidence):
        choices, _ = self._choices(evidence)
        self._kernel_counts = {'sparse_products': 0, 'dense_products': 0}
        constant, initial, terminal = self._outside()
        if self.contradictory or any(not c for c in choices) or None in initial or not math.isfinite(constant):
            return (-np.inf, None, None, None,
                    self._stats(prefix_layers_reused=0, transfers_rebuilt=0, zero_support=True))
        if not evidence and self._base_trace is not None:
            z, layers, transfers, final, stats = self._base_trace
            return z, layers, transfers, final, {**stats, 'cache_hit': True, 'message_cache_hit': True,
                'prefix_layers_reused': self.width, 'transfers_rebuilt': 0,
                'sparse_products': 0, 'dense_products': 0}
        base = self._base_trace
        self._plans = dict(self._base_plans) if base is not None else {}
        first_change = self.width
        rebuilt = 0
        transfers = []
        for step, pair in enumerate(zip(*self.spans)):
            matrices = []
            for side, pos in enumerate(pair):
                if base is not None and choices[pos] == self._base_choices[pos]:
                    matrices.append(base[2][step][side])
                else:
                    matrices.append(self._one_transfer(pos, choices[pos]))
                    rebuilt += 1
                    first_change = min(first_change, step)
            transfers.append(matrices)
        if base is None:
            first_change = 0
            alpha = np.full((self.count_limit+1, self.size, self.size), -np.inf)
            alpha[0, initial[0], initial[1]] = 0.0
            layers = [alpha]
        else:
            # ALWAYS branch from unconditional messages, never the preceding
            # query's suffix. Original arrays are not modified in place.
            layers = list(base[1][:first_change+1])
            alpha = layers[-1]
        for step in range(first_change, self.width):
            first, second = transfers[step]
            following = np.full_like(alpha, -np.inf)
            for count in range(min(step+1, self.count_limit)+1):
                if count <= step and np.isfinite(alpha[count]).any():
                    following[count] = self._contract(alpha[count], first[0], second[0])
                if count and np.isfinite(alpha[count-1]).any():
                    following[count] = np.logaddexp(following[count],
                        self._contract(alpha[count-1], first[1], second[1]))
            alpha = following
            layers.append(alpha)
        final = alpha + terminal[0][None, :, None] + terminal[1][None, None, :]
        if self.target is not None:
            final[:self.target] = -np.inf
        z = float(logsumexp(final)) + constant
        stats = self._stats(prefix_layers_reused=first_change, transfers_rebuilt=rebuilt,
                            saved_message_bytes=sum(layer.nbytes for layer in layers),
                            retained_base_message_bytes=sum(layer.nbytes for layer in base[1]) if base else 0)
        trace = z, layers, transfers, final, stats
        if not evidence and self._cache_enabled:
            for array in (*layers, final, *(matrix for pair in transfers for matrix in pair)):
                array.flags.writeable = False
            self._base_trace = trace
            self._base_choices = choices
            self._base_plans = dict(self._plans)
        # A nonempty first query is computed directly; it never builds a larger
        # unconditional trace merely to enable caching.
        self._plans = dict(self._base_plans)
        return trace

    def log_partition(self, evidence=None):
        evidence = self._evidence(evidence)
        key = tuple(sorted(evidence.items()))
        if key in self._cache:
            z, stats = self._cache[key]
            self._cache.move_to_end(key)
            self.last_stats = {**stats, 'cache_hit': True, 'transfers_rebuilt': 0,
                               'sparse_products': 0, 'dense_products': 0}
            return z
        z, _, _, _, stats = self._trace(evidence)
        self._cache[key] = z, dict(stats)
        if len(self._cache) > 128:
            self._cache.popitem(last=False)
        self.last_stats = stats
        return z

    def sample_batch(self, variables, rng, evidence=None):
        if isinstance(variables, str):
            raise InvalidSpecification('Batch variables must be a sequence')
        variables = tuple(variables)
        if len(set(variables)) != len(variables) or any(v not in self.graph.domains for v in variables):
            raise InvalidSpecification('Batch variables must be distinct original-token names')
        if not isinstance(rng, np.random.Generator):
            raise InvalidSpecification('Expected numpy Generator')
        evidence = self._evidence(evidence)
        base, layers, transfers, final, stats = self._trace(evidence)
        if not math.isfinite(base):
            raise ZeroMass('Cannot sample a zero-mass paired query')
        if not variables:
            self.last_stats = stats
            return BatchSample({}, 0.0, base)
        count, current_a, current_b = self._draw_array(final, rng)
        full = [self.spec.observed.get(i) for i in range(self.spec.length)]
        for step in range(self.width-1, -1, -1):
            first, second = transfers[step]
            incoming = np.full((2, self.size, self.size), -np.inf)
            for bit in (0, 1):
                if count >= bit:
                    incoming[bit] = (layers[step][count-bit] + first[bit, :, current_a][:, None] +
                                     second[bit, :, current_b][None, :])
            bit, old_a, old_b = self._draw_array(incoming, rng)
            for span, current in zip(self.spans, (current_a, current_b)):
                full[span[step]] = self.pitches[current]+2 if bit else (0 if current == 0 else 1)
            current_a, current_b, count = old_a, old_b, count-bit
        checked = verify_music(full, self.spec)
        if not checked.valid or any(full[self._positions[name]] != value for name, value in evidence.items()):
            raise VerificationError('Cached paired sample violates music or query evidence')
        assignment = {name: int(full[self._positions[name]]) for name in variables}
        combined = {**evidence, **assignment}
        unknown = {f'y{i}' for i in range(self.spec.length) if i not in self.spec.observed}
        if unknown <= combined.keys():
            clamped = checked.soft_score + sum(self._q(i, token) for i, token in enumerate(full))
            probability_mode = 'complete_sequence_weight'
        else:
            # A conditional sample's suffix is not the retained base. Drop it
            # before allocating the projection suffix: at most base + ONE
            # query trace may coexist under the cache workspace bound.
            del layers, transfers, final, first, second, incoming
            clamped = self.log_partition(combined)
            probability_mode = 'projected_partition'
        self.last_stats = {**stats, 'sample_probability': probability_mode}
        return BatchSample(assignment, clamped-base, clamped)
