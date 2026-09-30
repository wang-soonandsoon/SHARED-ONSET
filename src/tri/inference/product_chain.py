"""Standard factorial-HMM baseline for two relation-ordered music chains.

This is an independent numerical implementation, not a wrapper around paired.
It reuses MusicExactInference's immutable probability/query validation and local
music transition semantics. Corresponding onset labels select a Kronecker
product transition, evaluated as two matrix products rather than materializing
D**4 entries. This is the standard factorial-HMM/product-chain construction,
with O(L (K+1) D**3) arithmetic (O(L D**3) without a count constraint).

Products use row/column-scaled BLAS when all finite scaled inputs are within
300 log units. Otherwise a log-semiring contraction retains arbitrarily tiny
supported paths. No finite transition is truncated or q renormalized. Saved
forward messages support one-pass backward sampling and forward/backward token
marginals. These are standard HMM techniques, not claimed new optimizations.
"""

import math

import numpy as np
from scipy.special import logsumexp

from tri.domain.music import SILENCE, verify_music
from tri.errors import InvalidSpecification, UnsupportedSpec, VerificationError, ZeroMass
from tri.inference.exact import BatchSample
from tri.inference.music_backends import MusicExactInference


class ProductChainMusicInference(MusicExactInference):
    """Exact sum-product inference over a count-augmented product chain."""

    def __init__(self, spec, log_probs, budget=None):
        super().__init__(spec, log_probs, 'template', budget)
        relations = sorted({tuple(sorted(pair)) for pair in spec.equal_onsets})
        if not relations:
            raise UnsupportedSpec('product_chain needs two nonempty aligned spans')
        first, second = tuple(zip(*relations))
        if (any(a >= b for a, b in relations) or
                first != tuple(range(first[0], first[0] + len(first))) or
                second != tuple(range(second[0], second[0] + len(second))) or
                second[0] <= first[-1] + 1):
            raise UnsupportedSpec('product_chain needs ordered contiguous spans and an observed separator')
        self.spans = first, second
        self.width = len(first)
        self._inside = frozenset(first + second)
        self._outside_slots = tuple(i for i in range(spec.length) if i not in self._inside)
        if any(i not in spec.observed or i not in spec.fixed_soundings for i in self._outside_slots):
            raise UnsupportedSpec('product_chain needs observed tokens and sounding anchors outside spans')
        targets, flags = set(), {}
        self._contradiction = False
        for rule in spec.onset_counts:
            if not rule.positions:
                continue
            if rule.positions in self.spans:
                targets.add(rule.count)
            elif len(rule.positions) == 1 and rule.positions[0] in self._inside:
                pos = rule.positions[0]
                if pos in flags and flags[pos] != rule.count:
                    self._contradiction = True
                flags[pos] = rule.count
            else:
                raise UnsupportedSpec('product_chain counters must cover a complete span or a single span slot')
        self._contradiction |= len(targets) > 1
        self.target = min(targets) if targets else None
        self._flags = flags
        # A counter conveys no information if no total has been requested.
        self.count_size = 1 if self.target is None else self.target + 1
        carried = {value for value in (spec.initial_pitch, spec.end_pitch, *spec.fixed_soundings.values())
                   if value is not None}
        self.pitches = (SILENCE, *sorted(set(spec.pitches) | carried))
        self.size = len(self.pitches)
        self._pitch_indices = {pitch: index for index, pitch in enumerate(self.pitches)}
        self.backend_name = self.requested_backend = 'product_chain'
        state_entries = self.count_size * self.size ** 2
        contraction_entries = self.size ** 3
        self.budget.check_factor_entries(max(state_entries, contraction_entries), context='Product-chain contraction')
        self.storage_bytes = self._base_bytes() + 8 * (
            (self.width + 5) * state_entries + 4 * self.width * self.size ** 2 +
            4 * contraction_entries + 12 * self.size ** 2)
        self.budget.check_workspace_bytes(self.storage_bytes, context='Product-chain messages and contraction workspace')
        self.planning_stats = {
            'selected_backend': 'product_chain', 'algorithm': 'factorial_hmm',
            'width': self.width, 'count_states': self.count_size,
            'pitch_states_per_span': self.size, 'state_entries': state_entries,
            'contraction_entries': contraction_entries,
            'numeric_kernel': 'guarded_scaled_blas_with_log_semiring_fallback',
        }

    def _stats(self):
        return {**self.planning_stats, 'backend': self.backend_name, 'cache_hit': False,
                'peak_workspace_bytes': self.storage_bytes, 'blas_products': 0,
                'log_semiring_products': 0}

    @staticmethod
    def _log_matrix_product(left, right, stats):
        """Log(exp(left) @ exp(right)), using safe per-row/column gauges."""
        row = np.max(left, axis=1)
        col = np.max(right, axis=0)
        # All-zero rows/columns are represented by -inf, not NaN gauges.
        row = np.where(np.isfinite(row), row, 0.0)
        col = np.where(np.isfinite(col), col, 0.0)
        a, b = left - row[:, None], right - col[None, :]
        fa, fb = a[np.isfinite(a)], b[np.isfinite(b)]
        if (not fa.size or np.min(fa) >= -300.0) and (not fb.size or np.min(fb) >= -300.0):
            # Every nonzero product is >= exp(-600), safely above underflow.
            product = np.exp(a) @ np.exp(b)
            with np.errstate(divide='ignore'):
                result = np.log(product) + row[:, None] + col[None, :]
            stats['blas_products'] += 1
            return result
        stats['log_semiring_products'] += 1
        return logsumexp(left[:, :, None] + right[None, :, :], axis=1)

    def _compile_query(self, evidence):
        """Independent product-chain layout; only _local/_q are shared."""
        choices, _ = self._choices(evidence)
        beginnings = []
        endings = []
        guard_slots = set()
        for span in self.spans:
            before = self.spec.initial_pitch if span[0] == 0 else self.spec.fixed_soundings[span[0] - 1]
            beginnings.append(self._pitch_indices[SILENCE if before is None else before])
            after = span[-1] + 1
            tail = np.zeros(self.size)
            if after < self.spec.length:
                guard_slots.add(after)
                tail[:] = -np.inf
                for old, pitch in enumerate(self.pitches):
                    edge = self._local(after, pitch, self.spec.observed[after])
                    if edge is not None:
                        tail[old] = edge[1]
            endings.append(tail)
        constant = 0.0
        for pos in self._outside_slots:
            if pos in guard_slots:
                continue
            before = self.spec.initial_pitch if pos == 0 else self.spec.fixed_soundings[pos - 1]
            edge = self._local(pos, SILENCE if before is None else before, self.spec.observed[pos])
            if edge is None:
                constant = -np.inf
                break
            constant += edge[1]
        matrices = []
        for pair in zip(*self.spans):
            local_pair = []
            for pos in pair:
                block = np.full((2, self.size, self.size), -np.inf)
                for old, pitch in enumerate(self.pitches):
                    for token in choices[pos]:
                        bit = int(token >= 2)
                        if pos in self._flags and self._flags[pos] != bit:
                            continue
                        edge = self._local(pos, pitch, token)
                        if edge is not None:
                            new = self._pitch_indices[edge[0]]
                            block[bit, old, new] = np.logaddexp(block[bit, old, new], self._q(pos, token) + edge[1])
                local_pair.append(block)
            matrices.append(tuple(local_pair))
        if self._contradiction or any(not c for c in choices):
            constant = -np.inf
        return choices, tuple(beginnings), tuple(endings), constant, matrices

    def _next_count(self, count, bit):
        return 0 if self.target is None else count + bit

    def _forward_chain(self, evidence, save):
        stats = self._stats()
        query = self._compile_query(evidence)
        _, beginnings, endings, constant, matrices = query
        alpha = np.full((self.count_size, self.size, self.size), -np.inf)
        alpha[0, beginnings[0], beginnings[1]] = 0.0
        history = [alpha] if save else None
        if math.isfinite(constant):
            for step, (first, second) in enumerate(matrices):
                following = np.full_like(alpha, -np.inf)
                for count in range(self.count_size):
                    if not np.isfinite(alpha[count]).any():
                        continue
                    for bit in (0, 1):
                        nc = self._next_count(count, bit)
                        if nc >= self.count_size or (self.target is not None and nc + self.width - step - 1 < self.target):
                            continue
                        # Eliminate B's old state, then A's: standard factored
                        # transition, with no materialized Kronecker matrix.
                        part = self._log_matrix_product(alpha[count], second[bit], stats)
                        part = self._log_matrix_product(first[bit].T, part, stats)
                        following[nc] = np.logaddexp(following[nc], part)
                alpha = following
                if save:
                    history.append(alpha)
        else:
            alpha[:] = -np.inf
        final = alpha[-1] + endings[0][:, None] + endings[1][None, :]
        normalizer = float(logsumexp(final)) + constant
        stats['saved_message_bytes'] = sum(layer.nbytes for layer in history) if save else 0
        stats['live_message_bytes'] = 2 * alpha.nbytes
        stats['transfer_bytes'] = sum(block.nbytes for pair in matrices for block in pair)
        return normalizer, history, final, query, stats

    def log_partition(self, evidence=None):
        evidence = self._evidence(evidence)
        key = tuple(sorted(evidence.items()))
        if key in self._cache:
            result, stats = self._cache[key]
            self._cache.move_to_end(key)
            self.last_stats = {**stats, 'cache_hit': True}
            return result
        result, _, _, _, stats = self._forward_chain(evidence, save=False)
        self._cache[key] = (result, dict(stats))
        if len(self._cache) > 128:
            self._cache.popitem(last=False)
        self.last_stats = stats
        return result

    @staticmethod
    def _choose_index(weights, rng):
        z = float(logsumexp(weights))
        if not math.isfinite(z):
            raise ZeroMass('No supported product-chain continuation')
        probabilities = np.exp(weights.ravel() - z)
        probabilities /= probabilities.sum()
        index = int(rng.choice(probabilities.size, p=probabilities))
        return tuple(int(i) for i in np.unravel_index(index, weights.shape))

    def marginal_log_probs(self, variable, evidence=None):
        if variable not in self.graph.domains:
            raise InvalidSpecification(f'Unknown original-token variable {variable!r}')
        evidence = self._evidence(evidence)
        z, forward, _, query, stats = self._forward_chain(evidence, save=True)
        if not math.isfinite(z):
            raise ZeroMass('Conditioning event has zero mass')
        domain = self.graph.domains[variable]
        pos = self._positions[variable]
        values = np.full(len(domain), -np.inf)
        if variable in evidence or pos in self.spec.observed:
            token = evidence.get(variable, self.spec.observed.get(pos))
            values[domain.index(token)] = 0.0
            self.last_stats = stats
            return values
        choices, _, endings, constant, matrices = query
        which = 0 if pos in self.spans[0] else 1
        offset = self.spans[which].index(pos)
        beta = np.full((self.count_size, self.size, self.size), -np.inf)
        beta[-1] = endings[0][:, None] + endings[1][None, :]
        for step in range(self.width - 1, offset, -1):
            first, second = matrices[step]
            preceding = np.full_like(beta, -np.inf)
            for count in range(self.count_size):
                for bit in (0, 1):
                    nc = self._next_count(count, bit)
                    if nc >= self.count_size:
                        continue
                    part = self._log_matrix_product(first[bit], beta[nc], stats)
                    part = self._log_matrix_product(part, second[bit].T, stats)
                    preceding[count] = np.logaddexp(preceding[count], part)
            beta = preceding
        first, second = matrices[offset]
        # Edge beliefs excluding the queried chain's own local transition.
        edge_beliefs = np.full((2, self.size, self.size), -np.inf)
        for count in range(self.count_size):
            for bit in (0, 1):
                nc = self._next_count(count, bit)
                if nc >= self.count_size:
                    continue
                if which == 0:
                    part = self._log_matrix_product(forward[offset][count], second[bit], stats)
                    edge = self._log_matrix_product(part, beta[nc].T, stats)
                else:
                    part = self._log_matrix_product(first[bit].T, forward[offset][count], stats)
                    edge = self._log_matrix_product(part.T, beta[nc], stats)
                edge_beliefs[bit] = np.logaddexp(edge_beliefs[bit], edge)
        for index, token in enumerate(domain):
            bit = int(token >= 2)
            if token not in choices[pos] or (pos in self._flags and self._flags[pos] != bit):
                continue
            mass = []
            for old, pitch in enumerate(self.pitches):
                edge = self._local(pos, pitch, token)
                if edge is not None:
                    new = self._pitch_indices[edge[0]]
                    mass.append(edge_beliefs[bit, old, new] + self._q(pos, token) + edge[1])
            values[index] = float(logsumexp(mass)) + constant - z
        self.last_stats = {**stats, 'query': 'forward_backward_marginal'}
        return values

    def sample_batch(self, variables, rng, evidence=None):
        if isinstance(variables, str):
            raise InvalidSpecification('Batch variables must be a sequence')
        variables = tuple(variables)
        if len(set(variables)) != len(variables) or any(v not in self.graph.domains for v in variables):
            raise InvalidSpecification('Batch variables must be distinct original-token names')
        if not isinstance(rng, np.random.Generator):
            raise InvalidSpecification('Expected numpy Generator')
        evidence = self._evidence(evidence)
        z, history, final, query, stats = self._forward_chain(evidence, save=True)
        if not math.isfinite(z):
            raise ZeroMass('Cannot sample a zero-mass product chain')
        if not variables:
            self.last_stats = stats
            return BatchSample({}, 0.0, z)
        _, _, _, _, matrices = query
        current_a, current_b = self._choose_index(final, rng)
        count = self.count_size - 1
        sequence = [self.spec.observed.get(i) for i in range(self.spec.length)]
        for step in range(self.width - 1, -1, -1):
            first, second = matrices[step]
            incoming = np.full((2, self.size, self.size), -np.inf)
            for bit in (0, 1):
                previous_count = 0 if self.target is None else count - bit
                if previous_count < 0:
                    continue
                incoming[bit] = (history[step][previous_count] +
                                 first[bit, :, current_a][:, None] + second[bit, :, current_b][None, :])
            bit, previous_a, previous_b = self._choose_index(incoming, rng)
            # Within a fixed onset bit and old/new sounding state, this token
            # vocabulary has exactly one emitting edge (no path multiplicity).
            for span, state in zip(self.spans, (current_a, current_b)):
                sequence[span[step]] = self.pitches[state] + 2 if bit else (0 if self.pitches[state] == SILENCE else 1)
            current_a, current_b = previous_a, previous_b
            count = 0 if self.target is None else count - bit
        checked = verify_music(sequence, self.spec)
        if not checked.valid:
            raise VerificationError(f'Product-chain sample failed independent checker: {checked.violations}')
        assignment = {name: int(sequence[self._positions[name]]) for name in variables}
        clamped_evidence = {**evidence, **assignment}
        # The probability of a fully specified original sequence is directly
        # available; a projected subset still requires summing its complement.
        covered = {self._positions[name] for name in clamped_evidence} | set(self.spec.observed)
        if len(covered) == self.spec.length:
            clamped = checked.soft_score + sum(self._q(i, sequence[i]) for i in range(self.spec.length))
            stats['sample_probability'] = 'complete_sequence_weight'
        else:
            clamped = self.log_partition(clamped_evidence)
            stats['sample_probability'] = 'projected_partition'
            stats['projection_stats'] = self.last_stats
        self.last_stats = {**stats, 'query': 'forward_backward_joint_sample'}
        return BatchSample(assignment, clamped - z, clamped)
