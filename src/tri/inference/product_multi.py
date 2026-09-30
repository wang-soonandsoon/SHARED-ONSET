"""Standard factorized product-state HMM for R aligned melody spans.

A shared onset bit selects R local pitch-transition matrices. Each forward
step contracts one old-pitch axis at a time, never constructing a D**(2R)
transition. Arithmetic is O(L C R D**(R+1)), where C=K+1 (C=1 without a count).
Saved FFBS history is O(L C D**R), with O(L R D**2) local transitions.
When observations, finite q support, flags and K determine the full rhythm,
standard conditional independence dispatches to R separate chain FFBS passes
in O(R L D**2) time and O(R L D + R L D**2) cached storage. No reset
assumption is needed, so this dispatch retains motion and interval constraints.

This numerical recurrence is independent of onset-reset interval inference.
Only its pure structural layout/relationship helpers are shared; this class
never constructs an onset-reset engine and never removes motion or hard
interval rules. All original full130 q, observation deltas, temporary evidence,
HOLD/sounding boundaries and local soft scores retain their original meaning.

Each engine keeps one immutable evidence-keyed forward/transition trace. Query
replacement releases that trace before allocating another. Per-axis matrix
products use guarded scaled BLAS with blocked log-semiring fallback, retaining
finite extreme weights without allocating D**(R+1) fallback temporaries.
"""
from dataclasses import dataclass
import math
from time import perf_counter
from types import MappingProxyType

import numpy as np
from scipy.special import logsumexp

from tri.domain.music import SILENCE, verify_music
from tri.errors import InvalidSpecification, UnsupportedSpec, VerificationError, ZeroMass
from tri.inference.exact import BatchSample
from tri.inference.music_backends import MusicExactInference
from tri.inference.onset_reset import OnsetResetMusicInference


@dataclass(frozen=True)
class _ProductTrace:
    key: tuple
    choices: tuple
    matrices: tuple
    tails: tuple
    constant: float
    history: tuple
    final: np.ndarray
    log_z: float
    stats: object
    known_bits: tuple | None = None


class MultiSpanProductChainMusicInference(MusicExactInference):
    """Exact fixed-q product-chain queries, including nonzero motion scores."""

    # Shared input/layout plumbing only, not reset restrictions or numerics.
    _query_metadata_bytes = OnsetResetMusicInference._query_metadata_bytes
    _build_relations = OnsetResetMusicInference._build_relations
    _sounding = staticmethod(OnsetResetMusicInference._sounding)

    def __init__(self, spec, logq, budget=None):
        super().__init__(spec, logq, 'template', budget)
        try:
            OnsetResetMusicInference._compile_layout(self)
        except UnsupportedSpec as error:
            raise UnsupportedSpec(str(error).replace('onset_reset', 'product_multi')) from error
        self.backend_name = self.requested_backend = 'product_multi'
        self.count_size = 1 if self.K is None else self.K + 1
        self.joint_shape = (self.D,) * self.R
        self.joint_states = self.D ** self.R
        self.state_entries = self.count_size * self.joint_states
        self._prepared = None
        self.storage_bytes = None
        self._kernel_entries = 0
        self.planning_stats = {'selected_backend': self.backend_name, 'algorithm': 'factorial_hmm_multi_axis',
            'spans': self.R, 'span_count': self.R, 'span_length': self.L, 'pitch_states': self.D,
            'count_states': self.count_size, 'onset_count': self.K, 'state_entries': self.state_entries,
            'joint_pitch_states': self.joint_states,
            'numeric_kernel': 'blocked_guarded_scaled_blas_with_log_semiring_fallback',
            'cartesian_state_entries': self.state_entries, 'dispatch': 'unprepared'}
        self.last_stats = self._stats()

    def _plan_workspace(self, known_bits=None):
        """Bound retained trace plus the largest forward/backward/query workspace.

        Forward layers are separate C*D**R arrays, not one concatenated factor.
        Kernel temporaries are blocked by real factor/workspace allowances.
        D**R exhaustion is BudgetExceeded, never an unsupported-layout result.
        """
        known = known_bits is not None
        actual_arrays = ((self.L + 1) * self.D, 2 * self.D ** 2) if known else (
            self.state_entries, 2 * self.D ** 2, 2 * self.joint_states)
        for entries in actual_arrays:
            self.budget.check_factor_entries(entries, context='Multi-product actual state/transfer/sampling array')
        history = self.R * (self.L + 1) * self.D if known else (self.L + 1) * self.state_entries
        transfers = 2 * self.L * self.R * self.D ** 2
        tails = self.R * self.D
        cached_arrays = history + transfers + tails + (self.R * self.D if known else self.joint_states)
        metadata = self._metadata_allowance()
        # Cached history remains live for FFBS and marginals. Known templates
        # retain separate chains, so no Cartesian pitch state is allocated.
        scratch = (6 * (self.L + 1) * self.D + 12 * self.D ** 2 if known else
                   4 * self.state_entries + 10 * self.joint_states + 12 * self.D ** 2)
        fixed = self._base_bytes() + metadata + 8 * (cached_arrays + scratch)
        minimum_kernel = self.D ** 2
        self.budget.check_factor_entries(minimum_kernel, context='Multi-product minimum matrix block')
        self.budget.check_workspace_bytes(fixed + 8 * 8 * minimum_kernel,
                                         context='Multi-product cached FFBS plus query workspace')
        available = (self.budget.max_workspace_bytes - fixed) // (8 * 8)
        kernel_cap = self.D ** 2 if known else min(1024 * self.D ** 2, self.joint_states * self.D)
        self._kernel_entries = min(self.budget.max_factor_entries, available, kernel_cap)
        self.storage_bytes = fixed + 8 * 8 * self._kernel_entries
        self._cached_array_bound = cached_arrays * 8
        self.budget.check_workspace_bytes(self.storage_bytes, context='Multi-product cache and blocked contractions')
        self.planning_stats.update(dispatch='known_template_chains' if known else 'multi_axis_product',
            product_state_materialized=not known, state_entries=self.D if known else self.state_entries,
            kernel_entry_limit=self._kernel_entries, workspace_bytes_estimate=self.storage_bytes,
            peak_workspace_bytes=self.storage_bytes)

    def _metadata_allowance(self):
        return self.spec.length * (self.D + 2) * 64 + self.L * self.R * 512 + (self.L + 1) * 256

    def _fixed_template(self, choices):
        allowed = []
        contradiction = self._contradiction
        for step, positions in enumerate(zip(*self.spans)):
            bits = {0, 1} if step not in self._flags else {self._flags[step]}
            for position in positions:
                bits &= {int(token >= 2) for token in choices[position]}
            contradiction |= not bits
            allowed.append(bits)
        if contradiction:
            return None, True
        minimum = sum(bits == {1} for bits in allowed)
        maximum = sum(1 in bits for bits in allowed)
        if self.K is not None:
            if not minimum <= self.K <= maximum:
                return None, True
            if minimum == self.K or maximum == self.K:
                forced = 0 if minimum == self.K else 1
                allowed = [{forced} if len(bits) == 2 else bits for bits in allowed]
        return (tuple(next(iter(bits)) for bits in allowed) if all(len(bits) == 1 for bits in allowed)
                else None), False

    def _stats(self):
        return {**getattr(self, 'planning_stats', {}), 'backend': self.backend_name,
                'cache_hit': False, 'message_cache_hit': False,
                'blas_products': 0, 'log_semiring_products': 0,
                'axis_contractions': 0, 'maximum_kernel_entries': 0}

    def _log_matmul(self, left, right, stats):
        """General rectangular log-matmul with bounded row/inner blocks.

        Gauges match the two-span standard implementation. Finite scaled
        entries below -300 trigger logsumexp; hence BLAS never drops finite
        products through exp/product underflow (each is at least exp(-600)).
        Splitting the inner axis combines exact partial sums with logaddexp.
        """
        m, inner = left.shape
        if right.shape[0] != inner:
            raise AssertionError('Product contraction shape mismatch')
        n = right.shape[1]
        result = np.full((m, n), -np.inf)
        rows = min(m, 1024, max(1, self._kernel_entries // n))
        columns = min(inner, max(1, self._kernel_entries // (rows * n)))
        for start in range(0, m, rows):
            end = min(m, start + rows)
            for first in range(0, inner, columns):
                last = min(inner, first + columns)
                a, b = left[start:end, first:last], right[first:last]
                entries = (end - start) * (last - first) * n
                stats['maximum_kernel_entries'] = max(stats['maximum_kernel_entries'], entries)
                row = np.max(a, axis=1)
                col = np.max(b, axis=0)
                row = np.where(np.isfinite(row), row, 0.)
                col = np.where(np.isfinite(col), col, 0.)
                aa, bb = a - row[:, None], b - col[None, :]
                fa, fb = aa[np.isfinite(aa)], bb[np.isfinite(bb)]
                if (not fa.size or np.min(fa) >= -300.) and (not fb.size or np.min(fb) >= -300.):
                    mass = np.exp(aa) @ np.exp(bb)
                    with np.errstate(divide='ignore', over='raise', invalid='raise'):
                        value = np.log(mass) + row[:, None] + col[None, :]
                    stats['blas_products'] += 1
                else:
                    with np.errstate(over='raise', invalid='raise'):
                        value = logsumexp(a[:, :, None] + b[None, :, :], axis=1)
                    stats['log_semiring_products'] += 1
                result[start:end] = np.logaddexp(result[start:end], value)
        return result

    def _contract_axis(self, values, matrix, axis, stats):
        moved = np.moveaxis(values, axis, -1)
        result = self._log_matmul(moved.reshape(-1, self.D), matrix, stats)
        stats['axis_contractions'] += 1
        return np.moveaxis(result.reshape(moved.shape), -1, axis)

    def _tail_tensor(self, tails):
        result = np.zeros(self.joint_shape)
        for axis, tail in enumerate(tails):
            shape = [1] * self.R
            shape[axis] = self.D
            result += tail.reshape(shape)
        return result

    def _next_count(self, count, bit):
        return 0 if self.K is None else count + bit

    def _compile_query(self, evidence, choices):
        constant = -math.inf if self._contradiction or any(not values for values in choices) else 0.
        for position in self._outside:
            if position in self._guards:
                continue
            previous = self._sounding(self.spec.initial_pitch if position == 0
                                      else self.spec.fixed_soundings[position - 1])
            edge = self._local(position, previous, self.spec.observed[position])
            constant += edge[1] if edge is not None else -math.inf
        tails = []
        for span in self.spans:
            position = span[-1] + 1
            tail = np.zeros(self.D)
            if position < self.spec.length:
                for old, pitch in enumerate(self._pitches):
                    edge = self._local(position, pitch, self.spec.observed[position])
                    tail[old] = edge[1] if edge is not None else -math.inf
            tails.append(tail)
        matrices = []
        for step, positions in enumerate(zip(*self.spans)):
            local = []
            for position in positions:
                block = np.full((2, self.D, self.D), -np.inf)
                for old, pitch in enumerate(self._pitches):
                    for token in choices[position]:
                        bit = int(token >= 2)
                        if step in self._flags and bit != self._flags[step]:
                            continue
                        edge = self._local(position, pitch, token)
                        if edge is not None:
                            new = self._pitch_index[edge[0]]
                            block[bit, old, new] = np.logaddexp(block[bit, old, new], self._q(position, token) + edge[1])
                local.append(block)
            matrices.append(tuple(local))
        return choices, tuple(matrices), tuple(tails), constant

    def _prepare(self, evidence=None):
        evidence = self._evidence(evidence)
        key = tuple(sorted(evidence.items()))
        if self._prepared is not None and key == self._prepared.key:
            self.last_stats = {**self._prepared.stats, 'cache_hit': True, 'message_cache_hit': True,
                               'prepare_seconds_this_call': 0., 'blas_products': 0,
                               'log_semiring_products': 0, 'axis_contractions': 0, 'maximum_kernel_entries': 0}
            return self._prepared
        # Public callers retain scalar Z or original tokens, not this history,
        # before asking for another query. Thus caches never coexist here.
        self._prepared = None
        started = perf_counter()
        self.budget.check_workspace_bytes(self._base_bytes() + self._metadata_allowance(),
                                         context='Multi-product layout and token choices')
        choices, _ = self._choices(evidence)
        known_bits, contradiction = self._fixed_template(choices)
        if contradiction:
            self.storage_bytes = self._base_bytes() + self._metadata_allowance()
            self.planning_stats.update(dispatch='zero_mass', product_state_materialized=False,
                state_entries=0, kernel_entry_limit=0, workspace_bytes_estimate=self.storage_bytes,
                peak_workspace_bytes=self.storage_bytes)
            return self._save_trace(key, choices, (), (), -math.inf, (), np.empty(0), -math.inf,
                                    self._stats(), started, ())
        self._plan_workspace(known_bits)
        choices, matrices, tails, constant = self._compile_query(evidence, choices)
        stats = self._stats()
        if known_bits is not None:
            history, final = [], []
            normalizer = constant
            for r in range(self.R):
                alpha = np.full((self.L + 1, self.D), -np.inf)
                alpha[0, self._left[r]] = 0.
                for step, bit in enumerate(known_bits):
                    alpha[step + 1] = self._log_matmul(alpha[step:step + 1], matrices[step][r][bit], stats)[0]
                history.append(alpha)
                final.append(alpha[-1] + tails[r])
                normalizer += float(logsumexp(final[-1]))
            return self._save_trace(key, choices, matrices, tails, constant, tuple(history),
                                    tuple(final), normalizer, stats, started, known_bits)
        alpha = np.full((self.count_size, *self.joint_shape), -np.inf)
        alpha[(0, *self._left)] = 0.
        history = [alpha]
        if math.isfinite(constant):
            for step, blocks in enumerate(matrices):
                following = np.full_like(alpha, -np.inf)
                for count in range(self.count_size):
                    if not np.isfinite(alpha[count]).any():
                        continue
                    for bit in (0, 1):
                        nc = self._next_count(count, bit)
                        if nc >= self.count_size or (self.K is not None and nc + self.L - step - 1 < self.K):
                            continue
                        part = alpha[count]
                        for axis in reversed(range(self.R)):
                            part = self._contract_axis(part, blocks[axis][bit], axis, stats)
                        following[nc] = np.logaddexp(following[nc], part)
                alpha = following
                history.append(alpha)
        else:
            alpha[:] = -math.inf
        final = alpha[-1] + self._tail_tensor(tails)
        log_z = float(logsumexp(final)) + constant
        return self._save_trace(key, choices, matrices, tails, constant, tuple(history), final,
                                log_z, stats, started, None)

    def _save_trace(self, key, choices, matrices, tails, constant, history, final, log_z, stats, started, known_bits):
        final_arrays = final if isinstance(final, tuple) else (final,)
        arrays = (*history, *final_arrays, *tails, *(block for step in matrices for block in step))
        for array in arrays:
            array.flags.writeable = False
        stats.update(saved_message_bytes=sum(array.nbytes for array in history),
                     transfer_bytes=sum(block.nbytes for step in matrices for block in step),
                     retained_array_bytes=sum(array.nbytes for array in arrays),
                     known_onset_template=known_bits,
                     prepare_seconds=perf_counter() - started)
        stats['prepare_seconds_this_call'] = stats['prepare_seconds']
        self._prepared = _ProductTrace(key, choices, matrices, tails, constant, tuple(history), final, log_z,
                                       MappingProxyType(dict(stats)), known_bits)
        self.last_stats = stats
        return self._prepared

    def log_partition(self, evidence=None):
        return self._prepare(evidence).log_z

    @staticmethod
    def _draw_index(weights, rng):
        normalizer = float(logsumexp(weights))
        if not math.isfinite(normalizer):
            raise ZeroMass('No positive-mass product-chain continuation')
        probabilities = np.exp(weights.ravel() - normalizer)
        probabilities /= probabilities.sum()
        flat = int(rng.choice(len(probabilities), p=probabilities))
        return tuple(int(i) for i in np.unravel_index(flat, weights.shape))

    def sample_full(self, rng, evidence=None):
        if not isinstance(rng, np.random.Generator):
            raise InvalidSpecification('rng must be a numpy Generator')
        plan = self._prepare(evidence)
        if not math.isfinite(plan.log_z):
            raise ZeroMass('Multi-product conditioning event has zero mass')
        started = perf_counter()
        if plan.known_bits is not None:
            return self._sample_known(plan, rng, started)
        current = self._draw_index(plan.final, rng)
        count = self.count_size - 1
        sequence = [self.spec.observed.get(i, 0) for i in range(self.spec.length)]
        for step in range(self.L - 1, -1, -1):
            incoming = np.full((2, *self.joint_shape), -np.inf)
            for bit in (0, 1):
                before = 0 if self.K is None else count - bit
                if before < 0:
                    continue
                incoming[bit] = plan.history[step][before]
                for axis, state in enumerate(current):
                    shape = [1] * self.R
                    shape[axis] = self.D
                    incoming[bit] += plan.matrices[step][axis][bit, :, state].reshape(shape)
            chosen = self._draw_index(incoming, rng)
            bit, previous = chosen[0], chosen[1:]
            for span, state in zip(self.spans, current):
                sequence[span[step]] = self._pitches[state] + 2 if bit else (0 if state == 0 else 1)
            current = previous
            count = 0 if self.K is None else count - bit
        tokens = tuple(sequence)
        checked = verify_music(tokens, self.spec)
        if not checked.valid:
            raise VerificationError('Multi-product sample failed independent checker: ' + '; '.join(checked.violations))
        self.last_stats = {**self.last_stats, 'sample_seconds': perf_counter() - started, 'query': 'ffbs_full_sample'}
        return tokens

    def _sample_known(self, plan, rng, started):
        tokens = [self.spec.observed.get(i, 0) for i in range(self.spec.length)]
        for r, span in enumerate(self.spans):
            current = self._draw_index(plan.final[r], rng)[0]
            for step in range(self.L - 1, -1, -1):
                bit = plan.known_bits[step]
                tokens[span[step]] = self._pitches[current] + 2 if bit else (0 if current == 0 else 1)
                current = self._draw_index(plan.history[r][step] + plan.matrices[step][r][bit, :, current], rng)[0]
        tokens = tuple(tokens)
        checked = verify_music(tokens, self.spec)
        if not checked.valid:
            raise VerificationError('Known-template chain sample failed checker: ' + '; '.join(checked.violations))
        self.last_stats = {**self.last_stats, 'sample_seconds': perf_counter() - started,
                           'query': 'independent_known_template_chain_ffbs'}
        return tokens

    def log_weight(self, tokens):
        checked = verify_music(tokens, self.spec)
        if not checked.valid:
            return -math.inf
        return float(checked.soft_score + sum(self._q(i, token) for i, token in enumerate(tokens)))

    def sample_batch(self, variables, rng, evidence=None):
        if isinstance(variables, str):
            raise InvalidSpecification('Batch variables must be a sequence')
        variables = tuple(variables)
        if len(set(variables)) != len(variables) or any(v not in self.graph.domains for v in variables):
            raise InvalidSpecification('Batch variables must be unique original-token variables')
        if not isinstance(rng, np.random.Generator):
            raise InvalidSpecification('rng must be a numpy Generator')
        evidence = self._evidence(evidence)
        base = self.log_partition(evidence)
        if not math.isfinite(base):
            raise ZeroMass('Multi-product conditioning event has zero mass')
        if not variables:
            return BatchSample({}, 0., base)
        tokens = self.sample_full(rng, evidence)
        stats = dict(self.last_stats)
        assignment = {variable: tokens[self._positions[variable]] for variable in variables}
        clamped = {**evidence, **assignment}
        covered = {self._positions[name] for name in clamped} | set(self.spec.observed)
        if len(covered) == self.spec.length:
            mass = self.log_weight(tokens)
            stats['sample_probability'] = 'complete_sequence_weight'
        else:
            # sample_full has returned, releasing its local trace reference.
            mass = self.log_partition(clamped)
            stats.update(sample_probability='projected_partition', projection_stats=dict(self.last_stats))
        self.last_stats = stats
        return BatchSample(assignment, mass - base, mass)

    def _previous_beta(self, beta, matrices, stats):
        result = np.full_like(beta, -np.inf)
        for count in range(self.count_size):
            for bit in (0, 1):
                nc = self._next_count(count, bit)
                if nc >= self.count_size or not np.isfinite(beta[nc]).any():
                    continue
                part = beta[nc]
                for axis in reversed(range(self.R)):
                    part = self._contract_axis(part, matrices[axis][bit].T, axis, stats)
                result[count] = np.logaddexp(result[count], part)
        return result

    def marginal_log_probs(self, variable, evidence=None):
        if variable not in self.graph.domains:
            raise InvalidSpecification('Unknown original-token variable')
        evidence = self._evidence(evidence)
        plan = self._prepare(evidence)
        if not math.isfinite(plan.log_z):
            raise ZeroMass('Multi-product conditioning event has zero mass')
        domain = self.graph.domains[variable]
        result = np.full(len(domain), -np.inf)
        position = self._positions[variable]
        if variable in evidence or position in self.spec.observed:
            token = evidence.get(variable, self.spec.observed.get(position))
            result[domain.index(token)] = 0.
            return result
        which = next(r for r, span in enumerate(self.spans) if position in span)
        offset = self.spans[which].index(position)
        stats = dict(self.last_stats)
        if plan.known_bits is not None:
            beta = plan.tails[which]
            for step in range(self.L - 1, offset, -1):
                beta = self._log_matmul(plan.matrices[step][which][plan.known_bits[step]], beta[:, None], stats)[:, 0]
            chain_z = float(logsumexp(plan.final[which]))
            for index, token in enumerate(domain):
                if token not in plan.choices[position] or int(token >= 2) != plan.known_bits[offset]:
                    continue
                weights = []
                for old, pitch in enumerate(self._pitches):
                    edge = self._local(position, pitch, token)
                    if edge is not None:
                        weights.append(plan.history[which][offset, old] + self._q(position, token) +
                                       edge[1] + beta[self._pitch_index[edge[0]]])
                result[index] = float(logsumexp(weights)) - chain_z
            self.last_stats = {**stats, 'query': 'known_template_chain_marginal'}
            return result
        beta = np.full((self.count_size, *self.joint_shape), -np.inf)
        beta[-1] = self._tail_tensor(plan.tails)
        for step in range(self.L - 1, offset, -1):
            beta = self._previous_beta(beta, plan.matrices[step], stats)
        beliefs = np.full((2, self.D, self.D), -np.inf)
        for count in range(self.count_size):
            for bit in (0, 1):
                nc = self._next_count(count, bit)
                if nc >= self.count_size:
                    continue
                part = plan.history[offset][count]
                for axis in reversed(range(self.R)):
                    if axis != which:
                        part = self._contract_axis(part, plan.matrices[offset][axis][bit], axis, stats)
                left = np.moveaxis(part, which, 0).reshape(self.D, -1)
                right = np.moveaxis(beta[nc], which, 0).reshape(self.D, -1).T
                beliefs[bit] = np.logaddexp(beliefs[bit], self._log_matmul(left, right, stats))
        for index, token in enumerate(domain):
            bit = int(token >= 2)
            if token not in plan.choices[position] or (offset in self._flags and bit != self._flags[offset]):
                continue
            weights = []
            for old, pitch in enumerate(self._pitches):
                edge = self._local(position, pitch, token)
                if edge is not None:
                    weights.append(beliefs[bit, old, self._pitch_index[edge[0]]] + self._q(position, token) + edge[1])
            result[index] = float(logsumexp(weights)) + plan.constant - plan.log_z
        self.last_stats = {**stats, 'query': 'forward_backward_token_marginal'}
        return result
