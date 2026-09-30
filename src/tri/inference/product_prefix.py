"""Standard factorial-HMM control with immutable conditional-prefix reuse.

This extends product_reuse without changing its guarded scaled/log-semiring
matrix kernel, count recurrence, marginal equations, or original-token target.
Every conditional trace branches from the same immutable unconditional trace:
unchanged local transfers and all forward layers before the earliest changed
transfer are shared, and the remaining suffix is recomputed. The preceding
conditional query is never a numerical starting point.

A first nonempty query uses the ordinary conditional routine and does not build
an unconditional cache. The inherited conservative cache-plus-query workspace
bound also covers a fully changed suffix; insufficient space disables reuse.
New model probabilities require a new engine, as in the original product API.
"""

import math

import numpy as np
from scipy.special import logsumexp

from tri.inference.product_cached import CachedProductChainMusicInference


class PrefixCachedProductChainMusicInference(CachedProductChainMusicInference):
    """Budgeted standard product chain with transfer and forward-prefix reuse."""

    def __init__(self, spec, log_probs, budget=None):
        super().__init__(spec, log_probs, budget)
        self.backend_name = self.requested_backend = 'product_prefix'
        self.planning_stats = {**self.planning_stats, 'selected_backend': self.backend_name,
                               'conditional_reuse': 'immutable_unconditional_prefix'}

    def _stats(self):
        return {**super()._stats(), 'scalar_cache_hit': False,
                'prefix_layers_reused': 0, 'transfers_rebuilt': 0,
                'transfers_reused': 0, 'suffix_layers_recomputed': 0}

    def _one_transfer(self, position, choices):
        # Same loop order, token weights and local semantics as the original
        # product compiler; only unchanged blocks are omitted from rebuilding.
        block = np.full((2, self.size, self.size), -np.inf)
        for old, pitch in enumerate(self.pitches):
            for token in choices:
                bit = int(token >= 2)
                if position in self._flags and self._flags[position] != bit:
                    continue
                edge = self._local(position, pitch, token)
                if edge is not None:
                    new = self._pitch_indices[edge[0]]
                    block[bit, old, new] = np.logaddexp(
                        block[bit, old, new], self._q(position, token) + edge[1])
        return block

    def _forward_chain(self, evidence, save):
        base = self._base_trace
        if not evidence or base is None or not self._cache_enabled:
            z, history, final, query, stats = super()._forward_chain(evidence, save)
            hit = stats['message_cache_hit']
            stats = {**stats, 'scalar_cache_hit': False,
                     'prefix_layers_reused': self.width if hit else 0,
                     'transfers_rebuilt': 0 if hit else 2 * self.width,
                     'transfers_reused': 2 * self.width if hit else 0,
                     'suffix_layers_recomputed': 0 if hit or not math.isfinite(query[3]) else self.width}
            return z, history, final, query, stats

        choices, _ = self._choices(evidence)
        base_z, base_history, base_final, base_query, _ = base
        base_choices, beginnings, endings, constant, base_matrices = base_query
        stats = self._stats()
        if not math.isfinite(base_z):
            # Restricting the support of a zero-mass target cannot create mass.
            # This also handles invalid fixed boundaries, whose saved history
            # can end before L without incorrectly indexing a missing layer.
            query = choices, beginnings, endings, constant, base_matrices
            self._last_forward_normalizer = -math.inf
            stats.update(zero_mass_base_shortcut=True)
            return -math.inf, base_history if save else None, base_final, query, stats

        first_change = self.width
        rebuilt_bytes = 0
        matrices = []
        for step, pair in enumerate(zip(*self.spans)):
            local_pair = []
            for side, position in enumerate(pair):
                if choices[position] == base_choices[position]:
                    local_pair.append(base_matrices[step][side])
                    stats['transfers_reused'] += 1
                else:
                    block = self._one_transfer(position, choices[position])
                    local_pair.append(block)
                    rebuilt_bytes += block.nbytes
                    stats['transfers_rebuilt'] += 1
                    first_change = min(first_change, step)
            matrices.append(tuple(local_pair))
        matrices = tuple(matrices)
        query = choices, beginnings, endings, constant, matrices
        stats.update(prefix_layers_reused=first_change, new_transfer_bytes=rebuilt_bytes,
                     transfer_bytes=sum(m.nbytes for pair in matrices for m in pair))
        if first_change == self.width:
            self._last_forward_normalizer = base_z
            stats.update(cache_hit=True, message_cache_hit=True,
                         saved_message_bytes=sum(a.nbytes for a in base_history) if save else 0,
                         new_saved_message_bytes=0, live_message_bytes=0)
            return base_z, base_history if save else None, base_final, query, stats

        # Evidence-induced onset propagation is already reflected in choices.
        # Sharing is safe exactly before the earliest changed local transfer.
        alpha = base_history[first_change]
        history = list(base_history[:first_change + 1]) if save else None
        for step in range(first_change, self.width):
            first, second = matrices[step]
            following = np.full_like(alpha, -np.inf)
            for count in range(self.count_size):
                if not np.isfinite(alpha[count]).any():
                    continue
                for bit in (0, 1):
                    nc = self._next_count(count, bit)
                    if nc >= self.count_size or (self.target is not None and nc + self.width - step - 1 < self.target):
                        continue
                    part = self._log_matrix_product(alpha[count], second[bit], stats)
                    part = self._log_matrix_product(first[bit].T, part, stats)
                    following[nc] = np.logaddexp(following[nc], part)
            alpha = following
            if save:
                history.append(alpha)
        final = alpha[-1] + endings[0][:, None] + endings[1][None, :]
        z = float(logsumexp(final)) + constant
        self._last_forward_normalizer = z
        stats.update(suffix_layers_recomputed=self.width - first_change,
                     saved_message_bytes=sum(a.nbytes for a in history) if save else 0,
                     new_saved_message_bytes=sum(a.nbytes for a in history[first_change + 1:]) if save else 0,
                     live_message_bytes=2 * alpha.nbytes)
        return z, history, final, query, stats

    def log_partition(self, evidence=None):
        evidence = self._evidence(evidence)
        key = tuple(sorted(evidence.items()))
        if key in self._cache:
            z, saved_stats = self._cache[key]
            self._cache.move_to_end(key)
            # A scalar hit performs no transfer construction or contractions;
            # do not repeat the work counters from the original cached query.
            self.last_stats = {**saved_stats, 'cache_hit': True, 'scalar_cache_hit': True,
                               'message_cache_hit': False, 'prefix_layers_reused': 0,
                               'transfers_rebuilt': 0, 'transfers_reused': 0,
                               'suffix_layers_recomputed': 0, 'new_transfer_bytes': 0,
                               'new_saved_message_bytes': 0, 'blas_products': 0,
                               'log_semiring_products': 0,
                               'retained_base_array_bytes': self._retained_array_bytes}
            self._last_forward_normalizer = z
            return z
        z, _, _, _, stats = self._forward_chain(evidence, save=False)
        self._cache[key] = z, dict(stats)
        if len(self._cache) > 128:
            self._cache.popitem(last=False)
        self.last_stats = stats
        return z
