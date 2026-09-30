"""Equal-cache control for the standard factorial-HMM implementation.

One immutable unconditional forward trace is retained when its conservative
coexistence bound fits the existing budget. Conditional queries use the original
exact routine; no unconditional trace is built for a first conditional query.
The numerical contractions and backward/marginal equations are unchanged from
ProductChainMusicInference. This isolates generic HMM caching from any claimed
advantage of the paired implementation.
"""

from types import MappingProxyType

from tri.errors import InvalidSpecification
from tri.inference.exact import BatchSample
from tri.inference.product_chain import ProductChainMusicInference


class CachedProductChainMusicInference(ProductChainMusicInference):
    """Standard product chain with one budgeted, read-only unconditional trace."""

    def __init__(self, spec, log_probs, budget=None):
        super().__init__(spec, log_probs, budget)
        self.backend_name = self.requested_backend = 'product_reuse'
        self.planning_stats = {**self.planning_stats, 'selected_backend': self.backend_name}
        self._base_trace = None
        self._last_forward_normalizer = None
        self._retained_array_bytes = 0
        # In addition to the uncached algorithm's complete forward/marginal
        # workspace, allow one base history, transfers, final, boundary arrays,
        # immutable choice tuples and conservative Python container overhead.
        array_bound = 8 * ((self.width + 1) * self.count_size * self.size**2 +
                           4 * self.width * self.size**2 + self.size**2 + 2 * self.size)
        metadata_bound = (256 * (self.width + 1) + 1024 * self.width +
                          spec.length * (128 + 8 * (self.size + 1)))
        self._retained_bound = array_bound + metadata_bound
        self._cache_bound = self.storage_bytes + self._retained_bound
        self._cache_enabled = self._cache_bound <= self.budget.max_workspace_bytes
        self._workspace_bound = self._cache_bound if self._cache_enabled else self.storage_bytes

    def _stats(self):
        return {**super()._stats(), 'message_cache_enabled': self._cache_enabled,
                'message_cache_hit': False, 'retained_base_array_bytes': self._retained_array_bytes,
                'peak_workspace_bytes': self._workspace_bound}

    def _forward_chain(self, evidence, save):
        if not evidence and self._cache_enabled and self._base_trace is not None:
            z, history, final, query, saved_stats = self._base_trace
            self._last_forward_normalizer = z
            stats = {**saved_stats, 'cache_hit': True, 'message_cache_hit': True,
                     'blas_products': 0, 'log_semiring_products': 0,
                     'retained_base_array_bytes': self._retained_array_bytes,
                     'peak_workspace_bytes': self._workspace_bound}
            return z, history if save else None, final, query, stats
        retain = not evidence and self._cache_enabled
        z, history, final, query, stats = super()._forward_chain(evidence, save=save or retain)
        self._last_forward_normalizer = z
        if retain:
            choices, beginnings, endings, constant, matrices = query
            history = tuple(history)
            matrices = tuple(tuple(pair) for pair in matrices)
            query = choices, beginnings, endings, constant, matrices
            arrays = (*history, final, *endings, *(matrix for pair in matrices for matrix in pair))
            for array in arrays:
                array.flags.writeable = False
            self._retained_array_bytes = sum(array.nbytes for array in arrays)
            stats = {**stats, 'retained_base_array_bytes': self._retained_array_bytes}
            self._base_trace = z, history, final, query, MappingProxyType(dict(stats))
        stats = {**stats, 'message_cache_hit': False,
                 'retained_base_array_bytes': self._retained_array_bytes,
                 'peak_workspace_bytes': self._workspace_bound}
        return z, history if save else None, final, query, stats

    def sample_batch(self, variables, rng, evidence=None):
        if isinstance(variables, str):
            raise InvalidSpecification('Batch variables must be a sequence')
        variables = tuple(variables)
        if len(set(variables)) != len(variables) or any(v not in self.graph.domains for v in variables):
            raise InvalidSpecification('Batch variables must be distinct original-token names')
        evidence = self._evidence(evidence)
        covered = set(variables) | set(evidence) | {f'y{i}' for i in self.spec.observed}
        if not variables or len(covered) == self.spec.length:
            return super().sample_batch(variables, rng, evidence)
        # The original sampler already draws a full path before projection.
        # Return from its full-path call FIRST, releasing its conditional trace
        # before computing a projected partition. Thus at most the retained
        # base plus ONE transient query trace coexist under the cache bound.
        complete = super().sample_batch(tuple(self.graph.domains), rng, evidence)
        base = self._last_forward_normalizer
        stats = dict(self.last_stats)
        assignment = {name: complete.assignment[name] for name in variables}
        clamped = self.log_partition({**evidence, **assignment})
        self.last_stats = {**stats, 'sample_probability': 'projected_partition',
                           'projection_stats': dict(self.last_stats),
                           'peak_workspace_bytes': self._workspace_bound}
        return BatchSample(assignment, clamped - base, clamped)
