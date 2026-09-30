"""Independent, streaming exhaustive sums for tiny graphs only.

No variable-elimination arithmetic, ordering, or factor alignment is reused.
"""

from itertools import product
import math

import numpy as np

from tri.errors import BudgetExceeded, InvalidSpecification, ZeroMass
from tri.inference.exact import Budget, _Queries
from tri.inference.factors import FactorGraph


class EnumeratedInference(_Queries):
    def __init__(self, graph: FactorGraph, budget: Budget | None = None):
        if not isinstance(graph, FactorGraph):
            raise InvalidSpecification("Expected a FactorGraph")
        self.graph = graph
        self.budget = budget or Budget()
        self.last_stats = {}
        for factor in graph.factors:
            self.budget.check_factor_entries(factor.values.size, context="Oracle input factor")

    def _assignments(self, evidence):
        names = tuple(self.graph.domains)
        choices = [((self.graph.domains[v].index(evidence[v]),) if v in evidence else range(len(self.graph.domains[v]))) for v in names]
        count = math.prod(len(options) for options in choices)
        if count > self.budget.max_oracle_assignments:
            raise BudgetExceeded(f"Exhaustive oracle needs {count} assignments; budget is {self.budget.max_oracle_assignments}")
        self.budget.check_workspace_bytes(sum(f.values.nbytes for f in self.graph.factors) + 16 * len(names), context="Streaming oracle storage")
        self.last_stats = {"assignments": count, "backend": "independent_enumeration"}
        for indices in product(*choices):
            index = dict(zip(names, indices))
            log_weight = 0.0
            for factor in self.graph.factors:
                log_weight += float(factor.values[tuple(index[v] for v in factor.scope)])
            yield index, log_weight

    def log_partition(self, evidence=None) -> float:
        evidence = self._evidence(evidence)
        total = -math.inf
        for _, weight in self._assignments(evidence):
            total = float(np.logaddexp(total, weight))
        return total

    def marginal_log_probs(self, variable, evidence=None) -> np.ndarray:
        if variable not in self.graph.domains:
            raise InvalidSpecification(f"Unknown variable {variable!r}")
        evidence = self._evidence(evidence)
        size = len(self.graph.domains[variable])
        self.budget.check_factor_entries(size, context="Oracle marginal")
        self.budget.check_workspace_bytes(8 * size, context="Oracle marginal")
        masses = np.full(size, -math.inf)
        for indices, weight in self._assignments(evidence):
            i = indices[variable]
            masses[i] = np.logaddexp(masses[i], weight)
        total = float(np.logaddexp.reduce(masses))
        if np.isneginf(total):
            raise ZeroMass("Conditioning event has zero mass")
        return masses - total
