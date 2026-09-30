"""Budgeted log-space variable elimination and exact conditional sampling.

Exactness is for the supplied immutable factor graph. It says nothing about
the terminal distribution of a neural model queried repeatedly while decoding.
"""

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import numbers

import numpy as np
from scipy.special import logsumexp

from tri.errors import BudgetExceeded, InvalidSpecification, ZeroMass
from tri.inference.factors import FactorGraph


@dataclass(frozen=True)
class Budget:
    max_factor_entries: int = 2_000_000
    max_workspace_bytes: int = 536_870_912
    max_oracle_assignments: int = 1_000_000

    def __post_init__(self):
        for name in ("max_factor_entries", "max_workspace_bytes", "max_oracle_assignments"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise InvalidSpecification(f"{name} must be a positive integer")

    def check_factor_entries(self, entries: int, *, context: str = "factor") -> None:
        if entries > self.max_factor_entries:
            raise BudgetExceeded(f"{context} needs {entries} entries; budget is {self.max_factor_entries}")

    def check_workspace_bytes(self, size: int, *, context: str = "inference workspace") -> None:
        if size > self.max_workspace_bytes:
            raise BudgetExceeded(f"{context} needs approximately {size} bytes; budget is {self.max_workspace_bytes}")


@dataclass(frozen=True)
class BatchSample:
    assignment: dict[str, int]
    log_probability: float
    log_clamped_partition: float


class _Queries:
    """Input/conditional-query plumbing; backends supply independent sums."""

    graph: FactorGraph

    def _evidence(self, evidence=None) -> dict[str, int]:
        if evidence is None:
            return {}
        if not isinstance(evidence, Mapping):
            raise InvalidSpecification("Evidence must map variable names to domain values")
        result = {}
        for name, value in evidence.items():
            if name not in self.graph.domains:
                raise InvalidSpecification(f"Unknown variable {name!r}")
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value not in self.graph.domains[name]:
                raise InvalidSpecification(f"Invalid value {value!r} for {name!r}")
            result[name] = int(value)
        return result

    def log_clamped_partition(self, assignment) -> float:
        """Keep every original factor, including q at newly clamped positions."""
        return self.log_partition(assignment)

    def log_probability(self, assignment, evidence=None) -> float:
        evidence = self._evidence(evidence)
        assignment = self._evidence(assignment)
        base = self.log_partition(evidence)
        if np.isneginf(base):
            raise ZeroMass("Conditioning event has zero mass; this is not a proof of logical infeasibility")
        if any(name in evidence and evidence[name] != value for name, value in assignment.items()):
            return -math.inf
        return self.log_partition({**evidence, **assignment}) - base

    def sample_batch(self, variables: Sequence[str], rng: np.random.Generator, evidence=None) -> BatchSample:
        if isinstance(variables, str):
            raise InvalidSpecification("Batch variables must be a sequence, not a string")
        variables = tuple(variables)
        if len(set(variables)) != len(variables) or any(v not in self.graph.domains for v in variables):
            raise InvalidSpecification("Batch variables must be distinct known names")
        if not isinstance(rng, np.random.Generator):
            raise InvalidSpecification("Sampling requires a numpy.random.Generator")
        current = self._evidence(evidence)
        base = self.log_partition(current)
        if np.isneginf(base):
            raise ZeroMass("Cannot sample a zero-mass graph")
        assignment = {}
        for name in variables:
            # Update the condition after EVERY choice: independent marginals
            # would violate relations even if each marginal were exact.
            log_probs = self.marginal_log_probs(name, current)
            probabilities = np.exp(log_probs)
            probabilities /= probabilities.sum()  # roundoff only, not q truncation
            index = int(rng.choice(len(probabilities), p=probabilities))
            assignment[name] = self.graph.domains[name][index]
            current[name] = assignment[name]
        clamped = self.log_partition(current)
        return BatchSample(assignment, clamped - base, clamped)


class ExactInference(_Queries):
    """Exact variable elimination, with at most 128 cached scalar partitions.

    Explicit ``order`` must be a full permutation of graph variable names.
    The default weighted min-fill is recomputed after applying evidence.
    Budget checks conservatively include retained factor storage plus live
    arithmetic work arrays, before the corresponding dense allocations.
    """

    def __init__(self, graph: FactorGraph, budget: Budget | None = None, order: Sequence[str] | None = None):
        if not isinstance(graph, FactorGraph):
            raise InvalidSpecification("Expected a FactorGraph")
        self.graph = graph
        self.budget = budget or Budget()
        self.order = None if order is None else tuple(order)
        if self.order is not None and (len(self.order) != len(graph.domains) or set(self.order) != set(graph.domains)):
            raise InvalidSpecification("Elimination order must be a permutation of all graph variables")
        self._indices = {v: {value: i for i, value in enumerate(domain)} for v, domain in graph.domains.items()}
        self._cache: OrderedDict[tuple, tuple[float, dict]] = OrderedDict()
        self.last_stats: dict = {}
        self._input_bytes = sum(f.values.nbytes for f in graph.factors)
        for factor in graph.factors:
            self.budget.check_factor_entries(factor.values.size, context=f"Input factor {factor.scope}")
        self.budget.check_workspace_bytes(self._input_bytes, context="Input factor storage")

    def _key(self, evidence):
        return tuple((v, evidence[v]) for v in self.graph.domains if v in evidence)

    def _choose_order(self, factors, remaining, keep):
        if self.order is not None:
            return [v for v in self.order if v in remaining and v != keep]
        adjacency = {v: set() for v in remaining}
        for scope, _ in factors:
            for v in scope:
                adjacency[v].update(set(scope) - {v})
        available = set(remaining) - ({keep} if keep is not None else set())
        order = []
        while available:
            def cost(v):
                neighbors = sorted(adjacency[v])
                fill = sum(len(self.graph.domains[a]) * len(self.graph.domains[b])
                           for i, a in enumerate(neighbors) for b in neighbors[i + 1:] if b not in adjacency[a])
                entries = math.prod(len(self.graph.domains[n]) for n in neighbors) * len(self.graph.domains[v])
                return fill, entries, v
            chosen = min(available, key=cost)
            neighbors = adjacency.pop(chosen)
            for v in neighbors:
                adjacency[v].discard(chosen)
                adjacency[v].update(neighbors - {v})
            available.remove(chosen)
            order.append(chosen)
        return order

    def _combine(self, bucket, retained, stats, reduce_variable=None):
        scope = tuple(dict.fromkeys(v for names, _ in bucket for v in names))
        shape = tuple(len(self.graph.domains[v]) for v in scope)
        entries = math.prod(shape)
        self.budget.check_factor_entries(entries, context=f"Intermediate factor {scope}")
        # scipy.logsumexp may hold shifted inputs and exponentials; reserve
        # three full intermediate arrays plus output and retained inputs.
        reduced_entries = entries if reduce_variable is None else entries // len(self.graph.domains[reduce_variable])
        workspace = self._input_bytes + sum(values.nbytes for _, values in retained + bucket) + 8 * (3 * entries + reduced_entries)
        self.budget.check_workspace_bytes(workspace)
        stats["max_factor_entries"] = max(stats["max_factor_entries"], entries)
        stats["peak_workspace_bytes"] = max(stats["peak_workspace_bytes"], workspace)
        result = np.zeros(shape, dtype=np.float64)
        for names, values in bucket:
            positions = [scope.index(v) for v in names]
            permutation = np.argsort(positions)
            if names:
                values = values.transpose(tuple(int(i) for i in permutation))
            expanded = [1] * len(scope)
            for i in permutation:
                expanded[positions[i]] = len(self.graph.domains[names[i]])
            try:
                with np.errstate(over="raise", invalid="raise"):
                    np.add(result, values.reshape(expanded), out=result)
            except FloatingPointError as exc:
                raise InvalidSpecification("Combined log weights exceed float64 range") from exc
            stats["factor_additions"] += 1
        if reduce_variable is None:
            return scope, result
        axis = scope.index(reduce_variable)
        result = logsumexp(result, axis=axis)
        return tuple(v for v in scope if v != reduce_variable), np.asarray(result)

    def _run(self, evidence, keep=None):
        factors = []
        for factor in self.graph.factors:
            # Basic indexing produces views, and never renormalizes factors.
            index = tuple(self._indices[v][evidence[v]] if v in evidence else slice(None) for v in factor.scope)
            scope = tuple(v for v in factor.scope if v not in evidence)
            factors.append((scope, np.asarray(factor.values[index])))
        remaining = set(self.graph.domains) - set(evidence)
        order = self._choose_order(factors, remaining, keep)
        stats = {"elimination_order": order, "max_factor_entries": max((f.values.size for f in self.graph.factors), default=1),
                 "peak_workspace_bytes": sum(f.values.nbytes for f in self.graph.factors), "factor_additions": 0, "cache_hit": False}
        for v in order:
            bucket = [f for f in factors if v in f[0]]
            factors = [f for f in factors if v not in f[0]]
            if bucket:
                factors.append(self._combine(bucket, factors, stats, v))
            else:
                factors.append(((), np.asarray(math.log(len(self.graph.domains[v])))))
        # At this point only one retained variable (if any) can remain.
        if keep is not None and keep not in evidence and not any(keep in names for names, _ in factors):
            entries = len(self.graph.domains[keep])
            self.budget.check_factor_entries(entries, context="Disconnected marginal")
            self.budget.check_workspace_bytes(self._input_bytes + sum(a.nbytes for _, a in factors) + entries * 8)
            factors.append(((keep,), np.zeros(entries, dtype=np.float64)))
        scope, values = self._combine(factors, [], stats)
        self.last_stats = stats
        return scope, values

    def log_partition(self, evidence=None) -> float:
        evidence = self._evidence(evidence)
        key = self._key(evidence)
        if key in self._cache:
            value, stats = self._cache[key]
            self._cache.move_to_end(key)
            self.last_stats = {**stats, "cache_hit": True}
            return value
        _, value = self._run(evidence)
        result = float(value)
        self._cache[key] = (result, dict(self.last_stats))
        if len(self._cache) > 128:
            self._cache.popitem(last=False)
        return result

    def marginal_log_probs(self, variable, evidence=None) -> np.ndarray:
        if variable not in self.graph.domains:
            raise InvalidSpecification(f"Unknown variable {variable!r}")
        evidence = self._evidence(evidence)
        if variable in evidence:
            if np.isneginf(self.log_partition(evidence)):
                raise ZeroMass("Conditioning event has zero mass")
            self.budget.check_factor_entries(len(self.graph.domains[variable]), context="Observed marginal")
            self.budget.check_workspace_bytes(self._input_bytes + 8 * len(self.graph.domains[variable]), context="Observed marginal")
            values = np.full(len(self.graph.domains[variable]), -np.inf)
            values[self._indices[variable][evidence[variable]]] = 0.0
            return values
        _, values = self._run(evidence, keep=variable)
        normalization = float(logsumexp(values))
        if np.isneginf(normalization):
            raise ZeroMass("Conditioning event has zero mass")
        return values - normalization
