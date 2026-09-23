"""Input validation, resource budgets, and conditional-query contracts."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import numbers
import numpy as np
from shared_onset.errors import BudgetExceeded, InvalidSpecification, ZeroMass

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

    graph: object

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
