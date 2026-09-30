"""Immutable, finite-domain factor graphs in log space.

Domain labels are values, not necessarily zero-based tensor indices.
"""

from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
import numbers

import numpy as np

from tri.errors import InvalidSpecification


@dataclass(frozen=True, eq=False)
class LogFactor:
    scope: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self):
        scope = tuple(self.scope)
        if any(not isinstance(v, str) or not v for v in scope):
            raise InvalidSpecification("Factor variable names must be nonempty strings")
        if len(scope) != len(set(scope)):
            raise InvalidSpecification("Factor scope contains duplicate variables")
        try:
            values = np.asarray(self.values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InvalidSpecification("Factor values must be numeric") from exc
        if values.ndim != len(scope):
            raise InvalidSpecification("Factor rank must equal scope length")
        if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
            raise InvalidSpecification("Log factors allow finite values and -inf only")
        # A bytes owner also prevents callers from re-enabling WRITEABLE.
        snapshot = np.frombuffer(values.tobytes(order="C"), dtype=np.float64).reshape(values.shape)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "values", snapshot)


@dataclass(frozen=True, eq=False)
class FactorGraph:
    domains: Mapping[str, tuple[int, ...]]
    factors: tuple[LogFactor, ...]

    def __post_init__(self):
        domains = {}
        for name, labels in self.domains.items():
            if not isinstance(name, str) or not name:
                raise InvalidSpecification("Variable names must be nonempty strings")
            labels = tuple(labels)
            if not labels or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Integral) for x in labels):
                raise InvalidSpecification(f"Domain {name!r} must contain integer values")
            labels = tuple(int(x) for x in labels)
            if len(labels) != len(set(labels)):
                raise InvalidSpecification(f"Domain {name!r} contains duplicate values")
            domains[name] = labels
        factors = tuple(self.factors)
        for factor in factors:
            if not isinstance(factor, LogFactor):
                raise InvalidSpecification("Graph factors must be LogFactor instances")
            if any(name not in domains for name in factor.scope):
                raise InvalidSpecification("Factor refers to an unknown variable")
            expected = tuple(len(domains[name]) for name in factor.scope)
            if factor.values.shape != expected:
                raise InvalidSpecification(f"Factor {factor.scope} has shape {factor.values.shape}, expected {expected}")
        object.__setattr__(self, "domains", MappingProxyType(domains))
        object.__setattr__(self, "factors", factors)
