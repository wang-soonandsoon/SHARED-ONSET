"""Exact shared-feature/count DP for independent categorical variables ONLY.

For replicas r and aligned slots j, all categorical variables Y[r,j] share a
boolean feature z[j], and exactly K features are true. Optional per-category
allowed masks and original observations are supported. The unnormalized target
is the product of original unknown-position probabilities and these indicators.

There are NO sounding-state chains, HOLD semantics, neighboring pitch rules,
or other cross-slot factors. This is not a drop-in MusicSpec backend. A HOLD
category here is just an independent category unless the caller forbids it.
Using this backend on a general music request would change the target.

After O(R L V) category aggregation, partition and template sampling use an
O(L K) DP (O(L (K+1)) including K=0). Conditional category draws are exact.
The algorithm is a strong structural baseline, not a novel inference claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

import numpy as np
from scipy.special import logsumexp

from tri.errors import InvalidSpecification, ZeroMass
from tri.inference.exact import Budget

Position = tuple[int, int]


@dataclass(frozen=True)
class RhythmBatchSample:
    assignment: dict[Position, int]
    log_probability: float
    log_clamped_partition: float


def _snapshot(array: np.ndarray) -> np.ndarray:
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class SharedOnsetCountInference:
    """Finite category indices 0..V-1, shared boolean onset and exact count.

    Unknown rows of log_probs must be normalized over the complete V-category
    vocabulary. allowed applies after normalization, without restoring removed
    probability mass. Original observed positions contribute a delta of weight
    one; their model rows are ignored. Temporary query clamps keep old weights.

    Observed/allowed contradictions and zero supported model mass yield logZ
    -inf. Sampling/normalized probability queries then raise ZeroMass; this
    class does not assert that every zero-mass event is logically infeasible.
    """

    def __init__(
        self,
        log_probs: np.ndarray,
        onset_flags: Sequence[bool],
        count: int,
        observed: Mapping[Position, int] | None = None,
        allowed: np.ndarray | None = None,
        budget: Budget | None = None,
    ) -> None:
        self.budget = budget or Budget()
        if np.iscomplexobj(log_probs):
            raise InvalidSpecification("log_probs must contain real log probabilities")
        try:
            raw = np.asarray(log_probs)
        except (TypeError, ValueError) as exc:
            raise InvalidSpecification("log_probs must be a numeric [R,L,V] array") from exc
        if raw.ndim != 3 or any(size < 1 for size in raw.shape):
            raise InvalidSpecification("log_probs must have nonempty shape [replicas, length, vocabulary]")
        self.replicas, self.length, self.vocabulary = raw.shape
        self.shape = raw.shape
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or not 0 <= count <= self.length:
            raise InvalidSpecification("count must be an integer between zero and length")
        self.count = int(count)
        try:
            flags = np.asarray(onset_flags)
        except (TypeError, ValueError) as exc:
            raise InvalidSpecification("onset_flags must be a boolean vector") from exc
        if flags.shape != (self.vocabulary,) or flags.dtype.kind != "b":
            raise InvalidSpecification("onset_flags must be a boolean vector of length V")
        table_entries = (self.length + 1) * (self.count + 1)
        self.budget.check_factor_entries(self.vocabulary, context="categorical row")
        self.budget.check_factor_entries(table_entries, context="shared-onset DP table")
        # Include immutable snapshots and temporary query/aggregation storage.
        self._workspace_bytes = int(
            3 * raw.size * 8 + 2 * self.replicas * self.length * 2 * 8
            + 4 * self.length * 2 * 8 + 3 * table_entries * 8 + self.vocabulary
        )
        self.budget.check_workspace_bytes(self._workspace_bytes, context="shared-onset DP workspace")
        self.onset_flags = _snapshot(flags)
        try:
            raw = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InvalidSpecification("log_probs must be numeric") from exc
        observations = self._assignment({} if observed is None else observed)
        self.observed = MappingProxyType(observations)
        mask = None
        if allowed is not None:
            try:
                mask = np.asarray(allowed)
            except (TypeError, ValueError) as exc:
                raise InvalidSpecification("allowed must be a boolean array with shape [R,L,V]") from exc
            if mask.shape != raw.shape or mask.dtype.kind != "b":
                raise InvalidSpecification("allowed must be a boolean array with shape [R,L,V]")
        for r in range(self.replicas):
            for j in range(self.length):
                if (r, j) in observations:
                    continue
                row = raw[r, j]
                if np.isnan(row).any() or np.isposinf(row).any():
                    raise InvalidSpecification(f"unknown row {(r, j)} accepts only finite or -inf log probabilities")
                normalizer = float(logsumexp(row))
                if not np.isfinite(normalizer) or abs(normalizer) > 2e-6:
                    raise InvalidSpecification(f"unknown full row {(r, j)} must sum to one")
        weights = raw.copy()
        if mask is not None:
            weights[~mask] = -np.inf
        for (r, j), token in observations.items():
            weights[r, j, :] = -np.inf
            weights[r, j, token] = 0.0 if mask is None or mask[r, j, token] else -np.inf
        self._weights = _snapshot(weights)
        sums = np.empty((self.replicas, self.length, 2), dtype=np.float64)
        for feature in (0, 1):
            sums[:, :, feature] = logsumexp(self._weights[:, :, self.onset_flags == bool(feature)], axis=-1)
        self._sums = _snapshot(sums)
        self._aggregate = _snapshot(self._sums.sum(axis=0))
        self._backward_table = _snapshot(self._backward(self._aggregate))
        self._log_z = float(self._backward_table[0, self.count])
        self.last_stats = {
            "backend": "shared_onset_count_dp",
            "replicas": self.replicas,
            "length": self.length,
            "vocabulary": self.vocabulary,
            "count": self.count,
            "dp_cells": table_entries,
            "workspace_bytes_bound": self._workspace_bytes,
            "supports_local_pitch_transitions": False,
        }

    def _position(self, position: object) -> Position:
        if isinstance(position, (str, bytes)):
            raise InvalidSpecification("positions must be (replica, slot) pairs")
        try:
            coordinates = tuple(position)
        except TypeError as exc:
            raise InvalidSpecification("positions must be (replica, slot) pairs") from exc
        if len(coordinates) != 2:
            raise InvalidSpecification("positions must contain exactly two indices")
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in coordinates):
            raise InvalidSpecification("position indices must be integers")
        r, j = map(int, coordinates)
        if not 0 <= r < self.replicas or not 0 <= j < self.length:
            raise InvalidSpecification(f"position {(r, j)} is outside shape {self.shape}")
        return r, j

    def _assignment(self, assignment: Mapping[Position, int]) -> dict[Position, int]:
        if not isinstance(assignment, Mapping):
            raise InvalidSpecification("assignment must map (replica, slot) to a category index")
        result = {}
        for position, token in assignment.items():
            position = self._position(position)
            if isinstance(token, (bool, np.bool_)) or not isinstance(token, Integral) or not 0 <= token < self.vocabulary:
                raise InvalidSpecification("category indices must be integers in 0..V-1")
            result[position] = int(token)
        return result

    def _backward(self, aggregate: np.ndarray) -> np.ndarray:
        """B[j,k] is remaining mass for exactly k onsets in slots j..L-1."""
        table = np.full((self.length + 1, self.count + 1), -np.inf)
        table[self.length, 0] = 0.0
        for j in range(self.length - 1, -1, -1):
            high = min(self.count, self.length - j)
            table[j, : high + 1] = aggregate[j, 0] + table[j + 1, : high + 1]
            if high:
                table[j, 1 : high + 1] = np.logaddexp(
                    table[j, 1 : high + 1], aggregate[j, 1] + table[j + 1, :high]
                )
        return table

    def _clamped_aggregate(self, assignment: dict[Position, int]) -> np.ndarray:
        aggregate = self._aggregate.copy()
        by_slot: dict[int, list[tuple[int, int]]] = {}
        for (r, j), token in assignment.items():
            by_slot.setdefault(j, []).append((r, token))
        for j, items in by_slot.items():
            sums = self._sums[:, j, :].copy()
            for r, token in items:
                sums[r, :] = -np.inf
                sums[r, int(self.onset_flags[token])] = self._weights[r, j, token]
            aggregate[j, :] = sums.sum(axis=0)
        return aggregate

    def log_partition(self) -> float:
        return self._log_z

    def log_clamped_partition(self, assignment: Mapping[Position, int]) -> float:
        """Retain original q factors; never renormalize a query's category row."""
        assignment = self._assignment(assignment)
        if not assignment:
            return self._log_z
        return float(self._backward(self._clamped_aggregate(assignment))[0, self.count])

    def log_probability(self, assignment: Mapping[Position, int]) -> float:
        assignment = self._assignment(assignment)
        if np.isneginf(self._log_z):
            raise ZeroMass("Shared-onset/count target has zero mass; cannot normalize it")
        return self.log_clamped_partition(assignment) - self._log_z

    def sample_batch(self, positions: Sequence[Position], rng: np.random.Generator) -> RhythmBatchSample:
        """Sample a true joint marginal and return its probability.

        A complete latent rhythm is sampled by backward DP, then only requested
        categories are drawn given that rhythm. The returned batch probability
        sums all unreported categories AND rhythms via a clamped partition; it
        is not the probability of the sampled latent rhythm or a full candidate.
        """
        if isinstance(positions, (str, bytes)):
            raise InvalidSpecification("positions must be a sequence of (replica, slot) pairs")
        try:
            positions = tuple(self._position(position) for position in positions)
        except TypeError as exc:
            raise InvalidSpecification("positions must be a sequence of (replica, slot) pairs") from exc
        if len(set(positions)) != len(positions):
            raise InvalidSpecification("sample positions must not repeat")
        if not isinstance(rng, np.random.Generator):
            raise InvalidSpecification("sampling requires a numpy.random.Generator")
        if np.isneginf(self._log_z):
            raise ZeroMass("Cannot sample a zero-mass shared-onset/count target")
        if not positions:
            return RhythmBatchSample({}, 0.0, self._log_z)

        rhythm = np.empty(self.length, dtype=np.bool_)
        remaining = self.count
        for j in range(self.length):
            scores = np.full(2, -np.inf)
            for feature in (0, 1):
                following = remaining - feature
                if 0 <= following <= self.count:
                    scores[feature] = self._aggregate[j, feature] + self._backward_table[j + 1, following]
            probabilities = np.exp(scores - logsumexp(scores))
            probabilities /= probabilities.sum()  # roundoff only
            chosen = int(rng.choice(2, p=probabilities))
            rhythm[j] = bool(chosen)
            remaining -= chosen
        if remaining != 0:
            raise RuntimeError("Internal shared-onset DP count invariant failed")

        assignment = {}
        for r, j in positions:
            scores = np.where(self.onset_flags == rhythm[j], self._weights[r, j], -np.inf)
            probabilities = np.exp(scores - logsumexp(scores))
            probabilities /= probabilities.sum()
            assignment[r, j] = int(rng.choice(self.vocabulary, p=probabilities))
        clamped = self.log_clamped_partition(assignment)
        return RhythmBatchSample(assignment, clamped - self._log_z, clamped)
