"""Token semantics and query plumbing used by the onset-reset sampler."""
from dataclasses import dataclass
import math
from types import MappingProxyType
import numpy as np
from scipy.special import logsumexp
from shared_onset.music import MusicSpec, SILENCE, VOCAB_SIZE
from shared_onset.errors import InvalidSpecification, ZeroMass
from shared_onset.contracts import Budget, _Queries

@dataclass(frozen=True)
class _DomainView:
    """Original-token query domains only; intentionally not a factor graph."""
    domains: object


class _MusicQueryBase(_Queries):
    """Shared validation; the concrete sampler supplies sums and relations."""

    def __init__(self, spec: MusicSpec, log_probs, budget: Budget | None = None):
        if not isinstance(spec, MusicSpec):
            raise InvalidSpecification("Expected MusicSpec")
        self.spec, self.budget = spec, budget or Budget()
        if np.iscomplexobj(log_probs):
            raise InvalidSpecification("Model log probabilities must be real")
        raw = np.asarray(log_probs)
        if raw.shape != (spec.length, VOCAB_SIZE):
            raise InvalidSpecification(f"Expected log probabilities [{spec.length}, {VOCAB_SIZE}]")
        self._input_bytes = raw.size * 8
        self._metadata_bytes = self._query_metadata_bytes()
        self.budget.check_workspace_bytes(2 * self._input_bytes + self._metadata_bytes, context="Music query snapshot")
        try:
            q = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InvalidSpecification("Model log probabilities must be numeric") from exc
        for i in range(spec.length):
            if i in spec.observed:
                continue
            if np.isnan(q[i]).any() or np.isposinf(q[i]).any():
                raise InvalidSpecification(f"Unknown slot {i} has invalid model probabilities")
            total = float(logsumexp(q[i]))
            if not math.isfinite(total) or abs(total) > 2e-6:
                raise InvalidSpecification(f"Unknown slot {i} needs normalized full-vocabulary probabilities")
        self.log_probs = np.frombuffer(q.tobytes(), dtype=np.float64).reshape(q.shape)
        vocabulary = (0, 1) + tuple(p + 2 for p in spec.pitches)
        domains = {f"y{i}": (spec.observed[i],) if i in spec.observed else vocabulary for i in range(spec.length)}
        self.graph = _DomainView(MappingProxyType(domains))
        self._positions = {name: i for i, name in enumerate(domains)}
        self.last_stats = {}
        self._build_relations()

    @property
    def stats(self):
        return dict(self.last_stats)

    def _base_bytes(self):
        return self._input_bytes + self._metadata_bytes

    def _choices(self, evidence):
        result = []
        for i in range(self.spec.length):
            name = f"y{i}"
            domain = (evidence[name],) if name in evidence else self.graph.domains[name]
            result.append(tuple(token for token in domain if i in self.spec.observed or np.isfinite(self.log_probs[i, token])))
        allowed = []
        for group in self._groups:
            bits = {0, 1}
            for i in group:
                bits &= {int(token >= 2) for token in result[i]}
            allowed.append(tuple(sorted(bits)))
            for i in group:
                result[i] = tuple(token for token in result[i] if int(token >= 2) in bits)
        return tuple(result), tuple(allowed)

    def _local(self, position, previous, token):
        spec = self.spec
        if token == 0:
            current, score = SILENCE, 0.0
        elif token == 1:
            if previous == SILENCE:
                return None
            current, score = previous, 0.0
        else:
            current = token - 2
            jump = 0 if previous == SILENCE else abs(current - previous)
            if spec.max_adjacent_interval is not None and jump > spec.max_adjacent_interval:
                return None
            score = -spec.motion_cost * jump
        if current != SILENCE:
            if position in spec.pitch_ranges:
                low, high = spec.pitch_ranges[position]
                if not low <= current <= high:
                    return None
            if position in spec.pitch_classes and current % 12 not in spec.pitch_classes[position]:
                return None
        if position in spec.fixed_soundings:
            anchor = spec.fixed_soundings[position]
            if current != (SILENCE if anchor is None else anchor):
                return None
        if position == spec.length - 1 and spec.enforce_end:
            if current != (SILENCE if spec.end_pitch is None else spec.end_pitch):
                return None
        return current, score

    def _q(self, position, token):
        return 0.0 if position in self.spec.observed else float(self.log_probs[position, token])

    def marginal_log_probs(self, variable, evidence=None):
        if variable not in self.graph.domains:
            raise InvalidSpecification(f"Unknown original-token variable {variable!r}")
        evidence = self._evidence(evidence)
        base = self.log_partition(evidence)
        if not math.isfinite(base):
            raise ZeroMass("Conditioning event has zero mass; no logical UNSAT claim")
        domain = self.graph.domains[variable]
        self.budget.check_factor_entries(len(domain), context="Music token marginal")
        self.budget.check_workspace_bytes(self._base_bytes() + len(domain) * 16, context="Music token marginal")
        result = np.full(len(domain), -np.inf)
        for j, value in enumerate(domain):
            if variable in evidence and evidence[variable] != value:
                continue
            result[j] = self.log_partition({**evidence, variable: value}) - base
        return result
