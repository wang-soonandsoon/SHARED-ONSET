"""Exact synchronous inference for two aligned, anchored music spans.

State is (shared onset count, sounding pitch A, sounding pitch B), rather
than a remembered binary template across the chronological inter-span gap.
The two local transitions contract separately in log space: O(w K P^3)
work and O(w K P^2) saved messages. This restricted backend requires all
outside positions to be observed AND sounding-anchored. Other specifications
raise UnsupportedSpec; they are never relaxed. Initial observed q is delta;
temporary query clamps retain the supplied old q exactly.
"""
import math

import numpy as np
from scipy.special import logsumexp

from tri.domain.music import SILENCE, verify_music
from tri.errors import InvalidSpecification, UnsupportedSpec, VerificationError, ZeroMass
from tri.inference.exact import BatchSample
from tri.inference.music_backends import MusicExactInference


class PairedMusicInference(MusicExactInference):
    def __init__(self, spec, log_probs, budget=None):
        super().__init__(spec, log_probs, 'template', budget)
        pairs = sorted(set(tuple(sorted(pair)) for pair in spec.equal_onsets))
        if not pairs or any(a == b for a, b in pairs):
            raise UnsupportedSpec('paired backend needs two aligned nonempty spans')
        left, right = tuple(a for a, _ in pairs), tuple(b for _, b in pairs)
        if (left != tuple(range(left[0], left[0] + len(left))) or
                right != tuple(range(right[0], right[0] + len(right))) or
                left[-1] + 1 >= right[0]):
            raise UnsupportedSpec('paired spans must be disjoint contiguous runs with an observed separator')
        self.spans = left, right
        self.width = len(left)
        inside = set(left + right)
        outside = set(range(spec.length)) - inside
        if not outside <= set(spec.observed) or not outside <= set(spec.fixed_soundings):
            raise UnsupportedSpec('paired backend requires observed sounding anchors outside both spans')
        self.target = None
        self.forced_bits = {}
        self.contradictory = False
        for rule in spec.onset_counts:
            if not rule.positions:
                continue
            if rule.positions in self.spans:
                if self.target is not None and self.target != rule.count:
                    self.contradictory = True
                self.target = rule.count
            elif len(rule.positions) == 1 and rule.positions[0] in inside:
                p = rule.positions[0]
                if p in self.forced_bits and self.forced_bits[p] != rule.count:
                    self.contradictory = True
                self.forced_bits[p] = rule.count
            else:
                raise UnsupportedSpec('paired counters must be whole spans or individual span slots')
        self.count_limit = self.width if self.target is None else self.target
        # A carried-in pitch can be held even if it cannot be newly emitted.
        carried = {p for p in (spec.initial_pitch, spec.end_pitch, *spec.fixed_soundings.values()) if p is not None}
        self.pitches = (SILENCE,) + tuple(sorted(set(spec.pitches) | carried))
        self.pitch_index = {p: i for i, p in enumerate(self.pitches)}
        self.size = len(self.pitches)
        self.backend_name = self.requested_backend = 'paired'
        self.planning_stats = {
            'selected_backend': 'paired', 'width': self.width,
            'count_limit': self.count_limit, 'pitch_states_per_span': self.size,
            'state_entries': (self.count_limit + 1) * self.size ** 2,
            'contraction_entries': self.size ** 3,
        }
        self.budget.check_factor_entries(max(self.size ** 3, (self.count_limit + 1) * self.size ** 2),
                                         context='Paired log contraction')
        self.storage_bytes = (self._base_bytes() + 8 * (
            (self.width + 2) * (self.count_limit + 1) * self.size ** 2 +
            4 * self.width * self.size ** 2 + 4 * self.size ** 3))
        self.budget.check_workspace_bytes(self.storage_bytes, context='Paired saved messages and log workspace')

    def _outside(self):
        """All constant edges, two initial states and two right-boundary edges."""
        guards = {span[-1] + 1 for span in self.spans if span[-1] + 1 < self.spec.length}
        inside = set(self.spans[0] + self.spans[1])
        constant = 0.0
        initial, terminal = [], []
        for span in self.spans:
            p = self.spec.initial_pitch if span[0] == 0 else self.spec.fixed_soundings[span[0] - 1]
            initial.append(self.pitch_index.get(SILENCE if p is None else p))
            guard = span[-1] + 1
            tail = np.zeros(self.size)
            if guard < self.spec.length:
                for j, previous in enumerate(self.pitches):
                    edge = self._local(guard, previous, self.spec.observed[guard])
                    tail[j] = -np.inf if edge is None else edge[1]
            terminal.append(tail)
        for pos in sorted(set(range(self.spec.length)) - inside - guards):
            previous = self.spec.initial_pitch if pos == 0 else self.spec.fixed_soundings[pos - 1]
            edge = self._local(pos, SILENCE if previous is None else previous, self.spec.observed[pos])
            if edge is None:
                return -np.inf, initial, terminal
            constant += edge[1]
        return constant, initial, terminal

    def _transfers(self, choices):
        transfers = []
        for a, b in zip(*self.spans):
            step = []
            for pos in (a, b):
                matrices = np.full((2, self.size, self.size), -np.inf)
                for token in choices[pos]:
                    bit = int(token >= 2)
                    if pos in self.forced_bits and bit != self.forced_bits[pos]:
                        continue
                    for old, previous in enumerate(self.pitches):
                        edge = self._local(pos, previous, token)
                        if edge is not None:
                            new = self.pitch_index[edge[0]]
                            matrices[bit, old, new] = np.logaddexp(
                                matrices[bit, old, new], self._q(pos, token) + edge[1])
                step.append(matrices)
            transfers.append(step)
        return transfers

    @staticmethod
    def _contract(alpha, a, b):
        # alpha[oldA,oldB]; first sum oldA, then oldB. No q normalization
        # or exponentiation of tiny model masses is needed.
        temporary = logsumexp(alpha[:, :, None] + a[:, None, :], axis=0)
        return logsumexp(temporary[:, :, None] + b[:, None, :], axis=0)

    def _forward(self, evidence, save=False):
        choices, _ = self._choices(evidence)
        constant, initial, terminal = self._outside()
        stats = {**self.planning_stats, 'backend': 'paired', 'cache_hit': False,
                 'peak_workspace_bytes': self.storage_bytes}
        if self.contradictory or any(not c for c in choices) or None in initial or not math.isfinite(constant):
            return -np.inf, None, None, None, stats
        transfers = self._transfers(choices)
        alpha = np.full((self.count_limit + 1, self.size, self.size), -np.inf)
        alpha[0, initial[0], initial[1]] = 0.0
        layers = [alpha] if save else None
        for t, (a, b) in enumerate(transfers):
            next_alpha = np.full_like(alpha, -np.inf)
            for count in range(min(t + 1, self.count_limit) + 1):
                if count <= t and np.isfinite(alpha[count]).any():
                    next_alpha[count] = self._contract(alpha[count], a[0], b[0])
                if count and np.isfinite(alpha[count - 1]).any():
                    next_alpha[count] = np.logaddexp(next_alpha[count], self._contract(alpha[count - 1], a[1], b[1]))
            alpha = next_alpha
            if save:
                layers.append(alpha)
        final = alpha + terminal[0][None, :, None] + terminal[1][None, None, :]
        if self.target is not None:
            final[:self.target] = -np.inf
        return float(logsumexp(final)) + constant, layers, transfers, final, stats

    def log_partition(self, evidence=None):
        evidence = self._evidence(evidence)
        key = tuple(sorted(evidence.items()))
        if key in self._cache:
            value, stats = self._cache[key]
            self._cache.move_to_end(key)
            self.last_stats = {**stats, 'cache_hit': True}
            return value
        value, _, _, _, stats = self._forward(evidence)
        self._cache[key] = value, stats
        if len(self._cache) > 128:
            self._cache.popitem(last=False)
        self.last_stats = stats
        return value

    def _draw_array(self, log_weights, rng):
        z = float(logsumexp(log_weights))
        if not math.isfinite(z):
            raise ZeroMass('No supported paired continuation')
        p = np.exp(log_weights.ravel() - z)
        p /= p.sum()
        return tuple(int(v) for v in np.unravel_index(rng.choice(len(p), p=p), log_weights.shape))

    def sample_batch(self, variables, rng, evidence=None):
        if isinstance(variables, str):
            raise InvalidSpecification('Batch variables must be a sequence')
        variables = tuple(variables)
        if len(set(variables)) != len(variables) or any(v not in self.graph.domains for v in variables):
            raise InvalidSpecification('Batch variables must be distinct original-token names')
        if not isinstance(rng, np.random.Generator):
            raise InvalidSpecification('Expected numpy Generator')
        evidence = self._evidence(evidence)
        base = self.log_partition(evidence)
        if not math.isfinite(base):
            raise ZeroMass('Cannot sample a zero-mass paired query')
        if not variables:
            return BatchSample({}, 0.0, base)
        _, layers, transfers, final, stats = self._forward(evidence, save=True)
        count, current_a, current_b = self._draw_array(final, rng)
        full = [self.spec.observed.get(i) for i in range(self.spec.length)]
        for t in range(self.width - 1, -1, -1):
            a, b = transfers[t]
            weights = np.full((2, self.size, self.size), -np.inf)
            for bit in (0, 1):
                if count >= bit:
                    weights[bit] = layers[t][count - bit] + a[bit, :, current_a][:, None] + b[bit, :, current_b][None, :]
            bit, old_a, old_b = self._draw_array(weights, rng)
            for span, current in zip(self.spans, (current_a, current_b)):
                full[span[t]] = self.pitches[current] + 2 if bit else (0 if current == 0 else 1)
            current_a, current_b, count = old_a, old_b, count - bit
        if not verify_music(full, self.spec).valid:
            raise VerificationError('Paired sampler produced invalid music')
        assignment = {name: int(full[self._positions[name]]) for name in variables}
        clamped = self.log_partition({**evidence, **assignment})
        self.last_stats = {**self.last_stats, 'sampling': stats}
        return BatchSample(assignment, clamped - base, clamped)
