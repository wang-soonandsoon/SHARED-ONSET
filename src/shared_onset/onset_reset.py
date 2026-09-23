"""Exact shared-onset decomposition with independent pitch paths between resets.

The target is the supplied full-vocabulary q times hard MusicSpec constraints.
An optional proposal retains motion scores at OUTSIDE positions, including
right guards. Its partition is a proposal partition, not the soft target Z.
Internal onset transitions must forget the previous pitch: this implementation
requires zero internal motion cost and no hard adjacent-interval constraint.

For R aligned spans, interval preparation costs O(R L^2 D), exploiting the
diagonal HOLD and rank-one REST operators; the onset DAG costs O(L^2 K).
No onset-template enumeration or Cartesian product of R pitches is used.
"""

from dataclasses import dataclass
import math
from numbers import Real
from sys import float_info
from time import perf_counter

import numpy as np
from scipy.special import logsumexp

from shared_onset.music import SILENCE, verify_music
from shared_onset.errors import InvalidSpecification, UnsupportedSpec, VerificationError, ZeroMass
from shared_onset.contracts import BatchSample
from shared_onset._music_base import _MusicQueryBase


@dataclass
class _Prepared:
    key: tuple
    notes: np.ndarray
    holds: np.ndarray
    rests: np.ndarray
    tails: np.ndarray
    edges: np.ndarray
    forward: np.ndarray
    log_z: float
    stats: dict


class OnsetResetMusicInference(_MusicQueryBase):
    """Exact base/proposal queries and complete joint draws on original tokens.

    Supported layouts have R>=2 contiguous, disjoint, equal-length spans;
    all corresponding onset bits are equal and every outside position is
    observed with a sounding anchor. Whole-span counts and singleton onset
    flags are supported. A missing whole-span count means any onset count.
    Only one prepared query is cached; changing evidence keeps original q.
    """

    def __init__(self, spec, log_probs, budget=None, *, boundary_motion_cost=0.0):
        super().__init__(spec, log_probs, budget)
        if spec.motion_cost != 0 or spec.max_adjacent_interval is not None:
            raise UnsupportedSpec('onset_reset requires motion_cost=0 and no max_adjacent_interval')
        if (isinstance(boundary_motion_cost, bool) or not isinstance(boundary_motion_cost, Real)
                or not math.isfinite(boundary_motion_cost) or boundary_motion_cost < 0
                or boundary_motion_cost > float_info.max / (127 * spec.length)):
            raise InvalidSpecification('boundary_motion_cost must be finite, nonnegative and bounded')
        self.boundary_motion_cost = float(boundary_motion_cost)
        self.backend_name = self.requested_backend = 'onset_reset'
        self.planning_stats = {'selected_backend': 'onset_reset', 'planner': 'synchronized_onset_reset'}
        self._prepared = None
        self._compile_layout()

    def _query_metadata_bytes(self):
        # The general template engine stores a counter-by-position layout.
        # Reset inference only needs relation components and scalar K/flags;
        # neither allocate nor charge for a quadratic-in-R counter layout.
        spec = self.spec
        entries = sum(len(rule.positions) for rule in spec.onset_counts)
        return spec.length * 512 + 128 * (len(spec.equal_onsets) + entries)

    def _build_relations(self):
        parents = {}

        def find(p):
            parents.setdefault(p, p)
            while parents[p] != p:
                parents[p] = parents[parents[p]]
                p = parents[p]
            return p

        for a, b in self.spec.equal_onsets:
            if a != b:
                parents[find(b)] = find(a)
        components = {}
        for p in sorted(parents):
            components.setdefault(find(p), []).append(p)
        self._groups = tuple(tuple(c) for c in sorted(components.values(), key=lambda c: c[0]))

    def _compile_layout(self):
        spec = self.spec
        parents = {}

        def find(p):
            parents.setdefault(p, p)
            while parents[p] != p:
                parents[p] = parents[parents[p]]
                p = parents[p]
            return p

        for a, b in spec.equal_onsets:
            if a != b:
                parents[find(b)] = find(a)
        components = {}
        for position in sorted(parents):
            components.setdefault(find(position), []).append(position)
        columns = sorted((tuple(c) for c in components.values()), key=lambda c: c[0])
        if not columns or len(columns[0]) < 2:
            raise UnsupportedSpec('onset_reset needs at least two aligned related spans')
        R = len(columns[0])
        if any(len(c) != R for c in columns):
            raise UnsupportedSpec('All onset components must have the same number of spans')
        self.spans = tuple(tuple(c[r] for c in columns) for r in range(R))
        self.R, self.L = R, len(columns)
        if any(span != tuple(range(span[0], span[0] + self.L)) for span in self.spans):
            raise UnsupportedSpec('Related spans must be contiguous and correspond by position')
        if any(a[-1] + 1 >= b[0] for a, b in zip(self.spans, self.spans[1:])):
            raise UnsupportedSpec('Related spans need an observed separator')
        self._inside = frozenset(p for span in self.spans for p in span)
        self._outside = tuple(p for p in range(spec.length) if p not in self._inside)
        if any(p not in spec.observed or p not in spec.fixed_soundings for p in self._outside):
            raise UnsupportedSpec('Outside positions must be observed with fixed sounding anchors')
        at = {p: j for span in self.spans for j, p in enumerate(span)}
        counts, self._flags = set(), {}
        span_sets = set(self.spans)
        self._contradiction = False
        for rule in spec.onset_counts:
            if not rule.positions:
                continue
            if rule.positions in span_sets:
                counts.add(rule.count)
            elif len(rule.positions) == 1 and rule.positions[0] in at:
                j = at[rule.positions[0]]
                if j in self._flags and self._flags[j] != rule.count:
                    self._contradiction = True
                self._flags[j] = rule.count
            else:
                raise UnsupportedSpec('Only full-span counts and in-span singleton onset flags are supported')
        self._contradiction |= len(counts) > 1
        self.K = min(counts) if counts else None
        pitches = set(spec.pitches)
        pitches.update(p for p in spec.fixed_soundings.values() if p is not None)
        pitches.update(p for p in (spec.initial_pitch, spec.end_pitch) if p is not None)
        self._pitches = (SILENCE, *sorted(pitches))
        self.D = len(self._pitches)
        self._pitch_index = {p: i for i, p in enumerate(self._pitches)}
        self._left = tuple(self._pitch_index[self._sounding(spec.initial_pitch if span[0] == 0
                                else spec.fixed_soundings[span[0] - 1])] for span in self.spans)
        self._guards = frozenset(span[-1] + 1 for span in self.spans if span[-1] + 1 < spec.length)

    @staticmethod
    def _sounding(pitch):
        return SILENCE if pitch is None else pitch

    def _outside_local(self, position, previous):
        token = self.spec.observed[position]
        edge = self._local(position, previous, token)
        if edge is None:
            return -math.inf
        jump = abs(token - 2 - previous) if token >= 2 and previous != SILENCE else 0
        return -self.boundary_motion_cost * jump

    def _workspace(self):
        R, L, D = self.R, self.L, self.D
        F = (L + 1) * (1 if self.K is None else self.K + 1)
        arrays = (R * L * D, R * L, R * D, (L + 1) * (L + 2), F, (L + 1) * D)
        for entries in arrays:
            self.budget.check_factor_entries(entries, context='onset_reset actual array')
        # Both compiled NOTE/HOLD tensors, one live query, interval workspace,
        # sampling forward table and lse temporaries, plus Python token choices.
        estimate = self._base_bytes() + 8 * (2 * arrays[0] + sum(arrays[1:]) + 6 * (L + 1) * D)
        estimate += self.spec.length * (self.D + 2) * 64
        self.budget.check_workspace_bytes(estimate, context='onset_reset preparation and sampling workspace')
        return estimate

    @staticmethod
    def _non_onset(vector, holds, rest):
        result = vector + holds
        result[..., 0] = logsumexp(vector, axis=-1) + rest
        return result

    def _prepare(self, evidence=None):
        evidence = self._evidence(evidence)
        key = tuple(sorted(evidence.items()))
        if self._prepared is not None and key == self._prepared.key:
            self.last_stats = {**self._prepared.stats, 'cache_hit': True, 'prepare_seconds_this_call': 0.0}
            return self._prepared
        self._prepared = None
        start = perf_counter()
        workspace = self._workspace()
        choices, _ = self._choices(evidence)
        R, L, D = self.R, self.L, self.D
        notes = np.full((R, L, D), -np.inf)
        holds = np.full_like(notes, -np.inf)
        rests = np.full((R, L), -np.inf)
        tails = np.full((R, D), -np.inf)
        constant = -math.inf if self._contradiction else 0.0
        for p in self._outside:
            if p in self._guards:
                continue
            previous = self._sounding(self.spec.initial_pitch if p == 0 else self.spec.fixed_soundings[p - 1])
            constant += self._outside_local(p, previous)
        for r, span in enumerate(self.spans):
            for j, p in enumerate(span):
                allowed = choices[p]
                flag = self._flags.get(j)
                if flag != 0:
                    for token in allowed:
                        if token >= 2 and self._local(p, SILENCE, token) is not None:
                            notes[r, j, self._pitch_index[token - 2]] = self._q(p, token)
                if flag != 1:
                    if 0 in allowed and self._local(p, SILENCE, 0) is not None:
                        rests[r, j] = self._q(p, 0)
                    if 1 in allowed:
                        for d, pitch in enumerate(self._pitches):
                            if self._local(p, pitch, 1) is not None:
                                holds[r, j, d] = self._q(p, 1)
            right = span[-1] + 1
            if right == self.spec.length:
                tails[r, :] = 0.0
            else:
                tails[r, :] = [self._outside_local(right, pitch) for pitch in self._pitches]
        # Edge (t,u) sums the pitch path beginning with an onset at t and
        # ending just before the next onset u; t=0 denotes the fixed left
        # state and u=L+1 includes the right guard. Every factor appears once.
        edges = np.full((L + 1, L + 2), -np.inf)
        for t in range(L + 1):
            edges[t, t + 1:] = 0.0
        for r in range(R):
            vectors = np.full((L + 1, D), -np.inf)
            vectors[0, self._left[r]] = 0.0
            for u in range(1, L + 1):
                edges[:u, u] += logsumexp(vectors[:u], axis=1)
                vectors[:u] = self._non_onset(vectors[:u], holds[r, u - 1], rests[r, u - 1])
                vectors[u] = notes[r, u - 1]
            edges[:, L + 1] += logsumexp(vectors + tails[r], axis=1)
        forward = np.full((L + 1, 1 if self.K is None else self.K + 1), -np.inf)
        forward[0, 0] = 0.0
        for u in range(1, L + 1):
            if self.K is None:
                forward[u, 0] = logsumexp(forward[:u, 0] + edges[:u, u])
            else:
                stop = min(u, self.K)
                if stop:
                    forward[u, 1:stop + 1] = logsumexp(forward[:u, :stop] + edges[:u, u, None], axis=0)
        log_z = float(constant + logsumexp(forward[:, -1] + edges[:, L + 1]))
        stats = {'backend': 'onset_reset', 'spans': R, 'span_length': L, 'pitch_states': D,
                 'onset_count': self.K, 'interval_edges': (L + 1) * (L + 2) // 2,
                 'workspace_bytes_estimate': workspace, 'cache_hit': False,
                 'boundary_motion_cost': self.boundary_motion_cost,
                 'partition_semantics': 'proposal' if self.boundary_motion_cost else 'hard_target',
                 'prepare_seconds': perf_counter() - start}
        stats['prepare_seconds_this_call'] = stats['prepare_seconds']
        self._prepared = _Prepared(key, notes, holds, rests, tails, edges, forward, log_z, stats)
        self.last_stats = dict(stats)
        return self._prepared

    def log_partition(self, evidence=None):
        return self._prepare(evidence).log_z

    @staticmethod
    def _draw_index(log_weights, rng):
        normalizer = float(logsumexp(log_weights))
        if not math.isfinite(normalizer):
            raise ZeroMass('No positive-mass sampling continuation')
        probabilities = np.exp(log_weights - normalizer)
        probabilities /= probabilities.sum()
        return int(rng.choice(len(probabilities), p=probabilities))

    def sample_full(self, rng, evidence=None):
        if not isinstance(rng, np.random.Generator):
            raise InvalidSpecification('rng must be a numpy Generator')
        plan = self._prepare(evidence)
        if not math.isfinite(plan.log_z):
            raise ZeroMass('onset_reset conditioning event has zero mass')
        start = perf_counter()
        bits = np.zeros(self.L, dtype=np.int8)
        k = 0 if self.K is None else self.K
        t = self._draw_index(plan.forward[:, k] + plan.edges[:, self.L + 1], rng)
        while t:
            bits[t - 1] = 1
            previous_k = 0 if self.K is None else k - 1
            t = self._draw_index(plan.forward[:t, previous_k] + plan.edges[:t, t], rng)
            k = previous_k
        tokens = [self.spec.observed.get(p, 0) for p in range(self.spec.length)]
        for r, span in enumerate(self.spans):
            alpha = np.full((self.L + 1, self.D), -np.inf)
            alpha[0, self._left[r]] = 0.0
            for j, bit in enumerate(bits):
                alpha[j + 1] = (logsumexp(alpha[j]) + plan.notes[r, j] if bit
                                else self._non_onset(alpha[j], plan.holds[r, j], plan.rests[r, j]))
            current = self._draw_index(alpha[-1] + plan.tails[r], rng)
            for j in range(self.L - 1, -1, -1):
                if bits[j]:
                    tokens[span[j]] = self._pitches[current] + 2
                    current = self._draw_index(alpha[j], rng)
                elif current == 0:
                    tokens[span[j]] = 0
                    current = self._draw_index(alpha[j], rng)
                else:
                    tokens[span[j]] = 1
        result = tuple(tokens)
        checked = verify_music(result, self.spec)
        if not checked.valid:
            raise VerificationError('onset_reset produced an invalid full sample: ' + '; '.join(checked.violations))
        self.last_stats = {**self.last_stats, 'sample_seconds': perf_counter() - start,
                           'sample_onset_count': int(bits.sum())}
        return result

    def sample_batch(self, variables, rng, evidence=None):
        variables = tuple(variables)
        if len(set(variables)) != len(variables) or any(v not in self.graph.domains for v in variables):
            raise InvalidSpecification('Batch variables must be unique original-token variables')
        evidence = self._evidence(evidence)
        base = self.log_partition(evidence)
        sample = self.sample_full(rng, evidence)
        assignment = {v: sample[self._positions[v]] for v in variables}
        fixed = {**evidence, **assignment}
        if all(i in self.spec.observed or f'y{i}' in fixed for i in range(self.spec.length)):
            # A complete token clamp has exactly one path. Preserve the base
            # preparation for repeated draws; no second DP is needed.
            clamped = self.log_weight(sample)
        else:
            clamped = self.log_partition(fixed)
        return BatchSample(assignment, clamped - base, clamped)

    def retained_soft_score(self, tokens):
        checked = verify_music(tokens, self.spec)
        if not checked.valid:
            raise InvalidSpecification('retained_soft_score requires a valid complete sample')
        previous = self.spec.initial_pitch
        jumps = 0
        for position, token in enumerate(tokens):
            if token == 0:
                previous = None
            elif token >= 2:
                pitch = token - 2
                if previous is not None and position not in self._inside:
                    jumps += abs(pitch - previous)
                previous = pitch
        return -self.boundary_motion_cost * jumps

    def log_weight(self, tokens):
        if not verify_music(tokens, self.spec).valid:
            return -math.inf
        return float(sum(self._q(i, token) for i, token in enumerate(tokens)) + self.retained_soft_score(tokens))
