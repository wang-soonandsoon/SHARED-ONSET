"""Exact proposal with rank-one onset factors from initially visible conditions.

T_rj(u,a)=exp(incoming_rj(u)+notes_rj(a)) retains NOTE motion when
its predecessor is fixed or its NOTE destination is initially observed.
Outside scores are retained as in the boundary proposal. All other motion
is left to exact rejection. The supplied original soft target is unchanged.
"""
from dataclasses import dataclass, replace
import math
from time import perf_counter

import numpy as np
from scipy.special import logsumexp

from tri.domain.music import SILENCE, verify_music
from tri.errors import InvalidSpecification, UnsupportedSpec, VerificationError, ZeroMass
from tri.inference.onset_reset import OnsetResetMusicInference, _Prepared


@dataclass
class _RankOnePrepared(_Prepared):
    incoming: np.ndarray


class RankOneOnsetMusicInference(OnsetResetMusicInference):
    """Proposal-only partition and shared-pattern/conditional-span sampling."""

    def __init__(self, spec, log_probs, budget=None, *, retain_visible=True):
        if spec.max_adjacent_interval is not None:
            raise UnsupportedSpec('rank-one proposal does not relax hard intervals')
        if not isinstance(retain_visible, bool):
            raise InvalidSpecification('retain_visible must be bool')
        self.original_spec = spec
        self.motion_beta = spec.motion_cost
        self.retain_visible = retain_visible
        super().__init__(replace(spec, motion_cost=0.), log_probs, budget,
                         boundary_motion_cost=spec.motion_cost)
        self.backend_name = self.requested_backend = 'rank_one_onset'

    def _prepare(self, evidence=None):
        evidence = self._evidence(evidence)
        key = tuple(sorted(evidence.items()))
        if self._prepared is not None and key == self._prepared.key:
            self.last_stats = {**self._prepared.stats, 'cache_hit': True, 'prepare_seconds_this_call': 0.0}
            return self._prepared
        self._prepared = None
        start = perf_counter()
        workspace = self._workspace()
        self.budget.check_workspace_bytes(workspace + 8 * self.R * self.L * self.D, context="rank-one incoming factors")
        workspace += 8 * self.R * self.L * self.D
        choices, _ = self._choices(evidence)
        R, L, D = self.R, self.L, self.D
        notes = np.full((R, L, D), -np.inf)
        incoming = np.zeros_like(notes)
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
                # Eligibility depends on the initial specification, never on a
                # sampled/evidence destination. A known predecessor contributes
                # a destination-only term; an observed NOTE contributes an
                # incoming-only term. Both preserve T(old,new)=phi(old)psi(new).
                if self.retain_visible:
                    if p == 0 or p - 1 in self.spec.fixed_soundings:
                        previous = self.spec.initial_pitch if p == 0 else self.spec.fixed_soundings[p - 1]
                        if previous is not None:
                            notes[r, j] += [-self.motion_beta * abs(pitch - previous) if pitch != SILENCE else 0. for pitch in self._pitches]
                    elif self.spec.observed.get(p, 0) >= 2:
                        destination = self.spec.observed[p] - 2
                        incoming[r, j] = [-self.motion_beta * abs(destination - pitch) if pitch != SILENCE else 0. for pitch in self._pitches]
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
                edges[:u, u] += logsumexp(vectors[:u] + incoming[r, u - 1], axis=1)
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
        stats = {'backend': 'rank_one_onset', 'spans': R, 'span_length': L, 'pitch_states': D,
                 'onset_count': self.K, 'interval_edges': (L + 1) * (L + 2) // 2,
                 'workspace_bytes_estimate': workspace, 'cache_hit': False,
                 'boundary_motion_cost': self.boundary_motion_cost, 'retain_visible': self.retain_visible,
                 'partition_semantics': 'proposal',
                 'prepare_seconds': perf_counter() - start}
        stats['prepare_seconds_this_call'] = stats['prepare_seconds']
        self._prepared = _RankOnePrepared(key, notes, holds, rests, tails, edges, forward, log_z, stats, incoming)
        self.last_stats = dict(stats)
        return self._prepared

    def sample_pattern(self, rng):
        """Draw the entire shared rhythm before any conditional pitch paths."""
        if not isinstance(rng, np.random.Generator):
            raise InvalidSpecification('rng must be a numpy Generator')
        plan = self._prepare()
        if not math.isfinite(plan.log_z):
            raise ZeroMass('rank-one proposal has zero mass')
        bits = np.zeros(self.L, dtype=np.int8)
        k = 0 if self.K is None else self.K
        t = self._draw_index(plan.forward[:, k] + plan.edges[:, self.L + 1], rng)
        while t:
            bits[t - 1] = 1
            previous_k = 0 if self.K is None else k - 1
            t = self._draw_index(plan.forward[:t, previous_k] + plan.edges[:t, t], rng)
            k = previous_k
        return tuple(int(bit) for bit in bits)

    def sample_span(self, r, bits, rng):
        """Draw a whole conditional span; no within-span partial retry occurs."""
        if not isinstance(rng, np.random.Generator):
            raise InvalidSpecification('rng must be a numpy Generator')
        if isinstance(r, bool) or not isinstance(r, (int, np.integer)) or not 0 <= r < self.R:
            raise InvalidSpecification('span index out of range')
        if len(bits) != self.L or any(bit not in (0, 1) for bit in bits):
            raise InvalidSpecification('invalid shared rhythm')
        if self.K is not None and sum(bits) != self.K:
            raise InvalidSpecification('shared rhythm violates onset count')
        plan = self._prepare()
        alpha = np.full((self.L + 1, self.D), -np.inf)
        alpha[0, self._left[r]] = 0.
        for j, bit in enumerate(bits):
            alpha[j + 1] = (logsumexp(alpha[j] + plan.incoming[r, j]) + plan.notes[r, j] if bit
                else self._non_onset(alpha[j], plan.holds[r, j], plan.rests[r, j]))
        current = self._draw_index(alpha[-1] + plan.tails[r], rng)
        tokens = [0] * self.L
        for j in range(self.L - 1, -1, -1):
            if bits[j]:
                tokens[j] = self._pitches[current] + 2
                current = self._draw_index(alpha[j] + plan.incoming[r, j], rng)
            elif current == 0:
                tokens[j] = 0
                current = self._draw_index(alpha[j], rng)
            else:
                tokens[j] = 1
        return tuple(tokens)

    def sample_full(self, rng, evidence=None):
        # Parent normalized query interface supports evidence. Component APIs
        # intentionally use one fixed initial query, as required for rejection.
        if evidence:
            raise UnsupportedSpec('rank-one sampling components use the initial fixed proposal; conditional partition queries remain available')
        started = perf_counter()
        bits = self.sample_pattern(rng)
        tokens = [self.spec.observed.get(p, 0) for p in range(self.spec.length)]
        for r, span in enumerate(self.spans):
            for p, token in zip(span, self.sample_span(r, bits, rng)):
                tokens[p] = token
        result = tuple(tokens)
        checked = verify_music(result, self.original_spec)
        if not checked.valid:
            raise VerificationError('Rank-one proposal returned invalid original tokens')
        self.last_stats = {**self.last_stats, 'sample_seconds': perf_counter() - started,
                           'sample_onset_count': sum(bits)}
        return result

    def retained_soft_score(self, tokens):
        if not verify_music(tokens, self.original_spec).valid:
            raise InvalidSpecification('retained score requires a valid full path')
        previous, jumps = self.spec.initial_pitch, 0
        for p, token in enumerate(tokens):
            if token == 0:
                previous = None
            elif token >= 2:
                pitch = token - 2
                retained = p not in self._inside or (self.retain_visible and (
                    p == 0 or p - 1 in self.spec.fixed_soundings or self.spec.observed.get(p, 0) >= 2))
                if retained and previous is not None:
                    jumps += abs(pitch - previous)
                previous = pitch
        return -self.motion_beta * jumps

    def span_residual_score(self, r, tokens):
        """Unretained original motion factors in one complete span.

        All outside factors belong to the proposal, including right guards.
        Visible eligibility is fixed before any sampling takes place.
        """
        if len(tokens) != self.L:
            raise InvalidSpecification('Expected a complete span')
        span = self.spans[r]
        previous = self.spec.initial_pitch if span[0] == 0 else self.spec.fixed_soundings[span[0] - 1]
        jumps = 0
        for p, token in zip(span, tokens):
            if token == 0:
                previous = None
            elif token >= 2:
                pitch = token - 2
                retained = self.retain_visible and (p == 0 or p - 1 in self.spec.fixed_soundings or self.spec.observed.get(p, 0) >= 2)
                if not retained and previous is not None:
                    jumps += abs(pitch - previous)
                previous = pitch
        return -self.motion_beta * jumps
