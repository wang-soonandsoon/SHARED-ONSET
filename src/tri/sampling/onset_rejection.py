"""Exact full-path rejection sampling from an onset-reset proposal.

The proposal normalizer belongs to the relaxed/retained-factor proposal.
This sampler deliberately has no target partition or normalized batch API.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math
from numbers import Integral
from time import perf_counter

import numpy as np

from tri.domain.music import MusicSpec, verify_music
from tri.errors import BudgetExceeded, InvalidSpecification, UnsupportedSpec, VerificationError, ZeroMass
from tri.inference.exact import Budget


@dataclass(frozen=True)
class OnsetRejectionDraw:
    tokens: tuple[int, ...]
    diagnostics: dict


class OnsetRejectionSampler:
    """Sample the original fixed-q soft target without claiming its Z.

    ``drop`` removes all motion factors from the proposal. ``boundary`` keeps
    motion factors at positions outside the shared spans, including their
    right NOTE tails. Both proposals retain exactly the original hard support.
    ``max_proposals`` applies separately to each requested accepted sample.
    Repeated calls reuse proposal preprocessing, while lifetime counters retain
    the cost of every rejected candidate. Budget failures never return a path.
    An optional observational ``progress_callback(dict)`` receives proposal
    starts and completed acceptance decisions; its wall cost is included in
    the call, and it does not receive or modify the RNG or proposal tokens.
    """

    # Python records and the durable returned/last-stats snapshots coexist.
    # This is a conservative storage estimate, not just ndarray accounting.
    _DIAGNOSTIC_BASE_BYTES = 8192
    _DIAGNOSTIC_BYTES_PER_ATTEMPT = 4096

    def __init__(self, spec, log_probs, budget=None, *, proposal='drop', max_proposals=10_000,
                 progress_callback=None):
        self.spec = spec
        self.proposal = proposal
        self.max_proposals = max_proposals
        self.progress_callback = progress_callback
        self.budget = budget if budget is not None else Budget()
        self.proposal_engine = None
        self.proposal_log_z = None
        self.precompute_seconds = 0.0
        self.proposal_construction_seconds = 0.0
        self.proposal_partition_seconds = 0.0
        self._proposal_workspace_bytes = 0
        self._totals = dict(proposals=0, proposal_calls=0, rejections=0, accepted_samples=0,
                            proposal_seconds=0.0, score_acceptance_seconds=0.0,
                            verification_seconds=0.0, sampling_seconds=0.0)
        self.last_stats = {}
        start = perf_counter()
        phase = 'validate_specification'
        try:
            if not isinstance(spec, MusicSpec):
                raise InvalidSpecification('onset rejection requires a MusicSpec')
            if not isinstance(self.budget, Budget):
                raise InvalidSpecification('budget must be a Budget')
            if proposal not in ('drop', 'boundary'):
                raise InvalidSpecification('onset rejection proposal must be drop or boundary')
            if progress_callback is not None and not callable(progress_callback):
                raise InvalidSpecification('progress_callback must be callable or None')
            if isinstance(max_proposals, (bool, np.bool_)) or not isinstance(max_proposals, Integral) or max_proposals < 1:
                raise InvalidSpecification('max_proposals must be a positive integer')
            self.max_proposals = int(max_proposals)
            if spec.max_adjacent_interval is not None:
                raise UnsupportedSpec('onset rejection does not support max_adjacent_interval; no hard-rule relaxation is performed')
            # Lazy import keeps the independent sampling contract separate from
            # the general exact-engine factory and its normalized batch API.
            from tri.inference.onset_reset import OnsetResetMusicInference
            phase = 'proposal_construction'
            phase_start = perf_counter()
            try:
                self.proposal_engine = OnsetResetMusicInference(
                    replace(spec, motion_cost=0.0), log_probs, self.budget,
                    boundary_motion_cost=spec.motion_cost if proposal == 'boundary' else 0.0)
            finally:
                self.proposal_construction_seconds = perf_counter() - phase_start
            phase = 'proposal_partition'
            phase_start = perf_counter()
            try:
                self.proposal_log_z = float(self.proposal_engine.log_partition())
            finally:
                self.proposal_partition_seconds = perf_counter() - phase_start
            if self.proposal_log_z == -math.inf:
                # Finite nonnegative beta changes only weights, not support.
                raise ZeroMass('onset proposal has zero mass; original finite-motion target has the same support')
            if not math.isfinite(self.proposal_log_z):
                raise VerificationError('onset proposal returned an invalid partition value')
            self._proposal_workspace_bytes = int(self.proposal_engine.last_stats['workspace_bytes_estimate'])
            phase = 'diagnostic_workspace'
            self._check_diagnostic_workspace(0)
        except Exception as error:
            self.precompute_seconds = perf_counter() - start
            self._finish(self._blank_call(), self._status(error), error, phase)
            raise
        self.precompute_seconds = perf_counter() - start
        self._finish(self._blank_call(), 'ready')

    @staticmethod
    def _blank_call():
        return dict(proposals=0, proposal_calls=0, rejections=0, accepted=0,
                    proposal_seconds=0.0, score_acceptance_seconds=0.0,
                    verification_seconds=0.0, sampling_seconds=0.0, attempts=[])

    @staticmethod
    def _status(error):
        if isinstance(error, BudgetExceeded):
            return 'budget_exceeded'
        if isinstance(error, ZeroMass):
            return 'zero_mass'
        if isinstance(error, UnsupportedSpec):
            return 'unsupported'
        if isinstance(error, InvalidSpecification):
            return 'invalid_specification'
        return 'verification_failed' if isinstance(error, VerificationError) else 'error'

    def _finish(self, call, status, error=None, phase=None):
        stats = dict(call, status=status, proposal=self.proposal, max_proposals=self.max_proposals,
                     proposal_log_z=self.proposal_log_z, target_normalizer_available=False,
                     timing_scope='Precompute and proposal-loop wall; returned diagnostic copying excluded. Use external API wall for total latency.',
                     precompute_seconds=self.precompute_seconds,
                     proposal_construction_seconds=self.proposal_construction_seconds,
                     proposal_partition_seconds=self.proposal_partition_seconds,
                     diagnostic_workspace_bytes_estimate=self._diagnostic_bytes(len(call['attempts'])),
                     total_seconds=self.precompute_seconds + self._totals['sampling_seconds'],
                     **{f'cumulative_{key}': value for key, value in self._totals.items()})
        if self.proposal_engine is not None:
            stats['proposal_stats'] = deepcopy(self.proposal_engine.last_stats)
        if error is not None:
            stats.update(failed_phase=phase, error_type=type(error).__name__, error=str(error))
        self.last_stats = deepcopy(stats)
        if error is not None:
            error.diagnostics = deepcopy(stats)
        return deepcopy(stats)

    def _diagnostic_bytes(self, current_attempts):
        previous_attempts = len(self.last_stats.get('attempts', ()))
        return self._DIAGNOSTIC_BASE_BYTES + self._DIAGNOSTIC_BYTES_PER_ATTEMPT * (current_attempts + previous_attempts)

    def _check_diagnostic_workspace(self, current_attempts):
        self.budget.check_workspace_bytes(
            self._proposal_workspace_bytes + self._diagnostic_bytes(current_attempts),
            context='onset rejection proposal plus diagnostic snapshots')

    def _progress(self, event, call, attempt, start):
        if self.progress_callback is None:
            return
        elapsed = perf_counter() - start
        # A small immutable-value snapshot is enough for a supervisor to retain
        # partial work when it kills a process inside a later proposal.
        payload = {
            'event': event, 'proposal': self.proposal,
            'proposal_index': attempt['proposal_index'],
            'in_flight': event == 'proposal_started',
            **{key: call[key] for key in ('proposal_calls', 'proposals', 'rejections', 'accepted')},
            **{f'cumulative_{key}': self._totals[key]
               for key in ('proposal_calls', 'proposals', 'rejections', 'accepted_samples')},
            'precompute_seconds': self.precompute_seconds,
            'sampling_seconds_so_far': elapsed,
            'cumulative_sampling_seconds_so_far': self._totals['sampling_seconds'] + elapsed,
            'attempt': dict(attempt),
        }
        self.progress_callback(payload)

    def sample_full(self, rng):
        """Return one accepted original-target path and cost diagnostics."""
        call = self._blank_call()
        start = perf_counter()
        phase = 'validate_rng'
        try:
            if not isinstance(rng, np.random.Generator):
                raise InvalidSpecification('sampling requires a numpy.random.Generator')
            for index in range(self.max_proposals):
                phase = 'diagnostic_workspace'
                self._check_diagnostic_workspace(len(call['attempts']) + 1)
                attempt = {'proposal_index': index + 1, 'accepted': False}
                call['attempts'].append(attempt)
                call['proposal_calls'] += 1
                self._totals['proposal_calls'] += 1
                phase = 'progress_callback'
                self._progress('proposal_started', call, attempt, start)
                phase = 'proposal_sampling'
                stamp = perf_counter()
                try:
                    tokens = tuple(self.proposal_engine.sample_full(rng))
                finally:
                    elapsed = perf_counter() - stamp
                    attempt['proposal_seconds'] = elapsed
                    call['proposal_seconds'] += elapsed
                    self._totals['proposal_seconds'] += elapsed
                call['proposals'] += 1
                self._totals['proposals'] += 1
                phase = 'original_target_verification'
                stamp = perf_counter()
                try:
                    checked = verify_music(tokens, self.spec)
                    if not checked.valid:
                        raise VerificationError(f'onset proposal violated original hard constraints: {checked.violations}')
                    if not math.isfinite(checked.soft_score):
                        raise VerificationError('original motion score is not finite')
                finally:
                    elapsed = perf_counter() - stamp
                    attempt['verification_seconds'] = elapsed
                    call['verification_seconds'] += elapsed
                    self._totals['verification_seconds'] += elapsed
                phase = 'score_and_accept'
                stamp = perf_counter()
                try:
                    retained = float(self.proposal_engine.retained_soft_score(tokens))
                    if not math.isfinite(retained):
                        raise VerificationError('retained proposal motion score is not finite')
                    residual = checked.soft_score - retained
                    tolerance = max(1e-12, 32 * np.finfo(float).eps * max(1., abs(checked.soft_score), abs(retained)))
                    if retained > tolerance or residual > tolerance:
                        raise VerificationError('proposal motion factors do not give a nonpositive residual score')
                    log_acceptance = min(0.0, residual)
                    uniform = float(rng.random())
                    log_uniform = math.log(uniform) if uniform else -math.inf
                    accept = log_uniform < log_acceptance
                    attempt.update(original_soft_score=checked.soft_score, retained_soft_score=retained,
                                   log_acceptance=log_acceptance, accepted=accept)
                finally:
                    elapsed = perf_counter() - stamp
                    attempt['score_acceptance_seconds'] = elapsed
                    call['score_acceptance_seconds'] += elapsed
                    self._totals['score_acceptance_seconds'] += elapsed
                if accept:
                    call['accepted'] = 1
                    self._totals['accepted_samples'] += 1
                else:
                    call['rejections'] += 1
                    self._totals['rejections'] += 1
                phase = 'progress_callback'
                self._progress('proposal_completed', call, attempt, start)
                if accept:
                    call['sampling_seconds'] = perf_counter() - start
                    self._totals['sampling_seconds'] += call['sampling_seconds']
                    return OnsetRejectionDraw(tokens, self._finish(call, 'success'))
            phase = 'proposal_limit'
            raise BudgetExceeded(f'onset rejection exhausted {self.max_proposals} complete proposals; no accepted sample')
        except Exception as error:
            call['sampling_seconds'] = perf_counter() - start
            self._totals['sampling_seconds'] += call['sampling_seconds']
            self._finish(call, self._status(error), error, phase)
            raise
