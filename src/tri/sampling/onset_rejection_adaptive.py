"""Exact rejection with visible-factor proposals and optional early stopping.

Every attempt draws a fresh shared rhythm. Conditional spans are independent
given that rhythm; testing their nonpositive residual factors separately gives
the same acceptance probability as testing their sum on a full proposal.
An early rejection discards the rhythm and every already drawn span. Retrying
only the failed span would instead change the target and is never performed.
"""
from __future__ import annotations

import math
from numbers import Integral
from time import perf_counter

import numpy as np

from tri.domain.music import MusicSpec, verify_music
from tri.errors import BudgetExceeded, InvalidSpecification, VerificationError, ZeroMass
from tri.inference.exact import Budget
from tri.sampling.onset_rejection import OnsetRejectionDraw, OnsetRejectionSampler


class AdaptiveOnsetRejectionSampler(OnsetRejectionSampler):
    """Proposal-only sampler; no target Z or normalized partial-query API.

    ``visible`` samples a whole visible-factor proposal before one acceptance
    gate. ``boundary_early`` and ``visible_early`` test each conditional span
    and stop an attempt at its first rejection. In all modes ``proposals``
    counts completed shared-rhythm acceptance/rejection decisions, which can
    involve incomplete paths; ``full_candidates`` separately counts fully
    generated paths. ``max_proposals`` is a per-output rhythm-attempt limit.
    """

    _MODES = ('visible', 'boundary_early', 'visible_early')
    _PROPOSAL_UNIT = 'shared_rhythm_attempt_with_completed_accept_or_reject_decision'

    def __init__(self, spec, log_probs, budget=None, *, proposal='visible',
                 max_proposals=10_000, progress_callback=None):
        self.spec, self.proposal = spec, proposal
        self.max_proposals, self.progress_callback = max_proposals, progress_callback
        self.budget = Budget() if budget is None else budget
        self.proposal_engine = self.proposal_log_z = None
        self.precompute_seconds = self.proposal_construction_seconds = self.proposal_partition_seconds = 0.
        self._proposal_workspace_bytes = 0
        self._span_count = 0
        self._totals = {key: value for key, value in self._blank_call().items()
                        if key not in ('attempts', 'accepted')}
        self._totals['accepted_samples'] = 0
        self.last_stats = {}
        start, phase = perf_counter(), 'validate_specification'
        try:
            if not isinstance(spec, MusicSpec) or not isinstance(self.budget, Budget):
                raise InvalidSpecification('adaptive onset rejection requires MusicSpec and Budget')
            if proposal not in self._MODES:
                raise InvalidSpecification(f'adaptive proposal must be one of {self._MODES}')
            if progress_callback is not None and not callable(progress_callback):
                raise InvalidSpecification('progress_callback must be callable or None')
            if isinstance(max_proposals, (bool, np.bool_)) or not isinstance(max_proposals, Integral) or max_proposals < 1:
                raise InvalidSpecification('max_proposals must be a positive integer')
            self.max_proposals = int(max_proposals)
            from tri.inference.rank_one_onset import RankOneOnsetMusicInference
            phase, stamp = 'proposal_construction', perf_counter()
            try:
                self.proposal_engine = RankOneOnsetMusicInference(
                    spec, log_probs, self.budget, retain_visible=proposal != 'boundary_early')
                self._span_count = self.proposal_engine.R
            finally:
                self.proposal_construction_seconds = perf_counter() - stamp
            phase, stamp = 'proposal_partition', perf_counter()
            try:
                self.proposal_log_z = float(self.proposal_engine.log_partition())
            finally:
                self.proposal_partition_seconds = perf_counter() - stamp
            if self.proposal_log_z == -math.inf:
                raise ZeroMass('rank-one proposal and original finite-motion target have zero mass')
            if not math.isfinite(self.proposal_log_z):
                raise VerificationError('proposal returned an invalid normalizer')
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
        return {**OnsetRejectionSampler._blank_call(), 'patterns_sampled': 0,
                'spans_sampled': 0, 'full_candidates': 0, 'partial_candidates': 0}

    def _diagnostic_bytes(self, current_attempts):
        # Each attempt holds at most R compact span records. Account for both
        # current records and the previous durable diagnostic snapshot.
        count = current_attempts + len(self.last_stats.get('attempts', ()))
        working_tokens = 64 * getattr(self.spec, 'length', 0)
        return self._DIAGNOSTIC_BASE_BYTES + working_tokens + (self._DIAGNOSTIC_BYTES_PER_ATTEMPT +
                                                              2048 * self._span_count) * count

    def _finish(self, call, status, error=None, phase=None):
        call = {**call, 'proposal_unit': self._PROPOSAL_UNIT,
                'full_candidate_unit': 'all_spans_generated_before_acceptance_decision',
                'partial_candidate_unit': 'early_rejected_before_all_spans_generated',
                'early_rejection': isinstance(self.proposal, str) and self.proposal.endswith('_early')}
        return super()._finish(call, status, error, phase)

    def _progress(self, event, call, attempt, start):
        if self.progress_callback is None:
            return
        elapsed = perf_counter() - start
        keys = ('proposal_calls', 'proposals', 'rejections', 'patterns_sampled',
                'spans_sampled', 'full_candidates', 'partial_candidates')
        self.progress_callback({
            'event': event, 'proposal': self.proposal, 'proposal_unit': self._PROPOSAL_UNIT,
            'proposal_index': attempt['proposal_index'], 'in_flight': event != 'proposal_completed',
            'accepted': call['accepted'], **{key: call[key] for key in keys},
            **{f'cumulative_{key}': self._totals[key] for key in (*keys, 'accepted_samples')},
            'precompute_seconds': self.precompute_seconds,
            'sampling_seconds_so_far': elapsed,
            'cumulative_sampling_seconds_so_far': self._totals['sampling_seconds'] + elapsed,
            # No token path or mutable internal list is exposed to the hook.
            'attempt': {key: value for key, value in attempt.items() if key != 'span_decisions'},
        })

    def _increment(self, call, name, count=1):
        call[name] += count
        self._totals[name] += count

    def _timed(self, call, attempt, name, function):
        start = perf_counter()
        try:
            return function()
        finally:
            elapsed = perf_counter() - start
            attempt[name] = attempt.get(name, 0.) + elapsed
            call[name] += elapsed
            self._totals[name] += elapsed

    @staticmethod
    def _gate(residual, rng):
        if not math.isfinite(residual) or residual > 1e-12:
            raise VerificationError('proposal residual must be finite and nonpositive')
        uniform = float(rng.random())
        return (math.log(uniform) if uniform else -math.inf) < min(0., residual)

    def _checked_scores(self, tokens, residual_sum=None):
        checked = verify_music(tokens, self.spec)
        if not checked.valid:
            raise VerificationError(f'proposal violated original hard constraints: {checked.violations}')
        retained = float(self.proposal_engine.retained_soft_score(tokens))
        residual = checked.soft_score - retained
        tolerance = max(1e-12, 32 * np.finfo(float).eps *
                        max(1., abs(checked.soft_score), abs(retained)))
        if not all(math.isfinite(x) for x in (checked.soft_score, retained, residual)):
            raise VerificationError('original and retained scores must be finite')
        if retained > tolerance or residual > tolerance:
            raise VerificationError('proposal motion factors do not yield a nonpositive residual')
        if residual_sum is not None and abs(residual_sum - residual) > tolerance:
            raise VerificationError('sum of span residuals disagrees with original minus retained score')
        return checked.soft_score, retained, min(0., residual)

    def sample_full(self, rng):
        call, start, phase = self._blank_call(), perf_counter(), 'validate_rng'
        try:
            if not isinstance(rng, np.random.Generator):
                raise InvalidSpecification('sampling requires a numpy.random.Generator')
            early = self.proposal.endswith('_early')
            for index in range(self.max_proposals):
                phase = 'diagnostic_workspace'
                self._check_diagnostic_workspace(len(call['attempts']) + 1)
                attempt = {'proposal_index': index + 1, 'accepted': False,
                           'spans_sampled': 0, 'full_candidate': False, 'span_decisions': []}
                call['attempts'].append(attempt)
                self._increment(call, 'proposal_calls')
                phase = 'progress_callback'
                self._progress('proposal_started', call, attempt, start)
                phase = 'proposal_sampling'
                if not early:
                    tokens = tuple(self._timed(call, attempt, 'proposal_seconds',
                                              lambda: self.proposal_engine.sample_full(rng)))
                    self._increment(call, 'patterns_sampled')
                    self._increment(call, 'spans_sampled', self._span_count)
                    self._increment(call, 'full_candidates')
                    attempt.update(spans_sampled=self._span_count, full_candidate=True)
                    phase = 'original_target_verification'
                    original, retained, residual = self._timed(
                        call, attempt, 'verification_seconds', lambda: self._checked_scores(tokens))
                    phase = 'score_and_accept'
                    accepted = self._timed(call, attempt, 'score_acceptance_seconds',
                                           lambda: self._gate(residual, rng))
                    attempt.update(original_soft_score=original, retained_soft_score=retained,
                                   log_acceptance=residual)
                else:
                    # New bits are drawn on EVERY attempt, including after a
                    # rejection in span zero. No partial candidate is reused.
                    bits = self._timed(call, attempt, 'proposal_seconds',
                                       lambda: self.proposal_engine.sample_pattern(rng))
                    self._increment(call, 'patterns_sampled')
                    values = [self.spec.observed.get(p, 0) for p in range(self.spec.length)]
                    residual_sum, accepted = 0., True
                    for r, span in enumerate(self.proposal_engine.spans):
                        attempt['span_index'] = r
                        phase = 'progress_callback'
                        self._progress('span_started', call, attempt, start)
                        phase = 'proposal_sampling'
                        span_tokens = self._timed(call, attempt, 'proposal_seconds',
                            lambda: self.proposal_engine.sample_span(r, bits, rng))
                        if len(span_tokens) != len(span):
                            raise VerificationError('proposal returned an incomplete span')
                        for p, token in zip(span, span_tokens):
                            values[p] = token
                        self._increment(call, 'spans_sampled')
                        attempt['spans_sampled'] += 1
                        if r + 1 == self._span_count:
                            self._increment(call, 'full_candidates')
                            attempt['full_candidate'] = True
                        phase = 'score_and_accept'
                        def score_gate():
                            residual = float(self.proposal_engine.span_residual_score(r, span_tokens))
                            return residual, self._gate(residual, rng)
                        residual, accepted = self._timed(call, attempt, 'score_acceptance_seconds', score_gate)
                        residual_sum += residual
                        attempt['span_decisions'].append({'span_index': r, 'residual_score': residual,
                                                          'accepted': accepted})
                        attempt.update(last_span_accepted=accepted, residual_sum=residual_sum)
                        phase = 'progress_callback'
                        self._progress('span_completed', call, attempt, start)
                        if not accepted:
                            if r + 1 < self._span_count:
                                self._increment(call, 'partial_candidates')
                            break
                    if accepted:
                        tokens = tuple(values)
                        phase = 'original_target_verification'
                        original, retained, residual = self._timed(call, attempt, 'verification_seconds',
                            lambda: self._checked_scores(tokens, residual_sum))
                        attempt.update(original_soft_score=original, retained_soft_score=retained,
                                       log_acceptance=residual)
                self._increment(call, 'proposals')
                attempt['accepted'] = accepted
                if accepted:
                    call['accepted'] = 1
                    self._totals['accepted_samples'] += 1
                else:
                    self._increment(call, 'rejections')
                phase = 'progress_callback'
                self._progress('proposal_completed', call, attempt, start)
                if accepted:
                    call['sampling_seconds'] = perf_counter() - start
                    self._totals['sampling_seconds'] += call['sampling_seconds']
                    return OnsetRejectionDraw(tokens, self._finish(call, 'success'))
            phase = 'proposal_limit'
            raise BudgetExceeded(f'exhausted {self.max_proposals} shared-rhythm attempts; no accepted sample')
        except Exception as error:
            call['sampling_seconds'] = perf_counter() - start
            self._totals['sampling_seconds'] += call['sampling_seconds']
            self._finish(call, self._status(error), error, phase)
            raise
