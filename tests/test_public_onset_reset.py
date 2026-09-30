"""Independent original-token oracles for the onset-reset decomposition.

The oracle enumerates token assignments and calls verify_music; it does not use
new transition blocks, interval coefficients, onset DP, or compiler helpers.
"""
from collections import Counter
from dataclasses import replace
from itertools import product
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from shared_onset.music import CountRule, MusicSpec, verify_music
from shared_onset.errors import BudgetExceeded, InvalidSpecification, UnsupportedSpec, ZeroMass
from shared_onset.contracts import Budget
from shared_onset.onset_reset import OnsetResetMusicInference


def reset_case(R=2, L=2, K=1, *, pitches=(60, 62), guards=None, seed=91):
    spans = tuple(tuple(range(1 + r * (L + 1), 1 + r * (L + 1) + L)) for r in range(R))
    observed = {0: 62}
    anchors = {0: 60}
    for r, span in enumerate(spans):
        token, pitch = (guards or {}).get(r, (62, 60))
        observed[span[-1] + 1] = token
        anchors[span[-1] + 1] = pitch
    spec = MusicSpec(length=R * (L + 1) + 1, pitches=pitches, observed=observed,
                     fixed_soundings=anchors,
                     equal_onsets=tuple((spans[0][j], spans[r][j]) for r in range(1, R) for j in range(L)),
                     onset_counts=() if K is None else tuple(CountRule(span, K) for span in spans))
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=.7, size=(spec.length, 130))
    # Most probability can remain outside the working vocabulary. The oracle
    # and solver must both use the original full130 normalization unchanged.
    logits[:, :2] += 1.2
    logits[:, [p + 2 for p in pitches]] += 1.
    return spec, logits - logsumexp(logits, axis=1, keepdims=True), spans


def outside_motion(tokens, spec, spans, coefficient):
    inside = {position for span in spans for position in span}
    previous = spec.initial_pitch
    score = 0.
    for position, token in enumerate(tokens):
        if token == 0:
            previous = None
        elif token >= 2:
            pitch = token - 2
            if previous is not None and position not in inside:
                score -= coefficient * abs(pitch - previous)
            previous = pitch
    return score


def enumerate_tokens(spec, q, *, evidence=None, spans=(), boundary_motion_cost=0.):
    evidence = {int(name[1:]): token for name, token in (evidence or {}).items()}
    unknown = [i for i in range(spec.length) if i not in spec.observed]
    vocabulary = (0, 1, *(pitch + 2 for pitch in spec.pitches))
    domains = [tuple(token for token in vocabulary if np.isfinite(q[i, token])
                     and (i not in evidence or token == evidence[i])) for i in unknown]
    support = {}
    for assignment in product(*domains):
        values = dict(spec.observed)
        values.update(zip(unknown, assignment))
        tokens = tuple(values[i] for i in range(spec.length))
        checked = verify_music(tokens, spec)
        if not checked.valid or any(tokens[i] != token for i, token in evidence.items()):
            continue
        support[tokens] = (sum(float(q[i, tokens[i]]) for i in unknown) + checked.soft_score
                           + outside_motion(tokens, spec, spans, boundary_motion_cost))
    return support


def oracle_z(support):
    return float(logsumexp(list(support.values()))) if support else -math.inf


@pytest.mark.parametrize('R,L,K,pitches', [(2, 2, 0, (60, 62)), (2, 3, 1, (60, 62)),
    (2, 2, 2, (60, 62)), (3, 2, 1, (60, 62)), (4, 2, 1, (60,)), (4, 2, 1, (60, 62)),
    (2, 2, None, (60, 62)), (3, 2, None, (60,))])
def test_tiny_partition_and_token_weights_match_independent_enumeration(R, L, K, pitches):
    spec, q, spans = reset_case(R, L, K, pitches=pitches)
    engine = OnsetResetMusicInference(spec, q)
    support = enumerate_tokens(spec, q)
    assert len(support) > 1
    assert engine.log_partition() == pytest.approx(oracle_z(support), abs=2e-11)
    for tokens, weight in support.items():
        assert engine.log_weight(tokens) == pytest.approx(weight, abs=2e-11)
        assert engine.retained_soft_score(tokens) == 0.


@pytest.mark.parametrize('evidence', [{'y1': 62}, {'y2': 1}, {'y1': 62, 'y5': 1}, {'y1': 62, 'y4': 0}])
def test_temporary_clamps_preserve_original_q_and_correct_joint_mass(evidence):
    spec, q, spans = reset_case()
    engine = OnsetResetMusicInference(spec, q)
    original = oracle_z(enumerate_tokens(spec, q))
    support = enumerate_tokens(spec, q, evidence=evidence)
    expected = oracle_z(support)
    engine.log_partition()
    assert engine.log_partition(evidence) == pytest.approx(expected, abs=2e-11)
    assert engine.log_partition() == pytest.approx(original, abs=2e-11)
    if not support:
        with pytest.raises(ZeroMass):
            engine.sample_full(np.random.default_rng(12), evidence)
        return
    result = engine.sample_batch(['y2', 'y4'], np.random.default_rng(32), evidence)
    clamped = oracle_z(enumerate_tokens(spec, q, evidence={**evidence, **result.assignment}))
    assert result.log_probability == pytest.approx(clamped - expected, abs=2e-11)
    assert result.log_clamped_partition == pytest.approx(clamped, abs=2e-11)
    assert tuple(engine.sample_full(np.random.default_rng(4), evidence)) in support


def test_partial_observations_are_deltas_but_temporary_evidence_is_not():
    spec, q, spans = reset_case(R=3)
    observed = {**spec.observed, spans[0][0]: 62, spans[0][1]: 1}
    partial = replace(spec, observed=observed,
                      fixed_soundings={**spec.fixed_soundings, spans[0][0]: 60, spans[0][1]: 60})
    engine = OnsetResetMusicInference(partial, q)
    z_observed = engine.log_partition()
    assert z_observed == pytest.approx(oracle_z(enumerate_tokens(partial, q)), abs=2e-11)
    clamp = {f'y{position}': observed[position] for position in spans[0]}
    z_clamp = OnsetResetMusicInference(spec, q).log_partition(clamp)
    assert z_clamp == pytest.approx(z_observed + sum(q[i, observed[i]] for i in spans[0]), abs=2e-11)
    altered = q.copy()
    for i in observed:
        altered[i] = -np.log(130.)
    assert OnsetResetMusicInference(partial, altered).log_partition() == pytest.approx(z_observed, abs=2e-11)


@pytest.mark.parametrize('guard', [(1, 60), (0, None), (64, 62)])
def test_right_visible_hold_rest_note_and_local_sounding_rules(guard):
    spec, q, spans = reset_case(guards={0: guard})
    spec = replace(spec, pitch_classes={spans[1][0]: (0,)},
                   pitch_ranges={spans[0][1]: (60, 60)})
    support = enumerate_tokens(spec, q)
    engine = OnsetResetMusicInference(spec, q)
    assert support
    assert engine.log_partition() == pytest.approx(oracle_z(support), abs=2e-11)
    for seed in range(12):
        tokens = tuple(engine.sample_full(np.random.default_rng(seed)))
        assert tokens in support
        assert verify_music(tokens, spec).valid


def test_inherited_pitch_outside_new_onset_vocabulary_and_zero_onsets():
    # All visible HOLDs inherit MIDI55, which is not an allowed new NOTE pitch.
    spec, q, spans = reset_case(K=0, pitches=(60,), guards={0: (1, 55), 1: (1, 55)})
    spec = replace(spec, observed={i: 1 for i in spec.observed}, initial_pitch=55,
                   fixed_soundings={i: 55 for i in spec.observed}, enforce_end=True, end_pitch=55)
    support = enumerate_tokens(spec, q)
    assert len(support) == 1
    engine = OnsetResetMusicInference(spec, q)
    assert engine.log_partition() == pytest.approx(oracle_z(support), abs=2e-11)
    assert engine.sample_full(np.random.default_rng(9)) == (1,) * spec.length
    incompatible = replace(spec, fixed_soundings={**spec.fixed_soundings, spans[0][-1] + 1: 60})
    assert OnsetResetMusicInference(incompatible, q).log_partition() == -math.inf


def test_window_edge_spans_keep_initial_pitch_and_final_sounding_condition():
    spans = ((0, 1), (3, 4))
    spec = MusicSpec(length=5, pitches=(60, 62), observed={2: 62}, fixed_soundings={2: 60},
                     initial_pitch=62, enforce_end=True, end_pitch=60,
                     equal_onsets=((0, 3), (1, 4)),
                     onset_counts=tuple(CountRule(span, 1) for span in spans))
    q = np.full((5, 130), -np.log(130.))
    support = enumerate_tokens(spec, q)
    engine = OnsetResetMusicInference(spec, q)
    assert engine.log_partition() == pytest.approx(oracle_z(support), abs=2e-11)
    for seed in range(8):
        assert tuple(engine.sample_full(np.random.default_rng(seed))) in support


def test_single_total_count_and_singleton_flags_have_original_semantics():
    spec, q, spans = reset_case(R=3)
    spec = replace(spec, onset_counts=(CountRule(spans[1], 1), CountRule((spans[2][0],), 1)))
    support = enumerate_tokens(spec, q)
    engine = OnsetResetMusicInference(spec, q)
    assert engine.log_partition() == pytest.approx(oracle_z(support), abs=2e-11)
    assert all(tokens[spans[0][0]] >= 2 and tokens[spans[0][1]] < 2 for tokens in support)


@pytest.mark.parametrize('kind', ['unequal_totals', 'contradictory_singletons', 'contradictory_observations', 'zero_q', 'zero_onset_block'])
def test_contradictions_are_zero_mass_not_silent_repairs(kind):
    spec, q, spans = reset_case()
    if kind == 'unequal_totals':
        spec = replace(spec, onset_counts=(CountRule(spans[0], 1), CountRule(spans[1], 2)))
    elif kind == 'contradictory_singletons':
        spec = replace(spec, onset_counts=spec.onset_counts + (CountRule((1,), 0), CountRule((1,), 1)))
    elif kind == 'contradictory_observations':
        spec = replace(spec, observed={**spec.observed, 1: 62, 4: 0})
    elif kind == 'zero_q':
        q = np.full_like(q, -math.inf)
        q[:, 129] = 0.
    else:
        spec = replace(spec, pitch_classes={i: () for i in spans[1]})
    assert not enumerate_tokens(spec, q)
    engine = OnsetResetMusicInference(spec, q)
    assert engine.log_partition() == -math.inf
    with pytest.raises(ZeroMass):
        engine.sample_full(np.random.default_rng(5))


@pytest.mark.parametrize('K,token', [(0, 1), (2, 62)])
def test_extreme_finite_legal_mass_does_not_become_zero_through_exp_underflow(K, token):
    spec, _, spans = reset_case(R=3, K=K, pitches=(60,), guards={r: (1, 60) for r in range(3)})
    q = np.full((spec.length, 130), -math.inf)
    q[:, 129] = 0.
    q[:, token] = -1000.
    engine = OnsetResetMusicInference(spec, q)
    expected = -1000. * sum(map(len, spans))
    assert engine.log_partition() == pytest.approx(expected, abs=2e-10)
    tokens = tuple(engine.sample_full(np.random.default_rng(5)))
    assert verify_music(tokens, spec).valid
    assert engine.log_weight(tokens) == pytest.approx(expected, abs=2e-10)


def boundary_case():
    spans = ((2, 3), (6, 7))
    observed = {0: 62, 1: 64, 4: 62, 5: 64, 8: 62, 9: 64}
    spec = MusicSpec(length=10, pitches=(60, 62), observed=observed, initial_pitch=58,
                     fixed_soundings={i: token - 2 for i, token in observed.items()},
                     equal_onsets=((2, 6), (3, 7)),
                     onset_counts=tuple(CountRule(span, 1) for span in spans))
    rng = np.random.default_rng(710)
    logits = rng.normal(size=(10, 130))
    return spec, logits - logsumexp(logits, axis=1, keepdims=True), spans


@pytest.mark.parametrize('coefficient', [0., .2, 2.])
def test_boundary_proposal_keeps_external_and_right_guard_costs_exactly_once(coefficient):
    spec, q, spans = boundary_case()
    support = enumerate_tokens(spec, q, spans=spans, boundary_motion_cost=coefficient)
    engine = OnsetResetMusicInference(spec, q, boundary_motion_cost=coefficient)
    assert engine.log_partition() == pytest.approx(oracle_z(support), abs=2e-11)
    for tokens, weight in support.items():
        keep = outside_motion(tokens, spec, spans, coefficient)
        assert engine.retained_soft_score(tokens) == pytest.approx(keep, abs=2e-12)
        assert engine.log_weight(tokens) == pytest.approx(weight, abs=2e-11)
        total = verify_music(tokens, replace(spec, motion_cost=coefficient)).soft_score
        assert total <= keep + 1e-12
    evidence = {'y3': 1}
    assert engine.log_partition(evidence) == pytest.approx(oracle_z(enumerate_tokens(
        spec, q, evidence=evidence, spans=spans, boundary_motion_cost=coefficient)), abs=2e-11)


@pytest.mark.parametrize('boundary', [0., .15])
def test_full_and_projected_sample_probabilities_and_marginal_match_token_oracle(boundary):
    spec, q, spans = reset_case()
    support = enumerate_tokens(spec, q, spans=spans, boundary_motion_cost=boundary)
    z = oracle_z(support)
    engine = OnsetResetMusicInference(spec, q, boundary_motion_cost=boundary)
    unknown = [f'y{i}' for i in range(spec.length) if i not in spec.observed]
    complete = engine.sample_batch(unknown, np.random.default_rng(123))
    tokens = tuple(spec.observed[i] if i in spec.observed else complete.assignment[f'y{i}'] for i in range(spec.length))
    assert complete.log_probability == pytest.approx(support[tokens] - z, abs=2e-11)
    assert complete.log_clamped_partition == pytest.approx(support[tokens], abs=2e-11)
    partial = engine.sample_batch(['y2', 'y4'], np.random.default_rng(52))
    mass = oracle_z({t: w for t, w in support.items() if all(t[int(k[1:])] == v for k, v in partial.assignment.items())})
    assert partial.log_probability == pytest.approx(mass - z, abs=2e-11)
    assert partial.log_clamped_partition == pytest.approx(mass, abs=2e-11)
    values = engine.marginal_log_probs('y2')
    for token, actual in zip(engine.graph.domains['y2'], values):
        expected = oracle_z({t: w for t, w in support.items() if t[2] == token}) - z
        assert actual == pytest.approx(expected, abs=2e-11)
    assert engine.log_probability(partial.assignment) == pytest.approx(mass - z, abs=2e-11)


def test_actual_joint_draw_frequencies_match_nontrivial_small_support():
    spec, q, spans = reset_case(pitches=(60,))
    support = enumerate_tokens(spec, q)
    z = oracle_z(support)
    assert len(support) == 8
    engine = OnsetResetMusicInference(spec, q)
    rng = np.random.default_rng(439)
    observed = Counter(tuple(engine.sample_full(rng)) for _ in range(2200))
    assert set(observed) == set(support)
    for tokens, log_weight in support.items():
        expected = math.exp(log_weight - z)
        # Fixed seeded joint test: tolerance exceeds six binomial standard
        # errors yet rejects independent onset draws / uniform rhythm draws.
        assert abs(observed[tokens] / 2200 - expected) < .04
    assert all((t[1] >= 2) == (t[4] >= 2) for t in observed)


@pytest.mark.parametrize('change', ['motion', 'interval', 'no_relations', 'incomplete_relations', 'overlapping_count', 'unknown_separator'])
def test_out_of_scope_specs_are_explicitly_rejected(change):
    spec, q, spans = reset_case()
    if change == 'motion':
        spec = replace(spec, motion_cost=.1)
    elif change == 'interval':
        spec = replace(spec, max_adjacent_interval=12)
    elif change == 'no_relations':
        spec = replace(spec, equal_onsets=())
    elif change == 'incomplete_relations':
        spec = replace(spec, equal_onsets=((1, 4),))
    elif change == 'overlapping_count':
        spec = replace(spec, onset_counts=spec.onset_counts + (CountRule((1, 2, 4), 1),))
    else:
        spec = replace(spec, observed={k: v for k, v in spec.observed.items() if k != 3},
                       fixed_soundings={k: v for k, v in spec.fixed_soundings.items() if k != 3})
    with pytest.raises(UnsupportedSpec):
        OnsetResetMusicInference(spec, q)


@pytest.mark.parametrize('coefficient', [-.1, math.nan, math.inf, True, 1e308])
def test_boundary_coefficient_requires_finite_nonnegative_nonoverflowing_value(coefficient):
    spec, q, _ = reset_case()
    with pytest.raises(InvalidSpecification):
        OnsetResetMusicInference(spec, q, boundary_motion_cost=coefficient)


def test_budget_exhaustion_is_distinct_from_zero_mass():
    spec, q, _ = reset_case()
    with pytest.raises(BudgetExceeded):
        engine = OnsetResetMusicInference(spec, q, Budget(max_workspace_bytes=128))
        engine.log_partition()


def test_input_q_snapshot_is_immutable_and_changed_q_requires_fresh_engine():
    spec, q, _ = reset_case()
    engine = OnsetResetMusicInference(spec, q)
    original = engine.log_partition()
    q[:] = -np.log(130.)
    assert engine.log_partition() == original
    fresh = OnsetResetMusicInference(spec, q)
    assert fresh.log_partition() == pytest.approx(oracle_z(enumerate_tokens(spec, q)), abs=2e-11)
    assert abs(fresh.log_partition() - original) > 1e-3


@pytest.mark.parametrize('kind', ['shape', 'complex', 'nan', 'positive_infinity', 'unnormalized'])
def test_neural_q_requires_valid_original_full_vocabulary_probabilities(kind):
    spec, q, _ = reset_case()
    if kind == 'shape':
        q = q[:, :4]
    elif kind == 'complex':
        q = q.astype(complex)
    elif kind == 'nan':
        q[1, 129] = math.nan
    elif kind == 'positive_infinity':
        q[1, 129] = math.inf
    else:
        q[1] += .2
    with pytest.raises(InvalidSpecification):
        OnsetResetMusicInference(spec, q)
