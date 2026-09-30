from itertools import product
from dataclasses import replace

import numpy as np
import pytest

from tri.domain.compiler import compile_music
from tri.domain.music import (
    HOLD, MASK, PAD, REST, VOCAB_SIZE, CountRule, MusicSpec,
    note_token, token_pitch, verify_music,
)
from tri.errors import BudgetExceeded, InvalidSpecification
from tri.inference.exact import Budget, ExactInference


def uniform(length):
    return np.full((length, VOCAB_SIZE), -np.log(VOCAB_SIZE))


def sum_auxiliary_weights(graph, tokens):
    """Independent literal sum over auxiliary assignments, without VE."""
    fixed = {f"y{i}": token for i, token in enumerate(tokens)}
    if any(token not in graph.domains[name] for name, token in fixed.items()):
        return 0.0, 0
    auxiliary = tuple(name for name in graph.domains if name not in fixed)
    total, positive_paths = 0.0, 0
    indices = {name: {value: i for i, value in enumerate(domain)} for name, domain in graph.domains.items()}
    for values in product(*(graph.domains[name] for name in auxiliary)):
        assignment = {**fixed, **dict(zip(auxiliary, values))}
        weight = 0.0
        for factor in graph.factors:
            index = tuple(indices[name][assignment[name]] for name in factor.scope)
            weight += float(factor.values[index])
            if weight == -np.inf:
                break
        if np.isfinite(weight):
            total += np.exp(weight)
            positive_paths += 1
    return total, positive_paths


def test_token_roundtrip_and_input_only_symbols():
    for pitch in range(128):
        assert token_pitch(note_token(pitch)) == pitch
    assert token_pitch(REST) is None
    assert token_pitch(HOLD) is None
    for token in (MASK, PAD, -1, True):
        with pytest.raises(InvalidSpecification):
            token_pitch(token)


def test_spec_is_an_immutable_snapshot():
    observations = {0: note_token(60)}
    classes = {1: [0, 4]}
    positions = [0, 1]
    anchors = {0: 60}
    rule = CountRule(positions, 1)
    spec = MusicSpec(2, [64, 60], observed=observations, pitch_classes=classes, onset_counts=[rule], fixed_soundings=anchors)
    observations[0] = REST
    classes[1].append(7)
    positions.append(2)
    anchors[0] = None
    assert spec.observed[0] == note_token(60)
    assert spec.pitch_classes[1] == (0, 4)
    assert spec.onset_counts[0].positions == (0, 1)
    assert spec.pitches == (60, 64)
    assert spec.fixed_soundings[0] == 60
    with pytest.raises(TypeError):
        spec.observed[0] = REST


@pytest.mark.parametrize("kwargs", [
    {"length": 0}, {"length": True}, {"pitches": (60, 60)},
    {"pitches": (128,)}, {"observed": {1: MASK}},
    {"observed": {0: note_token(72)}}, {"observed": {2: REST}},
    {"initial_pitch": -1}, {"end_pitch": 60},
    {"equal_onsets": ((0, 2),)}, {"pitch_ranges": {0: (65, 60)}},
    {"pitch_classes": {0: (12,)}}, {"motion_cost": np.nan},
    {"motion_cost": -1}, {"max_adjacent_interval": -1},
    {"pitches": None}, {"observed": []}, {"equal_onsets": (1,)},
    {"pitch_ranges": {0: 60}}, {"pitch_classes": {0: "04"}},
    {"onset_counts": None}, {"motion_cost": 1e308},
    {"fixed_soundings": {0: 60}},
    {"observed": {0: HOLD}, "fixed_soundings": {0: 128}},
])
def test_invalid_spec_is_explicit(kwargs):
    with pytest.raises(InvalidSpecification):
        MusicSpec(**{"length": 2, "pitches": (60, 64), **kwargs})


def test_counts_validate_positions_and_target():
    for positions, count in [((0, 0), 1), ((0,), 2), ((-1,), 0), ((0,), True)]:
        with pytest.raises(InvalidSpecification):
            CountRule(positions, count)
    with pytest.raises(InvalidSpecification):
        MusicSpec(2, (60,), onset_counts=(CountRule((0, 2), 1),))


def test_onset_equality_does_not_tie_pitch_or_nononset_kind():
    spec = MusicSpec(2, (60, 64), equal_onsets=((0, 1),))
    assert verify_music([note_token(60), note_token(64)], spec).valid
    spec = MusicSpec(2, (60,), initial_pitch=60, equal_onsets=((0, 1),))
    assert verify_music([HOLD, REST], spec).valid
    assert not verify_music([note_token(60), HOLD], spec).valid


def test_hold_ranges_classes_and_right_boundary():
    assert not verify_music([HOLD], MusicSpec(1, (60,))).valid
    spec = MusicSpec(2, (60, 64), observed={1: HOLD}, enforce_end=True, end_pitch=60)
    assert verify_music([note_token(60), HOLD], spec).valid
    assert not verify_music([note_token(64), HOLD], spec).valid
    assert not verify_music([note_token(60), REST], spec).valid
    assert not verify_music([note_token(64), HOLD], MusicSpec(2, (60, 64), pitch_ranges={1: (60, 60)})).valid
    assert not verify_music([note_token(64), HOLD], MusicSpec(2, (60, 64), pitch_classes={1: (0,)})).valid
    assert verify_music([REST], MusicSpec(1, (), pitch_classes={0: ()}, enforce_end=True)).valid
    outside = MusicSpec(2, (60,), initial_pitch=72, observed={0: HOLD, 1: HOLD}, enforce_end=True, end_pitch=72)
    assert verify_music([HOLD, HOLD], outside).valid
    assert ExactInference(compile_music(outside, uniform(2))).log_partition() == pytest.approx(0.0)


def test_motion_is_counted_once_and_rest_breaks_it():
    spec = MusicSpec(4, (60, 64, 72), max_adjacent_interval=4, motion_cost=0.5)
    result = verify_music([note_token(60), note_token(64), REST, note_token(72)], spec)
    assert result.valid
    assert result.soft_score == -2.0
    assert not verify_music([note_token(60), HOLD, note_token(72), REST], spec).valid
    initial = MusicSpec(1, (64,), initial_pitch=60, motion_cost=0.5)
    assert verify_music([note_token(64)], initial).soft_score == -2.0


def test_internal_visible_hold_requires_a_sounding_anchor_to_preserve_audio():
    syntax_only = MusicSpec(3, (60, 64), observed={1: HOLD, 2: REST})
    assert verify_music([note_token(64), HOLD, REST], syntax_only).valid
    anchored = replace(syntax_only, fixed_soundings={1: 60, 2: None})
    assert not verify_music([note_token(64), HOLD, REST], anchored).valid
    assert verify_music([note_token(60), HOLD, REST], anchored).valid
    solver = ExactInference(compile_music(anchored, uniform(3)))
    assert solver.log_clamped_partition({"y0": note_token(64)}) == -np.inf
    assert solver.log_clamped_partition({"y0": note_token(60)}) == pytest.approx(-np.log(VOCAB_SIZE))
    # Revealing an unknown does not turn it into immutable original context.
    revealed = replace(anchored, observed={**anchored.observed, 0: note_token(60)})
    assert dict(revealed.fixed_soundings) == {1: 60, 2: None}
    assert verify_music([note_token(60), HOLD, REST], revealed).valid


def test_sounding_anchor_outside_generated_pitches_and_unreachable_anchor():
    spec = MusicSpec(1, (60,), observed={0: HOLD}, initial_pitch=72, fixed_soundings={0: 72})
    graph = compile_music(spec, uniform(1))
    assert 72 in graph.domains["h0"]
    assert ExactInference(graph).log_partition() == 0.0
    unreachable = replace(spec, initial_pitch=None)
    graph = compile_music(unreachable, uniform(1))
    assert 72 in graph.domains["h0"]
    assert ExactInference(graph).log_partition() == -np.inf
    assert not verify_music([HOLD], unreachable).valid


@pytest.mark.parametrize("spec", [
    MusicSpec(3, (60, 64), equal_onsets=((0, 2),), onset_counts=(CountRule((0, 1, 2), 2),), motion_cost=0.3),
    MusicSpec(3, (60, 64), observed={1: HOLD}, initial_pitch=64, enforce_end=True, end_pitch=60,
              pitch_classes={1: (0,)}, max_adjacent_interval=4, motion_cost=0.2),
    MusicSpec(3, (60,), initial_pitch=60, onset_counts=(CountRule((0, 2), 0), CountRule((0, 1, 2), 1))),
    MusicSpec(2, (60, 64), pitch_ranges={1: (60, 60)}, onset_counts=(CountRule((), 0),)),
    MusicSpec(3, (60, 64), observed={1: HOLD}, fixed_soundings={1: 60}, motion_cost=0.3),
])
def test_compiler_weight_identity_and_unique_auxiliaries(spec):
    rng = np.random.default_rng(821)
    q = rng.dirichlet(np.ones(VOCAB_SIZE), size=spec.length)
    graph = compile_music(spec, np.log(q))
    total = 0.0
    for tokens in product((REST, HOLD) + tuple(note_token(p) for p in spec.pitches), repeat=spec.length):
        result = verify_music(tokens, spec)
        expected = np.exp(result.soft_score) if result.valid else 0.0
        for pos, token in enumerate(tokens):
            if pos not in spec.observed:
                expected *= q[pos, token]
        actual, positive_paths = sum_auxiliary_weights(graph, tokens)
        assert actual == pytest.approx(expected, abs=1e-16, rel=1e-12)
        assert positive_paths == int(result.valid)
        total += expected
    assert np.exp(ExactInference(graph).log_partition()) == pytest.approx(total, abs=1e-16, rel=1e-12)


def test_cropped_vocab_keeps_mass_and_temporary_clamp_keeps_q():
    graph = compile_music(MusicSpec(1, (60,)), uniform(1))
    solver = ExactInference(graph)
    assert solver.log_partition() == pytest.approx(np.log(2 / VOCAB_SIZE))
    assert solver.log_clamped_partition({"y0": note_token(60)}) == pytest.approx(-np.log(VOCAB_SIZE))
    observed = MusicSpec(1, (60,), observed={0: note_token(60)})
    assert ExactInference(compile_music(observed, np.full((1, VOCAB_SIZE), np.nan))).log_partition() == 0.0


def test_invalid_hold_and_model_zero_mass_are_not_repaired():
    impossible = MusicSpec(1, (60,), observed={0: HOLD})
    assert ExactInference(compile_music(impossible, uniform(1))).log_partition() == -np.inf
    log_q = np.full((1, VOCAB_SIZE), -np.inf)
    log_q[0, note_token(127)] = 0.0
    assert ExactInference(compile_music(MusicSpec(1, (60,)), log_q)).log_partition() == -np.inf


@pytest.mark.parametrize("kind", ["shape", "normalization", "nan", "positive_inf", "complex"])
def test_model_rows_are_validated(kind):
    q = uniform(2)
    if kind == "shape":
        q = q[:, :3]
    elif kind == "normalization":
        q[0] += 0.1
    elif kind == "nan":
        q[1, 0] = np.nan
    elif kind == "positive_inf":
        q[1, 0] = np.inf
    else:
        q = q.astype(complex) + 1j
    with pytest.raises(InvalidSpecification):
        compile_music(MusicSpec(2, (60,)), q)


def test_count_compilation_uses_narrow_chain_and_respects_budget():
    spec = MusicSpec(40, (60,), onset_counts=(CountRule(tuple(range(40)), 20),))
    graph = compile_music(spec, uniform(spec.length))
    assert max(len(factor.scope) for factor in graph.factors) <= 3
    assert max(factor.values.size for factor in graph.factors) < 2_000
    with pytest.raises(BudgetExceeded):
        compile_music(spec, uniform(spec.length), Budget(max_factor_entries=10))
    with pytest.raises(BudgetExceeded):
        compile_music(spec, uniform(spec.length), Budget(max_workspace_bytes=100))


def test_observed_stretches_have_singleton_state_domains():
    spec = MusicSpec(5, (60, 64), observed={0: note_token(60), 1: HOLD, 2: REST, 3: note_token(64), 4: HOLD})
    graph = compile_music(spec, uniform(spec.length))
    assert all(len(graph.domains[f"h{i}"]) == 1 for i in range(spec.length))
    assert ExactInference(graph).log_partition() == 0.0


def test_mixed_requests_match_independent_original_sequence_enumeration():
    """Mix interacting rules, including inconsistent requests and boundaries."""
    rng = np.random.default_rng(512)
    vocabulary = (REST, HOLD, note_token(60), note_token(64))
    for _ in range(30):
        observed = {i: int(rng.choice(vocabulary)) for i in range(4) if rng.random() < 0.3}
        spec = MusicSpec(
            4, (60, 64), observed=observed,
            initial_pitch=rng.choice([None, 60, 72]),
            enforce_end=True, end_pitch=rng.choice([None, 60, 64, 72]),
            equal_onsets=((0, 2), (1, 3)),
            onset_counts=(CountRule((0, 1, 2, 3), int(rng.integers(5))),),
            pitch_classes={int(rng.integers(4)): (0,)},
            pitch_ranges={int(rng.integers(4)): (60, 64)},
            max_adjacent_interval=int(rng.choice([0, 4, 12])), motion_cost=0.27,
        )
        q = rng.dirichlet(np.ones(VOCAB_SIZE), size=4)
        graph = compile_music(spec, np.log(q))
        weights = {}
        for tokens in product(vocabulary, repeat=4):
            result = verify_music(tokens, spec)
            if result.valid:
                weights[tokens] = np.exp(result.soft_score) * np.prod([q[i, token] for i, token in enumerate(tokens) if i not in observed])
        total = sum(weights.values())
        solver = ExactInference(graph)
        assert np.exp(solver.log_partition()) == pytest.approx(total, abs=1e-16, rel=1e-11)
        if total:
            for token_index, token in enumerate(graph.domains["y0"]):
                expected = sum(w for y, w in weights.items() if y[0] == token) / total
                actual = np.exp(solver.marginal_log_probs("y0")[token_index])
                assert actual == pytest.approx(expected, abs=1e-12)
