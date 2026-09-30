import pytest

from tri.domain.music import MusicSpec
from tri.errors import InvalidSpecification
from tri.evaluation.shared_rhythm_motion import motion_parts


def example():
    spans = ((1, 2, 3, 4), (6, 7, 8, 9))
    observed = {0: 62, 3: 64, 5: 62, 8: 64, 10: 62}
    spec = MusicSpec(length=11, pitches=(60, 62, 64), observed=observed,
        fixed_soundings={i: t - 2 for i, t in observed.items()},
        equal_onsets=tuple(zip(*spans)), motion_cost=.02)
    tokens = (62, 64, 66, 64, 62, 62, 66, 62, 64, 66, 62)
    return tokens, spec, spans


def test_motion_factors_partition_exactly_and_only_initial_conditions_define_categories():
    tokens, spec, spans = example()
    result = motion_parts(tokens, spec, spans)
    assert result['jumps'] == {'outside': 4, 'fixed_predecessor': 10,
                               'observed_note': 4, 'unknown_endpoints': 6}
    assert result['original_soft_score'] == pytest.approx(-.48)
    assert result['boundary_residual_score'] == pytest.approx(-.40)
    assert sum(result['scores'].values()) == pytest.approx(-.48)


def test_rest_clears_motion_and_hold_carries_pitch():
    tokens, spec, spans = example()
    tokens = list(tokens)
    tokens[2] = 0
    tokens[7] = 1
    result = motion_parts(tuple(tokens), spec, spans)
    assert result['jumps']['unknown_endpoints'] == 0
    assert result['jumps']['observed_note'] == 2


def test_invalid_candidate_is_never_a_motion_diagnostic():
    tokens, spec, spans = example()
    with pytest.raises(InvalidSpecification):
        motion_parts((0, *tokens[1:]), spec, spans)
