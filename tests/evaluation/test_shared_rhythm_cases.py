from dataclasses import replace
from itertools import product
import json

import numpy as np
import pytest

from tri.domain.music import verify_music
from tri.evaluation.shared_rhythm_cases import make_shared_rhythm_case, build_shared_rhythm_cases, prepare_shared_rhythm_cases
from tri.evaluation.solver_cases import load_case


def small_grid():
    return {'R': [2, 3], 'L': 4, 'D': 3, 'K': 1, 'betas': [0., .02],
            'visibility': ['unknown', 'partial', 'known'], 'q_seeds': [17, 18]}


@pytest.mark.parametrize('R,beta,visibility', product([2, 3, 4, 6, 8], [0., .02], ['unknown', 'partial', 'known']))
def test_declared_pilot_fixtures_preserve_music_and_shared_rhythm(R, beta, visibility):
    case = make_shared_rhythm_case(R=R, beta=beta, visibility=visibility)
    assert case.metadata['family'] == 'synthetic_control'
    assert case.spec.motion_cost == beta
    assert verify_music(case.witness, case.spec).valid
    assert case.logq.shape == (R * 17 + 1, 130)
    assert np.isfinite(case.logq).all()
    np.testing.assert_allclose(np.exp(case.logq).sum(axis=1), 1., atol=1e-13)
    assert len(case.spec.pitches) == 15
    spans = case.metadata['spans']
    assert len(spans) == R
    assert len(case.spec.equal_onsets) == 16 * (R - 1)
    assert set(case.spec.fixed_soundings) == set(case.spec.observed)
    expected_templates = {'unknown': 1820, 'partial': 220, 'known': 1}[visibility]
    assert case.metadata['onset_templates_consistent_with_observations_and_count'] == expected_templates
    assert sum(case.witness[p] >= 2 for p in spans[0]) == 4


def test_beta_and_visibility_keep_identical_full_q_but_distinct_targets():
    base = make_shared_rhythm_case(seed=35)
    for beta, visibility in product([0., .02], ['unknown', 'partial', 'known']):
        case = make_shared_rhythm_case(seed=35, beta=beta, visibility=visibility)
        np.testing.assert_array_equal(case.logq, base.logq)
    other = make_shared_rhythm_case(seed=36)
    assert not np.array_equal(other.logq, base.logq)


def test_known_template_keeps_other_spans_unknown_and_partial_is_not_known():
    known = make_shared_rhythm_case(R=4, visibility='known')
    partial = make_shared_rhythm_case(R=4, visibility='partial')
    assert known.metadata['unknown_positions'] == 3 * 16
    assert partial.metadata['unknown_positions'] == 4 * 16 - 4
    assert partial.metadata['onset_templates_consistent_with_observations_and_count'] > 1


def test_preparation_is_reusable_and_rejects_different_config_or_q(tmp_path):
    grid = small_grid()
    manifest = prepare_shared_rhythm_cases(grid, tmp_path)
    assert manifest['count'] == 24
    assert prepare_shared_rhythm_cases(grid, tmp_path) == manifest
    for entry in manifest['cases']:
        loaded = load_case(entry['path'])
        assert loaded.metadata['R'] in [2, 3]
        assert verify_music(loaded.witness, loaded.spec).valid
    with pytest.raises(ValueError, match='different preparation'):
        prepare_shared_rhythm_cases({**grid, 'L': 8}, tmp_path)


@pytest.mark.parametrize('kwargs', [{'R': 1}, {'L': 0}, {'D': 1}, {'K': 17}, {'beta': -.1}, {'visibility': 'fake'}, {'logit_scale': 0}])
def test_bad_fixture_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        make_shared_rhythm_case(**kwargs)


def test_grid_deduplicates_nothing_silently():
    with pytest.raises(ValueError, match='distinct'):
        build_shared_rhythm_cases({**small_grid(), 'q_seeds': [17, 17]})
