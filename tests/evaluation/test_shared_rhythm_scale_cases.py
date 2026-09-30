from dataclasses import replace
import math

import numpy as np
import pytest

from tri.domain.music import CountRule, MusicSpec, verify_music
from tri.evaluation import shared_rhythm_scale_cases as scale
from tri.evaluation.solver_cases import SolverCase, _deserialize_spec, _serialize_spec
from tri.inference.music_backends import make_music_engine


def test_paired_scale_manifest_retains_all_requests_original_q_and_model_accounting(tmp_path):
    from tri.evaluation.solver_cases import save_cases, load_case
    source = tmp_path / 'learned'
    plan = scale.select_long_requests(windows())
    cases = [SolverCase(row['case_id'], _deserialize_spec(row['spec']),
                        np.full((256, 130), -math.log(130)),
                        {**row['metadata'], 'model_calls': 1, 'probability_provider_seconds': .01},
                        tuple(row['witness'])) for row in plan['requests']]
    original = save_cases(cases, source, policy={'scope': 'independent test runtime'})
    target = tmp_path / 'paired'
    paired = scale.prepare_long_scale_cases(source / 'cases.json', target)
    assert paired == scale.prepare_long_scale_cases(source / 'cases.json', target)
    assert paired['count'] == 16 and paired['policy']['unique_model_calls'] == 8
    assert sum(entry['metadata']['model_calls'] for entry in paired['cases']) == 8
    for index, original_entry in enumerate(original['cases']):
        narrow = load_case(paired['cases'][index * 2]['path'])
        wide = load_case(paired['cases'][index * 2 + 1]['path'])
        before = load_case(original_entry['path'])
        assert narrow.spec == before.spec and set(narrow.spec.pitches) < set(wide.spec.pitches)
        np.testing.assert_array_equal(narrow.logq, wide.logq)
        np.testing.assert_array_equal(narrow.logq, before.logq)
        assert narrow.metadata['q_pairing_key'] == wide.metadata['q_pairing_key'] == before.case_id


def windows(count=12):
    return [{'source_index': i, 'work_id': f'{i:03d}', 'source_start_cell': i * 256,
             'tokens': tuple([62, 1, 1, 1] * 64), 'initial_pitch': None} for i in range(count)]


@pytest.mark.parametrize('R,starts', [(3, [0, 96, 192]), (4, [0, 64, 128, 192])])
def test_long_gaps_are_two_bars_with_declared_visible_guards(R, starts):
    spans = scale.long_spans(R)
    assert [span[0] for span in spans] == starts
    assert all(len(span) == 32 for span in spans)
    assert all(b[0] - a[-1] >= 33 for a, b in zip(spans, spans[1:]))
    assert spans[-1][-1] == 223


@pytest.mark.parametrize('R,visibility', [(3, 'unknown'), (3, 'partial'), (4, 'unknown'), (4, 'partial')])
def test_long_target_uses_only_visible_projection_fixed_k_and_full_token_semantics(R, visibility):
    row = windows(1)[0]
    visible = scale.project_long_visible(row['tokens'], None, R, visibility)
    hidden_changed = [t if p in visible['observed'] else 0 for p, t in enumerate(row['tokens'])]
    projected = scale.project_long_visible(hidden_changed, None, R, visibility)
    assert projected == visible  # supplied visible soundings also remain identical
    spec, metadata, reason = scale.long_request_from_visible(visible, R, visibility)
    spec2, metadata2, reason2 = scale.long_request_from_visible(projected, R, visibility)
    assert _serialize_spec(spec) == _serialize_spec(spec2) and metadata == metadata2
    assert reason == reason2 is None
    assert metadata['L'] == 32 and metadata['K'] == 8
    assert metadata['visible_fraction_first_span'] == (.25 if visibility == 'partial' else 0.)
    assert metadata['observed_positions_in_first_span'] == (8 if visibility == 'partial' else 0)
    assert metadata['visible_onsets_in_first_span'] == (2 if visibility == 'partial' else 0)
    assert metadata['onset_templates_consistent_with_observations_and_count'] == (math.comb(24, 6) if visibility == 'partial' else math.comb(32, 8))
    witness, _ = scale.real._uniform_witness(spec, 15)
    assert len(witness) == 256 and verify_music(witness, spec).valid


def test_partial_exposing_all_eight_onsets_is_not_accepted_as_partial():
    tokens = list(windows(1)[0]['tokens'])
    tokens[:8] = [62] * 8
    spec, metadata, reason = scale.long_request_from_visible(scale.project_long_visible(tokens, None, 3, 'partial'), 3, 'partial')
    assert reason == 'partial_degenerated_to_known_template'
    assert metadata['remaining_onsets_in_first_span'] == 0
    assert metadata['onset_templates_consistent_with_observations_and_count'] == 1


def test_eight_long_requests_are_balanced_reproducible_and_diverse():
    first = scale.select_long_requests(windows())
    assert first == scale.select_long_requests(list(reversed(windows())))
    assert first['status'] == 'completed'
    assert first['selected_count'] == first['selected_work_count'] == 8
    assert first['selected_by_R_visibility'] == {'R3_unknown': 2, 'R3_partial': 2, 'R4_unknown': 2, 'R4_partial': 2}
    assert first['screened_status_counts'] == {'feasible': 8}
    assert first['possible_R_visibility_candidates'] == 48
    assert first['screened_candidate_count'] == 8
    for row in first['requests']:
        assert row['metadata']['scale_variant'] == 'long_L32'
        assert verify_music(row['witness'], _deserialize_spec(row['spec'])).valid


def test_long_audit_preserves_infeasible_candidates_and_work_caps(monkeypatch):
    def no_support(*args):
        raise scale.ZeroMass('public boundary mismatch')
    monkeypatch.setattr(scale.real, '_uniform_witness', no_support)
    plan = scale.select_long_requests(windows(2))
    assert plan['status'] == 'insufficient_eligible_requests'
    assert plan['screened_status_counts'] == {'infeasible_boundary_or_visible_context': 8}
    assert all('public boundary' in row['reason'] for row in plan['candidate_audit'])


def test_long_preparation_is_cpu_only_immutable_and_keeps_model_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(scale.real, '_provenance', lambda _: {'mock': 'immutable'})
    monkeypatch.setattr(scale.real, '_validation_windows', lambda _: windows())
    monkeypatch.setattr(scale.real, '_cuda_runtime', lambda *args: pytest.fail('No model allowed'))
    plan = scale.prepare_long_requests(tmp_path)
    assert plan['model_calls'] == 0 and plan['selected_count'] == 8
    assert not (tmp_path / 'cases.json').exists()
    assert 'not a strict single-variable' in plan['policy']['scope']
    assert scale.prepare_long_requests(tmp_path) == plan
    with pytest.raises(ValueError, match='differs'):
        scale.prepare_long_requests(tmp_path, options={'seed': 20260952})


def learned_case():
    spec = MusicSpec(length=3, pitches=(60, 110), observed={0: 112, 2: 1},
                     initial_pitch=105, fixed_soundings={0: 110, 2: 110},
                     onset_counts=(CountRule((1,), 0),), motion_cost=.02)
    logits = np.tile(np.linspace(-2, 3, 130), (3, 1))
    logq = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return SolverCase('learned_source', spec, logq,
        {'source_type': 'learned', 'D': 4, 'family': scale.real.FAMILY,
         'model_calls': 1, 'probability_provider_seconds': .015,
         'provider_preprocessing_seconds': .002, 'checkpoint': {'path': 'mock.pt'}}, (112, 1, 1))


def test_wide_only_changes_pitch_support_and_preserves_original_full_q_and_public_boundaries():
    source = learned_case()
    before = _serialize_spec(source.spec)
    wide = scale.make_wide_pitch_case(source)
    assert wide.spec.pitches == tuple(sorted(set(source.spec.pitches) | set(range(36, 97))))
    after = _serialize_spec(wide.spec)
    after['pitches'] = before['pitches']
    assert after == before
    assert wide.spec.initial_pitch == 105 and wide.spec.fixed_soundings[2] == 110
    assert 110 in wide.spec.pitches  # preserve original pitches above the requested wide interval
    np.testing.assert_array_equal(wide.logq, source.logq)
    np.testing.assert_allclose(np.exp(wide.logq).sum(axis=1), 1., atol=1e-13)
    assert wide.witness == source.witness and verify_music(wide.witness, wide.spec).valid
    assert wide.metadata['D'] == 64  # 61 widened pitches, original110, carried105, silence
    assert wide.metadata['source_case_id'] == source.case_id
    assert wide.metadata['q_pairing_key'] == source.case_id
    assert wide.metadata['model_calls'] == wide.metadata['additional_model_calls'] == 0
    assert wide.metadata['source_model_calls'] == 1
    assert wide.metadata['probability_provider_seconds'] == .015  # inherited, not a second call
    assert wide.metadata['additional_probability_provider_seconds'] == 0.
    assert 'target support/distribution changes' in wide.metadata['scale_comparison']
    wide.metadata['checkpoint']['path'] = 'only_copy.pt'
    wide.logq[0, 0] = -99
    assert source.metadata['checkpoint']['path'] == 'mock.pt' and source.logq[0, 0] != -99


def test_wider_target_has_different_partition_without_any_q_renormalization():
    spec = MusicSpec(length=1, pitches=(60, 62), onset_counts=(CountRule((0,), 1),))
    source = SolverCase('one_note', spec, np.full((1, 130), -math.log(130)), {'source_type': 'learned'}, (62,))
    wide = scale.make_wide_pitch_case(source, low=60, high=64)
    z = make_music_engine(source.spec, source.logq, backend='template_stream').log_partition()
    wide_z = make_music_engine(wide.spec, wide.logq, backend='template_stream').log_partition()
    assert z == pytest.approx(math.log(2 / 130))
    assert wide_z == pytest.approx(math.log(5 / 130))
    assert wide_z > z
    np.testing.assert_array_equal(source.logq, wide.logq)


@pytest.mark.parametrize('low,high', [(True, 96), (36, 128), (96, 36)])
def test_invalid_wide_ranges_are_rejected(low, high):
    with pytest.raises(ValueError, match='integer MIDI bounds'):
        scale.make_wide_pitch_case(learned_case(), low=low, high=high)


def test_wide_requires_learned_source_and_an_actual_expansion():
    source = learned_case()
    with pytest.raises(ValueError, match='does not expand'):
        scale.make_wide_pitch_case(source, low=60, high=60)
    with pytest.raises(ValueError, match='learned-probability'):
        scale.make_wide_pitch_case(replace(source, metadata={'source_type': 'synthetic'}))
