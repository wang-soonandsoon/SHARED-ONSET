from dataclasses import replace
import json
import math

import numpy as np
import pytest

from tri.domain.music import verify_music
from tri.evaluation import shared_rhythm_real_cases as real
from tri.evaluation.solver_cases import _identity, _serialize_spec, load_case
from tri.runtime import atomic_json


def options(**overrides):
    return {**real.DEFAULT_OPTIONS, 'quotas': {'3': 2, '4': 2},
            'minimum_distinct_works': 2, 'pitch_low': 60, 'pitch_high': 62, **overrides}


def window(index, *, work=None):
    return {'source_index': index, 'work_id': work or f'{index:03d}',
            'source_start_cell': 256 * index, 'tokens': tuple([62, 1, 1, 1] * 64),
            'initial_pitch': None}


def source_fixture(tmp_path, *, splits=None):
    revision = tmp_path / 'revision'
    stage = revision / 'bar16'
    stage.mkdir(parents=True)
    rows = [window(i) for i in range(8)]
    keys = {'work_ids': np.array([w['work_id'] for w in rows]),
            'start_cells': np.array([w['source_start_cell'] for w in rows]),
            'splits': np.array(splits or ['validation'] * 8)}
    dataset, chords, checkpoint = tmp_path / 'windows.npz', tmp_path / 'chords.npz', stage / 'best.pt'
    np.savez(dataset, **keys, tokens=np.array([w['tokens'] for w in rows]), initial_pitches=np.full(8, -1))
    features = np.zeros((8, 256, 38))
    for i in range(8):
        features[i, :, 0] = i
    np.savez(chords, **keys, chord_features=features)
    checkpoint.write_bytes(b'CPU mock checkpoint, never loaded by torch')
    atomic_json(stage / 'cohort.json', {'status': 'completed', 'options': {
        'split': 'validation', 'dataset': _identity(dataset), 'chords': _identity(chords)}})
    atomic_json(stage / 'evaluation/config.json', {'checkpoint': _identity(checkpoint), 'checkpoint_seed': 7})
    return revision


@pytest.mark.parametrize('R,starts', [(3, [0, 112, 224]), (4, [0, 64, 144, 224]),
                                    (6, [0, 32, 80, 128, 176, 224]),
                                    (8, [0, 32, 64, 96, 128, 160, 192, 224])])
def test_bar_aligned_positions_are_fixed_and_have_visible_separators(R, starts):
    spans = real.fixed_spans(R)
    assert [s[0] for s in spans] == starts
    assert all(len(s) == 16 for s in spans)
    assert all(b[0] - a[-1] >= 17 for a, b in zip(spans, spans[1:]))
    assert spans[-1][-1] == 239


@pytest.mark.parametrize('visibility', ['unknown', 'partial'])
def test_hidden_melody_changes_do_not_change_request_when_supplied_visible_anchors_are_same(visibility):
    tokens = list(window(0)['tokens'])
    visible = real.project_visible(tokens, None, 4, visibility, options=options())
    changed = [t if i in visible['observed'] else 0 for i, t in enumerate(tokens)]
    projected = real.project_visible(changed, None, 4, visibility, options=options())
    assert projected == visible  # each outside bar begins with a visible NOTE
    spec, metadata, reason = real.request_from_visible(visible, 4, visibility, options=options())
    other, metadata2, reason2 = real.request_from_visible(projected, 4, visibility, options=options())
    assert _serialize_spec(spec) == _serialize_spec(other)
    assert metadata == metadata2 and reason == reason2 is None
    assert metadata['K'] == 4 and metadata['beta'] == .02
    assert metadata['visible_onsets_in_first_span'] == (1 if visibility == 'partial' else 0)
    assert metadata['onset_templates_consistent_with_observations_and_count'] == (220 if visibility == 'partial' else 1820)


def test_partial_all_four_onsets_is_retained_as_degenerate_not_selected():
    tokens = list(window(0)['tokens'])
    tokens[:4] = [62] * 4
    visible = real.project_visible(tokens, None, 3, 'partial', options=options())
    _, metadata, reason = real.request_from_visible(visible, 3, 'partial', options=options())
    assert reason == 'partial_degenerated_to_known_template'
    assert metadata['remaining_onsets_in_first_span'] == 0
    plan = real.select_requests([{**window(i), 'tokens': tuple(tokens)} for i in range(8)], options=options())
    assert plan['status'] == 'insufficient_eligible_requests'
    assert plan['screened_status_counts']['partial_degenerated_to_known_template'] > 0
    assert all(r['visibility'] == 'unknown' for r in plan['requests'])


def test_visible_flags_that_exceed_count_are_infeasible_and_not_silently_repaired():
    visible = real.project_visible([62] * 256, None, 3, 'partial', options=options(K=3))
    _, metadata, reason = real.request_from_visible(visible, 3, 'partial', options=options(K=3))
    assert reason == 'infeasible_visible_onset_flags'
    assert metadata['onset_templates_consistent_with_observations_and_count'] == 0


def test_selection_is_deterministic_balanced_work_diverse_and_uses_full_token_witnesses():
    rows = [window(i, work=f'{i // 2:03d}') for i in range(8)]
    first = real.select_requests(rows, options=options())
    second = real.select_requests(list(reversed(rows)), options=options())
    assert first == second
    assert first['status'] == 'completed' and first['selected_count'] == 4
    assert first['selected_work_count'] == 4
    assert first['selected_by_R_visibility'] == {'R3_unknown': 1, 'R3_partial': 1, 'R4_unknown': 1, 'R4_partial': 1}
    assert len({r['source_index'] for r in first['requests']}) == 4
    assert sum(first['screened_status_counts'].values()) == first['screened_candidate_count']
    assert first['possible_R_visibility_candidates'] == 32
    for row in first['requests']:
        assert len(row['witness']) == 256
        assert verify_music(row['witness'], real._deserialize_spec(row['spec'])).valid


def test_work_cap_and_insufficient_pool_are_preserved():
    rows = [window(i, work='one_work') for i in range(8)]
    plan = real.select_requests(rows, options=options(max_per_work=2))
    assert plan['status'] == 'insufficient_eligible_requests'
    assert plan['selected_by_work'] == {'one_work': 2}


def test_boundary_infeasibility_is_audited_independently_of_visible_activity(monkeypatch):
    def no_support(spec, seed):
        raise real.ZeroMass('fixed HOLD cannot be connected')
    monkeypatch.setattr(real, '_uniform_witness', no_support)
    plan = real.select_requests([window(0), window(1)], options=options())
    assert plan['selected_count'] == 0
    assert plan['screened_status_counts'] == {'infeasible_boundary_or_visible_context': 8}
    assert all('fixed HOLD' in r['reason'] for r in plan['candidate_audit'])


def test_prepare_is_validation_only_idempotent_and_does_not_load_model(tmp_path, monkeypatch):
    revision = source_fixture(tmp_path, splits=['train'] * 4 + ['validation'] * 4)
    monkeypatch.setattr(real, '_cuda_runtime', lambda *a: pytest.fail('Preparation must not load model'))
    out = tmp_path / 'inputs'
    plan = real.prepare_real_requests(out, revision_root=revision, options=options())
    assert plan['model_calls'] == 0
    assert all(row['source_index'] >= 4 for row in plan['requests'])
    assert plan == real.prepare_real_requests(out, revision_root=revision, options=options())
    assert not (out / 'cases.json').exists()
    assert 'historical' in plan['policy']['source_preprocessing'].lower()
    with pytest.raises(ValueError, match='differs'):
        real.prepare_real_requests(out, revision_root=revision, options=options(K=3))


def test_test_containing_materialization_is_refused_before_token_access(tmp_path):
    revision = source_fixture(tmp_path, splits=['test'] + ['validation'] * 7)
    with pytest.raises(ValueError, match='no test reads'):
        real.prepare_real_requests(tmp_path / 'inputs', revision_root=revision, options=options())


def mock_runtime(log, *, bad=False):
    def factory(checkpoint, device):
        log.append(('load', checkpoint, device))
        def sync():
            log.append(('sync',))
        def make_provider(spec, condition):
            log.append(('prepare', spec, condition.copy()))
            def provider(state, noise):
                log.append(('call', state, noise))
                logits = np.tile(np.linspace(-3, 2, 130), (256, 1))
                q = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
                if bad:
                    q[:, 129] = np.nan
                return q
            return provider
        return real.ProbabilityRuntime(make_provider, sync, {'length': 256, 'condition_dim': 38}, {'kind': 'CPU_test_mock'})
    return factory


def test_freeze_calls_provider_once_per_request_preserves_full_q_and_resumes_without_runtime(tmp_path):
    revision, out, log = source_fixture(tmp_path), tmp_path / 'inputs', []
    plan = real.prepare_real_requests(out, revision_root=revision, options=options())
    manifest = real.freeze_real_probabilities(out, device='cpu_mock', runtime_factory=mock_runtime(log))
    assert manifest['count'] == manifest['model_calls'] == 4
    assert len([e for e in log if e[0] == 'load']) == 1
    assert len([e for e in log if e[0] == 'call']) == 4
    calls = [e for e in log if e[0] == 'call']
    for row, entry, call in zip(plan['requests'], manifest['cases'], calls):
        case = load_case(entry['path'])
        assert case.logq.shape == (256, 130)
        assert np.isfinite(case.logq[:, 129]).all()  # outside allowed pitch set still retained
        np.testing.assert_allclose(np.exp(case.logq).sum(axis=1), 1., atol=1e-13)
        assert call[1] == tuple(case.spec.observed.get(i) for i in range(256))
        assert call[2] == 1.
        assert verify_music(case.witness, case.spec).valid
        assert case.metadata['model_calls'] == 1 and case.metadata['probability_provider_seconds'] > 0
        assert case.metadata['provider_preprocessing_seconds'] > 0
        assert entry['metadata']['family'] == real.FAMILY
    assert manifest['timing']['sessions'][0]['model_calls_completed'] == 4
    assert real.freeze_real_probabilities(out, runtime_factory=lambda *a: pytest.fail('No model reload')) == manifest
    # Synchronization brackets every actual call without consuming random draws.
    for i, event in enumerate(log):
        if event[0] == 'call':
            assert log[i - 1][0] == log[i + 1][0] == 'sync'


def test_invalid_probability_failure_is_preserved_and_never_retried_or_replaced(tmp_path):
    revision, out, log = source_fixture(tmp_path), tmp_path / 'inputs', []
    plan = real.prepare_real_requests(out, revision_root=revision, options=options())
    with pytest.raises(ValueError, match='finite'):
        real.freeze_real_probabilities(out, device='cpu_mock', runtime_factory=mock_runtime(log, bad=True))
    status = json.loads((out / 'freeze_status.json').read_text())
    attempt = status['attempts'][plan['requests'][0]['case_id']]
    assert status['status'] == attempt['status'] == 'failed'
    assert attempt['model_calls_started'] == attempt['model_calls_completed'] == 1
    assert not (out / 'cases.json').exists()
    with pytest.raises(ValueError, match='cannot be repeated implicitly'):
        real.freeze_real_probabilities(out, runtime_factory=lambda *a: pytest.fail('No retry'))
    assert len([e for e in log if e[0] == 'call']) == 1


@pytest.mark.parametrize('custom_policy', [False, True])
def test_freeze_inherits_declared_cohort_policy_and_actual_entry_families(tmp_path, custom_policy):
    revision, out = source_fixture(tmp_path), tmp_path / 'inputs'
    plan = real.prepare_real_requests(out, revision_root=revision, options=options())
    if custom_policy:
        plan['policy'] = {'q': 'Retain the declared full130 objective.',
                          'scope': 'Alternate fixed validation mask layout.',
                          'layout_version': 'alternate_v1'}
    else:
        plan.pop('policy')  # Compatibility with a cohort lacking explicit policy.
    for i, row in enumerate(plan['requests']):
        row['metadata']['family'] = 'alternate_a' if i < 3 else 'alternate_b'
    atomic_json(out / 'cohort.json', plan)
    manifest = real.freeze_real_probabilities(out, device='cpu_mock', runtime_factory=mock_runtime([]))
    expected_policy = {**real.POLICY, **plan.get('policy', {})}
    assert manifest['policy'] == expected_policy
    assert manifest['families'] == {'alternate_a': 3, 'alternate_b': 1}
    for entry in manifest['cases']:
        case = load_case(entry['path'])
        assert case.metadata['q_policy'] == expected_policy['q']
        assert case.metadata['scope'] == expected_policy['scope']


def test_checkpoint_changed_after_selection_blocks_provider_call(tmp_path):
    revision, out = source_fixture(tmp_path), tmp_path / 'inputs'
    plan = real.prepare_real_requests(out, revision_root=revision, options=options())
    with open(plan['provenance']['checkpoint']['path'], 'ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='checkpoint identity changed'):
        real.freeze_real_probabilities(out, runtime_factory=lambda *a: pytest.fail('Must validate first'))


def test_frozen_visible_projection_rejects_unexpected_hidden_token_access():
    visible = real.project_visible(window(0)['tokens'], None, 3, 'unknown', options=options())
    visible['observed'][0] = 62
    with pytest.raises(ValueError, match='exactly'):
        real.request_from_visible(visible, 3, 'unknown', options=options())
