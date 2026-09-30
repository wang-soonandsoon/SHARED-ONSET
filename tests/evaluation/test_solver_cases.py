from dataclasses import replace
from itertools import product
import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import logsumexp

from tri.domain.music import verify_music
from tri.evaluation.solver_cases import (
    build_controlled_cases, build_tiny_cases, load_case, make_controlled_case,
    prepare_solver_cases, save_cases, _deserialize_spec, _serialize_spec,
)
from tri.inference.music_backends import make_music_engine


def brute(case, evidence=None):
    spec = case.spec
    missing = [i for i in range(spec.length) if i not in spec.observed]
    weights = []
    for assignment in product((0, 1) + tuple(p + 2 for p in spec.pitches), repeat=len(missing)):
        tokens = [spec.observed.get(i) for i in range(spec.length)]
        for pos, token in zip(missing, assignment):
            tokens[pos] = token
        if any(tokens[int(key[1:])] != val for key, val in (evidence or {}).items()):
            continue
        checked = verify_music(tokens, spec)
        if checked.valid:
            weights.append(sum(case.logq[i, tokens[i]] for i in missing) + checked.soft_score)
    return float(logsumexp(weights))


@pytest.mark.parametrize('case', build_tiny_cases(), ids=lambda case: case.case_id)
def test_tiny_support_and_clamps_match_original_token_oracle(case):
    expected = brute(case)
    assert np.isfinite(expected) == (case.metadata['expected_support'] == 'feasible')
    engine = make_music_engine(case.spec, case.logq, backend='paired')
    assert engine.log_partition() == pytest.approx(expected, abs=1e-11)
    evidence = case.metadata['query_evidence']
    assert engine.log_clamped_partition(evidence) == pytest.approx(brute(case, evidence), abs=1e-11)


def test_controlled_sweeps_are_deduplicated_reproducible_and_witnessed():
    cases = build_controlled_cases()
    assert len({case.case_id for case in cases}) == len(cases)
    assert len(cases) == 57
    assert {case.metadata['seed'] for case in cases} == {20260921, 20260922, 20260923}
    assert {case.metadata['D'] for case in cases if 'pitch_states' in case.metadata['sweeps']} == {8, 16, 32, 64}
    for case in cases:
        assert np.allclose(np.exp(case.logq).sum(axis=1), 1., atol=1e-12)
        if case.metadata['expected_support'] == 'feasible':
            assert verify_music(case.witness, case.spec).valid
            assert all(case.witness[int(p[1:])] == v for p, v in case.metadata['query_evidence'].items())
        assert case.metadata['unknown_positions'] > 0
    a = make_controlled_case(D=8)
    b = make_controlled_case(D=64)
    assert np.array_equal(a.logq, b.logq)  # No crop-dependent renormalization.
    assert np.exp(a.logq[1, [0, 1] + [p + 2 for p in a.spec.pitches]]).sum() < 1


def test_json_spec_schema_roundtrip_matches_existing_cohorts():
    from tri.evaluation.cohorts import serialize_spec, deserialize_spec
    for case in build_tiny_cases():
        assert _serialize_spec(case.spec) == serialize_spec(case.spec)
        assert _deserialize_spec(serialize_spec(case.spec)) == case.spec
        assert deserialize_spec(_serialize_spec(case.spec)) == case.spec


def test_save_load_is_immutable_and_resume_does_not_rebuild(tmp_path, monkeypatch):
    import tri.evaluation.solver_cases as module
    manifest = prepare_solver_cases(tmp_path, include_learned=False, seeds=(7,))
    before = (tmp_path / 'cases.json').stat().st_mtime_ns
    case = load_case(manifest['cases'][0]['path'])
    assert case.metadata == manifest['cases'][0]['metadata']

    def fail(*args, **kwargs):
        raise AssertionError('a frozen snapshot must not be rebuilt')

    monkeypatch.setattr(module, 'build_controlled_cases', fail)
    assert prepare_solver_cases(tmp_path, include_learned=False, seeds=(7,)) == manifest
    assert (tmp_path / 'cases.json').stat().st_mtime_ns == before
    with pytest.raises(ValueError, match='options changed'):
        prepare_solver_cases(tmp_path, include_learned=False, seeds=(8,))
    altered = case.logq.copy()
    altered[:, [0, 1]] = altered[:, [1, 0]]
    with pytest.raises(ValueError, match='different content'):
        save_cases([replace(case, logq=altered)], tmp_path)


def test_case_loading_does_not_import_torch(tmp_path):
    manifest = save_cases(build_tiny_cases()[:1], tmp_path)
    script = ('import sys; from tri.evaluation.solver_cases import load_case; '
              'load_case(sys.argv[1]); assert "torch" not in sys.modules')
    subprocess.run([sys.executable, '-c', script, manifest['cases'][0]['path']], check=True)


def test_known_template_retains_unknown_other_span():
    case = make_controlled_case(observed_fraction=1.)
    assert case.metadata['visible_fraction_first_span'] == 1.
    assert case.metadata['unknown_fraction_both_spans'] == .5
    assert len(case.metadata['probe_variables']) == 1
    assert case.metadata['unknown_positions'] == 16


def test_learned_snapshots_need_no_clean_tokens_and_only_select_validation(tmp_path):
    from dataclasses import asdict
    import torch
    from tri.evaluation.solver_cases import build_learned_cases, _identity
    from tri.models.grid import GridConfig, GridDenoiser

    source = build_tiny_cases()[0]
    length = source.spec.length
    data = tmp_path / 'windows.npz'
    # No clean token array is present. A test row contains no usable features.
    np.savez(data, work_ids=np.array(['001', '002']), start_cells=np.array([0, 0]),
             splits=np.array(['validation', 'test']))
    chords = tmp_path / 'chords.npz'
    features = np.zeros((2, length, 38))
    features[1] = np.nan
    np.savez(chords, chord_features=features)
    torch.manual_seed(81)
    model = GridDenoiser(GridConfig(length=length, hidden=8, layers=1, heads=2, condition_dim=38))
    checkpoint = tmp_path / 'model.pt'
    torch.save({'model_config': asdict(model.config), 'model_state': model.state_dict()}, checkpoint)
    for stage in ('short', 'bar8', 'bar16'):
        root = tmp_path / stage
        (root / 'evaluation').mkdir(parents=True)
        requests = [{'case_id': f'first-{kind}', 'source_index': 0, 'work_id': '001',
                     'source_start_cell': 0, 'kind': kind, 'spec': _serialize_spec(source.spec),
                     'witness_verified': True} for kind in ('unknown', 'partial', 'known', 'harmony')]
        requests += [{**row, 'case_id': f'never-{row["kind"]}', 'source_index': 1, 'work_id': '002'}
                     for row in requests]
        (root / 'cohort.json').write_text(json.dumps({'status': 'completed', 'requests': requests,
            'options': {'split': 'validation', 'dataset': _identity(data), 'chords': _identity(chords)}}))
        (root / 'evaluation/config.json').write_text(json.dumps({
            'checkpoint': _identity(checkpoint), 'checkpoint_seed': 81}))
    cases = build_learned_cases(tmp_path, device='cpu')
    assert len(cases) == 12
    for case in cases:
        assert case.metadata['source_case_id'].startswith('first-')
        assert case.metadata['model_calls'] == 1
        assert np.isfinite(case.logq).all()
        assert np.allclose(np.exp(case.logq).sum(axis=1), 1.)
        assert case.metadata['provider_state'] == [source.spec.observed.get(i) for i in range(length)]
