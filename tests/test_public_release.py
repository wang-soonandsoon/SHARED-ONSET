"""Check the portable release entry points rather than private author paths."""
import json
import os
from pathlib import Path

import pytest

from tri.evaluation.shared_rhythm_cases import make_shared_rhythm_case
from tri.evaluation.solver_cases import save_cases
from tri.evaluation.shared_rhythm_extension import prepare_plan, PROTOCOL
from tri.evaluation import shared_rhythm_real_cases as real
from tri.evaluation import shared_rhythm_scale_cases as long


def test_relative_manifest_works_outside_its_directory(tmp_path, monkeypatch):
    folder = tmp_path / 'frozen'
    manifest = save_cases([make_shared_rhythm_case(R=3, L=2, D=3, K=1)], folder)
    entry = manifest['cases'][0]
    entry['path'] = Path(entry['path']).name
    path = folder / 'cases.json'
    path.write_text(json.dumps(manifest))
    config = {'protocol': PROTOCOL, 'common': {'cpu': min(os.sched_getaffinity(0)),
              'budget': {}, 'rss_limit_mib': 512, 'max_proposals': 10000},
              'stages': {'synthetic': {'manifest': str(path), 'expected_cases': 1,
              'family': 'synthetic_control', 'backends': ['product_multi', 'onset_reset'],
              'timing_repeats': 1, 'samples_per_worker': 2, 'worker_seconds': 10,
              'sample_seed': 7, 'order_seed': 8}}}
    monkeypatch.chdir(tmp_path)
    _, plan = prepare_plan(config, 'synthetic')
    assert plan['planned_rows'] == 2
    assert all(Path(job['path']).is_file() for job in plan['jobs'])
    assert plan['jobs'][0]['draw_seeds'] == plan['jobs'][1]['draw_seeds']


@pytest.mark.parametrize('prepare', [real.prepare_real_requests, long.prepare_long_requests])
def test_explicit_provenance_checks_changed_inputs(tmp_path, prepare):
    data = tmp_path / 'windows.npz'
    data.write_bytes(b'fixture')
    provenance = {key: real._identity(data) for key in ('dataset', 'chords', 'checkpoint')}
    data.write_bytes(b'changed fixture')
    with pytest.raises(ValueError, match='Frozen dataset identity changed'):
        prepare(tmp_path / 'requests', provenance=provenance)


def test_public_api_uses_the_full_implementation():
    from shared_onset import MusicSpec, OnsetResetMusicInference, AdaptiveOnsetRejectionSampler
    from tri.domain.music import MusicSpec as DomainSpec
    from tri.inference.onset_reset import OnsetResetMusicInference as FullBase
    from tri.sampling.onset_rejection_adaptive import AdaptiveOnsetRejectionSampler as FullSampler
    assert (MusicSpec, OnsetResetMusicInference, AdaptiveOnsetRejectionSampler) == (DomainSpec, FullBase, FullSampler)
    from shared_onset.errors import SharedOnsetError, InvalidSpecification
    assert issubclass(InvalidSpecification, SharedOnsetError)
