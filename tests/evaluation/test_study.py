from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tri.data.chords import encode_chord_features
from tri.errors import BudgetExceeded, InvalidSpecification
from tri.evaluation import study
from tri.evaluation.suites import make_request
from tri.models.grid import GridConfig,GridDenoiser
from tri.runtime import read_jsonl
from tri.sampling.research_methods import research_decode


@pytest.fixture
def fixture(tmp_path):
    tokens=np.tile([62,1,0,0],(2,8)).astype(np.int64)
    works=np.array(['001','002'])
    starts=np.array([0,32])
    splits=np.array(['validation','train'])
    data=tmp_path/'windows.npz'
    chords=tmp_path/'chords.npz'
    np.savez(data,tokens=tokens,work_ids=works,start_cells=starts,splits=splits,
             initial_pitches=np.array([-1,-1]),time_signatures=np.array([[4,4],[4,4]]))
    labels=np.full((2,32),'C:maj',dtype='U8')
    features,known=encode_chord_features(labels)
    np.savez(chords,work_ids=works,start_cells=starts,splits=splits,chord_labels=labels,
             chord_known=known,chord_feature_known=known,chord_features=features,
             cell_seconds=np.tile(np.arange(33,dtype=float),(2,1)),row_status=np.array(['aligned']*2))
    model=GridDenoiser(GridConfig(length=32,hidden=8,layers=1,heads=2,condition_dim=38))
    checkpoint=tmp_path/'model.pt'
    torch.save({'model_config':asdict(model.config),'model_state':model.state_dict()},checkpoint)
    return data,chords,checkpoint,tmp_path/'out'


def run(fixture,**kwargs):
    return study.evaluate_study(*fixture,methods=('one_shot_joint','tri_direct'),suites=('unknown4',),
                                repeats=1,per_work=1,steps=2,device='cpu',**kwargs)


def test_successful_resume_skips_decodes_and_preserves_output(fixture,monkeypatch):
    first=run(fixture)
    before=Path(first['outputs']['results']).read_bytes()
    def fail(*args,**kwargs):
        raise AssertionError('completed cases must not decode again')
    monkeypatch.setattr(study,'research_decode',fail)
    second=run(fixture,resume=True)
    assert second['attempted']==2
    assert Path(second['outputs']['results']).read_bytes()==before
    assert all(r['valid'] for r in read_jsonl(second['outputs']['results']))


def test_changed_sidecar_is_rejected_before_resume(fixture):
    run(fixture)
    chord=fixture[1]
    stat=chord.stat()
    os.utime(chord,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000))
    with pytest.raises(ValueError,match='configuration'):
        run(fixture,resume=True)


def test_expected_request_failure_is_recorded_for_every_method(fixture,monkeypatch):
    def bad(*args,**kwargs):
        raise InvalidSpecification('unsupported fixture')
    monkeypatch.setattr(study,'make_request',bad)
    result=run(fixture)
    rows=read_jsonl(result['outputs']['results'])
    assert len(rows)==2
    assert all(r['status']=='unsupported_input' and r['model_calls']==0 for r in rows)
    assert all(v['attempted']==1 and v['completion_rate']==0 for v in result['results'].values())


def test_unexpected_retry_keeps_attempt_history_and_exports_unique_latest_rows(fixture,monkeypatch):
    def broken(*args,**kwargs):
        raise RuntimeError('transient fixture failure')
    monkeypatch.setattr(study,'research_decode',broken)
    with pytest.raises(RuntimeError,match='implementation error'):
        run(fixture)
    out=fixture[-1]
    old=read_jsonl(out/'attempts.jsonl')[0]
    stale=out/'midi'/f'000000_{old["method"]}.mid'
    stale.write_bytes(b'old output after interrupted export')
    def retry(method,*args,**kwargs):
        if method==old['method']:
            raise BudgetExceeded('known failure after retry')
        return research_decode(method,*args,**kwargs)
    monkeypatch.setattr(study,'research_decode',retry)
    result=run(fixture,resume=True)
    assert len(read_jsonl(out/'attempts.jsonl'))==3
    assert len(read_jsonl(out/'results.jsonl'))==2
    assert not stale.exists()
    assert result['attempted']==2


def test_partial_json_tail_recovers_but_corrupt_middle_does_not(tmp_path):
    path=tmp_path/'rows.jsonl'
    path.write_bytes(b'{"a":1}\n{"a":')
    assert read_jsonl(path,recover_tail=True)==[{'a':1}]
    assert path.read_bytes()==b'{"a":1}\n'
    path.write_bytes(b'broken\n{"a":2}\n')
    with pytest.raises(ValueError,match='corrupt'):
        read_jsonl(path,recover_tail=True)


def test_unknown_and_partial_specs_do_not_depend_on_hidden_interior_notes():
    original=np.tile([62,1,0,0],8)
    for suite in ('unknown4','unknown8','partial8','known8'):
        before=make_request(original,None,suite)
        changed=original.copy()
        for i in range(32):
            # Change a hidden onset that terminates before visible context.
            if i not in before.observed and original[i]>=2 and i+1 not in before.observed:
                changed[i]=95
        after=make_request(changed,None,suite)
        assert dict(before.observed)==dict(after.observed)
        assert before.pitches==after.pitches
        assert before.onset_counts==after.onset_counts
        assert dict(before.fixed_soundings)==dict(after.fixed_soundings)
