from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from tri.data.chords import encode_chord_features
from tri.evaluation import cohort_study
from tri.evaluation.cohorts import build_cohort,make_cohort_request,complete_feasibility_audit
from tri.evaluation.blind_audio import build_blind_audio
from tri.errors import BudgetExceeded
from tri.models.grid import GridConfig,GridDenoiser
from tri.runtime import read_jsonl,atomic_jsonl


@pytest.fixture
def inputs(tmp_path):
    tokens=np.tile([62,1,0,0],(4,8)).astype(np.int64)
    works=np.array(['001','002','003','004']);starts=np.zeros(4,dtype=int)
    splits=np.array(['validation']*3+['train']);data=tmp_path/'windows.npz';chords=tmp_path/'chords.npz'
    identity=dict(work_ids=works,start_cells=starts,splits=splits)
    np.savez(data,tokens=tokens,**identity,initial_pitches=np.full(4,-1),time_signatures=np.tile([4,4],(4,1)))
    labels=np.full((4,32),'C:maj');features,known=encode_chord_features(labels)
    np.savez(chords,**identity,chord_labels=labels,chord_known=known,chord_feature_known=known,
        chord_features=features,cell_seconds=np.tile(np.arange(33,dtype=float)/8,(4,1)),row_status=np.full(4,'aligned'))
    model=GridDenoiser(GridConfig(length=32,hidden=8,layers=1,heads=2,condition_dim=38))
    checkpoint=tmp_path/'model.pt';torch.save({'model_config':asdict(model.config),'model_state':model.state_dict()},checkpoint)
    cohort=tmp_path/'cohort.json'
    build_cohort(data,chords,cohort,gap_cells=4,per_kind=1,candidate_windows=3,max_per_work=2)
    return cohort,checkpoint,tmp_path/'evaluation'


def evaluate(inputs,**kwargs):
    return cohort_study.evaluate_cohort(*inputs,repeats=1,steps=2,device='cpu',**kwargs)


def test_cohort_is_visible_only_and_zero_template_is_explicit(inputs):
    plan=json.loads(inputs[0].read_text())
    assert plan['selected_count']==4 and len(plan['selected_by_kind'])==4
    assert plan['options']['split']=='validation'
    tokens=np.tile([62,1,0,0],8);features,_=encode_chord_features(np.full((1,32),'C:maj'))
    a=make_cohort_request(tokens,None,features[0],'unknown',8)
    tokens[8]=70;tokens[24]=74
    b=make_cohort_request(tokens,None,features[0],'unknown',8)
    assert a==b  # Changed hidden pitches terminate before the visible boundary.
    tokens[4:12]=0
    known=make_cohort_request(tokens,None,features[0],'known',8)
    assert known.onset_counts[0].count==0


def test_complete_resume_preserves_result_identity_and_blind_assignment(inputs,monkeypatch):
    result=evaluate(inputs);path=Path(result['outputs']['results']);before=path.stat().st_mtime_ns
    rows=read_jsonl(path)
    assert len(rows)==16 and {r['method'] for r in rows}=={'pooled_template','one_shot_joint','tri_direct','smc_4'}
    out=inputs[-1].parent/'listening'
    report=build_blind_audio(inputs[0],path,out,out,source_mode='aligned',render=False)
    def fail(*a,**k):
        raise AssertionError('completed attempts must not decode again')
    monkeypatch.setattr(cohort_study,'research_decode',fail)
    evaluate(inputs,resume=True)
    assert path.stat().st_mtime_ns==before
    assert build_blind_audio(inputs[0],path,out,out,source_mode='aligned',render=False)==report
    public=(out/'public/index.html').read_text()
    assert all(name not in public for name in ('pooled_template','one_shot_joint','tri_direct','smc_4'))
    assert report['groups']==4 and report['audio_files']==0


def test_failures_stay_in_denominator_and_keep_blind_placeholders(inputs,monkeypatch):
    decode=cohort_study.research_decode
    def limited(method,*a,**k):
        if method=='pooled_template':
            raise BudgetExceeded('test explicit budget outcome')
        return decode(method,*a,**k)
    monkeypatch.setattr(cohort_study,'research_decode',limited)
    result=evaluate(inputs)
    assert result['attempted']==16
    assert sum(r['valid'] for r in read_jsonl(result['outputs']['results']))==12
    out=inputs[-1].parent/'listening'
    pack=build_blind_audio(inputs[0],result['outputs']['results'],out,out,source_mode='aligned',render=False)
    assert pack['groups']==4 and pack['unavailable_variants']==4
    items=json.loads((out/'public/items.json').read_text())['items']
    assert all(len(i['variants'])==4 for i in items)


def test_unexpected_exception_stops_then_resume_retries_and_deduplicates(inputs,monkeypatch):
    original=cohort_study.research_decode
    def broken(*a,**k):
        raise RuntimeError('transient failure')
    monkeypatch.setattr(cohort_study,'research_decode',broken)
    with pytest.raises(RuntimeError,match='transient'):
        evaluate(inputs)
    assert json.loads((inputs[-1]/'status.json').read_text())['status']=='failed'
    monkeypatch.setattr(cohort_study,'research_decode',original)
    result=evaluate(inputs,resume=True)
    assert result['attempted']==16
    assert len(read_jsonl(inputs[-1]/'attempts.jsonl'))==17
    assert len(read_jsonl(inputs[-1]/'results.jsonl'))==16


def test_zero_onset_can_be_feasible_or_boundary_infeasible_without_changing_cohort(inputs):
    plan=json.loads(inputs[0].read_text());data=Path(plan['options']['dataset']['path'])
    with np.load(data) as f:
        arrays={k:f[k].copy() for k in f.files}
    arrays['tokens'][0,6:10]=0
    arrays['tokens'][1,6:10]=0
    arrays['tokens'][1,21]=0
    arrays['tokens'][1,26]=1
    np.savez(data,**arrays)
    dest=inputs[0].parent/'zero_cohort.json'
    build_cohort(data,plan['options']['chords']['path'],dest,gap_cells=4,per_kind=1,candidate_windows=3,max_per_work=4)
    before=dest.read_bytes();stat=dest.stat().st_mtime_ns
    audit=complete_feasibility_audit(dest)
    assert audit['zero_onset_support_counts']=={'feasible':1,'infeasible':1}
    assert sum(audit['support_status_counts'].values())==12
    assert any(r['eligibility_status']=='inactive_template' for r in audit['infeasible_requests'])
    assert dest.read_bytes()==before and dest.stat().st_mtime_ns==stat
    assert complete_feasibility_audit(dest)==audit


def test_quality_report_uses_per_attempt_calls_and_work_grouped_differences(inputs):
    from tri.evaluation.quality_report import quality_report
    plan=json.loads(inputs[0].read_text());rows=[]
    for c in plan['requests']:
        for method,calls,chord in [('one_shot_joint',1,.6),('tri_direct',8,.7)]:
            for rep in range(2):
                rows.append({'cohort_case_id':c['case_id'],'method':method,'valid':True,'model_calls':calls,
                    'elapsed_decode_seconds':calls*.1,'metrics':{'editable_tokens':[62,rep],
                    'onset_pattern':[1,0],'chord_tone_fraction':chord,
                    'mean_edit_boundary_or_internal_jump':2.,'editable_rest_fraction':.1}})
    dest=inputs[0].parent/'quality_rows.jsonl';atomic_jsonl(dest,rows)
    report=quality_report(inputs[0],dest,inputs[0].parent/'analysis')
    assert report['tables']['all']['tri_direct']['mean_model_calls_per_attempt']==8
    assert report['tables']['all']['tri_direct']['unique_token_fraction']==1
    assert report['tables']['all']['tri_direct']['unique_onset_fraction']==.5
    diff=next(r for r in report['comparisons'] if r['kind']=='all' and r['metric']=='chord_tone_fraction')
    assert diff['matched_requests']==4 and diff['works']==len(plan['selected_by_work'])
    assert diff['work_macro_mean_difference']==pytest.approx(.1)
