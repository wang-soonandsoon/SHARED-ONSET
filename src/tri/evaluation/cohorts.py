"""Validation-only, model-independent request design and frozen cohorts."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import math

import numpy as np

from tri.data.chords import load_chord_sidecar
from tri.domain.music import MusicSpec, CountRule, verify_music
from tri.errors import BudgetExceeded, ZeroMass
from tri.evaluation.batch import load_evaluation_windows
from tri.inference.music_backends import make_music_engine
from tri.request_builder import sounding_pitches
from tri.runtime import atomic_json

KINDS=('unknown','partial','known','harmony')


def file_identity(path):
    p=Path(path).resolve();s=p.stat()
    return {'path':str(p),'bytes':s.st_size,'modified_ns':s.st_mtime_ns}


def serialize_spec(spec):
    return {'length':spec.length,'pitches':list(spec.pitches),'observed':{str(k):v for k,v in spec.observed.items()},
            'initial_pitch':spec.initial_pitch,'fixed_soundings':{str(k):v for k,v in spec.fixed_soundings.items()},
            'equal_onsets':[list(p) for p in spec.equal_onsets],
            'onset_counts':[{'positions':list(c.positions),'count':c.count} for c in spec.onset_counts],
            'pitch_classes':{str(k):list(v) for k,v in spec.pitch_classes.items()},'motion_cost':spec.motion_cost,
            'pitch_ranges':{str(k):list(v) for k,v in spec.pitch_ranges.items()},
            'max_adjacent_interval':spec.max_adjacent_interval,'enforce_end':spec.enforce_end,'end_pitch':spec.end_pitch}


def deserialize_spec(data):
    data=dict(data)
    for key in ['observed','fixed_soundings','pitch_classes','pitch_ranges']:
        data[key]={int(k):v for k,v in data.get(key,{}).items()}
    data['onset_counts']=tuple(CountRule(tuple(c['positions']),c['count']) for c in data['onset_counts'])
    return MusicSpec(**data)


def make_cohort_request(tokens,initial_pitch,features,kind,gap_cells,*,bar_aligned=False):
    if kind not in KINDS or gap_cells<2 or len(tokens)<4*gap_cells:
        raise ValueError('unsupported cohort request dimensions or kind')
    length=len(tokens)
    starts=(length//4,3*length//4-gap_cells) if bar_aligned else (length//4-gap_cells//2,3*length//4-gap_cells//2)
    if bar_aligned and (gap_cells%16 or any(s%16 for s in starts)):
        raise ValueError('bar requests require complete 16-cell aligned spans')
    spans=tuple(tuple(range(s,s+gap_cells)) for s in starts)
    editable=set(spans[0]+spans[1])
    if kind=='known':
        editable.difference_update(spans[0])
    elif kind=='partial':
        editable.difference_update((spans[0][0],spans[0][min(3,gap_cells-1)]))
    observed={i:int(t) for i,t in enumerate(tokens) if i not in editable}
    sounding=sounding_pitches(tokens,initial_pitch)
    anchors={i:sounding[i] for i in observed}
    pitches=set(range(48,85))|{t-2 for t in observed.values() if t>=2}|{p for p in anchors.values() if p is not None}
    if initial_pitch is not None:
        pitches.add(initial_pitch)
    count=sum(observed[i]>=2 for i in spans[0]) if kind=='known' else max(2,gap_cells//4)
    classes={}
    if kind=='harmony':
        for i in editable:
            if features[i,37] and not features[i,36]:
                classes[i]=tuple(int(p) for p in np.flatnonzero(features[i,12:24]))
    return MusicSpec(length=length,pitches=tuple(sorted(pitches)),observed=observed,
                     initial_pitch=initial_pitch,fixed_soundings=anchors,
                     equal_onsets=tuple(zip(*spans)),onset_counts=tuple(CountRule(span,count) for span in spans),
                     pitch_classes=classes,motion_cost=.02)


def build_cohort(dataset,chords,output,*,gap_cells=8,bar_aligned=False,per_kind=6,
                 candidate_windows=64,max_per_work=3,min_visible_onsets=4,seed=20260915):
    """Audit the predefined candidate pool before ANY checkpoint is loaded.

    No hidden gap notes are used for eligibility. Boundary sounding values
    belong to fixed visible context. Known-template activity is visible too.
    Infeasible, inactive and budget-unknown candidates are separate outcomes.
    """
    path=Path(output).resolve()
    if min(per_kind,candidate_windows,max_per_work)<1 or min_visible_onsets<1:
        raise ValueError('cohort counts and activity threshold must be positive')
    options={'dataset':file_identity(dataset),'chords':file_identity(chords),'gap_cells':gap_cells,
             'bar_aligned':bar_aligned,'per_kind':per_kind,'candidate_windows':candidate_windows,
             'max_per_work':max_per_work,'min_visible_onsets':min_visible_onsets,'seed':seed,'split':'validation'}
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['options']!=options:
            raise ValueError('cohort or input identity changed; choose a new output')
        if saved['status']!='completed':
            raise ValueError('saved cohort has insufficient eligible requests')
        return saved
    windows=load_evaluation_windows(dataset,limit=2**31-1,split='validation')
    sidecar=load_chord_sidecar(dataset,chords)
    by_work=defaultdict(list)
    for window in windows:
        by_work[window.work_id].append(window)
    if not by_work:
        raise ValueError('dataset has no validation windows')
    rng=np.random.default_rng(seed)
    works=sorted(by_work);rng.shuffle(works)
    for work in works:
        rng.shuffle(by_work[work])
    # Round robin prevents sorted-file intros or a few long works dominating.
    candidates=[]
    for index in range(max(map(len,by_work.values()))):
        for work in works:
            if index<len(by_work[work]):
                candidates.append(by_work[work][index])
    candidates=candidates[:candidate_windows]
    audit=[];feasible=[]
    for window in candidates:
        for kind in KINDS:
            spec=make_cohort_request(window.tokens,window.initial_pitch,sidecar['chord_features'][window.source_index],kind,gap_cells,bar_aligned=bar_aligned)
            row={'case_id':f'{window.request_id}:{kind}:g{gap_cells}','source_index':window.source_index,
                 'work_id':window.work_id,'source_start_cell':window.start_cell,'kind':kind,
                 'suite':f'{kind}_g{gap_cells}','visible_onsets':sum(t>=2 for t in spec.observed.values()),
                 'shared_onsets':spec.onset_counts[0].count,'status':None}
            if row['shared_onsets']==0:
                row['status']='inactive_template'
            elif row['visible_onsets']<min_visible_onsets:
                row['status']='insufficient_visible_activity'
            elif kind=='harmony' and not spec.pitch_classes:
                row['status']='missing_harmony_condition'
            else:
                try:
                    engine=make_music_engine(spec,np.full((spec.length,130),-math.log(130)),backend='paired')
                    z=engine.log_partition()
                    row['uniform_log_partition']=z if math.isfinite(z) else None
                    row['status']='feasible' if math.isfinite(z) else 'infeasible'
                    if math.isfinite(z):
                        sample=engine.sample_batch([f'y{i}' for i in range(spec.length)],np.random.default_rng(seed))
                        if not verify_music(tuple(sample.assignment[f'y{i}'] for i in range(spec.length)),spec).valid:
                            raise RuntimeError('cohort witness failed independent verification')
                        row['witness_verified']=True
                        feasible.append({**row,'spec':serialize_spec(spec)})
                except ZeroMass:
                    row['status']='infeasible'
                except BudgetExceeded as error:
                    row.update(status='inference_budget_unknown',reason=str(error))
            audit.append(row)
    selected=[];per_work_count=Counter();kind_count=Counter();kind_works=defaultdict(set)
    # Round robin over kinds and works; eligibility never uses generation scores.
    for round_index in range(per_kind):
        for kind in KINDS:
            for row in feasible:
                if row['kind']==kind and row['work_id'] not in kind_works[kind] and per_work_count[row['work_id']]<max_per_work:
                    selected.append(row);kind_count[kind]+=1;per_work_count[row['work_id']]+=1;kind_works[kind].add(row['work_id'])
                    break
    if any(kind_count[kind]!=per_kind for kind in KINDS):
        status='insufficient_eligible_requests'
    else:
        status='completed'
    report={'status':status,'options':options,'candidate_count':len(audit),'status_counts':dict(Counter(r['status'] for r in audit)),
            'kind_status_counts':{k:dict(Counter(r['status'] for r in audit if r['kind']==k)) for k in KINDS},
            'selected_count':len(selected),'selected_by_kind':dict(kind_count),'selected_by_work':dict(per_work_count),
            'requests':selected,'candidate_audit':audit,
            'policy':'Validation-only, positive activity from visible context, model-independent exact support plus independent witness. Infeasible/inactive/unknown counts retained separately; no post-generation replacement.'}
    atomic_json(path,report)
    if status!='completed':
        raise ValueError(f'cohort has too few eligible requests: {dict(kind_count)}; inspect {path}')
    return report


def complete_feasibility_audit(cohort_path):
    """Separate support from musical activity without changing selected requests.

    In particular, a zero-onset template can be feasible via REST/HOLD, or
    impossible because its two visible boundary soundings cannot be connected.
    It remains excluded from this active-melody quality cohort in either case.
    """
    path=Path(cohort_path);plan=json.loads(path.read_text());audit_path=path.with_name('feasibility.json')
    if audit_path.exists():
        saved=json.loads(audit_path.read_text())
        if saved.get('cohort_identity')!=file_identity(path):
            raise ValueError('frozen cohort changed after its support audit')
        return saved
    dataset=plan['options']['dataset']['path'];chords=plan['options']['chords']['path']
    if file_identity(dataset)!=plan['options']['dataset'] or file_identity(chords)!=plan['options']['chords']:
        raise ValueError('cannot audit changed cohort data')
    windows={w.source_index:w for w in load_evaluation_windows(dataset,limit=2**31-1,split='validation')}
    sidecar=load_chord_sidecar(dataset,chords);infeasible=[]
    for row in plan['candidate_audit']:
        spec=None
        if row['status'] in ('feasible','infeasible','inference_budget_unknown'):
            support=row['status']
        else:
            w=windows[row['source_index']]
            spec=make_cohort_request(w.tokens,w.initial_pitch,sidecar['chord_features'][w.source_index],
                row['kind'],plan['options']['gap_cells'],bar_aligned=plan['options']['bar_aligned'])
            try:
                z=make_music_engine(spec,np.full((spec.length,130),-math.log(130)),backend='paired').log_partition()
                support='feasible' if math.isfinite(z) else 'infeasible'
                row['uniform_log_partition']=z if math.isfinite(z) else None
            except BudgetExceeded:
                support='inference_budget_unknown'
        row['support_status']=support
        if support=='infeasible':
            if spec is None:
                w=windows[row['source_index']]
                spec=make_cohort_request(w.tokens,w.initial_pitch,sidecar['chord_features'][w.source_index],
                    row['kind'],plan['options']['gap_cells'],bar_aligned=plan['options']['bar_aligned'])
            infeasible.append({'case_id':row['case_id'],'eligibility_status':row['status'],'spec':serialize_spec(spec)})
    plan.update(support_audit_version=1,
        support_status_counts=dict(Counter(r['support_status'] for r in plan['candidate_audit'])),
        zero_onset_support_counts=dict(Counter(r['support_status'] for r in plan['candidate_audit'] if r['shared_onsets']==0)),
        infeasible_requests=infeasible,
        zero_onset_policy='No NOTE is not automatically impossible: REST and inherited HOLD are allowed when all anchors agree. Excluded from active quality comparison; exact support is reported separately.')
    plan['cohort_identity']=file_identity(path)
    atomic_json(audit_path,plan)
    return plan
