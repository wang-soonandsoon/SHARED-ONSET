"""Resumable same-checkpoint suite evaluation; no tuning on test results."""
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from tri.data.chords import load_chord_sidecar
from tri.data.midi import write_grid_midi
from tri.evaluation.batch import load_evaluation_windows, _verify_returned, _error_status, _json_value, _sync_device
from tri.evaluation.suites import SUITES, make_request, music_statistics
from tri.inference.exact import Budget
from tri.models.grid import ModelProbabilityProvider
from tri.models.train import load_checkpoint
from tri.runtime import atomic_json, atomic_jsonl, exclusive_run, read_jsonl
from tri.sampling.research_methods import RESEARCH_METHODS, research_decode


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[(row['suite'],row['method'])].append(row)
    report={}
    for (suite,method),group in sorted(groups.items()):
        times=[r['elapsed_decode_seconds'] for r in group if r['elapsed_decode_seconds'] is not None]
        by_work=defaultdict(list)
        diversity=defaultdict(list)
        for row in group:
            by_work[row['work_id']].append(float(row['valid']))
            if row['valid'] and row['metrics'] is not None:
                diversity[(row.get('training_seed',row.get('checkpoint_seed')),row['work_id'],row['source_start_cell'])].append(row['metrics'])
        work_rates=np.asarray([np.mean(v) for _,v in sorted(by_work.items())])
        rng=np.random.default_rng(410)
        boot=np.mean(rng.choice(work_rates,size=(1000,len(work_rates)),replace=True),axis=1)
        token_div,pattern_div=[],[]
        for values in diversity.values():
            if len(values)<2:
                continue
            token_div.append(len({tuple(v['editable_tokens']) for v in values})/len(values))
            pattern_div.append(len({tuple(v['onset_pattern']) for v in values})/len(values))
        metric_mean={}
        for key in ('chord_tone_fraction','mean_edit_boundary_or_internal_jump','editable_rest_fraction','reconstruction_token_accuracy_diagnostic'):
            values=[r['metrics'][key] for r in group if r['valid'] and r['metrics'] is not None and r['metrics'][key] is not None]
            metric_mean[key]=float(np.mean(values)) if values else None
        report[f'{suite}/{method}']={
            'suite':suite,'method':method,'attempted':len(group),
            'returned':sum(r['returned'] for r in group),'valid':sum(r['valid'] for r in group),
            'exported':sum(r['exported'] for r in group),
            'completion_rate':sum(r['valid'] for r in group)/len(group),
            'work_macro_completion_rate':float(np.mean(work_rates)),
            'work_bootstrap_95_percentile_interval':np.percentile(boot,[2.5,97.5]).tolist(),
            'work_count':len(work_rates),'model_calls':sum(r['model_calls'] for r in group),
            'p50_seconds':float(np.percentile(times,50)) if times else None,
            'p95_seconds':float(np.percentile(times,95)) if times else None,
            'status_counts':dict(Counter(r['status'] for r in group)),
            'valid_output_metrics':metric_mean,
            'mean_unique_token_fraction':float(np.mean(token_div)) if token_div else None,
            'mean_unique_onset_pattern_fraction':float(np.mean(pattern_div)) if pattern_div else None,
            'diversity_request_groups':len(token_div),
        }
    return report


def evaluate_study(dataset, chords, checkpoint, output_dir, *, methods=('one_shot_joint','tri_direct'),
                   suites=('unknown4','unknown8'), split='validation', repeats=2, per_work=2,
                   limit=None, steps=8, seed=20260912, device='cpu', backend='auto',
                   budget=None, resume=False):
    """Run every planned case, preserving invalid/failing cases in denominators.

    Known terminal failures are completed observations. Unexpected errors stop
    the study after the record is saved, and are retried on explicit resume.
    """
    import torch
    torch.set_num_threads(2)
    methods,suites=tuple(methods),tuple(suites)
    if not methods or len(set(methods))!=len(methods) or any(m not in RESEARCH_METHODS for m in methods):
        raise ValueError('methods must be distinct supported research methods')
    if not suites or len(set(suites))!=len(suites) or any(s not in SUITES for s in suites):
        raise ValueError('suites must be distinct supported request suites')
    if repeats<1 or per_work<1 or (limit is not None and limit<1) or steps<1:
        raise ValueError('positive repeat/window/step limits required')
    output=Path(output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    budget=budget or Budget()
    checkpoint=Path(checkpoint).resolve()
    config={'dataset':str(Path(dataset).resolve()),'chords':str(Path(chords).resolve()),
            'checkpoint':str(checkpoint),'checkpoint_size':checkpoint.stat().st_size,
            'checkpoint_modified_ns':checkpoint.stat().st_mtime_ns,
            'methods':list(methods),'suites':list(suites),'split':split,'repeats':repeats,
            'per_work':per_work,'limit':limit,'steps':steps,'seed':seed,'backend':backend,
            'budget':asdict(budget),'device':str(device)}
    config['input_files']=[{'path':str(Path(p).resolve()),'bytes':Path(p).stat().st_size,
                           'modified_ns':Path(p).stat().st_mtime_ns} for p in (dataset,chords)]
    with exclusive_run(output/'evaluation.lock'):
        config_path=output/'config.json'
        if config_path.exists():
            if not resume:
                raise ValueError('evaluation output exists; use resume or a new output directory')
            if json.loads(config_path.read_text())!=config:
                raise ValueError('resume configuration or checkpoint changed; use a new output directory')
        atomic_json(config_path,config)
        windows=load_evaluation_windows(dataset,limit=2**31-1,split=split)
        sidecar=load_chord_sidecar(dataset,chords)
        counts=Counter()
        selected=[]
        for window in windows:
            if counts[window.work_id]<per_work:
                selected.append(window)
                counts[window.work_id]+=1
            if limit is not None and len(selected)>=limit:
                break
        model=load_checkpoint(checkpoint,device)
        if model.config.condition_dim not in (0,38):
            raise ValueError('research suite expects condition_dim=38 or the explicit no-chord ablation=0')
        if any(len(w.tokens)!=model.config.length for w in selected):
            raise ValueError('checkpoint/window lengths disagree')
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        results_path=output/'results.jsonl'
        attempts_path=output/'attempts.jsonl'
        old_rows=read_jsonl(attempts_path,recover_tail=resume)
        rows_by_key={(r['request_id'],r['method']):r for r in old_rows}
        planned=len(selected)*len(suites)*repeats*len(methods)
        planned_keys={(f's{seed}:{window.request_id}:{suite}:r{replicate}',method)
                      for window in selected for suite in suites for replicate in range(repeats) for method in methods}
        if not set(rows_by_key)<=planned_keys:
            raise ValueError('saved result identities do not belong to this evaluation plan')
        (output/'requests').mkdir(exist_ok=True)
        (output/'midi').mkdir(exist_ok=True)
        started=time.monotonic()
        new_count=0
        def progress(status,**extra):
            done=sum(r['status']!='unexpected_error' for r in rows_by_key.values())
            elapsed=time.monotonic()-started
            atomic_json(output/'status.json',{'status':status,'pid':__import__('os').getpid(),
                'updated_at':time.time(),'planned':planned,'completed':done,
                'elapsed_seconds_this_run':elapsed,'eta_seconds':elapsed/new_count*(planned-done) if new_count else None,
                **extra})
        progress('running')
        try:
            with attempts_path.open('a',encoding='utf-8') as stream:
                ordinal=0
                for window_index,window in enumerate(selected):
                    condition=sidecar['chord_features'][window.source_index]
                    for suite_index,suite in enumerate(suites):
                        spec,request_error=None,None
                        try:
                            spec=make_request(window.tokens,window.initial_pitch,suite,condition)
                            editable=tuple(i for i in range(spec.length) if i not in spec.observed)
                        except Exception as error:
                            request_error=error
                        for replicate in range(repeats):
                            request_id=f's{seed}:{window.request_id}:{suite}:r{replicate}'
                            case_seed=seed+window_index*1_000_000+suite_index*1000+replicate
                            manifest={'request_id':request_id,'work_id':window.work_id,'source_start_cell':window.start_cell,
                                'seed':case_seed,'suite':suite,'replicate':replicate,'initial_pitch':window.initial_pitch}
                            if spec is not None:
                                manifest.update({'observed_tokens':dict(spec.observed),'fixed_soundings':dict(spec.fixed_soundings),
                                'editable_positions':editable,'pitches':spec.pitches,'equal_onsets':spec.equal_onsets,
                                'onset_counts':[{'positions':c.positions,'count':c.count} for c in spec.onset_counts],
                                'pitch_classes':dict(spec.pitch_classes),'motion_cost':spec.motion_cost})
                            else:
                                manifest['construction_error']={'type':type(request_error).__name__,'message':str(request_error)}
                            atomic_json(output/'requests'/f'{ordinal:06d}.json',_json_value(manifest))
                            order=np.random.default_rng(case_seed+131).permutation(methods).tolist()
                            for method in order:
                                key=(request_id,method)
                                if key in rows_by_key and rows_by_key[key]['status']!='unexpected_error':
                                    continue
                                calls=0
                                def counted(state,noise):
                                    nonlocal calls
                                    calls+=1
                                    return provider(state,noise)
                                row={'request_id':request_id,'work_id':window.work_id,'source_start_cell':window.start_cell,
                                    'source_index':window.source_index,'initial_pitch':window.initial_pitch,'split':split,
                                    'suite':suite,'replicate':replicate,'method':method,'seed':case_seed,'checkpoint_seed':seed,
                                    'returned':False,'valid':False,'exported':False,'raw_tokens':None,'metrics':None,
                                    'verification':None,'model_calls':0,'elapsed_decode_seconds':None,'diagnostics':None,
                                    'error':None,'midi':None,'trace':None,'status':'pending'}
                                stage='request'
                                try:
                                    export_path=output/'midi'/f'{ordinal:06d}_{method}.mid'
                                    export_path.unlink(missing_ok=True)
                                    if request_error is not None:
                                        raise request_error
                                    stage='provider_setup'
                                    provider=ModelProbabilityProvider(model,editable,condition if model.config.condition_dim else None)
                                    stage='decode'
                                    _sync_device(device)
                                    t=time.monotonic()
                                    try:
                                        result=research_decode(method,spec,counted,steps=steps,seed=case_seed,backend=backend,budget=budget)
                                    finally:
                                        _sync_device(device)
                                        row['elapsed_decode_seconds']=time.monotonic()-t
                                    row.update(returned=True,raw_tokens=list(result.tokens),trace=_json_value(result.trace),
                                               diagnostics=_json_value(result.diagnostics))
                                    if result.model_calls!=calls:
                                        raise RuntimeError('decoder model call accounting mismatch')
                                    stage='verification'
                                    check=_verify_returned(result.tokens,window,spec)
                                    row['verification']=check
                                    row['valid']=check['valid']
                                    if row['valid']:
                                        row['metrics']=music_statistics(result.tokens,window.tokens,spec,condition)
                                        stage='export'
                                        row['midi']=write_grid_midi(list(result.tokens),export_path,
                                                               initial_pitch=window.initial_pitch,time_signature=window.time_signature)
                                        row.update(status='exported',exported=True)
                                    else:
                                        row['status']='returned_invalid'
                                except Exception as error:
                                    row['status']=_error_status(error)
                                    if stage=='export' and isinstance(error,OSError):
                                        row['status']='export_failure'
                                    row['error']={'type':type(error).__name__,'message':str(error),'stage':stage}
                                    if row['status']=='unexpected_error':
                                        row['error']['traceback']=traceback.format_exc()
                                row['model_calls']=calls
                                stream.write(json.dumps(_json_value(row),ensure_ascii=False,allow_nan=False)+'\n')
                                stream.flush()
                                rows_by_key[key]=row
                                new_count+=1
                                progress('running',current_request=request_id,current_method=method)
                                if row['status']=='unexpected_error':
                                    raise RuntimeError(f'implementation error in {method}: {row["error"]["message"]}')
                            ordinal+=1
        except BaseException as error:
            progress('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
            raise
        rows=list(rows_by_key.values())
        atomic_jsonl(results_path,[_json_value(row) for row in rows])
        report={'status':'completed','config':config,'selected_windows':len(selected),
                'work_ids':sorted(counts),'planned':planned,'attempted':len(rows),'results':summarize(rows),
                'outputs':{'results':str(results_path),'report':str(output/'report.json')},
                'limitations':['Objective statistics are descriptive, not listening scores.',
                    'Intervals resample works, not correlated windows; few works limit precision.',
                    'Finite-particle outputs are approximations; one-shot has a different terminal target.',
                    'Grid windows preserve recorded meter but are not downbeat-aligned bars.',
                    'Decode timing excludes artifact I/O; method order is randomized within requests.']}
        atomic_json(output/'report.json',_json_value(report))
        progress('completed')
        return report
