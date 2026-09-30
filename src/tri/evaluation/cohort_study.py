"""Durable evaluation of an independently selected validation cohort."""
from pathlib import Path
import json
import time
import traceback

import numpy as np

from tri.data.chords import load_chord_sidecar
from tri.data.midi import write_grid_midi
from tri.evaluation.batch import load_evaluation_windows, _verify_returned, _error_status, _json_value, _sync_device
from tri.evaluation.cohorts import deserialize_spec, file_identity
from tri.evaluation.study import summarize
from tri.evaluation.suites import music_statistics
from tri.inference.exact import Budget
from tri.models.grid import ModelProbabilityProvider
from tri.models.train import load_checkpoint
from tri.runtime import atomic_json, atomic_jsonl, exclusive_run, read_jsonl
from tri.sampling.research_methods import research_decode, RESEARCH_METHODS


def evaluate_cohort(cohort_path,checkpoint,output_dir,*,methods=('pooled_template','one_shot_joint','tri_direct','smc_4'),
                    repeats=8,steps=8,seed=20260915,device='cuda:0',backend='paired',resume=False,checkpoint_seed=None):
    import torch
    torch.set_num_threads(2)
    if repeats<1 or steps<1 or not methods or len(set(methods))!=len(methods) or any(m not in RESEARCH_METHODS for m in methods):
        raise ValueError('invalid cohort evaluation settings')
    plan=json.loads(Path(cohort_path).read_text())
    if plan['status']!='completed' or plan['options']['split']!='validation':
        raise ValueError('revised evaluation requires a completed validation-only cohort')
    dataset=Path(plan['options']['dataset']['path']);chords=Path(plan['options']['chords']['path'])
    for key,p in [('dataset',dataset),('chords',chords)]:
        if file_identity(p)!=plan['options'][key]:
            raise ValueError('data changed after cohort freezing')
    config={'cohort':file_identity(cohort_path),'checkpoint':file_identity(checkpoint),'methods':list(methods),
            'repeats':repeats,'steps':steps,'seed':seed,'device':device,'backend':backend,'checkpoint_seed':checkpoint_seed,
            'protocol':'validation_cohort_v2','template_sampler':'feature_count_dp_v2'}
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    with exclusive_run(output/'evaluation.lock'):
        if (output/'config.json').exists():
            if not resume or json.loads((output/'config.json').read_text())!=config:
                raise ValueError('evaluation exists or settings changed; explicit matching resume required')
        atomic_json(output/'config.json',config)
        windows={w.source_index:w for w in load_evaluation_windows(dataset,limit=2**31-1,split='validation')}
        sidecar=load_chord_sidecar(dataset,chords)
        model=load_checkpoint(checkpoint,device)
        if model.config.condition_dim not in (0,38):
            raise ValueError('unsupported checkpoint conditioning')
        for p in model.parameters():
            p.requires_grad_(False)
        cases=plan['requests']
        if len({r['case_id'] for r in cases})!=len(cases):
            raise ValueError('duplicate cohort case')
        for case in cases:
            w=windows[case['source_index']]
            if (w.work_id,w.start_cell)!=(case['work_id'],case['source_start_cell']) or len(w.tokens)!=model.config.length:
                raise ValueError('cohort/window/checkpoint identity mismatch')
        expected={(f"{case['case_id']}:s{seed}:r{rep}",method) for case in cases for rep in range(repeats) for method in methods}
        old=read_jsonl(output/'attempts.jsonl',recover_tail=resume)
        rows={(r['request_id'],r['method']):r for r in old}
        if not set(rows)<=expected:
            raise ValueError('saved attempts do not belong to this cohort')
        (output/'midi').mkdir(exist_ok=True)
        started=time.monotonic();new_count=0
        def state(status,**extra):
            done=sum(r['status'] not in ('unexpected_error','verifier_failure') for r in rows.values())
            elapsed=time.monotonic()-started
            atomic_json(output/'status.json',{'status':status,'pid':__import__('os').getpid(),'updated_at':time.time(),
                        'completed':done,'planned':len(expected),'elapsed_seconds_this_run':elapsed,
                        'eta_seconds':elapsed/new_count*(len(expected)-done) if new_count else None,**extra})
        state('running')
        try:
            with (output/'attempts.jsonl').open('a',encoding='utf-8') as stream:
                for ci,case in enumerate(cases):
                    w=windows[case['source_index']];spec=deserialize_spec(case['spec'])
                    condition=sidecar['chord_features'][w.source_index]
                    editable=tuple(i for i in range(spec.length) if i not in spec.observed)
                    provider=ModelProbabilityProvider(model,editable,condition if model.config.condition_dim else None)
                    for rep in range(repeats):
                        request_id=f"{case['case_id']}:s{seed}:r{rep}";draw_seed=seed+ci*10000+rep
                        for method in np.random.default_rng(draw_seed+131).permutation(methods).tolist():
                            key=(request_id,method)
                            if key in rows and rows[key]['status'] not in ('unexpected_error','verifier_failure'):
                                continue
                            calls=0
                            def counted(tokens,noise):
                                nonlocal calls
                                calls+=1
                                return provider(tokens,noise)
                            row={'request_id':request_id,'cohort_case_id':case['case_id'],'work_id':w.work_id,
                                 'source_index':w.source_index,'source_start_cell':w.start_cell,'initial_pitch':w.initial_pitch,
                                 'suite':case['suite'],'split':'validation','replicate':rep,'seed':draw_seed,'checkpoint_seed':checkpoint_seed,
                                 'method':method,'returned':False,'valid':False,'exported':False,'raw_tokens':None,
                                 'metrics':None,'verification':None,'model_calls':0,'elapsed_decode_seconds':None,
                                 'diagnostics':None,'error':None,'midi':None,'trace':None,'status':'pending'}
                            stage='decode';_sync_device(device);t=time.monotonic()
                            try:
                                result=research_decode(method,spec,counted,steps=steps,seed=draw_seed,backend=backend,budget=Budget())
                                _sync_device(device);row['elapsed_decode_seconds']=time.monotonic()-t
                                if result.model_calls!=calls:
                                    raise RuntimeError('model call accounting mismatch')
                                row.update(returned=True,raw_tokens=list(result.tokens),diagnostics=_json_value(result.diagnostics),trace=_json_value(result.trace))
                                stage='verification';check=_verify_returned(result.tokens,w,spec)
                                row.update(verification=check,valid=bool(check['valid']))
                                if row['valid']:
                                    row['metrics']=music_statistics(result.tokens,w.tokens,spec,condition)
                                    stage='export';dest=output/'midi'/f'{ci:04d}_r{rep:02d}_{method}.mid'
                                    write_grid_midi(result.tokens,dest,initial_pitch=w.initial_pitch,time_signature=w.time_signature)
                                    row.update(midi=str(dest),status='exported',exported=True)
                                else:
                                    row['status']='returned_invalid'
                                    if method in ('one_shot_joint','tri_direct','smc_4'):
                                        raise RuntimeError('complete-constraint method returned invalid music')
                            except Exception as error:
                                _sync_device(device)
                                if row['elapsed_decode_seconds'] is None:
                                    row['elapsed_decode_seconds']=time.monotonic()-t
                                row['status']=_error_status(error)
                                row['error']={'type':type(error).__name__,'message':str(error),'stage':stage}
                                if row['status'] in ('unexpected_error','verifier_failure'):
                                    row['error']['traceback']=traceback.format_exc()
                            row['model_calls']=calls
                            stream.write(json.dumps(_json_value(row),ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                            rows[key]=row;new_count+=1;state('running',current_request=request_id,current_method=method)
                            if row['status'] in ('unexpected_error','verifier_failure'):
                                raise RuntimeError(f"cohort {stage} error: {row['error']['message']}")
        except BaseException as error:
            state('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
            raise
        final=list(rows.values())
        if set(rows)!=expected:
            raise RuntimeError('completed evaluation has missing cases')
        # Preserve identity on a no-op resume: completed listening assignments
        # refer to these exact final results and must remain resumable too.
        if read_jsonl(output/'results.jsonl')!=final:
            atomic_jsonl(output/'results.jsonl',final)
        report={'status':'completed','config':config,'planned':len(expected),'attempted':len(final),
                'cohort_candidate_status_counts':plan['status_counts'],'selected_requests':len(cases),
                'results':summarize(final),'outputs':{'results':str(output/'results.jsonl'),'report':str(output/'report.json')},
                'limitations':['Selected feasible validation cohort, not original all-request completion rate.',
                               'Infeasible/inactive/budget-unknown candidate counts stay in cohort.json.',
                               'Repeated outputs do not constitute human listening judgments.']}
        atomic_json(output/'report.json',report);state('completed')
        return report
