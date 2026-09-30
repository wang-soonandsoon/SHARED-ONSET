"""Frozen validation revision: existing-weight quality study, then bar studies."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import yaml

from tri.paths import project_root, shared_data_root
from tri.runtime import atomic_json, exclusive_run
from tri.study_runner import _path, _process_start_tick, _runtime_check, _aggregate, study_status


def load_revision_config(path):
    c=yaml.safe_load(Path(path).read_text())
    if c.get('version')!=2 or c.get('protocol')!='validation_revision' or not c.get('stages'):
        raise ValueError('expected version 2 validation_revision config')
    names=[s['name'] for s in c['stages']]
    if len(set(names))!=len(names) or any(not n.replace('_','').isalnum() for n in names):
        raise ValueError('stage names must be unique plain identifiers')
    for stage in c['stages']:
        stage['data_directory']=str(_path(stage['data_directory']))
        if 'checkpoint' in stage:
            stage['checkpoint']=str(_path(stage['checkpoint']))
        if stage.get('split','validation')!='validation':
            raise ValueError('this revision does not evaluate test')
    return c


def _benchmark_worker(spec_data,backend,queue):
    from tri.evaluation.cohorts import deserialize_spec
    from tri.inference.music_backends import make_music_engine
    spec=deserialize_spec(spec_data);t=time.monotonic()
    try:
        engine=make_music_engine(spec,np.full((spec.length,130),-np.log(130)),backend=backend)
        z=engine.log_partition()
        if not np.isfinite(z):
            raise RuntimeError('positive-support frozen request returned nonfinite partition')
        queue.put({'backend':backend,'status':'completed','log_partition':z,'seconds':time.monotonic()-t,
                   'stats':engine.last_stats})
    except Exception as error:
        queue.put({'backend':backend,'status':'budget_exceeded' if type(error).__name__=='BudgetExceeded' else 'unexpected_error',
                   'type':type(error).__name__,'error':str(error),'seconds':time.monotonic()-t})


def scaling_benchmark(plans,output,timeout_seconds=10):
    path=Path(output)
    identities={name:{'options':plan['options'],'case':next(r['case_id'] for r in plan['requests'] if r['kind']=='unknown')} for name,plan in plans.items()}
    config={'cases':identities,'timeout_seconds':timeout_seconds,'backends':['paired','template','automaton','ve']}
    if path.exists():
        report=json.loads(path.read_text())
        if report['config']!=config or report['status']!='completed':
            raise ValueError('saved scaling benchmark is incompatible or failed')
        return report
    context=mp.get_context('spawn');rows=[]
    for name,plan in plans.items():
        case=next(r for r in plan['requests'] if r['kind']=='unknown')
        for backend in config['backends']:
            queue=context.Queue();process=context.Process(target=_benchmark_worker,args=(case['spec'],backend,queue))
            process.start()
            # Read while the child lives so large queue payloads cannot deadlock join.
            try:
                row=queue.get(timeout=timeout_seconds)
            except __import__('queue').Empty:
                row={'backend':backend,'status':'timeout','seconds':timeout_seconds}
            finally:
                if process.is_alive():
                    process.terminate()
                process.join();queue.close()
            rows.append({'stage':name,'length':case['spec']['length'],'gap_cells':plan['options']['gap_cells'],**row})
    errors=[]
    for name in plans:
        values=[r['log_partition'] for r in rows if r['stage']==name and r['status']=='completed']
        if not values or not all(np.isfinite(values)) or max(values)-min(values)>1e-8:
            errors.append(f'{name}: completed backends disagree or have no positive support')
    if any(r['status']=='unexpected_error' for r in rows):
        errors.append('unexpected backend error')
    report={'status':'failed' if errors else 'completed','config':config,'rows':rows,'errors':errors,
            'note':'Identical uniform full130 q and frozen visible spec. CPU single-case timing; timeout includes child startup. Budget/timeout remain outcomes, not infeasibility.'}
    atomic_json(path,report)
    if errors:
        raise RuntimeError(str(errors))
    return report


def run_revision(config_path,output_dir,*,resume=False):
    from tri.data.bars import prepare_bars
    from tri.evaluation.cohorts import build_cohort, file_identity, complete_feasibility_audit
    from tri.evaluation.cohort_study import evaluate_cohort
    from tri.evaluation.blind_audio import build_blind_audio
    from tri.evaluation.quality_report import quality_report
    from tri.models.grid import GridConfig
    from tri.models.training import fit
    c=load_revision_config(config_path);output=_path(output_dir);output.mkdir(parents=True,exist_ok=True)
    with exclusive_run(output/'study.lock'):
        snapshot=output/'config.json'
        if snapshot.exists() and (not resume or json.loads(snapshot.read_text())!=c):
            raise ValueError('revision exists or changed; matching --resume required')
        atomic_json(snapshot,c);started=time.time();stage='startup'
        def state(status,**extra):
            atomic_json(output/'status.json',{'status':status,'pid':os.getpid(),'process_start_tick':_process_start_tick(os.getpid()),
                        'stage':stage,'started_at':started,'updated_at':time.time(),'output_dir':str(output),**extra})
        state('running')
        try:
            atomic_json(output/'runtime.json',_runtime_check(c['device']))
            plans={};paths={}
            for item in c['stages']:
                name=item['name'];directory=Path(item['data_directory']);stage=f'{name}/data_and_cohort';state('running')
                if item.get('bars'):
                    prepare_bars(shared_data_root(),directory,bars=item['bars'],**c['bar_data'])
                dataset=directory/'windows.npz';chords=directory/'chords/chords.npz'
                if item.get('chord_path'):
                    chords=_path(item['chord_path'])
                cohort=output/name/'cohort.json'
                plans[name]=build_cohort(dataset,chords,cohort,seed=c['seed'],bar_aligned=bool(item.get('bars')),**item['cohort'])
                if plans[name]['status']!='completed':
                    raise ValueError('cohort preparation did not complete')
                plans[name]=complete_feasibility_audit(cohort)
                paths[name]=(dataset,chords,cohort)
            stage='scaling_benchmark';state('running')
            scaling_benchmark(plans,output/'scaling_benchmark.json',**c['scaling'])
            reports=[];training_reports=[];listening_reports=[]
            for item in c['stages']:
                name=item['name'];dataset,chords,cohort=paths[name];run=output/name
                if 'checkpoint' in item:
                    checkpoint=Path(item['checkpoint'])
                    identity=file_identity(checkpoint)
                    stamp=run/'checkpoint_source.json'
                    if stamp.exists() and json.loads(stamp.read_text())!=identity:
                        raise ValueError('existing source checkpoint changed')
                    atomic_json(stamp,identity)
                else:
                    stage=f'{name}/train';state('running',active_status=str(run/'train/status.json'))
                    config=GridConfig(length=item['bars']*16,**c['bar_model'])
                    train=fit(dataset,chords,run/'train',config=config,seed=item['training_seed'],device=c['device'],
                              max_gap_cells=item['cohort']['gap_cells'],min_gap_cells=item['cohort']['gap_cells'],
                              resume=(run/'train/last.pt').exists(),**c['bar_training'])
                    checkpoint=Path(train['best_checkpoint']);training_reports.append(str(run/'train/report.json'))
                stage=f'{name}/evaluate';state('running',active_status=str(run/'evaluation/status.json'))
                result=evaluate_cohort(cohort,checkpoint,run/'evaluation',methods=c['methods'],repeats=item['repeats'],
                            steps=c['steps'],seed=c['seed'],device=c['device'],backend='paired',
                            checkpoint_seed=item['training_seed'],resume=(run/'evaluation/config.json').exists())
                reports.append({'model':name,'seed':item['training_seed'],'split':'validation','report':result['outputs']['report']})
                stage=f'{name}/quality_cost';state('running')
                quality_report(cohort,result['outputs']['results'],run/'analysis')
                stage=f'{name}/blind_audio';state('running',active_status=str(run/'listening/status.json'))
                listening=build_blind_audio(cohort,result['outputs']['results'],shared_data_root(),run/'listening',
                                          source_mode='aligned' if item.get('bars') else 'original',seed=c['seed'],render=c['render_audio'])
                listening_reports.append(str(run/'listening/report.json'))
            stage='aggregate';state('running')
            _aggregate(output,reports)
            final={'status':'completed','evaluation_runs':len(reports),'new_training_runs':len(training_reports),
                   'planned_attempts':sum(plan['selected_count']*item['repeats']*len(c['methods']) for item,plan in [(s,plans[s['name']]) for s in c['stages']]),
                   'training_reports':training_reports,'listening_reports':listening_reports,'human_listening':'awaiting human ratings',
                   'elapsed_seconds_this_run':time.time()-started,'aggregate':str(output/'aggregate.json'),'summary_csv':str(output/'summary.csv'),
                   'note':'Validation revision only; original research outputs and test protocol remain preserved. Completion covers automated work, not human ratings.'}
            atomic_json(output/'report.json',final);state('completed',report=str(output/'report.json'))
            return final
        except BaseException as error:
            state('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error),traceback=traceback.format_exc())
            raise


def launch_revision(config_path,output_dir,*,resume=False):
    output=_path(output_dir);output.mkdir(parents=True,exist_ok=True)
    previous=study_status(output)
    if previous.get('status')=='running' and previous.get('process_alive'):
        return {**previous,'status':'already_running'}
    command=[sys.executable,'-m','tri','run-revision','--config',str(Path(config_path).resolve()),'--out',str(output)]
    if resume:
        command.append('--resume')
    with (output/'launcher.log').open('ab') as stream:
        process=subprocess.Popen(command,cwd=project_root(),stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    until=time.monotonic()+2
    while time.monotonic()<until and process.poll() is None:
        current=study_status(output)
        if current.get('pid')==process.pid and current.get('stage')!='startup':
            break
        time.sleep(.05)
    if process.poll() is not None and process.returncode:
        raise RuntimeError(f'revision launch failed; inspect {output/"launcher.log"}')
    return {'status':'launched','pid':process.pid,'output_dir':str(output),'log':str(output/'launcher.log')}
