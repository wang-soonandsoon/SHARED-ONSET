"""One foreground experiment runner, a detached launcher, and read-only status.

The runner serializes GPU jobs, resumes durable training/evaluation state, and
writes inspectable summaries. It never changes experiment settings in response
to test outcomes, and does not silently switch an unavailable GPU job to CPU.
"""
from dataclasses import asdict
import csv
import json
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


def _path(value):
    path=Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root()/path).resolve()


def load_config(path):
    config=yaml.safe_load(Path(path).read_text())
    if not isinstance(config,dict) or config.get('version')!=1:
        raise ValueError('study configuration must be a version:1 mapping')
    required={'device','seeds','data','models','training','evaluation','benchmarks','listening'}
    if required-set(config):
        raise ValueError(f'missing config keys: {sorted(required-set(config))}')
    if not config['seeds'] or len(set(config['seeds']))!=len(config['seeds']):
        raise ValueError('study seeds must be nonempty and unique')
    if not config['models'] or any(not str(name).replace('_','').isalnum() for name in config['models']):
        raise ValueError('model names must use letters, digits and underscores')
    config['data']['directory']=str(_path(config['data']['directory']))
    return config


def _runtime_check(device):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    torch.set_num_threads(2)
    report={'python':sys.executable,'torch':str(torch.__version__),'device':str(device)}
    if str(device).startswith('cuda'):
        try:
            a=torch.eye(8,device=device)
            if not bool(torch.equal(a@a,a)):
                raise RuntimeError('GPU numerical startup check failed')
            torch.cuda.synchronize(device)
            report['gpu']=torch.cuda.get_device_name(torch.device(device))
        except Exception as error:
            raise RuntimeError(f'CUDA startup failed: {error}. No CPU fallback was performed. Check study-status and the documented process-local driver setup.') from error
    return report


def _process_start_tick(pid):
    try:
        # Process comm may contain spaces/parentheses; starttime is field22
        # after the final ')' delimiter in Linux procfs stat.
        text=Path(f'/proc/{int(pid)}/stat').read_text()
        return text[text.rfind(')')+2:].split()[19]
    except (OSError,ValueError,IndexError,TypeError):
        return None


def _ensure_data(config):
    from tri.data.prepare import prepare_windows
    from tri.data.chords import align_chords, load_chord_sidecar
    options=config['data']
    directory=Path(options['directory'])
    dataset=directory/'windows.npz'
    chord_path=directory/'chords/chords.npz'
    generation={k:options[k] for k in ('seed','max_files','window_beats','max_windows_per_file')}
    stamp=directory/'generation_config.json'
    if stamp.exists() and json.loads(stamp.read_text())!=generation:
        raise ValueError('data generation configuration changed; choose a new data directory')
    if not dataset.exists():
        prepare_windows(shared_data_root(),directory,window_local=True,**generation)
    else:
        report=json.loads((directory/'report.json').read_text())
        if report['window_beats']!=generation['window_beats'] or report['seed']!=generation['seed'] or report['attempted_files']!=min(generation['max_files'],report['found_primary_files']):
            raise ValueError('existing data report does not match study preprocessing')
        if not stamp.exists() and generation['max_windows_per_file']>=100000 and report['windows']!=report['window_diagnostics']['valid_windows_before_cap']:
            raise ValueError('expected all valid windows, but existing data was capped')
    atomic_json(stamp,generation)
    if not chord_path.exists():
        result=align_chords(dataset,shared_data_root(),chord_path.parent)
        if result['status']=='needs_attention':
            raise RuntimeError('chord alignment reported unexpected implementation errors')
    load_chord_sidecar(dataset,chord_path)
    with np.load(dataset,allow_pickle=False) as archive:
        length=archive['tokens'].shape[1]
    return dataset,chord_path,length


def _aggregate(output, evaluation_reports):
    from tri.evaluation.study import summarize
    from tri.runtime import read_jsonl
    records=[]
    for item in evaluation_reports:
        report=json.loads(Path(item['report']).read_text())
        # Each within-run result already handles resumed attempts. Keep the
        # final observation for a retried unexpected implementation error.
        latest={}
        for row in read_jsonl(report['outputs']['results']):
            latest[(row['request_id'],row['method'])]=row
        for row in latest.values():
            records.append({**row,'model':item['model'],'split':item['split'],'training_seed':item['seed']})
    groups={}
    for model,split in sorted({(r['model'],r['split']) for r in records}):
        groups[f'{model}/{split}']=summarize([r for r in records if r['model']==model and r['split']==split])
    aggregate={'status':'completed','runs':evaluation_reports,'groups':groups,
               'note':'All planned outcomes retained. Bootstrap resamples works; replicates/seeds within works remain grouped. Music statistics are descriptive proxies.'}
    atomic_json(output/'aggregate.json',aggregate)
    with (output/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
        fields=['model','split','suite','method','attempted','valid','completion_rate','work_macro_completion_rate','work_count','model_calls','p50_seconds','p95_seconds','chord_tone_fraction','mean_edit_boundary_or_internal_jump','mean_unique_token_fraction']
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for name,table in groups.items():
            model,split=name.split('/')
            for values in table.values():
                row={k:values.get(k) for k in fields}
                row.update(model=model,split=split,chord_tone_fraction=values['valid_output_metrics']['chord_tone_fraction'],
                           mean_edit_boundary_or_internal_jump=values['valid_output_metrics']['mean_edit_boundary_or_internal_jump'])
                writer.writerow(row)
    # Static scientific figure, using the same unfiltered aggregated counts.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,table in groups.items():
        suites=sorted({v['suite'] for v in table.values()})
        methods=sorted({v['method'] for v in table.values()})
        matrix=np.asarray([[table[f'{suite}/{method}']['completion_rate'] for method in methods] for suite in suites])
        fig,ax=plt.subplots(figsize=(max(8,len(methods)*.8),max(3,len(suites)*.65)))
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues')
        ax.set_xticks(range(len(methods)),methods,rotation=40,ha='right',fontsize=8)
        ax.set_yticks(range(len(suites)),suites)
        ax.set_title(name+' — complete-request validity')
        for i in range(len(suites)):
            for j in range(len(methods)):
                ax.text(j,i,f'{matrix[i,j]:.0%}',ha='center',va='center',color='white' if matrix[i,j]>.6 else 'black',fontsize=8)
        fig.colorbar(im,ax=ax,label='valid / all attempts')
        fig.tight_layout()
        fig.savefig(output/(name.replace('/','_')+'_validity.png'),dpi=160)
        plt.close(fig)
    return aggregate


def run_study(config_path,output_dir,*,resume=False):
    from tri.models.grid import GridConfig
    from tri.models.training import fit
    from tri.evaluation.study import evaluate_study
    from tri.evaluation.listening import build_listening_pack
    from tri.inference.exact import Budget
    config=load_config(config_path)
    output=_path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    snapshot=output/'config.json'
    with exclusive_run(output/'study.lock'):
        if snapshot.exists():
            if not resume:
                raise ValueError('study exists; use --resume or a new --out')
            if json.loads(snapshot.read_text())!=config:
                raise ValueError('study configuration differs from saved experiment; use a new output directory')
        atomic_json(snapshot,config)
        started=time.time()
        stage='startup'
        def state(status,**kwargs):
            atomic_json(output/'status.json',{'status':status,'pid':os.getpid(),'stage':stage,
                       'process_start_tick':_process_start_tick(os.getpid()),
                       'updated_at':time.time(),'started_at':started,'output_dir':str(output),**kwargs})
        state('running')
        reports=[]
        try:
            runtime=_runtime_check(config['device'])
            atomic_json(output/'runtime.json',runtime)
            stage='data'
            state('running')
            dataset,chords,length=_ensure_data(config)
            stage='mathematical_benchmarks'
            state('running')
            benchmark_report=output/'benchmarks/report.json'
            cached_benchmark=json.loads(benchmark_report.read_text()) if benchmark_report.exists() else None
            expected_benchmark_config={**config['benchmarks'],'seed':config['seeds'][0]}
            if cached_benchmark is None or cached_benchmark.get('status')!='completed' or cached_benchmark.get('config')!=expected_benchmark_config:
                from tri.benchmarks_research import research_benchmarks
                benchmark_result=research_benchmarks(output/'benchmarks',**config['benchmarks'],seed=config['seeds'][0])
                if benchmark_result.get('status')!='completed':
                    raise RuntimeError(f'mathematical benchmark failed; inspect {benchmark_report}')
            for model_name,model_options in config['models'].items():
                model_config=GridConfig(length=length,**model_options)
                for seed in config['seeds']:
                    run=output/model_name/f'seed_{seed}'
                    train_dir=run/'train'
                    stage=f'train/{model_name}/{seed}'
                    state('running',active_status=str(train_dir/'status.json'))
                    # Even at the completed total step, fit validates saved
                    # optimizer/model/input identity and restores missing best
                    # safely. A stale report alone is never a completion proof.
                    train_report=fit(dataset,chords,train_dir,config=model_config,seed=seed,device=config['device'],
                                     resume=(train_dir/'last.pt').exists(),**config['training'])
                    for split in config['evaluation']['splits']:
                        evaluation_dir=run/split
                        stage=f'evaluate/{model_name}/{seed}/{split}'
                        state('running',active_status=str(evaluation_dir/'status.json'))
                        options={k:v for k,v in config['evaluation'].items() if k not in ('splits','max_factor_entries','max_workspace_mib')}
                        options['budget']=Budget(max_factor_entries=config['evaluation']['max_factor_entries'],
                                                 max_workspace_bytes=config['evaluation']['max_workspace_mib']*1024*1024)
                        evaluated=evaluate_study(dataset,chords,train_report['best_checkpoint'],evaluation_dir,
                            split=split,seed=seed,device=config['device'],resume=(evaluation_dir/'config.json').exists(),**options)
                        reports.append({'model':model_name,'seed':seed,'split':split,'report':str(evaluation_dir/'report.json')})
                        # This exports a frozen subset for later human listening;
                        # no listening scores are invented or used for selection.
                        stage=f'listening/{model_name}/{seed}/{split}'
                        state('running')
                        listening=build_listening_pack(evaluated['outputs']['results'],shared_data_root(),run/f'{split}_listening',
                                                      seed=seed,**config['listening'])
                        if listening['status']=='needs_attention':
                            raise RuntimeError('listening export reported an implementation error')
            stage='aggregate'
            state('running')
            aggregate=_aggregate(output,reports)
            final={'status':'completed','evaluation_runs':len(reports),'aggregate':str(output/'aggregate.json'),
                   'summary_csv':str(output/'summary.csv'),'elapsed_seconds_this_run':time.time()-started,
                   'config':str(snapshot),'limitations':'Executable finite monophonic shared-onset study. Human judgments and scientific conclusions remain to be established.'}
            atomic_json(output/'report.json',final)
            state('completed',report=str(output/'report.json'))
            return final
        except BaseException as error:
            state('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error),traceback=traceback.format_exc())
            raise


def study_status(output_dir):
    output=_path(output_dir)
    status_path=output/'status.json'
    if not status_path.exists():
        return {'status':'not_started','output_dir':str(output)}
    state=json.loads(status_path.read_text())
    pid=state.get('pid')
    actual_start=_process_start_tick(pid)
    alive=actual_start is not None and actual_start==state.get('process_start_tick')
    state['process_alive']=alive
    if state['status']=='running' and not alive:
        state['status']='stopped_without_final_status'
    active=state.get('active_status')
    if active and Path(active).exists():
        state['active']=json.loads(Path(active).read_text())
    return state


def launch_study(config_path,output_dir,*,resume=False):
    output=_path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    existing=study_status(output)
    if existing.get('status')=='running' and existing.get('process_alive'):
        return {**existing,'status':'already_running'}
    command=[sys.executable,'-m','tri','run-study','--config',str(Path(config_path).resolve()),'--out',str(output)]
    if resume:
        command.append('--resume')
    log=output/'launcher.log'
    with log.open('ab') as stream:
        process=subprocess.Popen(command,cwd=project_root(),stdout=stream,stderr=subprocess.STDOUT,
                                 start_new_session=True,stdin=subprocess.DEVNULL)
    until=time.monotonic()+2
    while time.monotonic()<until and process.poll() is None:
        status=study_status(output)
        if status.get('pid')==process.pid and status.get('stage')!='startup':
            break
        time.sleep(.05)
    if process.poll() is not None and process.returncode:
        raise RuntimeError(f'study failed during launch (exit {process.returncode}); inspect {log}')
    return {'status':'launched','pid':process.pid,'log':str(log),'output_dir':str(output),'command':command}
