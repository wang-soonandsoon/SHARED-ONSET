"""CLI declarations for complete research runs; legacy demos stay stable."""
from pathlib import Path
from tri.paths import project_root,processed_root,shared_data_root

COMMANDS={'align-chords','train','evaluate','bench-research','run-study','launch-study','study-status','listening-pack','run-revision','launch-revision'}


def add_parsers(sub):
    align=sub.add_parser('align-chords',help='align original MIDI chord annotations and encode features')
    align.add_argument('--dataset',type=Path,default=processed_root()/'bootstrap/windows.npz')
    align.add_argument('--data-root',type=Path,default=shared_data_root())
    align.add_argument('--out',type=Path,default=processed_root()/'chord_alignment')
    train=sub.add_parser('train',help='train on all training windows with durable resume and validation selection')
    train.add_argument('--dataset',type=Path,default=processed_root()/'research_8beats/windows.npz')
    train.add_argument('--chords',type=Path,default=processed_root()/'research_8beats/chords/chords.npz')
    train.add_argument('--out',type=Path,default=project_root()/'runs/training')
    train.add_argument('--steps',type=int,default=3000)
    train.add_argument('--batch-size',type=int,default=32)
    train.add_argument('--hidden',type=int,default=128)
    train.add_argument('--layers',type=int,default=4)
    train.add_argument('--heads',type=int,default=4)
    train.add_argument('--dropout',type=float,default=.1)
    train.add_argument('--condition-dim',type=int,choices=(0,38),default=38)
    train.add_argument('--lr',type=float,default=.0003)
    train.add_argument('--eval-every',type=int,default=100)
    train.add_argument('--save-every',type=int,default=100)
    train.add_argument('--max-gap-cells',type=int,default=8)
    train.add_argument('--device',default='cuda:0')
    train.add_argument('--resume',action='store_true')
    train.add_argument('--seed',type=int,default=20260912)
    evaluate=sub.add_parser('evaluate',help='resumable same-checkpoint request suites and generation controls')
    evaluate.add_argument('--dataset',type=Path,default=processed_root()/'research_8beats/windows.npz')
    evaluate.add_argument('--chords',type=Path,default=processed_root()/'research_8beats/chords/chords.npz')
    evaluate.add_argument('--checkpoint',type=Path,required=True)
    evaluate.add_argument('--out',type=Path,required=True)
    from tri.evaluation.suites import SUITES
    from tri.sampling.research_methods import RESEARCH_METHODS
    evaluate.add_argument('--methods',nargs='+',choices=RESEARCH_METHODS,default=['one_shot_joint','tri_direct','smc_4'])
    evaluate.add_argument('--suites',nargs='+',choices=SUITES,default=['unknown4','unknown8'])
    evaluate.add_argument('--split',choices=('validation','test'),default='validation')
    evaluate.add_argument('--repeats',type=int,default=2)
    evaluate.add_argument('--per-work',type=int,default=2)
    evaluate.add_argument('--limit',type=int)
    evaluate.add_argument('--steps',type=int,default=8)
    evaluate.add_argument('--backend',choices=('ve','template','automaton','auto'),default='auto')
    evaluate.add_argument('--device',default='cuda:0')
    evaluate.add_argument('--seed',type=int,default=20260912)
    evaluate.add_argument('--resume',action='store_true')
    bench=sub.add_parser('bench-research',help='coupled backend, exact path-target and linked-record experiments')
    bench.add_argument('--out',type=Path,default=project_root()/'runs/research_benchmarks')
    bench.add_argument('--trials',type=int,default=128)
    bench.add_argument('--seed',type=int,default=20260912)
    for name in ('run-study','launch-study'):
        p=sub.add_parser(name,help='run experiment plan in foreground' if name=='run-study' else 'launch the experiment plan as a detached process')
        p.add_argument('--config',type=Path,default=project_root()/'configs/research.yaml')
        p.add_argument('--out',type=Path,default=project_root()/'runs/research')
        p.add_argument('--resume',action='store_true')
    status=sub.add_parser('study-status',help='read study and active-stage progress without changing the run')
    status.add_argument('--out',type=Path,default=project_root()/'runs/research')
    for name in ('run-revision','launch-revision'):
        p=sub.add_parser(name,help='run or detach the frozen validation and aligned-bar revision')
        p.add_argument('--config',type=Path,default=project_root()/'configs/revision.yaml')
        p.add_argument('--out',type=Path,default=project_root()/'runs/revision')
        p.add_argument('--resume',action='store_true')
    listening=sub.add_parser('listening-pack',help='export matched original accompaniment and blind comparison MIDI')
    listening.add_argument('--results',type=Path,required=True)
    listening.add_argument('--out',type=Path,required=True)
    listening.add_argument('--data-root',type=Path,default=shared_data_root())
    listening.add_argument('--methods',nargs='+')
    listening.add_argument('--limit',type=int,default=12)
    listening.add_argument('--seed',type=int,default=20260912)


def run(args):
    if args.command in ('run-revision','launch-revision'):
        from tri.revision_runner import run_revision,launch_revision
        return (run_revision if args.command=='run-revision' else launch_revision)(args.config,args.out,resume=args.resume)
    if args.command=='align-chords':
        from tri.data.chords import align_chords
        report=align_chords(args.dataset,args.data_root,args.out)
        return {k:v for k,v in report.items() if k!='source_metadata'}
    if args.command=='train':
        import numpy as np
        import torch
        torch.set_num_threads(2)
        from tri.models.grid import GridConfig
        from tri.models.training import fit
        with np.load(args.dataset,allow_pickle=False) as archive:
            length=archive['tokens'].shape[1]
        config=GridConfig(length=length,hidden=args.hidden,layers=args.layers,heads=args.heads,
                          dropout=args.dropout,condition_dim=args.condition_dim)
        return fit(args.dataset,args.chords,args.out,steps=args.steps,batch_size=args.batch_size,
                   seed=args.seed,device=args.device,config=config,lr=args.lr,
                   eval_every=args.eval_every,save_every=args.save_every,
                   max_gap_cells=args.max_gap_cells,resume=args.resume)
    if args.command=='evaluate':
        from tri.evaluation.study import evaluate_study
        report=evaluate_study(args.dataset,args.chords,args.checkpoint,args.out,methods=args.methods,
            suites=args.suites,split=args.split,repeats=args.repeats,per_work=args.per_work,limit=args.limit,
            steps=args.steps,seed=args.seed,device=args.device,backend=args.backend,resume=args.resume)
        return {k:v for k,v in report.items() if k!='results'}
    if args.command=='bench-research':
        from tri.benchmarks_research import research_benchmarks
        return research_benchmarks(args.out,trials=args.trials,seed=args.seed)
    if args.command in ('run-study','launch-study','study-status'):
        from tri.study_runner import run_study,launch_study,study_status
        if args.command=='study-status':
            return study_status(args.out)
        return (run_study if args.command=='run-study' else launch_study)(args.config,args.out,resume=args.resume)
    from tri.evaluation.listening import build_listening_pack
    return build_listening_pack(args.results,args.data_root,args.out,seed=args.seed,limit=args.limit,methods=args.methods)
