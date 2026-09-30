"""Executable project entry points. No implicit dependency/data downloads."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

from tri.paths import project_root, shared_data_root, processed_root
from tri.errors import TRIError


def _device(name):
    if name != "auto":
        return name
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tri", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    from tri.cli_research import add_parsers, COMMANDS, run as run_research
    add_parsers(sub)
    doctor = sub.add_parser("doctor", help="inspect this Python environment and shared data")
    doctor.add_argument("--gpu", action="store_true")
    core = sub.add_parser("core-demo", help="exact two-gap relational demonstration")
    core.add_argument("--out", type=Path, default=project_root()/"runs/core")
    bench = sub.add_parser("bench-inference", help="compare exact rhythm/count DP and compact generic VE")
    bench.add_argument("--out", type=Path, default=project_root()/"runs/inference")
    data = sub.add_parser("prepare-data", help="prepare explicitly scoped POP909 windows")
    data.add_argument("--data-root", type=Path, default=shared_data_root())
    data.add_argument("--out", type=Path, default=processed_root()/"bootstrap")
    data.add_argument("--max-files", type=int, default=24)
    data.add_argument("--max-windows-per-file", type=int, default=4)
    data.add_argument("--window-beats", type=int, default=None, help="explicit beat blocks; default uses two recorded4/4 bars")
    data.add_argument("--window-local", action="store_true", help="accept clean windows while excluding problematic intervals")
    train = sub.add_parser("train-smoke", help="bounded real-window masked training/save/load smoke")
    train.add_argument("--dataset", type=Path, default=processed_root()/"bootstrap/windows.npz")
    train.add_argument("--out", type=Path, default=project_root()/"runs/bootstrap/train")
    train.add_argument("--steps", type=int, default=120)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--device", default="auto")
    demo = sub.add_parser("music-demo", help="generate a verified two-gap completion with a trained checkpoint")
    demo.add_argument("--dataset", type=Path, default=processed_root()/"bootstrap/windows.npz")
    demo.add_argument("--checkpoint", type=Path, default=project_root()/"runs/bootstrap/train/checkpoint.pt")
    demo.add_argument("--out", type=Path, default=project_root()/"runs/bootstrap/completion")
    demo.add_argument("--steps", type=int, default=4)
    demo.add_argument("--device", default="auto")
    evaluate = sub.add_parser("evaluate-smoke", help="batch-check fixed validation requests with one frozen checkpoint")
    evaluate.add_argument("--dataset", type=Path, default=processed_root()/"bootstrap/windows.npz")
    evaluate.add_argument("--checkpoint", type=Path, default=project_root()/"runs/bootstrap/train/checkpoint.pt")
    evaluate.add_argument("--out", type=Path, default=project_root()/"runs/advance2/batch")
    evaluate.add_argument("--limit", type=int, default=8)
    evaluate.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    evaluate.add_argument("--steps", type=int, default=4)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--methods", nargs="+", default=["tri_direct"],
                          choices=("raw_reference", "local_constraints", "one_shot_joint", "tri_direct"))
    evaluate.add_argument("--max-factor-entries", type=int, default=2_000_000)
    evaluate.add_argument("--max-workspace-mib", type=int, default=512)
    for p in (core, bench, data, train, demo, evaluate):
        p.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            report = {"python": sys.executable, "python_version": sys.version.split()[0],
                      "project_root": str(project_root()), "shared_data": str(shared_data_root()),
                      "data_present": (shared_data_root()/"raw/pop909").is_dir(),
                      "versions": {}}
            for name in ("numpy", "scipy", "mido", "torch", "pytest"):
                try:
                    report["versions"][name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    report["versions"][name] = None
            if args.gpu:
                import torch
                report["cuda_available"] = torch.cuda.is_available()
                if report["cuda_available"]:
                    x = torch.eye(8, device="cuda:0")
                    report["gpu0"] = torch.cuda.get_device_name(0)
                    report["gpu_matmul_ok"] = bool(torch.equal(x@x, x))
        elif args.command == "core-demo":
            from tri.experiments import core_demo
            report = core_demo(args.out, seed=args.seed)
        elif args.command == "bench-inference":
            from tri.benchmarks import benchmark_inference
            report = benchmark_inference(args.out, seed=args.seed)
        elif args.command == "prepare-data":
            from tri.data.prepare import prepare_windows
            report = prepare_windows(args.data_root, args.out, max_files=args.max_files,
                                     max_windows_per_file=args.max_windows_per_file,
                                     window_beats=args.window_beats, window_local=args.window_local, seed=args.seed)
            # Full per-work details live in report.json; keep console compact.
            report = {k: v for k, v in report.items() if k not in {"accepted", "skipped"}}
        elif args.command == "train-smoke":
            import torch
            torch.set_num_threads(2)
            from tri.models.train import train_smoke
            report = train_smoke(args.dataset, args.out, steps=args.steps, batch_size=args.batch_size,
                                 seed=args.seed, device=_device(args.device))
        elif args.command == "evaluate-smoke":
            import torch
            torch.set_num_threads(2)
            from tri.evaluation.batch import evaluate_smoke
            from tri.inference.exact import Budget
            report = evaluate_smoke(args.dataset, args.checkpoint, args.out,
                                    methods=tuple(args.methods), limit=args.limit, split=args.split,
                                    steps=args.steps, seed=args.seed, device=_device(args.device),
                                    budget=Budget(max_factor_entries=args.max_factor_entries,
                                                  max_workspace_bytes=args.max_workspace_mib*1024*1024))
            # Full rows and visible request manifests are saved to disk.
            report = {k: v for k, v in report.items() if k not in {"rows", "results", "requests", "by_work"}}
        elif args.command in COMMANDS:
            report = run_research(args)
        else:
            import torch
            torch.set_num_threads(2)
            from tri.experiments import music_demo
            report = music_demo(args.dataset, args.checkpoint, args.out, steps=args.steps,
                                seed=args.seed, device=_device(args.device))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if report.get("status") in {"needs_attention", "failed"}:
            raise SystemExit(2)
    except (TRIError, ValueError, OSError) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2) from exc
