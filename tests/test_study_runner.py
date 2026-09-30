"""Exercise orchestration safety without training, GPU, or research benchmarks."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tri import study_runner
from tri.runtime import atomic_json, exclusive_run


@pytest.fixture
def study(tmp_path, monkeypatch):
    from tri import benchmarks_research
    from tri.evaluation import listening, study as evaluation
    from tri.models import training

    config = {
        "version": 1, "device": "cpu", "seeds": [11],
        "data": {"directory": str(tmp_path / "data"), "seed": 11,
                 "max_files": 3, "window_beats": 4, "max_windows_per_file": 100000},
        "models": {"chord": {"hidden": 8, "layers": 1, "heads": 2,
                              "dropout": 0.0, "condition_dim": 38}},
        "training": {"steps": 10, "batch_size": 2, "lr": 0.001,
                     "eval_every": 2, "save_every": 2, "max_gap_cells": 4},
        "evaluation": {"splits": ["validation"], "methods": ["tri_direct"],
                       "suites": ["unknown4"], "repeats": 2, "per_work": 1,
                       "steps": 2, "backend": "ve", "max_factor_entries": 10000,
                       "max_workspace_mib": 4},
        "benchmarks": {"trials": 2},
        "listening": {"limit": 1, "methods": ["tri_direct"]},
    }
    calls = {name: [] for name in ("runtime", "data", "bench", "fit", "evaluate", "listen", "aggregate")}
    config_path, output = tmp_path / "study.yaml", tmp_path / "out"
    dataset, chords = tmp_path / "data/windows.npz", tmp_path / "data/chords.npz"

    def save_config():
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    def runtime(device):
        calls["runtime"].append(device)
        return {"device": device}

    def data(options):
        calls["data"].append(copy.deepcopy(options))
        return dataset, chords, 16

    def bench(directory, **options):
        calls["bench"].append(options)
        result = {"status": "completed", "config": options}
        atomic_json(Path(directory) / "report.json", result)
        return result

    def fit(dataset_arg, chords_arg, directory, **options):
        assert (dataset_arg, chords_arg) == (dataset, chords)
        directory = Path(directory)
        calls["fit"].append({"directory": directory, **options})
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("last.pt", "best.pt"):
            if not (directory / name).exists():
                (directory / name).write_bytes(b"mock checkpoint")
        result = {"status": "completed", "step": options["steps"],
                  "best_checkpoint": str(directory / "best.pt")}
        atomic_json(directory / "report.json", result)
        return result

    def evaluate(dataset_arg, chords_arg, checkpoint, directory, **options):
        assert (dataset_arg, chords_arg) == (dataset, chords)
        assert Path(checkpoint).is_file()
        calls["evaluate"].append({"checkpoint": checkpoint, "directory": directory, **options})
        directory = Path(directory)
        atomic_json(directory / "config.json", {"checkpoint": checkpoint})
        result = {"status": "completed", "outputs": {"results": str(directory / "results.jsonl")}}
        (directory / "results.jsonl").write_text("")
        atomic_json(directory / "report.json", result)
        return result

    def listen(results, source_root, directory, **options):
        calls["listen"].append({"results": results, "directory": directory, **options})
        return {"status": "completed"}

    def aggregate(directory, reports):
        calls["aggregate"].append(reports)
        result = {"status": "completed", "runs": reports}
        atomic_json(Path(directory) / "aggregate.json", result)
        return result

    monkeypatch.setattr(study_runner, "_runtime_check", runtime)
    monkeypatch.setattr(study_runner, "_ensure_data", data)
    monkeypatch.setattr(study_runner, "_aggregate", aggregate)
    monkeypatch.setattr(benchmarks_research, "research_benchmarks", bench)
    monkeypatch.setattr(training, "fit", fit)
    monkeypatch.setattr(evaluation, "evaluate_study", evaluate)
    monkeypatch.setattr(listening, "build_listening_pack", listen)
    save_config()
    return SimpleNamespace(config=config, path=config_path, output=output, calls=calls,
                           save_config=save_config, benchmarks=benchmarks_research,
                           evaluation=evaluation)


def test_full_cartesian_execution_preserves_conditioning_and_seeds(study):
    study.config["models"]["no_chord"] = {**study.config["models"]["chord"], "condition_dim": 0}
    study.config["seeds"] = [11, 12]
    study.config["evaluation"]["splits"] = ["validation", "test"]
    study.save_config()
    result = study_runner.run_study(study.path, study.output)
    assert result["status"] == "completed"
    assert result["evaluation_runs"] == 8
    assert len(study.calls["fit"]) == 4
    assert len(study.calls["evaluate"]) == len(study.calls["listen"]) == 8
    assert {(c["config"].condition_dim, c["seed"]) for c in study.calls["fit"]} == {
        (38, 11), (38, 12), (0, 11), (0, 12)}
    assert all(not c["resume"] for c in study.calls["fit"] + study.calls["evaluate"])
    assert all(c["config"].length == 16 and c["steps"] == 10 for c in study.calls["fit"])
    for call in study.calls["evaluate"]:
        assert f"seed_{call['seed']}/train/best.pt" in call["checkpoint"]
        assert call["directory"].name == call["split"]
        assert call["methods"] == ["tri_direct"]
        assert call["budget"].max_workspace_bytes == 4 * 1024 * 1024
    assert len(study.calls["aggregate"][0]) == 8
    assert json.loads((study.output / "status.json").read_text())["status"] == "completed"


def test_resume_revalidates_completed_training_and_restores_missing_best(study):
    study_runner.run_study(study.path, study.output)
    directory = study.output / "chord/seed_11/train"
    atomic_json(directory / "report.json", {"status": "completed", "step": 1,
                                          "best_checkpoint": str(directory / "best.pt")})
    (directory / "best.pt").unlink()
    study_runner.run_study(study.path, study.output, resume=True)
    assert len(study.calls["fit"]) == 2
    assert study.calls["fit"][-1]["resume"] is True
    assert study.calls["fit"][-1]["steps"] == 10
    assert (directory / "best.pt").is_file()
    assert study.calls["evaluate"][-1]["resume"] is True
    assert len(study.calls["bench"]) == 1


@pytest.mark.parametrize("cached", [
    {"status": "failed", "config": {"trials": 2, "seed": 11}},
    {"status": "completed", "config": {"trials": 999, "seed": 11}},
    {"status": "completed"},
])
def test_resume_recomputes_invalid_benchmark_cache(study, cached):
    study_runner.run_study(study.path, study.output)
    atomic_json(study.output / "benchmarks/report.json", cached)
    study_runner.run_study(study.path, study.output, resume=True)
    assert len(study.calls["bench"]) == 2
    assert study.calls["bench"][-1] == {"trials": 2, "seed": 11}


def test_failed_mathematical_checks_stop_before_training(study, monkeypatch):
    def failed(directory, **options):
        report = {"status": "failed", "config": options, "checks_passed": False}
        atomic_json(Path(directory) / "report.json", report)
        return report
    monkeypatch.setattr(study.benchmarks, "research_benchmarks", failed)
    with pytest.raises(RuntimeError):
        study_runner.run_study(study.path, study.output)
    assert study.calls["fit"] == []
    status = json.loads((study.output / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["stage"] == "mathematical_benchmarks"


def test_resume_rejects_configuration_change_before_work(study):
    study_runner.run_study(study.path, study.output)
    study.config["training"]["steps"] += 1
    study.save_config()
    with pytest.raises(ValueError, match="configuration differs"):
        study_runner.run_study(study.path, study.output, resume=True)
    assert len(study.calls["runtime"]) == len(study.calls["fit"]) == 1


def test_existing_study_requires_explicit_resume(study):
    study_runner.run_study(study.path, study.output)
    with pytest.raises(ValueError, match="use --resume"):
        study_runner.run_study(study.path, study.output)
    assert len(study.calls["fit"]) == 1


def test_cuda_startup_failure_is_persisted_without_cpu_fallback(study, monkeypatch):
    study.config["device"] = "cuda:0"
    study.save_config()
    def fail(device):
        assert device == "cuda:0"
        raise RuntimeError("simulated unavailable CUDA")
    monkeypatch.setattr(study_runner, "_runtime_check", fail)
    with pytest.raises(RuntimeError, match="unavailable CUDA"):
        study_runner.run_study(study.path, study.output)
    status = json.loads((study.output / "status.json").read_text())
    assert status["status"] == "failed" and status["stage"] == "startup"
    assert "unavailable CUDA" in status["traceback"]
    assert study.calls["data"] == study.calls["fit"] == []


def test_unexpected_evaluation_failure_is_not_aggregated(study, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("unexpected decoder failure")
    monkeypatch.setattr(study.evaluation, "evaluate_study", fail)
    with pytest.raises(RuntimeError, match="decoder failure"):
        study_runner.run_study(study.path, study.output)
    status = json.loads((study.output / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["stage"] == "evaluate/chord/11/validation"
    assert study.calls["aggregate"] == []
    assert not (study.output / "report.json").exists()


def test_exclusive_study_lock_rejects_concurrent_runner(study):
    with exclusive_run(study.output / "study.lock"):
        with pytest.raises(RuntimeError, match="another process"):
            study_runner.run_study(study.path, study.output)
    assert study.calls["runtime"] == []


def test_status_distinguishes_reused_pid_and_reads_active_progress(tmp_path):
    active_path = tmp_path / "train/status.json"
    atomic_json(active_path, {"status": "running", "step": 9})
    current_tick = study_runner._process_start_tick(os.getpid())
    assert current_tick is not None
    state = {"status": "running", "pid": os.getpid(), "process_start_tick": current_tick,
             "active_status": str(active_path)}
    atomic_json(tmp_path / "status.json", state)
    live = study_runner.study_status(tmp_path)
    assert live["process_alive"] is True
    assert live["active"]["step"] == 9
    atomic_json(tmp_path / "status.json", {**state, "process_start_tick": "wrong-start"})
    stopped = study_runner.study_status(tmp_path)
    assert stopped["process_alive"] is False
    assert stopped["status"] == "stopped_without_final_status"
    atomic_json(tmp_path / "status.json", {**state, "status": "completed", "pid": None})
    assert study_runner.study_status(tmp_path)["status"] == "completed"


def test_launch_already_running_does_not_spawn_or_overwrite_status(tmp_path, monkeypatch):
    monkeypatch.setattr(study_runner, "study_status", lambda _: {
        "status": "running", "process_alive": True, "pid": 123})
    def forbidden(*args, **kwargs):
        pytest.fail("already-running study must not create another child")
    monkeypatch.setattr(study_runner.subprocess, "Popen", forbidden)
    result = study_runner.launch_study(tmp_path / "unused.yaml", tmp_path / "out")
    assert result["status"] == "already_running" and result["pid"] == 123


@pytest.mark.parametrize("seed_field", ["training_seed", "checkpoint_seed"])
def test_aggregate_diversity_keeps_training_seeds_separate(seed_field):
    from tri.evaluation.study import summarize
    rows = []
    for seed in (11, 12):
        for token in (62, 66):
            rows.append({"suite": "unknown4", "method": "tri_direct", "work_id": "same-work",
                         "source_start_cell": 0, seed_field: seed, "valid": True,
                         "returned": True, "exported": True, "status": "exported",
                         "elapsed_decode_seconds": 0.01, "model_calls": 2,
                         "metrics": {"editable_tokens": [token], "onset_pattern": [1],
                                     "chord_tone_fraction": 1.0,
                                     "mean_edit_boundary_or_internal_jump": 0.0,
                                     "editable_rest_fraction": 0.0,
                                     "reconstruction_token_accuracy_diagnostic": None}})
    summary = summarize(rows)["unknown4/tri_direct"]
    assert summary["diversity_request_groups"] == 2
    assert summary["mean_unique_token_fraction"] == 1.0
    assert summary["mean_unique_onset_pattern_fraction"] == 0.5
    assert summary["attempted"] == 4
