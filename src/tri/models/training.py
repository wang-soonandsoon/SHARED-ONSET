"""Full-training-split optimization with resumable state and validation selection.

All training windows participate in shuffled epochs. Test targets are never
used for optimization, evaluation or checkpoint selection. Checkpoints preserve
the optimizer, shuffle cursor, mask RNG and dropout RNG; resume uses a TOTAL
target step count on the same device/software. No content hashes are required.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from tri.data.chords import CHORD_FEATURE_DIM, load_chord_sidecar
from tri.errors import InvalidSpecification
from tri.models.grid import GridConfig, GridDenoiser, MASK, masked_diffusion_loss, two_gap_mask


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def _training_lock(output: Path):
    """Prevent two trainers from writing the same run concurrently."""
    with (output / ".training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise InvalidSpecification("Another trainer already owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _positive(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise InvalidSpecification(f"{name} must be a positive integer")
    return int(value)


def _file_record(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "modified_ns": stat.st_mtime_ns}


def load_training_inputs(dataset: str | Path, chords: str | Path) -> dict:
    """Read exact row-aligned chord features, and enforce work-disjoint splits."""
    dataset, chords = Path(dataset).resolve(), Path(chords).resolve()
    if chords.is_dir():
        chords = chords / "chords.npz"
    before = [_file_record(dataset), _file_record(chords)]
    sidecar = load_chord_sidecar(dataset, chords)
    with np.load(dataset, allow_pickle=False) as archive:
        tokens = np.array(archive["tokens"], copy=True)
        works = np.asarray(archive["work_ids"]).astype(str)
        starts = np.array(archive["start_cells"], copy=True)
        splits = np.asarray(archive["splits"]).astype(str)
    if before != [_file_record(dataset), _file_record(chords)]:
        raise InvalidSpecification("Training input files changed while being read")
    if tokens.ndim != 2 or tokens.shape[1] < 8 or not np.issubdtype(tokens.dtype, np.integer):
        raise InvalidSpecification("Training tokens must be integer [N,L], L >= 8")
    if np.any((tokens < 0) | (tokens >= 130)):
        raise InvalidSpecification("Clean training archives cannot contain MASK/PAD or invalid token IDs")
    if len(tokens) == 0 or works.shape != (len(tokens),) or starts.shape != (len(tokens),) or splits.shape != (len(tokens),):
        raise InvalidSpecification("Training archive row shapes disagree")
    if any(not value.strip() for value in works):
        raise InvalidSpecification("Work IDs must not be empty")
    if len(set(zip(works.tolist(), starts.tolist()))) != len(tokens):
        raise InvalidSpecification("Duplicate (work_id,start_cell) in training archive")
    splits = np.where(splits == "val", "validation", splits)
    if not set(splits) <= {"train", "validation", "test"}:
        raise InvalidSpecification("Only train, validation/val and test splits are supported")
    work_split = {}
    for work, split in zip(works, splits):
        if work in work_split and work_split[work] != split:
            raise InvalidSpecification(f"Work {work!r} leaks across data splits")
        work_split[work] = split
    train_ids, validation_ids = np.flatnonzero(splits == "train"), np.flatnonzero(splits == "validation")
    if not len(train_ids) or not len(validation_ids):
        raise InvalidSpecification("Training requires nonempty train and validation splits; no test fallback")
    return {
        "tokens": torch.as_tensor(tokens.astype(np.int64, copy=False)),
        "condition": torch.as_tensor(sidecar["chord_features"].astype(np.float32, copy=False)),
        "known": sidecar["chord_feature_known"],
        "train_ids": torch.as_tensor(train_ids), "validation_ids": torch.as_tensor(validation_ids),
        "works": works, "splits": splits,
        "identity": {"files": before, "length": tokens.shape[1], "work_ids": works.tolist(),
                     "start_cells": starts.tolist(), "splits": splits.tolist()},
    }


class _EpochSampler:
    def __init__(self, train_ids: torch.Tensor, generator: torch.Generator, state: dict | None = None):
        self.ids, self.generator = train_ids, generator
        if state is None:
            self.order = self.ids[torch.randperm(len(self.ids), generator=self.generator)]
            self.cursor, self.epoch = 0, 0
        else:
            self.order = state["order"].cpu().long()
            self.cursor, self.epoch = int(state["cursor"]), int(state["epoch"])
            if not torch.equal(torch.sort(self.order).values, torch.sort(train_ids).values) or not 0 <= self.cursor <= len(self.ids) or self.epoch < 0:
                raise InvalidSpecification("Invalid resumed epoch sampler state")

    def take(self, count: int) -> torch.Tensor:
        parts = []
        while count:
            if self.cursor == len(self.order):
                self.order = self.ids[torch.randperm(len(self.ids), generator=self.generator)]
                self.cursor = 0
                self.epoch += 1
            size = min(count, len(self.order) - self.cursor)
            parts.append(self.order[self.cursor:self.cursor + size])
            self.cursor += size
            count -= size
        return torch.cat(parts)

    def state(self) -> dict:
        return {"order": self.order.clone(), "cursor": self.cursor, "epoch": self.epoch}


@torch.no_grad()
def validation_ce(model: GridDenoiser, inputs: dict, batch_size: int, device: torch.device,
                  max_gap_cells: int = 8, min_gap_cells: int | None = None) -> float:
    """Fixed full masks at u=1; aggregate CE over eligible slots, not batches."""
    previous_mode = model.training
    model.eval()
    total_loss, eligible_count = 0.0, 0
    try:
        ids = inputs["validation_ids"]
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            clean = inputs["tokens"][batch_ids].to(device)
            editable = two_gap_mask(len(clean), clean.shape[1], device, randomize=False,
                                    min_gap_cells=min(3, max_gap_cells) if min_gap_cells is None else min_gap_cells, max_gap_cells=max_gap_cells)
            noisy = clean.masked_fill(editable, MASK)
            condition = inputs["condition"][batch_ids].to(device) if model.config.condition_dim else None
            logits = model(noisy, torch.ones(len(clean), device=device), editable, condition)
            loss = F.cross_entropy(logits[editable], clean[editable], reduction="sum")
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite validation loss")
            total_loss += float(loss)
            eligible_count += int(editable.sum())
    finally:
        model.train(previous_mode)
    if eligible_count == 0:
        raise InvalidSpecification("No eligible validation positions")
    return total_loss / eligible_count


def _resume_log(path: Path, checkpoint_step: int) -> bool:
    """Discard only uncommitted log tail after the last durable checkpoint."""
    if not path.exists():
        return checkpoint_step == 0
    retained = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            break  # A process can stop partway through its last append.
        if int(record["step"]) <= checkpoint_step:
            retained.append(json.dumps(record, ensure_ascii=False, allow_nan=False))
    complete = not checkpoint_step or bool(retained and json.loads(retained[-1])["step"] == checkpoint_step)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text("".join(line + "\n" for line in retained), encoding="utf-8")
    os.replace(temporary, path)
    return complete


def _restore_best_if_needed(output: Path, restored: dict) -> None:
    """Recover interruption between last.pt and best.pt's atomic replacements."""
    current = None
    try:
        current = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError):
        pass
    matches = (isinstance(current, dict)
               and current.get("step") == restored["best_step"]
               and current.get("best_validation_ce") == restored["best_validation_ce"]
               and current.get("model_config") == restored["model_config"]
               and current.get("input_identity") == restored["input_identity"])
    if matches:
        return
    if restored["best_step"] == restored["step"]:
        _atomic_checkpoint(output / "best.pt", restored)
    else:
        raise InvalidSpecification("Historical best.pt is missing or inconsistent with last.pt; its older model cannot be reconstructed from last.pt")


def fit(dataset: str | Path, chords: str | Path, output_dir: str | Path, *,
        steps: int, batch_size: int = 16, seed: int = 20260912, device: str = "cpu",
        config: GridConfig | None = None, lr: float = 3e-4, eval_every: int = 100,
        save_every: int = 100, max_gap_cells: int = 8, resume: bool = False,
        min_gap_cells: int | None = None) -> dict:
    """Train to TOTAL ``steps``, selecting best.pt by validation CE only.

    A nonperiodic final validation is recorded and may select best.pt, but does
    not consume RNG or change subsequent optimization if that run is resumed.
    Exact resumed model trajectories require unchanged inputs/hyperparameters
    and the same device/software. Checkpoint input checks use identities plus
    file size/mtime, not file hashes. Interruptions recover from atomic last.pt.
    """
    steps, batch_size = _positive(steps, "steps"), _positive(batch_size, "batch_size")
    eval_every, save_every = _positive(eval_every, "eval_every"), _positive(save_every, "save_every")
    max_gap_cells = _positive(max_gap_cells, "max_gap_cells")
    if min_gap_cells is not None:
        min_gap_cells = _positive(min_gap_cells, "min_gap_cells")
        if min_gap_cells > max_gap_cells:
            raise InvalidSpecification('minimum gap exceeds maximum gap')
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise InvalidSpecification("seed must be a nonnegative integer")
    if isinstance(lr, bool) or not isinstance(lr, Real) or not math.isfinite(lr) or lr <= 0:
        raise InvalidSpecification("lr must be finite and positive")
    if not isinstance(resume, bool):
        raise InvalidSpecification("resume must be boolean")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _training_lock(output):
        if not resume and ((output / "last.pt").exists() or (output / "training.jsonl").exists()):
            raise InvalidSpecification("Output already contains a training run; use resume=True or a new output directory")
        step, started = 0, time.perf_counter()
        phase = "input_validation"
        deterministic_before = torch.are_deterministic_algorithms_enabled()
        try:
            inputs = load_training_inputs(dataset, chords)
            target_device = torch.device(device)
            cuda = target_device.type == "cuda"
            if cuda and not torch.cuda.is_available():
                raise InvalidSpecification("CUDA requested but unavailable")
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            restored = None
            if resume:
                if not (output / "last.pt").exists():
                    raise InvalidSpecification("resume=True requires last.pt")
                restored = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
                if restored.get("format_version") != 1:
                    raise InvalidSpecification("Checkpoint does not contain resumable training state")
                if restored["input_identity"] != inputs["identity"]:
                    raise InvalidSpecification("Training inputs differ from the saved checkpoint")
                if config is None:
                    config = GridConfig(**restored["model_config"])
            config = config or GridConfig(length=inputs["tokens"].shape[1], hidden=128, layers=4,
                                          heads=4, dropout=0.1, condition_dim=CHORD_FEATURE_DIM)
            if config.length != inputs["tokens"].shape[1] or config.condition_dim not in (0, CHORD_FEATURE_DIM):
                raise InvalidSpecification("Model length/condition_dim does not match training inputs")
            training_config = {"batch_size": batch_size, "seed": int(seed), "device": str(target_device),
                               "lr": float(lr), "eval_every": eval_every, "max_gap_cells": max_gap_cells,
                               "weight_decay": 0.01, "gradient_clip": 1.0}
            if min_gap_cells is not None:
                training_config['min_gap_cells'] = min_gap_cells
            if restored is not None:
                if restored["model_config"] != asdict(config) or restored["training_config"] != training_config:
                    raise InvalidSpecification("Resume model/training configuration differs from checkpoint")
                if steps < int(restored["step"]):
                    raise InvalidSpecification("Target steps precede the saved checkpoint")
            phase = "model_setup"
            torch.manual_seed(int(seed))
            data_rng = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
            model = GridDenoiser(config).to(target_device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.01)
            best_ce, best_step, last_ce, last_ce_step, initial_ce = None, None, None, None, None
            previous_seconds = 0.0
            if restored is not None:
                model.load_state_dict(restored["model_state"])
                optimizer.load_state_dict(restored["optimizer_state"])
                step = int(restored["step"])
                best_ce, best_step = restored["best_validation_ce"], restored["best_step"]
                last_ce, last_ce_step = restored["last_validation_ce"], restored["last_validation_step"]
                initial_ce = restored["initial_validation_ce"]
                previous_seconds = float(restored["elapsed_seconds"])
                data_rng.set_state(restored["data_rng"].cpu())
                torch.set_rng_state(restored["torch_cpu_rng"].cpu())
                if cuda:
                    torch.cuda.set_rng_state(restored["torch_cuda_rng"].cpu(), target_device)
            sampler = _EpochSampler(inputs["train_ids"], data_rng,
                                    restored["sampler"] if restored is not None else None)
            log_history_complete = True
            if restored is not None:
                log_history_complete = _resume_log(output / "training.jsonl", step)
                _restore_best_if_needed(output, restored)
            started = time.perf_counter()
            starting_step = step

            def elapsed():
                return previous_seconds + time.perf_counter() - started

            def progress(state: str, **extra):
                completed_this_run = step - starting_step
                seconds_this_run = time.perf_counter() - started
                rate = completed_this_run / seconds_this_run if completed_this_run and seconds_this_run else None
                value = {"status": state, "phase": phase, "step": step, "target_steps": steps,
                         "progress": step / steps, "elapsed_seconds": elapsed(),
                         "steps_per_second": rate, "eta_seconds": (steps - step) / rate if rate else None,
                         "train_windows": len(inputs["train_ids"]), "validation_windows": len(inputs["validation_ids"]),
                         "epochs_covered": step * batch_size / len(inputs["train_ids"]),
                         "best_validation_ce": best_ce, "best_step": best_step,
                         "log_history_complete": log_history_complete,
                         "last_validation_ce": last_ce, "last_validation_step": last_ce_step,
                         "last_checkpoint": str(output / "last.pt"), "best_checkpoint": str(output / "best.pt"),
                         "pid": os.getpid(), **extra}
                _atomic_json(output / "status.json", value)
                return value

            def checkpoint():
                return {"format_version": 1, "model_config": asdict(config), "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(), "step": step,
                        "training_config": training_config, "input_identity": inputs["identity"],
                        "best_validation_ce": best_ce, "best_step": best_step,
                        "initial_validation_ce": initial_ce,
                        "last_validation_ce": last_ce, "last_validation_step": last_ce_step,
                        "sampler": sampler.state(), "data_rng": data_rng.get_state(),
                        "torch_cpu_rng": torch.get_rng_state(),
                        "torch_cuda_rng": torch.cuda.get_rng_state(target_device) if cuda else None,
                        "elapsed_seconds": elapsed()}

            if restored is None:
                phase = "initial_validation"
                progress("running")
                last_ce = validation_ce(model, inputs, batch_size, target_device, max_gap_cells, min_gap_cells)
                initial_ce = last_ce
                best_ce, best_step, last_ce_step = last_ce, 0, 0
                payload = checkpoint()
                _atomic_checkpoint(output / "last.pt", payload)
                _atomic_checkpoint(output / "best.pt", payload)
            phase = "training"
            progress("running")
            with (output / "training.jsonl").open("a", encoding="utf-8") as log:
                while step < steps:
                    model.train()
                    ids = sampler.take(batch_size)
                    clean_cpu = inputs["tokens"][ids]
                    editable_cpu = two_gap_mask(batch_size, config.length, "cpu", generator=data_rng,
                                                min_gap_cells=min(3, max_gap_cells) if min_gap_cells is None else min_gap_cells, max_gap_cells=max_gap_cells)
                    noise_cpu = 1.0 - torch.rand(batch_size, generator=data_rng)
                    masked_cpu = (torch.rand(clean_cpu.shape, generator=data_rng) < noise_cpu[:, None]) & editable_cpu
                    clean = clean_cpu.to(target_device)
                    editable, masked, noise = editable_cpu.to(target_device), masked_cpu.to(target_device), noise_cpu.to(target_device)
                    noisy = clean.masked_fill(masked, MASK)
                    condition = inputs["condition"][ids].to(target_device) if config.condition_dim else None
                    logits = model(noisy, noise, editable, condition)
                    loss = masked_diffusion_loss(logits, clean, editable, masked, noise)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Nonfinite training loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    step += 1
                    row = {"step": step, "loss": float(loss.detach()), "gradient_norm": float(gradient),
                           "masked_positions": int(masked_cpu.sum()), "eligible_positions": int(editable_cpu.sum()),
                           "epochs_covered": step * batch_size / len(inputs["train_ids"]),
                           "elapsed_seconds": elapsed()}
                    improved = False
                    if step % eval_every == 0 or step == steps:
                        phase = "validation"
                        progress("running")
                        last_ce = validation_ce(model, inputs, batch_size, target_device, max_gap_cells, min_gap_cells)
                        last_ce_step = step
                        row["validation_ce"] = last_ce
                        if last_ce < best_ce:
                            best_ce, best_step, improved = last_ce, step, True
                        phase = "training"
                    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    log.flush()
                    if step % save_every == 0 or step == steps or improved:
                        payload = checkpoint()
                        _atomic_checkpoint(output / "last.pt", payload)
                        if improved:
                            _atomic_checkpoint(output / "best.pt", payload)
                    if step == 1 or step % min(10, save_every) == 0 or step == steps:
                        progress("running", latest_loss=row["loss"])
            if cuda:
                torch.cuda.synchronize(target_device)
            phase = "completed"
            summary = progress("completed")
            summary.update({
                "scope": "all-training-window conditioned denoiser; validation-only checkpoint selection",
                "model_config": asdict(config), "training_config": training_config,
                "initial_validation_ce": initial_ce,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "training_work_ids": sorted(set(inputs["works"][inputs["splits"] == "train"])),
                "validation_work_ids": sorted(set(inputs["works"][inputs["splits"] == "validation"])),
                "training_chord_known_fraction": float(inputs["known"][inputs["train_ids"].numpy()].mean()),
                "validation_chord_known_fraction": float(inputs["known"][inputs["validation_ids"].numpy()].mean()),
                "test_targets_used": False, "checkpoint_selection": "minimum validation fixed-mask eligible-position CE",
                "resume": resume, "input_files": inputs["identity"]["files"],
                "limitations": ["Validation reconstruction CE is not listening quality or relational completion quality.",
                                "Conditioning uses external chord labels, not a trained accompaniment encoder.",
                                "Exact resumed optimization requires the same device/software and unchanged inputs."]})
            _atomic_json(output / "report.json", summary)
            return summary
        except BaseException as error:
            _atomic_json(output / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                                 "phase": phase, "step": step, "target_steps": steps,
                                                 "error": {"type": type(error).__name__, "message": str(error)},
                                                 "elapsed_attempt_seconds": time.perf_counter() - started,
                                                 "last_checkpoint": str(output / "last.pt"), "pid": os.getpid()})
            raise
        finally:
            torch.use_deterministic_algorithms(deterministic_before)
