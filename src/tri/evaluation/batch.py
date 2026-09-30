"""Batch completion checks on fixed requests; no training or quality claims.

Every selected request remains in each method's denominator. Returned tokens,
independent validity and MIDI export are different outcomes. A provider factory
receives only the visible MusicSpec, never the original clean reference window.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
import json
import math
from numbers import Integral
from pathlib import Path
import re
from time import perf_counter
import traceback
from typing import Callable, Sequence

import numpy as np

from tri.data.midi import MidiGridError, write_grid_midi
from tri.domain.music import MusicSpec, verify_music
from tri.errors import BudgetExceeded, InvalidSpecification, UnsupportedSpec, VerificationError, ZeroMass
from tri.inference.exact import Budget
from tri.request_builder import build_two_gap_request, sounding_pitches
from tri.sampling.baselines import METHODS, METHOD_DESCRIPTIONS, decode_method


@dataclass(frozen=True)
class EvaluationWindow:
    source_index: int
    work_id: str
    start_cell: int
    tokens: tuple[int, ...]
    initial_pitch: int | None
    time_signature: tuple[int, int]
    split: str

    @property
    def request_id(self) -> str:
        return f"{self.work_id}:cell{self.start_cell}"


def _split_name(value: str) -> str:
    value = "validation" if value == "val" else value
    if value not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported split {value!r}")
    return value


def _integer_array(array: np.ndarray, name: str) -> None:
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")


def load_evaluation_windows(dataset: str | Path, *, limit: int = 8,
                            split: str = "validation") -> tuple[EvaluationWindow, ...]:
    """Validate the complete archive, then select stable (work,start) requests.

    Selection never falls back to another split. Work disjointness and duplicate
    request identity checks cover the entire archive, including unselected rows.
    """
    if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1:
        raise ValueError("limit must be a positive integer")
    split = _split_name(split)
    required = {"tokens", "work_ids", "start_cells", "initial_pitches", "splits", "time_signatures"}
    with np.load(dataset, allow_pickle=False) as source:
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"dataset missing arrays: {sorted(missing)}")
        arrays = {name: np.array(source[name], copy=True) for name in required}
    tokens = arrays["tokens"]
    _integer_array(tokens, "tokens")
    if tokens.ndim != 2 or tokens.shape[1] < 16:
        raise ValueError("tokens must have shape [N,L] with L >= 16")
    count = len(tokens)
    for name, values in arrays.items():
        if values.ndim < 1 or len(values) != count:
            raise ValueError(f"array length mismatch: {name}")
        if name not in {"tokens", "time_signatures"} and values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
    if np.any((tokens < 0) | (tokens >= 130)):
        raise ValueError("dataset clean tokens must be output IDs 0..129")
    for name in ("start_cells", "initial_pitches", "time_signatures"):
        _integer_array(arrays[name], name)
    if np.any(arrays["start_cells"] < 0):
        raise ValueError("start_cells must be nonnegative")
    if np.any((arrays["initial_pitches"] < -1) | (arrays["initial_pitches"] > 127)):
        raise ValueError("initial_pitches must be -1 or MIDI pitches 0..127")
    if arrays["time_signatures"].shape != (count, 2):
        raise ValueError("time_signatures must have shape [N,2]")
    if any(tuple(int(x) for x in meter) not in {(1, 4), (2, 4), (3, 4), (4, 4)}
           for meter in arrays["time_signatures"]):
        raise ValueError("unsupported recorded time signature")
    for name in ("work_ids", "splits"):
        if arrays[name].dtype.kind not in {"U", "S"}:
            raise ValueError(f"{name} must contain strings")
    works = arrays["work_ids"].astype(str)
    names = [_split_name(str(value)) for value in arrays["splits"].astype(str)]
    work_splits, identities = {}, set()
    for work, start, name in zip(works, arrays["start_cells"], names):
        if not work.strip():
            raise ValueError("work_ids must not be empty")
        if work in work_splits and work_splits[work] != name:
            raise ValueError(f"work {work!r} appears in multiple splits")
        work_splits[work] = name
        identity = (work, int(start))
        if identity in identities:
            raise ValueError(f"duplicate (work_id,start_cell): {identity}")
        identities.add(identity)
    selected = sorted((i for i, name in enumerate(names) if name == split),
                      key=lambda i: (works[i], int(arrays["start_cells"][i])))[:int(limit)]
    if not selected:
        raise ValueError(f"no {split} windows; no fallback to another split")
    return tuple(EvaluationWindow(
        source_index=i, work_id=str(works[i]), start_cell=int(arrays["start_cells"][i]),
        tokens=tuple(int(t) for t in tokens[i]),
        initial_pitch=None if arrays["initial_pitches"][i] == -1 else int(arrays["initial_pitches"][i]),
        time_signature=tuple(int(t) for t in arrays["time_signatures"][i]), split=split,
    ) for i in selected)


class _CountedProvider:
    def __init__(self, provider: Callable):
        self.provider = provider
        self.calls = 0

    def __call__(self, tokens, noise):
        self.calls += 1  # Include an attempted call that raises.
        return self.provider(tokens, noise)


def _json_value(value):
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"unsupported_python_value": repr(value)}


def _error_status(error: Exception) -> str:
    if isinstance(error, BudgetExceeded):
        return "budget_exceeded"
    if isinstance(error, ZeroMass):
        return "zero_mass"
    if isinstance(error, UnsupportedSpec):
        return "unsupported"
    if isinstance(error, InvalidSpecification):
        return "unsupported_input"
    if isinstance(error, VerificationError):
        return "verifier_failure"
    return "unexpected_error"


def _sync_device(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)


def _clear_managed_midis(directory: Path) -> None:
    """Clear only this evaluator's filenames, never paths read from metadata.

    All prior request/method outputs are removed before truncating results, so
    a smaller selection or changed method list cannot expose stale completions.
    Unrelated files and directories are retained. Unlinking a matched symlink
    removes that link itself; the target is never followed for deletion.
    """
    if directory.is_symlink():
        raise ValueError("evaluation midi directory must not be a symbolic link")
    names = "|".join(re.escape(method) for method in METHODS)
    owned = re.compile(rf"request_[0-9]{{4,}}_(?:{names})\.mid")
    for entry in directory.iterdir():
        if owned.fullmatch(entry.name) and (entry.is_file() or entry.is_symlink()):
            entry.unlink()


def _request_manifest(window: EvaluationWindow, spec: MusicSpec, seed: int) -> dict:
    """Only visible reference information and explicitly specified constraints."""
    return {
        "request_id": window.request_id, "work_id": window.work_id,
        "source_index": window.source_index, "source_start_cell": window.start_cell,
        "split": window.split, "seed": seed, "length": spec.length,
        "initial_pitch": window.initial_pitch, "recorded_meter": list(window.time_signature),
        "observed_tokens": dict(spec.observed), "fixed_soundings": dict(spec.fixed_soundings),
        "editable_positions": [i for i in range(spec.length) if i not in spec.observed],
        "working_pitches": list(spec.pitches), "equal_onsets": spec.equal_onsets,
        "onset_counts": [{"positions": rule.positions, "count": rule.count} for rule in spec.onset_counts],
        "motion_cost": spec.motion_cost,
    }


def _verify_returned(tokens, window: EvaluationWindow, spec: MusicSpec) -> dict:
    checked = verify_music(tokens, spec)
    length_valid = len(tokens) == spec.length
    ids_valid = all(not isinstance(t, (bool, np.bool_)) and isinstance(t, Integral) and 0 <= t < 130 for t in tokens)
    fixed_tokens = length_valid and ids_valid and all(tokens[i] == window.tokens[i] for i in spec.observed)
    fixed_soundings = False
    sounding_error = None
    if length_valid and ids_valid:
        try:
            generated = sounding_pitches(tokens, window.initial_pitch)
            original = sounding_pitches(window.tokens, window.initial_pitch)
            fixed_soundings = all(generated[i] == original[i] for i in spec.observed)
        except InvalidSpecification as error:
            sounding_error = str(error)
    return {
        "independent_verifier_passed": checked.valid,
        "violations": list(checked.violations), "soft_score": checked.soft_score,
        "output_length_valid": length_valid, "output_token_ids_valid": ids_valid,
        "fixed_tokens_preserved": bool(fixed_tokens),
        "fixed_sounding_pitches_preserved": bool(fixed_soundings),
        "sounding_error": sounding_error,
        "valid": checked.valid and bool(fixed_tokens) and bool(fixed_soundings),
    }


def _summary(rows: list[dict]) -> dict:
    attempted = len(rows)
    elapsed = [row["elapsed_decode_seconds"] for row in rows if row["elapsed_decode_seconds"] is not None]
    returned = sum(row["returned"] for row in rows)
    valid = sum(row["valid"] for row in rows)
    exported = sum(row["exported"] for row in rows)
    return {
        "attempted": attempted, "returned": returned, "valid": valid, "exported": exported,
        "returned_rate": returned / attempted if attempted else None,
        "completion_rate": valid / attempted if attempted else None,
        "export_rate": exported / attempted if attempted else None,
        "timed_decodes": len(elapsed),
        "p50_elapsed_seconds": float(np.percentile(elapsed, 50)) if elapsed else None,
        "p95_elapsed_seconds": float(np.percentile(elapsed, 95)) if elapsed else None,
        "total_decode_seconds": float(sum(elapsed)),
        "model_calls_total": sum(row["model_calls"] for row in rows),
        "model_calls_mean_per_attempt": sum(row["model_calls"] for row in rows) / attempted if attempted else None,
        "failure_reasons": dict(Counter(row["status"] for row in rows if row["status"] != "exported")),
    }


def evaluate_windows(
    windows: Sequence[EvaluationWindow], output_dir: str | Path, *,
    provider_factory: Callable[[MusicSpec], Callable], decoder: Callable = decode_method,
    methods: Sequence[str] = ("tri_direct",), steps: int = 4, seed: int = 20260912,
    device: str = "cpu", budget: Budget | None = None, config: dict | None = None,
    exporter: Callable = write_grid_midi,
) -> dict:
    """Evaluate already selected windows with model-free injection for tests.

    The provider factory is called with only a frozen request specification.
    Unexpected implementation errors stay visible as tracebacks and make the
    report `needs_attention`; they are not disguised as constraint failures.
    """
    windows = tuple(sorted(windows, key=lambda w: (w.work_id, w.start_cell)))
    if not windows:
        raise ValueError("at least one evaluation window is required")
    if len({w.request_id for w in windows}) != len(windows):
        raise ValueError("duplicate evaluation request identity")
    methods = tuple(methods)
    if not methods or len(methods) != len(set(methods)) or any(method not in METHODS for method in methods):
        raise ValueError(f"methods must be distinct members of {METHODS}")
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    budget = Budget() if budget is None else budget
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "midi").mkdir(exist_ok=True)
    _clear_managed_midis(output / "midi")
    rows, manifests = [], []
    experiment_start = perf_counter()
    with (output / "results.jsonl").open("w", encoding="utf-8") as results_file:
        for ordinal, window in enumerate(windows):
            request_seed = int(seed) + ordinal
            spec, request_error = None, None
            try:
                spec = build_two_gap_request(window.tokens, initial_pitch=window.initial_pitch)
                manifests.append(_request_manifest(window, spec, request_seed))
            except Exception as error:
                request_error = error
                manifests.append({"request_id": window.request_id, "work_id": window.work_id,
                                  "source_index": window.source_index, "source_start_cell": window.start_cell,
                                  "split": window.split, "seed": request_seed,
                                  "construction_error": {"type": type(error).__name__, "message": str(error)}})
            for method in methods:
                pipeline_start = perf_counter()
                provider = None
                row = {
                    "request_id": window.request_id, "work_id": window.work_id,
                    "source_index": window.source_index, "source_start_cell": window.start_cell,
                    "split": window.split, "seed": request_seed, "method": method,
                    "status": "pending", "returned": False, "valid": False, "exported": False,
                    "raw_tokens": None, "verification": None, "model_calls": 0,
                    "backend_model_calls": None, "elapsed_decode_seconds": None,
                    "error": None, "trace": None, "midi": None,
                }
                stage = "request"
                try:
                    export_path = output / "midi" / f"request_{ordinal:04d}_{method}.mid"
                    if request_error is not None:
                        raise request_error
                    stage = "provider_setup"
                    provider = _CountedProvider(provider_factory(spec))
                    _sync_device(device)
                    stage = "decode"
                    decode_start = perf_counter()
                    try:
                        decoded = decoder(method, spec, provider, steps=int(steps), seed=request_seed, budget=budget)
                    finally:
                        _sync_device(device)
                        row["elapsed_decode_seconds"] = perf_counter() - decode_start
                    row["returned"] = True
                    row["raw_tokens"] = _json_value(decoded.tokens)
                    row["trace"] = _json_value(decoded.trace)
                    row["backend_model_calls"] = _json_value(decoded.model_calls)
                    if isinstance(decoded.model_calls, bool) or not isinstance(decoded.model_calls, Integral) or decoded.model_calls != provider.calls:
                        raise RuntimeError(f"backend model_calls={decoded.model_calls!r}, observed calls={provider.calls}")
                    stage = "verification"
                    values = tuple(decoded.tokens)
                    verification = _verify_returned(values, window, spec)
                    row["verification"] = verification
                    row["valid"] = verification["valid"]
                    if not row["valid"]:
                        row["status"] = "returned_invalid"
                    else:
                        stage = "export"
                        row["midi"] = exporter(values, export_path, initial_pitch=window.initial_pitch,
                                               time_signature=window.time_signature)
                        row["exported"] = True
                        row["status"] = "exported"
                except Exception as error:
                    expected_export_error = stage.startswith("export") and isinstance(error, (OSError, MidiGridError))
                    row["status"] = "export_failure" if expected_export_error else _error_status(error)
                    row["error"] = {"type": type(error).__name__, "message": str(error), "stage": stage}
                    if row["status"] == "unexpected_error":
                        row["error"]["traceback"] = traceback.format_exc()
                row["model_calls"] = 0 if provider is None else provider.calls
                row["elapsed_pipeline_seconds"] = perf_counter() - pipeline_start
                rows.append(row)
                results_file.write(json.dumps(_json_value(row), ensure_ascii=False, allow_nan=False) + "\n")
                results_file.flush()
    work_rows = defaultdict(list)
    for row in rows:
        work_rows[(row["method"], row["work_id"])].append(row)
    summaries = {method: _summary([row for row in rows if row["method"] == method]) for method in methods}
    work_summaries = {
        method: {work: _summary(group) for (group_method, work), group in sorted(work_rows.items()) if group_method == method}
        for method in methods
    }
    report = {
        "status": "needs_attention" if any(row["status"] == "unexpected_error" for row in rows) else "completed",
        "scope": "fixed-request frozen-checkpoint batch engineering evaluation",
        "selected_requests": len(windows), "selected_work_ids": sorted({w.work_id for w in windows}),
        "selected_source_indices": [w.source_index for w in windows],
        "config": {**(config or {}), "methods": list(methods), "steps": int(steps), "seed": int(seed),
                   "device": str(device), "budget": asdict(budget)},
        "method_descriptions": {method: METHOD_DESCRIPTIONS[method] for method in methods},
        "by_method": summaries, "by_work": work_summaries,
        "unexpected_errors": sum(row["status"] == "unexpected_error" for row in rows),
        "elapsed_experiment_seconds": perf_counter() - experiment_start,
        "timing_scope": "Decoder wall time includes provider calls and inference, synchronizes CUDA, and excludes verification/artifact I/O; failed decoder calls remain in timing summaries.",
        "outputs": {"report": str(output / "report.json"), "results": str(output / "results.jsonl"),
                    "requests": str(output / "requests.json"), "midi_dir": str(output / "midi")},
        "limitations": [
            "Windows within a work are correlated; per-work counts are provided and no independent-window confidence intervals are claimed.",
            "Completion rate means independently valid outputs / all selected requests, including failures; it is not a music-quality score.",
            "Fixed method order and no unreported warmup: cold-start effects may affect early timings; these are plumbing diagnostics, not algorithmic speedup evidence.",
            "Beat-window bootstrap has no newly trained chord/accompaniment conditioner or human listening evaluation.",
            "MIDI exports are standalone monophonic windows; a leading HOLD is explicitly anchored as a NOTE.",
        ],
    }
    (output / "requests.json").write_text(json.dumps(_json_value({"requests": manifests}), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.json").write_text(json.dumps(_json_value(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def evaluate_smoke(dataset: str | Path, checkpoint: str | Path, output_dir: str | Path, *,
                   methods: Sequence[str] = ("tri_direct",), limit: int = 8,
                   split: str = "validation", steps: int = 4, seed: int = 20260912,
                   device: str = "cpu", budget: Budget | None = None) -> dict:
    """Load one frozen checkpoint, evaluate the same windows for every method."""
    from tri.models.grid import ModelProbabilityProvider
    from tri.models.train import load_checkpoint

    windows = load_evaluation_windows(dataset, limit=limit, split=split)
    model = load_checkpoint(checkpoint, device)
    if any(len(window.tokens) != model.config.length for window in windows):
        raise ValueError("checkpoint and selected window lengths differ")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def provider_factory(spec):
        editable = tuple(i for i in range(spec.length) if i not in spec.observed)
        return ModelProbabilityProvider(model, editable)

    return evaluate_windows(windows, output_dir, provider_factory=provider_factory,
                            methods=methods, steps=steps, seed=seed, device=device, budget=budget,
                            config={"dataset": str(Path(dataset).resolve()), "checkpoint": str(Path(checkpoint).resolve()),
                                    "requested_limit": int(limit), "split": _split_name(split),
                                    "checkpoint_loads": 1, "training_performed": False})
