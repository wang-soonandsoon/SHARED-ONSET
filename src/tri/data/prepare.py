"""Prepare a deliberately small POP909 pilot without changing shared originals."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from .midi import MidiGridError, extract_windows, read_melody, read_window_candidates


def discover_primary_midis(data_root: str | Path) -> list[Path]:
    """Find observed POP909 primary work/name.mid layout, excluding versions."""
    root = Path(data_root)
    corpus = root / "raw" / "pop909"
    if not corpus.is_dir():
        corpus = root
    files = sorted(p for p in corpus.rglob("*.mid")
                   if p.parent.name.isdigit() and len(p.parent.name) == 3
                   and p.stem == p.parent.name and "versions" not in p.parts)
    ids = [p.stem for p in files]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate primary work IDs: pass one POP909 corpus root")
    if not files:
        raise FileNotFoundError(f"no primary POP909 work/work.mid files below {corpus}")
    return files


def make_work_splits(work_ids: list[str], *, seed: int = 20260912) -> dict[str, str]:
    """Fixed 80/10/10 work split, defined before any filtering or slicing.

    Primary arrangements alone are used. Every window from a work has one split;
    adding alternate arrangements would require using this same work mapping.
    """
    if len(work_ids) != len(set(work_ids)):
        raise ValueError("work_ids must be unique")
    ids = sorted(work_ids)
    random.Random(seed).shuffle(ids)
    train_end, val_end = int(len(ids) * 0.8), int(len(ids) * 0.9)
    return {work: "train" if i < train_end else "validation" if i < val_end else "test"
            for i, work in enumerate(ids)}


def prepare_windows(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    max_files: int = 24,
    bars: int = 2,
    cells_per_bar: int = 16,
    max_windows_per_file: int = 4,
    seed: int = 20260912,
    window_beats: int | None = None,
    window_local: bool = False,
) -> dict[str, Any]:
    """Write strict 4/4 sixteenth-grid melody windows and transparent skips.

    This initial task uses four cells per quarter and sixteen cells per 4/4 bar.
    Recorded 1/4 or other signatures are not silently interpreted as 4/4. A
    corpus with no eligible sampled tracks produces an empty NPZ plus reasons;
    callers must not treat that as a successful training dataset. Explicit
    window_beats selects beat blocks rather than bars, retaining recorded meter.
    Explicit window_local=True selects clean windows from otherwise imperfect
    tracks; all intersecting notes and one-cell boundary guards must be valid.
    """
    if max_files < 1 or bars < 1 or max_windows_per_file < 1:
        raise ValueError("max_files, bars and max_windows_per_file must be positive")
    if cells_per_bar != 16:
        raise ValueError("initial pipeline requires cells_per_bar=16 (4/4, four cells per beat)")
    if window_beats is not None and (not isinstance(window_beats, int) or window_beats < 1):
        raise ValueError("window_beats must be a positive integer or None")
    root, output = Path(data_root).resolve(), Path(output_dir).resolve()
    raw_root = root / "raw" if (root / "raw").exists() else root
    if output == raw_root or raw_root in output.parents:
        raise ValueError("output_dir must not be inside original data")
    primary = discover_primary_midis(root)
    splits = make_work_splits([p.stem for p in primary], seed=seed)
    sample = primary.copy()
    random.Random(seed).shuffle(sample)
    sample = sample[:max_files]
    rows, ids, starts, initials, row_splits, meters = [], [], [], [], [], []
    skipped, accepted, skipped_reasons = [], [], Counter()
    skipped_window_reasons = Counter()
    candidate_windows = valid_windows = rejected_windows = touching_overlaps = 0
    length = bars * cells_per_bar if window_beats is None else window_beats * 4
    for source in sample:
        try:
            if window_local:
                candidates = read_window_candidates(source, length)
                metadata, windows = candidates.metadata, candidates.windows
                candidate_windows += metadata["candidate_windows"]
                valid_windows += metadata["valid_windows"]
                rejected_windows += metadata["rejected_windows"]
                touching_overlaps += metadata["raw_overlaps_quantized_to_touching"]
                skipped_window_reasons.update(candidates.skipped_window_reasons)
            else:
                track = read_melody(source)
                metadata = track.metadata
                windows = [w for w in extract_windows(track, length) if np.any(w.tokens >= 2)]
            if window_beats is None and metadata["time_signature"] != [4, 4]:
                raise MidiGridError("window_meter_mismatch", f"16-cell bars require 4/4, got {metadata['time_signature']}")
            if not windows:
                raise MidiGridError("no_windows", f"no complete {length}-cell window containing an onset")
            rng = random.Random(seed + int(source.stem))
            chosen = sorted(rng.sample(range(len(windows)), min(max_windows_per_file, len(windows))))
            for index in chosen:
                window = windows[index]
                rows.append(window.tokens)
                ids.append(source.stem)
                starts.append(window.start_cell)
                initials.append(-1 if window.initial_pitch is None else window.initial_pitch)
                row_splits.append(splits[source.stem])
                meters.append(metadata["time_signature"])
            accepted.append({"work_id": source.stem, "windows": len(chosen), "split": splits[source.stem],
                             "note_count": metadata["note_count"],
                             "time_signature": metadata["time_signature"],
                             "quantization_max_error_cells": metadata["quantization_max_error_cells"]})
        except MidiGridError as error:
            skipped_reasons[error.reason] += 1
            skipped.append({"work_id": source.stem, "reason": error.reason, "detail": str(error)})
        except (OSError, EOFError) as error:
            skipped_reasons["midi_read_error"] += 1
            skipped.append({"work_id": source.stem, "reason": "midi_read_error", "detail": str(error)})

    output.mkdir(parents=True, exist_ok=True)
    npz_path, split_path, report_path = output / "windows.npz", output / "splits.json", output / "report.json"
    np.savez_compressed(npz_path,
                        tokens=np.stack(rows).astype(np.int64) if rows else np.empty((0, length), dtype=np.int64),
                        work_ids=np.asarray(ids, dtype="U3"), start_cells=np.asarray(starts, dtype=np.int64),
                        initial_pitches=np.asarray(initials, dtype=np.int64), splits=np.asarray(row_splits, dtype="U10"),
                        time_signatures=np.asarray(meters, dtype=np.int64).reshape(-1, 2))
    split_path.write_text(json.dumps({"seed": seed, "unit": "primary_work", "work_splits": splits}, indent=2) + "\n")
    report = {
        "status": "prepared" if rows else "no_eligible_windows",
        "data_root": str(root), "output_dir": str(output), "seed": seed,
        "found_primary_files": len(primary), "attempted_files": len(sample),
        "accepted_files": len(accepted), "skipped_files": len(skipped),
        "windows": len(rows), "window_length": length, "cells_per_beat": 4,
        "window_unit": "bars" if window_beats is None else "beats",
        "selection_mode": "valid_windows" if window_local else "strict_whole_track",
        "window_diagnostics": ({"candidate_windows": candidate_windows, "valid_windows_before_cap": valid_windows,
                                "rejected_windows": rejected_windows, "rejection_reason_counts": dict(skipped_window_reasons),
                                "raw_overlaps_quantized_to_touching": touching_overlaps, "boundary_guard_cells": 1,
                                "reason_counts_may_overlap": True} if window_local else None),
        "window_beats": window_beats, "cells_per_bar": cells_per_bar if window_beats is None else None,
        "bars": bars if window_beats is None else None,
        "accepted_meter_counts": dict(Counter(f"{a['time_signature'][0]}/{a['time_signature'][1]}" for a in accepted)),
        "split_counts": dict(Counter(row_splits)), "skip_reasons": dict(skipped_reasons),
        "accepted": accepted, "skipped": skipped,
        "quantization": {"domain": "MIDI ticks", "rounding": "nearest, half upward",
                         "max_endpoint_error_cells": 0.25, "reject_short_collapsed_notes": True},
        "outputs": {"windows": str(npz_path), "splits": str(split_path), "report": str(report_path)},
        "limitations": ["Original MELODY tracks only; no cleaned corpus or alternate arrangements.",
                        ("Window-local filtering keeps all notes intersecting accepted windows; bad windows are skipped explicitly."
                         if window_local else "Every accepted source track passes the strict whole-song parser."),
                        ("16-cell bars require recorded 4/4; no meter repair or dropped notes." if window_beats is None
                         else "Explicit beat blocks preserve recorded meter; they are not bar/chord-aligned examples."),
                        "Work splits are assigned before eligibility filtering; small pilots may lack evaluation splits.",
                        "Pilot readiness does not establish corpus-wide feasibility or music quality."],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report
