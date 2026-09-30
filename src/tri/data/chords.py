"""Exact MIDI-time chord alignment and deterministic, vocabulary-free features.

Labels are sampled at each grid cell's original absolute left endpoint using
half-open annotation intervals. This supplies chord conditions, not downbeat
alignment or a claim that the chord is constant throughout each cell.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
import json
from numbers import Integral
from pathlib import Path
import traceback

import mido
import mir_eval.chord
import numpy as np

from .prepare import discover_primary_midis

CHORD_FEATURE_DIM = 38
FEATURE_SCHEMA = {
    "dimension": CHORD_FEATURE_DIM,
    "absolute_root_one_hot": [0, 12], "absolute_pitch_class_multi_hot": [12, 24],
    "absolute_bass_one_hot": [24, 36], "is_no_chord": 36, "feature_known": 37,
    "encoder": "mir_eval.chord.encode", "reduce_extended_chords": True,
    "strict_bass_intervals": False,
    "semantics": "Pitch-class bitmap includes the explicit slash bass, following mir_eval; extensions are reduced modulo 12. Raw labels are retained.",
    "unknown": "Uncovered, X, or unsupported labels have all-zero features and feature_known=false.",
    "no_chord": "N has is_no_chord=1 and feature_known=1, with zero root/chroma/bass.",
}


class ChordSourceError(ValueError):
    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(f"{reason}: {message}")


@dataclass(frozen=True)
class TempoMap:
    ticks_per_beat: int
    ticks: tuple[int, ...]
    tempos: tuple[int, ...]
    cumulative_tick_microseconds: tuple[int, ...]
    default_initial_tempo_used: bool

    def tick_to_seconds(self, tick: int) -> float:
        if isinstance(tick, bool) or not isinstance(tick, Integral) or tick < 0:
            raise ValueError("tick must be a nonnegative integer")
        tick = int(tick)
        index = bisect_right(self.ticks, tick) - 1
        numerator = self.cumulative_tick_microseconds[index] + (tick - self.ticks[index]) * self.tempos[index]
        return numerator / (self.ticks_per_beat * 1_000_000)

    def cell_seconds(self, start_cell: int, length: int, cells_per_beat: int = 4) -> np.ndarray:
        if isinstance(cells_per_beat, bool) or not isinstance(cells_per_beat, Integral) or cells_per_beat <= 0:
            raise ValueError("cells_per_beat must be a positive integer")
        if self.ticks_per_beat % cells_per_beat:
            raise ChordSourceError("unsupported_resolution", "ticks_per_beat must be divisible by cells_per_beat")
        if isinstance(start_cell, bool) or not isinstance(start_cell, Integral) or start_cell < 0:
            raise ValueError("start_cell must be a nonnegative integer")
        step = self.ticks_per_beat // int(cells_per_beat)
        return np.asarray([self.tick_to_seconds((int(start_cell) + i) * step) for i in range(length + 1)], dtype=np.float64)


def read_tempo_map(path: str | Path) -> TempoMap:
    """Read all MIDI tracks in ticks, accumulating integer tick*tempo products."""
    try:
        midi = mido.MidiFile(path)
    except (OSError, EOFError, ValueError) as error:
        raise ChordSourceError("midi_read_error", str(error)) from error
    if midi.type == 2:
        raise ChordSourceError("unsupported_midi_type", "type-2 tracks lack a shared time axis")
    if midi.ticks_per_beat <= 0:
        raise ChordSourceError("unsupported_resolution", "positive integer PPQ required; SMPTE is unsupported")
    changes: dict[int, int] = {}
    for track in midi.tracks:
        absolute = 0
        for message in track:
            if not isinstance(message.time, Integral) or message.time < 0:
                raise ChordSourceError("invalid_delta_ticks", "MIDI track deltas must be nonnegative integers")
            absolute += int(message.time)
            if message.type != "set_tempo":
                continue
            tempo = int(message.tempo)
            if tempo <= 0:
                raise ChordSourceError("invalid_tempo", f"nonpositive tempo at tick {absolute}")
            if absolute in changes and changes[absolute] != tempo:
                raise ChordSourceError("conflicting_tempo", f"different tempo values at tick {absolute}")
            changes[absolute] = tempo
    default_used = 0 not in changes
    changes.setdefault(0, 500000)
    ticks, tempos, accumulated = [], [], []
    numerator, previous_tick, previous_tempo = 0, 0, changes[0]
    for tick, tempo in sorted(changes.items()):
        numerator += (tick - previous_tick) * previous_tempo
        if not tempos or tempo != tempos[-1]:
            ticks.append(tick)
            tempos.append(tempo)
            accumulated.append(numerator)
        previous_tick, previous_tempo = tick, tempo
    return TempoMap(midi.ticks_per_beat, tuple(ticks), tuple(tempos), tuple(accumulated), default_used)


@dataclass(frozen=True)
class ChordIntervals:
    starts: tuple[float, ...]
    ends: tuple[float, ...]
    labels: tuple[str, ...]

    def label_at(self, seconds: float) -> str | None:
        if not np.isfinite(seconds):
            raise ValueError("seconds must be finite")
        index = bisect_right(self.starts, float(seconds)) - 1
        if index < 0 or seconds >= self.ends[index]:
            return None
        return self.labels[index]


def read_chord_intervals(path: str | Path) -> ChordIntervals:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ChordSourceError("annotation_read_error", str(error)) from error
    starts, ends, labels = [], [], []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        pieces = line.split(maxsplit=2)
        if len(pieces) != 3:
            raise ChordSourceError("malformed_annotation", f"line {line_number}: expected start end label")
        try:
            start, end = float(pieces[0]), float(pieces[1])
        except ValueError as error:
            raise ChordSourceError("malformed_annotation", f"line {line_number}: invalid time") from error
        if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
            raise ChordSourceError("invalid_annotation_interval", f"line {line_number}: {start}..{end}")
        if ends and start < ends[-1]:
            raise ChordSourceError("overlapping_annotation", f"line {line_number}: intervals are unordered or overlap")
        label = pieces[2].strip()
        if not label:
            raise ChordSourceError("malformed_annotation", f"line {line_number}: empty chord label")
        starts.append(start)
        ends.append(end)
        labels.append(label)
    if not starts:
        raise ChordSourceError("empty_annotation", "no chord intervals")
    return ChordIntervals(tuple(starts), tuple(ends), tuple(labels))


def _encode_label(label: str) -> tuple[np.ndarray, str | None]:
    features = np.zeros(CHORD_FEATURE_DIM, dtype=np.float32)
    if label in {"", "X"}:
        return features, "uncovered" if label == "" else "explicit_unknown_chord"
    try:
        root, relative, bass = mir_eval.chord.encode(label, reduce_extended_chords=True,
                                                    strict_bass_intervals=False)
    except mir_eval.chord.InvalidChordException as error:
        return features, str(error)
    if label == "N":
        features[36:] = 1.0
        return features, None
    if not 0 <= root < 12 or not 0 <= bass < 12 or np.any(np.asarray(relative) < 0):
        return features, "unsupported chord encoding"
    features[int(root)] = 1.0
    features[12:24] = np.roll(np.asarray(relative, dtype=np.float32), int(root))
    features[24 + (int(root) + int(bass)) % 12] = 1.0
    features[37] = 1.0
    return features, None


def encode_chord_features(labels: np.ndarray | list, annotation_known: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Encode arbitrary-shaped label arrays with fixed 38-dimensional semantics.

    Returns (features, feature_known), retaining annotation coverage as a separate
    concept. Unsupported labels remain unknown rather than losing slash/quality.
    """
    labels = np.asarray(labels)
    if labels.dtype.kind not in {"U", "S"}:
        raise ValueError("labels must be a Unicode/string array")
    labels = labels.astype(str)
    if annotation_known is None:
        annotation_known = labels != ""
    annotation_known = np.asarray(annotation_known)
    if annotation_known.shape != labels.shape or annotation_known.dtype != np.bool_:
        raise ValueError("annotation_known must be a boolean array matching labels")
    features = np.zeros((*labels.shape, CHORD_FEATURE_DIM), dtype=np.float32)
    for label in np.unique(labels[annotation_known]):
        vector, _ = _encode_label(str(label))
        features[(labels == label) & annotation_known] = vector
    return features, features[..., 37].astype(bool)


def _window_identity(dataset: str | Path) -> tuple[dict[str, np.ndarray], int]:
    required = {"work_ids", "start_cells", "splits", "tokens"}
    with np.load(dataset, allow_pickle=False) as archive:
        if required - set(archive.files):
            raise ValueError(f"dataset missing arrays {sorted(required - set(archive.files))}")
        shape = archive["tokens"].shape  # Only shape is used; melody contents do not inform alignment.
        if len(shape) != 2 or shape[1] < 1:
            raise ValueError("dataset tokens must have shape [N,L]")
        identity = {name: np.array(archive[name], copy=True) for name in required - {"tokens"}}
    count, length = shape
    if any(array.shape != (count,) for array in identity.values()):
        raise ValueError("dataset identity arrays must each have shape [N]")
    if identity["work_ids"].dtype.kind not in {"U", "S"} or identity["splits"].dtype.kind not in {"U", "S"}:
        raise ValueError("work IDs and splits must be string arrays")
    if not np.issubdtype(identity["start_cells"].dtype, np.integer) or np.any(identity["start_cells"] < 0):
        raise ValueError("start_cells must be nonnegative integers")
    return identity, length


def load_chord_sidecar(dataset: str | Path, sidecar: str | Path) -> dict[str, np.ndarray]:
    """Load and validate exact row identity/order and fixed feature semantics."""
    identity, length = _window_identity(dataset)
    count = len(identity["work_ids"])
    path = Path(sidecar)
    if path.is_dir():
        path = path / "chords.npz"
    with np.load(path, allow_pickle=False) as source:
        required = set(identity) | {"chord_labels", "chord_known", "cell_seconds", "row_status", "chord_features", "chord_feature_known"}
        if required - set(source.files):
            raise ValueError(f"chord sidecar missing arrays {sorted(required - set(source.files))}")
        arrays = {key: np.array(source[key], copy=True) for key in required}
    for key, values in identity.items():
        if arrays[key].shape != values.shape or not np.array_equal(arrays[key], values):
            raise ValueError(f"chord sidecar {key} differs from dataset row order/identity")
    expected_shapes = {"chord_labels": (count, length), "chord_known": (count, length),
                       "cell_seconds": (count, length + 1), "row_status": (count,),
                       "chord_features": (count, length, CHORD_FEATURE_DIM), "chord_feature_known": (count, length)}
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"chord sidecar {name} must have shape {shape}")
    if arrays["chord_labels"].dtype.kind not in {"U", "S"} or arrays["row_status"].dtype.kind not in {"U", "S"}:
        raise ValueError("labels/status must be string arrays")
    for name in ("chord_known", "chord_feature_known"):
        if arrays[name].dtype != np.bool_:
            raise ValueError(f"{name} must be boolean")
    features, known = arrays["chord_features"], arrays["chord_feature_known"]
    if not np.issubdtype(features.dtype, np.floating) or not np.all(np.isfinite(features)):
        raise ValueError("chord features must be finite floating-point values")
    if np.any((features != 0) & (features != 1)):
        raise ValueError("chord feature values must be binary")
    if not np.array_equal(features[..., 37].astype(bool), known) or np.any(known & ~arrays["chord_known"]):
        raise ValueError("chord feature known mask is inconsistent")
    if np.any(features[~known] != 0):
        raise ValueError("unknown chord features must be zero")
    expected, expected_known = encode_chord_features(arrays["chord_labels"], arrays["chord_known"])
    if not np.array_equal(features, expected) or not np.array_equal(known, expected_known):
        raise ValueError("chord feature values do not match the documented label encoding")
    states = arrays["row_status"].astype(str)
    if not set(states) <= {"aligned", "partial", "source_error"}:
        raise ValueError("invalid chord row status")
    times = arrays["cell_seconds"]
    if not np.issubdtype(times.dtype, np.floating):
        raise ValueError("cell_seconds must be floating-point")
    for index, state in enumerate(states):
        covered = arrays["chord_known"][index]
        labels = arrays["chord_labels"][index].astype(str)
        if np.any((labels == "") != ~covered):
            raise ValueError("empty chord labels and annotation coverage disagree")
        if state == "aligned" and not np.all(covered):
            raise ValueError("aligned rows must have full annotation coverage")
        if state == "partial" and np.all(covered):
            raise ValueError("partial rows must contain an uncovered cell")
        if state == "source_error" and np.any(covered):
            raise ValueError("source_error rows must remain unknown")
        finite = np.isfinite(times[index])
        if not np.all(finite):
            if state != "source_error" or not np.all(np.isnan(times[index])):
                raise ValueError("unavailable cell times must be all NaN on a source_error row")
        elif np.any(times[index] < 0) or np.any(np.diff(times[index]) <= 0):
            raise ValueError("cell_seconds must be increasing nonnegative absolute times")
    return arrays


def align_chords(dataset: str | Path, data_root: str | Path, output_dir: str | Path,
                 cells_per_beat: int = 4) -> dict:
    """Align all input rows, retaining failures and original row order/splits."""
    if isinstance(cells_per_beat, bool) or not isinstance(cells_per_beat, Integral) or cells_per_beat <= 0:
        raise ValueError("cells_per_beat must be a positive integer")
    dataset, root, output = Path(dataset).resolve(), Path(data_root).resolve(), Path(output_dir).resolve()
    raw = (root / "raw").resolve() if (root / "raw").exists() else root
    if output == raw or raw in output.parents or output == dataset.parent:
        raise ValueError("chord output must be separate from originals and the source window directory")
    if output / "chords.npz" == dataset:
        raise ValueError("chord output cannot overwrite dataset")
    identity, length = _window_identity(dataset)
    files = {path.stem: path for path in discover_primary_midis(root)}
    works = identity["work_ids"].astype(str)
    count = len(works)
    labels = [[""] * length for _ in range(count)]
    known = np.zeros((count, length), dtype=bool)
    seconds = np.full((count, length + 1), np.nan, dtype=np.float64)
    statuses = np.full(count, "source_error", dtype="U12")
    failures, source_metadata = [], {}
    unexpected_errors = 0
    for work in sorted(set(works)):
        indices = np.flatnonzero(works == work)
        stage, tempo_map = "midi", None
        try:
            if work not in files:
                raise ChordSourceError("missing_midi", f"no primary MIDI for work {work!r}")
            tempo_map = read_tempo_map(files[work])
            for index in indices:
                seconds[index] = tempo_map.cell_seconds(int(identity["start_cells"][index]), length, int(cells_per_beat))
            source_metadata[work] = {"tempo_segments": len(tempo_map.ticks),
                                     "tempo_ticks": list(tempo_map.ticks), "tempos": list(tempo_map.tempos),
                                     "ticks_per_beat": tempo_map.ticks_per_beat,
                                     "default_initial_tempo_used": tempo_map.default_initial_tempo_used}
            stage = "annotations"
            intervals = read_chord_intervals(files[work].parent / "chord_midi.txt")
            for index in indices:
                for cell, timestamp in enumerate(seconds[index, :-1]):
                    label = intervals.label_at(float(timestamp))
                    if label is not None:
                        labels[index][cell] = label
                        known[index, cell] = True
                statuses[index] = "aligned" if np.all(known[index]) else "partial"
        except Exception as error:
            # Any failed work leaves all its rows unknown, even after partial
            # execution. Valid tempo-derived timestamps survive annotation errors.
            for index in indices:
                labels[index] = [""] * length
            known[indices] = False
            statuses[indices] = "source_error"
            failure = {"work_id": work, "rows": [int(i) for i in indices], "stage": stage,
                       "reason": error.reason if isinstance(error, ChordSourceError) else "unexpected_error",
                       "type": type(error).__name__, "message": str(error)}
            if not isinstance(error, ChordSourceError):
                unexpected_errors += 1
                failure["traceback"] = traceback.format_exc()
            failures.append(failure)
    chord_labels = np.asarray(labels, dtype=str).reshape(count, length)
    features, feature_known = encode_chord_features(chord_labels, known)
    unsupported_labels = {}
    for label in np.unique(chord_labels[known & ~feature_known]):
        _, reason = _encode_label(str(label))
        unsupported_labels[str(label)] = {"cells": int(np.count_nonzero((chord_labels == label) & known)), "reason": reason}
    output.mkdir(parents=True, exist_ok=True)
    sidecar = output / "chords.npz"
    np.savez_compressed(sidecar, **identity, chord_labels=chord_labels, chord_known=known,
                        cell_seconds=seconds, row_status=statuses,
                        chord_features=features, chord_feature_known=feature_known)
    report = {
        "status": "needs_attention" if unexpected_errors else "completed_with_source_errors" if failures else "completed",
        "dataset": str(dataset), "data_root": str(root), "cells_per_beat": int(cells_per_beat),
        "rows": count, "works": len(set(works)), "window_length": length,
        "row_status_counts": dict(Counter(statuses.tolist())), "known_cells": int(known.sum()),
        "ordinary_chord_cells": int(np.count_nonzero(known & (chord_labels != "N"))),
        "no_chord_cells": int(np.count_nonzero(known & (chord_labels == "N"))),
        "uncovered_cells": int(np.count_nonzero(~known)), "feature_known_cells": int(feature_known.sum()),
        "unsupported_labels": unsupported_labels,
        "variable_tempo_works": sum(item["tempo_segments"] > 1 for item in source_metadata.values()),
        "default_initial_tempo_works": sum(item["default_initial_tempo_used"] for item in source_metadata.values()),
        "sampling_convention": "Each cell's original absolute left endpoint, with [start,end) annotation intervals; cell_seconds includes the final right boundary.",
        "time_conversion": "Integer accumulated delta_ticks*microseconds_per_quarter, divided once by PPQ*1_000_000; all MIDI tracks contribute tempo events.",
        "feature_schema": FEATURE_SCHEMA, "source_metadata": source_metadata,
        "failures": failures, "unexpected_errors": unexpected_errors,
        "outputs": {"chords": str(sidecar), "report": str(output / "report.json")},
        "limitations": ["Beat cells retain original absolute timing; no downbeat/bar alignment is claimed.",
                        "A left-endpoint chord does not imply that the entire cell has one chord.",
                        "Annotation preprocessing includes existing train/validation/test rows without fitting a vocabulary or model.",
                        "Pitch-class features omit voicing/octave while raw quality/slash labels remain available."],
    }
    load_chord_sidecar(dataset, sidecar)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
