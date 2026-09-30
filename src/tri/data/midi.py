"""Strict tick-based monophonic MIDI/grid conversion.

REST=0, HOLD=1 and NOTE(p)=p+2. MASK/PAD are never legal MIDI output.
The writer exports an extracted monophonic window, not a full-song multitrack
edit. A leading HOLD needs an external initial pitch and is exported as an
anchored NOTE at tick zero; its token identity cannot survive standalone MIDI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mido
import numpy as np

REST, HOLD, MASK, PAD = 0, 1, 130, 131


class MidiGridError(ValueError):
    """An explicitly unsupported or lossy source conversion."""

    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(f"{reason}: {message}")


@dataclass(frozen=True)
class GridTrack:
    tokens: np.ndarray
    ticks_per_beat: int
    cells_per_beat: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GridWindow:
    tokens: np.ndarray
    start_cell: int
    initial_pitch: int | None


@dataclass(frozen=True)
class WindowCandidates:
    windows: list[GridWindow]
    metadata: dict[str, Any]
    skipped_window_reasons: dict[str, int]


def _integer_grid(cells_per_beat: int, ticks_per_beat: int) -> int:
    if not isinstance(cells_per_beat, int) or cells_per_beat <= 0:
        raise ValueError("cells_per_beat must be a positive integer")
    if ticks_per_beat <= 0 or ticks_per_beat % cells_per_beat:
        raise MidiGridError("unsupported_resolution", "PPQ must be positive and divisible by cells_per_beat")
    return ticks_per_beat // cells_per_beat


def _read_events(path: str | Path, cells_per_beat: int, track_name: str):
    """Shared source validation, before whole-song or window-local note checks."""
    path = Path(path)
    try:
        midi = mido.MidiFile(path)
    except (OSError, EOFError, ValueError) as error:
        raise MidiGridError("midi_read_error", str(error)) from error
    if midi.type == 2:
        raise MidiGridError("unsupported_midi_type", "asynchronous type-2 tracks have no shared timeline")
    cell_ticks = _integer_grid(cells_per_beat, midi.ticks_per_beat)
    targets = [track for track in midi.tracks if track.name == track_name]
    if len(targets) != 1:
        raise MidiGridError("target_track", f"expected one {track_name!r} track, found {len(targets)}")
    target_channels = {m.channel for m in targets[0] if m.type in {"note_on", "note_off"}}

    signatures, tempos = [], []
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += message.time
            # MIDI controllers are channel-wide even if serialized in another
            # track. Accompaniment pedal on a different channel is irrelevant.
            if getattr(message, "channel", None) in target_channels:
                if message.type == "control_change" and message.control == 64 and message.value >= 64:
                    raise MidiGridError("sustain_pedal", "target-channel pedal prolongation is not implemented")
                if message.type == "control_change" and message.control in {120, 123}:
                    raise MidiGridError("channel_mode", "target-channel all-sound/notes-off is not implemented")
                if message.type == "pitchwheel" and message.pitch != 0:
                    raise MidiGridError("pitch_bend", "target-channel pitch bending is not represented")
            if message.type == "time_signature":
                signatures.append((tick, message.numerator, message.denominator))
            elif message.type == "set_tempo":
                tempos.append((tick, message.tempo))
    signatures.sort()
    if signatures and signatures[0][0] != 0:
        signatures.insert(0, (0, 4, 4))
    unique_signatures = {(num, den) for _, num, den in signatures} or {(4, 4)}
    if len(unique_signatures) != 1:
        raise MidiGridError("meter_change", f"time signatures: {sorted(unique_signatures)}")
    signature = next(iter(unique_signatures))
    if signature not in {(1, 4), (2, 4), (3, 4), (4, 4)}:
        raise MidiGridError("unsupported_meter", f"time signature {signature[0]}/{signature[1]}")

    events: dict[int, list[Any]] = {}
    tick = 0
    for message in targets[0]:
        tick += message.time
        if int(tick) != tick:
            raise MidiGridError("noninteger_tick", f"event at {tick}")
        if message.type == "control_change" and message.control == 64 and message.value >= 64:
            raise MidiGridError("sustain_pedal", "target-track pedal prolongation is not implemented")
        if message.type == "control_change" and message.control in {120, 123}:
            raise MidiGridError("channel_mode", "target-track all-sound/notes-off is not implemented")
        if message.type == "pitchwheel" and message.pitch != 0:
            raise MidiGridError("pitch_bend", "target-track pitch bending is not represented")
        if message.type in {"note_on", "note_off"}:
            events.setdefault(int(tick), []).append(message)
    target_end_tick = int(tick)
    metadata = {
        "source": str(path.resolve()), "track_name": track_name,
        "time_signature": list(signature), "implicit_time_signature": not signatures,
        "tempo_changes": [[int(t), int(v)] for t, v in sorted(tempos)],
    }
    return midi, events, target_end_tick, metadata


def read_melody(
    path: str | Path,
    cells_per_beat: int = 4,
    track_name: str = "MELODY",
    *,
    max_quantization_error_cells: float = 0.25,
) -> GridTrack:
    """Read one complete target track; reject unsupported notes, never drop them.

    Constant 1/4, 2/4, 3/4 and 4/4 signatures are supported. Missing signatures use
    MIDI's default 4/4 and are marked implicit. Meter changes, expressive pedal,
    pitch bend, polyphony and endpoint drift over the stated threshold fail.
    Tempo changes do not affect this beat/tick representation.
    """
    if not 0 <= max_quantization_error_cells < 0.5:
        raise ValueError("max_quantization_error_cells must lie in [0, 0.5)")
    midi, events, target_end_tick, metadata = _read_events(path, cells_per_beat, track_name)
    cell_ticks = _integer_grid(cells_per_beat, midi.ticks_per_beat)

    active: tuple[int, int, int] | None = None  # channel, pitch, onset tick
    notes: list[tuple[int, int, int]] = []
    for tick, messages in sorted(events.items()):
        # MIDI files can serialize an old note-off after its replacement note-on
        # at the same tick. Off-before-on gives the same half-open note intervals.
        off = [m for m in messages if m.type == "note_off" or m.velocity == 0]
        on = [m for m in messages if m.type == "note_on" and m.velocity > 0]
        for message in off:
            if active is None or (message.channel, message.note) != active[:2]:
                raise MidiGridError("unmatched_note_off", f"tick {tick}, channel {message.channel}, pitch {message.note}")
            if tick <= active[2]:
                raise MidiGridError("nonpositive_duration", f"pitch {message.note} at tick {tick}")
            notes.append((active[2], tick, active[1]))
            active = None
        for message in on:
            if active is not None:
                raise MidiGridError("polyphony", f"overlapping notes at tick {tick}: {active[1]} and {message.note}")
            active = (message.channel, message.note, tick)
    if active is not None:
        raise MidiGridError("unclosed_note", f"pitch {active[1]} from tick {active[2]}")
    if not notes:
        raise MidiGridError("empty_melody", "no positive-duration notes")

    grid_notes, errors = [], []
    for onset, offset, pitch in notes:
        start, end = int(np.floor(onset / cell_ticks + 0.5)), int(np.floor(offset / cell_ticks + 0.5))
        errors.extend((abs(onset / cell_ticks - start), abs(offset / cell_ticks - end)))
        if max(errors[-2:]) > max_quantization_error_cells + 1e-10:
            raise MidiGridError("quantization_drift", f"pitch {pitch} at tick {onset}: endpoint error {max(errors[-2:]):.4f} cells exceeds {max_quantization_error_cells}")
        if end <= start:
            raise MidiGridError("quantized_zero_duration", f"pitch {pitch} ticks [{onset}, {offset})")
        if grid_notes and start < grid_notes[-1][1]:
            raise MidiGridError("quantized_overlap", f"pitch {pitch} starts at cell {start}")
        grid_notes.append((start, end, pitch))
    grid_end = max(grid_notes[-1][1], int(np.floor(target_end_tick / cell_ticks + 0.5)))
    tokens = np.full(grid_end, REST, dtype=np.int64)
    for start, end, pitch in grid_notes:
        tokens[start] = pitch + 2
        tokens[start + 1:end] = HOLD
    metadata.update({
        "note_count": len(notes), "grid_length": len(tokens),
        "quantization_max_error_cells": float(max(errors)),
        "quantization_mean_error_cells": float(np.mean(errors)),
        "quantization_tolerance_cells": max_quantization_error_cells,
        "pitch_min": min(n[2] for n in notes), "pitch_max": max(n[2] for n in notes),
    })
    return GridTrack(tokens, midi.ticks_per_beat, cells_per_beat, metadata)


def extract_windows(track: GridTrack, length: int, *, stride: int | None = None) -> list[GridWindow]:
    """Return complete windows and the sounding pitch immediately before each.

    Initial pitch is a boundary condition, not a hidden reconstruction target.
    Windows contain their original REST/NOTE/HOLD tokens unchanged.
    """
    if length <= 0 or (stride is not None and stride <= 0):
        raise ValueError("length and stride must be positive")
    stride = length if stride is None else stride
    starts = set(range(0, len(track.tokens) - length + 1, stride))
    windows, active = [], None
    for index, token in enumerate(track.tokens):
        if index in starts:
            windows.append(GridWindow(track.tokens[index:index + length].copy(), index, active))
        token = int(token)
        if token == REST:
            active = None
        elif 2 <= token <= 129:
            active = token - 2
        elif token != HOLD or active is None:
            raise MidiGridError("invalid_grid", f"token {token} at cell {index} with active pitch {active}")
    return windows


def read_window_candidates(
    path: str | Path,
    length: int,
    *,
    cells_per_beat: int = 4,
    track_name: str = "MELODY",
    stride: int | None = None,
    max_quantization_error_cells: float = 0.25,
) -> WindowCandidates:
    """Select intact monophonic windows without requiring an intact whole song.

    Pair every source note first. A candidate and its one-cell boundary guard
    must contain no excessive endpoint drift, collapsed notes or shared-cell
    polyphony. No offending note is deleted to make a window pass. Short raw
    overlaps may round to touching endpoints within the stated drift bound;
    these quantization changes are counted. Pedal/bend/meter changes and
    ambiguous note lifetimes still reject the source. This is an explicit
    alternative dataset mode, not a relaxed read_melody implementation.
    """
    if length <= 0 or (stride is not None and stride <= 0):
        raise ValueError("length and stride must be positive")
    if not 0 <= max_quantization_error_cells < 0.5:
        raise ValueError("max_quantization_error_cells must lie in [0, 0.5)")
    midi, events, target_end_tick, metadata = _read_events(path, cells_per_beat, track_name)
    cell_ticks = _integer_grid(cells_per_beat, midi.ticks_per_beat)
    active: dict[tuple[int, int], int] = {}
    notes = []
    for tick, messages in sorted(events.items()):
        off = [m for m in messages if m.type == "note_off" or m.velocity == 0]
        on = [m for m in messages if m.type == "note_on" and m.velocity > 0]
        for message in off:
            key = (message.channel, message.note)
            if key not in active:
                raise MidiGridError("unmatched_note_off", f"tick {tick}, channel/pitch {key}")
            onset = active.pop(key)
            if tick <= onset:
                raise MidiGridError("nonpositive_duration", f"channel/pitch {key} at tick {tick}")
            notes.append((onset, tick, message.note))
        for message in on:
            key = (message.channel, message.note)
            if key in active:
                raise MidiGridError("ambiguous_note_lifetimes", f"overlapping same-channel/pitch onsets {key} at tick {tick}")
            active[key] = tick
    if active:
        raise MidiGridError("unclosed_note", f"still active channel/pitches {sorted(active)}")
    if not notes:
        raise MidiGridError("empty_melody", "no positive-duration notes")
    notes.sort()
    grid_end = max(1, int(np.floor(target_end_tick / cell_ticks + 0.5)))
    # Bits allow one rejected window to report all applicable reasons.
    drift_bit, collapse_bit, overlap_bit = 1, 2, 4
    invalid = np.zeros(grid_end, dtype=np.uint8)
    occupancy = np.zeros(grid_end, dtype=np.int32)
    owner = np.full(grid_end, -1, dtype=np.int64)
    grid_notes, endpoint_errors = [], []
    for index, (onset, offset, pitch) in enumerate(notes):
        start = int(np.floor(onset / cell_ticks + 0.5))
        end = int(np.floor(offset / cell_ticks + 0.5))
        error = max(abs(onset / cell_ticks - start), abs(offset / cell_ticks - end))
        endpoint_errors.append(error)
        grid_notes.append((start, end, pitch))
        # Union of original and quantized support; a bad note at the edge cannot
        # vanish merely because its rounded onset falls outside the window.
        lo = max(0, min(start, int(np.floor(onset / cell_ticks))))
        hi = min(grid_end, max(end, int(np.ceil(offset / cell_ticks)), lo + 1))
        if error > max_quantization_error_cells + 1e-10:
            invalid[lo:hi] |= drift_bit
        if end <= start:
            invalid[lo:hi] |= collapse_bit
            continue
        occupancy[start:end] += 1
        owner[start:end] = index
    invalid[occupancy > 1] |= overlap_bit

    tokens = np.full(grid_end, REST, dtype=np.int64)
    for cell in np.flatnonzero(occupancy == 1):
        start, _, pitch = grid_notes[int(owner[cell])]
        tokens[cell] = pitch + 2 if cell == start else HOLD
    # Count all pairwise raw overlaps that disappear under bounded quantization.
    touching_overlaps = 0
    active_indices: list[int] = []
    for index, (onset, offset, _) in enumerate(notes):
        active_indices = [other for other in active_indices if notes[other][1] > onset]
        for other in active_indices:
            left, right = grid_notes[other], grid_notes[index]
            if (left[1] > left[0] and right[1] > right[0] and left[1] <= right[0]
                    and max(endpoint_errors[other], endpoint_errors[index]) <= max_quantization_error_cells + 1e-10):
                touching_overlaps += 1
        active_indices.append(index)

    windows, skipped = [], {"quantization_drift": 0, "quantized_zero_duration": 0,
                           "quantized_overlap": 0, "no_onsets": 0}
    accepted_note_indices: set[int] = set()
    candidate_count = 0
    for start in range(0, grid_end - length + 1, length if stride is None else stride):
        candidate_count += 1
        end = start + length
        guard = invalid[max(0, start - 1):min(grid_end, end + 1)]
        reasons = [(drift_bit, "quantization_drift"), (collapse_bit, "quantized_zero_duration"),
                   (overlap_bit, "quantized_overlap")]
        rejected = False
        for bit, reason in reasons:
            if np.any(guard & bit):
                skipped[reason] += 1
                rejected = True
        if rejected:
            continue
        values = tokens[start:end].copy()
        if not np.any(values >= 2):
            skipped["no_onsets"] += 1
            continue
        initial = None
        if start and occupancy[start - 1] == 1:
            initial = grid_notes[int(owner[start - 1])][2]
        if values[0] == HOLD and initial is None:
            raise AssertionError("valid-window construction produced an unanchored HOLD")
        windows.append(GridWindow(values, start, initial))
        accepted_note_indices.update(int(x) for x in owner[start:end] if x >= 0)
    metadata.update({
        "selection_mode": "valid_windows", "note_count": len(notes), "grid_length": grid_end,
        "ticks_per_beat": midi.ticks_per_beat, "cells_per_beat": cells_per_beat,
        "candidate_windows": candidate_count, "valid_windows": len(windows),
        "rejected_windows": candidate_count - len(windows), "boundary_guard_cells": 1,
        "quantization_tolerance_cells": max_quantization_error_cells,
        "source_quantization_max_error_cells": float(max(endpoint_errors)),
        "quantization_max_error_cells": float(max((endpoint_errors[i] for i in accepted_note_indices), default=0)),
        "raw_overlaps_quantized_to_touching": touching_overlaps,
        "invalid_cell_counts": {name: int(np.count_nonzero(invalid & bit)) for bit, name in
                                [(drift_bit, "quantization_drift"), (collapse_bit, "quantized_zero_duration"), (overlap_bit, "quantized_overlap")]},
    })
    return WindowCandidates(windows, metadata, {k: v for k, v in skipped.items() if v})


def write_grid_midi(
    tokens: np.ndarray | list[int],
    path: str | Path,
    *,
    ticks_per_beat: int = 480,
    cells_per_beat: int = 4,
    initial_pitch: int | None = None,
    tempo: int = 500000,
    time_signature: tuple[int, int] = (4, 4),
) -> dict[str, Any]:
    """Export a standalone monophonic grid; preserve repeated NOTE re-onsets.

    Returns whether a leading HOLD was anchored as a NOTE. Existing context
    tokens should be concatenated before calling when exact token roundtrip at
    the left boundary is needed. This does not preserve source accompaniment.
    """
    cell_ticks = _integer_grid(cells_per_beat, ticks_per_beat)
    if tuple(time_signature) not in {(1, 4), (2, 4), (3, 4), (4, 4)}:
        raise ValueError("time_signature must be supported constant 1/4, 2/4, 3/4 or 4/4")
    values = np.asarray(tokens)
    if values.ndim != 1 or len(values) == 0 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("tokens must be a nonempty one-dimensional integer sequence")
    if initial_pitch is not None and (not isinstance(initial_pitch, (int, np.integer)) or not 0 <= initial_pitch <= 127):
        raise ValueError("initial_pitch must be None or a MIDI pitch")
    events, active = [], None
    anchored = int(values[0]) == HOLD
    for index, value in enumerate(values):
        token, tick = int(value), index * cell_ticks
        if token < 0 or token > 129:
            raise MidiGridError("invalid_grid", f"token {token} cannot be exported at cell {index}")
        if token == HOLD:
            if index == 0 and initial_pitch is not None:
                active = int(initial_pitch)
                events.append((0, mido.Message("note_on", note=active, velocity=80)))
            if active is None:
                raise MidiGridError("orphan_hold", f"HOLD at cell {index} has no sounding pitch")
            continue
        if active is not None:
            events.append((tick, mido.Message("note_off", note=active, velocity=0)))
            active = None
        if token >= 2:
            active = token - 2
            events.append((tick, mido.Message("note_on", note=active, velocity=80)))
    if active is not None:
        events.append((len(values) * cell_ticks, mido.Message("note_off", note=active, velocity=0)))
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    meta, track = mido.MidiTrack(), mido.MidiTrack()
    meta.extend([mido.MetaMessage("set_tempo", tempo=tempo),
                 mido.MetaMessage("time_signature", numerator=int(time_signature[0]), denominator=int(time_signature[1]))])
    track.append(mido.MetaMessage("track_name", name="MELODY"))
    last = 0
    for tick, message in events:
        track.append(message.copy(time=tick - last))
        last = tick
    track.append(mido.MetaMessage("end_of_track", time=len(values) * cell_ticks - last))
    midi.tracks.extend([meta, track])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)
    return {"path": str(path.resolve()), "leading_hold_anchored_as_note": anchored,
            "standalone_monophonic_export": True, "cells": len(values), "time_signature": list(time_signature)}
