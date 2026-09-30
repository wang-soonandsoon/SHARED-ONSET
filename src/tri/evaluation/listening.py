"""Blinded MIDI context bundles; preparing a bundle is not a listening study."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from numbers import Integral
from pathlib import Path
import random
import re
import traceback

import mido

from tri.data.chords import read_tempo_map
from tri.data.prepare import discover_primary_midis
from tri.request_builder import sounding_pitches


class ListeningSourceError(ValueError):
    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(f"{reason}: {message}")


@dataclass(frozen=True)
class _Voice:
    track: int
    channel: int
    pitch: int
    velocity: int
    onset: int


class _State:
    """Channel-global key and CC64 sustain state, independent of track storage."""
    def __init__(self):
        self.pressed = {}
        self.sustained = defaultdict(list)
        self.controls = {}
        self.programs = {}
        self.bends = {}
        self.pressure = {}

    def apply(self, tick, track, message):
        if message.type == "sysex":
            raise ListeningSourceError("unsupported_sysex", "instrument SysEx state is not reconstructed")
        if not hasattr(message, "channel"):
            return
        channel = message.channel
        if message.type == "polytouch":
            raise ListeningSourceError("unsupported_poly_pressure", "polyphonic pressure state is not reconstructed")
        if message.type == "control_change":
            cc, value = message.control, message.value
            if cc in {66, 69} and value >= 64:
                raise ListeningSourceError("unsupported_pedal", f"CC{cc} hold semantics are not reconstructed")
            if cc in {6, 38, 96, 97, 98, 99, 100, 101, 120, 121, 123}:
                raise ListeningSourceError("unsupported_controller", f"CC{cc} state-changing semantics require a separate adapter")
            self.controls[(channel, cc)] = value
            if cc == 64 and value < 64:
                self.sustained[channel].clear()
        elif message.type == "program_change":
            bank = (self.controls.get((channel, 0), 0), self.controls.get((channel, 32), 0))
            self.programs[channel] = (message.program, bank)
        elif message.type == "pitchwheel":
            self.bends[channel] = message.pitch
        elif message.type == "aftertouch":
            self.pressure[channel] = message.value
        elif message.type == "note_on" and message.velocity > 0:
            key = (channel, message.note)
            if key in self.pressed:
                raise ListeningSourceError("ambiguous_note_lifetime", f"overlapping pressed key {key} at tick {tick}")
            self.pressed[key] = _Voice(track, channel, message.note, message.velocity, tick)
        elif message.type in {"note_on", "note_off"}:
            key = (channel, message.note)
            if key not in self.pressed:
                raise ListeningSourceError("unmatched_note_off", f"channel/pitch {key} at tick {tick}")
            voice = self.pressed.pop(key)
            if tick <= voice.onset:
                raise ListeningSourceError("nonpositive_note", f"channel/pitch {key} at tick {tick}")
            if self.controls.get((channel, 64), 0) >= 64:
                self.sustained[channel].append(voice)

    def setup_messages(self, channels):
        messages = []
        for channel in sorted(channels):
            program, active_bank = self.programs.get(channel, (0, (0, 0)))
            # Restore the bank used by the last program change, then restore any
            # later pending bank-select state without changing that instrument.
            messages.extend([mido.Message("control_change", channel=channel, control=0, value=active_bank[0]),
                             mido.Message("control_change", channel=channel, control=32, value=active_bank[1]),
                             mido.Message("program_change", channel=channel, program=program)])
            for (cc_channel, cc), value in sorted(self.controls.items()):
                if cc_channel == channel:
                    messages.append(mido.Message("control_change", channel=channel, control=cc, value=value))
            messages.append(mido.Message("pitchwheel", channel=channel, pitch=self.bends.get(channel, 0)))
            if channel in self.pressure:
                messages.append(mido.Message("aftertouch", channel=channel, value=self.pressure[channel]))
        return messages


def _track_from_events(name, events, duration):
    track = mido.MidiTrack([mido.MetaMessage("track_name", name=name)])
    last = 0
    for tick, message in sorted(events, key=lambda event: event[0]):
        track.append(message.copy(time=tick - last))
        last = tick
    track.append(mido.MetaMessage("end_of_track", time=duration - last))
    return track


def export_context_window(source_midi, tokens, start_cell, out, initial_pitch=None,
                          cells_per_beat=4) -> dict:
    """Export an 8-beat-style window with original non-MELODY accompaniment.

    Timing and controllers are clipped on the original tick axis. Cross-boundary
    keys and CC64-sustained voices are re-anchored at zero; those anchors create a
    new attack, so this is not a waveform-preserving crop. Unsupported stateful
    MIDI semantics fail explicitly. No original MIDI is modified.
    """
    source, out = Path(source_midi).resolve(), Path(out).resolve()
    if source == out:
        raise ValueError("listening export must not overwrite the source MIDI")
    if isinstance(start_cell, bool) or not isinstance(start_cell, Integral) or start_cell < 0:
        raise ValueError("start_cell must be a nonnegative integer")
    if isinstance(cells_per_beat, bool) or not isinstance(cells_per_beat, Integral) or cells_per_beat <= 0:
        raise ValueError("cells_per_beat must be a positive integer")
    tokens = tuple(tokens)
    if not tokens:
        raise ValueError("tokens must be nonempty")
    sounding_pitches(tokens, initial_pitch)
    tempo_map = read_tempo_map(source)
    if tempo_map.ticks_per_beat % cells_per_beat:
        raise ListeningSourceError("unsupported_resolution", "integer ticks per grid cell are required")
    midi = mido.MidiFile(source)
    melody_indices = [i for i, track in enumerate(midi.tracks) if track.name == "MELODY"]
    if len(melody_indices) != 1:
        raise ListeningSourceError("target_track", "exactly one MELODY track is required")
    melody_index = melody_indices[0]
    melody_channels = {m.channel for m in midi.tracks[melody_index] if m.type in {"note_on", "note_off"}}
    if len(melody_channels) != 1:
        raise ListeningSourceError("melody_channels", "replacement requires exactly one original melody channel")
    melody_channel = next(iter(melody_channels))
    step = tempo_map.ticks_per_beat // int(cells_per_beat)
    start, duration = int(start_cell) * step, len(tokens) * step
    end = start + duration
    events, signatures, channels = [], {}, {melody_channel}
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        for ordinal, message in enumerate(track):
            tick += message.time
            if message.type == "time_signature":
                signature = (message.numerator, message.denominator, message.clocks_per_click, message.notated_32nd_notes_per_beat)
                if tick in signatures and signatures[tick] != signature:
                    raise ListeningSourceError("conflicting_meter", f"different time signatures at tick {tick}")
                signatures[tick] = signature
            if message.is_meta:
                continue
            if hasattr(message, "channel"):
                channels.add(message.channel)
            if message.type in {"note_on", "note_off"}:
                if track_index == melody_index:
                    continue
                if message.channel == melody_channel:
                    raise ListeningSourceError("shared_melody_channel", "accompaniment notes share the replaced melody channel")
            if tick < end:
                events.append((int(tick), track_index, ordinal, message))
    events.sort(key=lambda event: event[:3])
    state = _State()
    for tick, track, _, message in events:
        if tick <= start:
            state.apply(tick, track, message)
    if state.controls.get((melody_channel, 64), 0) >= 64 or state.bends.get(melody_channel, 0) != 0:
        raise ListeningSourceError("melody_controller", "melody sustain/bend would change generated grid semantics")
    setup = state.setup_messages(channels)
    accompaniment = {i: [] for i in range(len(midi.tracks)) if i != melody_index}
    # Controls from the removed target track still affect its channel; put them
    # in a dedicated control track, separate from the replacement note sequence.
    accompaniment[melody_index] = []
    anchored_keys = anchored_sustain = 0
    for voices in state.sustained.values():
        for voice in voices:
            if voice.onset < start:
                accompaniment[voice.track].extend([
                    (0, mido.Message("note_on", channel=voice.channel, note=voice.pitch, velocity=voice.velocity)),
                    (0, mido.Message("note_off", channel=voice.channel, note=voice.pitch, velocity=0)),
                ])
                anchored_sustain += 1
    for voice in state.pressed.values():
        if voice.onset < start:
            accompaniment[voice.track].append((0, mido.Message("note_on", channel=voice.channel, note=voice.pitch, velocity=voice.velocity)))
            anchored_keys += 1
    for tick, track, _, message in events:
        if tick < start:
            continue
        if tick == start:
            # State at the boundary has already consumed controls and releases.
            # Boundary note-ons are new attacks and must retain their events.
            if message.type == "note_on" and message.velocity > 0:
                accompaniment[track].append((0, message.copy(time=0)))
            continue
        state.apply(tick, track, message)
        if getattr(message, "channel", None) == melody_channel:
            if (message.type == "control_change" and message.control == 64 and message.value >= 64) or (message.type == "pitchwheel" and message.pitch != 0):
                raise ListeningSourceError("melody_controller", "melody sustain/bend would change generated grid semantics")
        accompaniment[track].append((tick - start, message.copy(time=0)))
    for voice in state.pressed.values():
        accompaniment[voice.track].append((duration, mido.Message("note_off", channel=voice.channel, note=voice.pitch, velocity=0)))
    # Release sustain at the right edge after active keys are closed. Use the
    # final track so equal-tick ordering cannot extend the listening excerpt.
    ending = [(duration, mido.Message("control_change", channel=channel, control=64, value=0))
              for channel in sorted(channels) if state.controls.get((channel, 64), 0) >= 64]

    meta = []
    active_tempo = max(i for i, tick in enumerate(tempo_map.ticks) if tick <= start)
    meta.append((0, mido.MetaMessage("set_tempo", tempo=tempo_map.tempos[active_tempo])))
    for tick, tempo in zip(tempo_map.ticks, tempo_map.tempos):
        if start < tick < end:
            meta.append((tick - start, mido.MetaMessage("set_tempo", tempo=tempo)))
    signatures.setdefault(0, (4, 4, 24, 8))
    initial_signature = signatures[max(tick for tick in signatures if tick <= start)]
    for tick, signature in [(start, initial_signature), *[(t, s) for t, s in sorted(signatures.items()) if start < t < end]]:
        meta.append((tick - start, mido.MetaMessage("time_signature", numerator=signature[0], denominator=signature[1],
                                                   clocks_per_click=signature[2], notated_32nd_notes_per_beat=signature[3])))
    melody_events, active = [], None
    for index, token in enumerate(tokens):
        tick = index * step
        if token == 1:
            if index == 0:
                active = int(initial_pitch)
                melody_events.append((0, mido.Message("note_on", channel=melody_channel, note=active, velocity=80)))
            continue
        if active is not None:
            melody_events.append((tick, mido.Message("note_off", channel=melody_channel, note=active, velocity=0)))
            active = None
        if token >= 2:
            active = int(token) - 2
            melody_events.append((tick, mido.Message("note_on", channel=melody_channel, note=active, velocity=80)))
    if active is not None:
        melody_events.append((duration, mido.Message("note_off", channel=melody_channel, note=active, velocity=0)))
    result = mido.MidiFile(type=1, ticks_per_beat=tempo_map.ticks_per_beat)
    result.tracks.extend([_track_from_events("CONTEXT_TIMING", meta, duration),
                          _track_from_events("INITIAL_CHANNEL_STATE", [(0, message) for message in setup], duration)])
    for track_index in sorted(accompaniment):
        if accompaniment[track_index]:
            name = "MELODY_CONTROLS" if track_index == melody_index else (midi.tracks[track_index].name or f"SOURCE_TRACK_{track_index}")
            result.tracks.append(_track_from_events(name, accompaniment[track_index], duration))
    result.tracks.append(_track_from_events("MELODY", melody_events, duration))
    if ending:
        result.tracks.append(_track_from_events("RIGHT_BOUNDARY_RELEASE", ending, duration))
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(out)
    return {"path": str(out), "source_midi": str(source), "start_cell": int(start_cell),
            "source_start_tick": start, "duration_ticks": duration, "ticks_per_beat": tempo_map.ticks_per_beat,
            "cells_per_beat": int(cells_per_beat), "melody_channel": melody_channel,
            "melody_velocity": 80, "leading_hold_anchored": tokens[0] == 1,
            "anchored_pressed_notes": anchored_keys, "anchored_sustained_voices": anchored_sustain,
            "initial_signature": list(initial_signature[:2]),
            "tempo_events": [{"tick": t, "tempo": m.tempo} for t, m in meta if m.type == "set_tempo"],
            "duration_seconds": tempo_map.tick_to_seconds(end) - tempo_map.tick_to_seconds(start),
            "scope": "standalone context MIDI crop with replaced melody; anchored notes have a new attack"}


def _context_signature(path):
    midi = mido.MidiFile(path)
    return (midi.ticks_per_beat, [[message.dict() for message in track]
                                 for track in midi.tracks if track.name != "MELODY"])


def _blind_label(index):
    label, value = "", index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(65 + remainder) + label
    return label


def build_listening_pack(results_path, data_root, output_dir, seed=20260912, limit=12, *, methods=None) -> dict:
    """Build blinded, paired MIDI excerpts only from common valid requests."""
    results_path, root, output = Path(results_path).resolve(), Path(data_root).resolve(), Path(output_dir).resolve()
    if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    raw_root = (root / "raw").resolve() if (root / "raw").exists() else root
    if output == raw_root or raw_root in output.parents or output == results_path.parent:
        raise ValueError("listening output must be separate from raw data and input result directory")
    groups, invalid_rows, all_methods, input_rows = defaultdict(list), [], set(), 0
    required = {"request_id", "work_id", "source_start_cell", "initial_pitch", "valid", "method"}
    for line_number, line in enumerate(results_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        input_rows += 1
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or required - set(row):
                raise ValueError("missing required request context fields")
            if not isinstance(row["request_id"], str) or not row["request_id"] or not isinstance(row["method"], str) or not row["method"]:
                raise ValueError("request_id and method must be nonempty strings")
            all_methods.add(row["method"])
            groups[row["request_id"]].append(row)
        except (ValueError, TypeError) as error:
            invalid_rows.append({"line": line_number, "reason": str(error)})
    selected_methods = tuple(sorted(all_methods)) if methods is None else tuple(methods)
    if not selected_methods or len(set(selected_methods)) != len(selected_methods) or any(not isinstance(method, str) or not method for method in selected_methods):
        raise ValueError("methods must be distinct nonempty strings")
    eligible, excluded = [], []
    for request_id, rows in sorted(groups.items()):
        relevant = [row for row in rows if row["method"] in selected_methods]
        counts = Counter(row["method"] for row in relevant)
        reason = None
        if any(counts[method] == 0 for method in selected_methods):
            reason = "missing_method"
        elif any(counts[method] != 1 for method in selected_methods):
            reason = "duplicate_method"
        elif any(row["valid"] is not True for row in relevant):
            reason = "not_common_valid"
        elif any(not isinstance(row.get("raw_tokens"), list) or not row["raw_tokens"] for row in relevant):
            reason = "missing_raw_tokens"
        else:
            first = relevant[0]
            fields = ("work_id", "source_start_cell", "initial_pitch", "suite", "replicate", "checkpoint_seed")
            if any(any(row.get(field) != first.get(field) for field in fields) or len(row["raw_tokens"]) != len(first["raw_tokens"]) for row in relevant):
                reason = "inconsistent_context"
        if reason:
            excluded.append({"request_id": request_id, "reason": reason})
        else:
            eligible.append((request_id, relevant))
    rng = random.Random(int(seed))
    rng.shuffle(eligible)
    chosen = eligible[:int(limit)]
    sources = {path.stem: path for path in discover_primary_midis(root)}
    public = output / "public"
    public.mkdir(parents=True, exist_ok=True)
    if public.is_symlink():
        raise ValueError("public listening directory must not be a symbolic link")
    for path in public.iterdir():
        if re.fullmatch(r"item_[0-9]{4,}_[A-Z]+\.mid", path.name) and (path.is_file() or path.is_symlink()):
            path.unlink()
    items, private, failures = [], [], []
    for ordinal, (request_id, rows) in enumerate(chosen):
        shuffled = sorted(rows, key=lambda row: row["method"])
        rng.shuffle(shuffled)
        item_id = f"item_{ordinal:04d}"
        exported, variant_key, signature = [], [], None
        try:
            first = rows[0]
            work = first["work_id"]
            if not isinstance(work, str) or work not in sources:
                raise ListeningSourceError("missing_source", f"unknown primary work {work!r}")
            for variant, row in enumerate(shuffled):
                label = _blind_label(variant)
                path = public / f"{item_id}_{label}.mid"
                exported.append(path)
                metadata = export_context_window(sources[work], row["raw_tokens"], row["source_start_cell"], path,
                                                 initial_pitch=row["initial_pitch"])
                current_signature = _context_signature(path)
                if signature is not None and current_signature != signature:
                    raise RuntimeError("paired variants have different accompaniment/controller/tempo events")
                signature = current_signature
                variant_key.append({"blind_label": label, "method": row["method"], "filename": path.name,
                                    "export": metadata})
            items.append({"item_id": item_id, "variants": [{"label": entry["blind_label"], "midi": entry["filename"]} for entry in variant_key]})
            private.append({"item_id": item_id, "request_id": request_id, "work_id": work,
                            "source_start_cell": first["source_start_cell"], "variants": variant_key})
        except Exception as error:
            for path in exported:
                path.unlink(missing_ok=True)
            failure = {"request_id": request_id, "reason": getattr(error, "reason", "export_error"),
                       "type": type(error).__name__, "message": str(error)}
            if not isinstance(error, (ValueError, OSError)):
                failure["reason"] = "unexpected_error"
                failure["traceback"] = traceback.format_exc()
            failures.append(failure)
    (public / "items.json").write_text(json.dumps({"items": items}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (public / "instructions.md").write_text(
        "# 盲听材料\n\n每个 item 的 A/B/C 等版本具有相同原始伴奏与速度，仅替换旋律。请使用同一音源、音量和播放器比较，随机播放顺序可自行调整。\n\n"
        "可分别记录和声适配、旋律自然度、两个片段的关系是否清楚，以及总体偏好。允许并列、不确定或无法评价。请保存 item 编号和盲标签，勿先查看 private_key.json。\n\n"
        "这些是 MIDI 文件，需要播放器与音源才能发声。本次只准备材料，尚未开展听评；裁剪边界的持续音会重新触发起音。\n", encoding="utf-8")
    (output / "private_key.json").write_text(json.dumps({"seed": int(seed), "items": private}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "status": "needs_attention" if any(failure["reason"] == "unexpected_error" for failure in failures)
                  else "completed_with_export_failures" if failures else "completed",
        "results_path": str(results_path), "methods": list(selected_methods), "seed": int(seed), "limit": int(limit),
        "input_rows": input_rows, "invalid_input_rows": invalid_rows, "request_groups": len(groups),
        "common_valid_groups": len(eligible), "selected_groups": len(chosen), "exported_groups": len(items),
        "exported_midis": sum(len(item["variants"]) for item in items),
        "excluded_group_counts": dict(Counter(entry["reason"] for entry in excluded)), "excluded_groups": excluded,
        "export_failures": failures, "paired_contexts_verified_equal": True, "paired_context_groups_checked": len(items),
        "outputs": {"public_dir": str(public), "private_key": str(output / "private_key.json"), "report": str(output / "report.json")},
        "limitations": ["Only common valid requests are exported; missing/failed requests remain counted in this report.",
                        "No audio rendering, listening judgments or music-quality conclusions were produced.",
                        "Boundary anchors reattack sustained notes. Unsupported MIDI state is reported rather than silently simplified.",
                        "Send listeners the public directory only; the private key reveals method assignments."],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
