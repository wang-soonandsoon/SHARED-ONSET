from pathlib import Path
import json
import tempfile
import unittest

import mido

from tri.evaluation.listening import (ListeningSourceError, _context_signature,
                                     build_listening_pack, export_context_window)


class ListeningPackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def track(name, events):
        track = mido.MidiTrack([mido.MetaMessage("track_name", name=name)])
        previous = 0
        for tick, message in sorted(events, key=lambda event: event[0]):
            track.append(message.copy(time=tick - previous))
            previous = tick
        return track

    def source(self, accompaniment=None, *, work="001", melody_program=73):
        directory = self.data / "raw" / "pop909" / work
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{work}.mid"
        midi = mido.MidiFile(type=1, ticks_per_beat=480)
        meta = self.track("", [(0, mido.MetaMessage("set_tempo", tempo=500000)),
                               (0, mido.MetaMessage("time_signature", numerator=1, denominator=4)),
                               (720, mido.MetaMessage("set_tempo", tempo=1000000)),
                               (840, mido.MetaMessage("time_signature", numerator=3, denominator=4))])
        melody = self.track("MELODY", [(0, mido.Message("program_change", channel=0, program=melody_program)),
                                         (0, mido.Message("note_on", channel=0, note=60, velocity=30)),
                                         (1920, mido.Message("note_off", channel=0, note=60))])
        if accompaniment is None:
            accompaniment = [(0, mido.Message("program_change", channel=2, program=4)),
                             (240, mido.Message("note_on", channel=2, note=48, velocity=51)),
                             (840, mido.Message("note_off", channel=2, note=48)),
                             (900, mido.Message("note_on", channel=2, note=52, velocity=61)),
                             (1680, mido.Message("note_off", channel=2, note=52))]
        midi.tracks.extend([meta, melody, self.track("PIANO", accompaniment)])
        midi.save(path)
        return path

    @staticmethod
    def absolute_messages(path, track_name):
        track = next(track for track in mido.MidiFile(path).tracks if track.name == track_name)
        output, tick = [], 0
        for message in track:
            tick += message.time
            output.append((tick, message))
        return output

    def export(self, tokens=(62, 1, 62, 1, 0, 66, 1, 0), source=None, name="out.mid", **kwargs):
        return export_context_window(source or self.source(), tokens, 4, self.root / name, **kwargs)

    def test_original_accompaniment_is_clipped_and_program_channel_preserved(self):
        metadata = self.export()
        piano = self.absolute_messages(metadata["path"], "PIANO")
        notes = [(tick, message.type, message.note, message.channel, message.velocity)
                 for tick, message in piano if message.type in {"note_on", "note_off"}]
        self.assertEqual(notes, [(0, "note_on", 48, 2, 51), (360, "note_off", 48, 2, 64),
                                 (420, "note_on", 52, 2, 61), (960, "note_off", 52, 2, 0)])
        setup = self.absolute_messages(metadata["path"], "INITIAL_CHANNEL_STATE")
        programs = {message.channel: message.program for _, message in setup if message.type == "program_change"}
        self.assertEqual(programs[0], 73)
        self.assertEqual(programs[2], 4)
        self.assertEqual(metadata["anchored_pressed_notes"], 1)

    def test_tempo_meter_changes_and_absolute_window_start_are_preserved(self):
        metadata = self.export()
        timing = self.absolute_messages(metadata["path"], "CONTEXT_TIMING")
        self.assertEqual([(tick, message.tempo) for tick, message in timing if message.type == "set_tempo"],
                         [(0, 500000), (240, 1000000)])
        self.assertEqual([(tick, message.numerator, message.denominator) for tick, message in timing if message.type == "time_signature"],
                         [(0, 1, 4), (360, 3, 4)])
        self.assertEqual(metadata["duration_seconds"], 1.75)

    def test_repeated_generated_onsets_and_leading_hold_are_preserved(self):
        metadata = self.export(tokens=(1, 1, 62, 1, 62, 0, 0, 0), initial_pitch=60)
        notes = [(tick, message.type, message.note) for tick, message in self.absolute_messages(metadata["path"], "MELODY")
                 if message.type in {"note_on", "note_off"}]
        self.assertEqual(notes, [(0, "note_on", 60), (240, "note_off", 60), (240, "note_on", 60),
                                 (480, "note_off", 60), (480, "note_on", 60), (600, "note_off", 60)])
        self.assertTrue(metadata["leading_hold_anchored"])

    def test_sustain_released_note_crossing_left_boundary_is_anchored(self):
        events = [(0, mido.Message("control_change", channel=2, control=64, value=127)),
                  (120, mido.Message("note_on", channel=2, note=48, velocity=77)),
                  (240, mido.Message("note_off", channel=2, note=48)),
                  (960, mido.Message("control_change", channel=2, control=64, value=0))]
        metadata = self.export(source=self.source(events))
        piano = self.absolute_messages(metadata["path"], "PIANO")
        self.assertEqual([(tick, message.type) for tick, message in piano if message.type in {"note_on", "note_off"}],
                         [(0, "note_on"), (0, "note_off")])
        self.assertEqual(metadata["anchored_sustained_voices"], 1)
        self.assertTrue(any(tick == 480 and message.type == "control_change" and message.control == 64 and message.value == 0 for tick, message in piano))

    def test_note_ending_exactly_at_left_boundary_does_not_reattack(self):
        events = [(0, mido.Message("note_on", channel=2, note=48)),
                  (480, mido.Message("note_off", channel=2, note=48))]
        metadata = self.export(source=self.source(events))
        self.assertEqual(metadata["anchored_pressed_notes"], 0)
        self.assertEqual(metadata["anchored_sustained_voices"], 0)
        self.assertFalse(any(track.name == "PIANO" for track in mido.MidiFile(metadata["path"]).tracks))

    def test_right_boundary_releases_pedal_after_closing_pressed_keys(self):
        events = [(0, mido.Message("control_change", channel=2, control=64, value=127)),
                  (0, mido.Message("note_on", channel=2, note=48)),
                  (1920, mido.Message("note_off", channel=2, note=48))]
        metadata = self.export(source=self.source(events))
        ending = self.absolute_messages(metadata["path"], "RIGHT_BOUNDARY_RELEASE")
        self.assertTrue(any(tick == 960 and message.type == "control_change" and message.control == 64 and message.value == 0 for tick, message in ending))
        self.assertEqual(mido.MidiFile(metadata["path"]).tracks[-1].name, "RIGHT_BOUNDARY_RELEASE")

    def test_unsupported_sostenuto_and_ambiguous_lifetimes_fail_explicitly(self):
        examples = [([(0, mido.Message("control_change", channel=2, control=66, value=127))], "unsupported_pedal"),
                    ([(0, mido.Message("note_on", channel=2, note=48)),
                      (120, mido.Message("note_on", channel=2, note=48))], "ambiguous_note_lifetime")]
        for events, reason in examples:
            with self.assertRaises(ListeningSourceError) as caught:
                self.export(source=self.source(events))
            self.assertEqual(caught.exception.reason, reason)

    def test_pair_variants_have_identical_context_tracks(self):
        source = self.source()
        first = self.export(source=source, name="a.mid")
        second = self.export(tokens=(64, 1, 64, 1, 0, 69, 1, 0), source=source, name="b.mid")
        self.assertEqual(_context_signature(first["path"]), _context_signature(second["path"]))

    def result_file(self, rows):
        directory = self.root / "research"
        directory.mkdir(exist_ok=True)
        path = directory / "results.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    @staticmethod
    def row(request="r1", method="tri_direct", valid=True, work="001"):
        return {"request_id": request, "method": method, "valid": valid, "work_id": work,
                "source_start_cell": 4, "initial_pitch": None, "raw_tokens": [62, 1, 62, 1, 0, 66, 1, 0]}

    def test_pack_blinds_methods_and_keeps_missing_failed_denominators(self):
        self.source()
        rows = [self.row(method="tri_direct"), self.row(method="one_shot_joint"),
                self.row("missing"), self.row("failed", valid=False), self.row("failed", method="one_shot_joint")]
        report = build_listening_pack(self.result_file(rows), self.data, self.root / "pack", methods=("one_shot_joint", "tri_direct"))
        self.assertEqual((report["request_groups"], report["common_valid_groups"], report["exported_groups"], report["exported_midis"]), (3, 1, 1, 2))
        self.assertEqual(report["excluded_group_counts"], {"missing_method": 1, "not_common_valid": 1})
        public = Path(report["outputs"]["public_dir"])
        manifest = json.loads((public / "items.json").read_text())
        self.assertNotIn("tri_direct", json.dumps(manifest))
        key = json.loads(Path(report["outputs"]["private_key"]).read_text())
        self.assertEqual({variant["method"] for variant in key["items"][0]["variants"]}, {"one_shot_joint", "tri_direct"})

    def test_missing_source_is_counted_without_path_traversal(self):
        self.source()
        rows = [self.row(work="../outside"), self.row(method="one_shot_joint", work="../outside")]
        report = build_listening_pack(self.result_file(rows), self.data, self.root / "pack")
        self.assertEqual(report["selected_groups"], 1)
        self.assertEqual(report["exported_groups"], 0)
        self.assertEqual(report["export_failures"][0]["reason"], "missing_source")

    def test_rerun_with_smaller_limit_removes_stale_owned_midis(self):
        self.source()
        rows = [self.row(request, method=method) for request in ("r1", "r2") for method in ("one_shot_joint", "tri_direct")]
        path, output = self.result_file(rows), self.root / "pack"
        report = build_listening_pack(path, self.data, output, limit=2)
        public = Path(report["outputs"]["public_dir"])
        self.assertEqual(len(list(public.glob("*.mid"))), 4)
        unrelated = public / "my_midi.mid"
        unrelated.write_bytes(b"keep")
        build_listening_pack(path, self.data, output, limit=1)
        self.assertEqual(len(list(public.glob("item_*.mid"))), 2)
        self.assertEqual(unrelated.read_bytes(), b"keep")

    def test_blind_assignment_is_reproducible_under_input_row_reordering(self):
        self.source()
        rows = [self.row(method="one_shot_joint"), self.row(method="tri_direct"), self.row(method="smc_4")]
        first = build_listening_pack(self.result_file(rows), self.data, self.root / "pack1", seed=42)
        second = build_listening_pack(self.result_file(list(reversed(rows))), self.data, self.root / "pack2", seed=42)
        mappings = []
        for report in (first, second):
            key = json.loads(Path(report["outputs"]["private_key"]).read_text())
            mappings.append([(v["blind_label"], v["method"]) for v in key["items"][0]["variants"]])
        self.assertEqual(mappings[0], mappings[1])

    def test_one_bad_variant_removes_all_partial_group_files(self):
        self.source()
        rows = [self.row(method="one_shot_joint"), self.row(method="tri_direct")]
        rows[1]["raw_tokens"][0] = 1  # Claimed valid but missing a required left anchor.
        report = build_listening_pack(self.result_file(rows), self.data, self.root / "pack")
        self.assertEqual(report["selected_groups"], 1)
        self.assertEqual(report["exported_groups"], 0)
        self.assertEqual(list(Path(report["outputs"]["public_dir"]).glob("item_*.mid")), [])


if __name__ == "__main__":
    unittest.main()
