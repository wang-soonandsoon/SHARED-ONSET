from pathlib import Path
import tempfile
import unittest

import mido
import numpy as np

from tri.data.midi import GridTrack, MidiGridError, extract_windows, read_melody, read_window_candidates, write_grid_midi
from tri.data.prepare import discover_primary_midis, make_work_splits, prepare_windows


class MidiGridTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def source(self, messages, *, signature=(4, 4), extra_meta=()):
        path = self.root / "source.mid"
        midi = mido.MidiFile(type=1, ticks_per_beat=480)
        meta = mido.MidiTrack()
        meta.append(mido.MetaMessage("time_signature", numerator=signature[0], denominator=signature[1]))
        meta.extend(extra_meta)
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name="MELODY"))
        track.extend(messages)
        midi.tracks.extend([meta, track])
        midi.save(path)
        return path

    def assert_reason(self, reason, path):
        with self.assertRaises(MidiGridError) as caught:
            read_melody(path)
        self.assertEqual(caught.exception.reason, reason)

    def interval_source(self, notes, end_cell):
        events = []
        for onset, offset, pitch in notes:
            events.append((onset, 1, mido.Message("note_on", note=pitch)))
            events.append((offset, 0, mido.Message("note_off", note=pitch)))
        messages, last = [], 0
        for tick, _, message in sorted(events, key=lambda event: event[:2]):
            messages.append(message.copy(time=tick - last))
            last = tick
        messages.append(mido.MetaMessage("end_of_track", time=end_cell * 120 - last))
        return self.source(messages)

    def test_roundtrip_preserves_repeated_onsets_rests_and_trailing_silence(self):
        tokens = np.array([0, 62, 1, 62, 1, 0, 66, 1, 0, 0])
        path = self.root / "roundtrip.mid"
        report = write_grid_midi(tokens, path)
        np.testing.assert_array_equal(read_melody(path).tokens, tokens)
        self.assertFalse(report["leading_hold_anchored_as_note"])

    def test_same_tick_off_and_on_order_does_not_create_false_overlap(self):
        for ordering in (False, True):
            off = mido.Message("note_off", note=60, time=120)
            on = mido.Message("note_on", note=60, velocity=70, time=0)
            middle = [off, on] if not ordering else [on.copy(time=120), off.copy(time=0)]
            path = self.source([mido.Message("note_on", note=60), *middle,
                                mido.Message("note_off", note=60, time=120)])
            np.testing.assert_array_equal(read_melody(path).tokens, [62, 62])

    def test_note_on_zero_velocity_is_note_off(self):
        path = self.source([mido.Message("note_on", note=60),
                            mido.Message("note_on", note=60, velocity=0, time=240)])
        np.testing.assert_array_equal(read_melody(path).tokens, [62, 1])

    def test_rejects_actual_overlap(self):
        self.assert_reason("polyphony", self.source([
            mido.Message("note_on", note=60), mido.Message("note_on", note=64, time=120),
            mido.Message("note_off", note=60, time=120), mido.Message("note_off", note=64, time=120)]))

    def test_rejects_unclosed_and_unmatched_notes(self):
        self.assert_reason("unclosed_note", self.source([mido.Message("note_on", note=60)]))
        self.assert_reason("unmatched_note_off", self.source([mido.Message("note_off", note=60)]))

    def test_rejects_sustain_and_pitch_bend(self):
        self.assert_reason("sustain_pedal", self.source([mido.Message("control_change", control=64, value=127)]))
        self.assert_reason("pitch_bend", self.source([mido.Message("pitchwheel", pitch=4)]))

    def test_controllers_follow_midi_channels_across_tracks(self):
        path = self.source([mido.Message("note_on", note=60),
                            mido.Message("note_off", note=60, time=240)])
        midi = mido.MidiFile(path)
        accompaniment = mido.MidiTrack()
        accompaniment.append(mido.Message("control_change", channel=1, control=64, value=127))
        midi.tracks.append(accompaniment)
        midi.save(path)
        np.testing.assert_array_equal(read_melody(path).tokens, [62, 1])
        accompaniment[0] = accompaniment[0].copy(channel=0)
        midi.save(path)
        self.assert_reason("sustain_pedal", path)

    def test_malformed_file_has_explicit_read_reason(self):
        path = self.root / "broken.mid"
        path.write_bytes(b"not a midi")
        self.assert_reason("midi_read_error", path)

    def test_rejects_large_quantization_drift(self):
        self.assert_reason("quantization_drift", self.source([
            mido.Message("note_on", note=60, time=60), mido.Message("note_off", note=60, time=240)]))

    def test_small_drift_is_measured_in_ticks_not_seconds(self):
        path = self.source([mido.Message("note_on", note=60, time=12),
                            mido.Message("note_off", note=60, time=240)],
                           extra_meta=[mido.MetaMessage("set_tempo", tempo=900000)])
        track = read_melody(path)
        np.testing.assert_array_equal(track.tokens, [62, 1])
        self.assertAlmostEqual(track.metadata["quantization_max_error_cells"], 0.1)

    def test_rejects_meter_change_and_unsupported_denominator(self):
        notes = [mido.Message("note_on", note=60), mido.Message("note_off", note=60, time=240)]
        self.assert_reason("unsupported_meter", self.source(notes, signature=(6, 8)))
        self.assert_reason("meter_change", self.source(notes, extra_meta=[
            mido.MetaMessage("time_signature", numerator=3, denominator=4, time=120)]))

    def test_window_boundary_carries_sounding_pitch_without_changing_tokens(self):
        tokens = np.array([62, 1, 1, 1, 0, 64, 1, 0])
        track = GridTrack(tokens, 480, 4, {})
        windows = extract_windows(track, 2)
        self.assertIsNone(windows[0].initial_pitch)
        self.assertEqual(windows[1].initial_pitch, 60)
        self.assertEqual(windows[2].initial_pitch, 60)
        np.testing.assert_array_equal(windows[1].tokens, [1, 1])
        np.testing.assert_array_equal(np.concatenate([w.tokens for w in windows]), tokens)

    def test_leading_hold_export_is_explicitly_anchored(self):
        path = self.root / "anchored.mid"
        report = write_grid_midi([1, 1, 0, 66], path, initial_pitch=60)
        self.assertTrue(report["leading_hold_anchored_as_note"])
        np.testing.assert_array_equal(read_melody(path).tokens, [62, 1, 0, 66])
        with self.assertRaises(MidiGridError):
            write_grid_midi([1, 1], path)

    def test_export_rejects_mask_pad_and_orphan_hold(self):
        for tokens in ([62, 130], [62, 131], [62, 0, 1]):
            with self.assertRaises(MidiGridError):
                write_grid_midi(tokens, self.root / "bad.mid")

    def test_split_and_preparation_are_work_disjoint_and_deterministic(self):
        root = self.root / "shared"
        for index in range(1, 11):
            work = f"{index:03d}"
            tokens = np.tile([62, 1, 0, 64], 24)
            write_grid_midi(tokens, root / "raw" / "pop909" / work / f"{work}.mid")
        # Alternate arrangements are deliberately excluded, even if accessible.
        write_grid_midi([62, 1], root / "raw" / "pop909" / "001" / "versions" / "001-v2.mid")
        self.assertEqual(len(discover_primary_midis(root)), 10)
        report = prepare_windows(root, self.root / "derived", max_files=10)
        self.assertEqual(report["accepted_files"], 10)
        with np.load(report["outputs"]["windows"], allow_pickle=False) as arrays:
            for work in np.unique(arrays["work_ids"]):
                work_splits = arrays["splits"][arrays["work_ids"] == work]
                self.assertEqual(len(set(work_splits)), 1)
            self.assertEqual(set(arrays["splits"]), {"train", "validation", "test"})
        repeated = prepare_windows(root, self.root / "derived_again", max_files=10)
        with np.load(report["outputs"]["windows"]) as first, np.load(repeated["outputs"]["windows"]) as second:
            for key in first.files:
                np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(make_work_splits(["001", "002"]), make_work_splits(["002", "001"]))

    def test_beat_blocks_preserve_meter_and_bar_mode_rejects_mismatch(self):
        raw = self.root / "shared" / "raw" / "pop909" / "001"
        raw.mkdir(parents=True)
        path = self.source([mido.Message("note_on", note=60),
                            mido.Message("note_off", note=60, time=480 * 16)], signature=(1, 4))
        (raw / "001.mid").write_bytes(path.read_bytes())
        strict = prepare_windows(self.root / "shared", self.root / "strict")
        self.assertEqual(strict["status"], "no_eligible_windows")
        self.assertEqual(strict["skip_reasons"], {"window_meter_mismatch": 1})
        beats = prepare_windows(self.root / "shared", self.root / "beats", window_beats=8)
        self.assertEqual(beats["window_unit"], "beats")
        self.assertIsNone(beats["bars"])
        self.assertEqual(beats["accepted_meter_counts"], {"1/4": 1})
        with np.load(beats["outputs"]["windows"]) as arrays:
            np.testing.assert_array_equal(arrays["time_signatures"], [[1, 4]])

    def test_output_cannot_write_beneath_originals(self):
        raw = self.root / "shared" / "raw"
        raw.mkdir(parents=True)
        with self.assertRaises(ValueError):
            prepare_windows(self.root / "shared", raw / "derived")

    def test_local_selection_rejects_bad_interval_but_keeps_distant_windows(self):
        path = self.interval_source([(0, 240, 60), (480, 960, 62), (720, 1200, 64),
                                     (1920, 2160, 65), (2880, 3120, 67)], 32)
        candidates = read_window_candidates(path, 8)
        self.assertEqual([w.start_cell for w in candidates.windows], [16, 24])
        self.assertEqual(candidates.skipped_window_reasons["quantized_overlap"], 2)
        np.testing.assert_array_equal(candidates.windows[0].tokens, [67, 1, 0, 0, 0, 0, 0, 0])

    def test_bounded_micro_overlap_may_quantize_to_touching(self):
        path = self.interval_source([(0, 246, 60), (234, 480, 64)], 4)
        self.assert_reason("polyphony", path)
        candidates = read_window_candidates(path, 4)
        self.assertEqual(len(candidates.windows), 1)
        self.assertEqual(candidates.metadata["raw_overlaps_quantized_to_touching"], 1)
        np.testing.assert_array_equal(candidates.windows[0].tokens, [62, 1, 66, 1])

    def test_true_shared_grid_cell_is_rejected_locally(self):
        path = self.interval_source([(0, 360, 60), (240, 480, 64)], 4)
        candidates = read_window_candidates(path, 4)
        self.assertEqual(candidates.windows, [])
        self.assertEqual(candidates.skipped_window_reasons, {"quantized_overlap": 1})

    def test_window_local_drift_and_collapsed_notes_are_not_dropped(self):
        for bad_note, reason in [((540, 780, 60), "quantization_drift"),
                                 ((490, 500, 60), "quantized_zero_duration")]:
            path = self.interval_source([bad_note, (1920, 2160, 64)], 24)
            candidates = read_window_candidates(path, 8)
            self.assertIn(reason, candidates.skipped_window_reasons)
            self.assertEqual([w.start_cell for w in candidates.windows], [16])

    def test_crossing_conflict_protects_both_window_boundaries(self):
        path = self.interval_source([(720, 1200, 60), (840, 1080, 64), (1920, 2160, 65)], 24)
        candidates = read_window_candidates(path, 8)
        self.assertEqual([w.start_cell for w in candidates.windows], [16])
        self.assertEqual(candidates.skipped_window_reasons["quantized_overlap"], 2)

    def test_local_window_can_anchor_an_unambiguous_long_note_after_old_conflict(self):
        path = self.interval_source([(0, 2400, 60), (480, 720, 64), (2640, 2880, 65)], 24)
        candidates = read_window_candidates(path, 8)
        window = next(w for w in candidates.windows if w.start_cell == 16)
        self.assertEqual(window.initial_pitch, 60)
        np.testing.assert_array_equal(window.tokens, [1, 1, 1, 1, 0, 0, 67, 1])

    def test_ambiguous_same_pitch_lifetimes_reject_source_in_local_mode(self):
        path = self.interval_source([(0, 360, 60), (240, 480, 60)], 4)
        with self.assertRaises(MidiGridError) as caught:
            read_window_candidates(path, 4)
        self.assertEqual(caught.exception.reason, "ambiguous_note_lifetimes")

    def test_writer_preserves_recorded_meter(self):
        path = self.root / "one_four.mid"
        write_grid_midi([62, 1, 0, 64], path, time_signature=(1, 4))
        self.assertEqual(read_melody(path).metadata["time_signature"], [1, 4])


if __name__ == "__main__":
    unittest.main()
